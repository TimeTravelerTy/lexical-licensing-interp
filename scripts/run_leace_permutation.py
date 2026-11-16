#!/usr/bin/env python3
"""Permutation control for the in-sample LEACE conditions (run_leace_necessity.py --insample), Pythia-1.4B fp32.

Same items, sites, position and fitting as the in-sample run (one eraser set fit on all items, applied to the same
items; LeaceFitter defaults, float64 statistics; rank 1), but the concept is a permuted verb label instead of verb class:
for permutation j, exactly half of the T verbs and half of the I verbs (floor) are flipped, chosen at random (seed j),
and every prompt takes its verb's permuted label. The labels are balanced and their correlation with the true class
over prompts is checked to be < 0.05 (with 125 prompts per verb it is 0).

Conditions (as in the in-sample run, fit with the permuted labels):
- scrub_0_8, scrub_0_12, scrub_all (0..23): sequential scrubbing; eraser L fit on site-L states with erasers 0..L-1
  of the same permutation applied and L not applied. The three are prefixes of one sequential fit.
- single_8: LEACE at site 8 alone, fit on natural states.

Measured, as before, on the true items: log P(" by") at the verb position and KL(P_natural || P_erased), natural
recomputed in the same batch. --with-true also runs the true labels through this code; its per-prompt log P(by) must
reproduce the in-sample run (--reference-items).

Outputs (--out-dir): items.csv.gz (per prompt: lp_by and kl per permutation x condition), labels.csv (permuted label
per verb), fit_diagnostics.csv, checks.json, meta.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_leace_necessity import DAS_SITE, N_SITES, Model, Stop, apply_erasers, fit_stats, load_items, sha

CONDS = {"scrub_0_8": 8, "scrub_0_12": 12, "scrub_all": N_SITES - 1, "single_8": None}


def permuted_labels(P, seed):
    """Flip exactly half (floor) of the T verbs and half of the I verbs; label per prompt via its verb."""
    rng = np.random.default_rng(seed)
    verbs = P.drop_duplicates("verb")[["verb", "cls"]].sort_values("verb").reset_index(drop=True)
    lab = verbs.cls.to_numpy().copy()
    for c in (0, 1):
        idx = np.flatnonzero(verbs.cls.to_numpy() == c)
        lab[rng.choice(idx, len(idx) // 2, replace=False)] ^= 1
    verbs["label"] = lab
    z = P.verb.map(dict(zip(verbs.verb, verbs.label))).to_numpy()
    return verbs, z


def run(args):
    import torch
    from concept_erasure import LeaceFitter

    args.dtype = "float32"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    M = Model(args)
    dev, d = M.dev, M.d
    pas, P, pairs = load_items(args, M.R.tok)
    N = len(P)
    sites = list(range(N_SITES))
    # the in-sample run's batching (all items, stable sort by length) for fitting and evaluation alike
    order = np.argsort(P.n_tokens.to_numpy(), kind="stable")
    batches = [order[i:i + args.batch_size] for i in range(0, N, args.batch_size)]
    prompts = P.prompt.tolist()
    cls = P.cls.to_numpy()
    a, b = (int(x) for x in args.perms.split("-"))
    labelsets = ([("true", cls)] if args.with_true else []) + [(f"p{j}", None) for j in range(a, b + 1)]

    # natural site-8 states (single-site fits)
    nat8 = torch.zeros((N, d), dtype=torch.float32, device=dev)
    for bt in batches:
        cap = {}
        M.forward([prompts[i] for i in bt], {DAS_SITE: lambda v: cap.__setitem__("x", v.clone())}, logits=False)
        nat8[torch.as_tensor(bt, device=dev)] = cap["x"]
    log(f"{N} prompts; natural site-{DAS_SITE} states captured")

    erasers, diag, lab_rows, checks = {}, [], [], {"label_corr": {}}
    for name, z in labelsets:
        if z is None:
            verbs, z = permuted_labels(P, args.seed_base + int(name[1:]))
            lab_rows += [{"perm": name, "verb": v, "cls": c, "label": l} for v, c, l in verbs.itertuples(index=False)]
        corr = float(np.corrcoef(z, cls)[0, 1])
        checks["label_corr"][name] = corr
        assert abs(z.mean() - 0.5) < 0.05, (name, z.mean())
        assert name == "true" or abs(corr) < 0.05, (name, corr)
        z_t = torch.as_tensor(z, device=dev, dtype=torch.float64)
        # single site 8, natural states
        fitter = LeaceFitter(d, 1, device=dev, dtype=torch.float64)
        for i in range(0, N, 4096):
            j = torch.arange(i, min(i + 4096, N), device=dev)
            fitter.update(nat8[j].double(), z_t[j])
        E, st = fit_stats(fitter, torch)
        single = {0: E}
        diag.append({"labels": name, "condition": "single", "site": DAS_SITE, **st})
        # sequential scrub 0..23
        scrub = {s: {} for s in sites}
        for L in sites:
            fitter = LeaceFitter(d, 1, device=dev, dtype=torch.float64)
            for bt in batches:
                cap = {}
                ef = torch.zeros(len(bt), dtype=torch.long, device=dev)
                hooks = {s: (lambda v, s=s: apply_erasers(scrub[s], ef, v)) for s in range(L)}

                def capture(v):
                    cap["x"] = v.clone()  # site L, before any erasure at L
                    raise Stop
                hooks[L] = capture
                M.forward([prompts[i] for i in bt], hooks)
                fitter.update(cap["x"].double(), z_t[torch.as_tensor(bt, device=dev)])
            E, st = fit_stats(fitter, torch)
            scrub[L][0] = E
            diag.append({"labels": name, "condition": "scrub", "site": L, **st})
        erasers[name] = (single, scrub)
        log(f"{name}: erasers fit (label corr {corr:+.3f})")

    assert all(r["erased_rank"] == 1 for r in diag), "an eraser is not rank 1"
    cols = {"lp_by__natural": np.zeros(N)}
    for name, _ in labelsets:
        for c in CONDS:
            cols[f"lp_by__{name}__{c}"] = np.zeros(N)
            cols[f"kl__{name}__{c}"] = np.zeros(N)
    for bi, bt in enumerate(batches):
        bp = [prompts[i] for i in bt]
        ef = torch.zeros(len(bt), dtype=torch.long, device=dev)
        lp0, _ = M.forward(bp, {})
        p0 = lp0.exp()
        cols["lp_by__natural"][bt] = lp0[:, M.by].cpu().numpy()
        for name, _ in labelsets:
            single, scrub = erasers[name]
            for c, top in CONDS.items():
                if top is None:
                    hooks = {DAS_SITE: lambda v: apply_erasers(single, ef, v)}
                else:
                    hooks = {s: (lambda v, s=s: apply_erasers(scrub[s], ef, v)) for s in range(top + 1)}
                lp, _ = M.forward(bp, hooks)
                cols[f"lp_by__{name}__{c}"][bt] = lp[:, M.by].cpu().numpy()
                cols[f"kl__{name}__{c}"][bt] = (p0 * (lp0 - lp)).sum(-1).cpu().numpy()
        if bi % 20 == 0:
            log(f"eval batch {bi + 1}/{len(batches)}")

    res = P[["item", "side", "cls", "verb", "verb_pair", "verb_band", "context_id", "context_band", "prompt"]].copy()
    for k, v in cols.items():
        res[k] = v
    if args.reference_items and Path(args.reference_items).exists():  # the in-sample run's per-prompt readout
        R = pd.read_csv(args.reference_items)
        assert (R.prompt.to_numpy() == res.prompt.to_numpy()).all()
        checks["natural_vs_reference_max_abs"] = float((R.lp_by__natural - res.lp_by__natural).abs().max())
        if args.with_true:
            checks["true_vs_insample_max_abs"] = {c: float((R[f"lp_by__{c}"] - res[f"lp_by__true__{c}"]).abs().max())
                                                  for c in CONDS}
        checks["reference_items_sha256"] = sha(args.reference_items)
    res.to_csv(out / "items.csv.gz", index=False)
    pd.DataFrame(lab_rows).to_csv(out / "labels.csv", index=False)
    pd.DataFrame(diag).to_csv(out / "fit_diagnostics.csv", index=False)
    (out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    meta = {"args": vars(args), "runner_settings": M.R.settings, "n_prompts": N, "labelsets": [n for n, _ in labelsets],
            "conditions": CONDS, "scores_sha256": sha(args.scores), "minutes": (time.time() - t0) / 60}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    log(f"checks {checks}")
    log(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default="results/passive_band_cross/pythia14b_scores.csv")
    ap.add_argument("--v2-pairs", default="data/matched_passives_v2/pairs.jsonl")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-folds", type=int, default=5, help="only for load_items' fold column (unused here)")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--perms", default="0-19", help="permutation indices a-b (inclusive)")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--with-true", action="store_true", help="also run the true labels (reproduction check)")
    ap.add_argument("--reference-items", default="", help="items.csv.gz of the in-sample run")
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--max-contexts", type=int, default=0)
    ap.add_argument("--out-dir", default="results/pythia/leace_permutation")
    run(ap.parse_args())
