#!/usr/bin/env python3
"""Downstream re-derivation after LEACE scrubbing (Pythia-1.4B fp32), with the saved rank-1 erasers of the in-sample
and cross-fitted runs (run_leace_necessity.py erasers.npz). Rules: reports/pythia/leace_learning_curve_rules.md,
Parts B and C, committed before any run.

Part B (site 20, split by verb token count; a prompt is "single" if its participle is one token):
- |cos| between the single-site eraser and the post-scrub eraser at site 20, refit on each subset (all / single /
  multi): single on natural site-20 states, post-scrub on site-20 states with the saved erasers 0..19 applied.
  In-sample: all of the subset's prompts. Cross-fitted: per fold f, the subset's fold-f training prompts with fold f's
  erasers 0..19 applied. The all-verbs refit must reproduce the saved erasers.
- Linear-probe predictions at site 20 after scrubbing 0..19 (before eraser 20): in-sample (verb-held-out split, seed 1)
  and cross-fitted (nested splits, seed 1); the analysis splits accuracy by subset. Natural site 20 likewise.

Part C: linear-probe predictions at sites 12..23 after scrub 0..12 (site 12 erased, 13..23 not), in-sample and
cross-fitted, with the natural states at the same sites and splits.

Checks: per-prompt log P(by) of scrub 0..12 reproduces each run's items.csv.gz (loading the saved erasers), natural
log P(by) likewise; probe accuracies of natural states and of the erased site-12 states reproduce each run's
probes.csv.

Outputs (--out-dir): preds.csv.gz (prompt metadata + per-prompt predictions, columns
pred__<fitting>__<states>__s<site>), cosines.csv, checks.json, meta.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_leace_necessity import Model, Stop, apply_erasers, fit_stats, inner_splits, load_items, rank1, sha
from run_leace_learning_curve import META_COLS, probe_pred

SITE = 20
TOP = 12
DOWN = list(range(TOP, 24))  # 12 (erased) .. 23


def load_erasers(path, torch, dev):
    from concept_erasure import LeaceEraser

    z = np.load(path)
    F = z["scrub_edit"].shape[0]
    t = lambda a: torch.as_tensor(a, device=dev, dtype=torch.float64)
    out = {c: {s: {f: LeaceEraser(t(z[f"{c}_edit"][f, s])[:, None], t(z[f"{c}_read"][f, s])[None, :],
                                  t(z[f"{c}_bias"][f, s])) for f in range(F)} for s in range(24)}
           for c in ("single", "scrub")}
    return out, F, z


def abs_cos(a, b):
    return float(abs((a * b).sum()) / (np.linalg.norm(a) * np.linalg.norm(b)))


def run(args):
    import torch
    from concept_erasure import LeaceFitter

    args.dtype = "float32"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    M = Model(args)
    dev, d = M.dev, M.d
    _, P, pairs = load_items(args, M.R.tok)
    N = len(P)
    prompts = P.prompt.tolist()
    ntok = P.n_tokens.to_numpy()
    cls = P.cls.to_numpy()
    z_t = torch.as_tensor(cls, device=dev, dtype=torch.float64)
    pair_np = P.verb_pair.to_numpy()
    fold_np = P.fold.to_numpy()
    single_tok = (P.ntok_part == 1).to_numpy()
    verb_np = P.verb.to_numpy()
    subsets = {"all": np.ones(N, bool), "single": single_tok, "multi": ~single_tok}

    def batches(idx):
        idx = idx[np.argsort(ntok[idx], kind="stable")]
        return [idx[i:i + args.batch_size] for i in range(0, len(idx), args.batch_size)]

    runs = {"insample": (Path(args.insample), np.zeros(N, dtype=np.int64)),
            "crossfit": (Path(args.crossfit), fold_np)}
    ers, Fs, raw = {}, {}, {}
    for k, (dir_, ef_np) in runs.items():
        ers[k], Fs[k], raw[k] = load_erasers(dir_ / "erasers.npz", torch, dev)
        assert ef_np.max() < Fs[k]
    checks = {"erasers_sha256": {k: sha(dir_ / "erasers.npz") for k, (dir_, _) in runs.items()}}

    # natural pass: states at sites 12..23, log P(by)
    nat = np.zeros((N, len(DOWN), d), dtype=np.float32)
    lp_nat = np.zeros(N)
    for b in batches(np.arange(N)):
        cap = {}
        lp, _ = M.forward([prompts[i] for i in b], {s: (lambda v, s=s: cap.__setitem__(s, v.clone())) for s in DOWN})
        for j, s in enumerate(DOWN):
            nat[b, j] = cap[s].cpu().numpy()
        lp_nat[b] = lp[:, M.by].cpu().numpy()
    log("natural pass done")

    down, pre20, chk_lp = {}, {}, {}
    for k, (dir_, ef_np) in runs.items():
        sc = ers[k]["scrub"]
        X = np.zeros((N, len(DOWN), d), dtype=np.float32)
        X20 = np.zeros((N, d), dtype=np.float32)
        lp12 = np.zeros(N)
        for b in batches(np.arange(N)):
            bp = [prompts[i] for i in b]
            ef = torch.as_tensor(ef_np[b], device=dev)
            cap = {}

            def h(v, s):  # scrub 0..12, capture 12 (erased) and 13..23 (untouched)
                new = apply_erasers(sc[s], ef, v) if s <= TOP else None
                cap[s] = (new if new is not None else v).clone()
                return new
            lp, _ = M.forward(bp, {s: (lambda v, s=s: h(v, s)) for s in range(24)})
            for j, s in enumerate(DOWN):
                X[b, j] = cap[s].cpu().numpy()
            lp12[b] = lp[:, M.by].cpu().numpy()
            cap20 = {}
            hooks = {s: (lambda v, s=s: apply_erasers(sc[s], ef, v)) for s in range(SITE)}

            def c20(v):
                cap20["x"] = v.clone()  # site 20 after erasers 0..19, before eraser 20
                raise Stop
            hooks[SITE] = c20
            M.forward(bp, hooks)
            X20[b] = cap20["x"].cpu().numpy()
        R = pd.read_csv(dir_ / "items.csv.gz")
        assert (R.prompt.to_numpy() == P.prompt.to_numpy()).all()
        chk_lp[k] = {"scrub_0_12_max_abs": float(np.abs(R.lp_by__scrub_0_12.to_numpy() - lp12).max()),
                     "natural_max_abs": float(np.abs(R.lp_by__natural.to_numpy() - lp_nat).max()),
                     "items_sha256": sha(dir_ / "items.csv.gz")}
        down[k], pre20[k] = X, X20
        log(f"{k}: scrub passes done, {chk_lp[k]}")
    checks["lp_by_reproduction"] = chk_lp

    # Part B: cosines at site 20 per subset
    def fit1(X, z):
        fitter = LeaceFitter(d, 1, device=dev, dtype=torch.float64)
        for i in range(0, len(X), 4096):
            fitter.update(torch.as_tensor(X[i:i + 4096], device=dev).double(), z[i:i + 4096])
        E, st = fit_stats(fitter, torch)
        if st["erased_rank"] != 1:
            return None, st
        return [v.double().cpu().numpy() for v in rank1(E)], st

    nat20 = nat[:, DOWN.index(SITE)]
    crow, repro = [], {}
    for k in runs:
        F = Fs[k]
        for f in range(F):
            if k == "insample":
                train = np.arange(N)
                Xs = pre20[k]  # the in-sample fitting distribution: all prompts, erasers 0..19 applied
            else:  # fold f's training prompts with fold f's erasers 0..19 applied
                train = np.flatnonzero(fold_np != f)
                Xs = np.zeros((N, d), dtype=np.float32)
                sc = ers[k]["scrub"]
                for b in batches(train):
                    ef = torch.full((len(b),), f, device=dev)
                    cap20 = {}
                    hooks = {s: (lambda v, s=s: apply_erasers(sc[s], ef, v)) for s in range(SITE)}

                    def c20(v):
                        cap20["x"] = v.clone()
                        raise Stop
                    hooks[SITE] = c20
                    M.forward([prompts[i] for i in b], hooks)
                    Xs[b] = cap20["x"].cpu().numpy()
            for sub, mk in subsets.items():
                idx = train[mk[train]]
                zt = z_t[torch.as_tensor(idx, device=dev)]
                vs, sts = fit1(nat20[idx], zt)
                vc, stc = fit1(Xs[idx], zt)
                r = {"fitting": k, "fold": f, "subset": sub, "n_prompts": len(idx),
                     "n_T_verbs": len(np.unique(verb_np[idx][cls[idx] == 1])),
                     "n_I_verbs": len(np.unique(verb_np[idx][cls[idx] == 0])),
                     "single_sv_top": sts["sv_top"], "scrub_sv_top": stc["sv_top"],
                     "single_rank": sts["erased_rank"], "scrub_rank": stc["erased_rank"]}
                if vs is not None and vc is not None:
                    r["cos_edit"], r["cos_read"] = abs_cos(vs[0], vc[0]), abs_cos(vs[1], vc[1])
                    if sub == "all":
                        repro[f"{k}_fold{f}"] = {
                            "single_edit": abs_cos(vs[0], raw[k]["single_edit"][f, SITE]),
                            "scrub_edit": abs_cos(vc[0], raw[k]["scrub_edit"][f, SITE]),
                            "saved_cos_edit": abs_cos(raw[k]["single_edit"][f, SITE], raw[k]["scrub_edit"][f, SITE])}
                crow.append(r)
            log(f"{k} fold {f}: " + ", ".join(f"{r['subset']} edit {r.get('cos_edit', np.nan):.3f} read "
                                               f"{r.get('cos_read', np.nan):.3f}" for r in crow[-3:]))
    checks["site20_refit_vs_saved_abs_cos"] = repro
    pd.DataFrame(crow).to_csv(out / "cosines.csv", index=False)

    # probes (Parts B and C), per-prompt predictions
    res = P[META_COLS].copy()
    res.insert(0, "row", np.arange(N))
    splits = {"insample": inner_splits(np.zeros(N, dtype=np.int64), pair_np, 5, seed=1),
              "crossfit": inner_splits(fold_np, pair_np, 5, seed=1)}
    for k in runs:
        sp = splits[k]
        res[f"pred__{k}__pre{SITE}__s{SITE}"] = probe_pred(pre20[k], cls, sp, torch, dev)
        for j, s in enumerate(DOWN):
            res[f"pred__{k}__scrub_0_12__s{s}"] = probe_pred(down[k][:, j], cls, sp, torch, dev)
            res[f"pred__{k}__natural__s{s}"] = probe_pred(nat[:, j], cls, sp, torch, dev)
        acc = {c: float((res[c] == cls).mean()) for c in res.columns if c.startswith(f"pred__{k}__")}
        log(f"{k} probes: " + ", ".join(f"{c.split('__', 2)[2]} {v:.3f}" for c, v in acc.items()))
        # stored probes (same splits and seeds): natural at 12..23 and the erased site-12 states (= scrub_all at 12)
        st = pd.read_csv(runs[k][0] / "probes.csv")
        st = st[st.probe == "linear"]
        cmp = {}
        for s in DOWN:
            cmp[f"natural_s{s}"] = (float(st[(st.condition == "natural") & (st.site == s)].acc.iloc[0]),
                                    acc[f"pred__{k}__natural__s{s}"])
        cmp[f"scrub_s{TOP}"] = (float(st[(st.condition == "scrub_all") & (st.site == TOP)].acc.iloc[0]),
                                acc[f"pred__{k}__scrub_0_12__s{TOP}"])
        checks[f"probe_reproduction_{k}"] = {"max_abs": max(abs(a - b) for a, b in cmp.values()),
                                             "pairs_stored_vs_now": cmp}
    res.to_csv(out / "preds.csv.gz", index=False)
    (out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    meta = {"args": vars(args), "runner_settings": M.R.settings, "n_prompts": N, "site": SITE, "top": TOP,
            "scores_sha256": sha(args.scores), "minutes": (time.time() - t0) / 60}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    log(f"checks: lp {chk_lp}; site-20 refit {repro}; probes "
        f"{ {k: checks[f'probe_reproduction_{k}']['max_abs'] for k in runs} }")
    log(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default="results/passive_band_cross/pythia14b_scores.csv")
    ap.add_argument("--v2-pairs", default="data/matched_passives_v2/pairs.jsonl")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--insample", default="results/pythia/leace_insample")
    ap.add_argument("--crossfit", default="results/pythia/leace_necessity")
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--max-contexts", type=int, default=0)
    ap.add_argument("--out-dir", default="results/pythia/leace_postscrub")
    run(ap.parse_args())
