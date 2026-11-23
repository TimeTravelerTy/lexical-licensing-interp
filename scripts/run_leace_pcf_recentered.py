#!/usr/bin/env python3
"""Stored cross-fitted passive erasers (P-cf, run_leace_necessity.py) re-centered on the evaluated fold's passive mean
(follow-up in reports/pythia/leace_crossframe_rules.md, declared 2026-11-18 after the full cross-frame run).

The stored P-cf eraser for fold f has bias = mean of the training folds' passives; A-cf (c) in the cross-frame run is
centered on fold f's own passive mean. Here the P-cf erasers are rebuilt from erasers.npz (rank-1 edit / read / bias)
and evaluated on the same passive items:
  u  as stored (reproduction check against the stored per-prompt log P(by))
  c  bias = fold f's label-free passive mean at that site; scrub: the mean under re-centered erasers 0..L-1,
     computed sequentially on the passives (as A-cf c in run_leace_crossframe.py)
Single sites 0..23 and scrub prefixes 0..k (k = 0..23). Outputs: items_passive.csv.gz, checks.json, meta.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_leace_necessity as nec

N_SITES = nec.N_SITES


def run(args):
    import torch
    from concept_erasure import LeaceEraser

    args.dtype = "float32"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)
    out, ref = Path(args.out_dir), Path(args.ref_cf)
    out.mkdir(parents=True, exist_ok=True)
    M = nec.Model(args)
    dev, d, tok = M.dev, M.d, M.R.tok
    pas, P, pairs = nec.load_items(args, tok)
    stored_folds = pd.read_csv(ref / "pairs_folds.csv").set_index("verb_pair").fold
    checks = {}
    if args.max_pairs or args.max_contexts:  # smoke subset: take each pair's stored fold
        P["fold"] = P.verb_pair.map(stored_folds).to_numpy()
        assert P.fold.notna().all()
    else:
        checks["folds_equal_stored"] = bool(len(stored_folds) == len(pairs) and
                                            (pairs.set_index("verb_pair").fold.reindex(stored_folds.index)
                                             == stored_folds).all())
        assert checks["folds_equal_stored"]
    meta_ref = json.loads((ref / "meta.json").read_text())
    assert not meta_ref["args"].get("insample"), "reference must be the cross-fitted run"
    N, sites = len(P), list(range(N_SITES))
    fold_np = P.fold.to_numpy().astype(np.int64)
    F = int(np.load(ref / "erasers.npz")["single_edit"].shape[0])
    assert F == args.n_folds and set(np.unique(fold_np)) <= set(range(F))
    pprompts = P.prompt.tolist()

    def batches_of(idx, ntok):
        idx = np.asarray(idx)
        idx = idx[np.argsort(np.asarray(ntok)[idx], kind="stable")]
        return [idx[i:i + args.batch_size] for i in range(0, len(idx), args.batch_size)]
    p_batches = batches_of(np.arange(N), P.n_tokens.to_numpy())

    # rebuild the stored erasers: x' = x - (read . (x - bias)) edit
    Z = np.load(ref / "erasers.npz")
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=dev, dtype=torch.float64)
    ers = {}
    for kind in ("single", "scrub"):
        e, r, b = Z[f"{kind}_edit"], Z[f"{kind}_read"], Z[f"{kind}_bias"]
        assert e.shape == r.shape == b.shape == (F, N_SITES, d), (kind, e.shape)
        ers[("u", kind)] = {s: {f: LeaceEraser(T(e[f, s])[:, None], T(r[f, s])[None, :], T(b[f, s]))
                                for f in range(F)} for s in sites}

    # c: single sites, natural fold means
    psum, pcnt = np.zeros((F, N_SITES, d)), np.bincount(fold_np, minlength=F).astype(float)
    for b in p_batches:
        cap = {}
        M.forward([pprompts[i] for i in b], {s: (lambda v, s=s: cap.__setitem__(s, v.clone())) for s in sites},
                  logits=False)
        for s in sites:
            x = cap[s].double().cpu().numpy()
            for f in range(F):
                psum[f, s] += x[fold_np[b] == f].sum(0)
    recenter = lambda E, mu: LeaceEraser(E.proj_left, E.proj_right, T(mu))
    ers[("c", "single")] = {s: {f: (recenter(ers[("u", "single")][s][f], psum[f, s] / pcnt[f]) if pcnt[f]
                                 else ers[("u", "single")][s][f]) for f in range(F)}
                            for s in sites}
    # c: scrub, sequential fold means under the re-centered erasers
    sc = ers[("c", "scrub")] = {}
    for L in sites:
        msum, mcnt = np.zeros((F, d)), np.zeros(F)
        for b in p_batches:
            cap = {}
            ef = torch.as_tensor(fold_np[b], device=dev)
            hooks = {s: (lambda v, s=s: nec.apply_erasers(sc[s], ef, v)) for s in range(L)}

            def capture(v):
                cap["x"] = v.clone()
                raise nec.Stop
            hooks[L] = capture
            M.forward([pprompts[i] for i in b], hooks)
            x = cap["x"].double().cpu().numpy()
            for f in range(F):
                m = fold_np[b] == f
                msum[f] += x[m].sum(0)
                mcnt[f] += m.sum()
        sc[L] = {f: (recenter(ers[("u", "scrub")][L][f], msum[f] / mcnt[f]) if mcnt[f] else ers[("u", "scrub")][L][f])
                 for f in range(F)}
    log("re-centered erasers built")

    conds = [(f"Pcf_{var}_single_{s}", var, "single", s) for var in ("c", "u") for s in sites]
    conds += [(f"Pcf_{var}_scrub_0_{k}", var, "scrub", k) for var in ("c", "u") for k in sites]
    lp_by = {c[0]: np.zeros(N) for c in conds}
    lp_by["natural"] = np.zeros(N)
    kl = {c[0]: np.zeros(N) for c in conds}
    for bi, b in enumerate(p_batches):
        bp = [pprompts[i] for i in b]
        ef = torch.as_tensor(fold_np[b], device=dev)
        lp0, _ = M.forward(bp, {})
        lp_by["natural"][b] = lp0[:, M.by].cpu().numpy()
        p0 = lp0.exp()
        for col, var, kind, k in conds:
            E = ers[(var, kind)]
            site_list = [k] if kind == "single" else list(range(k + 1))
            lp, _ = M.forward(bp, {s: (lambda v, s=s: nec.apply_erasers(E[s], ef, v)) for s in site_list})
            lp_by[col][b] = lp[:, M.by].cpu().numpy()
            kl[col][b] = (p0 * (lp0 - lp)).sum(-1).cpu().numpy()
        if bi % 10 == 0:
            log(f"passive eval batch {bi + 1}/{len(p_batches)}")

    # reproduction of the stored per-prompt log P(by) by the as-stored (u) erasers
    R = pd.read_csv(ref / "items.csv.gz")
    kc = ["verb_pair", "context_id", "side"]  # a verb can sit in several pairs, so prompts are not unique
    rk, pk = pd.MultiIndex.from_frame(R[kc]), pd.MultiIndex.from_frame(P[kc])
    assert rk.is_unique and pk.is_unique
    pos = rk.get_indexer(pk)
    assert (pos >= 0).all(), "passive items missing from the stored run"
    assert (R.prompt.to_numpy()[pos] == P.prompt.to_numpy()).all()
    rep = {"natural": float(np.abs(R.lp_by__natural.to_numpy()[pos] - lp_by["natural"]).max())}
    for col, var, kind, k in conds:
        if var != "u":
            continue
        sc_name = f"single_{k}" if kind == "single" else ("scrub_all" if k == N_SITES - 1 else f"scrub_0_{k}")
        if f"lp_by__{sc_name}" in R.columns:
            rep[col] = float(np.abs(R[f"lp_by__{sc_name}"].to_numpy()[pos] - lp_by[col]).max())
    checks["stored_repro_max_abs"] = max(rep.values())
    checks["stored_repro_n"] = len(rep)
    checks["stored_repro_worst"] = max(rep, key=rep.get)
    log(f"reproduction of the stored P-cf: max |Δ| {checks['stored_repro_max_abs']:.2e} over {len(rep)} columns "
        f"(worst {checks['stored_repro_worst']})")

    cols = {**{f"lp_by__{c}": v for c, v in lp_by.items()}, **{f"kl__{c}": v for c, v in kl.items()}}
    keep = [c for c in ("item", "side", "prompt", "verb", "cls", "fold", "verb_pair", "verb_band", "context_id",
                        "context_band") if c in P.columns]
    pd.concat([P[keep].reset_index(drop=True), pd.DataFrame(cols)], axis=1).to_csv(out / "items_passive.csv.gz",
                                                                                   index=False)
    (out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    meta = {"args": vars(args), "n_prompts": N, "n_pairs": int(P.verb_pair.nunique()), "sites": sites,
            "smoke": bool(args.max_pairs or args.max_contexts), "ref_erasers_sha256": nec.sha(ref / "erasers.npz"),
            "ref_items_sha256": nec.sha(ref / "items.csv.gz"), "minutes": (time.time() - t0) / 60}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
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
    ap.add_argument("--ref-cf", required=True, help="stored leace_necessity results dir")
    ap.add_argument("--max-pairs", type=int, default=0, help="smoke test: pairs per verb band")
    ap.add_argument("--max-contexts", type=int, default=0, help="smoke test: contexts kept")
    ap.add_argument("--out-dir", default="results/pythia/leace_pcf_recentered")
    run(ap.parse_args())
