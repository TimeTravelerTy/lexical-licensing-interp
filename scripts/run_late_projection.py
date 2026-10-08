#!/usr/bin/env python3
"""Late-layer projection onto the site-8 direction (round3_plan.md, A2). No patching.

One unpatched forward pass (bf16) over every passive-test prompt; the
last-token residual at sites 4..23 (site s = hidden_states[s], the output of
layer s - 1; hidden_states[24] is after the final LayerNorm and is not used)
is projected onto the 15 fold bases of
- d_8 and d_17 at every site (`p8`, `p17`: [prompt, site, basis]);
- the own-site d_s at the other DAS sites 4, 6, 10, 12, 14, 16 (`pown`,
  descriptive; at 8 and 17 the own site is p8 / p17).
Raw dot products: sign alignment, cross-fitting, the fixed d_8-orthogonal-
to-d_17 read-out and the metrics are computed in `analyze_late_projection.py`.
Bases come from `bases_rank1.npz`; the basis order is s{split}_f{fold}.

Check: the site-8 projections onto d_8 must match the passive test's
`projections_site8.parquet` (same prompts, same bases) up to bf16 batch noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_passive_test import PassiveRunner

SITES = list(range(4, 24))
OWN = (4, 6, 10, 12, 14, 16)
KEYS = [(k, f) for k in range(3) for f in range(5)]


def run(args):
    import torch

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz = np.load(args.bases)
    B = {s: torch.tensor(np.stack([npz[f"das_site{s}_s{k}_f{f}"].reshape(-1) for k, f in KEYS], 1),
                         dtype=torch.float32) for s in (8, 17) + OWN}
    runner = PassiveRunner(args)
    B = {s: v.to(runner.device) for s, v in B.items()}
    P = pd.read_csv(args.prompts)
    prompts = P.prompt.tolist()
    p8 = np.empty((len(P), len(SITES), 15), np.float32)
    p17 = np.empty((len(P), len(SITES), 15), np.float32)
    pown = np.empty((len(P), len(OWN), 15), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(P), args.batch):
            enc, anchors = runner.encode(prompts[i:i + args.batch])
            o = runner.model(**enc, output_hidden_states=True, use_cache=False)
            rows = torch.arange(len(anchors), device=runner.device)
            x = torch.stack([o.hidden_states[s][rows, anchors].float() for s in SITES], 1)  # [b, sites, d]
            sl = slice(i, i + len(anchors))
            p8[sl] = (x @ B[8]).cpu().numpy()
            p17[sl] = (x @ B[17]).cpu().numpy()
            for j, s in enumerate(OWN):
                pown[sl, j] = (x[:, SITES.index(s)] @ B[s]).cpu().numpy()
    print(f"forward: {len(P)} prompts in {time.time() - t0:.0f}s", flush=True)
    cos = [float(B[8][:, j] @ B[17][:, j]) for j in range(15)]
    chk = {}
    if Path(args.check).exists():
        ref = pd.read_parquet(args.check).set_index("pid").loc[P.pid]
        cols = [f"s{k}_f{f}" for k, f in KEYS]
        diff = np.abs(ref[cols].to_numpy() - p8[:, SITES.index(8)])
        chk = {"site8_vs_passive_test_max_abs": float(diff.max()), "median_abs": float(np.median(diff)),
               "scale_median_abs_proj": float(np.median(np.abs(ref[cols].to_numpy())))}
        print(f"check: {chk}", flush=True)
    np.savez_compressed(out / "late_projection.npz", pid=P.pid.to_numpy(), sites=np.array(SITES), own=np.array(OWN),
                        keys=np.array([f"s{k}_f{f}" for k, f in KEYS]), p8=p8, p17=p17, pown=pown)
    meta = {"args": vars(args), "prompts": len(P), "sites": SITES, "own_sites": OWN, "cos_d8_d17": cos,
            "check": chk, "bases_sha256": hashlib.sha256(Path(args.bases).read_bytes()).hexdigest()}
    (out / "late_projection_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", default="data/das_round2/passive_test/prompts.csv")
    ap.add_argument("--bases", default="results/das_round2/bases_rank1.npz")
    ap.add_argument("--check", default="results/das_round2/passive_test/projections_site8.parquet")
    ap.add_argument("--out-dir", default="results/das_round2/round3")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=1024)
    run(ap.parse_args())
