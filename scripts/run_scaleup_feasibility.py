#!/usr/bin/env python3
"""Round 5, D10: memory and time for DAS on larger Pythia models (reports/round5/plan.md, D10). fp32, TF32 off.

One model per process. Measures: memory after load; forward throughput on the round-2 DAS active prompts; one
round-2 DAS training step (rank-1 rotation as run_das_round2.train: frozen model, Adam on the rotation only, the
patched M readout, BCE loss) at the given sites, with warm-up, CUDA synchronization and timed repetitions, peak
allocated / reserved memory; the evaluation pass (no grad) per batch. Falls back to half the batch on OOM.
The rotation updates are discarded; nothing scientific is produced.

Output: --out JSON.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_das_round2 import Runner


def run(args):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.dtype = "float32"
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    r = Runner(args)
    torch.cuda.synchronize()
    rep = {"model": args.model, "load_s": round(time.time() - t0, 1),
           "load_alloc_GiB": torch.cuda.memory_allocated() / 2 ** 30,
           "layers": r.model.config.num_hidden_layers, "hidden": r.model.config.hidden_size}
    items = pd.read_csv(args.items)
    prompts = items.prompt.tolist()[:1024]
    # forward throughput (natural pass with hidden states at every site)
    torch.cuda.synchronize()
    t1 = time.time()
    _, _, reps = r.natural(prompts, batch=args.eval_batch)
    torch.cuda.synchronize()
    rep["forward_prompts_per_s"] = len(prompts) / (time.time() - t1)
    rep["sites"] = {}
    rng = np.random.default_rng(0)
    for site in [int(x) for x in args.sites.split(",")]:
        bs = args.batch
        while True:
            try:
                torch.cuda.reset_peak_memory_stats()
                raw = torch.nn.Parameter(torch.randn(reps.shape[-1], 1, device=r.device) * 0.02)
                opt = torch.optim.Adam([raw], lr=1e-2)
                times = []
                for k in range(args.warmup + args.reps):
                    b = rng.integers(0, len(prompts), bs)
                    j = rng.integers(0, len(prompts), bs)
                    y = torch.tensor(rng.integers(0, 2, bs).astype(float), device=r.device, dtype=torch.float32)
                    torch.cuda.synchronize()
                    ts = time.time()
                    basis, _ = torch.linalg.qr(raw)
                    m, _ = r.patched([prompts[x] for x in b], reps[j, site], site, basis)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(m, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    torch.cuda.synchronize()
                    if k >= args.warmup:
                        times.append(time.time() - ts)
                peak_a, peak_r = torch.cuda.max_memory_allocated() / 2 ** 30, torch.cuda.max_memory_reserved() / 2 ** 30
                # evaluation pass (no grad) per batch of eval_batch swaps
                with torch.no_grad():
                    basis = torch.linalg.qr(raw.detach())[0]
                    tev = []
                    for k in range(3):
                        b = rng.integers(0, len(prompts), args.eval_batch)
                        torch.cuda.synchronize()
                        ts = time.time()
                        r.patched([prompts[x] for x in b], reps[b, site], site, basis)
                        torch.cuda.synchronize()
                        tev.append(time.time() - ts)
                rep["sites"][str(site)] = {"batch": bs, "step_s": float(np.median(times)), "peak_alloc_GiB": peak_a,
                                           "peak_reserved_GiB": peak_r, "eval_batch": args.eval_batch,
                                           "eval_s_per_batch": float(np.median(tev))}
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs <= 4:
                    rep["sites"][str(site)] = {"error": "OOM at batch 4"}
                    break
                bs //= 2
        print(json.dumps({site: rep["sites"][str(site)]}), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sites", required=True, help="comma-separated residual sites (layer outputs)")
    ap.add_argument("--items", default="results/das_round2/final_strict/items.csv")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval-batch", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    run(ap.parse_args())
