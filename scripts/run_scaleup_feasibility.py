#!/usr/bin/env python3
"""Round 5, D10: memory and time for DAS on larger Pythia models (reports/round5/plan.md, D10). fp32, TF32 off.

One model per process. Measures (each with warm-up, CUDA synchronization and timed repetitions; peaks taken from a
fresh allocator baseline per measurement):
- memory after load;
- forward throughput on the passive-test prompts (natural pass with hidden states);
- one round-2 DAS training step (rank-1 rotation as run_das_round2.train: frozen model, Adam on the rotation only,
  patched M readout, BCE loss) at the given sites, on the round-2 active prompts; batch halved on OOM;
- the evaluation pass (no grad) per batch of patched swaps; batch halved on OOM.
The rotation updates are discarded; nothing scientific is produced.

Output: --out JSON.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_das_round2 import Runner


def timed(torch, fn, warmup, reps):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.time()
        fn()
        torch.cuda.synchronize()
        out.append(time.time() - t)
    return float(np.median(out))


def with_fallback(torch, measure, bs, floor=4):
    """measure(bs) -> dict; halves bs on OOM, with locals released between attempts."""
    while True:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            res = measure(bs)
            res.update({"batch": bs, "peak_alloc_GiB": torch.cuda.max_memory_allocated() / 2 ** 30,
                        "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2 ** 30})
            return res
        except torch.cuda.OutOfMemoryError:
            if bs <= floor:
                return {"error": f"OOM at batch {bs}"}
            bs //= 2


def run(args):
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.dtype = "float32"
    t0 = time.time()
    r = Runner(args)
    torch.cuda.synchronize()
    rep = {"model": args.model, "load_s": round(time.time() - t0, 1), "load_alloc_GiB": torch.cuda.memory_allocated() / 2 ** 30,
           "layers": r.model.config.num_hidden_layers, "hidden": r.model.config.hidden_size}
    passive = pd.read_csv(args.passive_items).good_prompt.drop_duplicates().tolist()[:1024]
    active = pd.read_csv(args.items).prompt.tolist()
    rng = np.random.default_rng(0)

    def fwd(bs):
        n = 4 * bs
        s = timed(torch, lambda: r.natural(passive[:n], batch=bs), 1, 3)
        return {"prompts_per_s": n / s}

    rep["forward_passive"] = with_fallback(torch, fwd, args.eval_batch)
    _, _, reps = r.natural(active, batch=64)  # source states at every site (active prompts)
    rep["sites"] = {}
    for site in [int(x) for x in args.sites.split(",")]:
        def train(bs):
            raw = torch.nn.Parameter(torch.randn(reps.shape[-1], 1, device=r.device) * 0.02)
            opt = torch.optim.Adam([raw], lr=1e-2)

            def step():
                b, j = rng.integers(0, len(active), bs), rng.integers(0, len(active), bs)
                y = torch.tensor(rng.integers(0, 2, bs).astype(np.float32), device=r.device)
                basis, _ = torch.linalg.qr(raw)
                m, _ = r.patched([active[x] for x in b], reps[j, site], site, basis)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(m, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

            return {"step_s": timed(torch, step, args.warmup, args.reps)}

        def evaluate(bs):
            basis = torch.linalg.qr(torch.randn(reps.shape[-1], 1, device=r.device))[0]

            def ev():
                with torch.no_grad():
                    b = rng.integers(0, len(active), bs)
                    r.patched([active[x] for x in b], reps[b, site], site, basis)

            return {"eval_s": timed(torch, ev, 1, 3)}

        rep["sites"][str(site)] = {"train": with_fallback(torch, train, args.batch),
                                   "eval": with_fallback(torch, evaluate, args.eval_batch)}
        print(json.dumps({site: rep["sites"][str(site)]}), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sites", required=True, help="comma-separated residual sites (layer outputs)")
    ap.add_argument("--items", default="results/das_round2/final_strict/items.csv")
    ap.add_argument("--passive-items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval-batch", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    run(ap.parse_args())
