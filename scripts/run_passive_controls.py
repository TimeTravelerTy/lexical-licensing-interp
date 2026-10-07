#!/usr/bin/env python3
"""Passive-side controls for the site-8 transfer result (post hoc; spec: passive_test_plan.md, last section).

Same plan rows as the main test, restricted to the primary population (plain
bad verb), bad passive bases, conditions T and I.

- shuffled_labels: the 15 shuffled-label DAS bases of the site, retrained
  exactly as the control in `run_das_round2.py final` (same label
  permutation seeds and training seeds); bases are saved. All 3 splits.
- random / random_normmatched (`--controls` picks which): per draw, one random rank-1 direction per
  fold basis; raw, and with each patch's displacement rescaled to the norm
  the DAS basis would produce. Split-0 rows only.

Outputs (`--out-dir`):
- `controls_shuffled_site{s}.parquet`: plan row + readouts (all rows).
- `controls_random_site{s}.parquet`: per control x draw x item, mean
  Delta readout after T donors minus after I donors (D per item).
- `shuffled_bases_site{s}.pt`, `controls_meta_site{s}.json` (held-out
  active metrics of the retrained shuffled bases, for comparison with the
  original run's summary.csv).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_das_round2 as r2
from run_passive_test import READ, SINGLE, PassiveRunner


def upgrade(runner):
    """Turn a run_das_round2.Runner into a PassiveRunner (adds the extra readout ids)."""
    runner.__class__ = PassiveRunner
    for k, t in SINGLE.items():
        runner.ids[k] = runner.torch.tensor(runner.tok.encode(t, add_special_tokens=False), device=runner.device)
    return runner


def run(args):
    import torch

    cfg = json.loads(Path(args.config).read_text())
    site, epochs = cfg["site"], cfg["epochs"]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    controls = set(args.controls.split(","))
    runner, pairs, items, reps, nat_items, _ = r2.setup(args)
    runner = upgrade(runner)
    labels = items.cls.to_numpy()

    # ---- shuffled-label bases, exactly as in run_final
    t0 = time.time()
    shuffled, summ = {}, []
    for split in range(args.n_splits if "shuffled" in controls else 0):
        assign = r2.folds(pairs, args.n_folds, split)
        for f in range(args.n_folds):
            tr, te = r2.fold_indices(items, assign, f)
            vcls = items.groupby("verb").cls.first()
            tv = sorted(set(items.verb.values[tr]))
            sh = dict(zip(tv, np.random.default_rng(100 * split + f).permutation(vcls[tv].to_numpy())))
            perm = np.array([sh.get(v, c) for v, c in zip(items.verb, labels)])
            sb, _ = r2.train(runner, items, reps, tr, perm, site, 1, epochs, args, seed=777 + 100 * split + f)
            shuffled[(split, f)] = sb
            df = r2.evaluate(runner, items, reps, nat_items, te, site, sb)
            summ.append({"split": split, "fold": f, **r2.summarize(df, items)})
    if shuffled:
        torch.save({f"r1_s{k[0]}_f{k[1]}": v.cpu() for k, v in shuffled.items()},
                   out / f"shuffled_bases_site{site}.pt")
        print(f"shuffled bases trained in {time.time() - t0:.0f}s", flush=True)

    das = {(int(k.split("_s")[1].split("_")[0]), int(k.split("_f")[1])): v.float().to(runner.device)
           for k, v in torch.load(Path(args.das_dir) / "bases.pt").items() if k.startswith("r1_")}

    # ---- plan subset and states
    it = pd.read_csv(args.items)
    prim = set(it[it.bad_class == "plain"].item_id)
    plan = pd.read_csv(args.plan)
    plan = plan[plan.item_id.isin(prim) & (plan.side == "bad") & plan.cond.isin(["T", "I"])].reset_index(drop=True)
    P = pd.read_csv(args.prompts)
    prompts = P.prompt.tolist()
    need = np.unique(np.concatenate([plan.base.to_numpy(), plan.donor.to_numpy()]))
    states = torch.zeros(len(P), runner.model.config.hidden_size, device=runner.device)
    nat = {k: np.full(len(P), np.nan, np.float32) for k in READ}
    with torch.no_grad():
        for i in range(0, len(need), args.batch):
            ids = need[i:i + args.batch]
            enc, anchors = runner.encode([prompts[j] for j in ids])
            o = runner.model(**enc, output_hidden_states=True, use_cache=False)
            _, parts = runner.readout(o.logits, anchors, full=True)
            rows = torch.arange(len(anchors), device=runner.device)
            states[torch.as_tensor(ids, device=runner.device)] = o.hidden_states[site][rows, anchors].float()
            for k in READ:
                nat[k][ids] = parts[k].cpu().numpy()

    def patch_rows(sub, basis_of, match_norm_of=None):
        res = {k: np.empty(len(sub), np.float32) for k in READ}
        with torch.no_grad():
            for key, g in sub.groupby(["split", "fold"]):
                ix = np.flatnonzero((sub.split == key[0]).to_numpy() & (sub.fold == key[1]).to_numpy())
                base, donor = sub.base.to_numpy()[ix], sub.donor.to_numpy()[ix]
                for i in range(0, len(ix), args.batch):
                    sl = slice(i, i + args.batch)
                    _, parts = runner.patched([prompts[j] for j in base[sl]],
                                              states[torch.as_tensor(donor[sl], device=runner.device)], site,
                                              basis_of[key], full=True,
                                              match_norm=None if match_norm_of is None else match_norm_of[key])
                    for k in READ:
                        res[k][ix[sl]] = parts[k].cpu().numpy()
        return res

    # ---- shuffled-label transfer (all splits)
    if shuffled:
        t0 = time.time()
        res = patch_rows(plan, shuffled)
        pd.DataFrame({"row": plan.row, **res}).to_parquet(out / f"controls_shuffled_site{site}.parquet", index=False)
        print(f"shuffled transfer: {len(plan)} patches in {time.time() - t0:.0f}s", flush=True)

    # ---- random directions (split 0)
    sub = plan[plan.split == 0].reset_index(drop=True)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    keys = sorted(k for k in das if k[0] == 0)
    rows = []
    t0 = time.time()
    for draw in range(args.n_random):
        rnd = {k: torch.linalg.qr(torch.randn(das[k].shape, generator=gen))[0].to(runner.device) for k in keys}
        for name, mn in [(n, m) for n, m in (("random", None), ("random_normmatched", das)) if n in controls]:
            res = patch_rows(sub, rnd, mn)
            d = pd.DataFrame({"item_id": sub.item_id, "cond": sub.cond})
            for k in READ:
                d[k] = res[k] - nat[k][sub.base.to_numpy()]
            g = d.groupby(["item_id", "cond"])[list(READ)].mean().unstack("cond")
            D = pd.DataFrame({k: g[(k, "T")] - g[(k, "I")] for k in READ}).reset_index()
            D.insert(0, "draw", draw)
            D.insert(0, "control", name)
            rows.append(D)
        if draw % 10 == 9:
            print(f"random draw {draw + 1}/{args.n_random} ({time.time() - t0:.0f}s)", flush=True)
    pd.concat(rows, ignore_index=True).to_parquet(out / f"controls_random_site{site}.parquet", index=False)
    meta = {"site": site, "epochs": epochs, "plan_rows": len(plan), "random_rows_per_draw": len(sub),
            "n_random": args.n_random, "seed": args.seed, "shuffled_heldout_active": summ,
            "das_bases_sha256": hashlib.sha256((Path(args.das_dir) / "bases.pt").read_bytes()).hexdigest(),
            "args": vars(args)}
    (out / f"controls_meta_site{site}.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="results/das_round2/frozen_config_site8.json")
    ap.add_argument("--das-dir", default="results/das_round2/final_strict_site8")
    ap.add_argument("--pairs", default="data/das_round2/train_pairs.csv")
    ap.add_argument("--items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--prompts", default="data/das_round2/passive_test/prompts.csv")
    ap.add_argument("--plan", default="data/das_round2/passive_test/plan.csv.gz")
    ap.add_argument("--out-dir", default="results/das_round2/passive_test")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--n-random", type=int, default=100)
    ap.add_argument("--n-cross", type=int, default=4)
    ap.add_argument("--n-same", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--controls", default="shuffled,random,random_normmatched",
                    help="comma list from shuffled, random, random_normmatched")
    run(ap.parse_args())
