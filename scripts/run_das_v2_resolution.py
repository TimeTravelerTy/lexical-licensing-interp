#!/usr/bin/env python3
"""Verb-sample resolution of a rank-1 DAS direction (rules: reports/v2_retrain_rules.md, measure 2).

At one site: for split 0, folds 0..n_folds_used-1, v2's training pool of that fold (stored training pairs of the fold
+ the kept new plain pairs of data/das_v2/train_pairs_v2.csv) is split into two disjoint random samples of n pairs
(n in --ns; --draws draws per n and fold). A direction is trained on each sample with the v2 recipe (stored epochs,
seed 50000 + 1000*draw + 100*fold + n and + 500 for the second sample) and evaluated on the stored held-out fold.
Writes resolution.csv (one row per draw: |cos(A, B)|, |cos| of each with the v2 and stored v bases of the fold,
held-out cross IIA of each) and meta.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import run_das_round2 as d2
from run_das_stability_ext import filtered


def run(a):
    import torch

    t0 = time.time()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = a.site
    stored, v2 = Path(a.stored_dir), Path(a.v2_dir)
    fm = json.loads((stored / "final_meta.json").read_text())
    vm = json.loads((v2 / "final_meta.json").read_text())
    assert fm["site"] == vm["site"] == s and not vm["v2"]["smoke"]
    import hashlib
    assert Path(vm["args"]["model"]).name == Path(a.model).name, (vm["args"]["model"], a.model)
    assert hashlib.sha256(Path(a.pairs).read_bytes()).hexdigest() == vm["v2"]["pairs_sha256"], "pairs differ from v2's"
    sa = fm["args"]
    args = SimpleNamespace(pairs=a.stored_pairs, model=a.model, device=a.device, dtype=sa["dtype"], lr=sa["lr"],
                           batch_size=sa["batch_size"], n_cross=sa["n_cross"], n_same=sa["n_same"], bos=sa.get("bos"),
                           fp32_unembed=sa.get("fp32_unembed"))
    runner, pairsS, itemsS, repsS, natS, droppedS = d2.setup(args)
    itemsP, repsP, natP, _, n_keep = filtered(runner, a.pairs, pairsS.pair_id)
    assert n_keep == vm["v2"]["n_new_pairs"], (n_keep, vm["v2"]["n_new_pairs"])
    folds_df = pd.read_csv(stored / "pairs_folds.csv")
    assign = dict(zip(folds_df.pair_id, folds_df.fold_split0))
    vb = {k: v.float() for k, v in torch.load(v2 / "bases.pt").items()}
    sb = {k: v.float() for k, v in torch.load(stored / "bases.pt").items() if k.startswith("r1_")}
    labels = itemsP.cls.to_numpy()
    rows = []
    for f in range(a.n_folds_used):
        trS, teS = d2.fold_indices(itemsS, assign, f)
        trP, _ = d2.fold_indices(itemsP, assign, f)
        pool = sorted(set(itemsP.pair_id.values[trP]))
        key = f"r1_s0_f{f}"
        for n in [int(x) for x in a.ns.split(",")]:
            assert 2 * n <= len(pool), (n, len(pool))
            for r in range(a.draws):
                rng = np.random.default_rng(1000 * r + 100 * f + n)
                pick = rng.permutation(pool)[: 2 * n]
                res = {"site": s, "fold": f, "n": n, "draw": r, "pool": len(pool)}
                dirs = []
                for h, half in enumerate((set(pick[:n]), set(pick[n:]))):
                    idx = trP[np.isin(itemsP.pair_id.values[trP], list(half))]
                    seed = 50000 + 1000 * r + 100 * f + n + 500 * h
                    b, _ = d2.train(runner, itemsP, repsP, idx, labels, s, 1, fm["epochs"], args, seed)
                    df = d2.evaluate(runner, itemsS, repsS, natS, teS, s, b)
                    u = b[:, 0].float().cpu()
                    dirs.append(u)
                    res[f"iia_{'AB'[h]}"] = d2.summarize(df, itemsS)["iia_cross"]
                    res[f"cos_v2_{'AB'[h]}"] = float(abs(u @ vb[key][:, 0]))
                    res[f"cos_v_{'AB'[h]}"] = float(abs(u @ sb[key][:, 0]))
                res["cos_AB"] = float(abs(dirs[0] @ dirs[1]))
                rows.append(res)
                print(f"[{time.time() - t0:6.0f}s] {res}", flush=True)
                pd.DataFrame(rows).to_csv(out / "resolution.csv", index=False)
    R = pd.DataFrame(rows)
    print(R.groupby("n")[["cos_AB", "cos_v2_A", "cos_v_A", "iia_A"]].median().round(3).to_string())
    (out / "meta.json").write_text(json.dumps({"args": vars(a), "n_new_pairs_kept": n_keep, "dropped_stored": droppedS,
                                               "minutes": (time.time() - t0) / 60}, indent=2, default=str) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", type=int, required=True)
    ap.add_argument("--stored-dir", required=True)
    ap.add_argument("--v2-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ns", default="10,20,30")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--n-folds-used", type=int, default=3)
    ap.add_argument("--pairs", default="data/das_v2/train_pairs_v2.csv")
    ap.add_argument("--stored-pairs", default="data/das_round2/train_pairs.csv")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    run(ap.parse_args())
