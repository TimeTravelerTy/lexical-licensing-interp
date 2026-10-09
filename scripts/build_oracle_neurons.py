#!/usr/bin/env python3
"""Round 4, E13-T: the neuron sets of the translation-stage oracle (reports/round4/plan.md, E13-T). No model.

From D11 (`conjunction_switch_neurons.csv`, `conjunction.npz`):
- `neurons`: ranks 1-50 of the passive-conjunction set ranked by half-A direct effect on the switch;
- `random1`..`random5`: per set, the same number of neurons per layer (MLP11, MLP14) drawn uniformly from the
  neurons outside the top 50 (seed 17, sets drawn in order);
- `wmatched`: for each top-50 neuron in rank order, the not-yet-chosen neuron of the same layer that D11 did not
  select (BH q >= 0.05 on half A) with the closest |w_by| (the " by" output weight).
Output: `--out` (set, layer, index, w_by).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_conjunction import bh, cell_means, pair_halves, t_pvalue

LAYERS = (11, 14)
D = 8192


def run(args):
    z = np.load(args.npz, allow_pickle=True)
    assert [int(l) for l in z["layers"]] == list(LAYERS)
    A = np.concatenate([cell_means(z, f"a_{l}") for l in LAYERS], -1)
    I = (A[:, 0, 0] - A[:, 0, 1]) - (A[:, 1, 0] - A[:, 1, 1])
    xa = I[pair_halves(z["bands"], args.seed) == 0]
    sel = bh(t_pvalue(xa.mean(0) / (xa.std(0, ddof=1) / np.sqrt(len(xa))), len(xa) - 1), args.q)
    wby = np.concatenate([z[f"w_by_{l}"] for l in LAYERS])
    sw = pd.read_csv(args.switch_neurons).sort_values("rank").head(args.n)
    top = list(zip(sw.layer.astype(int), sw["index"].astype(int)))
    flat = lambda l, i: LAYERS.index(l) * D + i
    rows = [("neurons", l, i) for l, i in top]
    rng = np.random.default_rng(args.seed)
    for k in range(1, args.n_random + 1):
        for l in LAYERS:
            m = sum(1 for ll, _ in top if ll == l)
            pool = np.setdiff1d(np.arange(D), [i for ll, i in top if ll == l])
            rows += [(f"random{k}", l, int(i)) for i in np.sort(rng.choice(pool, m, replace=False))]
    taken = set(top)
    for l, i in top:
        cand = np.array([j for j in range(D) if not sel[flat(l, j)] and (l, j) not in taken])
        j = int(cand[np.argmin(np.abs(np.abs(wby[[flat(l, c) for c in cand]]) - abs(wby[flat(l, i)])))])
        rows.append(("wmatched", l, j))
        taken.add((l, j))
    nd = pd.DataFrame(rows, columns=["set", "layer", "index"])
    nd["w_by"] = [float(wby[flat(l, i)]) for l, i in zip(nd.layer, nd["index"])]
    nd["d11_selected"] = [bool(sel[flat(l, i)]) for l, i in zip(nd.layer, nd["index"])]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    nd.to_csv(args.out, index=False)
    print(nd.groupby("set", sort=False).agg(n=("index", "size"), mlp11=("layer", lambda x: int((x == 11).sum())),
                                            mean_abs_w_by=("w_by", lambda x: float(np.abs(x).mean())),
                                            d11_selected=("d11_selected", "sum")).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="results/round4/translation/conjunction.npz")
    ap.add_argument("--switch-neurons", default="results/round4/translation/conjunction_switch_neurons.csv")
    ap.add_argument("--out", default="data/round4/oracle_translation/neurons.csv")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
