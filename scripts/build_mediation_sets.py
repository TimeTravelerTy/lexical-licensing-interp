#!/usr/bin/env python3
"""Neuron sets for `run_conjunction_mediation.py` (reports/pythia/mechanism_v2_rules.md, stage 6).

From a stage-4 run (v2): `conjunction.npz`, `conjunction_selection.npz` and `conjunction_switch_neurons.csv`.
Sets, as masks over MLP 11 then MLP 14 (2 x 8192):
- P: the passive-conjunction set; ALL: every neuron;
- rand_k (k < 20): per layer, as many neurons as P has there, uniform over the non-P neurons (seed 900101 + k);
- mag_k (k < 20): per layer and per decile of the half-A |mean was T - I| (pairs equally weighted; deciles within the
  layer over all 8192 neurons, ties by index), as many non-P neurons as P has in that cell; a cell with too few non-P
  neurons takes all of them and the shortfall is drawn from the nearest decile below, then the next (recorded);
- de_k (k < 20): the same, matched on the half-A |direct effect of was T - I on " by"| (activation x output coupling,
  one reference final-LN scale per pair, as analyze_conjunction);
- P_mlp11 / P_mlp14: the layer-only parts of P;
- top10 / top50 / top100: P ranked by half-A direct effect on the switch (exploratory).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_conjunction import cell_means

D = 8192


def run(args):
    d = Path(args.dir)
    z = np.load(d / "conjunction.npz", allow_pickle=True)
    s = np.load(d / "conjunction_selection.npz", allow_pickle=True)
    layers = [int(l) for l in z["layers"]]
    assert layers == [11, 14]
    A = np.concatenate([cell_means(z, f"a_{l}") for l in layers], -1)
    was = A[:, 0, 0] - A[:, 0, 1]
    mag = np.abs(was[s["halves"] == 0].mean(0))  # half A
    pc = s["pc"].astype(bool)
    names, masks, notes = ["P", "ALL"], [pc.copy(), np.ones(2 * D, bool)], {}
    for k in range(args.n_ctrl):
        rng = np.random.default_rng(900101 + k)
        m = np.zeros(2 * D, bool)
        for j in range(2):
            sl = slice(j * D, (j + 1) * D)
            pool = np.flatnonzero(~pc[sl])
            m[sl][rng.choice(pool, int(pc[sl].sum()), replace=False)] = True
        names.append(f"rand_{k}")
        masks.append(m)
    sig = cell_means(z, "sigma")[:, 0].mean(1)
    wby = np.concatenate([z[f"w_by_{l}"] for l in layers])
    de = np.abs((was[s["halves"] == 0] * wby[None] / sig[s["halves"] == 0][:, None]).mean(0))  # half-A |direct effect|
    short = {}
    for fam, var, off in (("mag", mag, 1000), ("de", de, 2000)):
        short[fam] = 0
        for k in range(args.n_ctrl):
            rng = np.random.default_rng(900101 + off + k)
            m = np.zeros(2 * D, bool)
            for j in range(2):
                sl = slice(j * D, (j + 1) * D)
                mg, p_ = var[sl], pc[sl]
                dec = np.minimum((np.argsort(np.argsort(mg, kind="stable"), kind="stable") * 10) // D, 9)
                taken = np.zeros(D, bool)
                for q in range(9, -1, -1):
                    n = int((p_ & (dec == q)).sum())
                    for qq in range(q, -1, -1):
                        pool = np.flatnonzero(~p_ & (dec == qq) & ~taken)
                        t = min(n, len(pool))
                        taken[rng.choice(pool, t, replace=False)] = True
                        n -= t
                        if qq != q:
                            short[fam] += t
                        if n == 0:
                            break
                    assert n == 0, "not enough non-P neurons"
                m[sl] = taken
            names.append(f"{fam}_{k}")
            masks.append(m)
    notes["shortfall_drawn_from_lower_deciles_total"] = short
    sw = pd.read_csv(d / "conjunction_switch_neurons.csv")
    for k in (10, 50, 100):
        m = np.zeros(2 * D, bool)
        for r in sw.head(k).itertuples():
            m[(0 if r.layer == 11 else D) + int(r.index)] = True
        names.append(f"top{k}")
        masks.append(m)
    for j, l in enumerate(layers):  # layer-only parts of P
        m = np.zeros(2 * D, bool)
        m[j * D:(j + 1) * D] = pc[j * D:(j + 1) * D]
        names.append(f"P_mlp{l}")
        masks.append(m)
    M = np.stack(masks)
    assert (M[2:2 + 3 * args.n_ctrl] & pc).sum() == 0
    np.savez_compressed(args.out, names=np.array(names), masks=M)
    notes.update({"sizes": {n: int(m.sum()) for n, m in zip(names, M) if not n.startswith(("rand_", "mag_", "de_")) or n.endswith("_0")},
                  "P_mean_halfA_mag": float(mag[pc].mean()),
                  "mag_ctrl_mean_halfA_mag": float(np.mean([mag[m].mean() for n, m in zip(names, M) if n.startswith("mag_")])),
                  "P_mean_halfA_de": float(de[pc].mean()),
                  "de_ctrl_mean_halfA_de": float(np.mean([de[m].mean() for n, m in zip(names, M) if n.startswith("de_")])),
                  "rand_ctrl_mean_halfA_mag": float(np.mean([mag[m].mean() for n, m in zip(names, M) if n.startswith("rand_")]))})
    Path(args.out).with_suffix(".json").write_text(json.dumps(notes, indent=2) + "\n")
    print(json.dumps(notes, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="stage-4 (v2) conjunction output dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-ctrl", type=int, default=20)
    run(ap.parse_args())
