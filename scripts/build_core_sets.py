#!/usr/bin/env python3
"""Core-neuron sets for C4 phase 1b, B3b (reports/v2_controls_mech_b_rules.md).

Core = the top 5 of the v2 passive-conjunction set P by half-A direct effect on the switch
(conjunction_switch_neurons.csv). Sets, masks over MLP 11 then MLP 14 (2 x 8192), as build_mediation_sets.py:
- core_<neuron> (each alone), core_top3, core_top5;
- ctl<layer>_<k> (k < 20): single non-P neurons, the 20 of each layer with the largest half-A |direct effect of was T - I
  on " by"| (computed as build_mediation_sets.py; ties by index). No non-P neuron reaches the core's coupling (recorded),
  so these are the strongest available non-conjunction comparators, not matched ones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_conjunction import cell_means

D = 8192


def run(a):
    d = Path(a.dir)
    z = np.load(d / "conjunction.npz", allow_pickle=True)
    s = np.load(d / "conjunction_selection.npz", allow_pickle=True)
    layers = [int(l) for l in z["layers"]]
    assert layers == [11, 14]
    A = np.concatenate([cell_means(z, f"a_{l}") for l in layers], -1)
    was = A[:, 0, 0] - A[:, 0, 1]
    hA = s["halves"] == 0
    sig = cell_means(z, "sigma")[:, 0].mean(1)
    wby = np.concatenate([z[f"w_by_{l}"] for l in layers])
    de = np.abs((was[hA] * wby[None] / sig[hA][:, None]).mean(0))
    pc = s["pc"].astype(bool)
    core = pd.read_csv(d / "conjunction_switch_neurons.csv").head(5)
    gix = [(0 if r.layer == 11 else D) + int(r.index) for r in core.itertuples()]
    assert pc[gix].all(), "a core neuron is not in P"
    names, masks, notes = [], [], {"core": core.neuron.tolist(), "controls": {}}

    def one(ix):
        m = np.zeros(2 * D, bool)
        m[list(ix)] = True
        return m

    for n, g in zip(core.neuron, gix):
        names.append(f"core_{n}")
        masks.append(one([g]))
    names += ["core_top3", "core_top5"]
    masks += [one(gix[:3]), one(gix)]
    for j, l in enumerate(layers):
        pool = np.flatnonzero(~pc[j * D:(j + 1) * D]) + j * D
        order = pool[np.lexsort((pool, -de[pool]))][:a.n_ctrl]
        notes["controls"][f"MLP{l}"] = {"ctrl_de_mean": float(de[order].mean()), "ctrl_de_min": float(de[order].min()),
                                        "ctrl_de_max": float(de[order].max()),
                                        "core_de": {n: float(de[g]) for n, g in zip(core.neuron, gix) if (g >= D) == bool(j)}}
        for k, c in enumerate(order):
            names.append(f"ctl{l}_{k}")
            masks.append(one([c]))
    M = np.stack(masks)
    np.savez_compressed(a.out, names=np.array(names), masks=M)
    Path(a.out).with_suffix(".json").write_text(json.dumps(notes, indent=2) + "\n")
    print(json.dumps(notes, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/pythia/das_v2/mech/conj_v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-ctrl", type=int, default=20)
    run(ap.parse_args())
