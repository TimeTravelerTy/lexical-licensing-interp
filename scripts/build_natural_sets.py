#!/usr/bin/env python3
"""Natural-matched control sets for C4 phase 1b, B2 (reports/v2_controls_mech_b_rules.md).

From the half-A natural run (`run_mech_b.py --mode natural_halfA`: pair-averaged good - bad post-activation per neuron
of MLPs 11 / 14) and phase 1's P:
- magnat_k (k < 20): per layer and per decile of |half-A mean good - bad| (deciles within the layer over all 8192
  neurons, ties by index), as many non-P neurons as P has in that cell; shortfall from the nearest lower decile, then
  the next (recorded), as build_mediation_sets.py. Seed 900101 + 3000 + k.
- denat_k: the same on |half-A mean good - bad x the neuron's " by" output weight w_by| (unscaled). Seed 900101 + 4000 + k.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

D = 8192


def matched(var, pc, rng):
    m = np.zeros(2 * D, bool)
    short = 0
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
                    short += t
                if n == 0:
                    break
            assert n == 0, "not enough non-P neurons"
        m[sl] = taken
    return m, short


def run(a):
    h = np.load(a.halfA, allow_pickle=True)
    z = np.load(Path(a.conj_dir) / "conjunction.npz", allow_pickle=True)
    s = np.load(Path(a.conj_dir) / "conjunction_selection.npz", allow_pickle=True)
    assert [int(l) for l in z["layers"]] == [11, 14]
    half_a = set(np.asarray(s["pairs"])[np.asarray(s["halves"]) == 0])
    assert set(h["pairs"]) == half_a, "half-A pairs differ from the selection's"
    pc = s["pc"].astype(bool)
    diff = h["mean_diff"].astype(float)
    wby = np.concatenate([z[f"w_by_{l}"] for l in (11, 14)])
    var = {"magnat": np.abs(diff), "denat": np.abs(diff * wby)}
    names, masks, notes = [], [], {"shortfall": {}, "means": {}}
    for fam, off in (("magnat", 3000), ("denat", 4000)):
        notes["shortfall"][fam] = 0
        ms = []
        for k in range(a.n_ctrl):
            m, sh = matched(var[fam], pc, np.random.default_rng(900101 + off + k))
            assert not (m & pc).any() and m.sum() == pc.sum()
            notes["shortfall"][fam] += sh
            names.append(f"{fam}_{k}")
            masks.append(m)
            ms.append(m)
        notes["means"][fam] = {"P": float(var[fam][pc].mean()), "ctrl": float(np.mean([var[fam][m].mean() for m in ms])),
                               "all": float(var[fam].mean())}
    np.savez_compressed(a.out, names=np.array(names), masks=np.stack(masks))
    Path(a.out).with_suffix(".json").write_text(json.dumps(notes, indent=2) + "\n")
    print(json.dumps(notes, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--halfA", required=True, help="natural_halfA.npz")
    ap.add_argument("--conj-dir", default="results/pythia/das_v2/mech/conj_v2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-ctrl", type=int, default=20)
    run(ap.parse_args())
