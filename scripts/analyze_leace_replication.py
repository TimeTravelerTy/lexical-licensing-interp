#!/usr/bin/env python3
"""Declared readings N1-N3 of reports/leace_replication_rules.md for one model (and Pythia for reference).

N1: cross-fitted scrub 0..NEC_TOP share remaining, 95% CI upper bound below 1.
N2: in-sample share below the cross-fitted share on the same bootstrap draws (paired difference CI below 0).
N3: in-sample true-label CI upper bound below the 5th percentile of the 20 permutations (leace_permutation_summary.csv).
Bootstrap: analyze_das_prep.Boot on the task-1 item table (1000 draws, seed 17), as the stored analyses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_das_prep import Boot


def table(path):
    P = pd.read_csv(path)
    G, B = P[P.side == "good"].set_index("item"), P[P.side == "bad"].set_index("item")
    it = G[["verb_pair", "verb_band", "context_id", "context_band"]].reset_index()
    return G, B, it


def ci(d):
    return [float(d[0]), float(np.nanpercentile(d[1:], 2.5)), float(np.nanpercentile(d[1:], 97.5))]


def run(a):
    k = a.nec_top
    Gc, Bc, it = table(Path(a.cf) / "items.csv.gz")
    Gi, Bi, it2 = table(Path(a.ins) / "items.csv.gz")
    assert (Gc.prompt.to_numpy() == Gi.prompt.to_numpy()).all()
    boot = Boot(it, a.n_boot, np.random.default_rng(a.seed))
    m = np.ones(len(it), bool)
    nat = boot.means((Gc.lp_by__natural - Bc.lp_by__natural).to_numpy(), m)
    cond = f"scrub_0_{k}" if f"lp_by__scrub_0_{k}" in Gc.columns else "scrub_0_8"
    cf = boot.means((Gc[f"lp_by__{cond}"] - Bc[f"lp_by__{cond}"]).to_numpy(), m) / nat
    ins = boot.means((Gi[f"lp_by__{cond}"] - Bi[f"lp_by__{cond}"]).to_numpy(), m) / nat
    perm = pd.read_csv(Path(a.perm_summary)).set_index("condition")
    p5 = float(perm.loc[[cond], "perm_share_p5"].iloc[0])
    r = {"model": a.model_name, "condition": cond, "natural_gap": ci(nat), "cf_share": ci(cf), "cf_removed": ci(1 - cf),
         "insample_share": ci(ins), "insample_minus_cf": ci(ins - cf), "perm_p5": p5}
    r["N1"] = r["cf_share"][2] < 1
    r["N2"] = r["insample_minus_cf"][2] < 0
    r["N3"] = r["insample_share"][2] < p5
    r["replicates"] = r["N1"] and r["N2"] and r["N3"]
    Path(a.out).write_text(json.dumps(r, indent=1) + "\n")
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cf", required=True)
    ap.add_argument("--ins", required=True)
    ap.add_argument("--perm-summary", required=True)
    ap.add_argument("--nec-top", type=int, required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
