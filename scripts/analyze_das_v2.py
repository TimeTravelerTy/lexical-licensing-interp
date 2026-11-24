#!/usr/bin/env python3
"""Analysis of the v2 instrument measures (rules: reports/v2_retrain_rules.md).

gate        measure 1: per declared rule site, v2 - v held-out cross IIA on the same swaps (matched on split, fold,
            base, src; natural readouts must agree), averaged per stored pair over its cross swaps as base, 95%
            percentile bootstrap over pairs (10,000). v2 replaces v unless the CI lies entirely below -0.02.
            Also, descriptively at every site: fold-mean cross IIA and median patched fraction from summary.csv.
resolution  measure 2: medians and ranges of the resolution runs by site and n.
Writes reports/v2/instrument_measures.md (+ csvs in results/v2_summary/).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RULE = {"pythia": [8, 12], "gemma": [5, 8, 12], "olmo": [9, 11, 12]}
PRIMARY = {"pythia": 17, "gemma": 15, "olmo": 13}
SITES = {"pythia": [*range(15), 16, 17], "gemma": list(range(16)), "olmo": list(range(16))}
EARLY_PYTHIA = {0, 1, 2, 3, 5, 7, 9, 11, 13}


def stored_dir(m, s):
    name = "final_strict" if s == PRIMARY[m] else f"final_strict_site{s}"
    if m == "pythia":
        return Path("results/gemma/das/pythia_early" if s in EARLY_PYTHIA else "results/das_round2") / name
    return Path(f"results/{m}/das/transfer") / name


def v2_dir(m, s):
    return Path(f"results/{m}/das_v2") / ("final_strict" if s == PRIMARY[m] else f"final_strict_site{s}")


def das_r1(df):
    return df[(df.control == "das") & (df["rank"] == 1)]


def gate(m, s, n_boot=10000, seed=0):
    key = ["split", "fold", "base", "src"]
    a = das_r1(pd.read_csv(stored_dir(m, s) / "heldout_swaps.csv.gz"))
    b = das_r1(pd.read_csv(v2_dir(m, s) / "heldout_swaps.csv.gz"))
    assert len(a) == len(b) and not a.duplicated(key).any() and not b.duplicated(key).any()
    j = a.merge(b, on=key, suffixes=("_v", "_v2"), validate="one_to_one")
    assert len(j) == len(a)
    for c in ["M_base", "M_src"]:
        assert np.abs(j[f"{c}_v"] - j[f"{c}_v2"]).max() <= 1e-4, (m, s, c)
    assert (j.cross_v == j.cross_v2).all() and (j.base_pair_id_v == j.base_pair_id_v2).all()
    c = j[j.cross_v]
    per = c.groupby("base_pair_id_v").agg(v=("iia_v", "mean"), v2=("iia_v2", "mean"))
    d = (per.v2 - per.v).to_numpy()
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    fv = c.groupby("base_pair_id_v").frac_v.median()
    fv2 = c.groupby("base_pair_id_v").frac_v2.median()
    return {"model": m, "site": s, "n_pairs": len(d), "n_cross_swaps": len(c), "iia_v": c.iia_v.mean(),
            "iia_v2": c.iia_v2.mean(), "diff": d.mean(), "lo": lo, "hi": hi, "keeps_v": hi < -0.02,
            "frac_med_v": c[c.base_cls_v == 0].frac_v.median(), "frac_med_v2": c[c.base_cls_v == 0].frac_v2.median(),
            "pairs_v2_higher": int((d > 0).sum()), "pairs_v2_lower": int((d < 0).sum())}


def summary_row(m, s):
    out = {"model": m, "site": s}
    for tag, d in [("v", stored_dir(m, s)), ("v2", v2_dir(m, s))]:
        S = pd.read_csv(d / "summary.csv")
        S = S[(S.control == "das") & (S["rank"] == 1)] if "control" in S else S
        out[f"iia_{tag}"] = S.iia_cross.mean()
        out[f"frac_{tag}"] = S.pc_frac_median.mean()
    out["diff"] = out["iia_v2"] - out["iia_v"]
    return out


def resolution(m, s):
    R = pd.read_csv(Path(f"results/{m}/das_v2/resolution_site{s}/resolution.csv"))
    R["iia"] = R[["iia_A", "iia_B"]].mean(axis=1)
    R["cos_v2"] = R[["cos_v2_A", "cos_v2_B"]].mean(axis=1)
    R["cos_v"] = R[["cos_v_A", "cos_v_B"]].mean(axis=1)
    g = R.groupby("n")
    t = g[["cos_AB", "cos_v2", "cos_v", "iia"]].median()
    t["cos_AB_min"], t["cos_AB_max"], t["n_draws"], t["pool"] = g.cos_AB.min(), g.cos_AB.max(), g.size(), g.pool.min()
    return t.reset_index().assign(model=m, site=s)


def main(a):
    out = Path("results/v2_summary")
    out.mkdir(parents=True, exist_ok=True)
    G = pd.DataFrame([gate(m, s) for m in RULE for s in RULE[m]])
    G.to_csv(out / "gate.csv", index=False)
    A = pd.DataFrame([summary_row(m, s) for m in SITES for s in SITES[m]])
    A.to_csv(out / "fold_mean_iia.csv", index=False)
    R = pd.concat([resolution(m, s) for m, ss in {"pythia": [8, 12], "gemma": [5, 8], "olmo": [9, 11]}.items()
                   for s in ss])
    R.to_csv(out / "resolution.csv", index=False)
    pd.set_option("display.width", 200)
    print(G.round(3).to_string(index=False))
    print(A.round(3).to_string(index=False))
    print(R.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    main(ap.parse_args())
