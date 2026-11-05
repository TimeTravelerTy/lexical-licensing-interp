#!/usr/bin/env python3
"""Regression check: compare a rerun DAS output table with the stored one (same seeds), key by key.

Modes: final (summary.csv of run_das_round2.py final; keys rank, split, fold, control, draw) and sweep (sweep.csv of
run_das_round2.py / run_passive_das.py sweep; keys site, split, fold, epoch). --sites restricts the stored sweep to
the rerun sites. Every numeric column is compared; exit 1 if any key is missing or any |diff| > --tol.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

KEYS = {"final": ["rank", "split", "fold", "control", "draw"], "sweep": ["site", "split", "fold", "epoch"]}


def main(args):
    new, old = pd.read_csv(args.new), pd.read_csv(args.stored)
    k = KEYS[args.mode]
    if args.mode == "final":
        for d in (new, old):
            d["draw"] = d["draw"].fillna(-1).astype(int)
    if args.sites:
        old = old[old.site.isin([int(s) for s in args.sites.split(",")])]
    cols = [c for c in old.columns if c not in k and np.issubdtype(old[c].dtype, np.number)]
    absent = [c for c in cols if c not in new]
    assert not absent, f"columns missing from the rerun: {absent}"
    for name, d in (("stored", old), ("new", new)):
        assert d[k].notna().all().all() and not d.duplicated(k).any(), f"null or duplicate keys in {name}"
    m = old.merge(new, on=k, how="outer", suffixes=("_stored", "_new"), indicator=True, validate="one_to_one")
    missing = int((m._merge != "both").sum())
    both = m[m._merge == "both"]
    nan_mismatch = {c: int((both[c + "_stored"].isna() != both[c + "_new"].isna()).sum()) for c in cols}
    diff = {c: float(np.nanmax(np.abs(both[c + "_stored"] - both[c + "_new"]).to_numpy())) if both[c + "_stored"].notna().any()
            else 0.0 for c in cols}
    worst = float(np.max(list(diff.values()))) if diff else float("nan")
    iia = {c: diff[c] for c in cols if c.startswith("iia")}
    res = {"mode": args.mode, "n_stored": len(old), "n_new": len(new), "unmatched_rows": missing,
           "nan_pattern_mismatches": sum(nan_mismatch.values()),
           "max_abs_diff_iia": iia, "max_abs_diff_all": worst, "worst_column": max(diff, key=diff.get) if diff else None}
    print(json.dumps(res, indent=1))
    if missing or sum(nan_mismatch.values()) or not np.isfinite(worst) or worst > args.tol:
        sys.exit(f"REGRESSION FAIL (tol {args.tol})")
    print("REGRESSION PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=sorted(KEYS))
    ap.add_argument("new")
    ap.add_argument("stored")
    ap.add_argument("--sites", default="")
    ap.add_argument("--tol", type=float, default=1e-6)
    main(ap.parse_args())
