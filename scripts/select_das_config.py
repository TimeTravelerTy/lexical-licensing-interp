#!/usr/bin/env python3
"""Freeze the DAS site and epoch count from the actives-only sweep.

Rule (fixed before the sweep was read):
- For each site take its best epoch by mean held-out cross-class IIA (folds
  pooled). Late sites control the readout almost linearly, so the highest
  IIA alone would favour them; the rule instead takes the **earliest** site
  whose best IIA is within 0.02 of the overall best and whose median
  positive-control fraction at that epoch is >= 0.5.
- Epochs: the smallest epoch at that site whose IIA is within 0.01 of the
  site's best.
Rank is chosen later, in `run_das_round2.py final`, by its own rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def run(args):
    sweep = pd.read_csv(args.sweep)
    g = sweep.groupby(["site", "epoch"]).agg(iia=("iia_cross", "mean"), pc=("pc_frac_median", "mean"),
                                             iia_same=("iia_same", "mean"), david=("iia_cross_david", "mean"),
                                             rev=("rev_frac_median", "mean")).reset_index()
    best = g.loc[g.groupby("site").iia.idxmax()].set_index("site")
    top = best.iia.max()
    ok = best[(best.iia >= top - args.iia_tol) & (best.pc >= args.min_pc)]
    if ok.empty:
        raise SystemExit("no site meets the rule")
    site = int(ok.index.min())
    s = g[g.site == site]
    epochs = int(s[s.iia >= s.iia.max() - args.epoch_tol].epoch.min())
    cfg = {"site": site, "layer_output": site - 1, "epochs": epochs, "rank": None,
           "rule": __doc__.split("Rule (fixed before the sweep was read):")[1].strip(),
           "selected_metrics": s[s.epoch == epochs].iloc[0].to_dict(), "overall_best_iia": float(top),
           "sweep_sha256": hashlib.sha256(Path(args.sweep).read_bytes()).hexdigest()}
    Path(args.out).write_text(json.dumps(cfg, indent=2) + "\n")
    best.round(3).to_csv(Path(args.out).with_name("sweep_by_site.csv"))
    print(best.round(3).to_string())
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", default="results/das_round2/sweep_strict/sweep.csv")
    ap.add_argument("--out", default="results/das_round2/frozen_config.json")
    ap.add_argument("--iia-tol", type=float, default=0.02)
    ap.add_argument("--min-pc", type=float, default=0.5)
    ap.add_argument("--epoch-tol", type=float, default=0.01)
    run(ap.parse_args())
