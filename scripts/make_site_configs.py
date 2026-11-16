#!/usr/bin/env python3
"""Write per-site rank-1 DAS configs (frozen_config_site{s}.json) from an active-task sweep, by the frozen epoch rule.

Epochs: the smallest epoch at the site whose fold-mean held-out cross-class IIA is within --epoch-tol of the site's
best (the rule in select_das_config.py). Each config records the role and rule given on the command line, the
sweep's SHA-256 and the IIA at the best and chosen epochs. This is the procedure behind the Gemma transfer and
depth-curve configs, written as a script for the OLMo replication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main(args):
    sweep_path = Path(args.sweep)
    sweep = pd.read_csv(sweep_path)
    sha = hashlib.sha256(sweep_path.read_bytes()).hexdigest()
    g = sweep.groupby(["site", "epoch"]).iia_cross.mean().reset_index()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    roles = dict(r.split("=", 1) for r in args.roles.split(";") if r) if args.roles else {}
    for site in [int(s) for s in args.sites.split(",")]:
        s = g[g.site == site]
        best = float(s.iia_cross.max())
        epochs = int(s[s.iia_cross >= best - args.epoch_tol].epoch.min())
        cfg = {"site": site, "layer_output": site - 1, "epochs": epochs, "rank": 1,
               "role": roles.get(str(site), args.default_role), "declared": args.declared,
               "epoch_rule": f"smallest epoch whose mean held-out cross-class IIA is within {args.epoch_tol} of the "
                             f"site's best ({sweep_path.name})",
               "best_epoch_iia": best, "chosen_epoch_iia": float(s[s.epoch == epochs].iia_cross.iloc[0]),
               "sweep_sha256": sha}
        p = out / f"frozen_config_site{site}.json"
        if p.exists() and not args.overwrite:
            raise SystemExit(f"{p} exists (use --overwrite)")
        p.write_text(json.dumps(cfg, indent=2) + "\n")
        print(site, epochs, round(best, 4))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sites", required=True)
    ap.add_argument("--roles", default="", help='"site=role;site=role"')
    ap.add_argument("--default-role", default="depth curve (passive test only)")
    ap.add_argument("--declared", required=True)
    ap.add_argument("--epoch-tol", type=float, default=0.01)
    ap.add_argument("--overwrite", action="store_true")
    main(ap.parse_args())
