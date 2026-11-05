#!/usr/bin/env python3
"""Gemma replication: compare two score files of the same scorer on common items (max |difference| per column).

Modes (files):
- frame:   frame_scores.parquet of run_frame_contrast.py; key (set, context_id, frame, verb); columns lp, lp1
           (+ lp_next_by when both have it).
- readout: CSV of run_next_token_readout.py; key prompt_id; every lp[...] column.
- fb:      CSV of score_matched_passives.py; key pair_id; good/bad whole, verb, suffix, by log-probs.

Prints n common items and max |a - b| per column; with --tol, exits non-zero if any exceeds it. With --exact,
the two files must agree to the last stored digit (0 difference).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

KEYS = {"frame": ["set", "context_id", "frame", "verb"], "readout": ["prompt_id"], "fb": ["pair_id"]}


def load(mode, path):
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, low_memory=False)


def main(args):
    a, b = load(args.mode, args.a), load(args.mode, args.b)
    k = KEYS[args.mode]
    if args.mode == "frame":
        cols = [c for c in ("lp", "lp1", "lp_next_by") if c in a and c in b]
    elif args.mode == "readout":
        cols = [c for c in a.columns if c.startswith("lp[") and c in b]
    else:
        cols = [f"{s}_{p}_lp" for s in ("good", "bad") for p in ("whole", "verb", "suffix", "by")]
    assert not a.duplicated(k).any() and not b.duplicated(k).any(), "duplicate keys"
    m = a[k + cols].merge(b[k + cols], on=k, suffixes=("_a", "_b"))
    assert len(m), "no common items"
    vals = m[[c + s for c in cols for s in ("_a", "_b")]].to_numpy(float)
    finite = bool(np.isfinite(vals).all())
    diff = {c: float((m[c + "_a"] - m[c + "_b"]).abs().max()) for c in cols}
    worst = max(diff.values())
    out = {"mode": args.mode, "a": args.a, "b": args.b, "n_a": len(a), "n_b": len(b), "n_common": len(m),
           "finite": finite, "max_abs_diff": diff, "worst": worst}
    print(json.dumps(out, indent=1))
    if (args.exact or args.tol is not None) and not (finite and len(m) == len(a) == len(b)):
        sys.exit(f"FAIL: non-finite scores or incomplete overlap ({len(m)} of {len(a)} / {len(b)})")
    if args.exact and worst != 0.0:
        sys.exit(f"FAIL: not identical (worst {worst:.3g})")
    if args.tol is not None and worst > args.tol:
        sys.exit(f"FAIL: worst {worst:.3g} > tol {args.tol}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=sorted(KEYS))
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--exact", action="store_true")
    main(ap.parse_args())
