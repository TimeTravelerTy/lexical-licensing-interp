#!/usr/bin/env python3
"""Control site directories for `run_passive_test.py` (reports/v2_controls_mech_b_rules.md, part A).

--kind shuffled: the v2 final's own `shuffled_bases.pt` (trained with permuted class labels) as `bases.pt`.
--kind random:   per (split, fold) key one standard-normal direction, normalized; generator seed
                 910001 + 100 * draw + 10 * split + fold.
The output directory gets `bases.pt` (keys and shapes as v2's), the final's `pairs_folds.csv` (copied), and
`control_meta.json` (kind, draw, sha256 of the bases, |cos| with v2 per key).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def run(a):
    import torch

    fin, out = Path(a.final), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    v2 = torch.load(fin / "bases.pt")
    keys = sorted(k for k in v2 if k.startswith("r1_"))
    if a.kind == "shuffled":
        b = torch.load(fin / "shuffled_bases.pt")
        assert sorted(k for k in b if k.startswith("r1_")) == keys, "shuffled keys differ from v2's"
        b = {k: b[k].float() for k in keys}
    else:
        b = {}
        for k in keys:
            s, f = int(k.split("_s")[1].split("_")[0]), int(k.split("_f")[1])
            g = torch.Generator().manual_seed(910001 + 100 * a.draw + 10 * s + f)
            x = torch.randn(v2[k].shape, generator=g)
            b[k] = x / x.norm()
    for k in keys:
        assert b[k].shape == v2[k].shape, k
        assert abs(float(b[k].norm()) - 1) < 1e-4, k
    torch.save(b, out / "bases.pt")
    shutil.copy(fin / "pairs_folds.csv", out / "pairs_folds.csv")
    cos = {k: abs(float(b[k][:, 0].float() @ v2[k][:, 0].float())) for k in keys}
    meta = {"kind": a.kind, "draw": a.draw, "final": str(fin), "keys": keys,
            "sha256": hashlib.sha256((out / "bases.pt").read_bytes()).hexdigest(),
            "abs_cos_with_v2": cos, "abs_cos_mean": sum(cos.values()) / len(cos)}
    (out / "control_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "abs_cos_with_v2"}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--final", required=True, help="v2 final dir (bases.pt, shuffled_bases.pt, pairs_folds.csv)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", required=True, choices=("shuffled", "random"))
    ap.add_argument("--draw", type=int, default=0)
    run(ap.parse_args())
