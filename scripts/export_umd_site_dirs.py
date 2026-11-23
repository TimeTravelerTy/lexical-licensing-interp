#!/usr/bin/env python3
"""Lay out the stored u_md bases (bases_umd.npz, keys das_site{s}_s{k}_f{f}) as run_passive_test.py site dirs
(bases.pt with keys r1_s{k}_f{f}, shape d x 1; items.csv and pairs_folds.csv copied from the stored v site dir).
No held-out swaps exist for u_md, so run_passive_test.py runs only its self-patch check. Used by the expanded eval set
(reports/expanded_eval_rules.md, part C).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def run(a):
    import torch

    z = np.load(a.bases)
    for spec in a.site_dirs.split(","):
        s, src = spec.split("=")
        d = Path(a.out_dir) / f"umd_site{s}"
        d.mkdir(parents=True, exist_ok=True)
        keys = sorted(k for k in z.files if k.startswith(f"das_site{s}_"))
        assert len(keys) == 15, (s, len(keys))
        b = {"r1_" + k.split("_", 2)[2]: torch.tensor(z[k], dtype=torch.float32).reshape(-1, 1) for k in keys}
        for v in b.values():
            assert abs(float(v.norm()) - 1) < 1e-4
        torch.save(b, d / "bases.pt")
        for f in ("items.csv", "pairs_folds.csv"):
            shutil.copy(Path(src) / f, d / f)
        print(s, sorted(b)[:2], "->", d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bases", default="results/das_round2/umd/bases_umd.npz")
    ap.add_argument("--site-dirs", default="8=results/das_round2/final_strict_site8,"
                    "12=results/das_round2/final_strict_site12,17=results/das_round2/final_strict")
    ap.add_argument("--out-dir", default="results/pythia/expanded_eval/umd_bases")
    run(ap.parse_args())
