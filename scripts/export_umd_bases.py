#!/usr/bin/env python3
"""Export the per-basis class mean-difference directions (u_md) as a bases file the transfer runners read.

Source: the split edit's directions_site{s}.npz (keys [15, 2] = (split, fold); u [15, d] unit vectors), so each
u is the class mean difference of exactly the training items of the v basis with the same (split, fold).
Output: an npz with keys das_site{s}_s{k}_f{f} -> u (float32), the format of bases_rank1.npz, so
run_construction_test.py / run_inrange_patch.py / run_fillergap.py patch along u instead of v. An interchange
patch is invariant to the sign of u, so u's sign is left as stored.

Checks: unit norm (1e-4), and, where bases_rank1.npz is given, the stored v of each key equals that file's
basis up to sign (1e-5), so the (split, fold) mapping is the one the transfer runs used.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-edit", required=True, help="results/{model}/split_edit")
    ap.add_argument("--v-bases", default="", help="the model's bases_rank1.npz, for the mapping check")
    ap.add_argument("--sites", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sites = [int(s) for s in a.sites.split(",")]
    vb = np.load(a.v_bases) if a.v_bases else None
    out, chk = {}, {"sites": sites, "max_norm_dev": 0.0, "max_v_vs_bases": None}
    vdev = 0.0
    for s in sites:
        z = np.load(Path(a.split_edit) / f"directions_site{s}.npz")
        for i, (k, f) in enumerate(z["keys"].tolist()):
            u = z["u"][i].astype(np.float32)
            chk["max_norm_dev"] = max(chk["max_norm_dev"], abs(float(np.linalg.norm(u)) - 1))
            key = f"das_site{s}_s{k}_f{f}"
            if vb is not None:
                b = vb[key].reshape(-1).astype(np.float64)
                v = z["v"][i].astype(np.float64)
                vdev = max(vdev, float(min(np.abs(b - v).max(), np.abs(b + v).max())))
            out[key] = u
        if len(z["keys"]) != 15:
            raise SystemExit(f"site {s}: {len(z['keys'])} bases, expected 15")
    if vb is not None:
        chk["max_v_vs_bases"] = vdev
        if vdev > 1e-5:
            raise SystemExit(f"stored v differs from {a.v_bases} by {vdev}")
    if chk["max_norm_dev"] > 1e-4:
        raise SystemExit(f"u not unit norm: {chk['max_norm_dev']}")
    np.savez(a.out, **out)
    chk["sha256"] = hashlib.sha256(Path(a.out).read_bytes()).hexdigest()
    chk["n_keys"] = len(out)
    Path(a.out).with_suffix(".json").write_text(json.dumps(chk, indent=1) + "\n")
    print(json.dumps(chk))


if __name__ == "__main__":
    main()
