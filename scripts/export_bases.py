#!/usr/bin/env python3
"""Export rank-1 DAS bases to one torch-free .npz, so analyses can run without torch.

Keys: `das_site{s}_s{split}_f{fold}` for every `final_strict*` run found, and
`shuffled_site{s}_s{split}_f{fold}` for every `passive_test*/shuffled_bases_site{s}.pt`.
Each value is the unit vector (d_model,) as float32, in the stored sign.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def run(args):
    import torch

    root = Path(args.root)
    out, src = {}, {}
    for d in sorted(root.glob("final_strict*")):
        cfg = json.loads((d / "final_meta.json").read_text())
        for k, v in torch.load(d / "bases.pt").items():
            if k.startswith("r1_"):
                out[f"das_site{cfg['site']}_{k[3:]}"] = v[:, 0].float().numpy()
        src[str(cfg["site"])] = str(d)
    for f in sorted(root.glob("passive_test*/shuffled_bases_site*.pt")):
        s = int(re.search(r"site(\d+)", f.name).group(1))
        for k, v in torch.load(f).items():
            out[f"shuffled_site{s}_{k[3:]}"] = v[:, 0].float().numpy()
    np.savez_compressed(args.out, **out)
    print(f"{len(out)} bases from {src}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results/das_round2")
    ap.add_argument("--out", default="results/das_round2/bases_rank1.npz")
    run(ap.parse_args())
