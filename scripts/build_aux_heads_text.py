#!/usr/bin/env python3
"""Round 4, A5(a): natural-text passages for generic previous-token scores. No model.

256 passages, each 8 good sentences drawn at random (seed 17) from the 67 BLiMP paradigms
(local HF cache, `nyu-mll/blimp`), joined by spaces; one passage per line. With ~10 tokens per
sentence this gives >= 48 tokens per passage; the runner checks the length.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def run(args):
    files = sorted(glob.glob(str(Path(args.blimp).expanduser() / "snapshots" / "*" / "*" / "*.parquet")))
    assert len(files) == 67, len(files)
    sents = pd.concat([pd.read_parquet(f, columns=["sentence_good"]) for f in files]).sentence_good.tolist()
    rng = np.random.default_rng(args.seed)
    lines = [" ".join(sents[i] for i in rng.choice(len(sents), args.per_passage, replace=False))
             for _ in range(args.n)]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} passages from {len(sents)} sentences -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blimp", default="~/.cache/huggingface/hub/datasets--nyu-mll--blimp")
    ap.add_argument("--out", default="data/round4/aux_heads/natural_text.txt")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--per-passage", type=int, default=8)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
