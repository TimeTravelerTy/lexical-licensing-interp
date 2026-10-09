#!/usr/bin/env python3
"""Round 4, D12: per-prompt direct effects of the mid MLPs on " by" (reports/round4/plan.md, D12). fp32.

Natural pass (no patching) over the good and bad prompts of every primary passive-test item. Per prompt,
at the participle's last token (`run_site8_mechanism.Mech.run`): each MLP's and each layer's summed
heads' direct contribution to the centered " by" logit for layers 8-23, divided by the final-LN scale,
and the actual centered " by" logit.

Output (`--out-dir`): `translation_gain.parquet` (prompt, by_actual, sigma, mlp{l}, heads{l}),
`translation_gain_meta.json`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_passive_test import PassiveRunner
from run_site8_mechanism import SITE, Mech


def run(args):
    import torch

    args.dtype = "float32"
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runner = PassiveRunner(args)
    mech = Mech(runner, args)
    items = pd.read_csv(args.items)
    prim = items[items.bad_class == "plain"]
    prompts = sorted(set(prim.good_prompt) | set(prim.bad_prompt))
    rows = []
    with torch.no_grad():
        for i in range(0, len(prompts), args.batch):
            ps = prompts[i:i + args.batch]
            cap = mech.run(ps)
            sig = cap["sigma"][:, None]
            mlp = (cap["mlp"][:, :, 0] / sig).cpu().numpy()  # [B, L-8]
            heads = (cap["heads"][:, :, :, 0].sum(-1) / sig).cpu().numpy()
            by = cap["zc"][:, 0].cpu().numpy()
            for j, p in enumerate(ps):
                rows.append({"prompt": p, "by_actual": float(by[j]), "sigma": float(cap["sigma"][j]),
                             **{f"mlp{l}": float(mlp[j, l - SITE]) for l in range(SITE, mech.L)},
                             **{f"heads{l}": float(heads[j, l - SITE]) for l in range(SITE, mech.L)}})
    df = pd.DataFrame(rows)
    df.to_parquet(out / "translation_gain.parquet", index=False)
    meta = {"args": vars(args), "prompts": len(df), "seconds": round(time.time() - t0, 1),
            "readout": "centered ' by' logit through the final LN (Mech.R[0]); contributions / sigma of the run"}
    (out / "translation_gain_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps(meta, default=str), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--out-dir", default="results/round4/translation")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=256)
    run(ap.parse_args())
