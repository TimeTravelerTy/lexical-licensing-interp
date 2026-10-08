#!/usr/bin/env python3
"""Get-passive version of the passive test (round3_plan.md, B6). No model.

Every passive prompt "The N was <participle>" becomes "The N got <participle>".
Prompt ids, items and the patch plan are otherwise unchanged, so each got-passive
patch is paired row by row with its was-passive counterpart; the plan file of
the passive test is reused as is (`plan.csv.gz`, hash in `plan_meta.json`).
Token alignment (" was" and " got" are single tokens, so the participle stays
the last token and the prompt length is unchanged) is checked on the GPU side
by `run_passive_test.py`'s self-patch check and by the runner's encode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def swap(s, src, dst):
    if s.count(src) != 1:
        raise SystemExit(f"expected exactly one {src!r} in {s!r}")
    return s.replace(src, dst)


def run(args):
    src, dst = f" {args.old} ", f" {args.new} "
    P = pd.read_csv(args.prompts)
    items = pd.read_csv(args.items)
    pas = P.kind == "passive"
    P.loc[pas, "prompt"] = [swap(s, src, dst) for s in P.prompt[pas]]
    if P.prompt.duplicated().any():
        raise SystemExit("prompt collision after the swap")
    for c in ("prefix", "good_prompt", "bad_prompt"):  # prefixes end in " was ", prompts contain it once
        items[c] = [swap(s, src, dst) for s in items[c]]
    pid = dict(zip(P.prompt, P.pid))
    assert items.good_prompt.map(pid).notna().all() and items.bad_prompt.map(pid).notna().all()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    P.to_csv(out / "prompts.csv", index=False)
    items.to_csv(out / "items.csv", index=False)
    meta = {"source_prompts": args.prompts, "source_items": args.items, "swap": [src, dst],
            "n_passive_prompts": int(pas.sum()), "n_items": len(items),
            "prompts_sha256": hashlib.sha256((out / "prompts.csv").read_bytes()).hexdigest(),
            "plan": "reuse data/das_round2/passive_test/plan.csv.gz (same prompt ids)",
            "example": P[pas].prompt.head(3).tolist()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", default="data/das_round2/passive_test/prompts.csv")
    ap.add_argument("--items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--old", default="was")
    ap.add_argument("--new", default="got")
    ap.add_argument("--out-dir", default="data/das_round2/passive_test_got")
    run(ap.parse_args())
