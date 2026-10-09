#!/usr/bin/env python3
"""Check that a plan in use is the committed one.

The builders record `plan_sha256` of the in-memory plan before it is written to CSV, so a CSV
read back hashes differently (dtypes). Rebuild the plan into a scratch directory with the same
builder, then call this: it asserts that the rebuilt in-memory hash (scratch `plan_meta.json`)
equals the committed one, and that the scratch and in-use CSVs have the same content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def content_hash(path):
    plan = pd.read_csv(path)
    return hashlib.sha256(pd.util.hash_pandas_object(plan, index=False).values.tobytes()).hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--committed", required=True, help="data dir with the committed plan_meta.json and the plan in use")
    ap.add_argument("--rebuilt", required=True, help="scratch dir written by the builder")
    a = ap.parse_args()
    c, r = Path(a.committed), Path(a.rebuilt)
    want = json.loads((c / "plan_meta.json").read_text())["plan_sha256"]
    got = json.loads((r / "plan_meta.json").read_text())["plan_sha256"]
    assert got == want, ("rebuilt plan differs from the committed one", got, want)
    h1, h2 = content_hash(c / "plan.csv.gz"), content_hash(r / "plan.csv.gz")
    assert h1 == h2, ("plan in use differs from the rebuilt plan", h1, h2)
    print(f"plan OK: {c} (in-memory {want[:12]}, content {h1[:12]})")
