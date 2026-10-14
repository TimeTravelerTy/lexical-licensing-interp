#!/usr/bin/env python3
"""Round 5, A3: Pile counts for the A1 participles via the infini-gram API (reports/round5/plan.md, A3).

Index `v4_piletrain_llama` (the Pile training set). For each participle V (data/round5/frame/verbs.csv): count("V")
and count("X V") for X in the be-forms {was, were, been, is, are, be} and the perfect forms {has, have, had}.
Counts are exact counts of the token sequence the API's Llama tokenizer assigns to the query string (lowercase;
the tokenizer's word-initial marker makes "V" match word-initial occurrences). Every query, its token ids and the
returned count are saved. Retries with backoff on failures; 4 concurrent requests.

Output: data/round5/counts/pile_counts.csv (one row per verb x query), meta.json.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

API = "https://api.infini-gram.io/"
BE = ("was", "were", "been", "is", "are", "be")
PERF = ("has", "have", "had")


def count(index, query, tries=6):
    body = json.dumps({"index": index, "query_type": "count", "query": query}).encode()
    for t in range(tries):
        try:
            req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            if "error" in d:
                raise RuntimeError(d["error"])
            return {"query": query, "count": int(d["count"]), "approx": bool(d.get("approx", False)),
                    "token_ids": " ".join(map(str, d.get("token_ids", [])))}
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(2 ** t)
    return {"query": query, "count": None, "approx": None, "token_ids": "", "error": err}


def main(args):
    V = pd.read_csv(args.verbs, keep_default_na=False)
    queries = []
    for v in V.verb:
        queries.append((v, "overall", v))
        queries += [(v, f"be_{x}", f"{x} {v}") for x in BE]
        queries += [(v, f"perf_{x}", f"{x} {v}") for x in PERF]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(lambda q: count(args.index, q[2]), queries))
    df = pd.DataFrame([{"verb": v, "slot": s, **r} for (v, s, _), r in zip(queries, res)])
    df.to_csv(out / "pile_counts.csv", index=False)
    failed = int(df["count"].isna().sum())
    meta = {"index": args.index, "api": API, "queries": len(df), "failed": failed, "seconds": round(time.time() - t0),
            "date": time.strftime("%Y-%m-%d"), "be_forms": BE, "perfect_forms": PERF}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbs", default="data/round5/frame/verbs.csv")
    ap.add_argument("--index", default="v4_piletrain_llama")
    ap.add_argument("--out-dir", default="data/round5/counts")
    ap.add_argument("--workers", type=int, default=4)
    main(ap.parse_args())
