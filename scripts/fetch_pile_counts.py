#!/usr/bin/env python3
"""Round 5, A3: Pile counts for the A1 participles via the infini-gram API (reports/round5/plan.md, A3).

Index `v4_piletrain_llama` (the Pile training set). For each participle V (data/round5/frame/verbs.csv): count("V")
and count("X V") for X in the be-forms {was, were, been, is, are, be} and the perfect forms {has, have, had}, also with
the auxiliary capitalized for the curated verbs (sensitivity).
Counts are exact counts of the token sequence the API's Llama tokenizer assigns to the query string (lowercase;
the tokenizer's word-initial marker makes "V" match word-initial occurrences). Every query, its token ids and the
returned count are saved. Sequential requests with a delay, retries with backoff, incremental saving and resume.

Output: data/round5/counts/pile_counts.csv (one row per verb x query), meta.json.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://api.infini-gram.io/"
BE = ("was", "were", "been", "is", "are", "be")
PERF = ("has", "have", "had")


def count(index, query, tries=40):
    body = json.dumps({"index": index, "query_type": "count", "query": query}).encode()
    for t in range(tries):
        try:
            req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            if "error" in d or "message" in d:
                raise RuntimeError(d.get("error") or d.get("message"))
            flat = lambda x: x if isinstance(x, str) else " ".join(map(str, x)) if not any(isinstance(y, list) for y in x) else json.dumps(x)
            return {"query": query, "count": int(d["count"]), "approx": bool(d.get("approx", False)),
                    "token_ids": flat(d.get("token_ids", [])), "tokens": json.dumps(d.get("tokens", []))}
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(300 if "Forbidden" in err else min(2 ** t, 60))  # throttled: wait it out
    return {"query": query, "count": None, "approx": None, "token_ids": "", "tokens": "", "error": err}


def main(args):
    V = pd.read_csv(args.verbs, keep_default_na=False)
    queries = []
    for v, cur in zip(V.verb, V.curated):
        queries.append((v, "overall", v))
        if cur == 1:  # curated (primary): exact single-sequence counts per auxiliary
            queries += [(v, f"be_{x}", f"{x} {v}") for x in BE]
            queries += [(v, f"perf_{x}", f"{x} {v}") for x in PERF]
            # capitalized auxiliaries (sensitivity): one OR query (infini-gram flags CNF counts as approximate)
            queries.append((v, "capbe_or", " OR ".join(f"{x.capitalize()} {v}" for x in BE)))
        else:  # pool (secondary): slot sums as OR queries (approximate)
            queries.append((v, "be_or", " OR ".join(f"{x} {v}" for x in BE)))
            queries.append((v, "perf_or", " OR ".join(f"{x} {v}" for x in PERF)))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    part = out / "pile_counts.partial.jsonl"
    done = set()
    if part.exists():  # resume
        for line in part.read_text().splitlines():
            r = json.loads(line)
            if r.get("count") is not None:
                done.add((r["verb"], r["slot"]))
    t0 = time.time()
    with part.open("a") as fh:
        for i, (v, sl, q) in enumerate(queries):
            if (v, sl) in done:
                continue
            r = count(args.index, q)
            fh.write(json.dumps({"verb": v, "slot": sl, **r}) + "\n")
            fh.flush()
            if r["count"] is None:
                raise SystemExit(f"query failed after retries: {q!r} ({r.get('error')}); rerun to resume")
            time.sleep(args.delay)
            if i % 500 == 0:
                print(f"{i}/{len(queries)} ({time.time() - t0:.0f}s)", flush=True)
    rows = {}
    for line in part.read_text().splitlines():
        r = json.loads(line)
        if r.get("count") is not None:
            rows[(r["verb"], r["slot"])] = r
    df = pd.DataFrame([rows[(v, sl)] for v, sl, _ in queries])
    df.to_csv(out / "pile_counts.csv", index=False)
    meta = {"index": args.index, "api": API, "queries": len(df), "failed": int(df["count"].isna().sum()),
            "date": time.strftime("%Y-%m-%d"), "be_forms": BE, "perfect_forms": PERF, "delay_s": args.delay}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbs", default="data/round5/frame/verbs.csv")
    ap.add_argument("--index", default="v4_piletrain_llama")
    ap.add_argument("--out-dir", default="data/round5/counts")
    ap.add_argument("--delay", type=float, default=0.2, help="seconds between requests (sequential)")
    main(ap.parse_args())
