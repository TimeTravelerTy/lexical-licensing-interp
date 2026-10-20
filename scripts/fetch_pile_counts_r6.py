#!/usr/bin/env python3
"""Round 6, A1: fetch the Pile exposure counts (reports/round6/plan.md, A1) from the infini-gram API.

Input: data/round6/counts/queries.csv (build_count_queries.py). Index `v4_piletrain_llama`. Rows marked for reuse take
round 5's count; the others are fetched sequentially: kind "string" as a count query on the string (the API's Llama-2
tokenizer, word-initial marker included); kind "ids_after_newline" as a token-id count of the ids that follow the
newline token (id 13) in the result of the row named in depends_on (fail if the first id is not 13 or nothing follows).
Every result is appended to a partial JSONL (resume by rerunning). Delay between requests adapts: +0.2 s after a
throttle (HTTP 429 / Forbidden; then a 60-300 s wait), never below --delay.

Output: data/round6/counts/pile_counts.csv (one row per query: verb, slot, kind, query, query_ids, count, approx,
token_ids, tokens, source), counts_meta.json.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://api.infini-gram.io/"
NEWLINE = 13


class Client:
    def __init__(self, index, delay):
        self.index, self.delay, self.base = index, delay, delay
        self.throttles = 0

    def count(self, query=None, ids=None, tries=40):
        body = {"index": self.index, "query_type": "count"}
        body.update({"query": query} if ids is None else {"query_ids": ids})
        err = ""
        for t in range(tries):
            try:
                req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=90) as r:
                    d = json.loads(r.read())
                if "error" in d or "message" in d:
                    raise RuntimeError(d.get("error") or d.get("message"))
                time.sleep(self.delay)
                return {"count": int(d["count"]), "approx": bool(d.get("approx", False)),
                        "token_ids": " ".join(map(str, d.get("token_ids", []))), "tokens": "|".join(d.get("tokens", []))}
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}"
                if e.code in (429, 403):
                    self.throttles += 1
                    self.delay = min(self.delay + 0.2, 5.0)
                    time.sleep(300 if e.code == 403 else 60)
                else:
                    time.sleep(min(2 ** t, 60))
            except Exception as e:  # noqa: BLE001
                err = str(e)
                time.sleep(min(2 ** t, 60))
        raise SystemExit(f"query failed after {tries} tries: {query or ids} ({err}); rerun to resume")


def main(args):
    Q = pd.read_csv(args.queries, keep_default_na=False, dtype=str)
    out = Path(args.out_dir)
    si, sn = map(int, args.shard.split("/"))
    part = out / ("pile_counts.partial.jsonl" if sn == 1 else f"pile_counts.partial.shard{si}.jsonl")
    vorder = {v: j for j, v in enumerate(dict.fromkeys(Q.verb))}
    mine = Q.verb.map(vorder) % sn == si  # this process fetches its own verbs; records of every shard are loaded
    done = {}
    want = {(v, sl): (k, q) for v, sl, k, q in zip(Q.verb, Q.slot, Q.kind, Q["query"])}
    for pf in sorted(out.glob("pile_counts.partial*.jsonl")):
        lines = pf.read_text().splitlines()
        for j, line in enumerate(lines):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                if j == len(lines) - 1 and pf == part:  # an interrupted final write: drop it and refetch
                    part.write_text("\n".join(lines[:-1]) + ("\n" if len(lines) > 1 else ""))
                    break
                if j == len(lines) - 1:  # another shard is mid-write
                    break
                raise
            key = (r["verb"], r["slot"])
            if key in want:  # provenance: the stored request must be the current one
                assert (r["kind"], r["query"]) == want[key], f"stale partial record for {key}: {r['query']!r}"
                assert r.get("index", args.index) == args.index, f"partial record for {key} from another index"
            done[key] = r
    cl = Client(args.index, args.delay)
    t0, n_new = time.time(), 0
    with part.open("a") as fh:
        for i, r in enumerate(Q.itertuples(index=False)):
            key = (r.verb, r.slot)
            if key in done or not mine.iloc[i]:
                continue
            rec = {"verb": r.verb, "slot": r.slot, "kind": r.kind, "query": r.query, "query_ids": "", "index": args.index}
            if r.reuse_count != "":
                rec.update({"count": int(float(r.reuse_count)), "approx": r.reuse_approx == "True",
                            "token_ids": r.reuse_token_ids, "tokens": r.reuse_tokens, "source": "round5"})
            elif r.kind == "string":
                rec.update(cl.count(query=r.query), source="round6")
                n_new += 1
            elif r.kind == "ids_after_newline":
                dep = done[(r.verb, r.depends_on)]
                ids = [int(x) for x in dep["token_ids"].split()]
                if not ids or ids[0] != NEWLINE or len(ids) < 2:
                    raise SystemExit(f"{key}: dependency {r.depends_on} token ids {ids} do not start with a newline")
                rec["query_ids"] = " ".join(map(str, ids[1:]))
                rec.update(cl.count(ids=ids[1:]), source="round6")
                n_new += 1
            else:
                raise SystemExit(f"unknown kind {r.kind}")
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done[key] = rec
            if n_new and n_new % 200 == 0:
                el = time.time() - t0
                print(f"{i + 1}/{len(Q)} rows; {n_new} fetched in {el / 60:.0f} min "
                      f"({el / n_new:.2f} s/query); delay {cl.delay:.1f} s; throttles {cl.throttles}", flush=True)
    for pf in sorted(out.glob("pile_counts.partial*.jsonl")):  # collect the other shards' records
        for line in pf.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.setdefault((r["verb"], r["slot"]), r)
    missing = [k for k in zip(Q.verb, Q.slot) if k not in done]
    if missing:
        print(f"shard {si}/{sn} finished; {len(missing)} rows still missing (other shards running); rerun to merge")
        return
    df = pd.DataFrame([done[(v, s)] for v, s in zip(Q.verb, Q.slot)])
    assert df["count"].notna().all()
    df.to_csv(out / "pile_counts.csv", index=False)
    meta = {"index": args.index, "api": API, "queries": len(df), "fetched_round6": int((df.source == "round6").sum()),
            "reused_round5": int((df.source == "round5").sum()), "approx": int(df.approx.astype(bool).sum()),
            "date": time.strftime("%Y-%m-%d"), "final_delay_s": cl.delay, "throttles_this_session": cl.throttles}
    (out / "counts_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", default="data/round6/counts/queries.csv")
    ap.add_argument("--index", default="v4_piletrain_llama")
    ap.add_argument("--out-dir", default="data/round6/counts")
    ap.add_argument("--delay", type=float, default=0.6, help="minimum seconds between requests")
    ap.add_argument("--shard", default="0/1", help="i/n: fetch the verbs whose position mod n is i (separate partial files)")
    main(ap.parse_args())
