#!/usr/bin/env python3
"""Round 5, A4: Pile-validation windows containing each edited participle (reports/round5/plan.md, A4).

infini-gram index `v4_pileval_llama` (the Pile validation set, held out from Pythia's training). For each verb of
data/round5/restore/manifest.csv: `find` gives the occurrence segments; up to --per-verb occurrences at evenly spaced
ranks are retrieved with `get_doc_by_rank` (a window of --disp tokens around the occurrence, decoded text). Windows
from a document already used are skipped. Sequential requests with a delay; retries with backoff.

Output: data/round5/restore/pileval_docs.jsonl (verb, doc_ix, rank, occ_char = the selected occurrence's character
offset, text), pileval_meta.json.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://api.infini-gram.io/"


def call(payload, tries=8):
    body = json.dumps(payload).encode()
    for t in range(tries):
        try:
            req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            if "error" in d:
                raise RuntimeError(d["error"])
            return d
        except Exception:  # noqa: BLE001
            time.sleep(min(2 ** t, 120))
    raise SystemExit(f"request failed: {payload}")


def main(args):
    man = pd.read_csv(args.manifest)
    out = Path(args.out)
    seen, rows, per = set(), [], {}
    for v in man.verb:
        f = call({"index": args.index, "query_type": "find", "query": v})
        segs = f["segment_by_shard"]
        cnt = int(f["cnt"])
        flat = [(s, r) for s, (a, b) in enumerate(segs) for r in range(a, b)] if cnt <= 50000 else None
        n = min(args.per_verb, cnt)
        picks = []
        if n:
            idx = np.unique(np.linspace(0, cnt - 1, n).round().astype(int))
            if flat is not None:
                picks = [flat[i] for i in idx]
            else:  # walk the segments without materializing them
                bounds = np.cumsum([b - a for a, b in segs])
                for i in idx:
                    s = int(np.searchsorted(bounds, i, side="right"))
                    off = i - (bounds[s - 1] if s else 0)
                    picks.append((s, segs[s][0] + int(off)))
        got = missing = 0
        for s, r in picks:
            # empty query: one continuously decoded span (highlight splitting can glue words); the window is centred on
            # the occurrence at this rank, so the occurrence nearest the middle is the selected one
            d = call({"index": args.index, "query_type": "get_doc_by_rank", "s": s, "rank": r, "max_disp_len": args.disp,
                      "query": ""})
            time.sleep(args.delay)
            dix = int(d["doc_ix"])
            if dix in seen:
                continue
            text = "".join(sp[0] for sp in d["spans"])
            occ = [m.start() for m in re.finditer(rf"(?<![A-Za-z]){re.escape(v)}(?![A-Za-z])", text)]
            if not occ:
                missing += 1
                continue
            seen.add(dix)
            mid = len(text) / 2
            rows.append({"verb": v, "doc_ix": dix, "shard": s, "rank": r, "occ_char": min(occ, key=lambda o: abs(o - mid)),
                         "text": text})
            got += 1
        per[v] = {"occurrences": cnt, "windows": got, "no_surface_match": missing}
        print(v, per[v], flush=True)
        time.sleep(args.delay)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out.parent / "pileval_meta.json").write_text(json.dumps({"index": args.index, "per_verb": per, "disp": args.disp,
                                                              "date": time.strftime("%Y-%m-%d")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/round5/restore/manifest.csv")
    ap.add_argument("--index", default="v4_pileval_llama")
    ap.add_argument("--out", default="data/round5/restore/pileval_docs.jsonl")
    ap.add_argument("--per-verb", type=int, default=20)
    ap.add_argument("--disp", type=int, default=600)
    ap.add_argument("--delay", type=float, default=0.3)
    main(ap.parse_args())
