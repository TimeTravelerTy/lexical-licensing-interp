#!/usr/bin/env python3
"""Round 5, A3 audit: what the "was V" proxy counts (reports/round5/plan.md, A3, Audit).

10 curated verbs (5 T, 5 I; per class 2 Head, 2 Tail, 1 XTail, drawn with seed 17). For each, up to 10 occurrences of
"was V" in `v4_piletrain_llama` at evenly spaced ranks (`find` + `get_doc_by_rank`), decoded as one span; the snippet
is cut around the occurrence nearest the window's centre. Labels (eventive passive / adjectival / other) are added by
hand in audit_labels.csv.

Output: data/round5/counts/audit_contexts.csv (verb, role, band, k, count, doc_ix, snippet).
"""

from __future__ import annotations

import argparse
import re
import time

import numpy as np
import pandas as pd

from fetch_pileval_docs import call


def main(args):
    V = pd.read_csv(args.verbs, keep_default_na=False)
    V = V[V.curated == 1].sort_values("verb")
    rng = np.random.default_rng(args.seed)
    pick = []
    for role in ("T", "I"):
        for band, k in (("head", 2), ("tail", 2), ("xtail", 1)):
            g = V[(V.role == role) & (V.band == band)]
            pick += [(v, role, band) for v in rng.choice(g.verb.to_numpy(), k, replace=False)]
    rows = []
    for v, role, band in pick:
        q = f"was {v}"
        f = call({"index": args.index, "query_type": "find", "query": q})
        segs, cnt = f["segment_by_shard"], int(f["cnt"])
        n = min(args.per_verb, cnt)
        bounds = np.cumsum([b - a for a, b in segs])
        for k, i in enumerate(np.unique(np.linspace(0, cnt - 1, n).round().astype(int)) if n else []):
            s = int(np.searchsorted(bounds, i, side="right"))
            r = segs[s][0] + int(i - (bounds[s - 1] if s else 0))
            d = call({"index": args.index, "query_type": "get_doc_by_rank", "s": s, "rank": r,
                      "max_disp_len": args.disp, "query": ""})
            time.sleep(args.delay)
            text = "".join(sp[0] for sp in d["spans"]).replace("\n", " ⏎ ")
            occ = [m.start() for m in re.finditer(rf"(?<![A-Za-z])was {re.escape(v)}(?![A-Za-z])", text)]
            if not occ:
                continue
            o = min(occ, key=lambda x: abs(x - len(text) / 2))
            rows.append({"verb": v, "role": role, "band": band, "k": k, "count": cnt, "doc_ix": int(d["doc_ix"]),
                         "snippet": text[max(0, o - 160):o].lstrip() + "[[" + text[o:o + 4 + len(v)] + "]]"
                         + text[o + 4 + len(v):o + 4 + len(v) + 80]})
        print(v, role, band, cnt, sum(r["verb"] == v for r in rows), flush=True)
        time.sleep(args.delay)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbs", default="data/round5/frame/verbs.csv")
    ap.add_argument("--index", default="v4_piletrain_llama")
    ap.add_argument("--out", default="data/round5/counts/audit_contexts.csv")
    ap.add_argument("--per-verb", type=int, default=10)
    ap.add_argument("--disp", type=int, default=80)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=17)
    main(ap.parse_args())
