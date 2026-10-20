#!/usr/bin/env python3
"""Round 6, A1: the infini-gram query list for the Pile exposure counts (reports/round6/plan.md, A1).

Verbs: the union of the D11 participles (data/round6/frame_d11/verbs.csv) and the curated participles
(data/round5/frame/verbs.csv, curated == 1): 608 forms. Per verb v (capitalized V; simple past p):
- overall variants: "v", "V" (word-initial marker); "\\nv", "\\nV" (their token ids after the newline give the
  unmarked forms, which are then counted by token ids: slots bare_low, bare_cap, kind "ids_after_newline");
- be-forms "X v", X in was, were, been, is, are, be, being;
- perfect "X v", X in has, have, had, having;
- competing frame, primary: clause-anchored simple past "a P p", a in that, when, and; P in he, she, they (no
  auxiliary inversion is possible after these anchors); secondary: broad pronoun adjacency "P p", P in he, she, it,
  they, He, She, It, They (also matches inverted questions such as "Was it abandoned?");
- B3 frames "P p the" and "P p.", P in he, she, they, He, She, They.
A query whose exact string was already counted in round 5 (data/round5/counts/pile_counts.csv; same index) is
marked for reuse. Output: data/round6/counts/queries.csv (verb, slot, kind, query, depends_on, reuse_count,
reuse_token_ids, reuse_tokens, reuse_approx), queries_meta.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

BE = ("was", "were", "been", "is", "are", "be", "being")
PERF = ("has", "have", "had", "having")
COMP = ("he", "she", "it", "they", "He", "She", "It", "They")  # broad pronoun adjacency (secondary)
ANCHORS = ("that", "when", "and")  # clause-anchored competing frame (primary)
ANCHOR_SUBJ = ("he", "she", "they")
FRAME = ("he", "she", "they", "He", "She", "They")
PAST = {"arisen": "arose", "striven": "strove", "browbeaten": "browbeat", "outdone": "outdid"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    d11 = pd.read_csv(args.d11, keep_default_na=False)
    r5 = pd.read_csv(args.r5_verbs, keep_default_na=False)
    verbs = sorted(set(d11.verb) | set(r5[r5.curated == 1].verb))
    assert all(v.isalpha() and v.islower() for v in verbs)
    # irregular forms: every participle not ending in -ed must be listed in PAST or be its own past
    irregular = [v for v in verbs if not v.endswith("ed")]
    same = {"bet", "crept", "lent", "outshone", "recast", "sent", "slept", "dwelt", "knelt", "slunk", "beheld", "waylaid"}
    assert set(irregular) <= same | set(PAST), sorted(set(irregular) - same - set(PAST))
    q = []
    for v in verbs:
        cap, past = v[0].upper() + v[1:], PAST.get(v, v)
        q += [(v, "overall", "string", v, ""), (v, "overall_cap", "string", cap, ""),
              (v, "nl_low", "string", "\n" + v, ""), (v, "bare_low", "ids_after_newline", "", "nl_low"),
              (v, "nl_cap", "string", "\n" + cap, ""), (v, "bare_cap", "ids_after_newline", "", "nl_cap")]
        q += [(v, f"be_{x}", "string", f"{x} {v}", "") for x in BE]
        q += [(v, f"perf_{x}", "string", f"{x} {v}", "") for x in PERF]
        q += [(v, f"comp_{p}", "string", f"{p} {past}", "") for p in COMP]
        q += [(v, f"compa_{a}_{p}", "string", f"{a} {p} {past}", "") for a in ANCHORS for p in ANCHOR_SUBJ]
        q += [(v, f"obj_{p}", "string", f"{p} {past} the", "") for p in FRAME]
        q += [(v, f"end_{p}", "string", f"{p} {past}.", "") for p in FRAME]
    Q = pd.DataFrame(q, columns=["verb", "slot", "kind", "query", "depends_on"])
    # fetch order (the API is rate-limited): primary slots first, then B3 frame counts, then the secondary measures
    tier = lambda sl: (0 if sl.startswith(("overall", "nl_", "be_", "perf_", "compa_")) else
                       1 if sl.startswith(("obj_", "end_")) else 2 if sl.startswith("comp_") else 3)
    Q["tier"] = Q.slot.map(tier)
    Q = Q.sort_values(["tier"], kind="stable").drop(columns="tier").reset_index(drop=True)
    old = pd.read_csv(args.r5_counts)
    old_meta = json.loads((Path(args.r5_counts).parent / "meta.json").read_text())
    assert old_meta["index"] == args.index, f"round-5 counts come from {old_meta['index']}, not {args.index}"
    old = old[old["count"].notna() & ~old.slot.str.endswith("_or")]
    assert old.groupby("query")["count"].nunique().max() == 1, "conflicting round-5 counts for one query string"
    old = old.drop_duplicates("query").set_index("query")
    hit = Q.kind.eq("string") & Q["query"].isin(old.index)
    Q["reuse_count"] = Q["query"].map(old["count"]).where(hit)
    Q["reuse_token_ids"] = Q["query"].map(old["token_ids"]).where(hit)
    Q["reuse_tokens"] = Q["query"].map(old["tokens"]).where(hit)
    Q["reuse_approx"] = Q["query"].map(old["approx"]).where(hit)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Q.to_csv(out / "queries.csv", index=False)
    meta = {"verbs": len(verbs), "queries": len(Q), "reused": int(hit.sum()), "to_fetch": int((~hit).sum()),
            "inputs": {p: sha(p) for p in (args.d11, args.r5_verbs, args.r5_counts)},
            "past_forms": PAST, "be": BE, "perfect": PERF, "competing_subjects_broad": COMP,
            "competing_anchors": ANCHORS, "competing_anchor_subjects": ANCHOR_SUBJ, "frame_subjects": FRAME, "index": args.index,
            "outputs": {"queries.csv": sha(out / "queries.csv")}}
    (out / "queries_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("verbs", "queries", "reused", "to_fetch")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d11", default="data/round6/frame_d11/verbs.csv")
    ap.add_argument("--r5-verbs", default="data/round5/frame/verbs.csv")
    ap.add_argument("--r5-counts", default="data/round5/counts/pile_counts.csv")
    ap.add_argument("--index", default="v4_piletrain_llama")
    ap.add_argument("--out-dir", default="data/round6/counts")
    main(ap.parse_args())
