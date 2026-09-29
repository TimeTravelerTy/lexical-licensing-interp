#!/usr/bin/env python3
"""Collect active-voice items for rating bad-side (intransitive verb) fit.

"The hat was salivated by the gentleman" cannot be rated for plausibility
without the rater also penalizing its grammar. Its active counterpart can:
is the noun a plausible subject of the intransitive verb? Each curated-cross
row gets two bad-side items:

- `agent`: "The gentleman salivated." (passive_1 only). The agent is seen
  after the participle, so it bears on the suffix and whole sentence.
- `patient`: "The hat salivated." The participle token is predicted from
  "The hat was", so only the patient can shape the participle margin.

All bad participles equal their simple past (the one irregular, crept, too).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def item_id(sentence: str) -> str:
    return hashlib.sha1(sentence.encode()).hexdigest()[:12]


def run(args):
    v2 = [json.loads(line) for line in Path(args.v2_pairs).open(encoding="utf-8")]
    context_nouns = {}
    for r in v2:
        if r["frame_id"] == "was":
            context_nouns[f"curated/{r['paradigm']}/{r['band']}/{r['good_lemma']}"] = (r["patient"], r["agent"])
    items, links = {}, []
    for line in Path(args.pairs).open(encoding="utf-8"):
        r = json.loads(line)
        if r["context_set"] != "curated":
            continue
        patient, agent = context_nouns[r["context_id"]]
        roles = [("patient", patient)] + ([("agent", agent)] if r["paradigm"] == "passive_1" else [])
        for role, noun in roles:
            sentence = f"The {noun} {r['bad_verb']}."
            iid = item_id(sentence)
            items.setdefault(iid, {"item_id": iid, "kind": "bad_active", "paradigm": "active",
                                   "sentence": sentence})
            links.append({"item_id": iid, "pair_id": r["pair_id"], "paradigm": r["paradigm"], "role": role,
                          "verb_pair": r["verb_pair"], "verb_band": r["verb_band"],
                          "context_id": r["context_id"], "own_context": r["own_context"],
                          "bad_verb": r["bad_verb"], "noun": noun})
    out_dir = Path(args.out_dir)
    with (out_dir / "bad_items.jsonl").open("w", encoding="utf-8") as f:
        for item in sorted(items.values(), key=lambda x: x["item_id"]):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (out_dir / "bad_item_links.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(links[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(links)
    print(json.dumps({"items": len(items), "links": len(links)}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="data/passive_band_cross/pairs.jsonl")
    ap.add_argument("--v2-pairs", default="data/matched_passives_v2/pairs.jsonl")
    ap.add_argument("--out-dir", default="data/fit_ratings")
    run(ap.parse_args())
