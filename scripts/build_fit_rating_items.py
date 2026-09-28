#!/usr/bin/env python3
"""Collect the curated-cross good sentences for plausibility (fit) rating.

Main items are the unique good sentences of the curated cross: every v2 good
verb in every curated `was` context. Two validation sets check that ratings
measure event plausibility rather than word rarity:

- `reversal`: each pair's own passive_1 context with patient and agent
  swapped (usually, but not always, less plausible);
- `human_sheet.csv`: a stratified blind sample for a human rater.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path


def item_id(sentence: str) -> str:
    return hashlib.sha1(sentence.encode()).hexdigest()[:12]


def run(args):
    rows = [json.loads(line) for line in Path(args.pairs).open(encoding="utf-8")]
    curated = [r for r in rows if r["context_set"] == "curated"]
    items: dict[str, dict] = {}
    links = []
    for r in curated:
        sentence = r["sentence_good"]
        iid = item_id(sentence)
        items.setdefault(iid, {"item_id": iid, "kind": "cross", "paradigm": r["paradigm"],
                               "sentence": sentence})
        links.append({"item_id": iid, "pair_id": r["pair_id"], "paradigm": r["paradigm"],
                      "verb_pair": r["verb_pair"], "verb_band": r["verb_band"],
                      "context_id": r["context_id"], "context_band": r["context_band"],
                      "own_context": r["own_context"], "good_verb": r["good_verb"]})
    n_cross = len(items)

    v2 = [json.loads(line) for line in Path(args.v2_pairs).open(encoding="utf-8")]
    for r in v2:
        if r["frame_id"] != "was" or r["paradigm"] != "passive_1":
            continue
        sentence = f"The {r['agent']} was {r['good_verb']} by the {r['patient']}."
        iid = item_id(sentence)
        items.setdefault(iid, {"item_id": iid, "kind": "reversal", "paradigm": "passive_1",
                               "sentence": sentence})
        links.append({"item_id": iid, "pair_id": "", "paradigm": "passive_1",
                      "verb_pair": f"{r['band']}/{r['good_lemma']}/{r['bad_lemma']}",
                      "verb_band": r["band"], "context_id": "reversal", "context_band": r["band"],
                      "own_context": "", "good_verb": r["good_verb"]})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "items.jsonl").open("w", encoding="utf-8") as f:
        for item in sorted(items.values(), key=lambda x: (x["kind"], x["paradigm"], x["item_id"])):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (out_dir / "item_links.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(links[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(links)

    # Blind human sample: 10 per paradigm x verb band x own/other, plus 12 reversals.
    rng = random.Random(args.seed)
    strata: dict[tuple, list[str]] = {}
    for link in links:
        if link["context_id"] == "reversal":
            key = ("reversal",)
        else:
            key = (link["paradigm"], link["verb_band"], "own" if link["own_context"] else "other")
        strata.setdefault(key, [])
        if link["item_id"] not in strata[key]:
            strata[key].append(link["item_id"])
    sample = []
    for key in sorted(strata):
        pool = sorted(strata[key])
        sample += rng.sample(pool, min(12 if key == ("reversal",) else 10, len(pool)))
    rng.shuffle(sample)
    with (out_dir / "human_sheet.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["item_id", "sentence", "rating_1_to_7"])
        for iid in sample:
            writer.writerow([iid, items[iid]["sentence"], ""])
    print(json.dumps({"cross_items": n_cross, "reversal_items": len(items) - n_cross,
                      "links": len(links), "human_sheet": len(sample)}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="data/passive_band_cross/pairs.jsonl")
    ap.add_argument("--v2-pairs", default="data/matched_passives_v2/pairs.jsonl")
    ap.add_argument("--out-dir", default="data/fit_ratings")
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
