#!/usr/bin/env python3
"""Put v2's exact verb pairings into released FreqBLiMP noun contexts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def run(args):
    v2 = [json.loads(line) for line in Path(args.v2_pairs).open(encoding="utf-8")]
    selected = {}
    for row in v2:
        if row["frame_id"] != "was":
            continue
        key = (row["band"], row["paradigm"], row["good_verb"].lower())
        if key in selected:
            raise ValueError(f"Duplicate selected good participle: {key}")
        selected[key] = row

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with Path(args.original_pairs).open(encoding="utf-8") as source, out.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            row = json.loads(line)
            key = (row["band"], row["paradigm"], row["good_verb"].lower())
            match = selected.get(key)
            if match is None:
                continue
            output = dict(row)
            output["source_bad_verb"] = row["bad_verb"]
            output["selected_pair_id"] = match["pair_id"]
            output["bad_verb"] = match["bad_verb"]
            output["sentence_bad"] = row["prefix"] + match["bad_verb"] + row["suffix"]
            target.write(json.dumps(output, ensure_ascii=False) + "\n")
            counts[(row["band"], row["paradigm"])] = counts.get(
                (row["band"], row["paradigm"]), 0
            ) + 1
    print(f"Wrote {sum(counts.values())} pairs to {out}: {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-pairs", default="data/matched_passives_v2/pairs.jsonl")
    parser.add_argument("--original-pairs", default="data/freqblimp_original_passives/pairs.jsonl")
    parser.add_argument("--out", default="data/freqblimp_original_contexts_v2_verbs/pairs.jsonl")
    run(parser.parse_args())
