#!/usr/bin/env python3
"""Adapt released FreqBLiMP passive pairs for the Pythia LP scorer."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PARTICIPLE = re.compile(
    r"\b(?:was|were|is|are|wasn't|weren't|isn't|aren't)\s+"
    r"([A-Za-z-]+)(?=\s+by\b|[.?!])"
)


def run(args):
    root = Path(args.freqblimp_root) / "data" / "freqblimp"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as target:
        for band in ("head", "tail", "xtail"):
            for paradigm in ("passive_1", "passive_2"):
                path = root / band / f"{paradigm}.jsonl"
                with path.open(encoding="utf-8") as source:
                    for index, line in enumerate(source):
                        original = json.loads(line)
                        good = original["sentence_good"]
                        bad = original["sentence_bad"]
                        gm, bm = PARTICIPLE.search(good), PARTICIPLE.search(bad)
                        if gm is None or bm is None:
                            raise ValueError(f"Unparsed passive in {path}:{index + 1}")
                        prefix, suffix = good[:gm.start(1)], good[gm.end(1):]
                        if prefix != bad[:bm.start(1)] or suffix != bad[bm.end(1):]:
                            raise ValueError(f"Nonmatching context in {path}:{index + 1}")
                        row = {
                            "pair_id": f"{band}/{paradigm}/{index}",
                            "band": band,
                            "paradigm": paradigm,
                            "source_index": index,
                            "source_pair_id": original.get("pairID", ""),
                            "prefix": prefix,
                            "suffix": suffix,
                            "good_verb": gm.group(1),
                            "bad_verb": bm.group(1),
                            "sentence_good": good,
                            "sentence_bad": bad,
                        }
                        target.write(json.dumps(row, ensure_ascii=False) + "\n")
                        count += 1
    if count != 6000:
        raise ValueError(f"Expected 6000 released passive pairs, got {count}")
    print(f"Wrote {count} pairs to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freqblimp-root", required=True)
    parser.add_argument("--out", default="data/freqblimp_original_passives/pairs.jsonl")
    run(parser.parse_args())
