#!/usr/bin/env python3
"""Cross selected passive verb pairs with contexts from every frequency band.

The released FreqBLiMP passives band nouns together with verbs, so released
Head-vs-XTail comparisons confound verb and noun frequency. This builder puts
every v2 verb pair (26/50/50) into the same sampled released contexts from all
three bands. Diagonal and off-diagonal cells are built identically: no context
keeps its original verb. The v2 curated `was` contexts are also crossed with
every verb pair; `own_context` marks the context written for that verb pair.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from wordfreq import zipf_frequency


BANDS = ("head", "tail", "xtail")
PARADIGMS = ("passive_1", "passive_2")
AUX = re.compile(r"\b(was|were|is|are|wasn't|weren't|isn't|aren't|had been)\s*$")
DETERMINERS = {"a", "an", "the", "some", "all", "every", "each", "this", "that",
               "these", "those", "most", "many", "few", "no", "both", "several"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def noun_phrase_features(words: list[str], role: str) -> dict:
    """Determiner class, proper-name flag, and head-noun Zipf of a simple NP."""
    first = words[0].lower() if words else ""
    is_name = bool(words) and first not in DETERMINERS and words[-1][:1].isupper()
    return {
        f"{role}_det": first if first in DETERMINERS else ("name" if is_name else "bare"),
        f"{role}_is_name": int(is_name),
        f"{role}_head": words[-1] if words else "",
        f"{role}_zipf": float(zipf_frequency(words[-1].lower(), "en")) if words else "",
    }


def context_features(prefix: str, suffix: str) -> dict:
    aux = AUX.search(prefix)
    if aux is None:
        raise ValueError(f"No auxiliary at end of prefix {prefix!r}")
    aux_word = aux.group(1)
    features = {
        "aux": aux_word,
        "negated": int("n't" in aux_word),
        "plural_aux": int(aux_word.startswith(("were", "are", "weren", "aren"))),
        "present_aux": int(aux_word.startswith(("is", "are"))),
    }
    features.update(noun_phrase_features(prefix[:aux.start()].split(), "patient"))
    agent_words = suffix.strip().rstrip(".").split()[1:] if " by " in suffix else []
    features.update(noun_phrase_features(agent_words, "agent") if agent_words else
                    {"agent_det": "", "agent_is_name": "", "agent_head": "", "agent_zipf": ""})
    return features


def released_contexts(path: Path, per_cell: int, seed: int) -> list[dict]:
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        by_cell.setdefault((row["band"], row["paradigm"]), []).append(row)
    contexts = []
    for band in BANDS:
        for paradigm in PARADIGMS:
            # Deduplicate on the context itself so each sampled frame is distinct.
            unique = {}
            for row in by_cell[band, paradigm]:
                unique.setdefault((row["prefix"], row["suffix"]), row)
            rows = sorted(unique.values(), key=lambda r: r["source_index"])
            if len(rows) < per_cell:
                raise ValueError(f"Only {len(rows)} distinct contexts for {band}/{paradigm}")
            rng = random.Random(f"{seed}|{band}|{paradigm}")
            for row in sorted(rng.sample(rows, per_cell), key=lambda r: r["source_index"]):
                contexts.append({
                    "context_id": f"released/{band}/{paradigm}/{row['source_index']}",
                    "context_set": "released", "context_band": band, "paradigm": paradigm,
                    "context_source": f"{row['source_index']}",
                    "context_orig_good": row["good_verb"], "context_orig_bad": row["bad_verb"],
                    "prefix": row["prefix"], "suffix": row["suffix"],
                })
    return contexts


def curated_contexts(v2_rows: list[dict]) -> list[dict]:
    contexts = []
    for row in v2_rows:
        if row["frame_id"] != "was":
            continue
        contexts.append({
            "context_id": f"curated/{row['paradigm']}/{row['band']}/{row['good_lemma']}",
            "context_set": "curated", "context_band": row["band"], "paradigm": row["paradigm"],
            "context_source": f"{row['band']}/{row['good_lemma']}",
            "context_orig_good": row["good_verb"], "context_orig_bad": row["bad_verb"],
            "prefix": row["prefix"], "suffix": row["suffix"],
        })
    return contexts


def run(args):
    v2_path, released_path = Path(args.v2_pairs), Path(args.released_pairs)
    v2_rows = [json.loads(line) for line in v2_path.open(encoding="utf-8")]
    verbs = {}
    for row in v2_rows:
        if row["frame_id"] == "was":
            verbs.setdefault(row["paradigm"], []).append(row)
    lexical_keys = ("good_lemma", "bad_lemma", "good_verb", "bad_verb", "bad_active_class",
                    "good_lemma_zipf", "bad_lemma_zipf", "good_form_zipf", "bad_form_zipf",
                    "lemma_zipf_gap", "form_zipf_gap", "both_lemmas_in_band")

    contexts = released_contexts(released_path, args.contexts_per_cell, args.seed)
    if not args.no_curated:
        contexts += curated_contexts(v2_rows)
    for context in contexts:
        context.update(context_features(context["prefix"], context["suffix"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    with out.open("w", encoding="utf-8") as f:
        for context in contexts:
            for verb in verbs[context["paradigm"]]:
                verb_pair = f"{verb['band']}/{verb['good_lemma']}/{verb['bad_lemma']}"
                row = {
                    "pair_id": f"{verb_pair}|{context['context_id']}",
                    "verb_pair": verb_pair, "verb_band": verb["band"],
                    "own_context": int(context["context_set"] == "curated"
                                       and context["context_source"] == f"{verb['band']}/{verb['good_lemma']}"),
                    **context,
                    **{key: verb[key] for key in lexical_keys},
                    "sentence_good": context["prefix"] + verb["good_verb"] + context["suffix"],
                    "sentence_bad": context["prefix"] + verb["bad_verb"] + context["suffix"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[context["context_set"], context["paradigm"], verb["band"], context["context_band"]] += 1

    context_out = Path(args.contexts_out)
    context_out.parent.mkdir(parents=True, exist_ok=True)
    with context_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(contexts[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(contexts)

    manifest = {
        "seed": args.seed, "contexts_per_cell": args.contexts_per_cell,
        "curated_included": not args.no_curated,
        "inputs": {str(v2_path): sha256(v2_path), str(released_path): sha256(released_path)},
        "verb_pairs": {p: dict(Counter(v["band"] for v in rows)) for p, rows in sorted(verbs.items())},
        "contexts": dict(Counter(f"{c['context_set']}/{c['paradigm']}/{c['context_band']}" for c in contexts)),
        "cells": {"/".join(k): n for k, n in sorted(counts.items())},
        "total_pairs": sum(counts.values()),
        "out": str(out), "out_sha256": sha256(out),
        "contexts_out": str(context_out), "contexts_sha256": sha256(context_out),
    }
    Path(args.manifest_out).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("verb_pairs", "contexts", "total_pairs", "out_sha256")}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v2-pairs", default="data/matched_passives_v2/pairs.jsonl")
    ap.add_argument("--released-pairs", default="data/freqblimp_original_passives/pairs.jsonl")
    ap.add_argument("--contexts-per-cell", type=int, default=100)
    ap.add_argument("--no-curated", action="store_true")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default="data/passive_band_cross/pairs.jsonl")
    ap.add_argument("--contexts-out", default="data/passive_band_cross/contexts.csv")
    ap.add_argument("--manifest-out", default="data/passive_band_cross/manifest.json")
    run(ap.parse_args())
