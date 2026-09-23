#!/usr/bin/env python3
"""Build a wider, manually checked FreqBLiMP passive verb pilot.

The released FreqBLiMP sentences supply the good verb inventory and the
tail/xtail bad inventory. A reviewed generator-branch CSV supplies head bad
verbs. Only realised participles determine the released frequency regime;
lemma and participle Zipf are then both matched within each verb pair.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from wordfreq import zipf_frequency

from build_matched_passives import (
    BANDS, PARADIGMS, collect_forms, pair_inventory, resolve_lemmas,
)


# These released-data bad forms have a direct-object, causative, or readily
# adjectival reading that can make a bare passive acceptable in some contexts.
# A strict label is more important than filling a numerical quota.
BAD_EXCLUDE = {
    "tail": frozenset("""
        balk branch congregate converge decay dine disapprove disembark flop
        gush materialize meditate mutate proliferate prosper resonate revert
        shiver spiral sprawl swell veer
    """.split()),
    "xtail": frozenset("""
        abut berth billet bluster calve clack clatter debark drone flounce
        gyrate hyperventilate intermarry jig molt ossify patter prattle procreate protrude
        putrefy scoot squawk sunbathe worm zigzag
    """.split()),
}
GOOD_EXCLUDE = {"xtail": frozenset({"gnash", "moralize", "overreach", "wheedle"})}
TEMPLATES = ("was", "had been")


def _rows_for_forms(forms, lemmas, key):
    band = key[0]
    lo, hi = BANDS[band]
    rows = []
    for form in forms[key]:
        options = lemmas.get(form, set())
        if len(options) != 1:
            continue
        form_zipf = float(zipf_frequency(form, "en"))
        if lo <= form_zipf <= hi:
            lemma = next(iter(options))
            rows.append({"lemma": lemma, "form": form,
                         "lemma_zipf": float(zipf_frequency(lemma, "en")),
                         "form_zipf": form_zipf})
    by_lemma = {}
    for row in sorted(rows, key=lambda r: (-r["form_zipf"], r["form"])):
        by_lemma.setdefault(row["lemma"], row)
    return by_lemma


def _shared_inventory(forms, lemmas, band, side):
    first = _rows_for_forms(forms, lemmas, (band, "passive_1", side))
    second = _rows_for_forms(forms, lemmas, (band, "passive_2", side))
    return [first[lemma] for lemma in sorted(first.keys() & second.keys())
            if first[lemma]["form"] == second[lemma]["form"]]


def _head_review(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reviewed = list(csv.DictReader(handle))
    rows = []
    for row in reviewed:
        lemma, form = row["lemma"], row["participle"]
        form_zipf = float(zipf_frequency(form, "en"))
        lo, hi = BANDS["head"]
        if not lo <= form_zipf <= hi:
            raise ValueError(f"Reviewed head participle outside band: {lemma}/{form} ({form_zipf})")
        rows.append({"lemma": lemma, "form": form,
                     "lemma_zipf": float(zipf_frequency(lemma, "en")),
                     "form_zipf": form_zipf, "active_class": row["active_class"]})
    if len({row["lemma"] for row in rows}) != len(rows):
        raise ValueError("Duplicate lemma in reviewed head inventory")
    return rows


def _contexts(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    out = {}
    for row in rows:
        key = (row["band"], row["good_lemma"])
        if key in out:
            raise ValueError(f"Duplicate context: {key}")
        if not row["patient"].strip() or not row["agent"].strip():
            raise ValueError(f"Missing context noun: {key}")
        out[key] = (row["patient"].strip(), row["agent"].strip())
    return out


def build(args):
    root = Path(args.freqblimp_root)
    forms, source_audit = collect_forms(root / "data" / "freqblimp")
    lemmas = resolve_lemmas(root, forms)
    reviewed_head = _head_review(Path(args.head_review))
    contexts = _contexts(Path(args.contexts))
    rows = []
    audit = {"source": source_audit, "head_review": str(args.head_review),
             "contexts": str(args.contexts), "matching_calipers": {
                 "lemma_zipf_max_abs_gap": 0.25, "form_zipf_max_abs_gap": 0.35},
             "band_definition": "realised participle Zipf only, as in released FreqBLiMP",
             "bad_exclusions": {k: sorted(v) for k, v in BAD_EXCLUDE.items()},
             "good_exclusions": {k: sorted(v) for k, v in GOOD_EXCLUDE.items()},
             "bands": {}}
    for band in BANDS:
        good = [r for r in _shared_inventory(forms, lemmas, band, "good")
                if r["lemma"] not in GOOD_EXCLUDE.get(band, ())]
        bad = (reviewed_head if band == "head" else
               [r for r in _shared_inventory(forms, lemmas, band, "bad")
                if r["lemma"] not in BAD_EXCLUDE[band]])
        pairs = pair_inventory(good, bad, args.max_pairs_per_band, args.seed)
        lo, hi = BANDS[band]
        audit["bands"][band] = {"good_candidates": len(good),
                                "bad_candidates": len(bad), "verb_pairs": len(pairs),
                                "both_lemmas_in_band": sum(
                                    lo <= g["lemma_zipf"] <= hi and lo <= b["lemma_zipf"] <= hi
                                    for g, b in pairs)}
        if not pairs:
            raise ValueError(f"No verb pairs in {band}")
        for g, b in pairs:
            key = (band, g["lemma"])
            if key not in contexts:
                raise ValueError(f"No manually checked context for {key}")
            patient, agent = contexts[key]
            for paradigm in PARADIGMS:
                for template in TEMPLATES:
                    prefix = f"The {patient} {template} "
                    suffix = f" by the {agent}." if paradigm == "passive_1" else "."
                    rows.append({
                        "pair_id": f"{band}/{paradigm}/{g['lemma']}/{template.replace(' ', '_')}",
                        "band": band, "paradigm": paradigm,
                        "frame_id": template.replace(" ", "_"),
                        "patient": patient, "agent": agent if paradigm == "passive_1" else "",
                        "prefix": prefix, "suffix": suffix,
                        "good_lemma": g["lemma"], "bad_lemma": b["lemma"],
                        "good_verb": g["form"], "bad_verb": b["form"],
                        "bad_active_class": b.get("active_class", "released_inventory"),
                        "good_lemma_zipf": g["lemma_zipf"], "bad_lemma_zipf": b["lemma_zipf"],
                        "good_form_zipf": g["form_zipf"], "bad_form_zipf": b["form_zipf"],
                        "lemma_zipf_gap": g["lemma_zipf"] - b["lemma_zipf"],
                        "form_zipf_gap": g["form_zipf"] - b["form_zipf"],
                        "both_lemmas_in_band": int(
                            lo <= g["lemma_zipf"] <= hi and lo <= b["lemma_zipf"] <= hi),
                        "sentence_good": prefix + g["form"] + suffix,
                        "sentence_bad": prefix + b["form"] + suffix,
                    })
    audit["n_sentence_pairs"] = len(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit_out = Path(args.audit_out)
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "sentence_pairs": len(rows),
                      "bands": audit["bands"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freqblimp-root", required=True)
    parser.add_argument("--head-review", required=True)
    parser.add_argument("--contexts", default="data/matched_passives_v2/contexts.csv")
    parser.add_argument("--max-pairs-per-band", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/matched_passives_v2/pairs.jsonl")
    parser.add_argument("--audit-out", default="data/matched_passives_v2/audit.json")
    build(parser.parse_args())
