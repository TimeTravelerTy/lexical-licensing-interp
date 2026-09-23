#!/usr/bin/env python3
"""Compare Pythia scores on released, crossed, and curated passive items."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from wordfreq import zipf_frequency

from summarize_matched_passives import write_csv


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def accuracy(rows):
    return fmean(int(row["correct"]) for row in rows)


def margin(rows):
    return fmean(float(row["whole_margin"]) for row in rows)


def bootstrap(values, seed, repeats=5000):
    rng = random.Random(seed)
    draws = sorted(fmean(rng.choices(values, k=len(values))) for _ in range(repeats))
    return draws[int(.025 * repeats)], draws[int(.975 * repeats)]


def run(args):
    original = read(args.original_scores)
    crossed = read(args.crossed_scores)
    curated = read(args.curated_scores)
    if (len(original), len(crossed), len(curated)) != (6000, 1339, 504):
        raise ValueError("Unexpected score counts")
    original_by_id = {row["pair_id"]: row for row in original}
    if len(original_by_id) != len(original):
        raise ValueError("Duplicate released pair ID")
    for row in crossed:
        source = original_by_id[row["pair_id"]]
        if (row["sentence_good"] != source["sentence_good"] or
                abs(float(row["good_whole_lp"]) - float(source["good_whole_lp"])) > 1e-5):
            raise ValueError(f"Crossed good sentence changed: {row['pair_id']}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for paradigm in ("passive_1", "passive_2"):
        for band in ("head", "tail", "xtail"):
            for condition, source in (("released", original),
                                      ("crossed", crossed),
                                      ("curated", curated)):
                group = [r for r in source if r["paradigm"] == paradigm and r["band"] == band]
                summary.append({"paradigm": paradigm, "band": band,
                                "condition": condition, "n_sentences": len(group),
                                "n_good_forms": len({r["good_verb"].lower() for r in group}),
                                "n_bad_forms": len({r["bad_verb"].lower() for r in group}),
                                "accuracy": accuracy(group), "mean_margin": margin(group)})
    write_csv(out / "summary.csv", summary)

    head_types = []
    for paradigm in ("passive_1", "passive_2"):
        groups = defaultdict(list)
        for row in original:
            if row["paradigm"] == paradigm and row["band"] == "head":
                groups[row["bad_verb"].lower()].append(row)
        for form, group in sorted(groups.items()):
            head_types.append({"paradigm": paradigm, "bad_form": form,
                               "n_sentences": len(group), "accuracy": accuracy(group),
                               "mean_margin": margin(group)})
    write_csv(out / "head_bad_types.csv", head_types)

    frequency_match = []
    for paradigm in ("passive_1", "passive_2"):
        for band in ("head", "tail", "xtail"):
            group = [row for row in original if row["paradigm"] == paradigm
                     and row["band"] == band]
            tight = [row for row in group if abs(
                zipf_frequency(row["good_verb"].lower(), "en") -
                zipf_frequency(row["bad_verb"].lower(), "en")) <= .35]
            frequency_match.append({"paradigm": paradigm, "band": band,
                                    "n_released": len(group),
                                    "n_form_gap_le_035": len(tight),
                                    "all_accuracy": accuracy(group),
                                    "form_gap_le_035_accuracy": accuracy(tight),
                                    "all_mean_margin": margin(group),
                                    "form_gap_le_035_mean_margin": margin(tight)})
    write_csv(out / "form_gap_sensitivity.csv", frequency_match)

    context = []
    for paradigm in ("passive_1", "passive_2"):
        for band in ("head", "tail", "xtail"):
            c = [r for r in crossed if r["paradigm"] == paradigm and r["band"] == band]
            o = [original_by_id[r["pair_id"]] for r in c]
            v = [r for r in curated if r["paradigm"] == paradigm and r["band"] == band]
            by_c, by_v = defaultdict(list), defaultdict(list)
            for row in c:
                by_c[row["good_verb"].lower()].append(row)
            for row in v:
                by_v[row["good_verb"].lower()].append(row)
            if set(by_c) != set(by_v):
                raise ValueError(f"Different good-verb inventory: {paradigm}/{band}")
            changes = {}
            for key, fun in (("accuracy", accuracy), ("margin", margin)):
                pair_diffs = [fun(by_v[verb]) - fun(by_c[verb]) for verb in sorted(by_v)]
                lo, hi = bootstrap(pair_diffs, args.seed)
                changes[key] = (fmean(pair_diffs), lo, hi)
            context.append({
                "paradigm": paradigm, "band": band,
                "n_crossed_sentences": len(c), "n_good_verb_types": len(by_c),
                "source_same_rows_accuracy": accuracy(o),
                "crossed_accuracy": accuracy(c),
                "curated_accuracy": accuracy(v),
                "source_same_rows_margin": margin(o),
                "crossed_margin": margin(c),
                "curated_margin": margin(v),
                "pair_equal_curated_minus_crossed_accuracy": changes["accuracy"][0],
                "accuracy_ci_lo": changes["accuracy"][1],
                "accuracy_ci_hi": changes["accuracy"][2],
                "pair_equal_curated_minus_crossed_margin": changes["margin"][0],
                "margin_ci_lo": changes["margin"][1],
                "margin_ci_hi": changes["margin"][2],
            })
    write_csv(out / "context_effect.csv", context)
    print(f"Wrote discrepancy summaries to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-scores", default="results/freqblimp_original_passives/pythia14b_scores.csv")
    parser.add_argument("--crossed-scores", default="results/freqblimp_original_contexts_v2_verbs/pythia14b_scores.csv")
    parser.add_argument("--curated-scores", default="results/matched_passives_v2/pythia14b_scores.csv")
    parser.add_argument("--out-dir", default="reports/passive_discrepancy_20260923")
    parser.add_argument("--seed", type=int, default=17)
    run(parser.parse_args())
