#!/usr/bin/env python3
"""Summarize passive accuracy and margins at the independent verb-pair level."""

from __future__ import annotations

import argparse
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return statistics.fmean(values)


def bootstrap_interval(values, seed, repeats=5000):
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(repeats))
    return draws[int(0.025 * repeats)], draws[int(0.975 * repeats)]


def bootstrap_difference(head, xtail, seed, repeats=5000):
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(head, k=len(head))) - mean(rng.choices(xtail, k=len(xtail)))
                   for _ in range(repeats))
    return draws[int(0.025 * repeats)], draws[int(0.975 * repeats)]


def markdown_report(summary, per_pair, comparison, leave_one_out):
    lines = ["# Pythia 1.4B matched-context passive pilot", "",
             "Positive margins favor the passivizable verb. The lexical unit is one verb pair; "
             "eight shared contexts are averaged within each pair. Intervals bootstrap verb pairs.", "",
             "## Regime summaries", "",
             "| Paradigm | Band | Verb pairs | Accuracy | Mean log-probability margin | 95% interval for margin |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for x in summary:
        lines.append(f"| {x['paradigm']} | {x['band']} | {x['n_verb_pairs']} | "
                     f"{x['accuracy']:.3f} | {x['mean_margin']:.3f} | "
                     f"[{x['margin_ci_lo']:.3f}, {x['margin_ci_hi']:.3f}] |")
    lines += ["", "## Head minus xtail sensitivity", "",
              "| Paradigm | Metric | Difference | 95% interval | Leave-one-out range | Sign changes |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for x in comparison:
        subset = [y for y in leave_one_out if y["paradigm"] == x["paradigm"] and y["metric"] == x["metric"]]
        values = [y["after_difference"] for y in subset]
        flips = sum(y["sign_change"] for y in subset)
        lines.append(f"| {x['paradigm']} | {x['metric']} | {x['difference']:.3f} | "
                     f"[{x['ci_lo']:.3f}, {x['ci_hi']:.3f}] | "
                     f"[{min(values):.3f}, {max(values):.3f}] | {flips}/{len(subset)} |")
    lines += ["", "## Individual verb pairs", "",
              "Each row is the mean across the eight contexts. See `per_verb_pair.csv` for "
              "participle frequencies, token counts, and verb/suffix contributions.", ""]
    for paradigm in ("passive_1", "passive_2"):
        lines += [f"### {paradigm}", "",
                  "| Band | Passivizable / intransitive | Accuracy | Mean margin |",
                  "| --- | --- | ---: | ---: |"]
        for x in per_pair:
            if x["paradigm"] == paradigm:
                lines.append(f"| {x['band']} | {x['good_lemma']} / {x['bad_lemma']} | "
                             f"{x['accuracy']:.3f} | {x['mean_margin']:.3f} |")
        lines.append("")
    lines += ["## Interpretation notes", "",
              "The selected verbs are unique within each paradigm and reused across the two "
              "paradigms by design. Good and bad verbs are matched within 0.25 lemma Zipf "
              "and 0.35 participle Zipf. Shared human-patient frames are broadly plausible "
              "for the selected good verbs, though individual meanings may still make a "
              "sentence unusual, especially among xtail verbs. This is a small curated gate; "
              "a flat or noisy effect is inconclusive.", ""]
    return "\n".join(lines)


def run(args):
    rows = read_csv(Path(args.scores))
    if not rows:
        raise SystemExit("No scores")
    by_pair = defaultdict(list)
    for row in rows:
        # All frames of one verb pairing are a single lexical observation.
        key = (row["band"], row["paradigm"], row["good_lemma"], row["bad_lemma"])
        by_pair[key].append(row)
    per_pair = []
    for (band, paradigm, good, bad), group in sorted(by_pair.items()):
        per_pair.append({
            "band": band, "paradigm": paradigm, "good_lemma": good, "bad_lemma": bad,
            "n_frames": len(group),
            "good_lemma_zipf": float(group[0]["good_lemma_zipf"]),
            "bad_lemma_zipf": float(group[0]["bad_lemma_zipf"]),
            "mean_lemma_zipf": mean([float(group[0]["good_lemma_zipf"]), float(group[0]["bad_lemma_zipf"])]),
            "mean_form_zipf": mean([float(group[0]["good_form_zipf"]), float(group[0]["bad_form_zipf"])]),
            "mean_margin": mean([float(x["whole_margin"]) for x in group]),
            "accuracy": mean([int(x["correct"]) for x in group]),
            "verb_margin": mean([float(x["verb_margin"]) for x in group]),
            "suffix_margin": mean([float(x["suffix_margin"]) for x in group]),
            "by_margin": mean([float(x["by_margin"]) for x in group]),
            "good_verb_tokens": int(group[0]["good_verb_tokens"]),
            "bad_verb_tokens": int(group[0]["bad_verb_tokens"]),
        })
    outdir = Path(args.out_dir)
    write_csv(outdir / "per_verb_pair.csv", per_pair)
    summary = []
    for paradigm in ("passive_1", "passive_2"):
        for band in ("head", "tail", "xtail"):
            group = [x for x in per_pair if x["band"] == band and x["paradigm"] == paradigm]
            if not group:
                continue
            margins = [x["mean_margin"] for x in group]
            accuracies = [x["accuracy"] for x in group]
            margin_ci = bootstrap_interval(margins, args.seed)
            accuracy_ci = bootstrap_interval(accuracies, args.seed + 1)
            summary.append({
                "paradigm": paradigm, "band": band,
                "n_verb_pairs": len(group), "n_sentence_pairs": sum(x["n_frames"] for x in group),
                "accuracy": mean(accuracies), "accuracy_ci_lo": accuracy_ci[0], "accuracy_ci_hi": accuracy_ci[1],
                "mean_margin": mean(margins), "margin_ci_lo": margin_ci[0], "margin_ci_hi": margin_ci[1],
                "mean_verb_margin": mean([x["verb_margin"] for x in group]),
                "mean_suffix_margin": mean([x["suffix_margin"] for x in group]),
                "mean_by_margin": mean([x["by_margin"] for x in group]),
                "mean_lemma_zipf": mean([x["mean_lemma_zipf"] for x in group]),
                "mean_form_zipf": mean([x["mean_form_zipf"] for x in group]),
                "mean_good_verb_tokens": mean([x["good_verb_tokens"] for x in group]),
                "mean_bad_verb_tokens": mean([x["bad_verb_tokens"] for x in group]),
            })
    write_csv(outdir / "summary.csv", summary)
    comparison = []
    leave_one_out = []
    for paradigm in ("passive_1", "passive_2"):
        head = [x for x in per_pair if x["band"] == "head" and x["paradigm"] == paradigm]
        xtail = [x for x in per_pair if x["band"] == "xtail" and x["paradigm"] == paradigm]
        for metric in ("mean_margin", "accuracy"):
            head_values = [x[metric] for x in head]
            xtail_values = [x[metric] for x in xtail]
            baseline = mean(head_values) - mean(xtail_values)
            lo, hi = bootstrap_difference(head_values, xtail_values, args.seed)
            comparison.append({"paradigm": paradigm, "metric": metric,
                               "difference": baseline, "ci_lo": lo, "ci_hi": hi})
            for removed_band, source in (("head", head), ("xtail", xtail)):
                for item in source:
                    kept = [x[metric] for x in source if x is not item]
                    after = (mean(kept) - mean(xtail_values) if removed_band == "head"
                             else mean(head_values) - mean(kept))
                    leave_one_out.append({
                        "paradigm": paradigm, "metric": metric,
                        "removed_band": removed_band,
                        "removed_good_lemma": item["good_lemma"],
                        "removed_bad_lemma": item["bad_lemma"],
                        "baseline_difference": baseline,
                        "after_difference": after,
                        "sign_change": int((baseline > 0) != (after > 0)),
                    })
    write_csv(outdir / "head_xtail_comparison.csv", comparison)
    write_csv(outdir / "leave_one_out.csv", leave_one_out)
    by_frame = defaultdict(list)
    for row in rows:
        by_frame[(row["paradigm"], row["band"], row["frame_id"])].append(row)
    frames = []
    for (paradigm, band, frame_id), group in sorted(by_frame.items()):
        frames.append({"paradigm": paradigm, "band": band, "frame_id": frame_id,
                       "n_verb_pairs": len(group), "accuracy": mean([int(x["correct"]) for x in group]),
                       "mean_margin": mean([float(x["whole_margin"]) for x in group])})
    write_csv(outdir / "by_frame.csv", frames)
    (outdir / "report.md").write_text(markdown_report(summary, per_pair, comparison, leave_one_out), encoding="utf-8")
    print(f"Wrote summaries and report to {outdir}")
    for item in summary:
        print(f"{item['paradigm']} {item['band']}: n={item['n_verb_pairs']}, "
              f"accuracy={item['accuracy']:.3f}, margin={item['mean_margin']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/matched_passives/pythia14b_scores.csv")
    ap.add_argument("--out-dir", default="reports/matched_passives")
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
