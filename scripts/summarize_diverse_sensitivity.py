#!/usr/bin/env python3
"""Summarize lexical-band and head-verb-class sensitivity for the diverse pilot."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from build_matched_passives import BANDS
from summarize_matched_passives import bootstrap_interval, mean, read_csv, write_csv


def run(args):
    scored = read_csv(Path(args.scores))
    by_pair = defaultdict(list)
    for row in scored:
        key = (row["paradigm"], row["band"], row["good_lemma"], row["bad_lemma"])
        by_pair[key].append(row)
    pairs = []
    for (paradigm, band, good, bad), group in by_pair.items():
        lo, hi = BANDS[band]
        gz, bz = float(group[0]["good_lemma_zipf"]), float(group[0]["bad_lemma_zipf"])
        pairs.append({"paradigm": paradigm, "band": band,
                      "good_lemma": good, "bad_lemma": bad,
                      "strict_lemma_band": lo <= gz <= hi and lo <= bz <= hi,
                      "bad_active_class": group[0]["bad_active_class"],
                      "margin": mean([float(r["whole_margin"]) for r in group]),
                      "accuracy": mean([int(r["correct"]) for r in group])})

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    band_rows = []
    for paradigm in ("passive_1", "passive_2"):
        for band in BANDS:
            for subset in ("all", "both_lemmas_in_band"):
                group = [p for p in pairs if p["paradigm"] == paradigm and p["band"] == band
                         and (subset == "all" or p["strict_lemma_band"])]
                if not group:
                    continue
                margins = [p["margin"] for p in group]
                accuracies = [p["accuracy"] for p in group]
                lo, hi = bootstrap_interval(margins, args.seed)
                band_rows.append({"paradigm": paradigm, "band": band, "subset": subset,
                                  "n_verb_pairs": len(group), "accuracy": mean(accuracies),
                                  "mean_margin": mean(margins), "margin_ci_lo": lo,
                                  "margin_ci_hi": hi})
    write_csv(out / "frequency_sensitivity.csv", band_rows)

    class_rows = []
    classes = sorted({p["bad_active_class"] for p in pairs if p["band"] == "head"})
    for paradigm in ("passive_1", "passive_2"):
        for active_class in classes:
            group = [p for p in pairs if p["paradigm"] == paradigm
                     and p["band"] == "head" and p["bad_active_class"] == active_class]
            if not group:
                continue
            margins = [p["margin"] for p in group]
            lo, hi = bootstrap_interval(margins, args.seed)
            class_rows.append({"paradigm": paradigm, "bad_active_class": active_class,
                               "n_verb_pairs": len(group),
                               "accuracy": mean([p["accuracy"] for p in group]),
                               "mean_margin": mean(margins),
                               "margin_ci_lo": lo, "margin_ci_hi": hi})
    write_csv(out / "head_active_classes.csv", class_rows)
    print(f"Wrote sensitivity summaries to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", default="results/matched_passives_v2/pythia14b_scores.csv")
    parser.add_argument("--out-dir", default="reports/matched_passives_v2")
    parser.add_argument("--seed", type=int, default=17)
    run(parser.parse_args())
