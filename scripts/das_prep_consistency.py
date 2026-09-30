#!/usr/bin/env python3
"""Task 0 consistency checks before DAS on passive argument structure.

1. Own-context advantage before and after the good-fit control, recomputed
   from one design family and one two-way bootstrap (same seed and weights as
   `analyze_bad_fit.py`), plus the same with the patient-side bad fit.
2. Per-verb-pair *by* margin on released `passive_1` contexts with a
   bootstrap CI over contexts; "reliable" means the CI excludes zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_bad_fit import fit
from analyze_fit_ratings import mean_rating, weights

BANDS = ("head", "tail", "xtail")


def curated_with_fit(args):
    _, good = mean_rating(args.good_ratings)
    _, badr = mean_rating(args.ratings)
    good_links = pd.read_csv(args.good_links, dtype={"item_id": str})[["pair_id", "item_id"]]
    links = pd.read_csv(args.links, dtype={"item_id": str})
    bad = links.merge(badr[["item_id", "rating"]], on="item_id") \
        .pivot_table(index="pair_id", columns="role", values="rating")
    bad.columns = [f"bad_{c}" for c in bad.columns]
    scores = pd.read_csv(args.scores, low_memory=False)
    cur = scores[scores.context_set == "curated"].merge(good_links, on="pair_id") \
        .merge(good[["item_id", "rating"]].rename(columns={"rating": "good_fit"}), on="item_id") \
        .merge(bad, left_on="pair_id", right_index=True)
    return scores, cur


def own_models(cur, n_boot, seed):
    rows = []
    rng = np.random.default_rng(seed)
    for paradigm, sub in cur.groupby("paradigm"):
        verbs = sub.drop_duplicates("verb_pair").sort_values("verb_pair")
        ctxs = sub.drop_duplicates("context_id").sort_values("context_id")
        vi = sub.verb_pair.map({v: i for i, v in enumerate(verbs.verb_pair)}).to_numpy()
        ci = sub.context_id.map({c: i for i, c in enumerate(ctxs.context_id)}).to_numpy()
        wv = weights(rng, verbs.verb_band.to_numpy(), n_boot)
        wc = weights(rng, ctxs.context_band.to_numpy(), n_boot)
        one = np.ones(len(sub))
        t, x = (sub.verb_band == "tail").to_numpy(float), (sub.verb_band == "xtail").to_numpy(float)
        good, patient = sub.good_fit.to_numpy() - 4, sub.bad_patient.to_numpy() - 4
        own = sub.own_context.to_numpy(float)
        designs = {
            "own": np.column_stack([one, t, x, own]),
            "good+own": np.column_stack([one, good, t, x, own]),
            "good+bad_patient+own": np.column_stack([one, good, patient, t, x, own]),
        }
        for metric in ("correct", "whole_margin", "verb_margin"):
            y = sub[metric].to_numpy(float) * (100 if metric == "correct" else 1)
            for model, X in designs.items():
                d = fit(sub, y, X, vi, ci, wv, wc)[:, -1]
                rows.append({"paradigm": paradigm, "metric": metric, "model": model,
                             "estimate": d[0], "ci_low": np.percentile(d[1:], 2.5),
                             "ci_high": np.percentile(d[1:], 97.5)})
    return pd.DataFrame(rows)


def by_reliability(scores, n_boot, seed):
    rng = np.random.default_rng(seed)
    rel = scores[(scores.paradigm == "passive_1") & (scores.context_set == "released")]
    rows = []
    for (band, pair), g in rel.groupby(["verb_band", "verb_pair"]):
        x = g.by_margin.to_numpy()
        draws = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        rows.append({"verb_pair": pair, "verb_band": band, "n_contexts": len(x),
                     "by_margin": x.mean(), "ci_low": lo, "ci_high": hi,
                     "by_class": "reliable_pos" if lo > 0 else ("reliable_neg" if hi < 0 else
                                                                 ("ns_neg" if x.mean() < 0 else "ns_pos"))})
    return pd.DataFrame(rows)


def run(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores, cur = curated_with_fit(args)
    own = own_models(cur, args.n_boot, args.seed)
    own.to_csv(out / "task0_own_context.csv", index=False)
    old = pd.read_csv(args.old_models)
    old = old[(old.term == "own") & old.model.isin(["own", "good+own"])]
    rel = by_reliability(scores, args.n_boot, args.seed)
    rel.to_csv(out / "task0_by_reliability.csv", index=False)

    fmt = lambda r: f"{r.estimate:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]"
    lines = ["# Task 0: consistency checks", "",
             "## Own-context advantage (own - other), pooled over bands", "",
             "Curated cross, WLS with shared good-fit slope and band dummies; two-way cluster bootstrap "
             f"(verb pairs within band x contexts within band), {args.n_boot} draws, seed {args.seed}. "
             "`committed` = `reports/fit_ratings/bad_fit_models.csv`.", "",
             "| Paradigm | Metric | Model | Recomputed | Committed |", "|---|---|---|---:|---:|"]
    for r in own.itertuples():
        o = old[(old.paradigm == r.paradigm) & (old.metric == r.metric) & (old.model == r.model)]
        lines.append(f"| {r.paradigm} | {r.metric} | {r.model} | {fmt(r)} | "
                     f"{fmt(o.iloc[0]) if len(o) else '-'} |")
    counts = rel.groupby(["verb_band", "by_class"]).size().unstack(fill_value=0) \
        .reindex(columns=["reliable_pos", "ns_pos", "ns_neg", "reliable_neg"], fill_value=0)
    lines += ["", "## *by* margin reliability per verb pair (`passive_1`, released contexts)", "",
              "Mean over 300 released contexts per pair; 95% bootstrap CI over contexts.", "",
              "| Verb band | n | Reliable + | n.s. + | n.s. - | Reliable - | Mean < 0 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for b in BANDS:
        c = counts.loc[b]
        lines.append(f"| {b} | {c.sum()} | {c.reliable_pos} | {c.ns_pos} | {c.ns_neg} | "
                     f"{c.reliable_neg} | {c.ns_neg + c.reliable_neg} |")
    for b in ("head", "xtail"):
        neg = rel[(rel.verb_band == b) & (rel.by_margin < 0)].sort_values("by_margin")
        lines += ["", f"Negative {b} pairs: " + ", ".join(
            f"{r.verb_pair.split('/', 1)[1]} ({r.by_margin:.2f})" for r in neg.itertuples())]
    (out / "task0_consistency.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ratings", default="results/fit_ratings/bad_gemma4_31b_it.csv")
    ap.add_argument("--links", default="data/fit_ratings/bad_item_links.csv")
    ap.add_argument("--good-ratings", default="results/fit_ratings/gemma4_31b_it.csv")
    ap.add_argument("--good-links", default="data/fit_ratings/item_links.csv")
    ap.add_argument("--scores", default="results/passive_band_cross/pythia14b_scores.csv")
    ap.add_argument("--old-models", default="reports/fit_ratings/bad_fit_models.csv")
    ap.add_argument("--out-dir", default="reports/passive_das_prep")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
