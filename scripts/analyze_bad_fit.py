#!/usr/bin/env python3
"""Validate bad-side (intransitive active) fit ratings and add them to margin models.

Each curated-cross row has a good-side fit (rating of the good passive) and
bad-side fits from active counterparts of the bad verb: `bad_patient`
("The hat salivated.") and, for passive_1, `bad_agent` ("The gentleman
salivated."). Models are WLS with a two-way (verb pair x context) bootstrap:

  unmatched       y ~ band
  good            y ~ good_fit + band
  good+bad        y ~ good_fit + bad fits + band

Expected signs: bad_patient < 0 on the participle margin (the participle is
predicted from "The PATIENT was"); bad_agent can only act after the verb.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_fit_ratings import mean_rating, spearman, weights

BANDS = ("head", "tail", "xtail")
METRICS = ("correct", "whole_margin", "verb_margin", "by_margin", "suffix_margin")


def validation(raw, rated, links, args):
    lines = ["## Validation", "",
             f"- **Format**: digit mass median {raw.digit_mass.median():.3f}; "
             f"{(raw.digit_mass < 0.9).mean() * 100:.1f}% below 0.9."]
    cols = [c for c in rated.columns if c.startswith("rating_")]
    if len(cols) == 2:
        lines.append(f"- **Prompt agreement**: Spearman {spearman(rated[cols[0]], rated[cols[1]]):.3f}")
    df = links.drop_duplicates(["item_id", "verb_band"]).merge(rated, on="item_id")
    by_band = df.groupby("verb_band").rating.mean()
    lines.append("- **Mean bad-side rating by bad-verb band**: " +
                 ", ".join(f"{b} {by_band[b]:.2f}" for b in BANDS))
    if args.second_rater:
        _, other = mean_rating(args.second_rater)
        m = rated.merge(other, on="item_id", suffixes=("", "_second"))
        lines.append(f"- **Second rater agreement**: Spearman {spearman(m.rating, m.rating_second):.3f}")
    if args.human:
        h = pd.read_csv(args.human, dtype={"item_id": str}).dropna(subset=["rating_1_to_7"])
        if len(h):
            hm = h.merge(rated, on="item_id")
            lines.append(f"- **Human agreement**: Spearman {spearman(hm.rating_1_to_7, hm.rating):.3f} "
                         f"on {len(hm)} items")
    items = rated.merge(pd.read_json(args.items, lines=True)[["item_id", "sentence"]], on="item_id")
    items = items.sort_values("rating")
    lines += ["", "Lowest rated: " + "; ".join(items.head(8).sentence),
              "", "Highest rated: " + "; ".join(items.tail(8).sentence), ""]
    return lines


def fit(sub, y, X, vi, ci, wv, wc):
    draws = []
    for b in range(wv.shape[0]):
        w = wv[b, vi] * wc[b, ci]
        k = w > 0
        sw = np.sqrt(w[k])
        beta, *_ = np.linalg.lstsq(X[k] * sw[:, None], y[k] * sw, rcond=None)
        draws.append(beta)
    return np.array(draws)


def models(df, n_boot, rng):
    rows = []
    for paradigm, sub in df.groupby("paradigm"):
        verbs = sub.drop_duplicates("verb_pair").sort_values("verb_pair")
        ctxs = sub.drop_duplicates("context_id").sort_values("context_id")
        vi = sub.verb_pair.map({v: i for i, v in enumerate(verbs.verb_pair)}).to_numpy()
        ci = sub.context_id.map({c: i for i, c in enumerate(ctxs.context_id)}).to_numpy()
        wv, wc = weights(rng, verbs.verb_band.to_numpy(), n_boot), weights(rng, ctxs.context_band.to_numpy(), n_boot)
        one = np.ones(len(sub))
        t, x = (sub.verb_band == "tail").to_numpy(float), (sub.verb_band == "xtail").to_numpy(float)
        good = sub.good_fit.to_numpy() - 4
        patient = sub.bad_patient.to_numpy() - 4
        own = sub.own_context.to_numpy(float)
        bad_terms = ["bad_patient"] + (["bad_agent"] if paradigm == "passive_1" else [])
        bad = [sub[c].to_numpy() - 4 for c in bad_terms]
        designs = {
            "unmatched": (["intercept", "tail", "xtail"], np.column_stack([one, t, x])),
            "good": (["intercept", "good_fit", "tail", "xtail"], np.column_stack([one, good, t, x])),
            # Primary bad-side control: the patient precedes the participle; the
            # agent term behaves like a verb-property confound (see docs).
            "good+bad_patient": (["intercept", "good_fit", "bad_patient", "tail", "xtail"],
                                 np.column_stack([one, good, patient, t, x])),
            "good+bad": (["intercept", "good_fit"] + bad_terms + ["tail", "xtail"],
                         np.column_stack([one, good] + bad + [t, x])),
            # Own-context advantage before and after good-side fit.
            "own": (["intercept", "tail", "xtail", "own"], np.column_stack([one, t, x, own])),
            "good+own": (["intercept", "good_fit", "tail", "xtail", "own"],
                         np.column_stack([one, good, t, x, own])),
        }
        for metric in METRICS:
            if metric == "by_margin" and paradigm == "passive_2":
                continue
            y = sub[metric].to_numpy(float) * (100 if metric == "correct" else 1)
            for model, (names, X) in designs.items():
                d = fit(sub, y, X, vi, ci, wv, wc)
                for j, name in enumerate(names):
                    rows.append({"paradigm": paradigm, "metric": metric, "model": model, "term": name,
                                 "estimate": d[0, j], "ci_low": np.percentile(d[1:, j], 2.5),
                                 "ci_high": np.percentile(d[1:, j], 97.5)})
    return pd.DataFrame(rows)


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw, rated = mean_rating(args.ratings)
    links = pd.read_csv(args.links, dtype={"item_id": str})
    lines = [f"# Bad-side fit: {Path(args.ratings).stem}", ""] + validation(raw, rated, links, args)

    bad = links.merge(rated[["item_id", "rating"]], on="item_id") \
        .pivot_table(index="pair_id", columns="role", values="rating")
    bad.columns = [f"bad_{c}" for c in bad.columns]
    _, good = mean_rating(args.good_ratings)
    good_links = pd.read_csv(args.good_links, dtype={"item_id": str})[["pair_id", "item_id"]]
    scores = pd.read_csv(args.scores, low_memory=False)
    cur = scores[scores.context_set == "curated"].merge(good_links, on="pair_id") \
        .merge(good[["item_id", "rating"]].rename(columns={"rating": "good_fit"}), on="item_id") \
        .merge(bad, left_on="pair_id", right_index=True)
    corr = cur.groupby("paradigm")[["good_fit", "bad_patient"]].corr().xs("good_fit", level=1)["bad_patient"]
    lines += ["- Correlation of good fit with bad-patient fit (Pearson): " +
              ", ".join(f"{p} {v:.2f}" for p, v in corr.items()), ""]
    coefs = models(cur, args.n_boot, np.random.default_rng(args.seed))
    coefs.to_csv(out_dir / "bad_fit_models.csv", index=False)

    lines += ["## Margin models", "", "Fit terms are per rating point (rating - 4). Accuracy in pp.", ""]
    for paradigm in sorted(coefs.paradigm.unique()):
        c = coefs[coefs.paradigm == paradigm]
        metrics = [m for m in METRICS if m in c.metric.unique()]
        lines += [f"**{paradigm}**, full model (good+bad)", "",
                  "| Term | " + " | ".join(metrics) + " |", "|---|" + "---:|" * len(metrics)]
        full = c[c.model == "good+bad"]
        for term in full.term.unique():
            if term == "intercept":
                continue
            vals = []
            for m in metrics:
                r = full[(full.term == term) & (full.metric == m)].iloc[0]
                vals.append(f"{r.estimate:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]")
            lines.append(f"| {term} | " + " | ".join(vals) + " |")
        lines += ["", f"XTail - Head across models ({paradigm}):", "",
                  "| Model | " + " | ".join(metrics) + " |", "|---|" + "---:|" * len(metrics)]
        for model in ("unmatched", "good", "good+bad_patient", "good+bad"):
            vals = []
            for m in metrics:
                r = c[(c.model == model) & (c.term == "xtail") & (c.metric == m)].iloc[0]
                vals.append(f"{r.estimate:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]")
            lines.append(f"| {model} | " + " | ".join(vals) + " |")
        lines += ["", f"Own-context advantage ({paradigm}):", "",
                  "| Model | " + " | ".join(metrics) + " |", "|---|" + "---:|" * len(metrics)]
        for model in ("own", "good+own"):
            vals = []
            for m in metrics:
                r = c[(c.model == model) & (c.term == "own") & (c.metric == m)].iloc[0]
                vals.append(f"{r.estimate:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]")
            lines.append(f"| {model} | " + " | ".join(vals) + " |")
        lines.append("")
    (out_dir / "bad_fit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ratings", default="results/fit_ratings/bad_gemma4_31b_it.csv")
    ap.add_argument("--items", default="data/fit_ratings/bad_items.jsonl")
    ap.add_argument("--links", default="data/fit_ratings/bad_item_links.csv")
    ap.add_argument("--good-ratings", default="results/fit_ratings/gemma4_31b_it.csv")
    ap.add_argument("--good-links", default="data/fit_ratings/item_links.csv")
    ap.add_argument("--scores", default="results/passive_band_cross/pythia14b_scores.csv")
    ap.add_argument("--second-rater", default="")
    ap.add_argument("--human", default="")
    ap.add_argument("--out-dir", default="reports/fit_ratings")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
