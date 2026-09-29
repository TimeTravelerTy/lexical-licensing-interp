#!/usr/bin/env python3
"""Validate LM plausibility ratings, then relate fit to Pythia passive margins.

Validation (run before trusting the ratings):
  format        digit probability mass per rating
  prompts       Spearman between prompt paraphrases
  direction     own context > other contexts; own > role reversal (passive_1)
  rarity        own-context ratings by verb band and vs participle Zipf; the own
                contexts were all written to be plausible, so a band gradient
                means the rater penalizes rare words
  human/rater   Spearman with a filled human sheet or a second rater, if given

Main analysis on the curated cross (15,876 verb x context pairs per paradigm):
Pythia margin ~ fit x verb band, with two-way (verb, context) bootstrap CIs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BANDS = ("head", "tail", "xtail")
FIT_BINS = [(1.0, 3.0), (3.0, 5.0), (5.0, 7.01)]


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def mean_rating(path):
    r = pd.read_csv(path, dtype={"item_id": str})
    wide = r.pivot_table(index="item_id", columns="prompt", values="rating")
    wide.columns = [f"rating_{c}" for c in wide.columns]
    wide["rating"] = wide.mean(axis=1)
    wide["min_mass"] = r.groupby("item_id").digit_mass.min()
    return r, wide.reset_index()


def validation(raw, rated, links, args):
    lines = ["## Validation", ""]
    mass = raw.digit_mass
    lines += [f"- **Format**: digit mass median {mass.median():.3f}; "
              f"{(mass < 0.9).mean() * 100:.1f}% of ratings below 0.9.", ""]
    prompt_cols = [c for c in rated.columns if c.startswith("rating_")]
    if len(prompt_cols) == 2:
        lines.append(f"- **Prompt agreement** (Spearman, all items): "
                     f"{spearman(rated[prompt_cols[0]], rated[prompt_cols[1]]):.3f}")
    df = links.merge(rated, on="item_id")
    cross = df[df.context_id != "reversal"]
    per_verb = []
    for (paradigm, verb_pair), g in cross.groupby(["paradigm", "verb_pair"]):
        own = g[g.own_context == 1].rating
        per_verb.append({"paradigm": paradigm, "verb_pair": verb_pair, "verb_band": g.verb_band.iloc[0],
                         "own": own.mean(), "other": g[g.own_context == 0].rating.mean(),
                         "good_verb": g.good_verb.iloc[0]})
    pv = pd.DataFrame(per_verb)
    rev = df[df.context_id == "reversal"].groupby("verb_pair").rating.mean()
    p1 = pv[pv.paradigm == "passive_1"].set_index("verb_pair")
    rev_diff = (p1.own - rev.reindex(p1.index)).dropna()
    lines += ["", "- **Direction checks**",
              f"  - own > mean of other contexts for {(pv.own > pv.other).mean() * 100:.1f}% of verb pairs "
              f"(mean own {pv.own.mean():.2f}, other {pv.other.mean():.2f})",
              f"  - own > role reversal (passive_1) for {(rev_diff > 0).mean() * 100:.1f}% of verb pairs "
              f"(mean difference {rev_diff.mean():.2f}); reversal is not always implausible", ""]
    # A rater that does not know a rare verb compresses toward the middle:
    # own ratings fall and reversal ratings rise with rarity. Lower *other*
    # ratings alone can instead reflect narrower selectional range.
    pv["reversal"] = pv.verb_pair.map(rev).where(pv.paradigm == "passive_1")
    lines += ["- **Rarity check**: mean rating by verb band (own and reversal should be flat)", "",
              "| Paradigm | Verb band | Own | Other | Reversal | n |", "|---|---|---:|---:|---:|---:|"]
    for (paradigm, band), g in pv.groupby(["paradigm", "verb_band"]):
        reversal = f"{g.reversal.mean():.2f}" if g.reversal.notna().any() else "-"
        lines.append(f"| {paradigm} | {band} | {g.own.mean():.2f} | {g.other.mean():.2f} | {reversal} | {len(g)} |")
    if args.v2_pairs:
        z = pd.read_json(args.v2_pairs, lines=True).drop_duplicates("good_verb").set_index("good_verb").good_form_zipf
        pv["good_form_zipf"] = pv.good_verb.map(z)
        lines += ["", f"- Spearman(own rating, participle Zipf): {spearman(pv.own, pv.good_form_zipf):.3f}; "
                  f"Spearman(other rating, participle Zipf): {spearman(pv.other, pv.good_form_zipf):.3f}"]
    if args.human:
        h = pd.read_csv(args.human, dtype={"item_id": str}).dropna(subset=["rating_1_to_7"])
        if len(h):
            hm = h.merge(rated, on="item_id")
            lines += ["", f"- **Human agreement**: Spearman {spearman(hm.rating_1_to_7, hm.rating):.3f} "
                      f"on {len(hm)} items"]
    if args.second_rater:
        _, other = mean_rating(args.second_rater)
        m = rated.merge(other, on="item_id", suffixes=("", "_second"))
        lines += ["", f"- **Second rater agreement**: Spearman {spearman(m.rating, m.rating_second):.3f} "
                  f"on {len(m)} items"]
    lines.append("")
    return lines, pv


def weights(rng, labels, n_boot):
    w = np.ones((n_boot + 1, len(labels)))
    for band in np.unique(labels):
        idx = np.flatnonzero(labels == band)
        w[1:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=n_boot)
    return w


def fit_models(df, n_boot, rng):
    """WLS of margin on fit x verb band; two-way bootstrap over verbs and contexts."""
    rows, bins = [], []
    for paradigm, sub in df.groupby("paradigm"):
        verbs = sub.drop_duplicates("verb_pair").sort_values("verb_pair")
        ctxs = sub.drop_duplicates("context_id").sort_values("context_id")
        vi = sub.verb_pair.map({v: i for i, v in enumerate(verbs.verb_pair)}).to_numpy()
        ci = sub.context_id.map({c: i for i, c in enumerate(ctxs.context_id)}).to_numpy()
        wv = weights(rng, verbs.verb_band.to_numpy(), n_boot)
        wc = weights(rng, ctxs.context_band.to_numpy(), n_boot)
        fit = sub.rating.to_numpy() - 4.0
        tail = (sub.verb_band == "tail").to_numpy(float)
        xtail = (sub.verb_band == "xtail").to_numpy(float)
        own = sub.own_context.to_numpy(float)
        designs = {
            "fit_x_band": (["intercept", "fit", "tail", "xtail", "fit:tail", "fit:xtail"],
                           np.column_stack([np.ones(len(sub)), fit, tail, xtail, fit * tail, fit * xtail])),
            "fit_band_own": (["intercept", "fit", "tail", "xtail", "own"],
                             np.column_stack([np.ones(len(sub)), fit, tail, xtail, own])),
        }
        for metric in ("whole_margin", "verb_margin", "correct"):
            y = sub[metric].to_numpy(float)
            for model, (names, X) in designs.items():
                draws = []
                for b in range(n_boot + 1):
                    w = wv[b, vi] * wc[b, ci]
                    keep = w > 0
                    sw = np.sqrt(w[keep])
                    beta, *_ = np.linalg.lstsq(X[keep] * sw[:, None], y[keep] * sw, rcond=None)
                    draws.append(beta)
                draws = np.array(draws)
                for j, name in enumerate(names):
                    lo, hi = np.percentile(draws[1:, j], [2.5, 97.5])
                    rows.append({"paradigm": paradigm, "metric": metric, "model": model, "term": name,
                                 "estimate": draws[0, j], "ci_low": lo, "ci_high": hi})
            # Verb-band accuracy/margin within fit bins (other contexts only).
            for lo_fit, hi_fit in FIT_BINS:
                for band in BANDS:
                    part = sub[(sub.own_context == 0) & (sub.rating >= lo_fit) & (sub.rating < hi_fit)
                               & (sub.verb_band == band)]
                    if len(part) == 0:
                        continue
                    by_verb = part.groupby("verb_pair")[metric].mean()
                    bins.append({"paradigm": paradigm, "metric": metric, "fit_bin": f"[{lo_fit:g}, {min(hi_fit, 7):g}]",
                                 "verb_band": band, "estimate": by_verb.mean(), "n_pairs": len(part),
                                 "n_verb_pairs": len(by_verb)})
    return pd.DataFrame(rows), pd.DataFrame(bins)


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw, rated = mean_rating(args.ratings)
    links = pd.read_csv(args.links, dtype={"item_id": str})
    lines = [f"# Fit ratings: {Path(args.ratings).stem}", ""]
    val_lines, per_verb = validation(raw, rated, links, args)
    lines += val_lines
    per_verb.to_csv(out_dir / "per_verb_ratings.csv", index=False)

    scores = pd.read_csv(args.scores, low_memory=False)
    cur = scores[scores.context_set == "curated"].merge(
        links[["pair_id", "item_id"]], on="pair_id").merge(rated[["item_id", "rating"]], on="item_id")
    coefs, bins = fit_models(cur, args.n_boot, np.random.default_rng(args.seed))
    coefs.to_csv(out_dir / "fit_models.csv", index=False)
    bins.to_csv(out_dir / "fit_bins.csv", index=False)
    lines += ["## Pythia margin vs fit (curated cross)", "",
              "`fit` is the rating minus 4 (per rating point); band terms are relative to Head verbs.", ""]
    for paradigm in sorted(coefs.paradigm.unique()):
        c = coefs[(coefs.paradigm == paradigm) & (coefs.model == "fit_x_band")]
        lines += [f"**{paradigm}**", "", "| Term | Whole margin | Verb margin | Accuracy |", "|---|---:|---:|---:|"]
        for term in c.term.unique():
            vals = [f"{r.estimate:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]" for _, r in
                    c[c.term == term].set_index("metric").loc[["whole_margin", "verb_margin", "correct"]].iterrows()]
            lines.append(f"| {term} | " + " | ".join(vals) + " |")
        o = coefs[(coefs.paradigm == paradigm) & (coefs.model == "fit_band_own") & (coefs.term == "own")
                  & (coefs.metric == "correct")].iloc[0]
        lines += ["", f"Own-context accuracy advantage after controlling fit: "
                  f"{o.estimate * 100:.1f} pp [{o.ci_low * 100:.1f}, {o.ci_high * 100:.1f}]", ""]
        b = bins[(bins.paradigm == paradigm) & (bins.metric == "correct")]
        lines += ["Accuracy (%) by fit bin, other contexts only:", "",
                  "| Fit bin | " + " | ".join(BANDS) + " |", "|---|" + "---:|" * len(BANDS)]
        for fit_bin in b.fit_bin.unique():
            cells = []
            for band in BANDS:
                r = b[(b.fit_bin == fit_bin) & (b.verb_band == band)]
                cells.append(f"{r.estimate.iloc[0] * 100:.1f} (n={r.n_pairs.iloc[0]})" if len(r) else "-")
            lines.append(f"| {fit_bin} | " + " | ".join(cells) + " |")
        lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((out_dir / "report.md").read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ratings", default="results/fit_ratings/gemma4_31b_it.csv")
    ap.add_argument("--links", default="data/fit_ratings/item_links.csv")
    ap.add_argument("--scores", default="results/passive_band_cross/pythia14b_scores.csv")
    ap.add_argument("--v2-pairs", default="data/matched_passives_v2/pairs.jsonl")
    ap.add_argument("--human", default="")
    ap.add_argument("--second-rater", default="")
    ap.add_argument("--out-dir", default="reports/fit_ratings")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
