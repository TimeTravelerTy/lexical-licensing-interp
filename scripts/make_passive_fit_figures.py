#!/usr/bin/env python3
"""Report figures for the band cross and fit-rating analyses.

1. band_cross_accuracy.png: accuracy by verb band x context band (released).
2. fit_accuracy.png: accuracy vs fit rating by verb band, and fit distributions.
3. xtail_effect_matched.png: XTail - Head with and without fit adjustment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BANDS = ("head", "tail", "xtail")
LABEL = {"head": "Head", "tail": "Tail", "xtail": "XTail"}
COLOR = {"head": "#2a78d6", "tail": "#eb6834", "xtail": "#1baf7a"}
MARKER = {"head": "o", "tail": "s", "xtail": "^"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
PARADIGM = {"passive_1": "passive_1  (was V-ed by NP)", "passive_2": "passive_2  (was V-ed.)"}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def weights(rng, labels, n_boot):
    w = np.ones((n_boot + 1, len(labels)))
    for band in np.unique(labels):
        idx = np.flatnonzero(labels == band)
        w[1:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=n_boot)
    return w


def band_cross_figure(cells, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, paradigm in zip(axes, PARADIGM):
        c = cells[(cells.paradigm == paradigm) & (cells.metric == "correct")]
        for j, cb in enumerate(BANDS):
            r = c[c.context_band == cb].set_index("verb_band").loc[list(BANDS)]
            x = np.arange(3) + (j - 1) * 0.18
            ax.errorbar(x, r.estimate * 100, yerr=[(r.estimate - r.ci_low) * 100, (r.ci_high - r.estimate) * 100],
                        fmt=MARKER[cb], color=COLOR[cb], ms=7, lw=2, capsize=0, mec="white", mew=1.5,
                        label=f"{LABEL[cb]} nouns")
        ax.set_xticks(range(3), [f"{LABEL[b]} verbs" for b in BANDS])
        ax.set_title(PARADIGM[paradigm], fontsize=10, color=INK, loc="left")
        style(ax)
    axes[0].set_ylabel("Accuracy (%)", color=INK)
    axes[0].set_ylim(55, 100)
    axes[1].legend(title="Context band", frameon=False, fontsize=8.5, title_fontsize=8.5, loc="lower left")
    fig.suptitle("Accuracy falls with verb band; noun (context) band has no effect",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, -0.02, "Pythia-1.4B, released contexts crossed with all verb pairs. "
             "95% two-way cluster bootstrap (verb pairs x contexts).", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fit_figure(cur, out, n_boot, rng):
    edges = [1, 2, 3, 4, 5, 6, 7.01]
    cur = cur.assign(fit_bin=pd.cut(cur.rating, edges, right=False, labels=range(6)))
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for col, paradigm in enumerate(PARADIGM):
        sub = cur[(cur.paradigm == paradigm) & (cur.own_context == 0)]
        ax = axes[0, col]
        for band in BANDS:
            b = sub[sub.verb_band == band]
            # Per verb pair x bin accuracy, then cluster-bootstrap verb pairs.
            per = b.groupby(["fit_bin", "verb_pair"], observed=True).correct.agg(["sum", "count"]).reset_index()
            xs, ys, lo, hi = [], [], [], []
            for k in range(6):
                p = per[per.fit_bin == k]
                if p["count"].sum() < 30:
                    continue
                s, n = p["sum"].to_numpy(float), p["count"].to_numpy(float)
                draws = [(s[i].sum() / n[i].sum()) for i in
                         (rng.integers(0, len(s), len(s)) for _ in range(n_boot))]
                xs.append(k + 1.5)
                ys.append(s.sum() / n.sum() * 100)
                lo.append(np.percentile(draws, 2.5) * 100)
                hi.append(np.percentile(draws, 97.5) * 100)
            ys, lo, hi = map(np.array, (ys, lo, hi))
            ax.fill_between(xs, lo, hi, color=COLOR[band], alpha=0.12, lw=0)
            ax.plot(xs, ys, color=COLOR[band], lw=2, marker=MARKER[band], ms=7, mec="white", mew=1.5,
                    label=f"{LABEL[band]} verbs")
        ax.set_title(PARADIGM[paradigm], fontsize=10, color=INK, loc="left")
        ax.set_ylim(50, 100)
        style(ax)
        ax2 = axes[1, col]
        bins = np.linspace(1, 7, 25)
        for band in BANDS:
            ax2.hist(sub[sub.verb_band == band].rating, bins=bins, density=True, histtype="step",
                     color=COLOR[band], lw=2)
            ax2.axvline(sub[sub.verb_band == band].rating.mean(), color=COLOR[band], lw=1.2, ls="--")
        ax2.set_xlabel("Plausibility rating (Gemma-4-31B-it, 1-7)", color=INK)
        style(ax2)
        ax2.set_yticks([])
    axes[0, 0].set_ylabel("Pythia accuracy (%)", color=INK)
    axes[1, 0].set_ylabel("Share of items", color=INK)
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle("Fit raises accuracy in every band; rare verbs sit lower at matched fit and get lower fit",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, -0.02, "Curated cross, other verbs' contexts only. Bins of one rating point "
             "(bins with <30 items omitted); 95% bootstrap over verb pairs. Dashed lines: band mean rating.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def adjusted_effects(cur, n_boot, rng):
    rows = []
    for paradigm, sub in cur.groupby("paradigm"):
        verbs = sub.drop_duplicates("verb_pair").sort_values("verb_pair")
        ctxs = sub.drop_duplicates("context_id").sort_values("context_id")
        vi = sub.verb_pair.map({v: i for i, v in enumerate(verbs.verb_pair)}).to_numpy()
        ci = sub.context_id.map({c: i for i, c in enumerate(ctxs.context_id)}).to_numpy()
        wv, wc = weights(rng, verbs.verb_band.to_numpy(), n_boot), weights(rng, ctxs.context_band.to_numpy(), n_boot)
        fit = sub.rating.to_numpy() - 4
        t, x = (sub.verb_band == "tail").to_numpy(float), (sub.verb_band == "xtail").to_numpy(float)
        one = np.ones(len(sub))
        designs = {"Unmatched": np.column_stack([one, t, x]),
                   "Fit-matched": np.column_stack([one, fit, t, x, fit * t, fit * x])}
        xcol = {"Unmatched": 2, "Fit-matched": 3}
        for metric in ("correct", "whole_margin", "verb_margin"):
            y = sub[metric].to_numpy(float) * (100 if metric == "correct" else 1)
            for name, X in designs.items():
                draws = []
                for b in range(n_boot + 1):
                    w = wv[b, vi] * wc[b, ci]
                    k = w > 0
                    sw = np.sqrt(w[k])
                    beta, *_ = np.linalg.lstsq(X[k] * sw[:, None], y[k] * sw, rcond=None)
                    draws.append(beta[xcol[name]])
                d = np.array(draws)
                rows.append({"paradigm": paradigm, "metric": metric, "model": name, "estimate": d[0],
                             "ci_low": np.percentile(d[1:], 2.5), "ci_high": np.percentile(d[1:], 97.5)})
    return pd.DataFrame(rows)


def effect_figure(eff, out):
    metrics = [("correct", "Accuracy (pp)"), ("whole_margin", "Full-sentence margin"),
               ("verb_margin", "Participle-token margin")]
    shade = {"Unmatched": "#9ec5f4", "Fit-matched": "#1c5cab"}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, (metric, title) in zip(axes, metrics):
        for j, model in enumerate(("Unmatched", "Fit-matched")):
            r = eff[(eff.metric == metric) & (eff.model == model)].set_index("paradigm").loc[list(PARADIGM)]
            x = np.arange(2) + (j - 0.5) * 0.28
            ax.errorbar(x, r.estimate, yerr=[r.estimate - r.ci_low, r.ci_high - r.estimate], fmt="o",
                        color=shade[model], ms=8, lw=2.2, capsize=0, mec="white", mew=1.5, label=model)
            for xi, v in zip(x, r.estimate):
                ax.annotate(f"{v:.1f}" if metric == "correct" else f"{v:.2f}", (xi, v), xytext=(7, 0),
                            textcoords="offset points", va="center", fontsize=8, color=INK)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xticks(range(2), ["passive_1", "passive_2"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        style(ax)
    axes[0].set_ylabel("XTail - Head", color=INK)
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle("Fit explains about half of the rare-verb deficit; a participle-token deficit remains",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, -0.03, "Curated cross (126 verb pairs x 126 contexts per paradigm). Fit-matched = XTail "
             "coefficient with fit x band terms, at rating 4. 95% two-way cluster bootstrap.",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    band_cross_figure(pd.read_csv(args.cells), out / "band_cross_accuracy.png")
    scores = pd.read_csv(args.scores, low_memory=False)
    links = pd.read_csv(args.links)[["pair_id", "item_id"]]
    rating = pd.read_csv(args.ratings).groupby("item_id").rating.mean().rename("rating")
    cur = scores[scores.context_set == "curated"].merge(links, on="pair_id").merge(rating, on="item_id")
    fit_figure(cur, out / "fit_accuracy.png", args.n_boot, rng)
    eff = adjusted_effects(cur, args.n_boot, rng)
    eff.to_csv(out / "xtail_effect_matched.csv", index=False)
    effect_figure(eff, out / "xtail_effect_matched.png")
    print(eff.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", default="reports/passive_band_cross/cells.csv")
    ap.add_argument("--scores", default="results/passive_band_cross/pythia14b_scores.csv")
    ap.add_argument("--links", default="data/fit_ratings/item_links.csv")
    ap.add_argument("--ratings", default="results/fit_ratings/gemma4_31b_it.csv")
    ap.add_argument("--out-dir", default="reports/figures")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
