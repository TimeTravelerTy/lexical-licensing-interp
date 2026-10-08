#!/usr/bin/env python3
"""Figure for round-3 A2: retained separation along d_8 and d_8-orthogonal-to-d_17, by site.

Retention = mean good - bad (passive) or transitive - intransitive (active) raw
gap at site s divided by the same gap at site 8, 95% pair-bootstrap CI
(`late_projection_ratios.csv`). Both voices are indexed to their own site-8
level, so they share one axis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PASSIVE, ACTIVE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 of the reference palette
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def run(args):
    r = pd.read_csv(args.ratios)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, constrained_layout=True)
    for ax, (ro, title) in zip(axes, (("d8", "Along d₈"), ("d8perp17", "Along d₈ with the d₁₇ component removed"))):
        d = r[r.readout == ro].sort_values("site")
        ax.axhline(1, color=MUTED, lw=1, ls=(0, (2, 2)), zorder=1)
        ax.axvline(8, color=GRID, lw=1, zorder=0)
        for col, color, ls, name in (("ret_passive", PASSIVE, "-", "passive (good − bad)"),
                                     ("ret_active", ACTIVE, (0, (5, 2)), "active (trans − intrans)")):
            ax.fill_between(d.site, d[f"{col}_lo"], d[f"{col}_hi"], color=color, alpha=0.15, lw=0, zorder=2)
            ax.plot(d.site, d[col], color=color, lw=2, ls=ls, marker="o", ms=4, zorder=3, label=name)
            last = d.iloc[-1]
            ax.annotate(f"{last[col]:.2f}", (last.site, last[col]), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("site (residual after layer s − 1)", fontsize=9, color=MUTED)
        ax.set_xticks([4, 8, 12, 16, 20, 23])
        ax.set_xlim(3.5, 24.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    axes[0].set_ylabel("gap relative to site 8", fontsize=9, color=MUTED)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Early verb-class separation persists in passives but is consumed in actives (64 pairs, 95% CI)",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ratios", default="results/das_round2/round3/late_projection_ratios.csv")
    ap.add_argument("--out", default="reports/passive_das_prep/figures/round3_late_projection.png")
    run(ap.parse_args())
