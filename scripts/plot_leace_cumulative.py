#!/usr/bin/env python3
"""Figure: cumulative LEACE scrub curve on Pythia-1.4B, with the shuffled-label control.

x: k (sequential scrub over sites 0..k); y: share of the natural *by* gap (T - I log P(by)) remaining.
Lines: in-sample and cross-fitted erasers, with 95% cluster-bootstrap CIs as bands
(reports/pythia/leace_necessity_cumulative.csv). Points: in-sample erasers fit on balanced permuted verb labels
(20 permutations) at k = 8, 12, 23, mean with 5th-95th percentile bars (leace_permutation_summary.csv for the mean,
leace_permutation_per_perm.csv for the percentiles). Sites k >= 13 are shaded: they sit next to the readout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # validated categorical slots 1-3 (dataviz default palette)
INK, INK2, GRID, SURF, SHADE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb", "#f0efec"
PERM_K = {"scrub_0_8": 8, "scrub_0_12": 12, "scrub_all": 23}
NEAR_READOUT = 13


def run(args):
    cum = pd.read_csv(args.cumulative)
    summ = pd.read_csv(args.perm_summary).set_index("condition")
    per = pd.read_csv(args.perm_per)
    per = per[per.labels != "true"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    kmax = int(cum.k.max())
    ax.axvspan(NEAR_READOUT - 0.5, kmax + 0.5, color=SHADE, lw=0, zorder=0)
    ax.text((NEAR_READOUT - 0.5 + kmax + 0.5) / 2, 1.0, "near readout", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=9, color=INK2)
    for y, lab in ((1.0, "no erasure"), (0.0, "gap fully removed")):
        ax.axhline(y, color=INK2, lw=0.8, ls=(0, (4, 3)) if y == 1 else "-", zorder=1)
        ax.text(-0.3, y, lab, ha="left", va="bottom", fontsize=8, color=INK2, zorder=1)

    for run_, col, mk, lab in (("in-sample", BLUE, "o", "in-sample"), ("cross-fitted", ORANGE, "s", "cross-fitted")):
        c = cum[cum.run == run_].sort_values("k")
        ax.fill_between(c.k, c.share_remaining_lo, c.share_remaining_hi, color=col, alpha=0.18, lw=0, zorder=2)
        ax.plot(c.k, c.share_remaining, color=col, lw=2, marker=mk, ms=4.5, mec=SURF, mew=1, zorder=3,
                label=f"{lab} (95% CI band)")
        ax.annotate(lab, (c.k.iloc[-1], c.share_remaining.iloc[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)

    xs, ys, lo, hi = [], [], [], []
    for cond, k in PERM_K.items():
        q = per[per.condition == cond].share_remaining.to_numpy()
        assert len(q) == summ.n_perms[cond]
        m = summ.perm_share_mean[cond]
        p5, p95 = np.percentile(q, [5, 95])
        assert np.isclose(p5, summ.perm_share_p5[cond])
        xs.append(k); ys.append(m); lo.append(m - p5); hi.append(p95 - m)
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="D", color=AQUA, ms=4.5, mec=SURF, mew=0.8, elinewidth=1.5, capsize=4, capthick=1.5,
                zorder=4, label="shuffled labels, in-sample (mean, 5th–95th pct, 20 perms)")
    ax.annotate("shuffled labels", (xs[1], ys[1]), xytext=(0, 9), textcoords="offset points", ha="center",
                va="bottom", fontsize=9, color=INK)

    ax.set_xticks(range(0, kmax + 1))
    ax.set_xlim(-0.5, kmax + 0.5)
    ax.set_ylim(top=1.17)
    ax.set_xlabel("k: sites 0..k scrubbed (sequential LEACE, rank 1, verb's last token)", color=INK2, fontsize=9)
    ax.set_ylabel("share of natural \" by\" gap remaining", color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.set_title("Pythia-1.4B: cumulative LEACE scrub of verb class", fontsize=10, color=INK, pad=16, loc="left")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURF)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURF)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cumulative", default="reports/pythia/leace_necessity_cumulative.csv")
    ap.add_argument("--perm-summary", default="reports/pythia/leace_permutation_summary.csv")
    ap.add_argument("--perm-per", default="reports/pythia/leace_permutation_per_perm.csv")
    ap.add_argument("--out", default="reports/pythia/figures/leace_cumulative.png")
    run(ap.parse_args())
