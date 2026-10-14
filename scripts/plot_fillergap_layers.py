#!/usr/bin/env python3
"""Round 4, Part B figure: the in-range filler-gap patch by layer.

From `results/round4/fillergap/patch_summary.csv` (half-B contexts, D_in = coordinate set to the T-verb vs the
I-verb level on d). Per experiment (direct questions "What did John V?", embedded "I know what John V"): the
patch effect on the clause end ("?" / ".") and on prepositions, with the filler ("what") and without
("that" / no wh), 95% CIs; layers where the declared reading is the licensing-like crossover are shaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# reference palette (dataviz skill), light mode
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SHADE = "#0b0b0b", "#52514e", "#e4e3df", "#f1f0ec"
EXPS = (("mat", "Direct questions", '"What did John V?"', '"?"', (4, 6, 8, 10)),
        ("emb", "Embedded questions", '"I know what John V"', '"."', (8, 10)))


def run(args):
    s = pd.read_csv(args.summary)
    s = s[(s.population == "half B") & (s.quantity == "D_in")]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "font.family": "sans-serif"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (exp, title, example, end, cross) in zip(axes, EXPS):
        e = s[s.exp == exp]
        ax.axvspan(min(cross) - 1, max(cross) + 1, color=SHADE, zorder=0, lw=0)
        ax.axhline(0, color=INK2, lw=1, zorder=1)
        series = ((f"{end} with filler", "num", "fill", BLUE, "-"), (f"{end} without filler", "num", "nofill", BLUE, "--"),
                  ("prepositions with filler", "PREP", "fill", ORANGE, "-"),
                  ("prepositions without filler", "PREP", "nofill", ORANGE, "--"))
        for label, rd, fr, col, ls in series:
            d = e[(e.readout == rd) & (e.frame == fr)].sort_values("site")
            ax.fill_between(d.site, d.lo95, d.hi95, color=col, alpha=0.12, lw=0, zorder=2)
            ax.plot(d.site, d.est, ls=ls, color=col, lw=2, marker="o", ms=4.5, zorder=3, label=label,
                    markeredgecolor="white", markeredgewidth=1)
        ax.set_title(f"{title}  {example}", loc="left", fontsize=11, color=INK)
        ax.set_xticks(sorted(e.site.unique()))
        ax.set_xlabel("layer (patch site)")
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.text((min(cross) + max(cross)) / 2, -3.8, "licensing-like crossover", ha="center", va="bottom",
                fontsize=8.5, color=INK2)
    axes[0].set_ylabel("patch effect (log-prob change)")
    axes[0].set_ylim(-3.9, 1.7)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, [x.replace('"?"', "clause end (?/.)") for x in l], loc="lower center", ncol=4, frameon=False,
               fontsize=9.5, labelcolor=INK)
    fig.suptitle("Setting the early verb-class coordinate to the transitive level: effect with vs. without a filler",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(out.with_suffix(".png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default="results/round4/fillergap/patch_summary.csv")
    ap.add_argument("--out", default="reports/round4/figures/fillergap_layers.png")
    run(ap.parse_args())
