#!/usr/bin/env python3
"""Figure: change in object-start and " by" log-probability after the T-vs-I donor patch, by DAS site.

D = mean change after a transitive active donor minus after an intransitive
one, on primary bad passive bases (64 pairs), with 95% CIs (three-way cluster
bootstrap). Background bands show the declared outcome at each site.
Reads `transfer_summary.csv` from the main and fill-in passive-test dirs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE = "#2a78d6", "#eb6834"  # validated categorical slots 1-2 (dataviz default palette)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
OUTCOME_FILL = {"abstract": "#fcfcfb", "mixed": "#f0efec", "surface": "#e4e3df"}  # neutral, so bands never read as series


def run(args):
    tr = pd.concat(pd.read_csv(Path(d) / "transfer_summary.csv") for d in args.dirs.split(","))
    d = tr[(tr.population == "primary (plain)") & (tr.band == "all") & (tr.condition == "D_bad")]
    dep = pd.read_csv(args.depth).set_index("site")
    sites = sorted(d.site.unique())

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF)
    # outcome bands (half-way between neighbouring sites)
    edges = [sites[0] - 1] + [(a + b) / 2 for a, b in zip(sites, sites[1:])] + [sites[-1] + 1]
    for i, s in enumerate(sites):
        oc = dep.outcome[s].split(" ")[0]
        ax.axvspan(edges[i], edges[i + 1], color=OUTCOME_FILL[oc], lw=0, zorder=0)
    for oc, xs in pd.Series({s: dep.outcome[s].split(" ")[0] for s in sites}).groupby(lambda s: dep.outcome[s].split(" ")[0]):
        lo, hi = edges[sites.index(min(xs.index))], edges[sites.index(max(xs.index)) + 1]
        ax.text((lo + hi) / 2, 1.0, oc, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=9, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    for rd, col, mk, lab in (("O", BLUE, "o", "object starts"), ("by", ORANGE, "s", '" by"')):
        x = d[d.readout == rd].set_index("site").loc[sites]
        ax.fill_between(sites, x.lo95, x.hi95, color=col, alpha=0.18, lw=0, zorder=2)
        ax.plot(sites, x.est, color=col, lw=2, marker=mk, ms=6, mec=SURF, mew=1.2, zorder=3, label=lab)
        ax.annotate(lab, (sites[-1], x.est.iloc[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK)
    ax.set_xticks(sites)
    ax.set_xticklabels([f"{s}" + ("*" if s == 17 else "") for s in sites])
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel("DAS site (output of layer site − 1); * = frozen primary site", color=INK2, fontsize=9)
    ax.set_ylabel("D: Δ log P, transitive − intransitive donor (nats)", color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("Patching d into bad passives: object starts vs \" by\", by layer", fontsize=10, color=INK, pad=16,
                 loc="left")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURF)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURF)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dirs", default="results/das_round2/passive_test,results/das_round2/passive_test_fill")
    ap.add_argument("--depth", default="results/das_round2/passive_test_fill/depth_summary.csv")
    ap.add_argument("--out", default="reports/passive_das_prep/figures/depth_D.png")
    run(ap.parse_args())
