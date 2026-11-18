#!/usr/bin/env python3
"""Staging curves per model: passive D(by) and D(O) against relative depth for the full DAS edit (v), its on-axis
piece (along u_md) and its off-axis piece, from results/{model}/split_edit/summary_passive.csv.

Relative depth = site / number of layers (Pythia 24, Gemma 26, OLMo 16). Bands are 95% bootstrap CIs; the dashed grey
line is the declared margin δ_s (a RISE needs the lower 95% bound above it). Filled markers are RISE, hollow are not.
Pythia's split edit was run at sites 4, 6, 8, 12 and 17 only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MODELS = [("pythia", "Pythia-1.4B", 24), ("gemma", "Gemma-3-1B", 26), ("olmo", "OLMo-2-1B", 16)]
PIECES = [("full", "v (full edit)", "#2a78d6"), ("on", "on-axis piece (along u_md)", "#eb6834"), ("off", "off-axis piece", "#1baf7a")]
READOUTS = [("by", "D(by)"), ("O", "D(O)")]


def main(a):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharex=True, sharey="row")
    for j, (m, name, L) in enumerate(MODELS):
        S = pd.read_csv(Path(a.results) / m / "split_edit" / "summary_passive.csv")
        for i, (ro, ylab) in enumerate(READOUTS):
            ax = axes[i, j]
            R = S[S.readout == ro].sort_values("site")
            x = R.site / L
            ax.axhline(0, color="#999999", lw=0.8)
            ax.plot(x, R.delta_s, ls="--", color="#777777", lw=1.2, label="margin δ" if (i, j) == (0, 0) else None)
            for p, lab, col in PIECES:
                ax.fill_between(x, R[f"{p}_lo95"], R[f"{p}_hi95"], color=col, alpha=0.15, lw=0)
                ax.plot(x, R[f"{p}_est"], color=col, lw=2, label=lab if (i, j) == (0, 0) else None)
                rise = R[f"class_{p}"].eq("RISE")
                ax.scatter(x[rise], R[f"{p}_est"][rise], s=36, color=col, zorder=3)
                ax.scatter(x[~rise], R[f"{p}_est"][~rise], s=36, facecolor="white", edgecolor=col, lw=1.5,
                           zorder=3)
            if i == 0:
                ax.set_title(name, fontsize=12)
            if j == 0:
                ax.set_ylabel(f"passive {ylab} (nats)")
            if i == 1:
                ax.set_xlabel("relative depth (site / layers)")
            ax.grid(axis="y", color="#e6e6e6", lw=0.7)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, fontsize=10, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.04))
    fig.text(0.5, -0.01, "Bands: 95% bootstrap CI. Filled marker: RISE (lower bound above δ); hollow: not RISE. "
             "Pythia split edit at sites 4, 6, 8, 12, 17 only.", ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="reports/figures/staging_curves_split_edit.png")
    main(ap.parse_args())
