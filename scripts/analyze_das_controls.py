#!/usr/bin/env python3
"""Gemma replication: trained DAS IIA vs random rank-1 directions vs a linear probe, per site (run_das_controls.py).

Per model x task x site, all on the sweep's held-out sets (split 0, 5 folds):
- trained: the sweep's best-epoch held-out cross-class IIA, mean over folds (the sweep tables' number). Pythia
  passive: the stored reverse sweep (sites 4-17, run in October) plus the fill-in sweep at the other sites
  (results/gemma/das/pythia_passive_sweep_fill, run today with the same arguments);
- random: per direction, the fold-mean cross-class IIA; mean and 95th percentile over the directions;
- unpatched: the same swaps' IIA with no patch (the base readout already on the source's side), mean over folds;
- probe: logistic-regression T vs I held-out accuracy, mean over folds; and the minimum over folds.
Gemma and Pythia are aligned by relative depth (site / number of layers): each Pythia site is paired with the
Gemma site round(p * 26 / 24). The last site (post-final-norm states) is shown but marked.

Writes reports/gemma/das_controls.md and results/gemma/das/controls/summary.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

L = {"pythia-1.4b": 24, "gemma-3-1b-pt": 26}
SWEEPS = {("pythia-1.4b", "active"): ["results/das_round2/sweep_strict/sweep.csv"],
          ("gemma-3-1b-pt", "active"): ["results/gemma/das/sweep_strict/sweep.csv"],
          ("pythia-1.4b", "passive"): ["results/das_round2/reverse/sweep.csv",
                                       "results/gemma/das/pythia_passive_sweep_fill/sweep.csv"],
          ("gemma-3-1b-pt", "passive"): ["results/gemma/das/reverse_sweep/sweep.csv"]}
DIRS = {("pythia-1.4b", "active"): "pythia_active", ("gemma-3-1b-pt", "active"): "gemma_active",
        ("pythia-1.4b", "passive"): "pythia_passive", ("gemma-3-1b-pt", "passive"): "gemma_passive"}


def trained(paths):
    out = []
    for i, p in enumerate(paths):
        if not Path(p).exists():
            continue
        g = pd.read_csv(p).groupby(["site", "epoch"]).iia_cross.mean().reset_index()
        b = g.loc[g.groupby("site").iia_cross.idxmax()].assign(trained_src="stored" if i == 0 else "fill-in")
        out.append(b.rename(columns={"iia_cross": "trained", "epoch": "trained_epoch"}))
    t = pd.concat(out)
    assert not t.site.duplicated().any(), "fill-in sweep overlaps the stored sweep"
    return t.set_index("site")


def main(args):
    rows = []
    for (model, task), d in DIRS.items():
        d = Path(args.root) / d
        if not (d / "random.csv").exists():
            continue
        R, P = pd.read_csv(d / "random.csv"), pd.read_csv(d / "probe.csv")
        per_draw = R.groupby(["site", "draw"]).iia_cross.mean()
        rnd = per_draw.groupby("site").agg(random_mean="mean", random_p95=lambda x: x.quantile(0.95),
                                           random_max="max", n_draws="size")
        unp = R[R.draw == 0].groupby("site").iia_cross_unpatched.mean().rename("unpatched")
        prb = P.groupby("site").acc.agg(probe="mean", probe_min="min")
        t = trained(SWEEPS[(model, task)])
        S = rnd.join(unp).join(prb).join(t[["trained", "trained_epoch", "trained_src"]], how="left").reset_index()
        S.insert(0, "model", model)
        S.insert(1, "task", task)
        S.insert(3, "depth", S.site / L[model])
        S["last_site"] = S.site == L[model]
        rows.append(S)
    A = pd.concat(rows, ignore_index=True)
    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    A.to_csv(out, index=False)

    f3 = lambda x: "—" if pd.isna(x) else f"{x:.3f}"
    lines = ["# DAS controls: trained IIA vs random rank-1 directions vs a linear probe", "",
             "Scripts `scripts/run_das_controls.py` (runs) and `scripts/analyze_das_controls.py` (this table; "
             "definitions in its docstring). Held-out cross-class swaps / items of the sweep (split 0, 5 folds; the "
             "token split's held-out verbs). Trained = sweep best-epoch IIA (fold mean); random = 20 untrained unit "
             "directions through the same interchange (mean and 95th percentile of the per-direction fold-mean IIA); "
             "unpatched = the same swaps with no patch; probe = logistic-regression T vs I held-out accuracy (fold "
             "mean). Rows: Pythia site p and Gemma site round(p × 26 / 24) (nearest relative depth). † = last site "
             "(post-final-norm states); ‡ = Pythia passive trained IIA from the fill-in sweep (run today; the stored "
             "sites 4–17 are from October).", ""]
    for task in ("active", "passive"):
        P = A[(A.model == "pythia-1.4b") & (A.task == task)].set_index("site")
        G = A[(A.model == "gemma-3-1b-pt") & (A.task == task)].set_index("site")
        if P.empty or G.empty:
            continue
        lines += [f"## {task} task", "",
                  "| P site | G site | depth P / G | P trained | P random mean / p95 | P unpatched | P probe | "
                  "G trained | G random mean / p95 | G unpatched | G probe |",
                  "|---:|---:|---|---:|---|---:|---:|---:|---|---:|---:|"]
        for p in P.index:
            g = int(round(p * L["gemma-3-1b-pt"] / L["pythia-1.4b"]))
            if g not in G.index:
                continue
            a, b = P.loc[p], G.loc[g]
            mark = lambda r, s: ("†" if r.last_site else "")
            pt = f3(a.trained) + ("‡" if a.trained_src == "fill-in" else "")
            lines.append(f"| {p}{mark(a, p)} | {g}{mark(b, g)} | {a.depth:.2f} / {b.depth:.2f} | {pt} | "
                         f"{f3(a.random_mean)} / {f3(a.random_p95)} | {f3(a.unpatched)} | {f3(a.probe)} | "
                         f"{f3(b.trained)} | {f3(b.random_mean)} / {f3(b.random_p95)} | {f3(b.unpatched)} | "
                         f"{f3(b.probe)} |")
        lines.append("")
    Path(args.report).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results/gemma/das/controls")
    ap.add_argument("--csv", default="results/gemma/das/controls/summary.csv")
    ap.add_argument("--report", default="reports/gemma/das_controls.md")
    main(ap.parse_args())
