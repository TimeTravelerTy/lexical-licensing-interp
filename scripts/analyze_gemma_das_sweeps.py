#!/usr/bin/env python3
"""Gemma replication: per-site summary of the rank-1 DAS sweeps (active round-2 task and passive *by* task).

For each sweep (model x task) and site: the best epoch by mean held-out cross-class IIA over the 5 folds (split 0),
with its fold mean / min / max IIA, same-class IIA, positive-control fraction (median, fold mean) and, for the passive
task, held-out-context IIA. Site picks:
- frozen rule (round 2, select_das_config.py): the earliest site whose best IIA is within 0.02 of the overall best
  and whose positive-control fraction is >= 0.5 (the active task's declared rule; applied descriptively to the
  passive task);
- best site: the highest best-epoch IIA (excluding the last site, whose source states are post-final-norm).
  Here the last site is also left out of the frozen rule's overall best; select_das_config.py keeps it, and its pick
  is printed too where the two differ (OLMo active: 12 here, 13 there).
Pythia's stored sweeps are listed beside Gemma's by relative depth (site / number of layers).

Writes reports/gemma/das_sweeps.md and results/gemma/das/sweep_summary_by_site.csv. --sweep model:task:path:n_layers
(repeatable) replaces the default list and --title the heading (OLMo replication: reports/olmo/das_sweep.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SWEEPS = [  # model, task, path, n_layers
    ("pythia-1.4b", "active", "results/das_round2/sweep_strict/sweep.csv", 24),
    ("gemma-3-1b-pt", "active", "results/gemma/das/sweep_strict/sweep.csv", 26),
    ("pythia-1.4b", "passive", "results/das_round2/reverse/sweep.csv", 24),
    ("gemma-3-1b-pt", "passive", "results/gemma/das/reverse_sweep/sweep.csv", 26),
]


def by_site(sw, task):
    agg = {"iia": ("iia_cross", "mean"), "iia_min": ("iia_cross", "min"), "iia_max": ("iia_cross", "max"),
           "iia_same": ("iia_same", "mean"), "pc": ("pc_frac_median", "mean")}
    if task == "passive":
        agg["iia_heldout_ctx"] = ("iia_cross_heldout_ctx", "mean")
        agg["pc_dM"] = ("pc_dM", "mean")
    g = sw.groupby(["site", "epoch"]).agg(**agg).reset_index()
    return g.loc[g.groupby("site").iia.idxmax()].reset_index(drop=True)


def picks(b, n_layers, tol=0.02, min_pc=0.5):
    inner = b[b.site < n_layers]  # last site (n_layers) excluded: post-final-norm source states
    top = inner.iia.max()
    ok = inner[(inner.iia >= top - tol) & (inner.pc >= min_pc)]
    ok_all = b[(b.iia >= b.iia.max() - tol) & (b.pc >= min_pc)]  # select_das_config.py: last site kept in the top
    return {"best_site": int(inner.loc[inner.iia.idxmax(), "site"]), "best_iia": float(top),
            "frozen_rule_site": int(ok.site.min()) if len(ok) else None,
            "select_das_config_site": int(ok_all.site.min()) if len(ok_all) else None}


def main(args):
    sweeps = [(m, t, p, int(n)) for m, t, p, n in (x.split(":") for x in args.sweep)] if args.sweep else SWEEPS
    rows, lines = [], [f"# {args.title}", "",
                       "Script `scripts/analyze_gemma_das_sweeps.py` (definitions in its docstring). Held-out "
                       "cross-class IIA at the participle's last subtoken, best epoch per site, mean over 5 folds "
                       "(split 0); [min, max] over folds. Site s = residual output of layer s − 1; relative depth = "
                       "s / number of layers. Last site excluded from the picks (post-final-norm source states).", ""]
    for model, task, path, L in sweeps:
        if not Path(path).exists():
            lines += [f"## {model}, {task}: not run yet", ""]
            continue
        b = by_site(pd.read_csv(path), task).assign(model=model, task=task, depth=lambda d: d.site / L)
        rows.append(b)
        pk = picks(b, L)
        lines += [f"## {model}, {task} task", "",
                  f"Best site {pk['best_site']} (IIA {pk['best_iia']:.3f}); frozen-rule site "
                  f"{pk['frozen_rule_site']}" + ("." if pk["select_das_config_site"] == pk["frozen_rule_site"] else
                  f" (`select_das_config.py`, which keeps the last site in the overall best, picks "
                  f"{pk['select_das_config_site']}).") , "",
                  "| site | depth | epoch | IIA cross [min, max] | IIA same | pc frac | " +
                  ("IIA held-out ctx | pc ΔM |" if task == "passive" else ""),
                  "|---:|---:|---:|---|---:|---:|" + ("---:|---:|" if task == "passive" else "")]
        for r in b.itertuples():
            extra = f" {r.iia_heldout_ctx:.3f} | {r.pc_dM:+.2f} |" if task == "passive" else ""
            lines.append(f"| {r.site} | {r.depth:.2f} | {r.epoch} | {r.iia:.3f} [{r.iia_min:.3f}, {r.iia_max:.3f}] | "
                         f"{r.iia_same:.3f} | {r.pc:.2f} |{extra}")
        lines.append("")
    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(out, index=False)
    Path(args.report).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="results/gemma/das/sweep_summary_by_site.csv")
    ap.add_argument("--report", default="reports/gemma/das_sweeps.md")
    ap.add_argument("--sweep", action="append", default=[], help="model:task:path:n_layers (replaces the default list)")
    ap.add_argument("--title", default="Rank-1 DAS sweeps: Gemma-3-1B-pt vs Pythia-1.4B")
    main(ap.parse_args())
