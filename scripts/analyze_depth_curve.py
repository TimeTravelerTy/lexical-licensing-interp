#!/usr/bin/env python3
"""Gemma replication, EXPLORATORY passive-only depth curve (rules: reports/gemma/exploratory.md, section 6).

Every site from 0 to ~60% relative depth (Gemma 0-15 of 26 layers, Pythia 0-14 of 24): the passive test's
declared quantities (analyze_passive_test.py `passive`: primary population (plain bad verbs), all bands, D on bad
passive bases = after a transitive minus after an intransitive active donor; RISE / NO RISE classes against
δ_s = 0.2 x the site's held-out active ΔlogP(O); outcome by analyze_passive_test.outcome), read from the run
directories in order (the first that has a site wins):
- Gemma: results/gemma/das/transfer/passive_test (sites 0-5, 8, 12, 15), then results/gemma/das/depth_curve/gemma_passive;
- Pythia: results/das_round2/passive_test (8, 12, 17), results/das_round2/passive_test_fill (4, 6, 10, 14, 16), then
  results/gemma/das/depth_curve/pythia_passive.
Active effect (ΔlogP(O), held-out, intransitive base <- transitive source): the `tost` block of the site's frozen config
in the root that ran it (Gemma results/gemma/das/transfer; Pythia stored sites results/das_round2, new sites
results/gemma/das/pythia_early).

O fraction = D(O) / active effect; its interval is D(O)'s 95% CI divided by the active effect (held fixed). On this
scale the NO RISE bound δ_s is 0.2 at every site. *by* effect = D(log P(" by")) in nats with its 95% CI; also as a
fraction of the natural *by* gap (frac_natgap_est, as in transfer.md).

Reproduction anchors: Gemma 5 and Pythia 4 are rerun in the new directories; every transfer_summary.csv row at the
anchor must equal the stored one (max |diff| reported).

Writes results/gemma/das/depth_curve/depth_curve.csv, reports/gemma/depth_curve.md and
reports/gemma/figures/depth_curve_passive.{png,svg}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_passive_test import outcome

NL = {"Gemma-3-1B": 26, "Pythia-1.4B": 24}
COL = {"Pythia-1.4B": "#2a78d6", "Gemma-3-1B": "#eb6834"}  # validated pair (dataviz validate_palette.js, light)
MK = {"Pythia-1.4B": "o", "Gemma-3-1B": "s"}


def tost(root, site):
    p = Path(root) / f"frozen_config_site{site}.json"
    if not p.exists():
        p = Path(root) / "frozen_config_final.json"
        assert json.loads(p.read_text())["site"] == site, (root, site)
    return json.loads(p.read_text())["tost"]


def rows(model, dirs, max_depth):
    out, seen = [], set()
    for d, root in dirs:
        if not (Path(d) / "transfer_summary.csv").exists():
            print(f"missing {d}")
            continue
        t = pd.read_csv(Path(d) / "transfer_summary.csv")
        t = t[(t.population == "primary (plain)") & (t.band == "all") & (t.condition == "D_bad")]
        for site, g in t.groupby("site"):
            site = int(site)
            if site in seen or site / NL[model] > max_depth:
                continue
            seen.add(site)
            g = g.set_index("readout")
            o, b = g.loc["O"], g.loc["by"]
            ts = tost(root, site)
            a = ts["active_effect_delta_logP_O"]
            out.append({"model": model, "site": site, "depth": site / NL[model], "source": str(d),
                        "active_effect": a, "delta": ts["delta"],
                        "O_est": o.est, "O_lo95": o.lo95, "O_hi95": o.hi95, "O_hi90": o.hi90, "O_class": o["class"],
                        "O_frac": o.est / a, "O_frac_lo95": o.lo95 / a, "O_frac_hi95": o.hi95 / a,
                        "by_est": b.est, "by_lo95": b.lo95, "by_hi95": b.hi95, "by_class": b["class"],
                        "by_frac_natgap": b.frac_natgap_est, "outcome": outcome(o["class"], b["class"])})
    return out


def anchor(stored, new, site):
    if not (Path(new) / "transfer_summary.csv").exists():
        return {"site": site, "rows_stored": 0, "rows_new": 0, "unmatched": -1, "max_abs_diff": float("nan")}
    a, b = (pd.read_csv(Path(x) / "transfer_summary.csv") for x in (stored, new))
    a, b = a[a.site == site], b[b.site == site]
    keys = [c for c in a.columns if not pd.api.types.is_float_dtype(a[c])]
    m = a.merge(b, on=keys, how="outer", suffixes=("_a", "_b"), indicator=True)
    num = [c for c in a.columns if c not in keys]
    md = max(float(np.nanmax(np.abs(m[c + "_a"] - m[c + "_b"]))) for c in num)
    return {"site": site, "rows_stored": len(a), "rows_new": len(b), "unmatched": int((m._merge != "both").sum()),
            "max_abs_diff": md}


def plot(R, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True, facecolor="#fcfcfb")
    for a_ in ax:
        a_.set_facecolor("#fcfcfb")
        a_.grid(axis="y", color="#e6e5e0", lw=0.6)
        a_.axhline(0, color="#52514e", lw=0.8)
    ax[0].axhline(0.2, color="#52514e", lw=0.8, ls="--")
    ax[0].text(0.605, 0.2, r"$\delta_s$ (NO RISE bound)", va="bottom", ha="right", fontsize=8, color="#52514e")
    for m, g in R.groupby("model"):
        g = g.sort_values("depth")
        for a_, y, lo, hi in ((ax[0], "O_frac", "O_frac_lo95", "O_frac_hi95"), (ax[1], "by_est", "by_lo95", "by_hi95")):
            a_.plot(g.depth, g[y], color=COL[m], lw=2, zorder=2)
            a_.errorbar(g.depth, g[y], yerr=[g[y] - g[lo], g[hi] - g[y]], fmt="none", ecolor=COL[m], elinewidth=1,
                        capsize=0, zorder=2)
            ab = g.outcome.str.startswith("abstract")
            a_.scatter(g.depth[~ab], g[y][~ab], s=30, marker=MK[m], facecolor="#fcfcfb", edgecolor=COL[m], lw=1.5,
                       zorder=3)
            a_.scatter(g.depth[ab], g[y][ab], s=30, marker=MK[m], facecolor=COL[m], edgecolor=COL[m], lw=1.5,
                       zorder=4)
        ax[1].plot(g.depth, g.delta, color=COL[m], lw=1, ls=":", zorder=1)
        last = g.iloc[-1]
        ax[0].annotate(m, (last.depth, last.O_frac), xytext=(4, 0), textcoords="offset points", fontsize=8,
                       color="#0b0b0b", va="center")
    ax[0].set_ylabel("D log P(O) / active effect")
    ax[0].set_title("Object continuation, fraction of the active effect", fontsize=9, loc="left")
    ax[1].set_ylabel("D log P(\" by\")  (nats)")
    ax[1].set_title(r"$by$ effect (dotted: each model's $\delta_s$)", fontsize=9, loc="left")
    for a_ in ax:
        a_.set_xlabel("relative depth (site / layers)")
        a_.set_xlim(-0.02, 0.62)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COL[m], marker=MK[m], lw=2, markerfacecolor="#fcfcfb", label=m) for m in COL]
    h += [Line2D([], [], color="#52514e", marker="o", lw=0, markerfacecolor="#52514e", label="filled: outcome abstract"),
          Line2D([], [], color="#52514e", marker="o", lw=0, markerfacecolor="#fcfcfb", label="hollow: other outcome")]
    ax[1].legend(handles=h, frameon=False, fontsize=8, loc="upper left")
    fig.text(0.01, 0.01, "Passive test, primary population, active T − I donors; 95% CIs (O fraction: D(O) CI / active "
             "effect). EXPLORATORY.", fontsize=7, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{path}.{ext}", dpi=200, facecolor="#fcfcfb")


def main(a):
    R = pd.DataFrame(
        rows("Gemma-3-1B", [(a.gemma_transfer + "/passive_test", a.gemma_transfer), (a.gemma_new, a.gemma_transfer)],
             a.max_depth)
        + rows("Pythia-1.4B", [(a.pythia_root + "/passive_test", a.pythia_root),
                               (a.pythia_root + "/passive_test_fill", a.pythia_root), (a.pythia_new, a.pythia_early)],
               a.max_depth)).sort_values(["model", "depth"])
    out = Path(a.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(out, index=False)
    anc = {"Gemma-3-1B": anchor(a.gemma_transfer + "/passive_test", a.gemma_new, 5),
           "Pythia-1.4B": anchor(a.pythia_root + "/passive_test_fill", a.pythia_new, 4)}
    (out.parent / "anchors.json").write_text(json.dumps(anc, indent=2) + "\n")
    plot(R, a.figure)
    f2 = lambda e, l, h: f"{e:+.2f} [{l:+.2f}, {h:+.2f}]"
    L = ["# Passive-only depth curve, 0 to ~60% depth (EXPLORATORY)", "",
         "Generated by `scripts/analyze_depth_curve.py` (sources and definitions in its docstring; rules in "
         "`exploratory.md`, section 6). Passive test, primary population, all bands, D on bad passive bases (active "
         "T − I donors), 95% CIs. O fraction = D(O) / the site's held-out active effect; δ_s is 0.2 on that scale. "
         "\"new\" = run in this batch (results/gemma/das/depth_curve).", "",
         f"Reproduction anchors (rerun in the new directories, all transfer_summary rows at the site): "
         + "; ".join(f"{m} site {v['site']}: {v['rows_new']} / {v['rows_stored']} rows, unmatched {v['unmatched']}, "
                     f"max |diff| {v['max_abs_diff']:.3g}" for m, v in anc.items()) + ".", "",
         f"![depth curve]({Path(a.figure).name}.png)".replace("](", "](figures/"), "",
         "| model | site | depth | run | active effect | δ_s | D log P(O) | class | O fraction [95% CI] | "
         "D log P(\" by\") | class | D(by) / natural gap | outcome |",
         "|---|---:|---:|---|---:|---:|---|---|---|---|---|---:|---|"]
    for r in R.itertuples():
        L.append(f"| {r.model} | {r.site} | {r.depth:.2f} | {'new' if 'depth_curve' in r.source else 'stored'} | "
                 f"{r.active_effect:.2f} | {r.delta:.2f} | {f2(r.O_est, r.O_lo95, r.O_hi95)} | {r.O_class} | "
                 f"{f2(r.O_frac, r.O_frac_lo95, r.O_frac_hi95)} | {f2(r.by_est, r.by_lo95, r.by_hi95)} | {r.by_class} | "
                 f"{r.by_frac_natgap:.2f} | {r.outcome} |")
    Path(a.report).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gemma-transfer", default="results/gemma/das/transfer")
    ap.add_argument("--gemma-new", default="results/gemma/das/depth_curve/gemma_passive")
    ap.add_argument("--pythia-root", default="results/das_round2")
    ap.add_argument("--pythia-early", default="results/gemma/das/pythia_early")
    ap.add_argument("--pythia-new", default="results/gemma/das/depth_curve/pythia_passive")
    ap.add_argument("--max-depth", type=float, default=0.6)
    ap.add_argument("--csv", default="results/gemma/das/depth_curve/depth_curve.csv")
    ap.add_argument("--report", default="reports/gemma/depth_curve.md")
    ap.add_argument("--figure", default="reports/gemma/figures/depth_curve_passive")
    main(ap.parse_args())
