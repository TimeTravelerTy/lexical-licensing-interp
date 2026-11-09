#!/usr/bin/env python3
"""Gemma replication, exploratory / secondary tables (rules: reports/gemma/exploratory.md).

2. In-range equivalence test: the passive test's classes (analyze_passive_test.classify: RISE = 95% CI > 0 and
   estimate >= δ_s; FALL mirrored; NO RISE = 90% CI within ±δ_s; else unresolved) applied to the in-range passive
   effects D_in = R(t_T) − R(t_I) on bad passive items (inrange_summary.csv; the in-range analysis's own two-way
   bootstrap), with the declared passive test's site-specific δ_s (deltas.json). Readouts O (the test) and " by"
   (secondary), and the in-range outcome under the declared rule (analyze_passive_test.outcome). The declared
   active-donor outcome at the same site is shown next to it.
3. Balanced shuffled-label control (run_das_shuffle_balanced.py): per permutation the held-out cross-class IIA
   averaged over the final run's 3 splits x 5 folds; mean, 95th percentile and max over the permutations; the
   largest |phi| between true and permuted training-verb classes; next to the final run's trained DAS IIA and its
   unconstrained shuffled-label control (fold means, final_strict*/summary.csv, chosen rank).

Writes reports/gemma/exploratory_tables.md and results/gemma/das/exploratory/{inrange_tost,shuffle_balanced}.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_passive_test import classify, outcome

NL = {"Pythia-1.4B": 24, "Gemma-3-1B": 26}


def deltas(paths):
    d = {}
    for p in paths:
        d.update({int(k): float(v) for k, v in json.loads(Path(p).read_text()).items()})
    return d


def donor_outcomes(paths):
    out = {}
    for p in paths:
        t = pd.read_csv(Path(p) / "transfer_summary.csv")
        t = t[(t.population == "primary (plain)") & (t.band == "all") & (t.condition == "D_bad")]
        for site, g in t.groupby("site"):
            g = g.set_index("readout")
            out[int(site)] = outcome(g.loc["O", "class"], g.loc["by", "class"])
    return out


def inrange_tost(cfg):
    rows = []
    for model, c in cfg.items():
        s = pd.read_csv(Path(c["inrange"]) / "inrange_summary.csv")
        s = s[s.construction == "passive"].set_index(["site", "readout"])
        dl, don = deltas(c["deltas"]), donor_outcomes(c["passive"])
        for site in sorted(s.index.get_level_values(0).unique()):
            r = {"model": model, "site": int(site), "depth": site / NL[model], "delta": dl[site],
                 "role": c["roles"].get(int(site), "")}
            cls = {}
            for rd in ("O", "by"):
                x = s.loc[(site, rd)]
                st = {k: x[f"D_in_{k}"] for k in ("est", "lo95", "hi95", "lo90", "hi90")}
                cls[rd] = classify(st, dl[site])
                r.update({f"{rd}_{k}": v for k, v in st.items()})
                r[f"{rd}_class"] = cls[rd]
            r["inrange_outcome"] = outcome(cls["O"], cls["by"])
            r["donor_outcome"] = don.get(int(site), "")
            rows.append(r)
    return pd.DataFrame(rows).sort_values(["depth", "model"])


def shuffle_balanced(cfg, root):
    rows = []
    for model, c in cfg.items():
        for d in sorted(Path(root).glob(f"{c['shb_prefix']}_site*")):
            if not (d / "shuffle_balanced.csv").exists():
                continue
            S = pd.read_csv(d / "shuffle_balanced.csv")
            site = int(S.site.iloc[0])
            fdir = Path(c["final_dirs"][site])
            fm = json.loads((fdir / "final_meta.json").read_text())
            fs = pd.read_csv(fdir / "summary.csv")
            fs = fs[fs["rank"] == fm["chosen_rank"]]
            per = S.groupby("draw").iia_cross.mean()
            n_folds = S.groupby("draw").size()
            rows.append({"model": model, "site": site, "depth": site / NL[model], "role": c["roles"].get(site, ""),
                         "n_perm": int(per.size), "folds_per_perm": int(n_folds.min()),
                         "complete": bool(n_folds.min() == 15),
                         "balanced_mean": per.mean(), "balanced_p95": per.quantile(0.95), "balanced_max": per.max(),
                         "balanced_fold_max": S.iia_cross.max(), "max_abs_phi": S.phi_verbs.abs().max(),
                         "unconstrained_mean": fs[fs.control == "shuffled_labels"].iia_cross.mean(),
                         "unconstrained_fold_max": fs[fs.control == "shuffled_labels"].iia_cross.max(),
                         "trained_das": fs[fs.control == "das"].iia_cross.mean(),
                         "random_normmatched": fs[fs.control == "random_normmatched"].iia_cross.mean()})
    return pd.DataFrame(rows).sort_values(["depth", "model"]) if rows else pd.DataFrame()


def table(df, cols, heads):
    L = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for _, r in df.iterrows():
        L.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return L


def main(a):
    roles_g = {15: "declared (frozen rule)", 12: "declared (within 0.05 of max)", 8: "declared (Pythia 0.9 rule)",
               5: "declared (Pythia 0.8 rule)", 4: "exploratory (IIA-matched to Pythia 8)",
               3: "exploratory (early span)", 2: "exploratory (early span)",
               1: "exploratory (IIA-matched to Pythia 6)", 0: "exploratory (IIA-matched to Pythia 4)"}
    roles_p = {s: "stored" for s in (4, 6, 8, 10, 12, 14, 16, 17)}
    cfg = {"Pythia-1.4B": {"inrange": a.pythia_inrange, "passive": a.pythia_passive.split(","),
                           "deltas": [f"{p}/deltas.json" for p in a.pythia_passive.split(",")], "roles": roles_p,
                           "shb_prefix": "pythia",
                           "final_dirs": {17: f"{a.pythia_final}/final_strict",
                                          **{s: f"{a.pythia_final}/final_strict_site{s}" for s in (4, 6, 8, 10, 12, 14, 16)}}},
           "Gemma-3-1B": {"inrange": f"{a.gemma}/inrange", "passive": [f"{a.gemma}/passive_test"],
                          "deltas": [f"{a.gemma}/passive_test/deltas.json"], "roles": roles_g, "shb_prefix": "gemma",
                          "final_dirs": {15: f"{a.gemma}/final_strict",
                                         **{s: f"{a.gemma}/final_strict_site{s}" for s in (0, 1, 2, 3, 4, 5, 8, 12)}}}}
    out = Path(a.csv_dir)
    out.mkdir(parents=True, exist_ok=True)
    f2 = lambda r, k: f"{r[k + '_est']:+.2f} [{r[k + '_lo95']:+.2f}, {r[k + '_hi95']:+.2f}] / 90% [{r[k + '_lo90']:+.2f}, {r[k + '_hi90']:+.2f}]"
    L = ["# Gemma replication: exploratory / secondary tables", "",
         "**Exploratory / secondary; the declared results are in `transfer.md` and `controls_transfer.md`.** "
         "Script `scripts/analyze_gemma_exploratory.py` (definitions in its docstring); rules in `exploratory.md`.", ""]
    I = inrange_tost(cfg)
    I.to_csv(out / "inrange_tost.csv", index=False)
    I["O_cell"] = [f"{f2(r, 'O')} {r.O_class}" for _, r in I.iterrows()]
    I["by_cell"] = [f"{f2(r, 'by')} {r.by_class}" for _, r in I.iterrows()]
    L += ["## 2. In-range equivalence test (passive, secondary)", "",
          "D_in = R(t_T) − R(t_I) on bad passive items, nats, 95% CI / 90% CI (two-way bootstrap); classes against "
          "the declared δ_s. The in-range outcome applies the declared surface / abstract / mixed rule to "
          "(O class, *by* class); the active-donor outcome is the declared result at the same site.", ""]
    L += table(I, ["model", "site", "depth", "role", "delta", "O_cell", "by_cell", "inrange_outcome", "donor_outcome"],
               ["model", "site", "depth", "role", "δ_s", "D_in log P(O)", "D_in log P(\" by\")", "in-range outcome",
                "active-donor outcome (declared)"]) + [""]
    S = shuffle_balanced(cfg, a.shb_root)
    L += ["## 3. Balanced shuffled-label control", ""]
    if S.empty:
        L += ["(no runs yet)", ""]
    else:
        S.to_csv(out / "shuffle_balanced.csv", index=False)
        L += ["Held-out cross-class IIA. Balanced: per permutation, the mean over the final run's 3 splits × 5 folds; "
              "mean / 95th percentile / max over the permutations, and the largest single-fold value. Unconstrained: "
              "the final run's shuffled-label control (one random permutation per fold), mean and fold max. "
              "Trained: the final run's DAS. max |phi| = largest correlation between true and permuted training-verb "
              "classes over all permutations and folds.", ""]
        L += table(S, ["model", "site", "depth", "role", "n_perm", "folds_per_perm", "balanced_mean", "balanced_p95",
                       "balanced_max", "balanced_fold_max", "max_abs_phi", "unconstrained_mean",
                       "unconstrained_fold_max", "trained_das", "random_normmatched"],
                   ["model", "site", "depth", "role", "perms", "folds / perm", "balanced mean", "balanced p95",
                    "balanced max", "balanced fold max", "max abs phi", "unconstrained mean", "unconstrained fold max",
                    "trained DAS", "random (norm-matched)"]) + [""]
    Path(a.report).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gemma", default="results/gemma/das/transfer")
    ap.add_argument("--pythia-passive", default="results/das_round2/passive_test,results/das_round2/passive_test_fill")
    ap.add_argument("--pythia-inrange", default="results/round4/inrange")
    ap.add_argument("--pythia-final", default="results/das_round2")
    ap.add_argument("--shb-root", default="results/gemma/das/shuffle_balanced")
    ap.add_argument("--csv-dir", default="results/gemma/das/exploratory")
    ap.add_argument("--report", default="reports/gemma/exploratory_tables.md")
    main(ap.parse_args())
