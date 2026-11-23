#!/usr/bin/env python3
"""Report for run_das_stability_ext.py + run_passive_test.py (rules: reports/das_stability_ext_rules.md, with its
Nov 23 amendment).

Per model and site: median |cos| with v for R, S1-S5, P, P2, P3, L; held-out active IIA; D_bad(by) with
analyze_passive_test's estimator (stored / S1-S5 / P on one set of bootstrap draws) on two populations:
  clean  primary (plain) minus pairs sharing a verb with P's training verbs (the reading)
  full   primary (plain), as Nov 18
Readings for P: direction (against the S1-S5 range) and function (ratio CI against [0.8, 1.25]); the item-14
three-level label is also computed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_passive_test as apt
from analyze_das_expanded_stability import RATIO_BAND, d_bad

S_CONDS = ("S1", "S2", "S3", "S4", "S5")
P_CONDS = ("P", "P2", "P3")


def bar_ok(ck, ref):
    """Amendment 2 recompute bar (das_expanded_stability_rules.md)."""
    med_bar = max(1e-3, 1.5 * ref.get("heldout_recompute_median_abs_dM", 0.0))
    return (ck.get("self_patch_max_abs_dM", 1) == 0.0 and ck.get("heldout_recompute_median_abs_dM", 1) <= med_bar
            and ck.get("heldout_recompute_max_abs_dM", 1) <= 0.5)


def overlap_mask(pr, verbs):
    return pr.good_lemma.isin(verbs).to_numpy() | pr.bad_lemma.isin(verbs).to_numpy()


def function_label(r):
    if RATIO_BAND[0] <= r["lo95"] and r["hi95"] <= RATIO_BAND[1]:
        return "equivalent"
    if r["lo95"] > RATIO_BAND[1]:
        return "larger"
    if r["hi95"] < RATIO_BAND[0]:
        return "smaller"
    return "unresolved"


def item14_label(gap, r):
    inside = RATIO_BAND[0] <= r["lo95"] and r["hi95"] <= RATIO_BAND[1]
    outside = r["hi95"] < RATIO_BAND[0] or r["lo95"] > RATIO_BAND[1]
    return "stable" if gap <= 0.05 and inside else "changed" if outside or gap > 0.15 else "partly stable"


def run(a):
    res = Path(a.results)
    checks = json.loads((res / "checks.json").read_text())
    C = pd.read_csv(res / "cosines.csv")
    A = pd.read_csv(res / "active_summary.csv")
    items = pd.read_csv(a.items)
    plan = pd.read_csv(a.plan, dtype={c: "category" for c in ("item_id", "side", "cond", "donor_pair")})
    pr, pops = apt.populations(items)
    Pp = pd.read_csv(res / "pairs_P.csv")
    pverbs = set(Pp[Pp.source == "expanded"].trans) | set(Pp[Pp.source == "expanded"].intrans)
    prim = pops["primary (plain)"].to_numpy()
    ov = overlap_mask(pr, pverbs)
    pixs = {"clean": np.flatnonzero(prim & ~ov), "full": np.flatnonzero(prim)}
    excluded = sorted(pr.index[prim & ov])
    pair_ix = {p: i for i, p in enumerate(pr.index)}
    ctx_ix = {c: i for i, c in enumerate(sorted(items.context_id.unique()))}
    das_q = sorted(plan[plan.cond == "T"].donor_pair.dropna().astype(str).unique())
    stored_meta = json.loads((Path(a.stored_passive) / "run_meta.json").read_text())["checks"]
    stored_sum = pd.read_csv(Path(a.stored_passive) / "transfer_summary.csv")
    problems, rows, readings = [], [], {}
    for k, v in checks.items():
        if k.startswith("stored_recompute_site") and not v["pass"]:
            problems.append(f"{k}: {v}")
    for s in sorted(C.site.unique()):
        if checks.get(f"R_median_abs_cos_site{s}", 0) < 0.99:
            problems.append(f"R does not reproduce v at site {s}")
        bt = apt.Boot(items.context_id.nunique(), a.n_boot, a.seed)
        wps = {p: bt.pair_weights(pr.band.to_numpy()[ix]) for p, ix in pixs.items()}
        D = {p: {} for p in pixs}
        for c in ("stored", *S_CONDS, "P"):
            pdir = Path(a.stored_passive) if c == "stored" else res / f"passive_{c}"
            nat = pd.read_parquet(pdir / "natural.parquet")
            Ap = d_bad(pdir, s, plan, items, nat, bt, pair_ix, ctx_ix, das_q)
            for p, ix in pixs.items():
                D[p][c] = apt.summarize(Ap, wps[p], ix)[:, 0]
            if c != "stored":
                ck = json.loads((pdir / "run_meta.json").read_text())["checks"][str(s)]
                if not bar_ok(ck, stored_meta[str(s)]):
                    problems.append(f"passive checks for {c} at site {s}: {ck}")
        st = stored_sum[(stored_sum.site == s) & (stored_sum.population == "primary (plain)") &
                        (stored_sum.band == "all") & (stored_sum.condition == "D_bad") & (stored_sum.readout == "by")]
        dref = abs(float(st.est.iloc[0]) - float(D["full"]["stored"][0]))
        if dref > 1e-4:
            problems.append(f"stored D_bad(by) at site {s} differs from transfer_summary by {dref:.2g}")
        cs = C[C.site == s].groupby("cond").abs_cos
        med = cs.median()
        ratio = {p: {c: apt.ci(D[p][c] / D[p]["stored"]) for c in (*S_CONDS, "P")} for p in pixs}
        sc = np.array([med[c] for c in S_CONDS])
        sr = np.array([ratio["clean"][c]["est"] for c in S_CONDS])
        pc = np.array([med[c] for c in P_CONDS if c in med])
        rP = ratio["clean"]["P"]
        gap = float(sc.mean() - med["P"])
        readings[int(s)] = {
            "direction": "inside seed range" if sc.min() <= med["P"] <= sc.max() else
                         ("above seed range" if med["P"] > sc.max() else "below seed range"),
            "function": function_label(rP), "ratio_ci_excludes_1": bool(rP["lo95"] > 1 or rP["hi95"] < 1),
            "item14_label": item14_label(gap, rP), "item14_label_full": item14_label(gap, ratio["full"]["P"]),
            "median_abs_cos": med.to_dict(), "seed_range_cos": [float(sc.min()), float(sc.max())],
            "S_bar_cos": float(sc.mean()), "P_seed_range_cos": [float(pc.min()), float(pc.max())],
            "L_cos": float(med.get("L", np.nan)), "P_cos_gap_in_seed_sd": gap / float(sc.std(ddof=1)),
            "seed_range_ratio": [float(sr.min()), float(sr.max())],
            "P_ratio_in_seed_range": bool(sr.min() <= rP["est"] <= sr.max()),
            "D": {p: {c: apt.ci(D[p][c]) for c in D[p]} for p in pixs}, "ratio": ratio}
        act = A[A.site == s].groupby("cond")[["iia_cross", "pc_frac_median"]].mean()
        for c in ("stored", "R", *S_CONDS, *P_CONDS, "L"):
            if c not in act.index:
                continue
            row = {"site": s, "cond": c, "median_abs_cos": med.get(c, np.nan), "min_abs_cos": cs.min().get(c, np.nan),
                   "iia_cross": act.loc[c, "iia_cross"], "pc_frac_median": act.loc[c, "pc_frac_median"]}
            for p in pixs:
                if c in D[p]:
                    row.update({f"D_{p}_{k}": v for k, v in apt.ci(D[p][c]).items()})
                if c in ratio[p]:
                    row.update({f"ratio_{p}_{k}": v for k, v in ratio[p][c].items()})
            rows.append(row)
    S = pd.DataFrame(rows)
    S.to_csv(res / "summary.csv", index=False)
    funcs = [r["function"] for r in readings.values()]
    dirs = [r["direction"] for r in readings.values()]
    overall = {"valid": not problems, "function": sorted(set(funcs)), "direction": sorted(set(dirs)),
               "item14_label": sorted(set(r["item14_label"] for r in readings.values()))}
    (res / "readings.json").write_text(json.dumps({"overall": overall, "sites": readings, "problems": problems,
                                                   "excluded_overlap_pairs": excluded, "checks": checks},
                                                  indent=1, default=float) + "\n")
    f3 = lambda d: f"{d['est']:.2f} [{d['lo95']:.2f}, {d['hi95']:.2f}]"
    L = [f"# DAS stability, extended: {a.model_name}", "",
         "Rules: `reports/das_stability_ext_rules.md` (with its Nov 23 amendment; recompute bar: Amendment 2 of "
         "`reports/pythia/das_expanded_stability_rules.md`). Scripts: `scripts/run_das_stability_ext.py`, "
         "`scripts/run_passive_test.py`, `scripts/analyze_das_stability_ext.py`.", "",
         f"**{'Valid' if not problems else 'INVALID'}.** Function (P vs stored, clean population): "
         f"{', '.join(funcs)}. Direction (P vs the seed range): {', '.join(dirs)}."
         + (f" Problems: {'; '.join(problems)}" if problems else ""), "",
         f"Clean population: primary (plain) minus {len(excluded)} pairs sharing a verb with P's training pairs "
         f"({', '.join(excluded) or 'none'}).", ""]
    for s, r in readings.items():
        m = r["median_abs_cos"]
        L += [f"- **Site {s}: function {r['function']}, direction {r['direction']}** (item-14 label "
              f"{r['item14_label']}). Median |cos| with v: P {m['P']:.3f} (P seeds {r['P_seed_range_cos'][0]:.3f}–"
              f"{r['P_seed_range_cos'][1]:.3f}); stored-pair seeds {r['seed_range_cos'][0]:.3f}–"
              f"{r['seed_range_cos'][1]:.3f}; L (stored pairs, P's update count) {r['L_cos']:.3f}; R {m['R']:.3f}. "
              f"D_bad(by) ratio P / stored: clean {f3(r['ratio']['clean']['P'])}, full {f3(r['ratio']['full']['P'])}; "
              f"seed range (clean) {r['seed_range_ratio'][0]:.2f}–{r['seed_range_ratio'][1]:.2f}."]
    L += ["", "| site | condition | median \\|cos\\| | min \\|cos\\| | held-out IIA (cross) | median pc frac | "
          "D_bad(by), clean | ratio, clean | ratio, full |", "|---:|---|---|---|---|---|---|---|---|"]
    for _, q in S.iterrows():
        cell = lambda k: (f"{q[k + '_est']:.2f} [{q[k + '_lo95']:.2f}, {q[k + '_hi95']:.2f}]"
                          if pd.notna(q.get(k + "_est")) else "—")
        L.append(f"| {q.site} | {q.cond} | {q.median_abs_cos:.3f} | {q.min_abs_cos:.3f} | {q.iia_cross:.3f} | "
                 f"{q.pc_frac_median:.2f} | {cell('D_clean')} | {cell('ratio_clean')} | {cell('ratio_full')} |")
    L += ["", "## Checks", "", "```", json.dumps(checks, indent=1, default=float), "```", ""]
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text("\n".join(L))
    print("\n".join(L[:16]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--stored-passive", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
