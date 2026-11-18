#!/usr/bin/env python3
"""Report for run_das_expanded_stability.py + run_passive_test.py (rules: reports/pythia/das_expanded_stability_rules.md).

Passive transfer: D_bad(by) on the primary (plain) population, all bands, with analyze_passive_test's estimator; stored,
E and S share one set of bootstrap draws per site (contexts, donor pairs, base pairs; 2000, seed 17), so ratios and
differences are paired.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_passive_test as apt

RATIO_BAND = (0.8, 1.25)


def d_bad(pdir, site, plan, items, nat, bt, pair_ix, ctx_ix, das_q):
    pat = pd.read_parquet(Path(pdir) / f"patches_site{site}.parquet")
    assert (pat.row.to_numpy() == plan.row.to_numpy()).all()
    df = plan[["item_id", "side", "cond", "base", "donor_pair"]].copy()
    df["by"] = pat["by"].to_numpy() - nat.set_index("pid").loc[plan.base.to_numpy(), "by"].to_numpy()
    df = df.join(items.set_index("item_id")[["pair_id", "context_id"]], on="item_id")
    d = df[(df.side == "bad") & df.cond.isin(["T", "I"])].assign(dq=lambda x: x.donor_pair)
    qix = {q: i for i, q in enumerate(das_q)}
    vT, mT = apt.cells(d[d.cond == "T"], pair_ix, ctx_ix, qix, len(das_q), ["by"])
    vI, mI = apt.cells(d[d.cond == "I"], pair_ix, ctx_ix, qix, len(das_q), ["by"])
    assert (mT == mI).all()
    return apt.pair_values(vT - vI, mT, bt.wc, bt.donor_weights("das", len(das_q)))


def run(a):
    res, out = Path(a.results), Path(a.report_dir)
    checks = json.loads((res / "checks.json").read_text())
    C = pd.read_csv(res / "cosines.csv")
    A = pd.read_csv(res / "active_summary.csv")
    items = pd.read_csv(a.items)
    plan = pd.read_csv(a.plan, dtype={c: "category" for c in ("item_id", "side", "cond", "donor_pair")})
    pr, pops = apt.populations(items)
    pix = np.flatnonzero(pops["primary (plain)"].to_numpy())
    pair_ix = {p: i for i, p in enumerate(pr.index)}
    ctx_ix = {c: i for i, c in enumerate(sorted(items.context_id.unique()))}
    das_q = sorted(plan[plan.cond == "T"].donor_pair.dropna().astype(str).unique())
    stored_sum = pd.read_csv(Path(a.stored_passive) / "transfer_summary.csv")
    problems, rows, readings = [], [], {}
    sites = sorted(C.site.unique())
    for s in sites:
        bt = apt.Boot(items.context_id.nunique(), a.n_boot, a.seed)
        wp = bt.pair_weights(pr.band.to_numpy()[pix])
        D = {}
        for c, pdir in (("stored", a.stored_passive), ("E", res / "passive_E"), ("S", res / "passive_S")):
            nat = pd.read_parquet(Path(pdir) / "natural.parquet")
            D[c] = apt.summarize(d_bad(pdir, s, plan, items, nat, bt, pair_ix, ctx_ix, das_q), wp, pix)[:, 0]
            rm = Path(pdir) / "run_meta.json"
            if c != "stored" and rm.exists():
                ck = json.loads(rm.read_text())["checks"][str(s)]
                if ck.get("self_patch_max_abs_dM", 1) > 1e-3 or ck.get("heldout_recompute_max_abs_dM", 1) > 1e-2:
                    problems.append(f"passive-test checks failed for {c} at site {s}: {ck}")
        ref = stored_sum[(stored_sum.site == s) & (stored_sum.population == "primary (plain)") &
                         (stored_sum.band == "all") & (stored_sum.condition == "D_bad") & (stored_sum.readout == "by")]
        dref = abs(float(ref.est.iloc[0]) - float(D["stored"][0]))
        if dref > 1e-4:
            problems.append(f"stored D_bad(by) at site {s} differs from transfer_summary by {dref:.2g}")
        if checks.get(f"stored_heldout_recompute_max_abs_site{s}", 1) >= 1e-2:
            problems.append(f"stored held-out recompute at site {s}")
        r_ok = checks.get(f"R_median_abs_cos_site{s}", 0) >= 0.99
        cs = C[C.site == s].groupby("cond").abs_cos
        med = cs.median()
        ratio = {c: apt.ci(D[c] / D["stored"]) for c in ("E", "S")}
        diff = {c: apt.ci(D[c] - D["stored"]) for c in ("E", "S")}
        rE = ratio["E"]
        inside = RATIO_BAND[0] <= rE["lo95"] and rE["hi95"] <= RATIO_BAND[1]
        outside = rE["hi95"] < RATIO_BAND[0] or rE["lo95"] > RATIO_BAND[1]
        cos_gap = med["S"] - med["E"]
        lab = ("stable" if cos_gap <= 0.05 and inside else
               "changed" if outside or cos_gap > 0.15 else "partly stable")
        readings[int(s)] = {"reading": lab, "reference": "stored v" if r_ok else "R (reproduction below 0.99)",
                            "median_abs_cos": med.to_dict(), "D": {c: apt.ci(D[c]) for c in D},
                            "ratio": ratio, "diff": diff}
        act = A[A.site == s].groupby("cond")[["iia_cross", "pc_frac_median"]].mean()
        for c in ("stored", "R", "S", "E"):
            rows.append({"site": s, "cond": c, "median_abs_cos": med.get(c, np.nan),
                         "min_abs_cos": cs.min().get(c, np.nan), "iia_cross": act.loc[c, "iia_cross"],
                         "pc_frac_median": act.loc[c, "pc_frac_median"],
                         **({f"D_{k}": v for k, v in apt.ci(D[c]).items()} if c in D else {}),
                         **({f"ratio_{k}": v for k, v in ratio[c].items()} if c in ratio else {})})
    labs = [r["reading"] for r in readings.values()]
    overall = "invalid" if problems else ("stable" if all(l == "stable" for l in labs) else
                                          "changed" if "changed" in labs else "partly stable")
    S = pd.DataFrame(rows)
    S.to_csv(res / "summary.csv", index=False)
    (res / "readings.json").write_text(json.dumps({"overall": overall, "sites": readings, "problems": problems,
                                                   "checks": checks}, indent=1, default=float) + "\n")
    f3 = lambda d: f"{d['est']:.2f} [{d['lo95']:.2f}, {d['hi95']:.2f}]"
    L = ["# DAS stability under more training verbs (Pythia-1.4B)", "",
         "Rules: `reports/pythia/das_expanded_stability_rules.md`. Scripts: `scripts/run_das_expanded_stability.py`, "
         "`scripts/run_passive_test.py`, `scripts/analyze_das_expanded_stability.py`.", "",
         f"**Overall: {overall}.**" + (f" Problems: {'; '.join(problems)}" if problems else ""), ""]
    for s, r in readings.items():
        L += [f"- **Site {s}: {r['reading']}** (reference: {r['reference']}). Median |cos| with v: "
              + ", ".join(f"{c} {v:.3f}" for c, v in r["median_abs_cos"].items())
              + f". D_bad(by): stored {f3(r['D']['stored'])}, E {f3(r['D']['E'])}, S {f3(r['D']['S'])}. "
              f"E / stored {f3(r['ratio']['E'])}; S / stored {f3(r['ratio']['S'])}."]
    L += ["", "| site | condition | median \\|cos\\| | min \\|cos\\| | held-out IIA (cross) | median pc frac | "
          "D_bad(by) | ratio to stored |", "|---:|---|---|---|---|---|---|---|"]
    for _, q in S.iterrows():
        dcell = f"{q.D_est:.2f} [{q.D_lo95:.2f}, {q.D_hi95:.2f}]" if pd.notna(q.get("D_est")) else "—"
        rcell = f"{q.ratio_est:.2f} [{q.ratio_lo95:.2f}, {q.ratio_hi95:.2f}]" if pd.notna(q.get("ratio_est")) else "—"
        L.append(f"| {q.site} | {q.cond} | {q.median_abs_cos:.3f} | {q.min_abs_cos:.3f} | {q.iia_cross:.3f} | "
                 f"{q.pc_frac_median:.2f} | {dcell} | {rcell} |")
    L += ["", "## Checks", "", "```", json.dumps(checks, indent=1, default=float), "```", ""]
    (out / "das_expanded_stability.md").write_text("\n".join(L))
    print("\n".join(L[:12]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/pythia/das_expanded_stability")
    ap.add_argument("--stored-passive", default="results/das_round2/passive_test")
    ap.add_argument("--items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--plan", default="data/das_round2/passive_test/plan.csv.gz")
    ap.add_argument("--report-dir", default="reports/pythia")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
