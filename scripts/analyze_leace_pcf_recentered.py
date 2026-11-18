#!/usr/bin/env python3
"""Report for run_leace_pcf_recentered.py: the verb-held-out cross-frame pairing with a matched reference.

A-cf (c) from the cross-frame run against P-cf as stored (u) and re-centered on the evaluated fold's passive mean (c),
on the same task-1 item table and bootstrap draws (1000, seed 17) as analyze_leace_crossframe.py. The declared reading
(reports/pythia/leace_crossframe_rules.md, follow-up section) is the amended A-cf vs P-cf (c) rule with P-cf (c) as
the reference; the permutation gate and flags come from the cross-frame readings.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_das_prep import Boot
from analyze_leace_crossframe import ci, item_table, sha

N_SITES, NEC_TOP = 24, 12
REPRO_TOL = 1e-4


def run(a):
    res, cf, out = Path(a.results), Path(a.crossframe), Path(a.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    X = pd.read_csv(cf / "items_passive.csv.gz")
    Y = pd.read_csv(res / "items_passive.csv.gz")
    checks = json.loads((res / "checks.json").read_text())
    meta = json.loads((res / "meta.json").read_text())
    R0 = json.loads((cf / "readings.json").read_text())
    smoke = bool(meta.get("smoke"))
    problems = []
    if not (X.prompt.to_numpy() == Y.prompt.to_numpy()).all():
        raise SystemExit("prompt order differs from the cross-frame run")
    nat_d = float(np.abs(X.lp_by__natural - Y.lp_by__natural).max())
    if nat_d >= 1e-3:
        problems.append(f"natural log P(by) differs from the cross-frame run by {nat_d}")
    if not checks.get("stored_repro_max_abs", np.inf) < REPRO_TOL:
        problems.append(f"stored P-cf reproduction {checks.get('stored_repro_max_abs')}")
    if not smoke and not checks.get("folds_equal_stored"):
        problems.append("folds not checked against the stored run")
    new = [c for c in Y.columns if c.startswith(("lp_by__Pcf_", "kl__Pcf_"))]
    X = pd.concat([X, Y[new]], axis=1)
    conds = ["natural"] + [f"cf_c_{t}" for t in [f"single_{s}" for s in range(N_SITES)] +
                           [f"scrub_0_{k}" for k in range(N_SITES)]]
    conds += [c[len("lp_by__"):] for c in new if c.startswith("lp_by__")]
    it = item_table(X, "lp_by__", conds)
    boot = Boot(it, a.n_boot, np.random.default_rng(a.seed))
    allm = np.ones(len(it), bool)
    nat = boot.means(it["natural"].to_numpy(), allm)
    share = {c: boot.means(it[c].to_numpy(), allm) / nat for c in conds if c != "natural"}

    rows = []
    for kind in ("scrub", "single"):
        for k in range(N_SITES):
            t = f"single_{k}" if kind == "single" else f"scrub_0_{k}"
            r = {"kind": kind, "site": k}
            for lab, c in (("A-cf_c", f"cf_c_{t}"), ("P-cf_stored", f"Pcf_u_{t}"), ("P-cf_c", f"Pcf_c_{t}")):
                e, lo, hi = ci(share[c])
                r.update({f"{lab}_share": e, f"{lab}_share_lo": lo, f"{lab}_share_hi": hi})
            for lab, c in (("stored", f"Pcf_u_{t}"), ("matched", f"Pcf_c_{t}")):
                with np.errstate(divide="ignore", invalid="ignore"):
                    e, lo, hi = ci((1 - share[f"cf_c_{t}"]) / (1 - share[c]))
                r.update({f"rho_{lab}": e, f"rho_{lab}_lo": lo, f"rho_{lab}_hi": hi})
            rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(out / "leace_pcf_recentered_summary.csv", index=False)

    # declared reading: amended A-cf vs P-cf (c) rule, P-cf (c) as the reference
    r = S[(S.kind == "scrub") & (S.site == NEC_TOP)].iloc[0]
    r0 = R0.get("A-cf vs P-cf (c)", {})
    a_repro = abs(r["A-cf_c_share"] - r0["share_A"][0]) if r0 else None
    if r0 and a_repro > 1e-9:
        problems.append(f"A-cf (c) share differs from the cross-frame readings by {a_repro}")
    pmin = R0["perm"]["c_scrub_12"]["min"]
    rem_lo, rem_hi = 1 - r["A-cf_c_share_hi"], 1 - r["A-cf_c_share_lo"]
    below = r["A-cf_c_share_hi"] < pmin
    rho, rlo, rhi = r.rho_matched, r.rho_matched_lo, r.rho_matched_hi
    if problems or 1 - r["P-cf_c_share_hi"] <= 0 or not np.isfinite(rho):
        lab = "invalid"
    elif rem_hi < 0:
        lab = "gap enhancement"
    elif rem_lo > 0 and below:
        lab = "majority transfer" if rlo >= 0.5 else "minority transfer" if rhi < 0.5 else \
            "positive transfer, share unresolved"
    elif rhi < 0.2:
        lab = "no meaningful transfer"
    else:
        lab = "inconclusive"
    reading = {"reading": lab, "flags": r0.get("flags", []), "reading_stored_ref": r0.get("reading"),
               "share_A": [r["A-cf_c_share"], r["A-cf_c_share_lo"], r["A-cf_c_share_hi"]],
               "share_P_matched": [r["P-cf_c_share"], r["P-cf_c_share_lo"], r["P-cf_c_share_hi"]],
               "share_P_stored": [r["P-cf_stored_share"], r["P-cf_stored_share_lo"], r["P-cf_stored_share_hi"]],
               "rho_matched": [rho, rlo, rhi], "rho_stored": [r.rho_stored, r.rho_stored_lo, r.rho_stored_hi],
               "perm_min": pmin, "below_perm_min": bool(below), "problems": problems,
               "checks": {**checks, "natural_vs_crossframe_max_abs": nat_d, "A_share_vs_readings": a_repro},
               "outputs_sha256": {"items_passive.csv.gz": sha(res / "items_passive.csv.gz")}}
    Path(a.readings).write_text(json.dumps(reading, indent=1, default=float) + "\n")

    f3 = lambda t: f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"
    L = ["# Cross-frame LEACE, verb-held-out pairing with a matched reference (Pythia-1.4B)", "",
         "Follow-up in `reports/pythia/leace_crossframe_rules.md` (declared 2026-11-18 after the full run). The stored "
         "cross-fitted passive erasers (P-cf) keep the mean of their training folds' passives; A-cf (c) is centered on "
         "the evaluated fold's passive mean. Here P-cf is rebuilt from the stored erasers and evaluated as stored and "
         "re-centered the same way as A-cf (c). Scripts: `scripts/run_leace_pcf_recentered.py`, "
         f"`scripts/analyze_leace_pcf_recentered.py`. Same items and bootstrap draws ({a.n_boot}, seed {a.seed}). "
         f"Run: {meta['minutes']:.0f} min.", ""]
    if problems:
        L += ["**Problems (reading invalid):** " + "; ".join(problems), ""]
    L += ["## Reading (scrub 0–12)", "",
          f"- **A-cf vs P-cf, matched (c): {lab}**"
          f"{' (' + '; '.join(reading['flags']) + ')' if reading['flags'] else ''}. Share remaining A-cf "
          f"{f3(reading['share_A'])}; P-cf re-centered {f3(reading['share_P_matched'])}, as stored "
          f"{f3(reading['share_P_stored'])}. ρ matched {f3(reading['rho_matched'])}, with the stored reference "
          f"{f3(reading['rho_stored'])}. Below the permutation minimum ({pmin:.2f}): {below}. "
          f"Reading with the stored reference: {reading['reading_stored_ref']}.", ""]
    for kind, title in (("scrub", "Scrub prefixes 0..k"), ("single", "Single site")):
        L += [f"## {title}", "", "| k | A-cf (c) | P-cf stored | P-cf re-centered | ρ stored | ρ matched |",
              "|---:|---|---|---|---|---|"]
        for _, q in S[S.kind == kind].iterrows():
            g = lambda p: f"{q[p + '_share']:.2f} [{q[p + '_share_lo']:.2f}, {q[p + '_share_hi']:.2f}]"
            h = lambda p: f"{q['rho_' + p]:.2f} [{q['rho_' + p + '_lo']:.2f}, {q['rho_' + p + '_hi']:.2f}]"
            note = " (next to readout)" if q.site > NEC_TOP else ""
            L.append(f"| {q.site}{note} | {g('A-cf_c')} | {g('P-cf_stored')} | {g('P-cf_c')} | {h('stored')} | "
                     f"{h('matched')} |")
        L.append("")
    L += ["## Checks", "",
          f"- Problems: {problems if problems else 'none'}.",
          f"- As-stored erasers reproduce the stored per-prompt log P(by): max |Δ| "
          f"{checks.get('stored_repro_max_abs', float('nan')):.2g} over {checks.get('stored_repro_n')} columns "
          f"(tolerance {REPRO_TOL}).",
          f"- Folds equal the stored run: {checks.get('folds_equal_stored')}. Natural log P(by) vs the cross-frame run: "
          f"max |Δ| {nat_d:.2g}. A-cf (c) scrub 0–12 share vs the cross-frame readings: |Δ| {a_repro}.",
          f"- Stored erasers sha256 {meta['ref_erasers_sha256'][:12]}…, stored items {meta['ref_items_sha256'][:12]}…; "
          f"output items {reading['outputs_sha256']['items_passive.csv.gz'][:12]}….", ""]
    (out / "leace_pcf_recentered.md").write_text("\n".join(L))
    print("\n".join(L[:10]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/pythia/leace_pcf_recentered")
    ap.add_argument("--crossframe", default="results/pythia/leace_crossframe")
    ap.add_argument("--report-dir", default="reports/pythia")
    ap.add_argument("--readings", default="results/pythia/leace_pcf_recentered/readings.json")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
