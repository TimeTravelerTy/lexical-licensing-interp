#!/usr/bin/env python3
"""Declared readings of the u_md transfer runs (reports/umd_transfer_rules.md), one model per call.

Inputs (all stored by the runs):
  {root}/umd/constructions/constructions_summary.csv   u_md, classes against v's delta_s (primary margin)
  {root}/constructions/constructions_summary.csv       v (or --v-summary)
  {draws_umd}, {draws_v}                               D_bad bootstrap draws (analyze_constructions.py --draws),
                                                       same seed and plan, so the u_md / v ratio is paired
  u_md active effect per site: split edit summary_active.csv (condition umd) or the Pythia direction check
  {root}/umd/fillergap/c8_decisions.json, {root}/umd/inrange report
Readings:
  T2-umd   OR and TC D_bad(O) RISE at both rule sites, primary margin
  own-effect margin: delta_u = 0.2 x u_md's active D(O); a RISE counts only if it holds with delta_u from the
           active hi95 (amendment 2026-11-17); point-estimate, 0.15 and 0.25 multipliers as sensitivity
  ratio    u_md / v D_bad(O), paired bootstrap
  T3-umd   in-range filler-gap 'licensing-like crossover' at a declared site (emb or mat)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_passive_test import classify  # noqa: E402

KEYS = ["est", "lo95", "hi95", "lo90", "hi90"]


def umd_active(model, root_repo):
    if model == "pythia":
        a = pd.read_csv(root_repo / "results/pythia/das_direction_check/summary_active_patching.csv")
        a = a[a.direction == "u_md"]
    else:
        a = pd.read_csv(root_repo / f"results/{model}/split_edit/summary_active.csv")
        a = a[a.condition == "umd"]
    return a.set_index("site")[["dO_est", "dO_lo95", "dO_hi95"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=("pythia", "gemma", "olmo"))
    ap.add_argument("--root", required=True)
    ap.add_argument("--v-summary", default="")
    ap.add_argument("--draws-umd", required=True)
    ap.add_argument("--draws-v", required=True)
    ap.add_argument("--rule-sites", required=True)
    ap.add_argument("--decl", required=True, help="declared T3 sites")
    ap.add_argument("--inrange-report", default="")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = Path(a.root)
    rs = [int(x) for x in a.rule_sites.split(",")]
    decl = [int(x) for x in a.decl.split(",")]
    U = pd.read_csv(root / "umd/constructions/constructions_summary.csv")
    V = pd.read_csv(a.v_summary or root / "constructions/constructions_summary.csv")
    du, dv = np.load(a.draws_umd), np.load(a.draws_v)
    act = umd_active(a.model, repo)
    sites = sorted(U.site.unique())
    rows = []
    for cons in ("OR", "TC"):
        for s in sites:
            u = U[(U.construction == cons) & (U.site == s) & (U.quantity == "D_bad") & (U.readout == "O")].iloc[0]
            v = V[(V.construction == cons) & (V.site == s) & (V.quantity == "D_bad") & (V.readout == "O")]
            v = v.iloc[0] if len(v) else None
            ci = {k: float(u[k]) for k in KEYS}
            r = {"construction": cons, "site": int(s), "delta_s": float(u.delta), **{f"umd_{k}": ci[k] for k in KEYS},
                 "class_primary": u["class"]}
            if v is not None:
                r.update({"v_est": float(v.est), "v_lo95": float(v.lo95), "v_hi95": float(v.hi95), "class_v": v["class"]})
            key = f"{cons}_site{s}_D_bad"
            if key in du.files and key in dv.files:
                x, y = du[key][:, 0].astype(np.float64), dv[key][:, 0].astype(np.float64)
                with np.errstate(divide="ignore", invalid="ignore"):
                    q = x / y
                r.update({"ratio_est": float(q[0]), "ratio_lo95": float(np.nanpercentile(q[1:], 2.5)),
                          "ratio_hi95": float(np.nanpercentile(q[1:], 97.5))})
                # check: the v draws' point estimate is the stored v estimate
                if v is not None:
                    r["v_draws_vs_stored"] = float(abs(y[0] - float(v.est)))
                r["umd_draws_vs_summary"] = float(abs(x[0] - ci["est"]))
            if s in act.index:
                e, hi = float(act.loc[s, "dO_est"]), float(act.loc[s, "dO_hi95"])
                r.update({"umd_active_dO": e, "umd_active_dO_hi95": hi,
                          "class_own_hi95": classify(ci, 0.2 * hi), "class_own_point": classify(ci, 0.2 * e),
                          "class_own_0.15": classify(ci, 0.15 * e), "class_own_0.25": classify(ci, 0.25 * e)})
            rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(root / "umd/t2_readings.csv", index=False)

    def t2(col):
        sub = T[T.site.isin(rs)]
        if col not in sub or len(sub) != 2 * len(rs):
            return None
        return bool((sub[col] == "RISE").all())
    dec = json.loads((root / "umd/fillergap/c8_decisions.json").read_text())
    t3 = {f"{e}|{s}": dec.get(f"{e}|in range|{s}") for e in ("emb", "mat") for s in decl}
    out = {"T2_umd_primary": t2("class_primary"), "T2_umd_own_hi95": t2("class_own_hi95"),
           "T2_umd_own_point": t2("class_own_point"), "T2_v": t2("class_v"),
           "T3_umd": {"in_range": t3, "pass": any(v == "licensing-like crossover" for v in t3.values())},
           "checks": {"max_v_draws_vs_stored": float(T.get("v_draws_vs_stored", pd.Series([np.nan])).max()),
                      "max_umd_draws_vs_summary": float(T.get("umd_draws_vs_summary", pd.Series([np.nan])).max())}}
    Path(a.out_json).write_text(json.dumps(out, indent=1) + "\n")

    f = lambda r, p: f"{r[p + 'est']:+.2f} [{r[p + 'lo95']:+.2f}, {r[p + 'hi95']:+.2f}]"
    L = [f"# u_md transfer readings ({a.model})", "",
         "Rules: `reports/umd_transfer_rules.md` (declared 2026-11-17 before running). Generated by "
         "`scripts/umd_transfer_readings.py`. D_bad(O): T − I donors on bad bases, nats, 95% CI. Primary class against "
         "δ_s (0.2 × v's active O effect); own-effect class against δ_u = 0.2 × u_md's active D(O) at its upper 95% "
         "bound (sensitivity: point estimate, multipliers 0.15 / 0.25). Ratio: u_md / v, paired bootstrap.", "",
         f"- **T2-umd (primary margin, rule sites {a.rule_sites}): {out['T2_umd_primary']}**; own-effect (hi95): "
         f"{out['T2_umd_own_hi95']}; v: {out['T2_v']}",
         f"- **T3-umd (in-range licensing-like at a declared site {a.decl}): {out['T3_umd']['pass']}**",
         f"- Checks: v draws reproduce the stored v estimate (max {out['checks']['max_v_draws_vs_stored']:.2g}); "
         f"u_md draws match the summary (max {out['checks']['max_umd_draws_vs_summary']:.2g}).", "",
         "| Cons | Site | δ_s | D(O) v | D(O) u_md | primary | δ_u (hi95) | own hi95 | own point / 0.15 / 0.25 | u_md / v |",
         "|---|---:|---:|---|---|---|---:|---|---|---|"]
    for _, r in T.iterrows():
        own = (f"{0.2 * r['umd_active_dO_hi95']:.2f} | {r['class_own_hi95']} | {r['class_own_point']} / "
               f"{r['class_own_0.15']} / {r['class_own_0.25']}") if "class_own_hi95" in r and pd.notna(
            r.get("class_own_hi95")) else "n/a | n/a | n/a"
        vv = f"{f(r, 'v_')} {r['class_v']}" if pd.notna(r.get("v_est")) else "n/a"
        ratio = (f"{r['ratio_est']:.2f} [{r['ratio_lo95']:.2f}, {r['ratio_hi95']:.2f}]"
                 if pd.notna(r.get("ratio_est")) else "n/a")
        L.append(f"| {r.construction} | {r.site} | {r.delta_s:.2f} | {vv} | {f(r, 'umd_')} | {r.class_primary} | {own} | "
                 f"{ratio} |")
    L += ["", "## Filler-gap, in range (primary population)", "", "| Exp | Site | u_md reading |", "|---|---:|---|"]
    L += [f"| {k.split('|')[0]} | {k.split('|')[1]} | {v} |" for k, v in t3.items()]
    if a.inrange_report and Path(a.inrange_report).exists():
        L += ["", "## In-range passive / OR / TC profile (from the u_md in-range report)", ""]
        L += [x for x in Path(a.inrange_report).read_text().splitlines() if x.startswith("- **")]
    Path(a.report).write_text("\n".join(L) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
