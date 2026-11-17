#!/usr/bin/env python3
"""Report for run_leace_crossframe.py (rules: reports/pythia/leace_crossframe_rules.md).

Share of the natural passive *by* gap remaining under active-fit erasure (A-in, A-cf), next to the passive-fit
references recomputed from their stored per-prompt files (P-in: leace_insample, P-cf: leace_necessity). Same item
table and task-1 two-way cluster bootstrap as analyze_leace_necessity.py (1000 draws, seed 17), the same draws for
every condition, so removed(A) / removed(P) is paired. Permutation control, active-side check, eraser geometry,
passive class-mean difference after erasure. Writes the report, a summary csv and readings.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_das_prep import Boot

N_SITES, DAS_SITE, NEC_TOP = 24, 8, 12


def ci(d):
    return float(d[0]), *(float(x) for x in np.nanpercentile(d[1:], [2.5, 97.5]))


def fmt(t, k=2, sign=False):
    s = "+" if sign else ""
    return f"{t[0]:{s}.{k}f} [{t[1]:{s}.{k}f}, {t[2]:{s}.{k}f}]"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def item_table(P, prefix, conds):
    G, B = P[P.side == "good"].set_index("item"), P[P.side == "bad"].set_index("item")
    assert (G.index == B.index).all()
    it = G[["verb_pair", "verb_band", "context_id", "context_band"]].reset_index()
    for c in conds:
        it[c] = (G[f"{prefix}{c}"] - B[f"{prefix}{c}"]).to_numpy()
    return it


def run(a):
    res, out = Path(a.results), Path(a.report_dir)
    X = pd.read_csv(res / "items_passive.csv.gz")
    Xa = pd.read_csv(res / "items_active.csv.gz")
    checks = json.loads((res / "checks.json").read_text())
    meta = json.loads((res / "meta.json").read_text())
    diag = pd.read_csv(res / "fit_diagnostics.csv")
    md = pd.read_csv(res / "meandiff.csv")
    ref = {}
    for name, d in (("P-in", a.ref_in), ("P-cf", a.ref_cf)):
        if not d:  # smoke test: no passive references
            continue
        R = pd.read_csv(Path(d) / "items.csv.gz")
        assert (R[["item", "side", "prompt"]].to_numpy() == X[["item", "side", "prompt"]].to_numpy()).all(), name
        ref[name] = {"natural_max_abs": float((R.lp_by__natural - X.lp_by__natural).abs().max()),
                     "items_sha256": sha(Path(d) / "items.csv.gz")}
        assert ref[name]["natural_max_abs"] < 1e-3, ref[name]
        for c in R.columns:
            if c.startswith("lp_by__") and c != "lp_by__natural":
                cond = c[len("lp_by__"):].replace("scrub_all", f"scrub_0_{N_SITES - 1}")
                X[f"lp_by__{name}_{cond}"] = R[c].to_numpy()
                X[f"kl__{name}_{cond}"] = R[f"kl__{c[len('lp_by__'):]}"].to_numpy()
    conds = [c[len("lp_by__"):] for c in X.columns if c.startswith("lp_by__")]
    it = item_table(X, "lp_by__", conds)
    boot = Boot(it, a.n_boot, np.random.default_rng(a.seed))
    allm = np.ones(len(it), bool)
    nat = boot.means(it["natural"].to_numpy(), allm)
    share = {c: boot.means(it[c].to_numpy(), allm) / nat for c in conds if c != "natural"}
    kl = {c: float(X[f"kl__{c}"].mean()) for c in conds if c != "natural"}

    def S(fit, kind, k):
        return f"{fit}_{kind}_{k}" if kind == "single" else f"{fit}_scrub_0_{k}"

    rows = []
    for kind in ("single", "scrub"):
        for k in range(N_SITES):
            r = {"kind": kind, "site": k}
            for fit in ("A-in", "A-cf", "P-in", "P-cf"):
                key = S({"A-in": "in", "A-cf": "cf"}.get(fit, fit), kind, k)
                if key not in share:
                    continue
                e, lo, hi = ci(share[key])
                r.update({f"{fit}_share": e, f"{fit}_share_lo": lo, f"{fit}_share_hi": hi, f"{fit}_kl": kl[key]})
            for A, Pn in (("A-in", "P-in"), ("A-cf", "P-cf")):
                ka, kp = S(A[2:], kind, k), S(Pn, kind, k)
                if ka in share and kp in share:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        rho = (1 - share[ka]) / (1 - share[kp])
                    e, lo, hi = ci(rho)
                    r.update({f"rho_{A}": e, f"rho_{A}_lo": lo, f"rho_{A}_hi": hi})
            rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(out / "leace_crossframe_summary.csv", index=False)

    # permutation control (A-in fitting, permuted verb labels)
    perm = {}
    for kind, k in (("single", DAS_SITE), ("scrub", NEC_TOP)):
        ps = np.array([share[c][0] for c in share if c.startswith("perm") and c.endswith(
            f"_single_{k}" if kind == "single" else f"_scrub_0_{k}")])
        if len(ps):
            perm[f"{kind}_{k}"] = {"n": len(ps), "mean": float(ps.mean()), "p5": float(np.percentile(ps, 5)),
                                   "min": float(ps.min()), "max": float(ps.max())}

    # active-side check
    acond = [c[len("lp_obj__"):] for c in Xa.columns if c.startswith("lp_obj__")]
    ita = item_table(Xa, "lp_obj__", acond)
    assert (ita[["item", "verb_pair", "context_id"]].to_numpy() == it[["item", "verb_pair", "context_id"]].to_numpy()).all()
    nat_a = boot.means(ita["natural"].to_numpy(), allm)
    active = {"natural_gap": ci(nat_a), **{c: ci(boot.means(ita[c].to_numpy(), allm) / nat_a)
                                           for c in acond if c != "natural"}}

    # geometry: |cos| between active-fit and passive-fit single-site edit vectors
    geo = []
    E = np.load(res / "erasers.npz")
    for fit, refd in (("in", a.ref_in), ("cf", a.ref_cf)):
        Rp = Path(refd) / "erasers.npz"
        if not refd or not Rp.exists():
            continue
        Q = np.load(Rp)
        ea, ep = E[f"{fit}_single_edit"], Q["single_edit"]
        assert ea.shape == ep.shape, (fit, ea.shape, ep.shape)
        for s in range(N_SITES):
            c = [abs(float(ea[f, s] @ ep[f, s] / np.linalg.norm(ea[f, s]) / np.linalg.norm(ep[f, s])))
                 for f in range(ea.shape[0])]
            geo.append({"fitting": fit, "site": s, "abs_cos_mean": float(np.mean(c)), "abs_cos_min": float(np.min(c))})
    G = pd.DataFrame(geo)

    # readings (primary: scrub 0-12)
    def reading(A, Pn):
        r = T[(T.kind == "scrub") & (T.site == NEC_TOP)].iloc[0]
        removed_lo = 1 - r[f"{A}_share_hi"]
        excl0 = removed_lo > 0
        rho_lo = r[f"rho_{A}_lo"]
        if A == "A-in":
            p = perm.get(f"scrub_{NEC_TOP}")
            below = p is not None and r[f"{A}_share_hi"] < p["p5"]
        else:
            below = True  # no cross-fitted permutation control (declared)
        if excl0 and below and rho_lo >= 0.5:
            lab = "shared"
        elif excl0 and below:
            lab = "partly shared"
        else:
            lab = "frame-specific"
        return {"reading": lab, "share_A": [r[f"{A}_share"], r[f"{A}_share_lo"], r[f"{A}_share_hi"]],
                "share_P": [r[f"{Pn}_share"], r[f"{Pn}_share_lo"], r[f"{Pn}_share_hi"]],
                "rho": [r[f"rho_{A}"], r[f"rho_{A}_lo"], r[f"rho_{A}_hi"]], "removed_ci_excludes_0": bool(excl0),
                "well_below_perm": bool(below) if A == "A-in" else "n/a"}
    # reproduction of the stored necessity report's passive-fit scrub shares (same items and bootstrap)
    repro = None
    cum = Path(a.stored_cumulative)
    if ref and cum.exists():
        C = pd.read_csv(cum)
        dev = []
        for _, c in C.iterrows():
            fit = "P-in" if c.run == "in-sample" else "P-cf"
            key = f"{fit}_scrub_0_{int(c.k)}"
            if key in share:
                e, lo, hi = ci(share[key])
                dev.append(max(abs(e - c.share_remaining), abs(lo - c.share_remaining_lo),
                               abs(hi - c.share_remaining_hi)))
        repro = {"n": len(dev), "max_abs": float(max(dev)) if dev else None}
    readings = {"natural_gap": ci(nat), "perm": perm, "active_side": active,
                "checks": {**checks, "refs": ref, "stored_share_repro": repro}}
    if ref:
        readings.update({"A-in vs P-in": reading("A-in", "P-in"), "A-cf vs P-cf": reading("A-cf", "P-cf")})
    Path(a.readings).parent.mkdir(parents=True, exist_ok=True)
    Path(a.readings).write_text(json.dumps(readings, indent=1, default=float) + "\n")

    L = ["# Cross-frame LEACE (Pythia-1.4B)", "",
         "Rules: `reports/pythia/leace_crossframe_rules.md` (declared 2026-11-17 before running). Scripts: "
         "`scripts/run_leace_crossframe.py`, `scripts/analyze_leace_crossframe.py`. Erasers fit on **active** "
         "verb-position states (\"The AGENT VERB-ed\", same verbs and contexts), applied to the passive items of "
         "`leace_necessity.md`. A-in: fit on all verbs (reference P-in, the in-sample passive fit). A-cf: fold f's "
         "eraser fit on the other folds' actives (reference P-cf, the cross-fitted passive fit). Share = erased / "
         "natural passive *by* gap; removed = 1 − share; ρ = removed(A) / removed(P), paired bootstrap "
         f"({a.n_boot} draws, seed {a.seed}). Run: {meta['minutes']:.0f} min.", "",
         f"Natural *by* gap {fmt(ci(nat), sign=True)} nats.", "",
         "## Readings (scrub 0–12)", ""]
    for k, r in readings.items():
        if k.startswith("A-"):
            L.append(f"- **{k}: {r['reading']}.** Share remaining A {r['share_A'][0]:.2f} [{r['share_A'][1]:.2f}, "
                     f"{r['share_A'][2]:.2f}], P {r['share_P'][0]:.2f} [{r['share_P'][1]:.2f}, {r['share_P'][2]:.2f}]; "
                     f"ρ {r['rho'][0]:.2f} [{r['rho'][1]:.2f}, {r['rho'][2]:.2f}]; removed CI excludes 0: "
                     f"{r['removed_ci_excludes_0']}; well below the permutation control: {r['well_below_perm']}.")
    L += ["", "## Permutation control (A-in fitting, permuted verb labels)", "",
          "| Condition | n | mean share | 5th pct | range |", "|---|---:|---:|---:|---|"]
    L += [f"| {k} | {v['n']} | {v['mean']:.2f} | {v['p5']:.2f} | {v['min']:.2f}–{v['max']:.2f} |" for k, v in perm.items()]
    L += ["", "## Active-side check (share of the natural active T − I log P(object start) gap remaining)", "",
          f"Natural active gap {fmt(active['natural_gap'], sign=True)} nats. "
          "A-cf erasers on each fold's held-out actives; A-in on the fitting actives.", ""]
    L += [f"- {c}: {fmt(v)}" for c, v in active.items() if c != "natural_gap"]
    for kind, title in (("scrub", "Scrub prefixes 0..k"), ("single", "Single site")):
        L += ["", f"## {title}", "", "| k | A-in | P-in | ρ in | A-cf | P-cf | ρ cf | KL A-in / A-cf |",
              "|---:|---|---|---|---|---|---|---|"]
        for _, r in T[T.kind == kind].iterrows():
            g = lambda f: (f"{r[f + '_share']:.2f} [{r[f + '_share_lo']:.2f}, {r[f + '_share_hi']:.2f}]"
                           if pd.notna(r.get(f + "_share")) else "")
            h = lambda A: (f"{r['rho_' + A]:.2f} [{r['rho_' + A + '_lo']:.2f}, {r['rho_' + A + '_hi']:.2f}]"
                           if pd.notna(r.get("rho_" + A)) else "")
            note = " (next to readout)" if r.site > NEC_TOP else ""
            L.append(f"| {r.site}{note} | {g('A-in')} | {g('P-in')} | {h('A-in')} | {g('A-cf')} | {g('P-cf')} | "
                     f"{h('A-cf')} | {r.get('A-in_kl', np.nan):.3f} / {r.get('A-cf_kl', np.nan):.3f} |")
    if len(G):
        L += ["", "## Geometry: |cos| of active-fit vs passive-fit single-site edit vectors (cf: mean / min over folds)",
              "", "| Site | in | cf |", "|---:|---:|---|"]
        for s in range(N_SITES):
            gi, gc = G[(G.fitting == "in") & (G.site == s)], G[(G.fitting == "cf") & (G.site == s)]
            L.append(f"| {s} | {gi.abs_cos_mean.iloc[0]:.2f} | "
                     + (f"{gc.abs_cos_mean.iloc[0]:.2f} / {gc.abs_cos_min.iloc[0]:.2f}" if len(gc) else "") + " |")
    mds = md[md.condition.str.contains("single")].copy()
    mds["fit"] = mds.condition.str.split("_").str[0]
    L += ["", "## Passive class-mean difference after / before erasure (norm ratio; single site)", "",
          "| Site | A-in | A-cf |", "|---:|---:|---:|"]
    for s in range(N_SITES):
        q = lambda f: mds[(mds.fit == f) & (mds.site == s) & (mds.condition == f"{f}_single_{s}")].ratio
        L.append(f"| {s} | {q('in').iloc[0]:.2f} | {q('cf').iloc[0]:.2f} |")
    sv = diag[~diag.fitting.str.startswith("perm")]
    L += ["", "## Checks", "",
          f"- Folds equal the stored necessity run: {checks.get('folds_equal_stored')}.",
          (f"- Natural passive log P(by) vs stored: P-in {ref['P-in']['natural_max_abs']:.2g}, P-cf "
           f"{ref['P-cf']['natural_max_abs']:.2g} (sha256 P-in {ref['P-in']['items_sha256'][:12]}…, P-cf "
           f"{ref['P-cf']['items_sha256'][:12]}…). Stored necessity scrub shares (estimate and CI) recomputed here: "
           f"max |Δ| {repro['max_abs'] if repro else 'n/a'} over {repro['n'] if repro else 0} values.")
          if ref else "- Smoke run: no passive references.",
          f"- Identity hook {checks['self_patch_max_abs']:.2g}; non-anchor positions {checks['outside_anchor_max_abs']:.2g}; "
          f"in-model vs offline single-site erasure {checks['single_offline_max_abs']:.2g}; active natural repeat "
          f"{checks['active_natural_vs_pass_max_abs']:.2g}.",
          f"- Scrub replay (in / cf): " + "; ".join(f"{k[-2:]} {v}" for k, v in checks.items() if k.startswith("scrub_replay")),
          f"- Erasers (true labels): rank {sorted(sv.erased_rank.unique())}, top singular value min {sv.sv_top.min():.3f}, "
          f"trace branch {int(sv.trace_branch.sum())}, train cross-covariance residual max {sv.train_xcov_resid.max():.2g}.",
          f"- Permuted labels: max |r| with true class {checks.get('perm_max_abs_r')}.", ""]
    (out / "leace_crossframe.md").write_text("\n".join(L))
    print("\n".join(L[:20]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/pythia/leace_crossframe")
    ap.add_argument("--ref-in", default="", help="leace_insample results dir (items.csv.gz, erasers.npz)")
    ap.add_argument("--ref-cf", default="", help="leace_necessity results dir")
    ap.add_argument("--stored-cumulative", default="reports/pythia/leace_necessity_cumulative.csv")
    ap.add_argument("--report-dir", default="reports/pythia")
    ap.add_argument("--readings", default="results/pythia/leace_crossframe/readings.json")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
