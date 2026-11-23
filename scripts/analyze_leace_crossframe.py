#!/usr/bin/env python3
"""Report for run_leace_crossframe.py (rules: reports/pythia/leace_crossframe_rules.md, amended 2026-11-18).

Share of the natural passive *by* gap remaining under active-fit erasure (fittings A-in, A-cf; variants c = centered
on the passive mean (primary), u = as fit, t = translation only), next to the passive-fit references recomputed from
their stored per-prompt files (P-in: leace_insample, P-cf: leace_necessity). Same item table and task-1 two-way
cluster bootstrap as analyze_leace_necessity.py (1000 draws, seed 17), one set of draws for every condition, so
removed(A) / removed(P) is paired (conditional on the fixed erasers and folds). Permutation control, active-side check,
edit norms, eraser geometry (vs the passive-fit erasers and vs u_md), passive class-mean difference. Readings follow
the amended rules; any failed check makes them "invalid".
"""

from __future__ import annotations

import argparse
import os
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_das_prep import Boot

N_SITES, DAS_SITE, NEC_TOP = (int(os.environ.get("LEACE_N_SITES", 24)), 8,
                              int(os.environ.get("LEACE_NEC_TOP", 12)))
TOL = {"self_patch_max_abs": 1e-6, "outside_anchor_max_abs": 1e-6, "single_offline_max_abs": 1e-5,
       "active_natural_vs_pass_max_abs": 1e-5, "replay_mean_maxabs": 1e-6, "replay_xcov_rel": 1e-6,
       "replay_post_xcov_resid_max": 1e-5, "ref_natural_max_abs": 1e-3, "stored_share_repro": 1e-6,
       "perm_max_abs_r": 0.05, "sv_top_min": 0.01}


def ci(d):
    d = np.asarray(d, float)
    return float(d[0]), *(float(x) for x in np.nanpercentile(d[1:], [2.5, 97.5]))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def item_table(P, prefix, conds):
    G, B = P[P.side == "good"].set_index("item"), P[P.side == "bad"].set_index("item")
    assert (G.index == B.index).all()
    it = G[["verb_pair", "verb_band", "context_id", "context_band"]].reset_index()
    cols = {c: (G[f"{prefix}{c}"] - B[f"{prefix}{c}"]).to_numpy() for c in conds}
    return pd.concat([it, pd.DataFrame(cols, index=it.index)], axis=1)


def run(a):
    res, out = Path(a.results), Path(a.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    X = pd.read_csv(res / "items_passive.csv.gz")
    Xa = pd.read_csv(res / "items_active.csv.gz")
    checks = json.loads((res / "checks.json").read_text())
    meta = json.loads((res / "meta.json").read_text())
    diag = pd.read_csv(res / "fit_diagnostics.csv")
    md = pd.read_csv(res / "meandiff.csv")
    en = pd.read_csv(res / "edit_norms.csv").set_index("condition").mean_edit_norm_per_site
    smoke = bool(meta.get("smoke"))
    if not smoke and not (a.ref_in and a.ref_cf):
        raise SystemExit("--ref-in and --ref-cf are required for a full run")
    problems = []

    # passive-fit references
    ref, extra = {}, {}
    myfolds = pd.read_csv(res / "pairs_folds.csv").set_index("verb_pair").fold
    for name, d, ins in (("P-in", a.ref_in, True), ("P-cf", a.ref_cf, False)):
        if not d:
            continue
        d = Path(d)
        R = pd.read_csv(d / "items.csv.gz")
        rm = json.loads((d / "meta.json").read_text())
        rf = pd.read_csv(d / "pairs_folds.csv").set_index("verb_pair").fold
        ref[name] = {"natural_max_abs": float((R.lp_by__natural - X.lp_by__natural).abs().max()),
                     "items_sha256": sha(d / "items.csv.gz"), "insample_flag": bool(rm["args"].get("insample")),
                     "same_prompts": bool((R[["item", "side", "prompt"]].to_numpy()
                                           == X[["item", "side", "prompt"]].to_numpy()).all()),
                     "same_folds": bool(len(rf) == len(myfolds) and (myfolds.reindex(rf.index) == rf).all())}
        if not (ref[name]["same_prompts"] and ref[name]["same_folds"] and ref[name]["insample_flag"] == ins
                and ref[name]["natural_max_abs"] < TOL["ref_natural_max_abs"]):
            problems.append(f"reference {name}: {ref[name]}")
        for c in R.columns:
            if c.startswith("lp_by__") and c != "lp_by__natural":
                cond = c[len("lp_by__"):].replace("scrub_all", f"scrub_0_{N_SITES - 1}")
                extra[f"lp_by__{name}_{cond}"] = R[c].to_numpy()
                extra[f"kl__{name}_{cond}"] = R[f"kl__{c[len('lp_by__'):]}"].to_numpy()
    if extra:
        X = pd.concat([X, pd.DataFrame(extra, index=X.index)], axis=1)
    conds = [c[len("lp_by__"):] for c in X.columns if c.startswith("lp_by__")]
    it = item_table(X, "lp_by__", conds)
    boot = Boot(it, a.n_boot, np.random.default_rng(a.seed))
    allm = np.ones(len(it), bool)
    nat = boot.means(it["natural"].to_numpy(), allm)
    share = {c: boot.means(it[c].to_numpy(), allm) / nat for c in conds if c != "natural"}
    kl = {c: float(X[f"kl__{c}"].mean()) for c in conds if c != "natural"}
    gapd = {c: boot.means(it[c].to_numpy(), allm) - nat for c in conds if c != "natural"}  # erased - natural, nats

    def key(fit, var, kind, k):
        base = {"A-in": "in", "A-cf": "cf"}.get(fit, fit)
        tail = f"single_{k}" if kind == "single" else f"scrub_0_{k}"
        return f"{base}_{tail}" if fit.startswith("P-") else f"{base}_{var}_{tail}"

    rows = []
    for var in ("c", "u", "t"):
        for kind in ("single", "scrub"):
            for k in range(N_SITES):
                r = {"variant": var, "kind": kind, "site": k}
                for fit in ("A-in", "A-cf", "P-in", "P-cf"):
                    kk = key(fit, var, kind, k)
                    if kk not in share:
                        continue
                    e, lo, hi = ci(share[kk])
                    g = ci(gapd[kk])
                    r.update({f"{fit}_share": e, f"{fit}_share_lo": lo, f"{fit}_share_hi": hi,
                              f"{fit}_dgap": g[0], f"{fit}_dgap_lo": g[1], f"{fit}_dgap_hi": g[2], f"{fit}_kl": kl[kk]})
                    if fit.startswith("A-"):
                        r[f"{fit}_edit_norm"] = float(en.get(kk, np.nan))
                for A, Pn in (("A-in", "P-in"), ("A-cf", "P-cf")):
                    ka, kp = key(A, var, kind, k), key(Pn, var, kind, k)
                    if ka in share and kp in share:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            rho = (1 - share[ka]) / (1 - share[kp])
                        e, lo, hi = ci(rho)
                        r.update({f"rho_{A}": e, f"rho_{A}_lo": lo, f"rho_{A}_hi": hi})
                rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(out / "leace_crossframe_summary.csv", index=False)

    # permutation control (A-in fitting, permuted verb labels), per variant
    perm = {}
    for var in ("c", "u"):
        for kind, k in (("single", DAS_SITE), ("scrub", NEC_TOP)):
            suf = f"_{var}_single_{k}" if kind == "single" else f"_{var}_scrub_0_{k}"
            cs = [c for c in share if c.startswith("perm") and c.endswith(suf)]
            if cs:
                ps = np.array([share[c][0] for c in cs])
                perm[f"{var}_{kind}_{k}"] = {"n": len(ps), "mean": float(ps.mean()), "min": float(ps.min()),
                                            "max": float(ps.max()), "kl_mean": float(np.mean([kl[c] for c in cs])),
                                            "edit_norm_mean": float(np.mean([en.get(c, np.nan) for c in cs]))}

    # active-side check (erasers as fit, on actives)
    acond = [c[len("lp_obj__"):] for c in Xa.columns if c.startswith("lp_obj__")]
    ita = item_table(Xa, "lp_obj__", acond)
    assert (ita[["item", "verb_pair", "context_id"]].to_numpy() == it[["item", "verb_pair", "context_id"]].to_numpy()).all()
    nat_a = boot.means(ita["natural"].to_numpy(), allm)
    active = {c: ci(boot.means(ita[c].to_numpy(), allm) / nat_a) for c in acond if c != "natural"}

    # checks -> problems
    for nm in ("self_patch_max_abs", "outside_anchor_max_abs", "single_offline_max_abs", "active_natural_vs_pass_max_abs"):
        if not checks.get(nm, np.inf) <= TOL[nm]:
            problems.append(f"{nm} = {checks.get(nm)}")
    for fit in ("in", "cf"):
        rp = checks.get(f"scrub_replay_{fit}", {})
        for k2 in ("mean_maxabs", "xcov_rel", "post_xcov_resid_max"):
            if not rp.get(k2, np.inf) <= TOL[f"replay_{k2}"]:
                problems.append(f"replay {fit} {k2} = {rp.get(k2)}")
    sv = diag[~diag.fitting.str.startswith("perm")]
    if not ((sv.erased_rank == 1).all() and sv.sv_top.min() > TOL["sv_top_min"]):
        problems.append("eraser rank / singular value")
    if not smoke and not (checks.get("perm_max_abs_r") is not None and checks["perm_max_abs_r"] < TOL["perm_max_abs_r"]):
        problems.append(f"permutation balance r = {checks.get('perm_max_abs_r')}")
    if not smoke and not checks.get("folds_equal_stored"):
        problems.append("folds not checked against the stored run")

    # reproduction of the stored necessity report's passive-fit scrub shares (same items and bootstrap)
    repro = None
    cum = Path(a.stored_cumulative)
    if ref and cum.exists():
        C = pd.read_csv(cum)
        dev = []
        for _, c in C.iterrows():
            kk = f"{'P-in' if c.run == 'in-sample' else 'P-cf'}_scrub_0_{int(c.k)}"
            if kk in share:
                e, lo, hi = ci(share[kk])
                dev.append(max(abs(e - c.share_remaining), abs(lo - c.share_remaining_lo), abs(hi - c.share_remaining_hi)))
        repro = {"n": len(dev), "max_abs": float(max(dev)) if dev else None}
        if not dev or repro["max_abs"] > TOL["stored_share_repro"]:
            problems.append(f"stored share reproduction {repro}")

    # geometry
    geo = []
    E = np.load(res / "erasers.npz")
    for fit, refd in (("in", a.ref_in), ("cf", a.ref_cf)):
        if not refd or not (Path(refd) / "erasers.npz").exists():
            continue
        Q = np.load(Path(refd) / "erasers.npz")
        ea, ep = E[f"{fit}_single_edit"], Q["single_edit"]
        assert ea.shape == ep.shape, (fit, ea.shape, ep.shape)
        for s in range(N_SITES):
            c = [abs(float(ea[f, s] @ ep[f, s] / np.linalg.norm(ea[f, s]) / np.linalg.norm(ep[f, s])))
                 for f in range(ea.shape[0])]
            geo.append({"what": f"A-{fit} vs P-{fit}", "site": s, "abs_cos_mean": float(np.mean(c)),
                        "abs_cos_min": float(np.min(c))})
    for s in (4, 6, 8, 12, 17):
        fp = Path(a.umd_dir) / f"directions_site{s}.npz"
        if not fp.exists():
            continue
        U = np.load(fp, allow_pickle=True)["u"]
        U = U / np.linalg.norm(U, axis=1, keepdims=True)
        comps = [("A-in vs u_md", E["in_single_edit"][0, s])]
        if a.ref_in:
            comps.append(("P-in vs u_md", np.load(Path(a.ref_in) / "erasers.npz")["single_edit"][0, s]))
        for what, vec in comps:
            c = np.abs(U @ (vec / np.linalg.norm(vec)))
            geo.append({"what": what, "site": s, "abs_cos_mean": float(c.mean()), "abs_cos_min": float(c.min())})
    G = pd.DataFrame(geo)
    G.to_csv(out / "leace_crossframe_geometry.csv", index=False)

    # readings (primary: variant c, scrub 0-12; also reported for u)
    def reading(A, Pn, var):
        r = T[(T.variant == var) & (T.kind == "scrub") & (T.site == NEC_TOP)].iloc[0]
        pm = perm.get(f"{var}_scrub_{NEC_TOP}")
        rem_lo, rem_hi = 1 - r[f"{A}_share_hi"], 1 - r[f"{A}_share_lo"]
        p_rem_lo = 1 - r[f"{Pn}_share_hi"]
        below = pm is not None and r[f"{A}_share_hi"] < pm["min"]
        rho, rlo, rhi = r[f"rho_{A}"], r[f"rho_{A}_lo"], r[f"rho_{A}_hi"]
        if problems or p_rem_lo <= 0 or not np.isfinite(rho) or pm is None:
            lab = "invalid"
        elif rem_hi < 0:
            lab = "gap enhancement"
        elif rem_lo > 0 and below:
            lab = ("majority transfer" if rlo >= 0.5 else "minority transfer" if rhi < 0.5
                   else "positive transfer, share unresolved")
        elif rhi < 0.2:
            lab = "no meaningful transfer"
        else:
            lab = "inconclusive"
        act = active.get(f"{A[2:]}_u_scrub_0_{NEC_TOP}")
        tr = T[(T.variant == "t") & (T.kind == "scrub") & (T.site == NEC_TOP)].iloc[0]
        flags = []
        if act is not None and not act[2] < 1:
            flags.append("eraser inert in its own frame (active removal CI includes 0)")
        if 1 - tr[f"{A}_share_hi"] > 0:
            flags.append("translation alone removes part of the gap")
        return {"reading": lab, "flags": flags, "share_A": [r[f"{A}_share"], r[f"{A}_share_lo"], r[f"{A}_share_hi"]],
                "share_P": [r[f"{Pn}_share"], r[f"{Pn}_share_lo"], r[f"{Pn}_share_hi"]],
                "removed_A_nats": [-r[f"{A}_dgap"], -r[f"{A}_dgap_hi"], -r[f"{A}_dgap_lo"]],
                "rho": [rho, rlo, rhi], "below_perm_min": bool(below),
                "perm_min": pm["min"] if pm else None,
                "translation_share": [tr[f"{A}_share"], tr[f"{A}_share_lo"], tr[f"{A}_share_hi"]],
                "active_share": list(act) if act else None}
    readings = {"natural_gap": ci(nat), "natural_active_gap": ci(nat_a), "perm": perm, "active_side": active,
                "problems": problems, "checks": {**checks, "refs": ref, "stored_share_repro": repro},
                "outputs_sha256": {f: sha(res / f) for f in ("items_passive.csv.gz", "items_active.csv.gz",
                                                              "erasers.npz")}}
    if ref:
        for var in ("c", "u"):
            for A, Pn in (("A-cf", "P-cf"), ("A-in", "P-in")):
                readings[f"{A} vs {Pn} ({var})"] = reading(A, Pn, var)
    Path(a.readings).parent.mkdir(parents=True, exist_ok=True)
    Path(a.readings).write_text(json.dumps(readings, indent=1, default=float) + "\n")

    f3 = lambda t: f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"
    L = ["# Cross-frame LEACE (Pythia-1.4B)", "",
         "Rules: `reports/pythia/leace_crossframe_rules.md` (declared 2026-11-17, amended 2026-11-18 before the full "
         "run). Scripts: `scripts/run_leace_crossframe.py`, `scripts/analyze_leace_crossframe.py`. Erasers fit on "
         "**active** verb-position states (\"The AGENT VERB-ed\", same verbs and contexts), applied to the passive "
         "items of `leace_necessity.md`. A-in: fit on all verbs (reference P-in). A-cf: fold f's eraser fit on the "
         "other folds' actives (reference P-cf). Variants: c = centered on the passive mean (primary), u = as fit "
         "(active mean), t = the constant shift of u alone. Share = erased / natural passive *by* gap; removed = 1 − "
         f"share; ρ = removed(A) / removed(P), paired bootstrap ({a.n_boot} draws, seed {a.seed}). Run: "
         f"{meta['minutes']:.0f} min.", "",
         f"Natural *by* gap {ci(nat)[0]:+.2f} [{ci(nat)[1]:+.2f}, {ci(nat)[2]:+.2f}] nats.", ""]
    if problems:
        L += ["**Problems (readings invalid):** " + "; ".join(problems), ""]
    L += [f"## Readings (scrub 0–{NEC_TOP})", ""]
    for k, r in readings.items():
        if " vs " in k:
            pmin = "n/a" if r["perm_min"] is None else f"{r['perm_min']:.2f}"
            L.append(f"- **{k}: {r['reading']}**{' (' + '; '.join(r['flags']) + ')' if r['flags'] else ''}. "
                     f"Share remaining A {f3(r['share_A'])}, P {f3(r['share_P'])}; removed by A "
                     f"{r['removed_A_nats'][0]:+.2f} [{r['removed_A_nats'][1]:+.2f}, {r['removed_A_nats'][2]:+.2f}] "
                     f"nats; ρ {f3(r['rho'])}; below the permutation minimum ({pmin}): {r['below_perm_min']}; "
                     f"translation alone leaves {f3(r['translation_share'])}; active-side share remaining "
                     f"{f3(r['active_share']) if r['active_share'] else 'n/a'}.")
    L += ["", "## Permutation control (A-in fitting, permuted verb labels)", "",
          "| Variant / condition | n | mean share | min | max | mean KL | mean edit norm |", "|---|---:|---:|---:|---:|---:|---:|"]
    L += [f"| {k} | {v['n']} | {v['mean']:.2f} | {v['min']:.2f} | {v['max']:.2f} | {v['kl_mean']:.3f} | "
          f"{v['edit_norm_mean']:.2f} |" for k, v in perm.items()]
    L += ["", "## Active-side check (share of the natural active T − I log P(object start) gap remaining)", "",
          f"Natural active gap {ci(nat_a)[0]:+.2f} [{ci(nat_a)[1]:+.2f}, {ci(nat_a)[2]:+.2f}] nats. Erasers as fit "
          "(active-centered); A-cf on each fold's held-out actives, A-in on the fitting actives.", ""]
    L += [f"- {c}: {f3(v)}" for c, v in active.items()]
    for var, vname in (("c", "centered (primary)"), ("u", "as fit"), ("t", "translation only")):
        for kind, title in (("scrub", "scrub prefixes 0..k"), ("single", "single site")):
            L += ["", f"## {title.capitalize()}, {vname}", "",
                  "| k | A-in | P-in | ρ in | A-cf | P-cf | ρ cf | KL A-in / A-cf | edit norm A-in / A-cf |",
                  "|---:|---|---|---|---|---|---|---|---|"]
            for _, r in T[(T.variant == var) & (T.kind == kind)].iterrows():
                g = lambda f: (f"{r[f + '_share']:.2f} [{r[f + '_share_lo']:.2f}, {r[f + '_share_hi']:.2f}]"
                               if pd.notna(r.get(f + "_share")) else "")
                h = lambda A: (f"{r['rho_' + A]:.2f} [{r['rho_' + A + '_lo']:.2f}, {r['rho_' + A + '_hi']:.2f}]"
                               if pd.notna(r.get("rho_" + A)) else "")
                note = " (next to readout)" if r.site > NEC_TOP else ""
                L.append(f"| {r.site}{note} | {g('A-in')} | {g('P-in')} | {h('A-in')} | {g('A-cf')} | {g('P-cf')} | "
                         f"{h('A-cf')} | {r.get('A-in_kl', np.nan):.3f} / {r.get('A-cf_kl', np.nan):.3f} | "
                         f"{r.get('A-in_edit_norm', np.nan):.2f} / {r.get('A-cf_edit_norm', np.nan):.2f} |")
    if len(G):
        L += ["", "## Geometry: |cos| of single-site edit vectors (mean / min over folds or u_md bases)", "",
              "| Comparison | Site | mean | min |", "|---|---:|---:|---:|"]
        L += [f"| {r.what} | {r.site} | {r.abs_cos_mean:.2f} | {r.abs_cos_min:.2f} |" for r in G.itertuples()]
    L += ["", "## Passive class-mean difference after / natural (norm ratio, single site; pooled and fold range)",
          "", "| Site | A-in c | A-in u | A-cf c (folds) | A-cf u (folds) |", "|---:|---:|---:|---|---|"]
    md["fold"] = md.fold.astype(str)
    for s in range(N_SITES):
        def q(fit, var):
            sub = md[(md.condition == f"{fit}_{var}_single_{s}") & (md.site == s)]
            pooled = float(sub[sub.fold == "pooled"].ratio.iloc[0])
            pf = sub[sub.fold != "pooled"].ratio
            return pooled, float(pf.min()), float(pf.max())
        ic, iu, cc, cu = q("in", "c"), q("in", "u"), q("cf", "c"), q("cf", "u")
        L.append(f"| {s} | {ic[0]:.2f} | {iu[0]:.2f} | {cc[0]:.2f} ({cc[1]:.2f}–{cc[2]:.2f}) | "
                 f"{cu[0]:.2f} ({cu[1]:.2f}–{cu[2]:.2f}) |")
    L += ["", "Scrub rows of `meandiff.csv` compare the passive class-mean difference at each site after all erasers up "
          "to that site with the natural one (not just before vs after the local eraser).", "",
          "## Checks", "",
          f"- Problems: {problems if problems else 'none'}.",
          f"- Folds equal the stored necessity run: {checks.get('folds_equal_stored')}.",
          (("- References: " + "; ".join(f"{k}: same prompts {v['same_prompts']}, same folds {v['same_folds']}, "
                                          f"in-sample flag {v['insample_flag']}, natural max |Δ| "
                                          f"{v['natural_max_abs']:.2g}, sha256 {v['items_sha256'][:12]}…"
                                          for k, v in ref.items())
            + f". Stored necessity scrub shares recomputed here: max |Δ| {repro['max_abs'] if repro else 'n/a'} over "
              f"{repro['n'] if repro else 0} values.") if ref else "- Smoke run: no passive references."),
          f"- Identity hook {checks['self_patch_max_abs']:.2g}; non-anchor positions {checks['outside_anchor_max_abs']:.2g}; "
          f"in-model vs offline single-site erasure {checks['single_offline_max_abs']:.2g}; active natural repeat "
          f"{checks['active_natural_vs_pass_max_abs']:.2g}.",
          "- Scrub replay: " + "; ".join(f"{k[-2:]} {v}" for k, v in checks.items() if k.startswith("scrub_replay")),
          f"- Erasers (true labels): rank {sorted(sv.erased_rank.unique())}, top singular value min {sv.sv_top.min():.3f}, "
          f"trace branch {int(sv.trace_branch.sum())}, train cross-covariance residual max {sv.train_xcov_resid.max():.2g}.",
          f"- Permuted labels: max |r| with true class {checks.get('perm_max_abs_r')}.",
          f"- OBJ tokens ({len(meta.get('obj_tokens', []))}): {meta.get('obj_tokens')}.",
          "- Output sha256: " + ", ".join(f"{k} {v[:12]}…" for k, v in readings["outputs_sha256"].items()), ""]
    (out / "leace_crossframe.md").write_text("\n".join(L))
    print("\n".join(L[:16]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/pythia/leace_crossframe")
    ap.add_argument("--ref-in", default="", help="leace_insample results dir (items.csv.gz, erasers.npz, meta.json)")
    ap.add_argument("--ref-cf", default="", help="leace_necessity results dir")
    ap.add_argument("--stored-cumulative", default="reports/pythia/leace_necessity_cumulative.csv")
    ap.add_argument("--umd-dir", default="results/pythia/split_edit")
    ap.add_argument("--report-dir", default="reports/pythia")
    ap.add_argument("--readings", default="results/pythia/leace_crossframe/readings.json")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
