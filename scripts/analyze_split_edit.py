#!/usr/bin/env python3
"""Tables for the split-edit test, frame invariance, passive-axis patching and side splits
(rules: reports/pythia/split_edit_plan.md). Reads run_split_edit.py outputs (and, for part D, the direction check's
passive ablation and the LEACE per-prompt dumps); writes summary CSVs next to the outputs and the report.
Run from scripts/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_das_direction_check import LAB, RD, ci, f3, pboot, q
from analyze_das_round2_results import per_pair
from analyze_passive_test import Boot, cells, classify, outcome, pair_values, populations, summarize

CONDS = ("full", "on", "off")


def unit_rows(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def md(mean, cls, keep=None):
    """Unit T - I difference of per-verb means over the kept verbs; [sites, d]."""
    k = np.ones(len(cls), bool) if keep is None else keep
    return unit_rows(mean[k & (cls == 1)].mean(0) - mean[k & (cls == 0)].mean(0))


def side_share(good_nat, bad_nat, good_after, bad_after, n_boot, seed):
    """Bad-side share of the gap closure, ratio of pair means, pair bootstrap. Inputs: per-pair arrays."""
    ix = pboot(len(good_nat), n_boot, seed)
    dg = (good_after - good_nat)[ix].mean(1)
    db = (bad_after - bad_nat)[ix].mean(1)
    closure = db - dg  # = -(Delta G)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = db / closure
    return {"gap_nat": float((good_nat - bad_nat).mean()), "dG": float(-closure[0]), "d_good": float(dg[0]),
            "d_bad": float(db[0]), **{f"bad_share_{k}": v for k, v in q(share).items()}}


def site_dir(root, s, primary):
    return Path(root) / ("final_strict" if s == primary else f"final_strict_site{s}")


def stored_dirs(root, spec):
    return [Path(x) if Path(x).is_absolute() else Path(root) / x for x in spec.split(",") if x]


def delta_for(root, s, primary, dirs):
    """delta_s: the site's frozen config TOST block, else the first deltas.json listing the site."""
    cfg = json.loads((Path(root) / ("frozen_config_final.json" if s == primary else
                                    f"frozen_config_site{s}.json")).read_text())
    if "tost" in cfg:
        return cfg["tost"]["delta"]
    for d in dirs:
        p = d / "deltas.json"
        if p.exists():
            js = json.loads(p.read_text())
            if str(s) in js:
                v = js[str(s)]
                return v["delta"] if isinstance(v, dict) else v
    raise SystemExit(f"no delta_s for site {s}")


def stored_summary(s, dirs):
    for d in dirs:
        p = d / "transfer_summary.csv"
        if p.exists():
            x = pd.read_csv(p)
            x = x[(x.site == s) & (x.population == "primary (plain)") & (x.band == "all") & (x.condition == "D_bad")]
            if len(x):
                return x.set_index("readout")
    return None


def main(args):
    res = Path(args.results)
    root = Path(args.root)
    checks = json.loads((res / "checks.json").read_text())
    meta = json.loads((res / "meta.json").read_text())
    sites = [int(s) for s in meta["args"]["sites"].split(",")]
    smoke = bool(checks.get("smoke"))
    umd = bool(meta["args"].get("umd"))
    PCONDS = CONDS + (("umd",) if umd else ())
    sdirs = stored_dirs(root, args.stored_passive)
    gate = {}

    # ================================================================ A. actives
    A = pd.read_csv(res / "active_summary.csv")
    det = pd.read_csv(res / "active_swaps.csv.gz")
    a_rows, gate["active_iia_full"] = [], {}
    for s in sites:
        stored = pd.read_csv(site_dir(root, s, args.primary_site) / "summary.csv")
        stored = stored[(stored.control == "das") & (stored["rank"] == 1)].set_index(["split", "fold"])
        mine = A[(A.site == s) & (A.condition == "full")].set_index(["split", "fold"])
        diff = (mine.iia_cross - stored.iia_cross.reindex(mine.index)).abs()
        gate["active_iia_full"][str(s)] = {"max_abs_diff": float(diff.max()), "pass": bool(diff.max() <= 0.01)}
        pp = {}
        for c in PCONDS + ("u_pas",):
            p, _ = per_pair(det[(det.site == s) & (det.condition == c)], 1)
            pp[c] = p[p.base_cls == 0].set_index("base_pair_id").d_O
        common = pp["full"].index
        for c in pp:
            common = common.intersection(pp[c].index)
        ix = pboot(len(common), args.n_boot, 0)
        E = {c: pp[c][common].to_numpy()[ix].mean(1) for c in pp}
        for c in pp:
            g = A[(A.site == s) & (A.condition == c)]
            a_rows.append({"site": s, "condition": c, "iia_cross": g.iia_cross.mean(), "iia_cross_min": g.iia_cross.min(),
                           "iia_cross_max": g.iia_cross.max(), "iia_same": g.iia_same.mean(),
                           **{f"dO_{k}": v for k, v in q(E[c]).items()},
                           **({f"share_{k}": v for k, v in q(E[c] / E["full"]).items()}
                              if c in ("on", "off", "umd") else {}),
                           "n_pairs": len(common)})
        add = E["full"] - E["on"] - E["off"]
        a_rows.append({"site": s, "condition": "additivity full-on-off", **{f"dO_{k}": v for k, v in q(add).items()}})
    AS = pd.DataFrame(a_rows)
    AS.to_csv(res / "summary_active.csv", index=False)

    # ================================================================ A + C. passives
    items = pd.read_csv(args.items)
    plan = pd.read_csv(args.plan, dtype={c: "category" for c in ("item_id", "side", "cond", "donor_pair")})
    nat = pd.read_parquet(res / "passive_natural.parquet").set_index("pid")
    pr, pops = populations(items)
    pix0 = np.flatnonzero(pops["primary (plain)"].to_numpy())
    pair_ix = {p: i for i, p in enumerate(pr.index)}
    ctx_ix = {c: i for i, c in enumerate(sorted(items.context_id.unique()))}
    item_pair = items.set_index("item_id")[["pair_id", "context_id"]]
    plan_i = plan.set_index("row")
    P_rows, gate["passive_D_full"] = [], {}
    for s in sites:
        sp = pd.read_parquet(res / f"split_patches_site{s}.parquet")
        ax = pd.read_parquet(res / f"axis_patches_site{s}.parquet").assign(condition="u_pas")
        bt = Boot(len(ctx_ix), args.n_boot, args.seed)
        rows0 = sp[sp.condition == "full"].row.to_numpy()
        for c in PCONDS:
            assert (sp[sp.condition == c].row.to_numpy() == rows0).all()
        assert (ax.row.to_numpy() == rows0).all()
        pids = item_pair.loc[plan_i.loc[rows0, "item_id"].astype(str).unique(), "pair_id"].unique()
        pix = np.array(sorted({pair_ix[p] for p in pids} & set(pix0)))
        wp = bt.pair_weights(pr.band.to_numpy()[pix])
        S = {}
        for c, x in [(c, sp[sp.condition == c]) for c in PCONDS] + [("u_pas", ax)]:
            x = x.reset_index(drop=True)
            dd = plan_i.loc[x.row.to_numpy(), ["item_id", "cond", "base", "donor_pair"]].reset_index(drop=True)
            for col in ("item_id", "cond", "donor_pair"):
                dd[col] = dd[col].astype(str)
            dd[list(RD)] = x[list(RD)].to_numpy() - nat.loc[dd.base.to_numpy(), list(RD)].to_numpy()
            dd = dd.join(item_pair, on="item_id")
            qs = sorted(dd.donor_pair.unique())
            qix = {v: i for i, v in enumerate(qs)}
            dd = dd.assign(dq=dd.donor_pair)
            vT, mT = cells(dd[dd.cond == "T"], pair_ix, ctx_ix, qix, len(qs), RD)
            vI, mI = cells(dd[dd.cond == "I"], pair_ix, ctx_ix, qix, len(qs), RD)
            assert (mT == mI).all(), "T and I cells not paired"
            S[c] = summarize(pair_values(vT - vI, mT, bt.wc, bt.donor_weights("das", len(qs))), wp, pix)
        dl = delta_for(root, s, args.primary_site, sdirs)
        stored = stored_summary(s, sdirs)
        gate["passive_D_full"][str(s)] = {}
        for r, rd in enumerate(RD):
            row = {"site": s, "readout": rd, "delta_s": dl}
            for c in S:
                cc = ci(S[c][:, r])
                row.update({f"{c}_{k}": v for k, v in cc.items()})
                row[f"class_{c}"] = classify(cc, dl) if rd in ("O", "by", "dot") else ""
            with np.errstate(divide="ignore", invalid="ignore"):
                for c in ("on", "off", "u_pas") + (("umd",) if umd else ()):
                    row.update({f"share_{c}_{k}": v for k, v in ci(S[c][:, r] / S["full"][:, r]).items()})
            row.update({f"additivity_{k}": v for k, v in ci(S["full"][:, r] - S["on"][:, r] - S["off"][:, r]).items()})
            P_rows.append(row)
            if rd in ("O", "by") and not smoke and stored is not None:
                dv = abs(row["full_est"] - stored.loc[rd, "est"])
                gate["passive_D_full"][str(s)][rd] = {"full": row["full_est"], "stored": float(stored.loc[rd, "est"]),
                                                      "abs_diff": float(dv), "pass": bool(dv <= 0.02)}
    PS = pd.DataFrame(P_rows)
    PS.to_csv(res / "summary_passive.csv", index=False)
    push = pd.read_csv(res / "push.csv")
    # v's sign is arbitrary per basis: v -> -v flips a and w (and the separations along w) but leaves every
    # patch unchanged. Align each basis to a >= 0 before averaging; u (T - I class mean difference) is fixed.
    sgn = np.where(push.a < 0, -1.0, 1.0)
    for c in ("a", "sep_act_w", "sep_pas_w"):
        push[c] = push[c] * sgn
    push["abs_sep_act_w"], push["abs_sep_pas_w"] = push.sep_act_w.abs(), push.sep_pas_w.abs()
    PU = push.groupby("site").mean(numeric_only=True).drop(columns=["split", "fold"]).reset_index()
    PU["n_bases_flipped"] = pd.Series(sgn < 0).groupby(push.site.values).sum().values
    PU.to_csv(res / "summary_push.csv", index=False)

    # declared readings, A
    reading = {}
    # generic readings (reports/split_edit_replication_rules.md): outcome per site and condition, R-A1 / R-A2, S1, T1
    oc = {}
    for c in ("full", "on") + (("umd",) if umd else ()) + ("u_pas",):
        oc[c] = {}
        for s in sites:
            r_o = PS[(PS.site == s) & (PS.readout == "O")].iloc[0]
            r_b = PS[(PS.site == s) & (PS.readout == "by")].iloc[0]
            oc[c][s] = outcome(r_o[f"class_{c}"], r_b[f"class_{c}"])
    reading["outcomes"] = {c: {str(s): v for s, v in d_.items()} for c, d_ in oc.items()}
    ra1, ra2 = {}, {}
    for s in sites:
        r_o = PS[(PS.site == s) & (PS.readout == "O")].iloc[0]
        r_b = PS[(PS.site == s) & (PS.readout == "by")].iloc[0]
        if r_o.class_full == "RISE":
            ra1[str(s)] = {"off_share_O": float(r_o.share_off_est), "class_on_O": r_o.class_on,
                           "held": bool(r_o.share_off_est >= 0.5 and r_o.class_on != "RISE")}
        if r_b.class_full == "RISE":
            ra2[str(s)] = {"on_share_by": float(r_b.share_on_est), "held": bool(r_b.share_on_est >= 0.5)}
    rs = [int(x) for x in args.rule_sites.split(",")] if args.rule_sites else []
    missing = [x for x in rs if x not in sites]
    if missing:
        raise SystemExit(f"rule sites {missing} not in the run's sites {sites}")

    def at_rule_sites(d_):
        # declared: held if true at every rule site where the reading applies (v's class RISE there);
        # "n/a" if it applies at no rule site
        app = [str(x) for x in rs if str(x) in d_]
        return "n/a" if not app else bool(all(d_[x]["held"] for x in app))
    for name, d_ in (("R-A1", ra1), ("R-A2", ra2)):
        reading[name] = {"sites": d_, "held_at_rule_sites": at_rule_sites(d_),
                         "rule_sites_applicable": [x for x in rs if str(x) in d_],
                         "held_all_applicable_sites": bool(d_) and all(v["held"] for v in d_.values()),
                         "held_n": f"{sum(v['held'] for v in d_.values())} / {len(d_)}"}
    s1 = {}
    for c, d_ in oc.items():
        ab = [s for s, v in d_.items() if v.startswith("abstract")]
        su = [s for s, v in d_.items() if v.startswith("surface")]
        s1[c] = {"abstract": ab, "surface": su, "pass": bool(ab) and (not su or max(ab) < min(su))}
    reading["S1"] = s1
    if rs:
        reading["T1"] = {c: {"by_class": {str(s): PS[(PS.site == s) & (PS.readout == "by")].iloc[0][f"class_{c}"]
                                          for s in rs}} for c in oc}
        for c in oc:
            reading["T1"][c]["pass"] = all(v == "RISE" for v in reading["T1"][c]["by_class"].values())
    if args.legacy_readings and 12 in sites:
        o12 = PS[(PS.site == 12) & (PS.readout == "O")].iloc[0]
        reading["A1"] = {"off_share_O_12": o12.share_off_est, "class_on_O_12": o12.class_on,
                         "held": bool(o12.share_off_est >= 0.5 and o12.class_on != "RISE"),
                         "on_share_O_12": o12.share_on_est}
        reading["decision"] = ("redraw staging with the on-axis piece / u_md; v as upper bound" if reading["A1"]["held"]
                               else "keep staging drawn with v" if o12.share_on_est >= 0.5 else
                               "between: report both curves")
    if args.legacy_readings:
        a2 = {int(s): float(PS[(PS.site == s) & (PS.readout == "by")].share_on_est.iloc[0]) for s in sites if s <= 12}
        reading["A2"] = {"on_share_by": a2, "held": bool(all(v >= 0.5 for v in a2.values()))}

    # ================================================================ B. frame invariance
    z = np.load(res / "class_states.npz", allow_pickle=True)
    pm, pc, ppair = z["pas_mean"].astype(np.float64), z["pas_cls"], z["pas_pair"]
    am, ac, apair = z["act_mean"].astype(np.float64), z["act_cls"], z["act_pair"]
    nS = pm.shape[1]
    u_pas, u_act = md(pm, pc), md(am, ac)
    cos_ap = (u_pas * u_act).sum(-1)
    if args.leace_insample and Path(args.leace_insample).exists():
        lin = np.load(args.leace_insample)
        edit = unit_rows(lin["single_edit"][0].astype(np.float64))
        b0 = np.abs((edit[:nS] * u_pas).sum(-1))
        gate["B0"] = {"min_abs_cos": float(b0.min()), "pass": bool(b0.min() >= 0.98)}
    else:
        b0 = np.full(nS, np.nan)

    def halves(pair, cls, mean, n_draw, seed):
        ps = np.array(sorted(set(pair)))
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n_draw):
            perm = rng.permutation(len(ps))
            h1 = set(ps[perm[: len(ps) // 2]])
            k1 = np.array([p in h1 for p in pair])
            out.append((md(mean, cls, k1) * md(mean, cls, ~k1)).sum(-1))
        return np.array(out)  # [draws, sites]

    r_pas = halves(ppair, pc, pm, args.n_half, 0)
    r_act = halves(apair, ac, am, args.n_half, 0)
    rng = np.random.default_rng(1)
    ps = np.array(sorted(set(ppair)))
    pidx = np.array([np.searchsorted(ps, p) for p in ppair])
    flips = []
    for _ in range(args.n_flip):
        fl = rng.random(len(ps)) < 0.5
        cf = np.where(fl[pidx], 1 - pc, pc)
        flips.append(np.abs((md(pm, cf) * u_act).sum(-1)))
    flips = np.array(flips)
    B_rows = []
    for s in range(nS):
        ra, rp = r_act[:, s].mean(), r_pas[:, s].mean()
        B_rows.append({"site": s, "cos_act_pas": cos_ap[s], "abs_cos_pas_leace_edit": b0[s],
                       "split_half_pas": rp, "split_half_pas_lo": np.percentile(r_pas[:, s], 2.5),
                       "split_half_pas_hi": np.percentile(r_pas[:, s], 97.5), "split_half_act": ra,
                       "split_half_act_lo": np.percentile(r_act[:, s], 2.5),
                       "split_half_act_hi": np.percentile(r_act[:, s], 97.5),
                       "flip_mean": flips[:, s].mean(), "flip_p975": np.percentile(flips[:, s], 97.5),
                       "FI": cos_ap[s] / np.sqrt(ra * rp) if ra > 0 and rp > 0 else np.nan})  # n/a if r <= 0
    BS = pd.DataFrame(B_rows)
    for s in sites:
        dz = np.load(res / f"directions_site{s}.npz")
        cv = np.abs(dz["v"] @ u_pas[s])
        BS.loc[BS.site == s, "abs_cos_v_pas_mean"] = cv.mean()
        BS.loc[BS.site == s, "abs_cos_v_pas_min"] = cv.min()
        BS.loc[BS.site == s, "abs_cos_v_pas_max"] = cv.max()
        cu = np.abs(dz["u"] @ u_pas[s])
        BS.loc[BS.site == s, "abs_cos_umd_fold_pas_mean"] = cu.mean()
    BS.to_csv(res / "summary_frame.csv", index=False)
    reading["B1"] = {"held": bool((BS.cos_act_pas > BS.flip_p975).all()),
                     "sites_failing": BS[~(BS.cos_act_pas > BS.flip_p975)].site.tolist()}
    fis = [int(x) for x in args.fi_sites.split(",")]
    if not fis or any(x >= nS or x < 0 or not np.isfinite(BS.FI[x]) for x in fis):
        raise SystemExit(f"FI sites {fis}: each must be in 0..{nS - 1} with a finite FI")
    reading["B2"] = {str(s): {"cos": float(BS.cos_act_pas[s]), "split_half_pas": float(BS.split_half_pas[s]),
                              "below": bool(BS.cos_act_pas[s] < BS.split_half_pas[s])} for s in fis}
    reading["FI_all"] = {str(s): (None if np.isnan(BS.FI[s]) else round(float(BS.FI[s]), 3)) for s in range(nS)}
    fi = {s: float(BS.FI[s]) for s in fis}
    reading["FI"] = {"values": fi, "decision": ("low: 'one direction every construction reads'"
                                                if all(v < 0.5 for v in fi.values()) else
                                                "'one feature' stands" if all(v >= 0.5 for v in fi.values())
                                                else "mixed: decide in the write-up")}

    # ================================================================ D. side splits
    D_rows = []
    dchk = Path(args.dirchk_results) if args.dirchk_results else None
    if dchk and (dchk / f"ablation_passive_site{sites[0]}.parquet").exists():
        for s in sites:
            x = pd.read_parquet(dchk / f"ablation_passive_site{s}.parquet")
            x["item_id"] = x.item_id.astype(str)
            m = x.groupby(["direction", "item_id", "side"]).by.mean().unstack("side").reset_index()
            m = m.merge(item_pair.reset_index(), on="item_id").groupby(["direction", "pair_id"])[["good", "bad"]].mean()
            n = m.loc["none"]
            for dname in ("v", "u_md"):
                a = m.loc[dname].reindex(n.index)
                D_rows.append({"source": "mean ablation (direction check)", "site": s, "condition": dname,
                               "n_pairs": len(n), **side_share(n.good.to_numpy(), n.bad.to_numpy(),
                                                               a.good.to_numpy(), a.bad.to_numpy(),
                                                               args.n_boot, args.seed)})
    for label, path in (("LEACE in-sample", args.leace_insample_items), ("LEACE cross-fitted", args.leace_crossfit_items)):
        if not path or not Path(path).exists():
            continue
        L = pd.read_csv(path, usecols=lambda c: c in ("verb_pair", "side", "item") or c.startswith("lp_by__"))
        for cond in ("single_8", "scrub_0_8", "scrub_0_12", "scrub_all"):
            col = f"lp_by__{cond}"
            if col not in L.columns:
                continue
            g = L.groupby(["verb_pair", "side"])[["lp_by__natural", col]].mean().unstack("side")
            D_rows.append({"source": label, "site": "", "condition": cond, "n_pairs": len(g),
                           **side_share(g[("lp_by__natural", "good")].to_numpy(), g[("lp_by__natural", "bad")].to_numpy(),
                                        g[(col, "good")].to_numpy(), g[(col, "bad")].to_numpy(),
                                        args.n_boot, args.seed)})
    DS = pd.DataFrame(D_rows)
    DS.to_csv(res / "summary_side_split.csv", index=False)
    if len(DS):
        sel = DS[(DS.condition == "scrub_0_12") & DS.source.str.startswith("LEACE")]
        reading["D1"] = {r.source: float(r.bad_share_est) for r in sel.itertuples()}
        reading["D1"]["held"] = bool(len(sel) == 2 and (sel.bad_share_est >= 0.5).all())
    (res / "gates.json").write_text(json.dumps(gate, indent=2) + "\n")
    (res / "readings.json").write_text(json.dumps(reading, indent=2, default=str) + "\n")

    # ================================================================ report
    yn = lambda b: "pass" if b else "FAIL"
    Lr = [f"# Split-edit test, frame invariance and side splits ({args.model_label})", "",
          f"Rules, declared and committed before running: `{args.rules}`. Run: `run_split_edit.py` "
          f"({args.run_label}); tables: `analyze_split_edit.py`; outputs `{args.results.replace('../', '')}/`." +
          (" **Smoke run: numbers are not the declared ones.**" if smoke else ""), "", "## Checks", "",
          "| Site | full: active IIA vs stored (max abs diff) | full: passive D(O) / D(by) vs stored | write = read "
          "max abs dM | a·u + w = v max err | max abs w·u | u vs direction check max abs |", "|---:|---|---|---:|---:|---:|---:|"]
    for s in sites:
        ck = checks["sites"][str(s)]
        gi = gate["active_iia_full"][str(s)]
        gp = gate["passive_D_full"].get(str(s), {})
        pdx = " / ".join(f"{gp[r]['full']:.3f} vs {gp[r]['stored']:.3f} ({yn(gp[r]['pass'])})" for r in ("O", "by")
                         if r in gp) or "n/a"
        Lr.append(f"| {s} | {gi['max_abs_diff']:.4f} ({yn(gi['pass'])}) | {pdx} | {ck['write_eq_read_max_abs_dM']:.2g} | "
                  f"{ck['decomp_max_err']:.1e} | {ck['w_dot_u_max']:.1e} | {ck.get('u_vs_dirchk_max_abs', float('nan')):.1e} |")
    b0s = (f"B0: min over sites 0–{nS - 1} of |cos(u_pas, LEACE in-sample edit)| = {gate['B0']['min_abs_cos']:.4f} "
           f"({yn(gate['B0']['pass'])}; threshold 0.98)." if "B0" in gate else "B0: no LEACE run for this model.")
    Lr += ["", f"{b0s} LEACE items: {checks['leace_items']}. C fits: {checks['c_fit']}.",
           "", "## A. Split edit", "", "### Passives (primary population, bad bases, D = T donor − I donor)", "",
           "95% CI: three-way cluster bootstrap (2,000 draws, seed 17); shares per draw on shared draws. Class: the "
           "declared passive-test rule with the site's δ_s.", "",
           "| Site | Readout | D full | class | D on | class | D off | class | on / full | off / full | full − on − off |"
           + (" D u_md | class | u_md / full |" if umd else ""),
           "|---:|---|---|---|---|---|---|---|---|---|---|" + ("---|---|---|" if umd else "")]
    pick = lambda r, p: {"est": r[f"{p}_est"], "lo95": r[f"{p}_lo95"], "hi95": r[f"{p}_hi95"]}
    for r in PS.to_dict("records"):
        Lr.append(f"| {r['site']} | {LAB[r['readout']]} | {f3(pick(r, 'full'))} | {r['class_full']} | "
                  f"{f3(pick(r, 'on'))} | {r['class_on']} | {f3(pick(r, 'off'))} | {r['class_off']} | "
                  f"{f3(pick(r, 'share_on'), 2)} | {f3(pick(r, 'share_off'), 2)} | {f3(pick(r, 'additivity'))} |"
                  + (f" {f3(pick(r, 'umd'))} | {r['class_umd']} | {f3(pick(r, 'share_umd'), 2)} |" if umd else ""))
    Lr += ["", "### Passive outcome per site and condition", "",
           "| Site | " + " | ".join(oc) + " |", "|---:|" + "---|" * len(oc)]
    for s in sites:
        Lr.append(f"| {s} | " + " | ".join(oc[c][s] for c in oc) + " |")
    Lr += ["", "### Held-out actives", "", "IIA: cross-class held-out IIA, mean [min, max] over the 15 bases. Δ log P(O): "
           "I base ← T source, mean over pairs, 95% CI over pairs (2,000 draws, seed 0); share = condition / full per draw.",
           "", "| Site | Condition | IIA | same-class IIA | Δ log P(O) | share of full |", "|---:|---|---|---:|---|---|"]
    for r in AS.to_dict("records"):
        if r["condition"].startswith("additivity"):
            Lr.append(f"| {r['site']} | full − on − off | | | {r['dO_est']:.3f} [{r['dO_lo95']:.3f}, {r['dO_hi95']:.3f}] | |")
            continue
        sh = (f"{r['share_est']:.2f} [{r['share_lo95']:.2f}, {r['share_hi95']:.2f}]"
              if not pd.isna(r.get("share_est", np.nan)) else "")
        Lr.append(f"| {r['site']} | {r['condition']} | {r['iia_cross']:.3f} [{r['iia_cross_min']:.3f}, "
                  f"{r['iia_cross_max']:.3f}] | {r['iia_same']:.3f} | {r['dO_est']:.3f} [{r['dO_lo95']:.3f}, "
                  f"{r['dO_hi95']:.3f}] | {sh} |")
    Lr += ["", "### Push size and class separation (mean over the 15 bases)", "",
           "a = v·u, b = ‖w‖. Push: median over rows of |δ·a| / sd(h·u) (on) and |δ·b| / sd(h·ŵ) (off); actives: "
           "cross-class I ← T held-out swaps, sd over the 406 items; passives: plan rows, sd over primary good and bad "
           "prompts. Separation: standardized mean difference T − I (actives) / good − bad (passives) along u and ŵ.", "",
           "| Site | a | b | act on | act off | act off > 2 sd | pas on | pas off | pas off > 2 sd | sep act u | "
           "sep act ŵ | sep pas u | sep pas ŵ |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in PU.to_dict("records"):
        Lr.append(f"| {r['site']} | {r['a']:.3f} | {r['b']:.3f} | {r.get('act_on_med', np.nan):.2f} | "
                  f"{r.get('act_off_med', np.nan):.2f} | {r.get('act_off_gt2sd', np.nan):.2f} | {r.get('pas_on_med', np.nan):.2f} | "
                  f"{r.get('pas_off_med', np.nan):.2f} | {r.get('pas_off_gt2sd', np.nan):.2f} | {r['sep_act_u']:.2f} | "
                  f"{r['sep_act_w']:.2f} | {r['sep_pas_u']:.2f} | {r['sep_pas_w']:.2f} |")
    Lr += ["", "## B. Frame invariance of the class axis", "",
           "u_act: T − I over the 406 active items; u_pas: T − I over the LEACE items (126 pairs × 125 contexts); both "
           "from per-verb means, same bf16 runner. Split-half: 200 random halves of the pairs, mean [2.5, 97.5 pct]. "
           "Sign-flip reference: 1,000 draws, |cos(u_act, u_pas,flip)| mean / 97.5th pct. FI = cos / sqrt(r_act r_pas) "
           "(approximate; an upper bound).", "",
           "| Site | cos(u_act, u_pas) | split-half act | split-half pas | sign-flip ref | FI | |cos(u_pas, LEACE edit)| | "
           "|cos(v, u_pas)| | |cos(u_md fold, u_pas)| |", "|---:|---:|---|---|---|---:|---:|---|---:|"]
    for r in BS.to_dict("records"):
        vv = (f"{r['abs_cos_v_pas_mean']:.3f} [{r['abs_cos_v_pas_min']:.3f}, {r['abs_cos_v_pas_max']:.3f}]"
              if not pd.isna(r.get("abs_cos_v_pas_mean", np.nan)) else "")
        uu = f"{r['abs_cos_umd_fold_pas_mean']:.3f}" if not pd.isna(r.get("abs_cos_umd_fold_pas_mean", np.nan)) else ""
        Lr.append(f"| {r['site']} | {r['cos_act_pas']:.3f} | {r['split_half_act']:.3f} [{r['split_half_act_lo']:.3f}, "
                  f"{r['split_half_act_hi']:.3f}] | {r['split_half_pas']:.3f} [{r['split_half_pas_lo']:.3f}, "
                  f"{r['split_half_pas_hi']:.3f}] | {r['flip_mean']:.3f} / {r['flip_p975']:.3f} | {r['FI']:.2f} | "
                  f"{r['abs_cos_pas_leace_edit']:.4f} | {vv} | {uu} |")
    Lr += ["", "## C. Patching along u_pas (cross-fitted; exploratory)", "",
           "Passive D for u_pas is in the A passive CSV (`summary_passive.csv`, columns `u_pas_*`, `share_u_pas_*` = "
           "u_pas / full v); actives in `summary_active.csv` (condition u_pas). Stored v and u_md (fold) rows: "
           "`../das_direction_check/summary_passive_patching.csv`.", "",
           "| Site | Readout | D full v | D u_pas | class | u_pas / full v |", "|---:|---|---|---|---|---|"]
    for r in PS.to_dict("records"):
        Lr.append(f"| {r['site']} | {LAB[r['readout']]} | {f3(pick(r, 'full'))} | {f3(pick(r, 'u_pas'))} | "
                  f"{r['class_u_pas']} | {f3(pick(r, 'share_u_pas'), 2)} |")
    if len(DS):
        Lr += ["", "## D. Side splits of gap closure", "",
               "Gap G = good − bad log P(\" by\"), pair means. Bad-side share = (bad after − bad natural) / (−ΔG); "
               "pair bootstrap (2,000 draws, seed 17), ratio of means per draw.", "",
               "| Source | Site | Condition | pairs | natural G | ΔG | Δ good | Δ bad | bad-side share |",
               "|---|---|---|---:|---:|---:|---:|---:|---|"]
        for r in DS.to_dict("records"):
            Lr.append(f"| {r['source']} | {r['site']} | {r['condition']} | {r['n_pairs']} | {r['gap_nat']:.3f} | "
                      f"{r['dG']:.3f} | {r['d_good']:.3f} | {r['d_bad']:.3f} | {r['bad_share_est']:.2f} "
                      f"[{r['bad_share_lo95']:.2f}, {r['bad_share_hi95']:.2f}] |")
    Lr += ["", "## Declared readings", "", "```", json.dumps(reading, indent=2, default=str), "```"]
    Path(args.report).write_text("\n".join(Lr) + "\n", encoding="utf-8")
    print("\n".join(Lr))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="../results/pythia/split_edit")
    ap.add_argument("--root", default="../results/das_round2")
    ap.add_argument("--items", default="../data/das_round2/passive_test/items.csv")
    ap.add_argument("--plan", default="../data/das_round2/passive_test/plan.csv.gz")
    ap.add_argument("--leace-insample", default="../results/pythia/leace_insample/erasers.npz")
    ap.add_argument("--dirchk-results", default="../results/pythia/das_direction_check")
    ap.add_argument("--leace-insample-items", default="../results/pythia/leace_insample/items.csv.gz")
    ap.add_argument("--leace-crossfit-items", default="../results/pythia/leace_necessity/items.csv.gz")
    ap.add_argument("--report", default="../reports/pythia/split_edit.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-half", type=int, default=200)
    ap.add_argument("--n-flip", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--run-label", default="TSUBAME")
    ap.add_argument("--model-label", default="Pythia-1.4B")
    ap.add_argument("--rules", default="split_edit_plan.md")
    ap.add_argument("--primary-site", type=int, default=17)
    ap.add_argument("--stored-passive", default="passive_test,passive_test_fill",
                    help="comma list of dirs (relative to --root unless absolute) holding transfer_summary.csv "
                         "and deltas.json for the stored v passive test")
    ap.add_argument("--rule-sites", default="", help="declared 0.8 / 0.9 rule sites, for the T1 reading per condition")
    ap.add_argument("--fi-sites", default="8,12", help="sites of the FI decision")
    ap.add_argument("--legacy-readings", action=argparse.BooleanOptionalAction, default=True,
                    help="Pythia's A1 (site 12) and A2 (sites <= 12) readings")
    main(ap.parse_args())
