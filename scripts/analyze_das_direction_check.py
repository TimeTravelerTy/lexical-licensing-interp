#!/usr/bin/env python3
"""Tables for the DAS direction check (rules: reports/pythia/das_direction_check_plan.md).

Reads run_das_direction_check.py outputs; writes summary CSVs next to them and the report. Run from scripts/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_das_round2_results import per_pair
from analyze_passive_test import Boot, cells, classify, pair_values, populations, summarize

SITE_DIRS = {4: "final_strict_site4", 6: "final_strict_site6", 8: "final_strict_site8",
             12: "final_strict_site12", 17: "final_strict"}
STORED_PASSIVE = {4: "passive_test_fill", 6: "passive_test_fill", 8: "passive_test", 12: "passive_test",
                  17: "passive_test"}
RD = ("M", "O", "by", "dot")
LAB = {"M": "M", "O": "log P(O)", "by": 'log P(" by")', "dot": 'log P(".")'}


def delta_s(root, site):
    cfg = json.loads((Path(root) / ("frozen_config_final.json" if site == 17 else
                                    f"frozen_config_site{site}.json")).read_text())
    if "tost" in cfg:
        return cfg["tost"]["delta"]
    for name in ("passive_test", "passive_test_fill"):
        p = Path(root) / name / "deltas.json"
        if p.exists() and str(site) in json.loads(p.read_text()):
            return json.loads(p.read_text())[str(site)]
    raise KeyError(site)


def pboot(x, B, seed):
    """Pair-bootstrap index draws (row 0 = identity)."""
    rng = np.random.default_rng(seed)
    return np.vstack([np.arange(x), rng.integers(0, x, (B, x))])


def q(a):
    """Point estimate (row 0) and 95% percentile CI over the draws; NaN draws dropped and counted."""
    d = np.asarray(a[1:], float)
    ok = d[~np.isnan(d)]
    return {"est": float(a[0]), "lo95": float(np.percentile(ok, 2.5)) if len(ok) else np.nan,
            "hi95": float(np.percentile(ok, 97.5)) if len(ok) else np.nan, "nan_draws": int(len(d) - len(ok))}


def ci(a):
    """As analyze_passive_test.ci (95% and 90% percentile CIs), NaN draws dropped and counted."""
    d = np.asarray(a[1:], float)
    ok = d[~np.isnan(d)]
    pc = lambda p: float(np.percentile(ok, p)) if len(ok) else np.nan
    return {"est": float(a[0]), "lo95": pc(2.5), "hi95": pc(97.5), "lo90": pc(5), "hi90": pc(95),
            "nan_draws": int(len(d) - len(ok))}


def f3(s, k=3):
    return f"{s['est']:.{k}f} [{s['lo95']:.{k}f}, {s['hi95']:.{k}f}]"


def main(args):
    res = Path(args.results)
    root = Path(args.root)
    cos = pd.read_csv(res / "cosines.csv")
    sites = sorted(cos.site.unique())
    checks = json.loads((res / "checks.json").read_text())
    meta = json.loads((res / "meta.json").read_text())
    smoke = bool(meta["args"].get("max_items"))
    gate = {}

    # ---------------------------------------------------------------- 1. cosines
    c_rows = []
    for s in sites:
        g = cos[cos.site == s]
        z = np.load(res / f"directions_site{s}.npz")
        row = {"site": s, "n_bases": len(g)}
        for col in ("cos_v_umd", "cos_v_leace_edit", "cos_v_leace_read", "cos_umd_leace_edit", "cos_umd_leace_read",
                    "cos_v_leace_edit_crossfit", "cos_v_leace_read_crossfit", "var_ratio_v", "var_ratio_umd"):
            row[f"{col}_mean"], row[f"{col}_min"], row[f"{col}_max"] = g[col].mean(), g[col].min(), g[col].max()
        row["cos_leace_edit_read"] = g.cos_leace_edit_read.iloc[0]
        row["rand_mean"], row["rand_p95"] = float(z["rand_cos"].mean()), float(np.percentile(z["rand_cos"], 95))
        row["shuf_mean"], row["shuf_p95"] = float(z["shuf_cos"].mean()), float(np.percentile(z["shuf_cos"], 95))
        c_rows.append(row)
    C = pd.DataFrame(c_rows)
    C.to_csv(res / "summary_cosines.csv", index=False)

    # ---------------------------------------------------------------- 2a. active patching
    A = pd.read_csv(res / "active_summary.csv")
    det = pd.read_csv(res / "active_swaps.csv.gz")
    a_rows, gate["active_iia"], gate["active_effect"] = [], {}, {}
    for s in sites:
        stored = pd.read_csv(root / SITE_DIRS[s] / "summary.csv")
        stored = stored[(stored.control == "das") & (stored["rank"] == 1)].set_index(["split", "fold"])
        mine = A[(A.site == s) & (A.direction == "v")].set_index(["split", "fold"])
        diff = (mine.iia_cross - stored.iia_cross.reindex(mine.index)).abs()
        gate["active_iia"][str(s)] = {"max_abs_diff": float(diff.max()), "pass": bool(diff.max() <= 0.01)}
        pp = {}
        for name in ("v", "u_md"):
            p, _ = per_pair(det[(det.site == s) & (det.direction == name)], 1)
            pp[name] = p[p.base_cls == 0].set_index("base_pair_id").d_O
        common = pp["v"].index.intersection(pp["u_md"].index)
        vv, uu = pp["v"][common].to_numpy(), pp["u_md"][common].to_numpy()
        ix = pboot(len(common), args.n_boot, 0)
        ev, eu = vv[ix].mean(1), uu[ix].mean(1)
        ref = delta_s(root, s) / 0.2
        gate["active_effect"][str(s)] = {"v": float(ev[0]), "stored": ref, "pass": bool(abs(ev[0] - ref) <= 0.05)}
        for name, e in (("v", ev), ("u_md", eu)):
            g = A[(A.site == s) & (A.direction == name)]
            a_rows.append({"site": s, "direction": name, "iia_cross": g.iia_cross.mean(),
                           "iia_cross_min": g.iia_cross.min(), "iia_cross_max": g.iia_cross.max(),
                           "pc_frac_median": g.pc_frac_median.mean(), "iia_same": g.iia_same.mean(),
                           **{f"dO_{k}": v for k, v in q(e).items()}, "n_pairs": len(common)})
        iv = A[(A.site == s) & (A.direction == "v")].iia_cross.mean()
        iu = A[(A.site == s) & (A.direction == "u_md")].iia_cross.mean()
        a_rows.append({"site": s, "direction": "ratio u_md/v", "iia_cross": iu / iv,
                       **{f"dO_{k}": v for k, v in q(eu / ev).items()}})
    AS = pd.DataFrame(a_rows)
    AS.to_csv(res / "summary_active_patching.csv", index=False)

    # ---------------------------------------------------------------- 2b. passive patching
    have_passive = (res / f"passive_patches_site{sites[0]}.parquet").exists()
    P_rows, gate["passive_D"] = [], {}
    if have_passive:
        items = pd.read_csv(args.items)
        plan = pd.read_csv(args.plan, dtype={c: "category" for c in ("item_id", "side", "cond", "donor_pair")})
        nat = pd.read_parquet(res / "passive_natural.parquet").set_index("pid")
        pr, pops = populations(items)
        pix0 = np.flatnonzero(pops["primary (plain)"].to_numpy())
        pair_ix = {p: i for i, p in enumerate(pr.index)}
        ctx_ix = {c: i for i, c in enumerate(sorted(items.context_id.unique()))}
        item_pair = items.set_index("item_id")[["pair_id", "context_id"]]
        plan_i = plan.set_index("row")
        for s in sites:
            pat = pd.read_parquet(res / f"passive_patches_site{s}.parquet")
            bt = Boot(len(ctx_ix), args.n_boot, args.seed)
            assert (pat[pat.direction == "v"].row.to_numpy() == pat[pat.direction == "u_md"].row.to_numpy()).all()
            rows_v = pat[pat.direction == "v"].row.to_numpy()
            pids = item_pair.loc[plan_i.loc[rows_v, "item_id"].astype(str).unique(), "pair_id"].unique()
            pix = np.array(sorted({pair_ix[p] for p in pids} & set(pix0)))
            wp = bt.pair_weights(pr.band.to_numpy()[pix])  # one draw of pair weights, shared by v and u_md
            S = {}
            for name in ("v", "u_md"):
                x = pat[pat.direction == name].reset_index(drop=True)
                dd = plan_i.loc[x.row.to_numpy(), ["item_id", "cond", "base", "donor_pair"]].reset_index(drop=True)
                for c in ("item_id", "cond", "donor_pair"):
                    dd[c] = dd[c].astype(str)
                dd[list(RD)] = x[list(RD)].to_numpy() - nat.loc[dd.base.to_numpy(), list(RD)].to_numpy()
                dd = dd.join(item_pair, on="item_id")
                qs = sorted(dd.donor_pair.astype(str).unique())
                qix = {v: i for i, v in enumerate(qs)}
                dd = dd.assign(dq=dd.donor_pair.astype(str))
                vT, mT = cells(dd[dd.cond == "T"], pair_ix, ctx_ix, qix, len(qs), RD)
                vI, mI = cells(dd[dd.cond == "I"], pair_ix, ctx_ix, qix, len(qs), RD)
                assert (mT == mI).all(), "T and I cells not paired"
                S[name] = summarize(pair_values(vT - vI, mT, bt.wc, bt.donor_weights("das", len(qs))), wp, pix)
            dl = delta_s(root, s)
            stored = pd.read_csv(root / STORED_PASSIVE[s] / "transfer_summary.csv")
            stored = stored[(stored.site == s) & (stored.population == "primary (plain)") & (stored.band == "all")
                            & (stored.condition == "D_bad")].set_index("readout")
            gate["passive_D"][str(s)] = {}
            for r, rd in enumerate(RD):
                cv, cu = ci(S["v"][:, r]), ci(S["u_md"][:, r])
                with np.errstate(divide="ignore", invalid="ignore"):
                    cr = ci(S["u_md"][:, r] / S["v"][:, r])
                P_rows.append({"site": s, "readout": rd, "delta_s": dl,
                               **{f"v_{k}": v for k, v in cv.items()}, **{f"umd_{k}": v for k, v in cu.items()},
                               **{f"ratio_{k}": v for k, v in cr.items()},
                               "class_v": classify(cv, dl) if rd in ("O", "by", "dot") else "",
                               "class_umd": classify(cu, dl) if rd in ("O", "by", "dot") else "",
                               "stored_v_est": float(stored.loc[rd, "est"]) if rd in stored.index else np.nan})
                if rd in ("O", "by"):
                    dv = abs(cv["est"] - stored.loc[rd, "est"])
                    gate["passive_D"][str(s)][rd] = {"v": cv["est"], "stored": float(stored.loc[rd, "est"]),
                                                     "abs_diff": float(dv), "pass": bool(dv <= 0.02)}
    PS = pd.DataFrame(P_rows)
    PS.to_csv(res / "summary_passive_patching.csv", index=False)

    # ---------------------------------------------------------------- 3a. active ablation
    ab = pd.read_csv(res / "ablation_active.csv.gz")
    an = pd.read_csv(res / "active_natural.csv")
    an["item"] = np.arange(len(an))
    pairs = sorted(an.pair_id.unique())
    ixb = pboot(len(pairs), args.n_boot, args.seed)

    def pair_gap(df, col):
        g = df.groupby(["pair_id", "cls"])[col].mean().unstack("cls")
        return (g[1] - g[0]).reindex(pairs).to_numpy()

    # natural = the "none" (zero-basis) rows, run in the same batches as the ablations; the separate natural pass
    # is reported alongside (gap_nat_pass) as a batch-effect check
    nat_pass_gap = {c: pair_gap(an, c) for c in ("O", "M")}
    ab_rows = []
    for s in sites:
        x = ab[ab.site == s]
        gaps = {}
        for name, g in x.groupby("direction"):
            per_item = g.groupby(["item", "pair_id", "cls"])[["O", "M"]].mean().reset_index()
            assert len(per_item) == len(an) and (g.groupby("item").size() == 3).all(), "cross-fitting incomplete"
            gaps[name] = {c: pair_gap(per_item, c) for c in ("O", "M")}
        for name in gaps:
            if name == "none":
                continue
            for c in ("O", "M"):
                nb, abl = gaps["none"][c][ixb].mean(1), gaps[name][c][ixb].mean(1)
                ab_rows.append({"site": s, "frame": "active", "direction": name, "readout": c,
                                "gap_nat": nb[0], "gap_nat_pass": float(nat_pass_gap[c].mean()), "gap_abl": abl[0],
                                **{f"share_{k}": v for k, v in q(1 - abl / nb).items()}})
    # ---------------------------------------------------------------- 3b. passive ablation
    if have_passive:
        for s in sites:
            x = pd.read_parquet(res / f"ablation_passive_site{s}.parquet")
            x["item_id"] = x.item_id.astype(str)
            bt = Boot(len(ctx_ix), args.n_boot, args.seed)
            one = np.ones((bt.B + 1, 1), np.float32)
            pix = np.array(sorted({pair_ix[p] for p in item_pair.loc[x.item_id.unique(), "pair_id"].unique()}))
            wp = bt.pair_weights(pr.band.to_numpy()[pix])

            def boot_gap(df):
                v, m = cells(df.assign(dq=0), pair_ix, ctx_ix, {0: 0}, 1, RD)
                return summarize(pair_values(v, m, bt.wc, one), wp, pix)

            # separate natural pass (batch-effect check)
            ns = x[x.direction == "none"].drop_duplicates(["item_id", "side"])[["item_id", "side", "base"]]
            ng = pd.DataFrame(nat.loc[ns.base.to_numpy(), list(RD)].to_numpy(), columns=list(RD)) \
                .assign(item_id=ns.item_id.to_numpy(), side=ns.side.to_numpy()) \
                .pivot_table(index="item_id", columns="side", values=list(RD))
            Spass = boot_gap(pd.DataFrame({c: ng[(c, "good")] - ng[(c, "bad")] for c in RD}).join(item_pair))
            S = {}
            for name, g in x.groupby("direction"):
                assert (g.groupby(["item_id", "side"]).size() == 3).all(), "passive cross-fitting incomplete"
                m = g.groupby(["item_id", "side"])[list(RD)].mean().unstack("side")
                S[name] = boot_gap(pd.DataFrame({c: m[(c, "good")] - m[(c, "bad")] for c in RD}).join(item_pair))
            for name, Sab in S.items():
                if name == "none":
                    continue
                for r, rd in enumerate(RD):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        sh = 1 - Sab[:, r] / S["none"][:, r]
                    ab_rows.append({"site": s, "frame": "passive", "direction": name, "readout": rd,
                                    "gap_nat": float(S["none"][0, r]), "gap_nat_pass": float(Spass[0, r]),
                                    "gap_abl": float(Sab[0, r]), **{f"share_{k}": v for k, v in q(sh).items()}})
    AB = pd.DataFrame(ab_rows)
    AB.to_csv(res / "summary_ablation.csv", index=False)
    (res / "gates.json").write_text(json.dumps(gate, indent=2) + "\n")

    # ---------------------------------------------------------------- report
    L = ["# DAS direction check (Pythia-1.4B)", "",
         "Rules, declared and committed before running: `das_direction_check_plan.md`. Run: "
         f"`run_das_direction_check.py` ({args.run_label}); tables: `analyze_das_direction_check.py`; outputs "
         "`results/pythia/das_direction_check/`. Sites 4, 6, 8, 12, 17; 15 final-run rank-1 bases per site "
         "(3 splits x 5 folds)." + (" **Smoke run (passive subset): passive numbers are not the declared ones.**"
                                    if smoke else ""), "",
         "## Checks", "",
         "| Site | v held-out IIA vs stored (max abs diff per basis) | v active Δ log P(O) vs δ_s/0.2 | "
         "v passive D(O) vs stored | v passive D(by) vs stored | u_md self-patch max abs dM | identity ablation "
         "max abs dM (active / passive) | unit-norm max dev | batch effect max abs dM |",
         "|---:|---|---|---|---|---:|---|---:|---:|"]
    for s in sites:
        ck = checks["sites"][str(s)]
        gi, ge = gate["active_iia"][str(s)], gate["active_effect"][str(s)]
        gp = gate["passive_D"].get(str(s), {})
        pd_ = lambda r: (f"{gp[r]['v']:.3f} vs {gp[r]['stored']:.3f} ({'pass' if gp[r]['pass'] else 'FAIL'})"
                         if r in gp else "n/a")
        L.append(f"| {s} | {gi['max_abs_diff']:.4f} ({'pass' if gi['pass'] else 'FAIL'}) | {ge['v']:.3f} vs "
                 f"{ge['stored']:.3f} ({'pass' if ge['pass'] else 'FAIL'}) | {pd_('O')} | {pd_('by')} | "
                 f"{ck['u_md_self_patch_max_abs_dM']:.2g} | {ck['identity_ablation_active_max_abs_dM']:.2g} / "
                 f"{ck.get('identity_ablation_passive_max_abs_dM', float('nan')):.2g} | {ck['unit_norm_max_dev']:.1e} | "
                 f"{ck.get('batch_effect_max_abs_dM', float('nan')):.2g} |")
    L += ["", "Thresholds (declared): IIA 0.01, active effect 0.05, passive D 0.02 nats. Self-patch and identity "
          "ablation are compared with an unpatched forward of the same batch (zero basis); the batch-effect column "
          "is the max |dM| between that and the separate natural pass (bf16 batch composition).", "",
          "## 1. Natural class axis", "",
          "|cos| over the 15 bases: mean [min, max]. u_leace = in-sample single-site LEACE eraser (fit on passive "
          "`was` items), edit and read vectors; cross-fitted = mean over the 5 cross-fitted erasers. References "
          "pooled over bases: random isotropic unit vectors (10,000 per basis) and shuffled-label mean differences "
          "(100 verb-level permutations per basis), mean / 95th pct.", "",
          "| Site | v·u_md | v·u_leace edit | v·u_leace read | v·edit (cross-fit) | v·read (cross-fit) | "
          "u_md·edit | u_md·read | random ref | shuffled-label ref |", "|---:|---|---|---|---:|---:|---|---|---|---|"]
    mm = lambda r, c: f"{r[c + '_mean']:.3f} [{r[c + '_min']:.3f}, {r[c + '_max']:.3f}]"
    for r in C.to_dict("records"):
        L.append(f"| {r['site']} | {mm(r, 'cos_v_umd')} | {mm(r, 'cos_v_leace_edit')} | {mm(r, 'cos_v_leace_read')} | "
                 f"{r['cos_v_leace_edit_crossfit_mean']:.3f} | {r['cos_v_leace_read_crossfit_mean']:.3f} | "
                 f"{mm(r, 'cos_umd_leace_edit')} | {mm(r, 'cos_umd_leace_read')} | "
                 f"{r['rand_mean']:.3f} / {r['rand_p95']:.3f} | {r['shuf_mean']:.3f} / {r['shuf_p95']:.3f} |")
    L += ["", "Variance of the projection of all 406 active items, as a multiple of the mean per-dimension variance "
          "(trace / d); mean [min, max] over bases.", "", "| Site | along v | along u_md | |cos(edit, read)| |",
          "|---:|---|---|---:|"]
    for r in C.to_dict("records"):
        L.append(f"| {r['site']} | {mm(r, 'var_ratio_v')} | {mm(r, 'var_ratio_umd')} | {r['cos_leace_edit_read']:.3f} |")
    L += ["", "## 2. Component patching: u_md vs v", "", "### Held-out actives", "",
          "IIA: cross-class held-out IIA, mean [min, max] over bases. PC frac: median positive-control fraction, mean "
          "over bases. Δ log P(O): intransitive base ← transitive source, mean over pairs (`per_pair`), 95% CI over "
          "pairs; the ratio row's CI is a paired pair bootstrap.", "",
          "| Site | Direction | IIA | PC frac | same-class IIA | Δ log P(O) |", "|---:|---|---|---:|---:|---|"]
    for r in AS.to_dict("records"):
        if r["direction"].startswith("ratio"):
            L.append(f"| {r['site']} | u_md / v | {r['iia_cross']:.3f} | | | {r['dO_est']:.3f} [{r['dO_lo95']:.3f}, "
                     f"{r['dO_hi95']:.3f}] |")
        else:
            L.append(f"| {r['site']} | {r['direction']} | {r['iia_cross']:.3f} [{r['iia_cross_min']:.3f}, "
                     f"{r['iia_cross_max']:.3f}] | {r['pc_frac_median']:.3f} | {r['iia_same']:.3f} | "
                     f"{r['dO_est']:.3f} [{r['dO_lo95']:.3f}, {r['dO_hi95']:.3f}] |")
    if len(PS):
        L += ["", "### Passives (primary population, bad bases, D = T donor − I donor)", "",
              "95% CI: three-way cluster bootstrap (2,000 draws, seed 17); ratio per draw on shared draws. Class: the "
              "declared passive-test rule with the site's δ_s.", "",
              "| Site | Readout | D, v | class | D, u_md | class | u_md / v | stored D, v |",
              "|---:|---|---|---|---|---|---|---:|"]
        for r in PS.to_dict("records"):
            pick = lambda p: {"est": r[f"{p}_est"], "lo95": r[f"{p}_lo95"], "hi95": r[f"{p}_hi95"]}
            L.append(f"| {r['site']} | {LAB[r['readout']]} | {f3(pick('v'))} | {r['class_v']} | {f3(pick('umd'))} | "
                     f"{r['class_umd']} | {f3(pick('ratio'), 2)} | {r['stored_v_est']:.3f} |")
    L += ["", "## 3. Mean ablation in natural runs", "",
          "h ← h + (μ − h·u) u at the verb's last token, μ = grand mean of h·u over the frame's items (per basis). "
          "Natural = unpatched readouts computed in the same batches as the ablations (zero basis); the gap from "
          "the separate natural pass is in `summary_ablation.csv` (gap_nat_pass). "
          "Actives: T − I gap over the 29 pairs (each item averaged over the 3 bases holding its pair out); 95% CI "
          "over pairs. Passives: good − bad gap over the 64 primary pairs (each item averaged over its 3 plan "
          "bases); 95% CI two-way (pairs within band, contexts). Random: 10 random unit directions per basis, mean "
          "[min, max] of the share's point estimate.", "",
          "| Site | Frame | Gap | natural | ablated v | share removed, v | ablated u_md | share removed, u_md | "
          "share removed, random |", "|---:|---|---|---:|---:|---|---:|---|---|"]
    for s in sites:
        for frame, rd in (("active", "O"), ("active", "M"), ("passive", "by"), ("passive", "O")):
            x = AB[(AB.site == s) & (AB.frame == frame) & (AB.readout == rd)].set_index("direction")
            if not len(x):
                continue
            sv = {k: x.loc["v", f"share_{k}"] for k in ("est", "lo95", "hi95")}
            su = {k: x.loc["u_md", f"share_{k}"] for k in ("est", "lo95", "hi95")}
            rr = x[x.index.str.startswith("rand")].share_est
            L.append(f"| {s} | {frame} | {'T − I' if frame == 'active' else 'good − bad'} {LAB[rd]} | "
                     f"{x.loc['v', 'gap_nat']:.3f} | {x.loc['v', 'gap_abl']:.3f} | {f3(sv)} | "
                     f"{x.loc['u_md', 'gap_abl']:.3f} | {f3(su)} | {rr.mean():.3f} [{rr.min():.3f}, {rr.max():.3f}] |")
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="../results/pythia/das_direction_check")
    ap.add_argument("--root", default="../results/das_round2")
    ap.add_argument("--items", default="../data/das_round2/passive_test/items.csv")
    ap.add_argument("--plan", default="../data/das_round2/passive_test/plan.csv.gz")
    ap.add_argument("--report", default="../reports/pythia/das_direction_check.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--run-label", default="TSUBAME")
    main(ap.parse_args())
