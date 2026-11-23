#!/usr/bin/env python3
"""Report for run_das_reverse_freq.py + run_passive_test.py (rules: reports/pythia/das_reverse_freq_rules.md, with its
Nov 23 amendment).

Per site and condition (F filtered, U unfiltered, B both sides rare): |cos| with v; held-out active IIA on v's
frequent pairs, swap-weighted, same swaps as v, paired difference with a pair bootstrap; D_bad(by) on the primary
(plain) population minus pairs sharing a verb with the condition's training verbs, by band, v and the rare direction
on one set of bootstrap draws; per-seed D (plan rows of split k). Sites 8 and 12 are the transfer tests; 17 is
signed retention (descriptive).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_passive_test as apt
from analyze_das_stability_ext import bar_ok, overlap_mask

BANDS = ("head", "tail", "xtail", "all")
CONDS = ("F", "U", "B")
TEST_SITES = (8, 12)


def d_bad(pdir, site, plan, items, nat, bt, pair_ix, ctx_ix, das_q, split=None):
    pat = pd.read_parquet(Path(pdir) / f"patches_site{site}.parquet", columns=["row", "by"])
    assert (pat.row.to_numpy() == plan.row.to_numpy()).all()
    df = plan[["item_id", "side", "cond", "base", "donor_pair", "split"]].copy()
    df["by"] = pat["by"].to_numpy() - nat.set_index("pid").loc[plan.base.to_numpy(), "by"].to_numpy()
    df = df.join(items.set_index("item_id")[["pair_id", "context_id"]], on="item_id")
    d = df[(df.side == "bad") & df.cond.isin(["T", "I"])]
    if split is not None:
        d = d[d.split == split]
    d = d.assign(dq=d.donor_pair)
    qix = {q: i for i, q in enumerate(das_q)}
    vT, mT = apt.cells(d[d.cond == "T"], pair_ix, ctx_ix, qix, len(das_q), ["by"])
    vI, mI = apt.cells(d[d.cond == "I"], pair_ix, ctx_ix, qix, len(das_q), ["by"])
    assert (mT == mI).all()
    return apt.pair_values(vT - vI, mT, bt.wc, bt.donor_weights("das", len(das_q)))


def iia(res, s, c, items, n_boot, seed):
    v = pd.read_csv(res / f"v_site{s}_swaps.csv.gz")
    r = pd.read_csv(res / f"{c}_site{s}" / "heldout_swaps.csv.gz")
    m = v.merge(r, on=["split", "fold", "base", "src"], suffixes=("_v", "_r"))
    assert len(m) == len(v) == len(r)
    m = m[m.cross_v].copy()
    m["pair"] = items.pair_id.values[m.base]
    g = m.groupby("pair").agg(sv=("iia_v", "sum"), sr=("iia_r", "sum"), n=("iia_v", "size"))
    rng = np.random.default_rng(seed)
    W = np.vstack([np.ones(len(g)), rng.multinomial(len(g), np.full(len(g), 1 / len(g)), size=n_boot)])
    pv, pr_ = (W @ g.sv.to_numpy()) / (W @ g.n.to_numpy()), (W @ g.sr.to_numpy()) / (W @ g.n.to_numpy())
    return apt.ci(pv), apt.ci(pr_), apt.ci(pr_ - pv)


def label(dr, ratio):
    if dr["lo95"] > 0:
        return "transfers" if ratio["est"] >= 0.5 else "weak transfer"
    return "opposite" if dr["hi95"] < 0 else "unresolved"


def run(a):
    res = Path(a.results)
    checks = json.loads((res / "checks.json").read_text())
    C = pd.read_csv(res / "cosines.csv")
    items = pd.read_csv(a.items)
    plan = pd.read_csv(a.plan, dtype={c: "category" for c in ("item_id", "side", "cond", "donor_pair")})
    stored_items = pd.read_csv(Path(a.stored_root) / "final_strict_site8" / "items.csv")
    pr, pops = apt.populations(items)
    prim = pops["primary (plain)"].to_numpy()
    pair_ix = {p: i for i, p in enumerate(pr.index)}
    ctx_ix = {c: i for i, c in enumerate(sorted(items.context_id.unique()))}
    das_q = sorted(plan[plan.cond == "T"].donor_pair.dropna().astype(str).unique())
    stored_meta = json.loads((Path(a.stored_passive) / "run_meta.json").read_text())["checks"]
    stored_sum = pd.read_csv(Path(a.stored_passive) / "transfer_summary.csv")
    ext = {}
    if a.ext_results and (Path(a.ext_results) / "readings.json").exists():
        ext = json.loads((Path(a.ext_results) / "readings.json").read_text())["sites"]
    problems = [f"{k}: {v}" for k, v in checks.items() if k.startswith("stored_recompute_site") and not v["pass"]]
    excl, pixs = {}, {}
    for c in CONDS:
        Pc = pd.read_csv(res / f"pairs_{c}.csv")
        ov = overlap_mask(pr, set(Pc.trans) | set(Pc.intrans))
        excl[c] = sorted(pr.index[prim & ov])
        pixs[c] = {b: np.flatnonzero(prim & ~ov & ((pr.band == b).to_numpy() if b != "all" else True)) for b in BANDS}
    full = {b: np.flatnonzero(prim & ((pr.band == b).to_numpy() if b != "all" else True)) for b in BANDS}
    rows, readings, seed_rows = [], {}, []
    for s in sorted(C.site.unique()):
        bt = apt.Boot(items.context_id.nunique(), a.n_boot, a.seed)
        wps = {c: {b: bt.pair_weights(pr.band.to_numpy()[pixs[c][b]]) for b in BANDS} for c in CONDS}
        wfull = {b: bt.pair_weights(pr.band.to_numpy()[full[b]]) for b in BANDS}
        nat_v = pd.read_parquet(Path(a.stored_passive) / "natural.parquet")
        Av = d_bad(a.stored_passive, s, plan, items, nat_v, bt, pair_ix, ctx_ix, das_q)
        for b in BANDS:
            st = stored_sum[(stored_sum.site == s) & (stored_sum.population == "primary (plain)") &
                            (stored_sum.band == b) & (stored_sum.condition == "D_bad") & (stored_sum.readout == "by")]
            if abs(float(st.est.iloc[0]) - float(apt.summarize(Av, wfull[b], full[b])[0, 0])) > 1e-4:
                problems.append(f"stored D_bad(by) site {s} band {b} differs from transfer_summary")
        readings[int(s)] = {}
        for c in CONDS:
            pdir = res / f"passive_{c}"
            ck = json.loads((pdir / "run_meta.json").read_text())["checks"][str(s)]
            if not bar_ok(ck, stored_meta[str(s)]):
                problems.append(f"passive checks for {c} at site {s}: {ck}")
            nat = pd.read_parquet(pdir / "natural.parquet")
            Ar = d_bad(pdir, s, plan, items, nat, bt, pair_ix, ctx_ix, das_q)
            Dr = {b: apt.summarize(Ar, wps[c][b], pixs[c][b])[:, 0] for b in BANDS}
            Dv = {b: apt.summarize(Av, wps[c][b], pixs[c][b])[:, 0] for b in BANDS}
            for k in range(3):  # per seed (split k), point estimates
                Ak = d_bad(pdir, s, plan, items, nat, bt, pair_ix, ctx_ix, das_q, split=k)
                Avk = d_bad(a.stored_passive, s, plan, items, nat_v, bt, pair_ix, ctx_ix, das_q, split=k)
                seed_rows.append({"site": s, "cond": c, "seed": k, **{
                    f"{b}_{w}": float(apt.summarize(A_, wps[c][b], pixs[c][b])[0, 0])
                    for b in ("head", "all") for w, A_ in (("rare", Ak), ("v", Avk))}})
            cs = C[(C.site == s) & (C.cond == c)].abs_cos
            iv, ir, idiff = iia(res, s, c, stored_items, a.n_boot, a.seed)
            dh, rh = apt.ci(Dr["head"]), apt.ci(Dr["head"] / Dv["head"])
            readings[int(s)][c] = {
                "reading": label(dh, rh) if s in TEST_SITES else "descriptive (signed retention)",
                "D_head": dh, "Dv_head": apt.ci(Dv["head"]), "ratio_head": rh,
                "half_contrast_head": apt.ci(Dr["head"] - 0.5 * Dv["head"]),
                "ratio_all": apt.ci(Dr["all"] / Dv["all"]), "iia_rare": ir, "iia_v": iv, "iia_diff": idiff,
                "median_abs_cos": float(cs.median()), "low_active_iia": bool(ir["est"] < 0.6)}
            for b in BANDS:
                rows.append({"site": s, "cond": c, "band": b, "n_pairs": len(pixs[c][b]),
                             "median_abs_cos": cs.median(), "min_abs_cos": cs.min(), "max_abs_cos": cs.max(),
                             "iia_rare": ir["est"], "iia_v": iv["est"],
                             **{f"D_{k}": v for k, v in apt.ci(Dr[b]).items()},
                             **{f"Dv_{k}": v for k, v in apt.ci(Dv[b]).items()},
                             **{f"ratio_{k}": v for k, v in apt.ci(Dr[b] / Dv[b]).items()}})
    overall = {}
    for c in CONDS:
        labs = [readings[s][c]["reading"] for s in TEST_SITES if s in readings]
        pos = [readings[s][c]["D_head"]["lo95"] > 0 for s in TEST_SITES if s in readings]
        overall[c] = ("invalid" if problems else "transfers" if labs.count("transfers") == len(TEST_SITES) else
                      "no positive transfer" if not any(pos) else "mixed")
    S = pd.DataFrame(rows)
    S.to_csv(res / "summary.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(res / "per_seed.csv", index=False)
    (res / "readings.json").write_text(json.dumps({"overall": overall, "sites": readings, "problems": problems,
                                                   "excluded_overlap_pairs": excl, "checks": checks},
                                                  indent=1, default=float) + "\n")
    f3 = lambda d: f"{d['est']:.2f} [{d['lo95']:.2f}, {d['hi95']:.2f}]"
    L = ["# Reverse frequency transfer: DAS trained only on rare verbs (Pythia-1.4B)", "",
         "Rules: `reports/pythia/das_reverse_freq_rules.md` (with its Nov 23 amendment). Scripts: "
         "`scripts/run_das_reverse_freq.py`, `scripts/run_passive_test.py`, `scripts/analyze_das_reverse_freq.py`.", "",
         "**Overall (sites 8 and 12, Head band): " + "; ".join(f"{c} {overall[c]}" for c in CONDS) + ".**"
         + (f" Problems: {'; '.join(problems)}" if problems else ""), "",
         f"Training pairs: {checks['n_rare_pairs']} rare plain pairs (F keeps {checks['n_kept_F']} after the behavior "
         f"filter; dropped: {', '.join(checks['dropped_F']) or 'none'}); B, both sides rare: {checks['n_both_rare']} "
         f"pairs, {checks['n_kept_B']} after the filter. Failing verbs (items on the class side, of 7): "
         + (", ".join(f"{r['verb']} ({'T' if r['cls'] == 1 else 'I'}, {r['n_correct']})" for r in checks["failing_verbs"])
            or "none") + ".",
         "Excluded overlap pairs (primary): " + "; ".join(f"{c}: {', '.join(v) or 'none'}" for c, v in excl.items()), ""]
    for s, rr in readings.items():
        e = ext.get(str(s), {})
        ref = (f" References: v's stored-pair seed range {e['seed_range_cos'][0]:.3f}–{e['seed_range_cos'][1]:.3f}, "
               f"P {e['median_abs_cos']['P']:.3f}." if e else "")
        for c, r in rr.items():
            L.append(f"- **Site {s}, {c}: {r['reading']}.** D_bad(by) on Head {f3(r['D_head'])} (v "
                     f"{f3(r['Dv_head'])}; ratio {f3(r['ratio_head'])}; D_rare − 0.5·D_v {f3(r['half_contrast_head'])}); "
                     f"all bands ratio {f3(r['ratio_all'])}. Held-out active IIA on v's frequent pairs "
                     f"{f3(r['iia_rare'])} (v {f3(r['iia_v'])}; difference {f3(r['iia_diff'])}). Median |cos| with v "
                     f"{r['median_abs_cos']:.3f}." + (ref if c == "F" else ""))
    L += ["", "| site | cond | band | pairs | median \\|cos\\| | D_bad(by) rare | D_bad(by) v | ratio |",
          "|---:|---|---|---:|---|---|---|---|"]
    for _, q in S.iterrows():
        L.append(f"| {q.site} | {q.cond} | {q.band} | {q.n_pairs} | {q.median_abs_cos:.3f} | "
                 f"{q.D_est:.2f} [{q.D_lo95:.2f}, {q.D_hi95:.2f}] | {q.Dv_est:.2f} [{q.Dv_lo95:.2f}, {q.Dv_hi95:.2f}] | "
                 f"{q.ratio_est:.2f} [{q.ratio_lo95:.2f}, {q.ratio_hi95:.2f}] |")
    L += ["", "Per seed (point estimates, plan rows of split k): `results/pythia/das_reverse_freq/per_seed.csv`.",
          "Forward direction (v on Tail / XTail, R2-ptest) is the `D_bad(by) v` column at those bands.", "",
          "## Checks", "", "```", json.dumps(checks, indent=1, default=float), "```", ""]
    Path(a.report).write_text("\n".join(L))
    print("\n".join(L[:18]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/pythia/das_reverse_freq")
    ap.add_argument("--stored-root", default="results/das_round2")
    ap.add_argument("--stored-passive", default="results/das_round2/passive_test")
    ap.add_argument("--ext-results", default="results/stability_ext/pythia")
    ap.add_argument("--items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--plan", default="data/das_round2/passive_test/plan.csv.gz")
    ap.add_argument("--report", default="reports/pythia/das_reverse_freq.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
