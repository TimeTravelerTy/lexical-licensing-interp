#!/usr/bin/env python3
"""Expanded eval set: behavior and passive transfer with v and u_md (rules: reports/expanded_eval_rules.md).

Populations (pairs; primary = plain bad verb):
  original plain / new plain / pooled plain; original prep / new prep / pooled prep (separate group);
  robustness (new plain): without selection-flagged intransitives, without FreqBLiMP-source transitives, single-token.
Passive: analyze_passive_test.transfer() (stored estimator; D_bad, natural gap, D / gap) for v and u_md at each site,
with the same bootstrap draws for both. Behavior: per-pair natural log P(" by") gap (T - I), share positive, and the
slope of the pair gap on the intransitive participle Zipf (pair bootstrap). Checks: v on the original pairs
reproduces the stored transfer_summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import analyze_passive_test as apt
from analyze_das_expanded_stability import d_bad

REPRO_TOL = 0.02  # max |dD_bad(by)| vs the stored run, original plain pairs (rules amendment 1)


def pops_for(items, meta):
    pr = items.drop_duplicates("pair_id").set_index("pair_id")
    m = meta.set_index("pair_id")
    orig, new = pr.source == "original", pr.source == "new"
    plain, prep = pr.bad_class == "plain", pr.bad_class == "prep_object"
    sel = pr.index.to_series().map(m.intrans_selection).notna() & new
    fbt = pr.index.to_series().map(m.trans_source).eq("freqblimp") & new
    single = pr.index.to_series().map(m.single_token).fillna(False).astype(bool) & new
    return pr, {
        "original plain": orig & plain, "new plain": new & plain, "pooled plain": plain,
        "original prep": orig & prep, "new prep": new & prep, "pooled prep": prep,
        "new plain, no selection-flagged": new & plain & ~sel,
        "new plain, no FreqBLiMP transitives": new & plain & ~fbt,
        "new plain, single-token": new & plain & single,
    }


def behavior(items, P, nat, pr, pops, zipf, n_boot, seed):
    pidx = dict(zip(P.prompt, P.pid))
    by = nat.set_index("pid")["by"]
    g = items[["pair_id"]].copy()
    g["gap"] = by.loc[items.good_prompt.map(pidx)].to_numpy() - by.loc[items.bad_prompt.map(pidx)].to_numpy()
    pg = g.groupby("pair_id").gap.mean()
    rng = np.random.default_rng(seed)
    rows = []
    for name, mask in pops.items():
        ids = pr.index[mask.to_numpy()]
        y = pg.loc[ids].to_numpy()
        x = zipf.reindex(ids).to_numpy()
        ok = np.isfinite(x)
        b = [np.polyfit(x[ok], y[ok], 1)[0]] if ok.sum() > 2 else [np.nan]
        for _ in range(n_boot if ok.sum() > 2 else 0):
            j = rng.integers(0, ok.sum(), ok.sum())
            b.append(np.polyfit(x[ok][j], y[ok][j], 1)[0])
        b = np.array(b)
        rows.append({"population": name, "n_pairs": len(ids), "mean_pair_gap": float(y.mean()),
                     "share_positive": float((y > 0).mean()), "slope_on_intrans_zipf": float(b[0]),
                     "slope_lo95": float(np.nanpercentile(b[1:], 2.5)) if len(b) > 1 else np.nan,
                     "slope_hi95": float(np.nanpercentile(b[1:], 97.5)) if len(b) > 1 else np.nan})
    return pd.DataFrame(rows), pg


def run(a):
    data, out = Path(a.data_dir), Path(a.out_dir)
    items = pd.read_csv(data / "items.csv")
    P = pd.read_csv(data / "prompts.csv")
    plan = pd.read_csv(data / "plan.csv.gz", dtype={c: "category" for c in ("item_id", "side", "cond", "donor_pair")})
    meta = pd.read_csv(a.pairs_meta)
    pr, pops = pops_for(items, meta)
    sites = [int(s) for s in a.sites.split(",")]
    # intransitive participle Zipf: new pairs from pairs_meta, originals from the v2 pairs
    v2 = pd.read_json(a.v2_pairs, lines=True).drop_duplicates("bad_verb").set_index("bad_verb").bad_form_zipf
    zipf = pd.Series({p: (meta.set_index("pair_id").intrans_participle_zipf.get(p) if r.source == "new"
                          else v2.get(r.bad_part)) for p, r in pr.iterrows()}, dtype=float)
    res, checks = {}, {}
    for inst, rdir in (("v", a.v_dir), ("umd", a.umd_dir)):
        nat = pd.read_parquet(Path(rdir) / "natural.parquet")
        if inst == "v":
            B, pair_gap = behavior(items, P, nat, pr, pops, zipf, a.n_boot, a.seed)
            B.to_csv(out / "behavior.csv", index=False)
            pair_gap.rename("nat_by_gap").to_frame().join(zipf.rename("intrans_participle_zipf")).to_csv(out / "pair_gaps.csv")
        args = SimpleNamespace(out_dir=rdir, _P=P)
        rows = []
        for s in sites:
            bt = apt.Boot(items.context_id.nunique(), a.n_boot, a.seed)
            rows += apt.transfer(args, s, items, plan, nat, bt, pr, pops, apt.site_delta(Path(a.root), s))
        T = pd.DataFrame(rows)
        T = T[(T.band == "all") & T.readout.isin(["by", "O"]) & T.condition.isin(["D_bad", "nat_gap"])]
        T.insert(0, "instrument", inst)
        res[inst] = T
        rm = json.loads((Path(rdir) / "run_meta.json").read_text())
        checks[f"{inst}_passive_checks"] = rm["checks"]
    T = pd.concat(res.values(), ignore_index=True)
    T.to_csv(out / "transfer.csv", index=False)
    # paired new vs original (one set of context and donor draws; pair weights within each population)
    pair_ix = {p: i for i, p in enumerate(pr.index)}
    ctx_ix = {c: i for i, c in enumerate(sorted(items.context_id.unique()))}
    das_q = sorted(plan[plan.cond == "T"].donor_pair.dropna().astype(str).unique())
    crow = []
    for inst, rdir in (("v", a.v_dir), ("umd", a.umd_dir)):
        nat = pd.read_parquet(Path(rdir) / "natural.parquet")
        for s in sites:
            bt = apt.Boot(items.context_id.nunique(), a.n_boot, a.seed)
            Ap = d_bad(rdir, s, plan, items, nat, bt, pair_ix, ctx_ix, das_q)
            Dp = {}
            for name in ("new plain", "original plain"):
                ix = np.flatnonzero(pops[name].to_numpy())
                Dp[name] = apt.summarize(Ap, bt.pair_weights(pr.band.to_numpy()[ix]), ix)[:, 0]
            dn, do = Dp["new plain"], Dp["original plain"]
            cn, cc, cr = apt.ci(dn), apt.ci(dn - 0.5 * do), apt.ci(dn / do)
            lab = ("descriptive" if s not in [int(x) for x in a.rule_sites.split(",")] else
                   "generalizes" if cn["lo95"] > 0 and cc["lo95"] > 0 else
                   "criterion not met: weaker" if cc["hi95"] < 0 else "criterion not met: inconclusive")
            crow.append({"instrument": inst, "site": s, "reading": lab, **{f"Dnew_{k}": v for k, v in cn.items()},
                         **{f"Dorig_{k}": v for k, v in apt.ci(do).items()},
                         **{f"ratio_{k}": v for k, v in cr.items()}, **{f"half_{k}": v for k, v in cc.items()}})
    Cn = pd.DataFrame(crow)
    Cn.to_csv(out / "new_vs_original.csv", index=False)
    # reproduction: v on the original plain pairs vs the stored transfer_summary's original-only primary population
    st = pd.read_csv(Path(a.stored_passive) / "transfer_summary.csv")
    st = st[(st.population == "primary, original pairs only") & (st.band == "all") & (st.condition == "D_bad")
            & (st.readout == "by")].set_index("site").est
    mine = T[(T.instrument == "v") & (T.population == "original plain") & (T.condition == "D_bad")
             & (T.readout == "by")].set_index("site").est
    checks["v_original_D_bad_by_vs_stored_max_abs"] = float((mine - st.reindex(mine.index)).abs().max())
    checks["reproduction_ok"] = checks["v_original_D_bad_by_vs_stored_max_abs"] <= REPRO_TOL
    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=float) + "\n")

    f3 = lambda r, p="": f"{r[p + 'est']:.2f} [{r[p + 'lo95']:.2f}, {r[p + 'hi95']:.2f}]"
    L = [f"# Expanded eval set: behavior and passive transfer ({a.model_name})", "",
         ("" if checks["reproduction_ok"] else f"**Reproduction check FAILED (> {REPRO_TOL} nats): do not read.**"), "",
         "Rules: `reports/expanded_eval_rules.md`. Scripts: `scripts/build_expanded_eval.py`, `scripts/build_passive_test.py`, "
         "`scripts/run_passive_test.py`, `scripts/analyze_expanded_eval.py`.", "",
         f"Reproduction: v's D_bad(by) on the original plain pairs differs from the stored run by at most "
         f"{checks['v_original_D_bad_by_vs_stored_max_abs']:.2g} nats.", "", "## Behavior (natural passive *by* gap)", "",
         "| population | pairs | mean pair gap | share of pairs > 0 | slope on intransitive Zipf |", "|---|---:|---|---|---|"]
    for r in B.itertuples():
        L.append(f"| {r.population} | {r.n_pairs} | {r.mean_pair_gap:.2f} | {r.share_positive:.2f} | "
                 f"{r.slope_on_intrans_zipf:+.2f} [{r.slope_lo95:+.2f}, {r.slope_hi95:+.2f}] |")
    L += ["", "## Passive transfer, D_bad(*by*) (nats) and D / natural gap", "",
          "| site | population | pairs | natural gap | v: D_bad(by) | v: D/gap | u_md: D_bad(by) | u_md / v | v: O class |",
          "|---:|---|---:|---|---|---|---|---|---|"]
    for s in sites:
        for pop in pops:
            q = lambda inst, cond, rd: T[(T.instrument == inst) & (T.site == s) & (T.population == pop)
                                         & (T.condition == cond) & (T.readout == rd)]
            dv, du, ng, oc = q("v", "D_bad", "by"), q("umd", "D_bad", "by"), q("v", "nat_gap", "by"), q("v", "D_bad", "O")
            if dv.empty:
                continue
            dv, du, ng, oc = dv.iloc[0], du.iloc[0], ng.iloc[0], oc.iloc[0]
            L.append(f"| {s} | {pop} | {dv.n_pairs} | {f3(ng)} | {f3(dv)} | {f3(dv, 'frac_natgap_')} | {f3(du)} | "
                     f"{du.est / dv.est:.2f} | {oc.get('class', '')} |")
    L += ["", "## Declared comparison 1: does transfer generalize to new plain verbs?", "",
          "Paired (one set of context and donor draws). Reading at the rule sites; other sites descriptive.", "",
          "| instrument | site | D_bad(by) new | D_bad(by) original | ratio new / original | D_new − 0.5·D_orig | reading |",
          "|---|---:|---|---|---|---|---|"]
    for r in Cn.itertuples():
        g = lambda k: f"{getattr(r, k + '_est'):.2f} [{getattr(r, k + '_lo95'):.2f}, {getattr(r, k + '_hi95'):.2f}]"
        L.append(f"| {r.instrument} | {r.site} | {g('Dnew')} | {g('Dorig')} | {g('ratio')} | {g('half')} | {r.reading} |")
    L += ["", "## Checks", "", "```", json.dumps(checks, indent=1, default=float), "```", ""]
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/verb_expansion/eval/passive_test")
    ap.add_argument("--pairs-meta", default="data/verb_expansion/eval/pairs_meta.csv")
    ap.add_argument("--v2-pairs", default="data/matched_passives_v2/pairs.jsonl")
    ap.add_argument("--root", default="results/das_round2")
    ap.add_argument("--stored-passive", default="results/das_round2/passive_test")
    ap.add_argument("--v-dir", default="results/pythia/expanded_eval/passive_v")
    ap.add_argument("--umd-dir", default="results/pythia/expanded_eval/passive_umd")
    ap.add_argument("--sites", default="8,12,17")
    ap.add_argument("--rule-sites", default="8,12")
    ap.add_argument("--out-dir", default="results/pythia/expanded_eval")
    ap.add_argument("--model-name", default="Pythia-1.4B")
    ap.add_argument("--report", default="reports/pythia/expanded_eval_passive.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
