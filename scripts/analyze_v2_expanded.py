#!/usr/bin/env python3
"""v2 on the expanded eval set (rules: reports/v2_retrain_rules.md, amendment Nov 25; criteria: reports/criteria_audit.md).

Populations come from data/das_v2/expanded_eval_pairs.csv. eval_v2 means every expanded pair except v2's 44 new training pairs; the 8 originals with v's stored training verbs are fold-protected.
  eval plain (primary), eval prep (separate group), eval original plain, eval new plain,
  v2 training pairs (in-sample, descriptive), all plain (268-set; v's population, descriptive for v2).

For v2 at every site (passive chunks in results/{m}/das_v2/expanded_eval/passive_*), and for v at its stored expanded
sites (results/{m}/expanded_eval/passive_v), on the same plan, natural readouts and bootstrap seed:
  transfer.csv  D_bad and nat_gap per readout and population, with D / natural gap (CIs from joint draws), and the
                stored class under the instrument's own delta and under delta_v
  paired.csv    v2 - v and v2 / v on D_bad(by) and D_bad(O), one set of draws, at the shared sites
  checks.json   natural readouts equal across runs; v's stored expanded transfer.csv reproduced
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import analyze_passive_test as apt

CFG = {
    "pythia": dict(data="data/verb_expansion/eval/passive_test", stored_root="results/das_round2", v_sites=[8, 12, 17]),
    "gemma": dict(data="results/gemma/expanded_eval/passive_test", stored_root="results/gemma/das/transfer",
                  v_sites=[5, 8, 12, 15]),
    "olmo": dict(data="results/olmo/expanded_eval/passive_test", stored_root="results/olmo/das/transfer",
                 v_sites=[9, 11, 12, 13]),
}


def classify_by(r):
    """Audited *by* class (reports/criteria_audit.md): margin 0.2 x the natural gap, tested on joint-draw CIs."""
    if r["minus20gap_lo95"] > 0:
        return "RISE"
    if r["hi95"] < 0:
        return "FALL"
    if r["lo95"] > 0:
        return "positive, magnitude unresolved"
    if r["minus20gap_hi90"] < 0 and r["plus20gap_lo90"] > 0:
        return "NO RISE"
    return "unresolved"


def classify_o(r, d):
    """Audited O class: margin 0.2 x the instrument's active O effect (the stored delta), tested on the CI."""
    if r["lo95"] > d:
        return "RISE"
    if r["hi95"] < -d:
        return "FALL"
    if r["lo95"] > 0:
        return "positive, magnitude unresolved"
    if r["lo90"] > -d and r["hi90"] < d:
        return "NO RISE"
    return "unresolved"


def outcome(o, by):
    if o == "RISE" and by == "RISE":
        return "mixed"
    if by == "RISE" and o == "NO RISE":
        return "abstract"
    if o == "RISE":
        return {"NO RISE": "surface, by unchanged", "FALL": "surface, by suppressed"}.get(by, "surface, by unresolved")
    return "unresolved"


def staging(oc):
    ab = sorted(s for s, v in oc.items() if v == "abstract")
    su = sorted(s for s, v in oc.items() if v.startswith("surface"))
    return {"abstract": ab, "surface": su, "holds": bool(ab) and bool(su) and max(ab) < min(su)}


def populations(items, frozen, old_eval):
    pr = items.drop_duplicates("pair_id").set_index("pair_id")
    f = frozen.set_index("pair_id").reindex(pr.index)
    assert f.eval_v2.notna().all(), "pairs missing from the frozen list"
    ev, plain, prep = f.eval_v2.astype(bool), pr.bad_class == "plain", pr.bad_class == "prep_object"
    orig, new = pr.source == "original", pr.source == "new"
    pops = {"eval plain": ev & plain, "eval prep": ev & prep, "eval original plain": ev & plain & orig,
            "eval new plain": ev & plain & new,
            "eval new plain, lemma-fresh": ev & plain & new & ~(pr.good_lemma.isin(old_eval) | pr.bad_lemma.isin(old_eval)),
            "v2 training pairs (in-sample)": ~ev & plain, "all plain (268-set)": plain}
    return pr, pops


def site_runs(xdir):
    m = {}
    for d in sorted(Path(xdir).glob("passive_*")):
        for p in d.glob("patches_site*.parquet"):
            s = int(p.stem.replace("patches_site", ""))
            assert s not in m, f"site {s} in two runs"
            m[s] = d
    return m


def main(a):
    cfg = CFG[a.model]
    data = Path(cfg["data"])
    items = pd.read_csv(data / "items.csv", low_memory=False)
    P = pd.read_csv(data / "prompts.csv")
    plan = pd.read_csv(data / "plan.csv.gz", dtype={c: "category" for c in ("item_id", "side", "cond", "donor_pair")})
    frozen = pd.read_csv(a.frozen)
    old = pd.read_csv(a.old_eval)
    pr, pops = populations(items, frozen, set(old.trans_lemma) | set(old.intrans_lemma))
    v2root, dvroot = Path(f"results/{a.model}/das_v2"), Path(f"results/{a.model}/das_v2_dv")
    xdir = v2root / "expanded_eval"
    runs = {"v2": site_runs(xdir), "v": {s: Path(f"results/{a.model}/expanded_eval/passive_v") for s in cfg["v_sites"]}}
    if a.sites:
        keep = {int(s) for s in a.sites.split(",")}
        runs = {k: {s: d for s, d in r.items() if s in keep} for k, r in runs.items()}
    checks = {"n_pairs": {k: int(v.sum()) for k, v in pops.items()}}
    nat0 = pd.read_parquet(runs["v"][cfg["v_sites"][0]] / "natural.parquet").set_index("pid").sort_index()
    num = [c for c in nat0.columns if pd.api.types.is_numeric_dtype(nat0[c])]
    checks["natural_max_abs_vs_v_run"] = {
        str(d): float(np.nanmax(np.abs(pd.read_parquet(d / "natural.parquet").set_index("pid").sort_index()[num].to_numpy(float)
                                       - nat0[num].to_numpy(float)))) for d in sorted(set(runs["v2"].values()))}
    nat = nat0.reset_index()
    rows = []
    for inst, rr in runs.items():
        for s, d in sorted(rr.items()):
            bt = apt.Boot(items.context_id.nunique(), a.n_boot, a.seed)
            own = apt.site_delta(v2root if inst == "v2" else Path(cfg["stored_root"]), s)
            dv = apt.site_delta(dvroot if inst == "v2" else Path(cfg["stored_root"]), s)
            for r in apt.transfer(SimpleNamespace(out_dir=d, _P=P), s, items, plan, nat, bt, pr, pops, own):
                if r.get("band") != "all":
                    continue
                r = dict(r)
                r.update(instrument=inst, delta_own=own, delta_v=dv)
                if r.get("condition") == "D_bad" and "est" in r:
                    r["class_own"], r["class_delta_v"] = apt.classify(r, own), apt.classify(r, dv)
                    if r["readout"] == "by":
                        r["class_audit"] = classify_by(r)
                    elif r["readout"] == "O":
                        r["class_audit"], r["class_audit_delta_v"] = classify_o(r, own), classify_o(r, dv)
                rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(xdir / "transfer.csv", index=False)
    # audited outcomes and staging per instrument and population (O margin: own delta; and delta_v)
    oc_rows, s1 = [], {}
    for (inst, pop), g in T[T.condition == "D_bad"].groupby(["instrument", "population"]):
        for col in ("class_audit", "class_audit_delta_v"):
            oc = {}
            for s_, gs in g.groupby("site"):
                ro, rb = gs[gs.readout == "O"].iloc[0], gs[gs.readout == "by"].iloc[0]
                if pd.isna(ro.get(col)):
                    continue
                oc[int(s_)] = outcome(ro[col], rb["class_audit"])
                oc_rows.append({"instrument": inst, "population": pop, "o_margin": col, "site": int(s_),
                                "O": ro[col], "by": rb["class_audit"], "outcome": oc[int(s_)]})
            if oc:
                s1[f"{inst} | {pop} | {col}"] = staging(oc)
    pd.DataFrame(oc_rows).to_csv(xdir / "outcomes.csv", index=False)
    (xdir / "staging.json").write_text(json.dumps(s1, indent=1) + "\n")
    # reproduction of v's stored expanded transfer.csv (population "pooled plain" = all plain here)
    st = pd.read_csv(Path(f"results/{a.model}/expanded_eval/transfer.csv"))
    st = st[(st.instrument == "v") & (st.population == "pooled plain") & (st.condition == "D_bad")].set_index(["site", "readout"]).est
    mine = T[(T.instrument == "v") & (T.population == "all plain (268-set)") & (T.condition == "D_bad")].set_index(["site", "readout"]).est
    j = mine.to_frame("mine").join(st.rename("stored"), how="inner")
    checks["v_stored_expanded_reproduced_max_abs"] = float((j.mine - j.stored).abs().max())
    # paired v2 - v on one set of draws (eval plain)
    pair_ix = {p: i for i, p in enumerate(pr.index)}
    ctx_ix = {c: i for i, c in enumerate(sorted(items.context_id.unique()))}
    das_q = sorted(plan[plan.cond == "T"].donor_pair.dropna().astype(str).unique())
    prow = []
    for s in sorted(set(runs["v"]) & set(runs["v2"])):
        bt = apt.Boot(items.context_id.nunique(), a.n_boot, a.seed)
        for pop in ("eval plain", "eval prep"):
            pix = np.flatnonzero(pops[pop].to_numpy())
            wp = bt.pair_weights(pr.band.to_numpy()[pix])
            D = {k: apt.summarize(d_bad_all(runs[k][s], s, plan, items, nat, bt, pair_ix, ctx_ix, das_q), wp, pix)
                 for k in ("v2", "v")}
            for ri, rd in enumerate(("by", "O")):
                x2, x1 = D["v2"][:, ri], D["v"][:, ri]
                for k, x in (("v2", x2), ("v", x1)):  # same estimate as transfer.csv's D_bad for this cell
                    t = T[(T.instrument == k) & (T.population == pop) & (T.condition == "D_bad") & (T.site == s) & (T.readout == rd)]
                    if len(t):
                        dev = abs(float(x[0]) - float(t.est.iloc[0]))
                        checks["paired_matches_transfer_max_abs"] = max(checks.get("paired_matches_transfer_max_abs", 0.0), dev)
                prow.append({"site": s, "population": pop, "readout": rd,
                             **{f"v2_{k}": q for k, q in apt.ci(x2).items()}, **{f"v_{k}": q for k, q in apt.ci(x1).items()},
                             **{f"diff_{k}": q for k, q in apt.ci(x2 - x1).items()},
                             **{f"ratio_{k}": q for k, q in apt.ci(x2 / x1).items()}})
    pd.DataFrame(prow).to_csv(xdir / "paired.csv", index=False)
    (xdir / "checks.json").write_text(json.dumps(checks, indent=1) + "\n")
    print(json.dumps(checks, indent=1))


def d_bad_all(pdir, site, plan, items, nat, bt, pair_ix, ctx_ix, das_q):
    """D_bad per pair and draw for log P(" by") and log P(O): [B+1, P, 2]."""
    pat = pd.read_parquet(Path(pdir) / f"patches_site{site}.parquet")
    assert (pat.row.to_numpy() == plan.row.to_numpy()).all()
    df = plan[["item_id", "side", "cond", "base", "donor_pair"]].copy()
    nn = nat.set_index("pid").loc[plan.base.to_numpy()]
    for c in ("by", "O"):
        df[c] = pat[c].to_numpy() - nn[c].to_numpy()
    df = df.join(items.set_index("item_id")[["pair_id", "context_id"]], on="item_id")
    d = df[(df.side == "bad") & df.cond.isin(["T", "I"])].assign(dq=lambda x: x.donor_pair.astype(str))
    qix = {q: i for i, q in enumerate(das_q)}
    vT, mT = apt.cells(d[d.cond == "T"], pair_ix, ctx_ix, qix, len(das_q), ["by", "O"])
    vI, mI = apt.cells(d[d.cond == "I"], pair_ix, ctx_ix, qix, len(das_q), ["by", "O"])
    assert (mT == mI).all()
    return apt.pair_values(vT - vI, mT, bt.wc, bt.donor_weights("das", len(das_q)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(CFG))
    ap.add_argument("--frozen", default="data/das_v2/expanded_eval_pairs.csv")
    ap.add_argument("--old-eval", default="data/verb_expansion/eval_pairs.csv", help="the earlier 19 eval pairs (lemma-fresh row)")
    ap.add_argument("--sites", default="", help="restrict to these sites (smoke)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    main(ap.parse_args())
