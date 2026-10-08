#!/usr/bin/env python3
"""Get-passive transfer: paired comparison with was-passives (round3_plan.md, B6).

Both frames use the same plan rows (same bases' pids, donors, bases), so for
every T / I row the change after patching can be compared row by row:
  D_frame = mean Delta after T donors - mean Delta after I donors (bad bases),
  Delta = patched - unpatched readout in that frame.
D_got, D_was and D_got - D_was use the passive test's three-way bootstrap
(base pairs within band, contexts, donor pairs; 2,000 draws, seed 17) on the
same resamples. Populations: the 64 primary pairs and the declared eventive
subset (48 pairs). The natural good - bad gaps in each frame are reported on
the same bootstrap. Decision rule (site 8): `round3_plan.md`, B6.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_passive_test import Boot, cells, ci, classify, frame_z, pair_values, populations, site_delta, summarize

READ = ("M", "O", "I", "det", "pron", "refl", "by", "dot", "the", "him")
EXCLUDE_EVENTIVE = ("earned", "bet", "exerted", "awed", "forested", "uttered", "blurted", "eschewed", "precluded",
                    "wadded", "larded", "tithed", "blabbed", "blasphemed", "disabused", "edified")


def frame_rows(out_dir, site, plan, sel):
    nat = pd.read_parquet(Path(out_dir) / "natural.parquet").set_index("pid")
    pat = pd.read_parquet(Path(out_dir) / f"patches_site{site}.parquet")
    assert (pat.row.to_numpy() == plan.row.to_numpy()).all()
    p = plan[sel]
    return pat.loc[sel, list(READ)].to_numpy() - nat.loc[p.base.to_numpy(), list(READ)].to_numpy()


def nat_gap(out_dir, items, P):
    nat = pd.read_parquet(Path(out_dir) / "natural.parquet").set_index("pid")
    pidx = dict(zip(P.prompt, P.pid))
    g = items[["pair_id", "context_id"]].copy()
    g[list(READ)] = nat.loc[items.good_prompt.map(pidx), list(READ)].to_numpy() - \
        nat.loc[items.bad_prompt.map(pidx), list(READ)].to_numpy()
    return g


def run(args):
    root = Path(args.root)
    items_w = pd.read_csv(args.items_was)
    items_g = pd.read_csv(args.items_got)
    Pw, Pg = pd.read_csv(args.prompts_was), pd.read_csv(args.prompts_got)
    assert (items_w.item_id == items_g.item_id).all()
    plan = pd.read_csv(args.plan, dtype={c: "category" for c in ("item_id", "side", "cond", "donor_pair")})
    pr, pops = populations(items_w)
    prim = pops["primary (plain)"]
    eventive = prim & ~pr.good_part.isin(EXCLUDE_EVENTIVE)
    popsel = {"primary (64)": prim, f"eventive subset ({int(eventive.sum())})": eventive}
    sel = (plan.side == "bad").to_numpy() & plan.cond.isin(["T", "I"]).to_numpy()
    df = plan[sel][["item_id", "cond", "donor_pair"]].copy()
    df = df.join(items_w.set_index("item_id")[["pair_id", "context_id"]], on="item_id")
    pair_ix = {p: i for i, p in enumerate(pr.index)}
    ctx_ix = {c: i for i, c in enumerate(sorted(items_w.context_id.unique()))}
    das_q = sorted(df.donor_pair.dropna().unique())
    qix = {q: i for i, q in enumerate(das_q)}
    rows = []
    for site in [int(s) for s in args.sites.split(",")]:
        was_dir = next(d for d in args.was_dirs.split(",") if (Path(d) / f"patches_site{site}.parquet").exists())
        dW = frame_rows(was_dir, site, plan, sel)
        dG = frame_rows(args.got_dir, site, plan, sel)
        bt = Boot(items_w.context_id.nunique(), args.n_boot, args.seed)
        A = {}
        for name, d in (("was", dW), ("got", dG), ("got_minus_was", dG - dW)):
            x = df.assign(dq=df.donor_pair, **{k: d[:, j] for j, k in enumerate(READ)})
            vT, mT = cells(x[x.cond == "T"], pair_ix, ctx_ix, qix, len(das_q), READ)
            vI, mI = cells(x[x.cond == "I"], pair_ix, ctx_ix, qix, len(das_q), READ)
            assert (mT == mI).all()
            A[name] = pair_values(vT - vI, mT, bt.wc, bt.donor_weights("das", len(das_q)))
        for fr, out_dir, items, P in (("was", was_dir, items_w, Pw), ("got", args.got_dir, items_g, Pg)):
            g = nat_gap(out_dir, items, P)
            v, m = cells(g.assign(dq=0), pair_ix, ctx_ix, {0: 0}, 1, READ)
            A[f"natgap_{fr}"] = pair_values(v, m, bt.wc, np.ones((bt.B + 1, 1), np.float32))
        delta = site_delta(root, site)
        for pname, mask in popsel.items():
            pix = np.flatnonzero(mask.to_numpy())
            wp = bt.pair_weights(pr.band.to_numpy()[pix])
            S = {k: summarize(a, wp, pix) for k, a in A.items()}
            for k, s in S.items():
                for j, rd in enumerate(READ):
                    row = {"site": site, "population": pname, "n_pairs": len(pix), "quantity": k, "readout": rd,
                           "delta": delta, **ci(s[:, j])}
                    if k in ("was", "got") and rd in ("O", "by", "dot"):
                        row["class"] = classify(row, delta)
                    rows.append(row)
        print(f"site {site} done", flush=True)
    res = pd.DataFrame(rows)
    out = Path(args.out_dir)
    res.to_csv(out / "got_vs_was_summary.csv", index=False)
    # natural projection: good - bad z gap per frame vs the same verbs' active gap (mean over primary pairs)
    pf = pd.read_csv(root / "final_strict" / "pairs_folds.csv")
    das = pd.read_csv(root / "final_strict" / "items.csv")
    prim_ids = list(pr.index[prim.to_numpy()])
    pj = []
    for site in [int(s) for s in args.sites.split(",")]:
        was_dir = next(d for d in args.was_dirs.split(",") if (Path(d) / f"projections_site{site}.parquet").exists())
        row = {"site": site}
        for fr, d, items, P in (("was", was_dir, items_w, Pw), ("got", args.got_dir, items_g, Pg)):
            it, at = frame_z(Path(d) / f"projections_site{site}.parquet", items, P, pf, das, pr)
            gp = it[it.pair_id.isin(prim_ids)].groupby("pair_id")["diff"].mean().mean()
            ga = at[at.pair_id.isin(prim_ids)].groupby("pair_id")["diff"].mean().mean()
            row.update({f"{fr}_gap": gp, f"{fr}_ratio": gp / ga, "active_gap": ga})
        pj.append(row)
    pj = pd.DataFrame(pj)
    pj.to_csv(out / "got_vs_was_projection.csv", index=False)
    write_report(args, res, pj)


def outcome(o, b):
    if o == "RISE":
        return "mixed" if b == "RISE" else "surface"
    if o == "NO RISE":
        return "abstract" if b == "RISE" else ("nothing moves" if b == "NO RISE" else "unresolved")
    return "unresolved"


def write_report(args, res, pj):
    get = lambda pop, site, q, rd: res[(res.population == pop) & (res.site == site) & (res.quantity == q)
                                       & (res.readout == rd)].iloc[0]
    f3 = lambda r: f"{r.est:+.2f} [{r.lo95:+.2f}, {r.hi95:+.2f}]"
    L = ["# B6. Get-passive transfer of the existing directions", "",
         "Spec: `round3_plan.md`, B6. Run: `run_passive_test.py` on the got prompts (same plan rows as the was test); "
         "analysis: `analyze_get_passive.py`. D = mean change after a transitive minus an intransitive active donor, "
         "on bad passive bases, nats. 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs), the "
         "same resamples for both frames.", ""]
    for pop in res.population.unique():
        L += [f"## {pop}", "",
              "| Site | δ | Natural *by* gap: was / got | D(O): was / got | D(by): was / got | D(\".\"): got | "
              "got − was: O [90% CI] | got − was: by | Outcome was / got |",
              "|---:|---:|---|---|---|---|---|---|---|"]
        for site in sorted(res.site.unique()):
            o_w, o_g = get(pop, site, "was", "O"), get(pop, site, "got", "O")
            b_w, b_g = get(pop, site, "was", "by"), get(pop, site, "got", "by")
            dg = get(pop, site, "got", "dot")
            do, db = get(pop, site, "got_minus_was", "O"), get(pop, site, "got_minus_was", "by")
            nw, ng = get(pop, site, "natgap_was", "by"), get(pop, site, "natgap_got", "by")
            L.append(f"| {site} | {o_w.delta:.2f} | {nw.est:.2f} / {ng.est:.2f} [{ng.lo95:.2f}, {ng.hi95:.2f}] | "
                     f"{o_w.est:+.2f} / {f3(o_g)} | {b_w.est:+.2f} / {f3(b_g)} | {dg.est:+.2f} | "
                     f"{do.est:+.2f} [{do.lo90:+.2f}, {do.hi90:+.2f}] | {f3(db)} | "
                     f"{outcome(o_w['class'], b_w['class'])} / {outcome(o_g['class'], b_g['class'])} |")
        # declared decision at site 8
        o8, d8 = get(pop, 8, "got_minus_was", "O"), get(pop, 8, "got", "by")
        bw8, ng8 = get(pop, 8, "was", "by"), get(pop, 8, "natgap_got", "by")
        delta = o8.delta
        equiv = -delta < o8.lo90 and o8.hi90 < delta
        by_ok = d8.lo95 > 0 and d8.est >= 0.5 * bw8.est
        by_lost = d8.lo95 <= 0 and ng8.est >= 0.2
        if equiv and by_ok:
            verdict = "generalizes to got"
        elif (o8.lo95 > 0 and o8.est >= delta) or by_lost:
            verdict = "does not generalize"
        else:
            verdict = "unresolved"
        L += ["", f"**Declared decision (site 8):** D(O) got − was {o8.est:+.2f}, 90% CI [{o8.lo90:+.2f}, {o8.hi90:+.2f}] "
              f"(δ = {delta:.2f}; within ±δ: {'yes' if equiv else 'no'}); D(by) got {f3(d8)} vs was {bw8.est:+.2f} "
              f"(preserved at ≥ half with CI > 0: {'yes' if by_ok else 'no'}); natural got *by* gap {ng8.est:.2f}. "
              f"**Outcome: {verdict}.**", ""]
    L += ["## Natural projection (no patching), primary pairs", "",
          "| Site | was: good − bad z | got: good − bad z | active gap | was / active | got / active |",
          "|---:|---:|---:|---:|---:|---:|"]
    for r in pj.itertuples():
        L.append(f"| {r.site} | {r.was_gap:.2f} | {r.got_gap:.2f} | {r.active_gap:.2f} | {r.was_ratio:.2f} | "
                 f"{r.got_ratio:.2f} |")
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results/das_round2")
    ap.add_argument("--items-was", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--items-got", default="data/das_round2/passive_test_got/items.csv")
    ap.add_argument("--prompts-was", default="data/das_round2/passive_test/prompts.csv")
    ap.add_argument("--prompts-got", default="data/das_round2/passive_test_got/prompts.csv")
    ap.add_argument("--plan", default="data/das_round2/passive_test/plan.csv.gz")
    ap.add_argument("--was-dirs", default="results/das_round2/passive_test,results/das_round2/passive_test_fill")
    ap.add_argument("--got-dir", default="results/das_round2/passive_test_got")
    ap.add_argument("--sites", default="4,6,8,10,12,14,16,17")
    ap.add_argument("--out-dir", default="results/das_round2/passive_test_got")
    ap.add_argument("--report", default="reports/passive_das_prep/round3_get_passive.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
