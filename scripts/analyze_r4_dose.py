#!/usr/bin/env python3
"""Round 4, A1 (dose check for C9) and A3 (OR verb forms). No model; existing outputs only.

A1, per construction c (passive, OR, TC) and site, 64 primary pairs:
- natural z levels of good and bad items (cross-fitted z as in round 3), gap g_c, and ratio
  rho_c = g_c / active gap of the same verbs (the passive test's active frame, one denominator;
  ratio of population means, the round-3 aggregation);
- donor levels z_T, z_I over the bad-base T / I plan rows; upper overshoot u_c = z_T - z_good,
  lower l_c = z_bad - z_I, span excess (z_T - z_I) - g_c;
- excess object log-probability E_c = D(O) - natural good - bad log P(O).
Primary inference: pair bootstrap (pairs within band, the same draws for every construction, so
differences are paired), contexts and donors averaged. Sensitivity at site 8: contexts and donor
pairs also resampled (independently per construction). Decision rules: plan.md, A1.

A3: the C9 object-relative result at site 8 without the pairs whose verbs have simple past !=
participle, with the C9 three-way bootstrap and readings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_constructions import add_L, reading
from analyze_nonce_passive import z_table
from analyze_passive_test import Boot, cells, ci, classify, frame_z, pair_values, site_delta, summarize

SITES = (4, 6, 8, 10, 12, 14, 16, 17)
CONS = ("passive", "OR", "TC")
COLS = ["zg", "zb", "gap", "active_gap", "zT", "zI", "u", "l", "span_excess", "D_O", "natO", "E"]


def passive_dir(root, site):
    return Path(root) / ("passive_test" if site in (8, 12, 17) else "passive_test_fill")


def pidx(P):
    return dict(zip(P.prompt, P.pid))


def zfull(proj, P, das, pf):
    """Cross-fitted z for every prompt (DAS verbs: held-out bases only)."""
    proj = proj.set_index("pid").loc[P.pid]
    cols = sorted(proj.columns)
    keys = [(int(c[1]), int(c.split("_f")[1])) for c in cols]
    Pz = P.assign(kind=np.where(P.lemma.isin(set(das.verb)), "real", "other"))
    Z, _ = z_table(Pz, proj[cols].to_numpy()[:, None, :], [0], keys, das, pf)
    return Z[:, 0]


def frames_for(rows_plan, O_patched, nat, zall, items, P):
    """Item-level and donor-level frames.

    items -> [pair_id, context_id, zg, zb, natO]; donor rows -> [pair_id, context_id, dq, D_O, zT, zI]
    (T - I change in log P(O) per item x donor pair, averaged over splits)."""
    ix = pidx(P)
    it = items[["item_id", "pair_id", "context_id"]].copy()
    it["zg"], it["zb"] = zall[items.good_prompt.map(ix)], zall[items.bad_prompt.map(ix)]
    it["natO"] = nat.loc[items.good_prompt.map(ix), "O"].to_numpy() - nat.loc[items.bad_prompt.map(ix), "O"].to_numpy()
    d = rows_plan[["item_id", "cond", "base", "donor", "donor_pair"]].copy()
    d["dO"] = O_patched - nat.loc[d.base.to_numpy(), "O"].to_numpy()
    d["zd"] = zall[d.donor.to_numpy()]
    m = d.groupby(["item_id", "donor_pair", "cond"], observed=True)[["dO", "zd"]].mean().unstack("cond")
    dr = pd.DataFrame({"D_O": m[("dO", "T")] - m[("dO", "I")], "zT": m[("zd", "T")], "zI": m[("zd", "I")]})
    dr = dr.reset_index().rename(columns={"donor_pair": "dq"})
    dr = dr.join(items.set_index("item_id")[["pair_id", "context_id"]], on="item_id")
    return it, dr


def passive_frames(root, site, items, P, plan, das, pf, pr):
    nat = pd.read_parquet(passive_dir(root, site) / "natural.parquet").set_index("pid")
    proj = pd.read_parquet(passive_dir(root, site) / f"projections_site{site}.parquet")
    _, at = frame_z(proj, items, P, pf, das, pr)
    active_gap = at.groupby("pair_id")["diff"].mean()
    pat = pd.read_parquet(passive_dir(root, site) / f"patches_site{site}.parquet")
    assert (pat.row.to_numpy() == plan.row.to_numpy()).all()
    sel = (plan.item_id.isin(set(items.item_id)) & (plan.side == "bad") & plan.cond.isin(["T", "I"])).to_numpy()
    it, dr = frames_for(plan[sel], pat.O.to_numpy()[sel], nat, zfull(proj, P, das, pf), items, P)
    return it, dr, active_gap


def cons_frames(root, cons, items, P, plan, nat, das, pf, pz, site):
    it = items[items.construction == cons]
    keys = [(int(c[1]), int(c.split("_f")[1])) for c in pz["keys"]]
    Pz = P.assign(kind=np.where(P.kind.isin(["OR", "TC", "active", "base_ref"]), "real", P.kind))
    Z, _ = z_table(Pz, pz["proj"], list(pz["sites"]), keys, das, pf)
    pat = pd.read_parquet(Path(root) / "constructions" / f"patches_site{site}.parquet")
    assert (pat.row.to_numpy() == plan.row.to_numpy()).all()
    sel = (plan.item_id.isin(set(it.item_id)) & (plan.side == "bad") & plan.cond.isin(["T", "I"])).to_numpy()
    return frames_for(plan[sel], pat.O.to_numpy()[sel], nat, Z[:, list(pz["sites"]).index(site)], it, P)


def derive(df):
    df = df.copy()
    df["gap"] = df.zg - df.zb
    df["u"], df["l"] = df.zT - df.zg, df.zb - df.zI
    df["span_excess"] = (df.zT - df.zI) - df.gap
    df["E"] = df.D_O - df.natO
    return df


def population(stats, cols=COLS):
    """stats [B+1, R] over COLS -> dict of per-draw arrays, with rho = ratio of means."""
    s = {c: stats[:, i] for i, c in enumerate(cols)}
    s["rho"] = s["gap"] / s["active_gap"]
    return s


def run_a1(args):
    root = Path(args.root)
    data = Path(args.data_root)
    pt = data / "das_round2" / "passive_test"
    items = pd.read_csv(pt / "items.csv")
    items = items[items.bad_class == "plain"]
    P = pd.read_csv(pt / "prompts.csv")
    plan = pd.read_csv(pt / "plan.csv.gz", usecols=["row", "item_id", "side", "cond", "base", "donor", "donor_pair"])
    das = pd.read_csv(root / "final_strict" / "items.csv")
    pf = pd.read_csv(root / "final_strict" / "pairs_folds.csv")
    pr = items.drop_duplicates("pair_id").set_index("pair_id")
    cd = data / "constructions"
    citems, cP = pd.read_csv(cd / "items.csv"), pd.read_csv(cd / "prompts.csv")
    cplan = pd.read_csv(cd / "plan.csv.gz", usecols=["row", "item_id", "side", "cond", "base", "donor", "donor_pair"])
    cnat = pd.read_parquet(root / "constructions" / "natural.parquet").set_index("pid")
    pz = np.load(root / "constructions" / "projections.npz")
    pairs = list(pr.index)
    bt = Boot(1, args.n_boot, args.seed)
    wp = bt.pair_weights(pr.band.to_numpy())  # [B+1, P]; the same draws for every construction and site
    rows, per_rows, sens_rows = [], [], []
    for site in SITES:
        fr = {}
        it, dr, active_gap = passive_frames(root, site, items, P, plan, das, pf, pr)
        fr["passive"] = (it, dr)
        for cons in ("OR", "TC"):
            fr[cons] = cons_frames(root, cons, citems, cP, cplan, cnat, das, pf, pz, site)
        stats = {}
        for c, (it, dr) in fr.items():
            per = it.groupby("pair_id")[["zg", "zb", "natO"]].mean().join(
                dr.groupby("pair_id")[["D_O", "zT", "zI"]].mean())
            per["active_gap"] = active_gap
            per = derive(per).loc[pairs]
            per_rows.append(per.assign(construction=c, site=site).reset_index())
            stats[c] = population((wp @ per[COLS].to_numpy()) / wp.sum(1, keepdims=True))
        for c in CONS:
            for q, v in stats[c].items():
                rows.append({"site": site, "construction": c, "quantity": q, **ci(v)})
        for a, b in (("passive", "TC"), ("TC", "OR"), ("passive", "OR")):
            for q in ("rho", "u", "E", "gap"):
                rows.append({"site": site, "construction": f"{a} - {b}", "quantity": q,
                             **ci(stats[a][q] - stats[b][q])})
        if site == 8:  # sensitivity: contexts and donor pairs resampled too, independently per construction
            pix = np.arange(len(pairs))
            pair_ix = {p: i for i, p in enumerate(pairs)}
            st = {}
            for j, (c, (it, dr)) in enumerate(fr.items()):
                ctx_ix = {x: i for i, x in enumerate(sorted(it.context_id.unique()))}
                b3 = Boot(len(ctx_ix), args.n_boot, args.seed + 101 + j)
                v, m = cells(it.assign(dq=0), pair_ix, ctx_ix, {0: 0}, 1, ["zg", "zb", "natO"])
                A1_ = pair_values(v, m, b3.wc, np.ones((b3.B + 1, 1), np.float32))
                qs = sorted(dr.dq.unique())
                v, m = cells(dr, pair_ix, ctx_ix, {q: i for i, q in enumerate(qs)}, len(qs), ["D_O", "zT", "zI"])
                A2_ = pair_values(v, m, b3.wc, b3.donor_weights("q", len(qs)))
                ag = np.broadcast_to(active_gap.loc[pairs].to_numpy()[None, :, None], (b3.B + 1, len(pairs), 1))
                A = np.concatenate([A1_, A2_, ag], -1)  # zg zb natO D_O zT zI active_gap
                S = summarize(A, wp, pix)
                s = dict(zip(["zg", "zb", "natO", "D_O", "zT", "zI", "active_gap"], S.T))
                s["gap"] = s["zg"] - s["zb"]
                s["rho"], s["u"], s["E"] = s["gap"] / s["active_gap"], s["zT"] - s["zg"], s["D_O"] - s["natO"]
                st[c] = s
            for c in CONS:
                for q in ("rho", "u", "E"):
                    sens_rows.append({"site": 8, "construction": c, "quantity": q, **ci(st[c][q])})
            for a, b in (("passive", "TC"), ("TC", "OR"), ("passive", "OR")):
                for q in ("rho", "u", "E"):
                    sens_rows.append({"site": 8, "construction": f"{a} - {b}", "quantity": q, **ci(st[a][q] - st[b][q])})
        print(f"A1 site {site} done", flush=True)
    per = pd.concat(per_rows, ignore_index=True)
    slopes = []  # descriptive per-pair slope of E on u within construction (confounded; see plan)
    for (c, s), g in per.groupby(["construction", "site"]):
        x, y = g.u.to_numpy(), g.E.to_numpy()
        b = []
        for w in wp:
            xm, ym = np.average(x, weights=w), np.average(y, weights=w)
            b.append(np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2))
        slopes.append({"construction": c, "site": s, **ci(np.array(b))})
    return pd.DataFrame(rows), per, pd.DataFrame(slopes), pd.DataFrame(sens_rows)


def run_a3(args):
    """OR at site 8 without past != participle pairs, with the C9 machinery."""
    root, data = Path(args.root), Path(args.data_root)
    cd = data / "constructions"
    items = pd.read_csv(cd / "items.csv")
    P = pd.read_csv(cd / "prompts.csv")
    plan = pd.read_csv(cd / "plan.csv.gz", dtype={c: "category" for c in ("item_id", "side", "cond", "donor_pair")})
    nat = pd.read_parquet(root / "constructions" / "natural.parquet").set_index("pid")
    vf = pd.read_csv(data / "das_round2" / "verb_forms.csv").set_index("lemma")
    it0 = items[items.construction == "OR"]
    differ = sorted({p.pair_id for p in it0.drop_duplicates("pair_id").itertuples()
                     for lem in (p.good_lemma, p.bad_lemma) if vf.past[lem] != vf.participle[lem]})
    pat = pd.read_parquet(root / "constructions" / "patches_site8.parquet")
    assert (pat.row.to_numpy() == plan.row.to_numpy()).all()
    R = ["O", "by", "PREP", "MAIN", "L"]
    base = ["O", "by", "PREP", "MAIN", "END", "END_BROAD"]
    out = []
    for label, drop in (("all 64 pairs", []), ("without past ≠ participle", differ)):
        it = it0[~it0.pair_id.isin(drop)]
        pr = it.drop_duplicates("pair_id").set_index("pair_id")
        pair_ix = {p: i for i, p in enumerate(pr.index)}
        ctx_ix = {c: i for i, c in enumerate(sorted(it.context_id.unique()))}
        sel = plan.item_id.isin(set(it.item_id)).to_numpy()
        pl = plan[sel]
        df = pl[["item_id", "side", "cond", "donor_pair"]].join(it.set_index("item_id")[["pair_id", "context_id"]],
                                                                 on="item_id").reset_index(drop=True)
        natb = add_L(nat.loc[pl.base.to_numpy(), base].reset_index(drop=True), "OR")
        pv = add_L(pat.loc[sel, base].reset_index(drop=True), "OR")
        df[R] = pv[R].to_numpy() - natb[R].to_numpy()
        bt = Boot(len(ctx_ix), args.n_boot, args.seed)
        x = df[(df.side == "bad") & df.cond.isin(["T", "I"])].assign(dq=lambda y: y.donor_pair)
        qs = sorted(x.donor_pair.unique())
        qix = {q: i for i, q in enumerate(qs)}
        vT, mT = cells(x[x.cond == "T"], pair_ix, ctx_ix, qix, len(qs), R)
        vI, mI = cells(x[x.cond == "I"], pair_ix, ctx_ix, qix, len(qs), R)
        assert (mT == mI).all()
        A = {"D_bad": pair_values(vT - vI, mT, bt.wc, bt.donor_weights("q", len(qs)))}
        ix = pidx(P)
        g = it[["pair_id", "context_id"]].copy()
        g[R] = add_L(nat.loc[it.good_prompt.map(ix), base].reset_index(drop=True), "OR")[R].to_numpy() - \
            add_L(nat.loc[it.bad_prompt.map(ix), base].reset_index(drop=True), "OR")[R].to_numpy()
        v, m = cells(g.assign(dq=0), pair_ix, ctx_ix, {0: 0}, 1, R)
        A["nat_gap"] = pair_values(v, m, bt.wc, np.ones((bt.B + 1, 1), np.float32))
        wpair = bt.pair_weights(pr.band.to_numpy())
        S = {k: summarize(a, wpair, np.arange(len(pr))) for k, a in A.items()}
        delta = site_delta(root, 8)
        dL = 0.2 * float(S["nat_gap"][0, R.index("L")])
        cls = {"O": classify(ci(S["D_bad"][:, 0]), delta), "L": classify(ci(S["D_bad"][:, R.index("L")]), abs(dL))}
        rd = reading(cls["O"], cls["L"], ci(S["D_bad"][:, R.index("MAIN")])["lo95"] > 0, True)
        for k, s in S.items():
            for j, r in enumerate(R):
                out.append({"population": label, "n_pairs": len(pr), "quantity": k, "readout": r, **ci(s[:, j]),
                            "class": cls.get(r, "") if k == "D_bad" else "", "reading": rd})
    return pd.DataFrame(out), differ


def write_report(args, res, slopes, sens, a3, differ):
    g = lambda s, c, q, t=res: t[(t.site == s) & (t.construction == c) & (t.quantity == q)].iloc[0]
    f = lambda r: f"{r.est:+.2f} [{r.lo95:+.2f}, {r.hi95:+.2f}]"
    f0 = lambda r: f"{r.est:.2f} [{r.lo95:.2f}, {r.hi95:.2f}]"
    L = ["# Round 4, A1 and A3", "",
         "Spec: `plan.md`, A1 and A3. Script: `analyze_r4_dose.py` (existing round-3 outputs only). 64 primary "
         "pairs. CIs: pair bootstrap within band, the same draws for every construction (paired differences); "
         "contexts and donors averaged (inference conditional on them). z: 0 = held-out active intransitive, "
         "1 = transitive level. E = D(O) − natural O gap, log-probability units.", "",
         "## A1. Natural range, donor overshoot and excess object log-probability", "",
         "| Site | Construction | z good | z bad | ratio to active gap | donor z T / I | upper overshoot u | D(O) | natural O gap | E |",
         "|---:|---|---:|---:|---|---|---|---|---|---|"]
    for s in SITES:
        for c in CONS:
            L.append(f"| {s} | {c} | {g(s, c, 'zg').est:.2f} | {g(s, c, 'zb').est:.2f} | {f0(g(s, c, 'rho'))} | "
                     f"{g(s, c, 'zT').est:.2f} / {g(s, c, 'zI').est:.2f} | {f(g(s, c, 'u'))} | {f(g(s, c, 'D_O'))} | "
                     f"{f(g(s, c, 'natO'))} | {f(g(s, c, 'E'))} |")
    L += ["", "Paired differences between constructions:", "", "| Site | Contrast | ratio | u | E |", "|---:|---|---|---|---|"]
    for s in SITES:
        for c in ("passive - TC", "TC - OR", "passive - OR"):
            L.append(f"| {s} | {c} | {f(g(s, c, 'rho'))} | {f(g(s, c, 'u'))} | {f(g(s, c, 'E'))} |")
    below = lambda c, q, t=res: g(8, c, q, t).hi95 < 0
    above = lambda c, q, t=res: g(8, c, q, t).lo95 > 0
    dose = below("passive - TC", "rho") and below("TC - OR", "rho")
    u_ord = above("passive - TC", "u") and above("TC - OR", "u")
    e_ord = above("passive - TC", "E") and above("TC - OR", "E")
    verdict = ("the excess tracks the overshoot" if u_ord and e_ord else
               "overshoot differs but does not explain the excess" if u_ord else "unresolved")
    L += ["", f"**Declared decisions (site 8):** dose differs by construction (ρ ordered): **{'yes' if dose else 'no'}**; "
          f"u ordered: **{'yes' if u_ord else 'no'}**; E ordered: **{'yes' if e_ord else 'no'}** → **{verdict}**.", "",
          "Sensitivity at site 8 (contexts and donor pairs also resampled):", "",
          "| Construction / contrast | ratio | u | E |", "|---|---|---|---|"]
    for c in CONS + ("passive - TC", "TC - OR", "passive - OR"):
        L.append(f"| {c} | {f(g(8, c, 'rho', sens))} | {f(g(8, c, 'u', sens))} | {f(g(8, c, 'E', sens))} |")
    L += ["", "Per-pair slope of E on u within construction (descriptive; confounded through the good verb):", "",
          "| Site | passive | OR | TC |", "|---:|---|---|---|"]
    for s in SITES:
        L.append(f"| {s} | " + " | ".join(f(slopes[(slopes.site == s) & (slopes.construction == c)].iloc[0])
                                         for c in CONS) + " |")
    L += ["", "## A3. Object relatives without past ≠ participle pairs (site 8)", "",
          f"Pairs dropped: {', '.join(differ) if differ else 'none'}. C9 three-way bootstrap.", "",
          "| Population | pairs | D(O) | D(L) | D(MAIN) | D(PREP) | natural L | natural O | reading |",
          "|---|---:|---|---|---|---|---|---|---|"]
    for pop, gg in a3.groupby("population", sort=False):
        q = lambda k, r: gg[(gg.quantity == k) & (gg.readout == r)].iloc[0]
        L.append(f"| {pop} | {int(gg.n_pairs.iloc[0])} | {f(q('D_bad', 'O'))} {q('D_bad', 'O')['class']} | "
                 f"{f(q('D_bad', 'L'))} {q('D_bad', 'L')['class']} | {f(q('D_bad', 'MAIN'))} | {f(q('D_bad', 'PREP'))} | "
                 f"{f(q('nat_gap', 'L'))} | {f(q('nat_gap', 'O'))} | {gg.reading.iloc[0]} |")
    a = a3[a3.population == "all 64 pairs"].set_index(["quantity", "readout"]).est
    b = a3[a3.population != "all 64 pairs"].set_index(["quantity", "readout"]).est
    mx = float((a - b).abs().max())
    same = a3.groupby("population").reading.first().nunique() == 1
    L += ["", f"Largest change in any estimate: {mx:.3f}. **Declared decision:** "
          f"**{'unaffected' if mx < 0.05 and same else 'affected'}** (every estimate moves by < 0.05 and the "
          "reading is unchanged).", ""]
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def run(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res, per, slopes, sens = run_a1(args)
    a3, differ = run_a3(args)
    res.to_csv(out / "a1_summary.csv", index=False)
    per.to_csv(out / "a1_per_pair.csv", index=False)
    slopes.to_csv(out / "a1_slopes.csv", index=False)
    sens.to_csv(out / "a1_sensitivity_site8.csv", index=False)
    a3.to_csv(out / "a3_or_forms.csv", index=False)
    (out / "a3_meta.json").write_text(json.dumps({"dropped_pairs": differ}, indent=2) + "\n")
    write_report(args, res, slopes, sens, a3, differ)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results/das_round2")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="results/round4/dose")
    ap.add_argument("--report", default="reports/round4/a1_a3_dose.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
