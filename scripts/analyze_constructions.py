#!/usr/bin/env python3
"""Object relatives and tough constructions (round3_plan.md, C9): analysis.

Per construction (OR, TC) and site, on bad items (primary) and good items:
- natural gate: good - bad licensing readout L_c (OR: log P(MAIN) - log P(PREP);
  TC: log P(END) - log P(PREP)) on the passive test's bootstrap (pairs within
  band, contexts);
- D = mean change after T - after I active donors (three-way bootstrap: pairs
  within band, contexts, donor pairs; 2,000 draws, seed 17) for O, L_c, the
  numerator, PREP, " by", and the other readouts; the within-construction swap;
- classification: O with delta_s; L_c with delta_L = 0.2 x the natural good -
  bad L_c gap; the licensing reading also needs the numerator's D CI above 0;
- TC base-form gate: "NAME can V" good vs bad verbs separate along d_s (AUC over
  verb means >= 0.8);
- natural projection: good - bad z gap in the construction and its ratio to the
  same verbs' active gap;
- tokens driving the natural and the patched effects (top 5 by probability change).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_passive_test import Boot, auc, cells, ci, classify, pair_values, site_delta, summarize
from analyze_nonce_passive import z_table

BASE = ("M", "O", "I", "det", "pron", "refl", "by", "dot", "the", "him", "PREP", "MAIN", "END", "END_BROAD")
NUM = {"OR": "MAIN", "TC": "END"}
SITES = (4, 6, 8, 10, 12, 14, 16, 17)


def add_L(df, cons):
    df = df.copy()
    df["L"] = df[NUM[cons]] - df["PREP"]
    df["L_broad"] = df["END_BROAD"] - df["PREP"] if cons == "TC" else df["L"]
    return df


def run(args):
    ddir, rdir = Path(args.data_dir), Path(args.out_dir)
    items = pd.read_csv(ddir / "items.csv")
    P = pd.read_csv(ddir / "prompts.csv")
    plan = pd.read_csv(ddir / "plan.csv.gz", dtype={c: "category" for c in ("item_id", "side", "cond", "donor_pair")})
    nat = pd.read_parquet(rdir / "natural.parquet").set_index("pid")
    das = pd.read_csv(Path(args.root) / "final_strict" / "items.csv")
    pf = pd.read_csv(Path(args.root) / "final_strict" / "pairs_folds.csv")
    pz = np.load(rdir / "projections.npz")
    keys = [(int(c[1]), int(c.split("_f")[1])) for c in pz["keys"]]
    Pz = P.assign(kind=np.where(P.kind.isin(["OR", "TC", "active", "base_ref"]), "real", P.kind))
    Z, _ = z_table(Pz, pz["proj"], list(pz["sites"]), keys, das, pf)
    pidx = dict(zip(P.prompt, P.pid))
    rows, proj_rows, gate_rows, tok_rows = [], [], [], []
    for cons in ("OR", "TC"):
        it = items[items.construction == cons]
        pr = it.drop_duplicates("pair_id").set_index("pair_id")
        pair_ix = {p: i for i, p in enumerate(pr.index)}
        ctx_ix = {c: i for i, c in enumerate(sorted(it.context_id.unique()))}
        sel = plan.item_id.isin(set(it.item_id)).to_numpy()
        pl = plan[sel]
        df0 = pl[["item_id", "side", "cond", "donor_pair"]].join(it.set_index("item_id")[["pair_id", "context_id"]],
                                                                  on="item_id")
        natb = add_L(nat.loc[pl.base.to_numpy(), list(BASE)].reset_index(drop=True), cons)
        R = list(BASE) + ["L", "L_broad"]
        # natural gap good - bad
        g = it[["pair_id", "context_id"]].copy()
        ng = add_L(nat.loc[it.good_prompt.map(pidx), list(BASE)].reset_index(drop=True), cons).to_numpy() - \
            add_L(nat.loc[it.bad_prompt.map(pidx), list(BASE)].reset_index(drop=True), cons).to_numpy()
        g[R] = ng
        for site in SITES:
            pat = pd.read_parquet(rdir / f"patches_site{site}.parquet")
            assert (pat.row.to_numpy() == plan.row.to_numpy()).all()
            pv = add_L(pat.loc[sel, list(BASE)].reset_index(drop=True), cons)
            df = df0.reset_index(drop=True).copy()
            df[R] = pv[R].to_numpy() - natb[R].to_numpy()
            bt = Boot(len(ctx_ix), args.n_boot, args.seed)
            A = {}
            for side in ("bad", "good"):
                x = df[(df.side == side) & df.cond.isin(["T", "I"])].assign(dq=lambda y: y.donor_pair)
                qs = sorted(x.donor_pair.unique())
                qix = {q_: i for i, q_ in enumerate(qs)}
                vT, mT = cells(x[x.cond == "T"], pair_ix, ctx_ix, qix, len(qs), R)
                vI, mI = cells(x[x.cond == "I"], pair_ix, ctx_ix, qix, len(qs), R)
                assert (mT == mI).all()
                A[f"D_{side}"] = pair_values(vT - vI, mT, bt.wc, bt.donor_weights("q", len(qs)))
                xs = df[(df.side == side) & (df.cond == "cswap")].assign(dq=0)
                v, m = cells(xs, pair_ix, ctx_ix, {0: 0}, 1, R)
                A[f"swap_{side}"] = pair_values(v, m, bt.wc, np.ones((bt.B + 1, 1), np.float32))
            v, m = cells(g.assign(dq=0), pair_ix, ctx_ix, {0: 0}, 1, R)
            A["nat_gap"] = pair_values(v, m, bt.wc, np.ones((bt.B + 1, 1), np.float32))
            delta = site_delta(Path(args.root), site)
            pix = np.arange(len(pr))
            wp = bt.pair_weights(pr.band.to_numpy())
            S = {k: summarize(a, wp, pix) for k, a in A.items()}
            jL = R.index("L")
            dL = 0.2 * float(S["nat_gap"][0, jL])
            for k, s in S.items():
                for j, rd in enumerate(R):
                    row = {"construction": cons, "site": site, "quantity": k, "readout": rd, "delta": delta,
                           "delta_L": dL, **ci(s[:, j])}
                    if k == "D_bad" and rd == "O":
                        row["class"] = classify(row, delta)
                    if k == "D_bad" and rd == "L":
                        row["class"] = classify(row, abs(dL))
                    rows.append(row)
            # natural projection: good - bad z in the construction vs the same verbs in actives
            si = list(pz["sites"]).index(site)
            zc = it.good_prompt.map(pidx).map(lambda i: Z[i, si]) - it.bad_prompt.map(pidx).map(lambda i: Z[i, si])
            act = P[P.kind == "active"].set_index(["lemma", "subject"]).pid
            aw = []
            for p in pr.itertuples():
                for s in ("She", "He", "They", "We", "I", "Maria", "David"):
                    gi, bi = act.get((p.good_lemma, s)), act.get((p.bad_lemma, s))
                    if gi is not None and bi is not None:
                        aw.append((p.Index, Z[gi, si] - Z[bi, si]))
            ag = pd.DataFrame(aw, columns=["pair_id", "d"]).groupby("pair_id").d.mean()
            cg = pd.Series(zc.to_numpy(), index=it.pair_id.to_numpy()).groupby(level=0).mean()
            proj_rows.append({"construction": cons, "site": site, "cons_gap": float(cg.mean()),
                              "active_gap": float(ag.mean()), "ratio": float(cg.mean() / ag.mean()),
                              "auc_items": auc(it.good_prompt.map(pidx).map(lambda i: Z[i, si]).to_numpy(),
                                               it.bad_prompt.map(pidx).map(lambda i: Z[i, si]).to_numpy())})
            if cons == "TC":  # base-form gate
                br = P[P.kind == "base_ref"].assign(z=lambda d: Z[d.pid.to_numpy(), si])
                gl, bl = set(pr.good_lemma), set(pr.bad_lemma)
                vg = br[br.lemma.isin(gl)].groupby("lemma").z.mean()
                vb = br[br.lemma.isin(bl)].groupby("lemma").z.mean()
                gate_rows.append({"site": site, "auc_verbs": auc(vg.to_numpy(), vb.to_numpy()),
                                  "pass": auc(vg.to_numpy(), vb.to_numpy()) >= 0.8})
            print(f"{cons} site {site} done", flush=True)
        # tokens driving the natural gap and the patched D (sites 8 and 17)
        nt = np.load(rdir / "natural_tokens.npz")
        toks = list(nt["tokens"])
        lpi = pd.DataFrame(nt["lp"], index=nt["pid"])
        dg = np.exp(lpi.loc[it.good_prompt.map(pidx)].to_numpy()).mean(0) - np.exp(lpi.loc[it.bad_prompt.map(pidx)].to_numpy()).mean(0)
        for j in np.argsort(-np.abs(dg))[:6]:
            tok_rows.append({"construction": cons, "what": "natural good - bad", "token": toks[j], "dprob": float(dg[j])})
        for site in (8, 17):
            ptk = np.load(rdir / f"patch_tokens_site{site}.npz")
            prow = pd.Series(np.arange(len(ptk["rows"])), index=ptk["rows"])
            sub = plan[sel & (plan.side == "bad").to_numpy() & plan.cond.isin(["T", "I"]).to_numpy()]
            lpT = np.exp(ptk["lp"][prow.loc[sub[sub.cond == "T"].row.to_numpy()].to_numpy()]).mean(0)
            lpI = np.exp(ptk["lp"][prow.loc[sub[sub.cond == "I"].row.to_numpy()].to_numpy()]).mean(0)
            dd = lpT - lpI
            for j in np.argsort(-np.abs(dd))[:6]:
                tok_rows.append({"construction": cons, "what": f"site {site} T - I", "token": toks[j], "dprob": float(dd[j])})
    res, pj, gt, tk = pd.DataFrame(rows), pd.DataFrame(proj_rows), pd.DataFrame(gate_rows), pd.DataFrame(tok_rows)
    res.to_csv(rdir / "constructions_summary.csv", index=False)
    pj.to_csv(rdir / "constructions_projection.csv", index=False)
    gt.to_csv(rdir / "tc_baseform_gate.csv", index=False)
    tk.to_csv(rdir / "constructions_tokens.csv", index=False)
    write_report(args, res, pj, gt, tk)


def reading(o, l, num_ok, gate_ok):
    if not gate_ok:
        return "unresolved (gate)"
    if o == "NO RISE" and l == "RISE" and num_ok:
        return "consistent with direct-gap licensing"
    if o == "RISE" and l in ("NO RISE", "FALL"):
        return "surface"
    if o == "RISE" and l == "RISE":
        return "mixed"
    if o == "NO RISE" and l == "NO RISE":
        return "nothing moves"
    return "unresolved"


def write_report(args, res, pj, gt, tk):
    g = lambda c, s, qn, rd: res[(res.construction == c) & (res.site == s) & (res.quantity == qn) & (res.readout == rd)].iloc[0]
    f3 = lambda r: f"{r.est:+.2f} [{r.lo95:+.2f}, {r.hi95:+.2f}]"
    L = ["# C9. Object relatives and tough constructions", "",
         "Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary "
         "pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the "
         "bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − "
         "log P(PREP); L_TC = log P(END) − log P(PREP).", ""]
    gate_tc = dict(zip(gt.site, gt["pass"]))
    for cons, name in (("OR", "Object relatives (\"The house that John destroyed\")"),
                       ("TC", "Tough constructions (\"The house is easy to destroy\")")):
        ng = g(cons, 8, "nat_gap", "L")
        natok = ng.lo95 > 0
        L += [f"## {name}", "",
              f"Natural gate: good − bad L = {f3(ng)} → **{'passes' if natok else 'fails'}**; numerator "
              f"{f3(g(cons, 8, 'nat_gap', NUM[cons]))}, PREP {f3(g(cons, 8, 'nat_gap', 'PREP'))}, O "
              f"{f3(g(cons, 8, 'nat_gap', 'O'))}.", "",
              "| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(\" by\") | swap good→bad: L | z gap / active | Reading |",
              "|---:|---|---|---|---|---|---|---|---|---|"]
        for s in SITES:
            o, l = g(cons, s, "D_bad", "O"), g(cons, s, "D_bad", "L")
            nu, pp = g(cons, s, "D_bad", NUM[cons]), g(cons, s, "D_bad", "PREP")
            by, sw = g(cons, s, "D_bad", "by"), g(cons, s, "swap_bad", "L")
            p = pj[(pj.construction == cons) & (pj.site == s)].iloc[0]
            gate_ok = natok and (cons == "OR" or gate_tc.get(s, False))
            rd = reading(o["class"], l["class"] if natok else "", nu.lo95 > 0, gate_ok)
            L.append(f"| {s} | {o.delta:.2f} / {l.delta_L:.2f} | {f3(o)} {o['class']} | {f3(l)} {l['class']} | {f3(nu)} | "
                     f"{f3(pp)} | {f3(by)} | {f3(sw)} | {p.cons_gap:.2f} / {p.active_gap:.2f} | **{rd}** |")
        if cons == "TC":
            L += ["", "TC base-form gate (\"John can read\" vs \"John can sleep\", AUC over verb means along d_s): "
                  + ", ".join(f"site {r.site} {r.auc_verbs:.2f}{'' if r['pass'] else ' (fails)'}" for _, r in gt.iterrows())]
        t = tk[tk.construction == cons]
        L += ["", "Tokens driving the effects (probability change):", ""]
        for w, gg in t.groupby("what", sort=False):
            L.append(f"- {w}: " + ", ".join(f"{r.token!r} {r.dprob:+.4f}" for r in gg.itertuples()))
        L.append("")
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results/das_round2")
    ap.add_argument("--data-dir", default="data/constructions")
    ap.add_argument("--out-dir", default="results/das_round2/constructions")
    ap.add_argument("--report", default="reports/passive_das_prep/round3_constructions.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
