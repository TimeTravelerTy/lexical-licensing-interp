#!/usr/bin/env python3
"""Round 4, Part B: filler-gap analysis (reports/round4/plan.md, B6 / B7).

Modes
- gates (stage 1, natural pass only), per experiment (emb = B6, mat = B7):
  * I-verb gate on half-A contexts: log P(PREP | fill) - log P(PREP | nofill), mean over contexts,
    kept if its context-bootstrap 95% CI is above 0 -> `data/.../kept_I_{exp}.csv`;
  * natural gate on half-B contexts (kept I, all T): filler x class interaction on L (fill - nofill,
    T - I) and on O (nofill - fill, T - I); T and I verbs resampled independently, contexts jointly;
    passes if both 95% CIs are above 0. Also on all contexts and with all I verbs; per matrix verb;
  * the prepositions the filler raises most after kept I verbs;
  * B7 base-form gate ("NAME can V", T vs I along d_s, AUC over verb means >= 0.8 per site);
  * class x frame means of the readouts and of z. -> `gates.json`, report.
- patch (stage 2): D (T - I donors) and D_in (t_T - t_I) on kept I bases per frame and site;
  bootstrap over base verbs, contexts, donor pairs; O level against the natural T level with T verbs
  resampled; patch x frame interactions; bounds with a 0.2-nat minimum natural gap; the reading;
  sensitivities (non-DAS verbs, multi-token verbs, each matrix verb). Primary: half-B contexts.
L = log P(".") - log P(PREP) (B6) or log P("?") - log P(PREP) (B7).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_nonce_passive import z_table
from analyze_passive_test import Boot, auc, cells, ci, classify, pair_values, site_delta, summarize

SITES = (4, 6, 8, 10, 12, 14, 16, 17)
DECL = (6, 8)
EXPS = {"emb": "B6 embedded wh (\"I know that / what NAME V-ed\")",
        "mat": "B7 matrix wh (\"AUX NAME V\" / \"What aux NAME V\")"}
RD = ("O", "PREP", "num", "L", "END")
MINGAP = 0.2


def readouts(items, nat):
    x = items.join(nat, on="pid")
    x["num"] = np.where(x.exp == "emb", x["dot"], x["q"])
    x["L"] = x.num - x.PREP
    x["matrix"] = x.context_id.str.split("/").str[1]
    return x


def multi(n, B, rng):
    return np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1 / n), B)]).astype(np.float64)


# ---------------------------------------------------------------- stage 1
def interaction(dif, ctxs, Tset, Iset, B, rng):
    """Per draw, class means of the frame difference (fill - nofill): T and I verbs resampled independently,
    contexts jointly. Returns {cls: [B+1, R]}."""
    cix = {c: i for i, c in enumerate(ctxs)}
    wc = multi(len(ctxs), B, rng)
    per = {}
    for cls, lemmas in (("T", sorted(Tset)), ("I", sorted(Iset))):
        li = {l: i for i, l in enumerate(lemmas)}
        sub = dif[(dif.cls == cls) & dif.lemma.isin(lemmas) & dif.context_id.isin(cix)]
        M = np.zeros((len(lemmas), len(ctxs), len(RD)))
        M[sub.lemma.map(li).to_numpy(), sub.context_id.map(cix).to_numpy()] = sub[list(RD)].to_numpy()
        vm = np.einsum("bc,vcr->bvr", wc, M) / wc.sum(1)[:, None, None]
        wv = multi(len(lemmas), B, rng)
        per[cls] = np.einsum("bv,bvr->br", wv, vm) / wv.sum(1)[:, None]
    return per


def run_gates(args):
    ddir, rdir = Path(args.data_dir), Path(args.out_dir)
    items, P = pd.read_csv(ddir / "items.csv"), pd.read_csv(ddir / "prompts.csv")
    nat = pd.read_parquet(rdir / "natural.parquet").set_index("pid")
    x = readouts(items, nat)
    ptok = np.load(rdir / "natural_prep_tokens.npz")
    plp = pd.DataFrame(ptok["lp"], index=ptok["pid"], columns=list(ptok["tokens"]))
    rng = np.random.default_rng(args.seed)
    out, rows = {}, []
    for exp in ("emb", "mat"):
        e = x[x.exp == exp]
        w = e.pivot_table(index=["lemma", "cls", "context_id", "half", "matrix"], columns="frame", values=list(RD))
        dif = pd.DataFrame({r: w[(r, "fill")] - w[(r, "nofill")] for r in RD}).reset_index()
        ctxA = sorted(e[e.half == 0].context_id.unique())
        ctxB = sorted(e[e.half == 1].context_id.unique())
        # I-verb gate, half A
        wcA = multi(len(ctxA), args.n_boot, rng)
        gate = []
        for lem, g in dif[(dif.cls == "I") & (dif.half == 0)].groupby("lemma"):
            v = np.zeros(len(ctxA))
            v[g.context_id.map({c: i for i, c in enumerate(ctxA)}).to_numpy()] = g.PREP.to_numpy()
            c = ci((wcA @ v) / wcA.sum(1))
            gate.append({"exp": exp, "lemma": lem, "dPREP_halfA": c["est"], "lo95": c["lo95"], "hi95": c["hi95"],
                         "kept": c["lo95"] > 0})
        gate = pd.DataFrame(gate)
        kept = set(gate[gate.kept].lemma)
        gate[gate.kept][["lemma"]].to_csv(ddir / f"kept_I_{exp}.csv", index=False)
        Tset, Iall = set(dif[dif.cls == "T"].lemma), set(dif[dif.cls == "I"].lemma)
        res = {}
        for label, cx, Iset in (("half B, kept I", ctxB, kept), ("all contexts, kept I", ctxA + ctxB, kept),
                                ("half B, all I", ctxB, Iall)):
            per = interaction(dif, cx, Tset, Iset, args.n_boot, rng)
            iL = per["T"][:, RD.index("L")] - per["I"][:, RD.index("L")]
            iO = -(per["T"][:, RD.index("O")] - per["I"][:, RD.index("O")])
            res[label] = {"interaction_L": ci(iL), "interaction_O": ci(iO), "n_I": len(Iset), "n_ctx": len(cx)}
            for cls in ("T", "I"):
                for j, r in enumerate(RD):
                    rows.append({"exp": exp, "set": label, "cls": cls, "quantity": f"fill - nofill {r}",
                                 **ci(per[cls][:, j])})
        strata = {}
        for mv in sorted(e.matrix.unique()):
            cx = sorted(e[(e.half == 1) & (e.matrix == mv)].context_id.unique())
            per = interaction(dif, cx, Tset, kept, args.n_boot, rng)
            strata[mv] = {"interaction_L": ci(per["T"][:, RD.index("L")] - per["I"][:, RD.index("L")]),
                          "interaction_O": ci(-(per["T"][:, RD.index("O")] - per["I"][:, RD.index("O")]))}
        passes = res["half B, kept I"]["interaction_L"]["lo95"] > 0 and res["half B, kept I"]["interaction_O"]["lo95"] > 0
        # prepositions the filler raises after kept I verbs (probability change, half B)
        ki = e[(e.cls == "I") & e.lemma.isin(kept) & (e.half == 1)]
        pf_ = np.exp(plp.loc[ki[ki.frame == "fill"].pid]).mean(0) - np.exp(plp.loc[ki[ki.frame == "nofill"].pid]).mean(0)
        top = pf_.sort_values(ascending=False).head(8)
        means = e[e.half == 1].groupby(["cls", "frame"])[list(RD)].mean()
        out[exp] = {"I_gate": gate, "kept": sorted(kept), "natural_gate": res, "strata": strata, "passes": bool(passes),
                    "means": means, "top_prep": {k: float(v) for k, v in top.items()}}
    pz = np.load(rdir / "projections.npz")
    keys = [(int(c[1]), int(c.split("_f")[1])) for c in pz["keys"]]
    das = pd.read_csv(Path(args.root) / "final_strict" / "items.csv")
    pf = pd.read_csv(Path(args.root) / "final_strict" / "pairs_folds.csv")
    Pz = P.assign(kind=np.where(P.kind.isin(["emb", "mat", "base_ref"]), "real", P.kind))
    Z, _ = z_table(Pz, pz["proj"], list(pz["sites"]), keys, das, pf)
    zrows, base_gate = [], []
    cl = dict(zip(items.lemma, items.cls))
    for si, s in enumerate(pz["sites"]):
        xi = x.assign(z=Z[x.pid.to_numpy(), si])
        for (exp, cls, frame), g in xi.groupby(["exp", "cls", "frame"]):
            zrows.append({"site": int(s), "exp": exp, "cls": cls, "frame": frame, "z": float(g.z.mean())})
        br = P[P.kind == "base_ref"].assign(z=lambda d: Z[d.pid.to_numpy(), si])
        vm = br.groupby("lemma").z.mean()
        a = auc(vm[[cl[l] == "T" for l in vm.index]].to_numpy(), vm[[cl[l] == "I" for l in vm.index]].to_numpy())
        base_gate.append({"site": int(s), "auc_verbs": a, "pass": a >= 0.8})
    zt, bg = pd.DataFrame(zrows), pd.DataFrame(base_gate)
    pd.concat([o["I_gate"] for o in out.values()]).to_csv(rdir / "I_gate.csv", index=False)
    pd.DataFrame(rows).to_csv(rdir / "natural_gate.csv", index=False)
    zt.to_csv(rdir / "z_by_class_frame.csv", index=False)
    bg.to_csv(rdir / "baseform_gate.csv", index=False)
    json.dump({exp: {"passes": o["passes"], "kept": o["kept"], "natural_gate": o["natural_gate"],
                     "strata": o["strata"], "top_prep": o["top_prep"]} for exp, o in out.items()},
              open(rdir / "gates.json", "w"), indent=2)
    write_gates(args, out, zt, bg)


def write_gates(args, out, zt, bg):
    f = lambda c: f"{c['est']:+.2f} [{c['lo95']:+.2f}, {c['hi95']:+.2f}]"
    L = ["# Round 4, Part B stage 1: natural gates", "",
         "Spec: `plan.md`, B6/B7. Run: `run_fillergap.py natural`; analysis: `analyze_fillergap.py gates`. "
         "fill = with the wh filler, nofill = without. L = log P(\".\") − log P(PREP) (B6) or log P(\"?\") − log "
         "P(PREP) (B7). Half A (12 contexts) selects I verbs; half B (12 contexts) tests. CIs: T and I verbs "
         "resampled independently, contexts jointly (2,000 draws, seed 17).", ""]
    for exp, o in out.items():
        g, ng = o["I_gate"], o["natural_gate"]
        L += [f"## {EXPS[exp]}", "",
              f"**I-verb gate (half A):** {int(g.kept.sum())} of {len(g)} I verbs kept (filler raises PREP, CI above "
              f"0). Dropped: {', '.join(g[~g.kept].lemma) or 'none'}.", "",
              "| Contexts, I set | n I | interaction on L | interaction on O |", "|---|---:|---|---|"]
        for label, r in ng.items():
            L.append(f"| {label} | {r['n_I']} | {f(r['interaction_L'])} | {f(r['interaction_O'])} |")
        L += ["", f"**Natural gate (half B, kept I verbs): {'passes' if o['passes'] else 'FAILS: no patching'}.**", "",
              "Per matrix verb / auxiliary (half B, kept I): " + "; ".join(
                  f"{mv}: L {f(r['interaction_L'])}, O {f(r['interaction_O'])}" for mv, r in o["strata"].items()), "",
              "Prepositions the filler raises most after kept I verbs (half B, probability change): " +
              ", ".join(f"{k!r} {v:+.3f}" for k, v in o["top_prep"].items()), "",
              "Class × frame means (half B, log-probs):", "", "| class | frame | O | PREP | . or ? | L | END |",
              "|---|---|---:|---:|---:|---:|---:|"]
        for (cls, frame), r in o["means"].iterrows():
            L.append(f"| {cls} | {frame} | {r.O:.2f} | {r.PREP:.2f} | {r.num:.2f} | {r.L:+.2f} | {r.END:.2f} |")
        L += ["", "Natural z along d_s (all contexts; 0 = active intransitive, 1 = transitive level):", "",
              "| site | T nofill | T fill | I nofill | I fill |", "|---:|---:|---:|---:|---:|"]
        for s in SITES:
            q = zt[(zt.site == s) & (zt.exp == exp)].set_index(["cls", "frame"]).z
            L.append(f"| {s} | {q[('T', 'nofill')]:.2f} | {q[('T', 'fill')]:.2f} | {q[('I', 'nofill')]:.2f} | "
                     f"{q[('I', 'fill')]:.2f} |")
        L.append("")
    L += ["B7 base-form gate (\"NAME can V\", T vs I verbs, AUC over verb means along d_s): "
          + ", ".join(f"site {r.site} {r.auc_verbs:.2f}{'' if r['pass'] else ' (fails)'}" for _, r in bg.iterrows()), ""]
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


# ---------------------------------------------------------------- stage 2
def stage2_stats(d, base_items, T_items, verbs, ctxs, delta, natgap, B, seed):
    """d: patch rows of one experiment (with readout changes and absolute patched O). Returns rows of stats per
    frame and quantity, plus D_fill - D_nofill interactions, on the given verbs and contexts."""
    vix, cix = {v: i for i, v in enumerate(verbs)}, {c: i for i, c in enumerate(ctxs)}
    d = d[d.pair_id.isin(vix) & d.context_id.isin(cix)]
    bt = Boot(len(ctxs), B, seed)
    wv = bt.pair_weights(np.zeros(len(verbs)))
    pix = np.arange(len(verbs))
    one = np.ones((bt.B + 1, 1), np.float32)
    # T level of O per draw: T verbs resampled (their own weights), the same context draws
    tl = {}
    for fr in ("nofill", "fill"):
        t = T_items[(T_items.frame == fr) & T_items.context_id.isin(cix)]
        tv = sorted(t.lemma.unique())
        v, m = cells(t.assign(pair_id=t.lemma, dq=0), {x: i for i, x in enumerate(tv)}, cix, {0: 0}, 1, ["O"])
        wt = bt.pair_weights(np.ones(len(tv)))
        tl[fr] = summarize(pair_values(v, m, bt.wc, one), wt, np.arange(len(tv)))[:, 0]
    cols = list(RD) + ["Oabs"]
    S, out = {}, []
    for fr in ("nofill", "fill"):
        f0 = d[d.frame == fr]
        qs = sorted(f0[f0.cond == "T"].dq.unique())
        qix = {q: i for i, q in enumerate(qs)}
        vT, mT = cells(f0[f0.cond == "T"], vix, cix, qix, len(qs), cols)
        vI, mI = cells(f0[f0.cond == "I"], vix, cix, qix, len(qs), cols)
        assert (mT == mI).all()
        wq = bt.donor_weights("q", len(qs))
        S[(fr, "D")] = summarize(pair_values(vT - vI, mT, bt.wc, wq), wv, pix)
        S[(fr, "T_only")] = summarize(pair_values(vT, mT, bt.wc, wq), wv, pix)
        vT2, mT2 = cells(f0[f0.cond == "inT"].assign(dq=0), vix, cix, {0: 0}, 1, cols)
        vI2, mI2 = cells(f0[f0.cond == "inI"].assign(dq=0), vix, cix, {0: 0}, 1, cols)
        S[(fr, "D_in")] = summarize(pair_values(vT2 - vI2, mT2, bt.wc, one), wv, pix)
        S[(fr, "inT_only")] = summarize(pair_values(vT2, mT2, bt.wc, one), wv, pix)
    for (fr, q), s in S.items():
        for j, r in enumerate(cols):
            if r == "Oabs":
                if q in ("T_only", "inT_only"):
                    out.append({"frame": fr, "quantity": q, "readout": "Olevel", "bound": delta, **ci(s[:, j] - tl[fr])})
                continue
            gap = float(natgap[fr][r])
            bound = delta if r == "O" else 0.2 * abs(gap)
            row = {"frame": fr, "quantity": q, "readout": r, "bound": bound, "natgap": gap, **ci(s[:, j])}
            if q in ("D", "D_in"):
                row["class"] = classify(row, bound) if (r == "O" or abs(gap) >= MINGAP) else "unresolved (small natural gap)"
            out.append(row)
    for q in ("D", "D_in"):
        for j, r in enumerate(RD):
            out.append({"frame": "fill - nofill", "quantity": q, "readout": r, **ci(S[("fill", q)][:, j] - S[("nofill", q)][:, j])})
    return out


def run_patch(args):
    ddir, rdir = Path(args.data_dir), Path(args.out_dir)
    items = pd.read_csv(ddir / "items.csv")
    nat = pd.read_parquet(rdir / "natural.parquet").set_index("pid")
    x = readouts(items, nat).set_index("item_id")
    bg = pd.read_csv(rdir / "baseform_gate.csv").set_index("site")["pass"]
    vt = pd.read_csv(rdir / "verb_tokens.csv")
    rows = []
    for exp in [e for e in ("emb", "mat") if (ddir / f"plan_{e}.csv.gz").exists()]:
        plan = pd.read_csv(ddir / f"plan_{exp}.csv.gz")
        kept = sorted(json.loads((ddir / f"plan_{exp}_meta.json").read_text())["kept_I_verbs"])
        e = x[x.exp == exp].reset_index()
        Ti = e[e.cls == "T"]
        base = x.loc[plan.item_id]
        df = pd.DataFrame({"pair_id": base.lemma.to_numpy(), "context_id": base.context_id.to_numpy(),
                           "frame": base.frame.to_numpy(), "cond": plan.cond.to_numpy(), "half": base.half.to_numpy(),
                           "matrix": base.matrix.to_numpy(), "dq": plan.donor_pair.fillna("").to_numpy()})
        das_verbs = set(e[e.das_pair.fillna("") != ""].lemma)
        for site in SITES:
            pat = pd.read_parquet(rdir / f"patches_{exp}_site{site}.parquet")
            assert (pat.row.to_numpy() == plan.row.to_numpy()).all()
            v = pat.assign(num=pat["dot"] if exp == "emb" else pat["q"])
            v["L"] = v.num - v.PREP
            d = df.copy()
            d[list(RD)] = v[list(RD)].to_numpy() - base[list(RD)].to_numpy()
            d["Oabs"] = v.O.to_numpy()
            delta = site_delta(Path(args.root), site)
            pops = {"half B": (kept, sorted(e[e.half == 1].context_id.unique())),
                    "all contexts": (kept, sorted(e.context_id.unique()))}
            if site in DECL:
                pops["half B, non-DAS verbs"] = ([v_ for v_ in kept if v_ not in das_verbs],
                                                 pops["half B"][1])
                tok = vt[vt.exp == exp].set_index("lemma").n_tokens
                pops["half B, multi-token verbs"] = ([v_ for v_ in kept if tok[v_] > 1], pops["half B"][1])
                for mv in sorted(e.matrix.unique()):
                    pops[f"half B, {mv}"] = (kept, sorted(e[(e.half == 1) & (e.matrix == mv)].context_id.unique()))
            for pop, (verbs, ctxs) in pops.items():
                ee = e[e.context_id.isin(ctxs)]
                natgap = {fr: ee[(ee.cls == "T") & (ee.frame == fr)][list(RD)].mean()
                          - ee[(ee.cls == "I") & (ee.frame == fr) & ee.lemma.isin(verbs)][list(RD)].mean()
                          for fr in ("nofill", "fill")}
                for r in stage2_stats(d, None, Ti, verbs, ctxs, delta, natgap, args.n_boot, args.seed):
                    rows.append({"exp": exp, "site": site, "population": pop, "n_verbs": len(verbs), **r})
            print(f"{exp} site {site} done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(rdir / "patch_summary.csv", index=False)
    write_patch(args, res, bg)


def reading(g, pq, oq):
    dot, o_n = g("nofill", pq, "num"), g("nofill", pq, "O")
    L, num, prep, o_f = g("fill", pq, "L"), g("fill", pq, "num"), g("fill", pq, "PREP"), g("fill", pq, "O")
    lev = g("fill", oq, "Olevel")
    not_above = lev.hi90 < lev.bound
    above = lev.lo95 > 0 and lev.est >= lev.bound
    if (dot["class"] == "FALL" and o_n["class"] == "RISE" and L["class"] == "RISE" and num.lo95 > 0
            and prep["class"] == "FALL" and not_above):
        return "licensing-like crossover", not_above, above
    if o_n["class"] == "RISE" and o_f["class"] == "RISE" and above:
        return "surface (object next)", not_above, above
    return "mixed / unresolved", not_above, above


def write_patch(args, res, bg):
    f = lambda r: f"{r.est:+.2f} [{r.lo95:+.2f}, {r.hi95:+.2f}]"
    L = ["# Round 4, Part B stage 2: filler-gap patches", "",
         "Spec: `plan.md`, B6/B7. Run: `run_fillergap.py patch`; analysis: `analyze_fillergap.py patch`. Bases: kept I "
         "verbs. D = after T − after I active donors; D_in = coordinate at t_T − at t_I (in range). nofill = \"that\" / "
         "no wh; fill = \"what\". O level = O of the patched base (T donor or t_T) − natural O of T verbs (resampled), "
         "same frame and contexts. 95% CIs: base verbs × contexts × donor pairs. Primary: half-B contexts.", ""]
    for exp in res.exp.unique():
        L += [f"## {EXPS[exp]}", ""]
        for pq, oq, nm in (("D", "T_only", "T − I donors"), ("D_in", "inT_only", "in range")):
            L += [f"### {nm} (half B)", "",
                  "| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |",
                  "|---:|---|---|---|---|---|---|---|---|---|"]
            for s in SITES:
                sub = res[(res.exp == exp) & (res.site == s) & (res.population == "half B")]
                g = lambda fr, q, r: sub[(sub.frame == fr) & (sub.quantity == q) & (sub.readout == r)].iloc[0]
                rd, na, ab = reading(g, pq, oq)
                if exp == "mat" and not bool(bg.get(s, True)):
                    rd = "inconclusive (base-form gate)"
                c = [g("nofill", pq, "num"), g("nofill", pq, "O"), g("fill", pq, "L"), g("fill", pq, "num"),
                     g("fill", pq, "PREP"), g("fill", pq, "O")]
                lev, it_ = g("fill", oq, "Olevel"), g("fill - nofill", pq, "O")
                L.append(f"| {s} | " + " | ".join(f"{f(r)} {r['class']}" for r in c) +
                         f" | {f(lev)} {'not above' if na else 'above' if ab else 'unresolved'} | {f(it_)} | **{rd}** |")
            L.append("")
        L += ["### Readings by population (sites 6 and 8)", "", "| Population | verbs | site | T − I donors | in range |",
              "|---|---:|---:|---|---|"]
        for pop in res[res.exp == exp].population.unique():
            for s in DECL:
                sub = res[(res.exp == exp) & (res.site == s) & (res.population == pop)]
                if sub.empty:
                    continue
                g = lambda fr, q, r: sub[(sub.frame == fr) & (sub.quantity == q) & (sub.readout == r)].iloc[0]
                L.append(f"| {pop} | {int(sub.n_verbs.iloc[0])} | {s} | {reading(g, 'D', 'T_only')[0]} | "
                         f"{reading(g, 'D_in', 'inT_only')[0]} |")
        nat = res[(res.exp == exp) & (res.site == 8) & (res.population == "half B") & (res.quantity == "D")
                  & res.frame.isin(["nofill", "fill"])].drop_duplicates(["frame", "readout"])
        L += ["", "Natural T − I gaps (half B, kept I verbs): " + "; ".join(
            f"{r.frame} {r.readout} {r.natgap:+.2f}" for r in nat.itertuples() if r.readout in ("O", "PREP", "num", "L")), ""]
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("gates", "patch"))
    ap.add_argument("--root", default="results/das_round2")
    ap.add_argument("--data-dir", default="data/round4/fillergap")
    ap.add_argument("--out-dir", default="results/round4/fillergap")
    ap.add_argument("--report", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    if a.report is None:
        a.report = "reports/round4/b_fillergap_gates.md" if a.mode == "gates" else "reports/round4/b_fillergap.md"
    run_gates(a) if a.mode == "gates" else run_patch(a)
