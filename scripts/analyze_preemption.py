#!/usr/bin/env python3
"""Round 6, C5 analysis: in-context preemption (reports/round6/plan.md, C5). No model.

Inputs: data/round6/preempt/{contexts.csv, i_frames.csv}; results/round6/preempt/preempt_scores.parquet; the 1.4B D11
frame scores (k = 0 reproduction check); D11 verbs (token counts).
Delta_pass(V | arm, k) = lp(V | ctx + "The N was") - lp(V | ctx + "The N has") (primary; secondary "has been" / "has"),
mean over the 24 nouns. Primary population: rare (Tail + XTail) intransitives annotated dynamic (natural -ing nominal and
progressive); all rare intransitives and the theme-subject subset as sensitivities. Contrasts at k = 8: E_arm = arm -
filler for pre, exp, mention, other, vs, ing; E_pre - E_exp, E_pre - E_mention, E_pre - E_other, E_pre - E_vs,
E_pre - E_ing; distance to the Head level (Head under filler) and its paired reduction; dose slope on log2 k; T control
E_T; raw log-prob levels and changes per frame. Claim of in-context preemption = the conjunction of: E_pre lowers,
beyond exposure, beyond exact-form mention, verb-specific, T control does not fall. Bootstrap 2,000 draws, seed 17: verbs within band x nouns within animacy.
Outputs: reports/round6/c5_preemption.md, results/round6/preempt/c5_summary.csv, c5_decisions.json, c5_verbs.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_frame_contrast import ci, classify, fmt, weights

ARMS = ("pre", "exp", "mention", "fill", "other", "vs", "ing", "trans", "fill_t")
KS = (1, 2, 4, 8)
BOUND = 0.25


def main(args):
    C = pd.read_csv(Path(args.data_dir) / "contexts.csv", keep_default_na=False)
    IF = pd.read_csv(Path(args.data_dir) / "i_frames.csv", keep_default_na=False).set_index("lemma")
    S = pd.read_parquet(Path(args.res_dir) / "preempt_scores.parquet")
    pre = pd.read_csv(args.prefixes, keep_default_na=False)
    fq = pre[pre.set == "freq"]
    nouns, anim = list(fq.noun), fq.animacy.to_numpy()
    assert not S.duplicated(["cid", "noun", "frame"]).any(), "duplicate scores"
    assert len(S) == len(C) * len(nouns) * 3 and set(S.cid) == set(C.cid), "missing scores"
    assert np.isfinite(S.lp).all(), "non-finite scores"
    S = S.merge(C[["cid", "verb", "role", "band", "arm", "k", "lemma"]], on="cid", validate="many_to_one")
    # k = 0 reproduces A1's D11 scores (freq: was / has; been: has been)
    A = pd.read_parquet(args.a1_scores)
    ctx_noun = dict(zip(pre.context_id, pre.noun))
    A = A[A.set.isin(["freq", "been"])].assign(noun=lambda d: d.context_id.map(ctx_noun))
    A["frame2"] = np.where(A.set == "freq", A.frame.map({"pass": "was", "act": "has"}), A.frame.map({"pass": "hasbeen", "act": "has_b"}))
    A = A[A.frame2 != "has_b"]
    k0 = S[S.k == 0].merge(A[["verb", "noun", "frame2", "lp"]], left_on=["verb", "noun", "frame"], right_on=["verb", "noun", "frame2"],
                           suffixes=("", "_a1"))
    assert len(k0) == (S.k == 0).sum(), f"k = 0 rows without an A1 score: {(S.k == 0).sum() - len(k0)}"
    rep = float((k0.lp - k0.lp_a1).abs().max())
    assert rep <= 1e-3, f"k = 0 does not reproduce A1 ({rep})"
    P = S.pivot_table(index=["verb", "role", "band", "arm", "k", "noun"], columns="frame", values="lp").reset_index()
    P["dp"], P["db"] = P.was - P.has, P.hasbeen - P.has
    verbs = C[["verb", "role", "band", "lemma"]].drop_duplicates("verb").reset_index(drop=True)
    vix = {v: i for i, v in enumerate(verbs.verb)}
    nix = {n: i for i, n in enumerate(nouns)}
    cells = {}
    for (arm, k), g in P.groupby(["arm", "k"]):
        for col in ("dp", "db", "was", "has", "hasbeen"):
            M = np.full((len(verbs), len(nouns)), np.nan)
            M[[vix[v] for v in g.verb], [nix[n] for n in g.noun]] = g[col].to_numpy()
            cells[(arm, k, col)] = M
    rng = np.random.default_rng(args.seed)
    wn = weights(rng, anim, args.n_boot)
    group = np.where(verbs.role == "T", "T_" + verbs.band, verbs.band)
    wv = weights(rng, group, args.n_boot)

    def vm(arm, k, col):
        """[draw, verb] noun-weighted verb means (k = 0 is shared by all arms of a verb); NaN for verbs without the arm."""
        M = cells[("none", 0, col)] if k == 0 else cells[(arm, k, col)]
        have = ~np.isnan(M).any(1)
        assert not (np.isnan(M).any(1) & ~np.isnan(M).all(1)).any(), "partially missing nouns"
        out = np.full((len(wn), len(M)), np.nan)
        out[:, have] = (wn @ M[have].T) / wn.sum(1, keepdims=True)
        return out, have

    def gm(Y, mask):
        assert np.isfinite(Y[:, mask]).all(), "an arm is missing for a verb in the population"
        return (wv[:, mask] * Y[:, mask]).sum(1) / wv[:, mask].sum(1)

    ann = {c: np.array([int(IF.loc[l, c]) if r == "I" else 0 for l, r in zip(verbs.lemma, verbs.role)])
           for c in ("dynamic", "theme")}
    rareI_all = ((verbs.role == "I") & verbs.band.isin(["tail", "xtail"])).to_numpy()
    rareI = rareI_all & (ann["dynamic"] == 1)  # primary population
    headI = ((verbs.role == "I") & (verbs.band == "head")).to_numpy() & (ann["dynamic"] == 1)
    rareT = (verbs.role == "T").to_numpy()
    nocomp = np.array([IF.loc[l, "comp"] == "" if r == "I" else True for l, r in zip(verbs.lemma, verbs.role)])
    ntok = pd.read_csv(args.d11_verbs, keep_default_na=False).set_index("verb").n_tok.loc[verbs.verb].to_numpy()
    lines = ["# Round 6, C5: in-context preemption", "",
             "Spec: `plan.md`, C5. Contexts: `build_preemption.py`; run: `run_preemption.py` (Pythia-1.4B, fp32); analysis: "
             "`analyze_preemption.py`. Δ_pass = log P(V | ctx + \"The N was\") − log P(V | ctx + \"The N has\"), mean over A1's 24 "
             f"nouns. k = 0 reproduces A1's D11 scores (max |Δ lp| {rep:.1e}). E_arm = Δ_pass(arm) − Δ_pass(filler) at k = 8. "
             "95% CIs: verbs within band × nouns within animacy, 2,000 draws. Rule (bound 0.25 nats): lowers = 95% CI below 0 and "
             "estimate ≤ −0.25; raises = mirror; negligible = 90% CI within ±0.25.", ""]
    dec, rows = {}, []
    # curves
    lines += ["## Δ_pass by arm and k (means; rare I = Tail + XTail intransitives)", "",
              "| group | arm | k=0 | k=1 | k=2 | k=4 | k=8 |", "|---|---|---|---|---|---|---|"]
    lines.insert(len(lines) - 1, f"Primary population: {int(rareI.sum())} rare dynamic intransitives (of {int(rareI_all.sum())}); "
                 f"Head reference {int(headI.sum())}; T control {int((verbs.role == 'T').sum())}.")
    for gname, mask, arms in (("rare I", rareI, ("pre", "exp", "mention", "fill", "other", "vs", "ing")),
                              ("Head I", headI, ("pre", "exp", "mention", "fill", "other", "vs", "ing")),
                              ("rare T", rareT, ("trans", "fill_t"))):
        for arm in arms:
            vals = [gm(vm(arm, k, "dp")[0], mask) for k in (0,) + KS]
            lines.append(f"| {gname} | {arm} | " + " | ".join(f"{v[0]:+.2f}" for v in vals) + " |")
    lines.append("")

    def E(arm, ref, col, mask, k=8):
        return gm(vm(arm, k, col)[0] - vm(ref, k, col)[0], mask)

    def lowers(d):
        lo, hi = ci(d)
        if hi < 0 and d[0] <= -BOUND:
            return "lowers"
        if lo > 0 and d[0] >= BOUND:
            return "raises"
        lo90, hi90 = ci(d, 5, 95)
        return "negligible" if -BOUND < lo90 and hi90 < BOUND else "unresolved"

    for col, lab in (("dp", "was / has (primary)"), ("db", "has been / has")):
        Epre, Eexp, Eoth = E("pre", "fill", col, rareI), E("exp", "fill", col, rareI), E("other", "fill", col, rareI)
        Emen = E("mention", "fill", col, rareI)
        Evs, Eing = E("vs", "fill", col, rareI), E("ing", "fill", col, rareI)
        # distance to the Head level (Head under filler): baseline gap G > 0 (rare less negative), paired reduction D
        head_ref = gm(vm("fill", 8, col)[0], headI)
        G = gm(vm("fill", 8, col)[0], rareI) - head_ref
        Dred = G - np.abs(gm(vm("pre", 8, col)[0], rareI) - head_ref)
        share = -Epre / G
        share_ok = bool(ci(G)[0] > 0)
        dose = np.empty(len(wv))
        ks = np.log2(np.array(KS, float))
        dk = np.stack([gm(vm("pre", k, col)[0] - vm("fill", k, col)[0], rareI) for k in KS], 1)  # [draw, k]
        dose = ((dk - dk.mean(1, keepdims=True)) * (ks - ks.mean())).sum(1) / ((ks - ks.mean()) ** 2).sum()
        ET = E("trans", "fill_t", col, rareT)
        EpreH = E("pre", "fill", col, headI)
        d = {"E_pre": [Epre[0], *ci(Epre)], "E_pre_reading": lowers(Epre),
             "E_pre_minus_E_exp": [(Epre - Eexp)[0], *ci(Epre - Eexp)],
             "beyond_exposure": bool(ci(Epre - Eexp)[1] < 0 and (Epre - Eexp)[0] <= -BOUND),
             "E_pre_minus_E_mention": [(Epre - Emen)[0], *ci(Epre - Emen)],
             "beyond_mention": bool(ci(Epre - Emen)[1] < 0 and (Epre - Emen)[0] <= -BOUND),
             "E_pre_minus_E_other": [(Epre - Eoth)[0], *ci(Epre - Eoth)], "verb_specific": bool(ci(Epre - Eoth)[1] < 0),
             "E_pre_minus_E_vs": [(Epre - Evs)[0], *ci(Epre - Evs)], "E_pre_minus_E_ing": [(Epre - Eing)[0], *ci(Epre - Eing)],
             "head_gap_G": [G[0], *ci(G)], "distance_reduction": [Dred[0], *ci(Dred)],
             "moves_toward_head": bool(ci(G)[0] > 0 and ci(Dred)[0] > 0),
             "share_gap_closed": [share[0], *ci(share)] if share_ok else "not interpreted (gap not resolved above 0)",
             "E_mention": [Emen[0], *ci(Emen)],
             "dose_slope": [dose[0], *ci(dose)], "E_T": [ET[0], *ci(ET)], "T_does_not_fall": bool(ci(ET)[0] > -BOUND),
             "E_exp": [Eexp[0], *ci(Eexp)], "E_other": [Eoth[0], *ci(Eoth)], "E_vs": [Evs[0], *ci(Evs)],
             "E_ing": [Eing[0], *ci(Eing)], "E_pre_head": [EpreH[0], *ci(EpreH)]}
        d["preemption_claim"] = bool(d["E_pre_reading"] == "lowers" and d["beyond_exposure"] and d["beyond_mention"]
                                     and d["verb_specific"] and d["T_does_not_fall"])
        for lab2, mask in (("all rare I (incl. stative)", rareI_all), ("theme subjects (rare, dynamic)", rareI & (ann["theme"] == 1))):
            d[f"E_pre {lab2}"] = (lambda x: [x[0], *ci(x)])(E("pre", "fill", col, mask))
            d[f"E_pre - E_mention {lab2}"] = (lambda x: [x[0], *ci(x)])(E("pre", "fill", col, mask) - E("mention", "fill", col, mask))
        dec[col] = d
        lines += [f"## Contrasts at k = 8, {lab}", "",
                  f"- **E_pre (rare I) {fmt(Epre)} → {d['E_pre_reading']}**; E_exp {fmt(Eexp)}; E_mention {fmt(Emen)}; "
                  f"E_other {fmt(Eoth)}; E_vs (present) {fmt(Evs)}; E_ing (is V-ing) {fmt(Eing)}.",
                  f"- E_pre − E_exp {fmt(Epre - Eexp)} → beyond exposure: **{d['beyond_exposure']}**; E_pre − E_mention "
                  f"{fmt(Epre - Emen)} → beyond exact-form mention: **{d['beyond_mention']}**; E_pre − E_other "
                  f"{fmt(Epre - Eoth)} → verb-specific: **{d['verb_specific']}**; E_pre − E_vs {fmt(Epre - Evs)}; E_pre − E_ing "
                  f"{fmt(Epre - Eing)}.",
                  f"- Head level (Head I under filler, k = 8) {fmt(head_ref)}; rare − Head gap G {fmt(G)}; distance reduction "
                  f"{fmt(Dred)} → moves toward Head: **{d['moves_toward_head']}**; share closed "
                  + (fmt(share) if share_ok else "not interpreted (G not above 0)") + f". Head I: E_pre {fmt(EpreH)}.",
                  "- Sensitivities: " + "; ".join(f"{k} {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]" for k, v in d.items()
                                                  if k.startswith("E_pre ") or k.startswith("E_pre - E_mention ")) + ".",
                  f"- Dose: slope of Δ_pass(pre) − Δ_pass(filler) on log2 k {fmt(dose)} per doubling.",
                  f"- **T control** E_T {fmt(ET)} → does not fall: **{d['T_does_not_fall']}**.", ""]
        for nm, v in d.items():
            if isinstance(v, list):
                rows.append({"frame": col, "stat": nm, "est": v[0], "lo": v[1], "hi": v[2]})
        if col == "dp":
            lines += ["Raw log-probs at k = 8 (arm − filler):", "",
                      "| group | arm | Δ lp(V \\| was) | Δ lp(V \\| has) | Δ lp(V \\| has been) |", "|---|---|---|---|---|"]
            for gname, mask, arms, ref in (("rare I", rareI, ("pre", "exp", "mention", "other", "vs", "ing"), "fill"),
                                           ("Head I", headI, ("pre", "exp", "mention", "other", "vs", "ing"), "fill"),
                                           ("rare T", rareT, ("trans",), "fill_t")):
                for arm in arms:
                    lines.append(f"| {gname} | {arm} | {fmt(E(arm, ref, 'was', mask))} | {fmt(E(arm, ref, 'has', mask))} | "
                                 f"{fmt(E(arm, ref, 'hasbeen', mask))} |")
            lines += ["", "Raw levels by dose (rare I, mean lp after was / has):", "", "| arm | k=0 | k=1 | k=2 | k=4 | k=8 |",
                      "|---|---|---|---|---|---|"]
            for arm in ("pre", "exp", "mention", "fill", "other", "vs", "ing"):
                lines.append(f"| {arm} | " + " | ".join(f"{gm(vm(arm, k, 'was')[0], rareI)[0]:.2f} / {gm(vm(arm, k, 'has')[0], rareI)[0]:.2f}"
                                                       for k in (0,) + KS) + " |")
            lines += ["", "Breakdowns (E_pre, was / has):", ""]
            for lab2, mask in (("Tail", rareI & (verbs.band == "tail").to_numpy()), ("XTail", rareI & (verbs.band == "xtail").to_numpy()),
                               ("single-token", rareI & (ntok == 1)), ("multi-token", rareI & (ntok > 1)),
                               ("no complement", rareI & nocomp), ("with complement", rareI & ~nocomp)):
                if mask.sum() >= 3:
                    lines.append(f"- {lab2} ({int(mask.sum())} verbs): E_pre {fmt(E('pre', 'fill', 'dp', mask))}, "
                                 f"E_exp {fmt(E('exp', 'fill', 'dp', mask))}.")
            lines.append("")
    lines += ["## Declared decisions (was / has)", "",
              f"- Preemption lowers Δ_pass(I): **{dec['dp']['E_pre_reading']}**; beyond exposure: **{dec['dp']['beyond_exposure']}**; "
              f"beyond exact-form mention: **{dec['dp']['beyond_mention']}**; verb-specific: **{dec['dp']['verb_specific']}**; "
              f"T control does not fall: **{dec['dp']['T_does_not_fall']}** → in-context preemption claimed (all five): "
              f"**{dec['dp']['preemption_claim']}**.",
              f"- Moves toward the Head level (secondary): **{dec['dp']['moves_toward_head']}**.", ""]
    out = Path(args.res_dir)
    Path(args.report).write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(out / "c5_summary.csv", index=False)
    (out / "c5_decisions.json").write_text(json.dumps({**dec, "k0_reproduction": rep}, indent=1, default=float))
    vt = []
    for arm in ARMS:
        for k in KS:
            if (arm, k, "dp") in cells:
                vt.append(pd.DataFrame({"verb": verbs.verb, "role": verbs.role, "band": verbs.band, "arm": arm, "k": k,
                                        "dp": np.nanmean(cells[(arm, k, "dp")], 1)}))
    vt.append(pd.DataFrame({"verb": verbs.verb, "role": verbs.role, "band": verbs.band, "arm": "none", "k": 0,
                            "dp": np.nanmean(cells[("none", 0, "dp")], 1)}))
    pd.concat(vt).to_csv(out / "c5_verbs.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/round6/preempt")
    ap.add_argument("--res-dir", default="results/round6/preempt")
    ap.add_argument("--prefixes", default="data/round6/frame_d11/prefixes.csv")
    ap.add_argument("--a1-scores", default="results/round6/frame_d11/pythia-1.4b/frame_scores.parquet")
    ap.add_argument("--d11-verbs", default="data/round6/frame_d11/verbs.csv")
    ap.add_argument("--report", default="reports/round6/c5_preemption.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    main(ap.parse_args())
