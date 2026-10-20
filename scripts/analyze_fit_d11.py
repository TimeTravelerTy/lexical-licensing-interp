#!/usr/bin/env python3
"""Round 6, A7 analysis: fit controls and robustness on the D11 pairs (reports/round6/plan.md, A7). No model.

Inputs: data/round6/fit_d11/links.csv; Gemma-4-31B-it ratings (round P: results/fit_ratings/gemma4_31b_it.csv,
bad_gemma4_31b_it.csv; round 6: results/round6/fit_d11/gemma4_31b_it.csv), mean of the two prompts per sentence;
A2's 1.4B D11 scores (results/round6/frame_d11/pythia-1.4b/frame_scores.parquet, cur set); A2's and round 5's decisions
(robustness table).
1. Matched-fit regression: M_was (primary), L, Delta_pass(T), Delta_pass(I), M_has per pair x curated passive_1 context on
   band dummies, + good fit, + bad-patient fit (round P's WLS, analyze_bad_fit.fit; fit terms per rating point minus 4);
   bootstrap pairs within band x contexts within context band, 2,000 draws, seed 17; attenuation (unmatched - matched,
   paired); sensitivities: pairs whose I verb forms a complete bare clause, quadratic fit terms, band-specific fit
   slopes. Token-standardized: fit-adjusted Tail - XTail contrasts within the cells 2x2, 2x3, 3x2, 3x3 combined with A2's
   weights 85/36/35/22. Rating validation: completeness, digit mass, prompt agreement, band means, Claude's blind audit.
2. Noun-rarity check: the D11 verbs in the 300 released passive_1 contexts (noun bands; data/round6/frame_d11_rel):
   L and the passive participle margin M_pass by verb band x noun band; verb and context effects. The curated-context
   cross (context band = the band of the context's original verb) is kept as a secondary check.
3. Robustness table: every A1 prefix set, D11 (1.4B) and curated (round 5); D11 freq for every suite size.
Outputs: reports/round6/a7_fit_controls.md, results/round6/fit_d11/a7_summary.csv, a7_decisions.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_bad_fit import fit as wls_fit
from analyze_frame_contrast import ci, classify, fmt, weights

BANDS = ("head", "tail", "xtail")
NAME = {"head": "Head", "tail": "Tail", "xtail": "XTail"}
CELLS = ("2×2", "2×3", "3×2", "3×3")


def ratings(paths):
    parts = []
    for p in paths:
        r = pd.read_csv(p, usecols=["sentence", "prompt", "rating", "digit_mass"])
        assert not r.duplicated(["sentence", "prompt"]).any(), f"duplicate (sentence, prompt) rows in {p}"
        parts.append(r.assign(src=str(Path(p).resolve())))
    R = pd.concat(parts).drop_duplicates(["sentence", "prompt"])  # round-P rows first: identical sentences reuse them
    w = R.pivot_table(index="sentence", columns="prompt", values="rating")
    assert w.notna().all().all(), "a sentence lacks a prompt's rating"
    return w.mean(1), R, w


def main(args):
    rng = np.random.default_rng(args.seed)
    L = pd.read_csv(args.links, keep_default_na=False)
    rat, raw, wide = ratings([args.good_r5, args.bad_r5, args.new_ratings])
    missing = (set(L.good_sentence) | set(L.bad_sentence)) - set(rat.index)
    assert not missing, f"{len(missing)} sentences without ratings"
    L["good_fit"] = L.good_sentence.map(rat)
    L["bad_patient"] = L.bad_sentence.map(rat)
    pairs = pd.read_csv(args.d11_pairs, keep_default_na=False).set_index("verb_pair")
    S = pd.read_parquet(args.scores)
    s = S[S.set == "cur"]
    lp = s.pivot_table(index=["verb", "frame"], columns="context_id", values="lp", aggfunc="first")
    rows_ = []
    for r in L.itertuples(index=False):
        p = pairs.loc[r.pair_id]
        tw, ta = lp.loc[(p.t_verb, "pass"), r.context_id], lp.loc[(p.t_verb, "act"), r.context_id]
        iw, ia = lp.loc[(p.i_verb, "pass"), r.context_id], lp.loc[(p.i_verb, "act"), r.context_id]
        rows_.append((tw - iw, (tw - ta) - (iw - ia), ta - ia, f"{p.t_ntok}×{p.i_ntok}", tw - ta, iw - ia))
    L["M_was"], L["L"], L["M_has"], L["cell"], L["dT"], L["dI"] = zip(*rows_)
    assert np.isfinite(L[["M_was", "L", "M_has", "dT", "dI"]].to_numpy()).all()
    IF = pd.read_csv(args.i_frames, keep_default_na=False).set_index("lemma")
    L["bare_ok"] = [int(IF.loc[pairs.loc[p, "i_lemma"], "bare_ok"]) for p in L.pair_id]
    lines = ["# Round 6, A7: fit controls and robustness on the D11 pairs", "",
             "Spec: `plan.md`, A7. Ratings: Gemma-4-31B-it (`rate_fit.py`, two prompts averaged; round-P ratings reused for "
             f"identical sentences); outcomes from A2's 1.4B cur set. {len(L)} pair × context rows (281 pairs × 126 curated "
             "passive_1 contexts). 95% CIs: pairs within band × contexts within context band, 2,000 draws.", "",
             f"Rating format: digit mass median {raw.digit_mass.median():.3f}; good-fit mean by verb band "
             + ", ".join(f"{NAME[b]} {L[L.band == b].good_fit.mean():.2f}" for b in BANDS) + "; bad-patient mean "
             + ", ".join(f"{NAME[b]} {L[L.band == b].bad_patient.mean():.2f}" for b in BANDS) + ".", ""]
    new = raw[raw.src == str(Path(args.new_ratings).resolve())]
    nw = wide.loc[wide.index.isin(set(new.sentence))]
    lines += ["Rating validation (new sentences): " + f"{len(nw)} sentences; digit mass median {new.digit_mass.median():.3f}, "
              f"{(new.digit_mass < 0.9).mean() * 100:.1f}% below 0.9; prompt agreement (Spearman) "
              f"{nw.iloc[:, 0].rank().corr(nw.iloc[:, 1].rank()):.3f}.", ""]
    au = Path(args.audit)
    if au.exists():
        A = pd.read_csv(au, keep_default_na=False)
        A = A[A.claude_rating != ""]
        A["gemma"] = A.sentence.map(rat)
        rho = A.claude_rating.astype(float).rank().corr(A.gemma.rank())
        lines += [f"Blind audit: Claude rated {len(A)} stratified new sentences before seeing Gemma's ratings; Spearman {rho:.3f}.", ""]
    up, uc = sorted(L.pair_id.unique()), sorted(L.context_id.unique())
    vi, cix = pd.Index(up).get_indexer(L.pair_id), pd.Index(uc).get_indexer(L.context_id)
    wv = weights(rng, L.groupby("pair_id").band.first().loc[up].to_numpy(), args.n_boot)
    wc = weights(rng, L.groupby("context_id").context_band.first().loc[uc].to_numpy(), args.n_boot)
    one = np.ones(len(L))
    t, x = (L.band == "tail").to_numpy(float), (L.band == "xtail").to_numpy(float)
    good, pat = L.good_fit.to_numpy() - 4, L.bad_patient.to_numpy() - 4
    designs = {"unmatched": (["intercept", "tail", "xtail"], np.column_stack([one, t, x])),
               "good": (["intercept", "good_fit", "tail", "xtail"], np.column_stack([one, good, t, x])),
               "good+bad_patient": (["intercept", "good_fit", "bad_patient", "tail", "xtail"], np.column_stack([one, good, pat, t, x]))}
    dec, rows = {}, []
    lines += ["## 1. Matched-fit regression", "", "| outcome | model | Tail (95% CI) | XTail (95% CI) | good fit | bad patient |",
              "|---|---|---|---|---|---|"]
    coefs = {}
    for y_name in ("M_was", "L", "dT", "dI", "M_has"):
        y = L[y_name].to_numpy(float)
        for model, (names, X) in designs.items():
            d = wls_fit(L, y, X, vi, cix, wv, wc)
            coefs[(y_name, model)] = d
            get = lambda n: fmt(d[:, names.index(n)]) if n in names else "—"
            lines.append(f"| {y_name} | {model} | {get('tail')} | {get('xtail')} | {get('good_fit')} | {get('bad_patient')} |")
            for j, n in enumerate(names):
                rows.append({"part": "fit", "outcome": y_name, "model": model, "term": n, "est": d[0, j],
                             "lo": ci(d[:, j])[0], "hi": ci(d[:, j])[1]})
    primary = coefs[("M_was", "good+bad_patient")]
    xt = primary[:, 4]
    tl = primary[:, 3]

    def survive(v, unmatched):
        """XTail - Head coefficients (deficit < 0). Requires an unmatched deficit."""
        if not (ci(unmatched)[1] < 0 and unmatched[0] <= -0.25):
            return "no unmatched deficit"
        if ci(v)[1] < 0 and v[0] <= -0.25:
            return "survives matched fit"
        lo90, hi90 = ci(v, 5, 95)
        return "negligible after fit adjustment" if -0.25 < lo90 and hi90 < 0.25 else "unresolved"

    acc = coefs[("L", "good+bad_patient")][:, 4] + coefs[("M_has", "good+bad_patient")][:, 4]
    dec["matched_fit"] = {}
    for y_name in ("M_was", "L"):
        un, mt = coefs[(y_name, "unmatched")], coefs[(y_name, "good+bad_patient")]
        for term, j_un, j_mt in (("xtail", 2, 4), ("tail", 1, 3)):
            att = un[:, j_un] - mt[:, j_mt]
            dec["matched_fit"][f"{y_name}_{term}"] = {"matched": [mt[0, j_mt], *ci(mt[:, j_mt])],
                                                      "unmatched": [un[0, j_un], *ci(un[:, j_un])],
                                                      "attenuation": [att[0], *ci(att)],
                                                      "reading": survive(mt[:, j_mt], un[:, j_un])}
    dec["matched_fit"]["accounting_identity_err"] = float(np.abs(acc - xt).max())
    sides = {k: coefs[(k, "good+bad_patient")][:, 4] for k in ("dT", "dI")}
    dec["matched_fit"]["sides_xtail"] = {"dT": [sides["dT"][0], *ci(sides["dT"])], "minus_dI": [-sides["dI"][0], *ci(-sides["dI"])]}
    # sensitivities on M_was (XTail coefficient)
    sens = {}
    bm = L.bare_ok.to_numpy() == 1
    Xb = np.column_stack([one, good, pat, t, x])[bm]
    sens["bare-clause I verbs only"] = wls_fit(L[bm], L.M_was.to_numpy(float)[bm], Xb, vi[bm], cix[bm], wv, wc)[:, 4]
    Xq = np.column_stack([one, good, pat, good ** 2, pat ** 2, t, x])
    sens["quadratic fit terms"] = wls_fit(L, L.M_was.to_numpy(float), Xq, vi, cix, wv, wc)[:, 6]
    Xs_ = np.column_stack([one, good, pat, good * t, good * x, pat * t, pat * x, t, x])
    sens["band-specific fit slopes (at fit = 4)"] = wls_fit(L, L.M_was.to_numpy(float), Xs_, vi, cix, wv, wc)[:, 8]
    dec["matched_fit"]["sensitivities_xtail_M_was"] = {k: [v[0], *ci(v)] for k, v in sens.items()}
    # token-standardized: fit-adjusted Tail - XTail within each cell, combined with A2's fixed weights
    m = L.band.isin(["tail", "xtail"]).to_numpy() & L.cell.isin(CELLS).to_numpy()
    Lm = L[m]
    W4 = {"2×2": 85, "2×3": 36, "3×2": 35, "3×3": 22}
    cd = [(Lm.cell == c).to_numpy(float) for c in CELLS]
    tl_c = [((Lm.cell == c) & (Lm.band == "tail")).to_numpy(float) for c in CELLS]
    Xm = np.column_stack(cd + tl_c + [Lm.good_fit - 4, Lm.bad_patient - 4]).astype(float)
    # pairs resampled within band x cell (as A2's token contrast), so every cell keeps Tail and XTail support
    upm = sorted(Lm.pair_id.unique())
    vim = pd.Index(upm).get_indexer(Lm.pair_id)
    wvm = weights(rng, Lm.groupby("pair_id").apply(lambda g: f"{g.band.iloc[0]}|{g.cell.iloc[0]}").loc[upm].to_numpy(), args.n_boot)
    for y_name in ("M_was", "L"):
        bm_ = wls_fit(Lm, Lm[y_name].to_numpy(float), Xm, vim, cix[m], wvm, wc)
        dm = sum(W4[c] * bm_[:, 4 + i] for i, c in enumerate(CELLS)) / sum(W4.values())  # Tail - XTail (deficit > 0)
        dec[f"matched_fit_token_std_{y_name}"] = {"tail_minus_xtail": [dm[0], *ci(dm)], "reading": classify(dm)}
    mf = dec["matched_fit"]
    lines += ["", f"- **XTail − Head on M_was at matched fit (good + bad patient): {fmt(xt)} → {mf['M_was_xtail']['reading']}** "
                  f"(unmatched {mf['M_was_xtail']['unmatched'][0]:+.2f}; attenuation {mf['M_was_xtail']['attenuation'][0]:+.2f} "
                  f"[{mf['M_was_xtail']['attenuation'][1]:+.2f}, {mf['M_was_xtail']['attenuation'][2]:+.2f}]); Tail − Head {fmt(tl)} → "
                  f"{mf['M_was_tail']['reading']}. L: XTail {mf['L_xtail']['matched'][0]:+.2f} [{mf['L_xtail']['matched'][1]:+.2f}, "
                  f"{mf['L_xtail']['matched'][2]:+.2f}] → {mf['L_xtail']['reading']}; its sides at matched fit: Δ_pass(T) "
                  f"{mf['sides_xtail']['dT'][0]:+.2f}, −Δ_pass(I) {mf['sides_xtail']['minus_dI'][0]:+.2f}. Accounting β(M_was) = β(L) + "
                  f"β(M_has): max error {mf['accounting_identity_err']:.1e}.",
              "- Sensitivities (XTail on M_was): " + "; ".join(f"{k} {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]"
                                                               for k, v in mf["sensitivities_xtail_M_was"].items()) + ".",
              "- Token-standardized (fit-adjusted Tail − XTail within cells 2×2, 2×3, 3×2, 3×3, weights 85/36/35/22; deficit > 0): "
              + "; ".join(f"{y} {dec[f'matched_fit_token_std_{y}']['tail_minus_xtail'][0]:+.2f} "
                          f"[{dec[f'matched_fit_token_std_{y}']['tail_minus_xtail'][1]:+.2f}, {dec[f'matched_fit_token_std_{y}']['tail_minus_xtail'][2]:+.2f}] → "
                          f"{dec[f'matched_fit_token_std_{y}']['reading']}" for y in ("M_was", "L")) + ".", ""]
    # 2. noun rarity: released contexts (noun bands), primary
    Sr = pd.read_parquet(args.rel_scores)
    pre_r = pd.read_csv(args.rel_prefixes, keep_default_na=False).set_index("context_id")
    lr = Sr[Sr.set == "rel"].pivot_table(index=["verb", "frame"], columns="context_id", values="lp", aggfunc="first")
    rr = []
    for pid, p in pairs.iterrows():
        tw, ta, iw, ia = (lr.loc[(p.t_verb, "pass")], lr.loc[(p.t_verb, "act")], lr.loc[(p.i_verb, "pass")], lr.loc[(p.i_verb, "act")])
        for c in lr.columns:
            rr.append({"pair_id": pid, "band": p.band, "context_id": c, "context_band": pre_r.loc[c, "context_band"],
                       "M_pass": tw[c] - iw[c], "L": (tw[c] - ta[c]) - (iw[c] - ia[c])})
    Rr = pd.DataFrame(rr)
    assert np.isfinite(Rr[["M_pass", "L"]].to_numpy()).all() and len(Rr) == len(pairs) * 300
    upr, ucr = sorted(Rr.pair_id.unique()), sorted(Rr.context_id.unique())
    vir, cir = pd.Index(upr).get_indexer(Rr.pair_id), pd.Index(ucr).get_indexer(Rr.context_id)
    wvr = weights(rng, Rr.groupby("pair_id").band.first().loc[upr].to_numpy(), args.n_boot)
    wcr = weights(rng, Rr.groupby("context_id").context_band.first().loc[ucr].to_numpy(), args.n_boot)
    lines += ["## 2. Noun-rarity check: D11 verbs in the 300 released passive_1 contexts (noun bands)", "",
              "| outcome | verb band | Head-noun contexts | Tail-noun contexts | XTail-noun contexts |", "|---|---|---|---|---|"]

    def cross(D, vi_, ci_, wv_, wc_, outcomes, tag):
        vb_, cb_ = D.band.to_numpy(), D.context_band.to_numpy()

        def cm_(y, mask):
            w = wv_[:, vi_[mask]] * wc_[:, ci_[mask]]
            return (w * y[mask][None]).sum(1) / w.sum(1)

        for y_name in outcomes:
            y = D[y_name].to_numpy(float)
            M = {(a, b): cm_(y, (vb_ == a) & (cb_ == b)) for a in BANDS for b in BANDS}
            for a in BANDS:
                lines.append(f"| {y_name} | {NAME[a]} | " + " | ".join(f"{M[(a, b)][0]:+.2f}" for b in BANDS) + " |")
            verb_eff = np.mean([M[("xtail", b)] - M[("head", b)] for b in BANDS], axis=0)
            ctx_eff = np.mean([M[(a, "xtail")] - M[(a, "head")] for a in BANDS], axis=0)
            lo90, hi90 = ci(ctx_eff, 5, 95)
            rd = "negligible" if -0.25 < lo90 and hi90 < 0.25 else "unresolved" if ci(ctx_eff)[0] < 0 < ci(ctx_eff)[1] else "context effect"
            dec.setdefault(tag, {})[y_name] = {"verb_effect": [verb_eff[0], *ci(verb_eff)], "verb_reading": classify(-verb_eff),
                                               "context_effect": [ctx_eff[0], *ci(ctx_eff)], "context_reading": rd}
            lines.append(f"| {y_name} | effects | verb (XTail − Head) {fmt(verb_eff)} → Head − XTail {classify(-verb_eff)} | "
                         f"context (XTail − Head) {fmt(ctx_eff)} → {rd} | |")

    cross(Rr, vir, cir, wvr, wcr, ("L", "M_pass"), "noun_rarity")
    lines += ["", "Secondary: the curated contexts (context band = the band of the context's original verb, not noun "
                  "frequency; patient Zipf medians 4.96 / 4.70 / 4.45):", "",
              "| outcome | verb band | Head contexts | Tail contexts | XTail contexts |", "|---|---|---|---|---|"]
    cross(L, vi, cix, wv, wc, ("L", "M_was"), "curated_context_cross")
    lines.append("")
    # 3. robustness table
    A2 = json.loads(Path(args.a2_decisions).read_text())
    R5 = pd.read_csv(args.r5_summary)
    lines += ["## 3. Robustness: every A1 prefix set", "",
              "| population | set | ΔL Head − XTail | reading | ΔL_tok (Tail − XTail) | reading | side (token-std.) |",
              "|---|---|---|---|---|---|---|"]
    for st in ("freq", "been", "cur", "they"):
        d = A2["pythia-1.4b"][st]
        lines.append(f"| D11 (1.4B) | {st} | {d['dL_head_xtail'][0]:+.2f} [{d['dL_head_xtail'][1]:+.2f}, {d['dL_head_xtail'][2]:+.2f}] | "
                     f"{d['dL_head_xtail_reading']} | {d['L_tok'][0]:+.2f} [{d['L_tok'][1]:+.2f}, {d['L_tok'][2]:+.2f}] | "
                     f"{d['L_tok_reading']} | {d['side_tok']} |")
    for st in ("freq", "been", "cur", "they"):
        r = R5[(R5.set == st) & (R5.quantity == "L") & (R5.stat == "head-xtail")]
        if len(r):
            r = r.iloc[0]
            lines.append(f"| curated 126 (1.4B, round 5) | {st} | {r.est:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}] | — | — | — | — |")
    for mdl, d in A2.items():
        if mdl.startswith("pythia") and mdl != "pythia-1.4b" and "freq" in d:
            f_ = d["freq"]
            lines.append(f"| D11 ({mdl}) | freq | {f_['dL_head_xtail'][0]:+.2f} [{f_['dL_head_xtail'][1]:+.2f}, {f_['dL_head_xtail'][2]:+.2f}] | "
                         f"{f_['dL_head_xtail_reading']} | {f_['L_tok'][0]:+.2f} [{f_['L_tok'][1]:+.2f}, {f_['L_tok'][2]:+.2f}] | "
                         f"{f_['L_tok_reading']} | {f_['side_tok']} |")
    lines.append("")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(out / "a7_summary.csv", index=False)
    (out / "a7_decisions.json").write_text(json.dumps(dec, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--links", default="data/round6/fit_d11/links.csv")
    ap.add_argument("--good-r5", default="results/fit_ratings/gemma4_31b_it.csv")
    ap.add_argument("--bad-r5", default="results/fit_ratings/bad_gemma4_31b_it.csv")
    ap.add_argument("--new-ratings", default="results/round6/fit_d11/gemma4_31b_it.csv")
    ap.add_argument("--d11-pairs", default="data/round6/frame_d11/pairs.csv")
    ap.add_argument("--scores", default="results/round6/frame_d11/pythia-1.4b/frame_scores.parquet")
    ap.add_argument("--a2-decisions", default="results/round6/frame_d11/a2_decisions.json")
    ap.add_argument("--r5-summary", default="results/round5/frame/a1_a2_summary.csv")
    ap.add_argument("--rel-scores", default="results/round6/frame_d11_rel/pythia-1.4b/frame_scores.parquet")
    ap.add_argument("--rel-prefixes", default="data/round6/frame_d11_rel/prefixes.csv")
    ap.add_argument("--i-frames", default="data/round6/preempt/i_frames.csv")
    ap.add_argument("--audit", default="data/round6/fit_d11/audit.csv")
    ap.add_argument("--out-dir", default="results/round6/fit_d11")
    ap.add_argument("--report", default="reports/round6/a7_fit_controls.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    main(ap.parse_args())
