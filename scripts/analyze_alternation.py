#!/usr/bin/env python3
"""Round 6, B3 analysis: object-onset and clause-end expectation after T and I verbs (reports/round6/plan.md, B3;
revised after review). No model.

Inputs: data/round6/alternation/prefixes.csv; results/round6/alternation/<model>/next_sets.parquet; D11 pairs / verbs
(data/round6/frame_d11); data/round6/preempt/i_frames.csv (bare_ok: the I verb forms a complete bare clause); A1's counts
(data/round6/counts/pile_counts.csv; count analyses skipped if absent); released FreqBLiMP items and round-5 A4's
unedited 1.4B scores; data/round5/verbs_expanded/pairs.csv (lemmas).
Readouts per verb and frame (past primary, perf secondary), mean over the 24 nouns: log P(O) (object-onset proxy; strict
objective pronouns as sensitivity), log P(DOT) (primary clause end; END set as sensitivity), log P(PREP).
M_obj = log P(O|T) - log P(O|I) on all pairs; M_end = log P(DOT|I) - log P(DOT|T) on eligible pairs (T obligatory for
both raters and I bare_ok). Band contrasts (token-standardized Tail - XTail, fixed weights 85/36/35/22; Head - XTail) with
sides; per-verb slopes on Zipf (frame-specific log token count as covariate) and the paired slope difference between the
classes (side specificity); counts (A1's CV machinery, token count in both models); a descriptive item-level association
with released accuracy. Bootstrap 2,000 draws, seed 17: pairs within band (x token cell) x nouns within animacy;
identical draws for every model.
Outputs: reports/round6/b3_alternations.md, results/round6/alternation/b3_summary.csv, b3_decisions.json, b3_verbs.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_exposure import cv_mse, decide, fold_map, wls
from analyze_frame_contrast import ci, classify, fmt, weights
from verb_forms import forms

MODELS = ("pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b", "pythia-2.8b", "pythia-6.9b", "pythia-12b")
W4 = {"2×2": 85, "2×3": 36, "3×2": 35, "3×3": 22}
SETS = ("O", "O_strict", "END", "DOT", "PREP")
PARADIGMS = {"causative": ("I", "O"), "transitive": ("I", "O"), "inchoative": ("T", "DOT"),
             "drop_argument": ("T", "DOT"), "intransitive": ("T", "DOT")}
BANDS = ("head", "tail", "xtail")
NAME = {"head": "Head", "tail": "Tail", "xtail": "XTail"}


def verb_means(R, P, frame, wn, nouns):
    """{set: ([draw, pair] for T, [draw, pair] for I)} noun-weighted means; also frame-specific token counts."""
    x = R[R.frame == frame]
    out = {}
    for s in SETS:
        tab = x.pivot(index="verb", columns="noun", values=s)
        assert tab.shape[1] == len(nouns) and tab.notna().all().all(), f"missing {s} scores"
        tab = tab[nouns]
        Xt, Xi = tab.loc[P.t_verb].to_numpy(), tab.loc[P.i_verb].to_numpy()
        out[s] = ((wn @ Xt.T) / wn.sum(1, keepdims=True), (wn @ Xi.T) / wn.sum(1, keepdims=True))
    nt = x.drop_duplicates("verb").set_index("verb").n_verb_tok
    return out, nt.loc[P.t_verb].to_numpy(float), nt.loc[P.i_verb].to_numpy(float)


def pmean(wv, Y, mask):
    return (wv[:, mask] * Y[:, mask]).sum(1) / wv[:, mask].sum(1)


def tok_contrast(wc, Y, P, subset):
    num, den = 0.0, 0.0
    for s, w in W4.items():
        mt = subset & (P.band == "tail").to_numpy() & (P.cell == s).to_numpy()
        mx = subset & (P.band == "xtail").to_numpy() & (P.cell == s).to_numpy()
        if mt.sum() == 0 or mx.sum() == 0:
            continue
        num = num + w * (pmean(wc, Y, mt) - pmean(wc, Y, mx))
        den += w
    return num / den


def side(total, t_part, i_part):
    if classify(total) != "deficit":
        return "no deficit (contributions descriptive)"
    if ci(i_part)[0] > 0 and ci(i_part - t_part)[0] > 0:
        return "I-side"
    if ci(t_part)[0] > 0 and ci(t_part - i_part)[0] > 0:
        return "T-side"
    return "mixed"


def slope(Y, z, ntok, wv, categorical=False):
    """Per-draw slope of Y on z with token count (log, or dummies for 1 / 2 / 3 / 4+) as covariates."""
    if categorical:
        c = np.minimum(ntok, 4)
        cols = [c == k for k in (2, 3, 4) if (c == k).any()]
        X = np.column_stack([np.ones(len(z)), z] + cols).astype(float)
    else:
        X = np.column_stack([np.ones(len(z)), z, np.log(ntok)])
    return np.array([wls(X, Y[b], wv[b])[1] for b in range(len(Y))])


def slope_reading(s, pred):
    """pred -1: tracks frequency (CI below 0) / reversed (CI above 0) / flat (90% CI within +-0.25) / unresolved;
    pred 0: flat / slope != 0 / unresolved."""
    lo, hi = ci(s)
    lo90, hi90 = ci(s, 5, 95)
    if pred < 0:
        if hi < 0:
            return "tracks frequency"
        if lo > 0:
            return "reversed"
    elif lo > 0 or hi < 0:
        return "slope ≠ 0"
    return "flat" if -0.25 < lo90 and hi90 < 0.25 else "unresolved"


def diff_reading(d):
    """Side specificity: the predicted-side slope minus the other side's slope; predicted < 0."""
    lo, hi = ci(d)
    return "side-specific" if hi < 0 else "reversed" if lo > 0 else "not side-specific (unresolved)"


def main(args):
    P = pd.read_csv(Path(args.d11) / "pairs.csv", keep_default_na=False)
    P["cell"] = [f"{a}×{b}" for a, b in zip(P.t_ntok, P.i_ntok)]
    IF = pd.read_csv(args.i_frames, keep_default_na=False).set_index("lemma")
    P["oblig"] = P.t_flag.eq("obligatory|obligatory")
    P["bare_ok"] = P.i_lemma.map(IF.bare_ok).astype(int).eq(1)
    P["eligible_end"] = P.oblig & P.bare_ok
    P["rt_any"] = P.i_flag.str.contains("rare_trans")
    P["rt_both"] = P.i_flag.eq("rare_trans|rare_trans")
    Pre = pd.read_csv(Path(args.data_dir) / "prefixes.csv", keep_default_na=False)
    nouns = list(dict.fromkeys(Pre[Pre.frame == "past"].noun))
    an = Pre[Pre.frame == "past"].drop_duplicates("noun").set_index("noun").animacy.loc[nouns].to_numpy()
    lines = ["# Round 6, B3: object-onset and clause-end expectation after T and I verbs", "",
             "Spec: `plan.md`, B3. Run: `run_next_sets.py` on `data/round6/alternation`; analysis: `analyze_alternation.py`. "
             "Frames: \"The N p\" (simple past; primary) and \"The N has v\" (perfect); 24 A1 nouns. O = round 2's object-onset "
             "set (an object-onset proxy; strict objective pronouns him / them / me / us as sensitivity); clause end = \".\" "
             "(primary; the punctuation set as sensitivity). M_obj = log P(O|T) − log P(O|I) (all 281 pairs); M_end = "
             "log P(.|I) − log P(.|T) on the eligible pairs (T obligatory for both raters and I forms a complete bare clause: "
             f"{int(P.eligible_end.sum())} pairs). Token-standardized Tail − XTail: cells 2×2, 2×3, 3×2, 3×3, weights "
             "85 / 36 / 35 / 22. Rule (bound 0.25 nats) as A2. Side specificity: the paired difference of class slopes on Zipf. "
             "95% CIs: pairs within band (× cell) × nouns within animacy, 2,000 draws, identical across models.", ""]
    dec, rows, summ, verbs_out = {}, [], [], []
    allp, el = np.ones(len(P), bool), P.eligible_end.to_numpy()
    for model in args.models:
        f = Path(args.res_dir) / model / "next_sets.parquet"
        if not f.exists():
            lines.append(f"- {model}: no output.")
            continue
        rng = np.random.default_rng(args.seed)
        X = pd.read_parquet(f)
        assert X.id.is_unique and set(X.id) == set(Pre.id), f"{model}: next-set rows do not match the prefixes"
        R = Pre.merge(X, on="id", validate="one_to_one")
        wn = weights(rng, an, args.n_boot)
        wv = weights(rng, P.band.to_numpy(), args.n_boot)
        wc = weights(rng, np.array([f"{b}|{c}" for b, c in zip(P.band, P.cell)]), args.n_boot)
        dm = dec[model] = {}
        for frame in ("past", "perf"):
            vm, ntT, ntI = verb_means(R, P, frame, wn, nouns)
            Q = {"O_T": vm["O"][0], "O_I": vm["O"][1], "Os_T": vm["O_strict"][0], "Os_I": vm["O_strict"][1],
                 "D_T": vm["DOT"][0], "D_I": vm["DOT"][1], "E_T": vm["END"][0], "E_I": vm["END"][1],
                 "P_T": vm["PREP"][0], "P_I": vm["PREP"][1]}
            fr = dm[frame] = {}
            specs = {"M_obj": ("O", allp, "obj"), "M_obj_strict": ("Os", allp, "obj"), "M_end": ("D", el, "end"),
                     "M_end_punct": ("E", el, "end"), "M_end_all_oblig": ("D", P.oblig.to_numpy(), "end"),
                     "M_obj_no_rare_trans_any": ("O", ~P.rt_any.to_numpy(), "obj"),
                     "M_obj_no_rare_trans_both": ("O", ~P.rt_both.to_numpy(), "obj"),
                     "M_end_no_rare_trans_any": ("D", el & ~P.rt_any.to_numpy(), "end"),
                     "M_end_no_rare_trans_both": ("D", el & ~P.rt_both.to_numpy(), "end")}
            for meas, (key, sub, kind) in specs.items():
                T_, I_ = Q[f"{key}_T"], Q[f"{key}_I"]
                Y = T_ - I_ if kind == "obj" else I_ - T_
                hb, xb = sub & (P.band == "head").to_numpy(), sub & (P.band == "xtail").to_numpy()
                hx = pmean(wv, Y, hb) - pmean(wv, Y, xb)
                tk = tok_contrast(wc, Y, P, sub)
                rec = {"n_pairs": int(sub.sum()), "band": {b: pmean(wv, Y, sub & (P.band == b).to_numpy())[0] for b in BANDS}}
                for nm, tot, dT, dI in (("tok", tk, tok_contrast(wc, T_, P, sub), tok_contrast(wc, I_, P, sub)),
                                        ("head_xtail", hx, pmean(wv, T_, hb) - pmean(wv, T_, xb), pmean(wv, I_, hb) - pmean(wv, I_, xb))):
                    tp, ip = (dT, -dI) if kind == "obj" else (-dT, dI)  # contributions to the deficit
                    rec[nm] = [tot[0], *ci(tot)]
                    rec[f"{nm}_reading"] = classify(tot)
                    rec[f"{nm}_side"] = side(tot, tp, ip)
                    rec[f"{nm}_T_part"], rec[f"{nm}_I_part"] = [tp[0], *ci(tp)], [ip[0], *ci(ip)]
                fr[meas] = rec
                for nm in ("tok", "head_xtail"):
                    rows.append({"model": model, "frame": frame, "measure": meas, "stat": nm, "est": rec[nm][0],
                                 "lo": rec[nm][1], "hi": rec[nm][2]})
            # per-verb slopes and side specificity (paired, same pairs for both classes)
            zt, zi = P.t_zipf.to_numpy(float), P.i_zipf.to_numpy(float)
            sl = {}
            for lab, key, sub, pred_side in (("O", "O", allp, "I"), ("O_strict", "Os", allp, "I"), ("DOT", "D", el, "T"),
                                             ("END", "E", el, "T")):
                sT = slope(Q[f"{key}_T"][:, sub], zt[sub], ntT[sub], wv[:, sub])
                sI = slope(Q[f"{key}_I"][:, sub], zi[sub], ntI[sub], wv[:, sub])
                sTc = slope(Q[f"{key}_T"][:, sub], zt[sub], ntT[sub], wv[:, sub], categorical=True)
                sIc = slope(Q[f"{key}_I"][:, sub], zi[sub], ntI[sub], wv[:, sub], categorical=True)
                pred_s, other_s = (sI, sT) if pred_side == "I" else (sT, sI)
                d = pred_s - other_s
                pc, oc = (sIc, sTc) if pred_side == "I" else (sTc, sIc)
                sl[lab] = {"predicted_side": pred_side, "slope_pred": [pred_s[0], *ci(pred_s)], "reading_pred": slope_reading(pred_s, -1),
                           "slope_other": [other_s[0], *ci(other_s)], "reading_other": slope_reading(other_s, 0),
                           "diff": [d[0], *ci(d)], "side_specificity": diff_reading(d),
                           "categorical_tokens": {"slope_pred": [pc[0], *ci(pc)], "diff": [(pc - oc)[0], *ci(pc - oc)],
                                                  "side_specificity": diff_reading(pc - oc)}}
                ms = sub & P.band.isin(["tail", "xtail"]).to_numpy() & (P.t_ntok >= 2).to_numpy() & (P.i_ntok >= 2).to_numpy()
                a = slope(Q[f"{key}_{pred_side}"][:, ms], (zi if pred_side == "I" else zt)[ms], (ntI if pred_side == "I" else ntT)[ms], wv[:, ms])
                b = slope(Q[f"{key}_{'T' if pred_side == 'I' else 'I'}"][:, ms], (zt if pred_side == "I" else zi)[ms],
                          (ntT if pred_side == "I" else ntI)[ms], wv[:, ms])
                sl[lab]["common_support_multi"] = {"slope_pred": [a[0], *ci(a)], "diff": [(a - b)[0], *ci(a - b)],
                                                   "side_specificity": diff_reading(a - b)}
            fr["slopes"] = sl
            # animacy split (nouns): side slopes on animate / inanimate nouns only
            for anim in ("animate", "inanimate"):
                wn_a = wn * (an == anim)[None, :]
                vma, _, _ = verb_means(R, P, frame, wn_a, nouns)
                aO_I, aO_T = slope(vma["O"][1], zi, ntI, wv), slope(vma["O"][0], zt, ntT, wv)
                aD_T, aD_I = (slope(vma["DOT"][0][:, el], zt[el], ntT[el], wv[:, el]),
                              slope(vma["DOT"][1][:, el], zi[el], ntI[el], wv[:, el]))
                fr[f"slopes_{anim}"] = {
                    "O": {"slope_I": [aO_I[0], *ci(aO_I)], "slope_T": [aO_T[0], *ci(aO_T)], "diff": [(aO_I - aO_T)[0], *ci(aO_I - aO_T)],
                          "side_specificity": diff_reading(aO_I - aO_T)},
                    "DOT": {"slope_T": [aD_T[0], *ci(aD_T)], "slope_I": [aD_I[0], *ci(aD_I)], "diff": [(aD_T - aD_I)[0], *ci(aD_T - aD_I)],
                            "side_specificity": diff_reading(aD_T - aD_I)}}
            if model == "pythia-1.4b":
                verbs_out.append(pd.DataFrame({"frame": frame, "verb": list(P.t_verb) + list(P.i_verb),
                                               "role": ["T"] * len(P) + ["I"] * len(P), "band": list(P.band) * 2,
                                               "logP_O": np.concatenate([Q["O_T"][0], Q["O_I"][0]]),
                                               "logP_DOT": np.concatenate([Q["D_T"][0], Q["D_I"][0]]),
                                               "logP_PREP": np.concatenate([Q["P_T"][0], Q["P_I"][0]])}))
            if model == "pythia-1.4b" and frame == "past":
                lines += ["## Pythia-1.4B, simple past (primary)", "",
                          "| measure | pairs | Head | Tail | XTail | Tail − XTail (token-std.) | reading | side | Head − XTail | reading | side |",
                          "|---|---:|---|---|---|---|---|---|---|---|---|"]
                for meas, rec in fr.items():
                    if not meas.startswith("M_"):
                        continue
                    lines.append(f"| {meas} | {rec['n_pairs']} | " + " | ".join(f"{rec['band'][b]:+.2f}" for b in BANDS) +
                                 f" | {rec['tok'][0]:+.2f} [{rec['tok'][1]:+.2f}, {rec['tok'][2]:+.2f}] | {rec['tok_reading']} | "
                                 f"{rec['tok_side']} | {rec['head_xtail'][0]:+.2f} [{rec['head_xtail'][1]:+.2f}, {rec['head_xtail'][2]:+.2f}] | "
                                 f"{rec['head_xtail_reading']} | {rec['head_xtail_side']} |")
                lines += ["", "Side parts (positive = contributes to the deficit): " + "; ".join(
                    f"{m} {nm}: T {fr[m][nm + '_T_part'][0]:+.2f} [{fr[m][nm + '_T_part'][1]:+.2f}, {fr[m][nm + '_T_part'][2]:+.2f}], "
                    f"I {fr[m][nm + '_I_part'][0]:+.2f} [{fr[m][nm + '_I_part'][1]:+.2f}, {fr[m][nm + '_I_part'][2]:+.2f}]"
                    for m in ("M_obj", "M_end") for nm in ("tok", "head_xtail")) + ".", "",
                          "Per-verb slopes on Zipf (nats per Zipf unit; frame-specific log token count as covariate) and side "
                          "specificity (predicted-side slope minus the other class's slope, same pairs; predicted < 0):", "",
                          "| readout | predicted side | predicted-side slope | reading | other-side slope | reading | difference | side specificity | categorical tokens | multi-token Tail/XTail only |",
                          "|---|---|---|---|---|---|---|---|---|---|"]
                for lab, s in sl.items():
                    lines.append(f"| {lab} | {s['predicted_side']} | {s['slope_pred'][0]:+.3f} [{s['slope_pred'][1]:+.3f}, {s['slope_pred'][2]:+.3f}] | "
                                 f"**{s['reading_pred']}** | {s['slope_other'][0]:+.3f} [{s['slope_other'][1]:+.3f}, {s['slope_other'][2]:+.3f}] | "
                                 f"{s['reading_other']} | {s['diff'][0]:+.3f} [{s['diff'][1]:+.3f}, {s['diff'][2]:+.3f}] | **{s['side_specificity']}** | "
                                 f"{s['categorical_tokens']['side_specificity']} | {s['common_support_multi']['side_specificity']} |")
                lines += ["", "Animacy split (nouns): " + "; ".join(
                    f"{a}: O slope I {fr[f'slopes_{a}']['O']['slope_I'][0]:+.3f} [{fr[f'slopes_{a}']['O']['slope_I'][1]:+.3f}, "
                    f"{fr[f'slopes_{a}']['O']['slope_I'][2]:+.3f}], T {fr[f'slopes_{a}']['O']['slope_T'][0]:+.3f}, "
                    f"{fr[f'slopes_{a}']['O']['side_specificity']}; '.' slope T {fr[f'slopes_{a}']['DOT']['slope_T'][0]:+.3f} "
                    f"[{fr[f'slopes_{a}']['DOT']['slope_T'][1]:+.3f}, {fr[f'slopes_{a}']['DOT']['slope_T'][2]:+.3f}], I "
                    f"{fr[f'slopes_{a}']['DOT']['slope_I'][0]:+.3f}, {fr[f'slopes_{a}']['DOT']['side_specificity']}"
                    for a in ("animate", "inanimate")) + ".",
                          "", "log P(PREP) by band (T / I): " + "; ".join(
                    f"{NAME[b]} {pmean(wv, Q['P_T'], (P.band == b).to_numpy())[0]:+.2f} / {pmean(wv, Q['P_I'], (P.band == b).to_numpy())[0]:+.2f}"
                    for b in BANDS) + ".", ""]
        p = dm["past"]
        summ.append(f"| {model} | {p['M_obj']['head_xtail'][0]:+.2f} ({p['M_obj']['head_xtail_reading']}; {p['M_obj']['head_xtail_side']}) | "
                    f"{p['M_obj']['tok'][0]:+.2f} ({p['M_obj']['tok_reading']}) | {p['M_end']['head_xtail'][0]:+.2f} "
                    f"({p['M_end']['head_xtail_reading']}; {p['M_end']['head_xtail_side']}) | {p['M_end']['tok'][0]:+.2f} ({p['M_end']['tok_reading']}) | "
                    f"{p['slopes']['O']['side_specificity']} | {p['slopes']['DOT']['side_specificity']} |")
    lines += ["## All models (simple past)", "",
              "| model | ΔM_obj Head − XTail | ΔM_obj token-std. | ΔM_end Head − XTail | ΔM_end token-std. | O side specificity | \".\" side specificity |",
              "|---|---|---|---|---|---|---|"] + summ + [""]
    if "pythia-1.4b" in dec:
        pf = dec["pythia-1.4b"]["perf"]
        lines += ["Perfect frame (\"The N has v\"), 1.4B: " + "; ".join(
            f"{m} Head − XTail {pf[m]['head_xtail'][0]:+.2f} ({pf[m]['head_xtail_reading']}, {pf[m]['head_xtail_side']}), token-std. "
            f"{pf[m]['tok'][0]:+.2f} ({pf[m]['tok_reading']})" for m in ("M_obj", "M_end"))
                  + "; side specificity " + ", ".join(f"{k} {v['side_specificity']}" for k, v in pf["slopes"].items()) + ".", ""]
    V = pd.concat(verbs_out) if verbs_out else None
    # counts (1.4B, past)
    cpath = Path(args.counts)
    if V is not None and cpath.exists():
        C = pd.read_csv(cpath, keep_default_na=False)
        C["count"] = C["count"].astype(float)
        Wc = C.pivot(index="verb", columns="slot", values="count")
        ctot = Wc.overall + Wc.overall_cap + Wc.nl_low + Wc.nl_cap
        cobj = Wc[[c for c in Wc.columns if c.startswith("obj_")]].sum(1)
        cend = Wc[[c for c in Wc.columns if c.startswith("end_")]].sum(1)
        lg = lambda s, vs: np.log10(s.loc[vs].to_numpy() + 0.5)
        rng = np.random.default_rng(args.seed)
        wn = weights(rng, an, args.n_boot)
        wv = weights(rng, P.band.to_numpy(), args.n_boot)
        R = Pre.merge(pd.read_parquet(Path(args.res_dir) / "pythia-1.4b" / "next_sets.parquet"), on="id")
        vm, ntT, ntI = verb_means(R, P, "past", wn, nouns)
        folds = fold_map(P.band.to_numpy(), args.seed)
        lines += ["## Against counts (1.4B, simple past; A1's c_total and the B3 frame counts; log token count in every model)", ""]
        dec["counts"] = {}
        for lab, Y, vs, frame_c, mask, nt in (("log P(O | I verb)", vm["O"][1], P.i_verb, cobj, allp, ntI),
                                              ("log P(. | T verb), eligible pairs", vm["DOT"][0], P.t_verb, cend, el, ntT)):
            Cx, Fx, Nx = lg(ctot, vs)[mask], lg(frame_c, vs)[mask], np.log(nt[mask])
            Yd, w, fo = Y[:, mask], wv[:, mask], folds[mask]
            one = np.ones(mask.sum())
            X0, X1, X2 = (np.column_stack([one, Nx]), np.column_stack([one, Nx, Cx]), np.column_stack([one, Nx, Cx, Fx]))
            mse = lambda X: np.array([cv_mse(X, Yd[b], w[b], fo) for b in range(len(Yd))])
            m0, m1, m2 = mse(X0), mse(X1), mse(X2)
            d01, dl01 = decide(m0, m1)
            d12, dl12 = decide(m1, m2)
            c1 = np.array([wls(X1, Yd[b], w[b])[2] for b in range(len(Yd))])
            c2 = np.array([wls(X2, Yd[b], w[b])[3] for b in range(len(Yd))])
            zero = pd.Series(frame_c.loc[vs].to_numpy()[mask] == 0).groupby(P.band.to_numpy()[mask]).mean()
            dec["counts"][lab] = {"null_to_C": d01, "C_coef": [c1[0], *ci(c1)], "C_to_C_plus_frame": d12,
                                  "frame_coef": [c2[0], *ci(c2)], "zero_share_frame_count": zero.to_dict()}
            lines += [f"- {lab}: tokens → + C {fmt(dl01, 3)} → **{d01}**, C coefficient {fmt(c1, 3)}; + log frame count "
                      f"{fmt(dl12, 3)} → **{d12}**, frame-count coefficient {fmt(c2, 3)}; share with frame count 0 by band "
                      + ", ".join(f"{NAME[b]} {v:.2f}" for b, v in zero.items()) + "."
                      + (" (The frame count \"P p the\" is itself an object-onset proxy: \"the\" can begin a temporal NP.)"
                         if "O |" in lab else " (Frame count: Σ count(\"P p.\").)")]
        lines.append("")
    else:
        lines += ["## Against counts", "", "A1's counts not available yet (rerun when they are).", ""]
    # descriptive association with released accuracy (1.4B LP)
    if V is not None:
        lem = pd.read_csv(args.d11_pairs, keep_default_na=False)
        form2 = {}
        for role in ("t", "i"):
            for l, p in zip(lem[f"{role}_lemma"], lem[f"{role}_participle"]):
                for fx in forms(l, p).values():
                    form2.setdefault((fx, role.upper()), p)
        Vp = V[V.frame == "past"].set_index("verb")
        zipf = pd.concat([P.set_index("t_verb").t_zipf, P.set_index("i_verb").i_zipf])
        ntk = pd.read_csv(Path(args.d11) / "verbs.csv", keep_default_na=False).set_index("verb").n_tok
        F = pd.read_parquet(args.lp_scores)
        F = F[F.cond == "unedited"].merge(pd.read_parquet(args.lp_items), on="row", validate="one_to_one")
        rng = np.random.default_rng(args.seed)
        lines += ["## Released FreqBLiMP (Pythia-1.4B LP): accuracy and a descriptive item-level association", "",
                  "Item margin (good − bad LP) on the bad verb's side measure, with band fixed effects and the bad verb's Zipf and "
                  "log token count as controls; verbs resampled within band. Analysed only with ≥ 3 mapped verb types in every band "
                  "and ≥ 15 in total; otherwise descriptive. This is an association, not a test of the mechanism (lexical, "
                  "inflectional and good-verb differences are not controlled).", "",
                  "| paradigm | accuracy Head / Tail / XTail (%, 95% CI) | bad verb | mapped items | verb types H / T / X | coefficient (95% CI) | association |",
                  "|---|---|---|---|---|---|---|"]
        dec["link"] = {}
        for par, (role, meas) in PARADIGMS.items():
            g = F[F.uid == par].copy()
            accs = []
            for b in BANDS:
                c = (g[g.band == b].good_lp > g[g.band == b].bad_lp).to_numpy(float)
                wb = weights(rng, np.zeros(len(c)), args.n_boot)
                a = 100 * (wb @ c) / wb.sum(1)
                accs.append(f"{a[0]:.1f} [{ci(a)[0]:.1f}, {ci(a)[1]:.1f}]")
            items = []
            for band in BANDS:
                for line in open(Path(args.freqblimp) / band / f"{par}.jsonl"):
                    r = json.loads(line)
                    gw, bw = r["sentence_good"].rstrip(".").split(), r["sentence_bad"].rstrip(".").split()
                    d = [i for i, (x, y) in enumerate(zip(gw, bw)) if x != y] if len(gw) == len(bw) else []
                    v = form2.get((bw[d[0]].lower(), role)) if len(d) == 1 else None
                    items.append({"band": band, "pairID": str(r["pairID"]), "verb": v})
            it = pd.DataFrame(items)
            g = g.assign(pairID=g.pairID.astype(str)).merge(it, on=["band", "pairID"], validate="one_to_one")
            g = g[g.verb.notna()]
            types = g.groupby("band").verb.nunique().reindex(BANDS, fill_value=0)
            ok = bool((types >= 3).all() and g.verb.nunique() >= 15)
            rec = {"n_items": len(g), "types": types.to_dict(), "analysed": ok}
            coef_txt, assoc = "—", "descriptive only (too few verb types)"
            if ok:
                col = "logP_O" if meas == "O" else "logP_DOT"
                xv = Vp.loc[g.verb, col].to_numpy()
                D = np.column_stack([np.ones(len(g)), xv, g.band == "tail", g.band == "xtail", zipf.loc[g.verb].to_numpy(float),
                                     np.log(ntk.loc[g.verb].to_numpy(float))]).astype(float)
                vlist = sorted(set(g.verb))
                vix = pd.Index(vlist).get_indexer(g.verb)
                vband = g.groupby("verb").band.first().loc[vlist].to_numpy()
                wvb = weights(rng, vband, args.n_boot)
                rank = np.linalg.matrix_rank(D)
                co = np.array([wls(D, g.margin.to_numpy() if "margin" in g else (g.good_lp - g.bad_lp).to_numpy(), wvb[b, vix])[1]
                               for b in range(args.n_boot + 1)])
                rec.update({"coef": [co[0], *ci(co)], "rank_full": bool(rank == D.shape[1])})
                coef_txt = fmt(co, 3) + ("" if rank == D.shape[1] else " (rank-deficient)")
                assoc = "negative" if ci(co)[1] < 0 else "positive" if ci(co)[0] > 0 else "unresolved"
                rec["association"] = assoc
            dec["link"][par] = rec
            lines.append(f"| {par} | {' / '.join(accs)} | {role} | {len(g)} | {' / '.join(str(types[b]) for b in BANDS)} | "
                         f"{coef_txt} | **{assoc}** |")
        lines.append("")
    out = Path(args.res_dir)
    out.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(out / "b3_summary.csv", index=False)
    (out / "b3_decisions.json").write_text(json.dumps(dec, indent=1, default=float))
    if V is not None:
        V.to_csv(out / "b3_verbs.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/round6/alternation")
    ap.add_argument("--res-dir", default="results/round6/alternation")
    ap.add_argument("--d11", default="data/round6/frame_d11")
    ap.add_argument("--d11-pairs", default="data/round5/verbs_expanded/pairs.csv")
    ap.add_argument("--i-frames", default="data/round6/preempt/i_frames.csv")
    ap.add_argument("--counts", default="data/round6/counts/pile_counts.csv")
    ap.add_argument("--freqblimp", default="/Users/tyronewhite/masters_research_code/freq-blimp/data/freqblimp")
    ap.add_argument("--lp-items", default="results/round5/restore/restore_freqblimp_items.parquet")
    ap.add_argument("--lp-scores", default="results/round5/restore/restore_freqblimp.parquet")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--report", default="reports/round6/b3_alternations.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    main(ap.parse_args())
