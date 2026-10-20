#!/usr/bin/env python3
"""Round 6, A2 analysis: the frame contrast on the D11 pairs across the Pythia suite (reports/round6/plan.md, A2).
No model; numpy/pandas only.

Inputs: data/round6/frame_d11/{prefixes,verbs,pairs}.csv; per model results/round6/frame_d11/<model>/{frame_scores.parquet,
frame_h.npz, frame_meta.json}; round 5's scores for the reproduction check (1.4B: results/round5/frame; other sizes:
results/round5/frame_suite/<model>).
Per model and prefix set (freq primary): band means of L, Delta_pass(T), Delta_pass(I), M_has, L1, L_rest; readings as
round 5's A1 (bound 0.25 nats); the primary token-standardized contrast
  Delta L_tok = sum_s w_s (L_Tail,s - L_XTail,s), s in the exact (T tokens x I tokens) cells 2x2, 2x3, 3x2, 3x3,
  w_s proportional to the pairs in s (both bands),
its side contributions and side reading; sensitivities (OLS L ~ band + T tokens + I tokens over the multi-multi
Tail/XTail pairs; the 2x2 cell alone); held-out and trainable pairs separately; per-verb a_V (Head AUC, singleton D_L).
Bootstrap: pairs within band x split (token-standardized contrast: within band x token cell x split) x contexts within
animacy strata (freq / been), context band (cur) or none (they); 2,000 draws, seed 17; one context draw per set and
model, shared by all its outcomes.

Outputs: reports/round6/a2_suite_d11.md, results/round6/frame_d11/a2_summary.csv, a2_decisions.json, a2_verbs.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_frame_contrast import BANDS, NAME, SETS, Boot, ci, classify, ctx_strata, fmt, pair_tables, reading, weights

MODELS = ("pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b", "pythia-2.8b", "pythia-6.9b", "pythia-12b")
CELLS = ("2×2", "2×3", "3×2", "3×3")
W4 = {"2×2": 85, "2×3": 36, "3×2": 35, "3×3": 22}  # declared fixed weights (pairs per cell, both bands)
W3 = {"2×2": 85, "2×3": 36, "3×2": 35}  # three-cell estimand (held-out Tail has no 3x3 pair)
KEYS = ("L", "dT", "dI", "Mhas", "L1", "Lrest")
LAB = {"L": "L", "dT": "Δ_pass(T)", "dI": "Δ_pass(I)", "Mhas": "M_has", "L1": "L₁", "Lrest": "L_rest"}


def boots_for(pairs, pre, st, ctx, n_boot, rng):
    """Band-level and token-cell-level draws sharing one context draw."""
    band, split, cell = pairs.band.to_numpy(), pairs.split.to_numpy(), pairs.cell.to_numpy()
    Bb = Boot(np.array([f"{b}|{s}" for b, s in zip(band, split)]), ctx_strata(pre, st, ctx), n_boot, rng)
    Bb.strata = band  # masks below use the band; the draws were made within band x split
    Bc = object.__new__(Boot)
    Bc.strata, Bc.wc = band, Bb.wc
    Bc.wv = weights(rng, np.array([f"{b}|{c}|{s}" for b, c, s in zip(band, cell, split)]), n_boot)
    return Bb, Bc


def tok_contrast(B, X, pairs, subset, W=W4):
    """Tail - XTail contrast standardized over the cells of W with fixed weights; every cell needs pairs in both bands."""
    band, cell = pairs.band.to_numpy(), pairs.cell.to_numpy()
    num, den, used = 0.0, 0.0, []
    for s, w in W.items():
        mt, mx = subset & (band == "tail") & (cell == s), subset & (band == "xtail") & (cell == s)
        assert mt.sum() >= 1 and mx.sum() >= 1, f"cell {s} has no pair in a band of this subset"
        num = num + w * (B.mean(X, mt) - B.mean(X, mx))
        den += w
        used.append(f"{s} ({int(mt.sum())}/{int(mx.sum())})")
    return num / den, used


def ols_tail(B, X, pairs, subset):
    """Coefficient of Tail in L ~ 1 + Tail + T tokens + I tokens, weighted by the pair draws (multi-multi pairs)."""
    m = subset & pairs.band.isin(["tail", "xtail"]).to_numpy() & (pairs.t_ntok >= 2).to_numpy() & (pairs.i_ntok >= 2).to_numpy()
    u = B.unit_means(X)[:, m]
    D = np.column_stack([np.ones(m.sum()), (pairs.band.to_numpy()[m] == "tail"), pairs.t_ntok.to_numpy()[m],
                         pairs.i_ntok.to_numpy()[m]]).astype(float)
    out = np.empty(len(u))
    for b in range(len(u)):
        sw = np.sqrt(B.wv[b, m])
        out[b] = np.linalg.lstsq(D * sw[:, None], u[b] * sw, rcond=None)[0][1]
    return out


def side_reading(dT, dI, total):
    """Only for a deficit: intransitive-side = -dDelta_pass(I) CI above 0 and above dDelta_pass(T) (paired);
    transitive-side = mirror; otherwise mixed. Without a deficit the contributions are descriptive."""
    if classify(total) != "deficit":
        return "no deficit (contributions descriptive)"
    i_part, t_part = -dI, dT
    if ci(i_part)[0] > 0 and ci(i_part - t_part)[0] > 0:
        return "intransitive-side"
    if ci(t_part)[0] > 0 and ci(t_part - i_part)[0] > 0:
        return "transitive-side"
    return "mixed / unresolved"


def reproduce(model, S, args):
    ref = Path(args.r5_frame) if model == "pythia-1.4b" else Path(args.r5_suite) / model
    if not (ref / "frame_scores.parquet").exists():
        return 0, float("nan")
    R = pd.read_parquet(ref / "frame_scores.parquet")
    m = S.merge(R, on=["set", "context_id", "frame", "verb"], suffixes=("", "_r5"))
    return len(m), float((m.lp - m.lp_r5).abs().max()) if len(m) else float("nan")


def affinity(model_dir, pre, V, Bb, rng, n_boot):
    z = np.load(model_dir / "frame_h.npz")
    keys = list(z["keys"])
    H, U = z["H"].astype(np.float64), z["U"].astype(np.float64)
    urow = {int(t): i for i, t in enumerate(z["U_ids"])}
    fctx = list(pre[pre.set == "freq"].context_id)
    dh = np.stack([H[keys.index(f"freq|{c}|pass")] - H[keys.index(f"freq|{c}|act")] for c in fctx])
    dbar = (Bb.wc @ dh) / Bb.wc.sum(1, keepdims=True)
    A = dbar @ np.stack([U[urow[int(t)]] for t in V.first_id]).T  # [draw, verb]
    s1 = (V.n_tok == 1).to_numpy()
    hv = s1 & (V.band == "head").to_numpy()
    at, ai = A[0, hv & (V.role == "T").to_numpy()], A[0, hv & (V.role == "I").to_numpy()]
    auc = float(np.mean(at[:, None] > ai[None, :]) + 0.5 * np.mean(at[:, None] == ai[None, :]))
    idx = np.flatnonzero(s1 & V.band.isin(["head", "tail"]).to_numpy())
    grp = np.array([f"{r}_{b}" for r, b in zip(V.role.to_numpy()[idx], V.band.to_numpy()[idx])])
    w = weights(rng, grp, n_boot)
    gm = lambda g: (w[:, grp == g] * A[:, idx][:, grp == g]).sum(1) / w[:, grp == g].sum(1)
    DT, DI = gm("T_head") - gm("T_tail"), gm("I_head") - gm("I_tail")
    n = {g: int((grp == g).sum()) for g in ("T_head", "T_tail", "I_head", "I_tail")}
    return A[0], auc, DT, DI, n


def main(args):
    rng = np.random.default_rng(args.seed)
    data = Path(args.data_dir)
    pre = pd.read_csv(data / "prefixes.csv", keep_default_na=False)
    V = pd.read_csv(data / "verbs.csv", keep_default_na=False)
    pairs = pd.read_csv(data / "pairs.csv", keep_default_na=False)
    pairs["cell"] = [f"{a}×{b}" for a, b in zip(pairs.t_ntok, pairs.i_ntok)]
    held, train = (pairs.split == "heldout").to_numpy(), (pairs.split == "trainable").to_numpy()
    tx = pairs[pairs.band.isin(["tail", "xtail"])].cell.value_counts()
    assert {c: int(tx.get(c, 0)) for c in W4} == W4, f"cell sizes differ from the declared weights: {tx.to_dict()}"
    allp = np.ones(len(pairs), bool)
    rows, dec, summ, verbs_out = [], {}, [], []
    lines = ["# Round 6, A2: the D11 frame contrast across the Pythia suite", "",
             "Spec: `plan.md`, A2. Run: `run_frame_contrast.py --cur-all` on the 562 D11 participles; analysis: "
             "`analyze_frame_d11.py`. Δ_pass(V) = log P(V | was-prefix) − log P(V | has-prefix); L = Δ_pass(T) − "
             "Δ_pass(I). 281 pairs (Head 47, Tail 132, XTail 102; 62 held out). Primary: ΔL_tok = token-cell-weighted "
             "Tail − XTail over the exact cells 2×2, 2×3, 3×2, 3×3 (fixed weights 85 / 36 / 35 / 22). Rule (bound 0.25 nats): "
             "deficit = 95% CI above 0 and estimate ≥ 0.25; surplus = mirror; negligible = 90% CI within ±0.25; "
             "otherwise unresolved. Side (only given a deficit): intransitive-side = −ΔΔ_pass(I) CI above 0 and above ΔΔ_pass(T) "
             "(paired); dominance, not exclusivity. Held-out pairs: DAS-unexposed subgroup (most verbs scored in round 5); "
             "three-cell estimand (2×2, 2×3, 3×2). Draws are identical across models. "
             "95% CIs: pairs within band × split (token contrast: band × cell × split) × contexts.", ""]
    detail = []
    for model in args.models:
        mdir = Path(args.res_dir) / model
        if not (mdir / "frame_scores.parquet").exists():
            lines.append(f"- {model}: no output (not run or failed).")
            continue
        rng = np.random.default_rng(args.seed)  # identical pair and context draws for every model (paired across models)
        S = pd.read_parquet(mdir / "frame_scores.parquet")
        meta = json.loads((mdir / "frame_meta.json").read_text())
        n_rep, rep_err = reproduce(model, S, args)
        tol = 1e-3 if model == "pythia-1.4b" else 1e-2
        assert n_rep == 0 or rep_err <= tol, f"{model}: round-5 reproduction error {rep_err}"
        tabs = pair_tables(S, pairs, pre)
        dm = dec[model] = {"checks": {**meta["checks"], "round5_rows_compared": n_rep, "round5_max_abs_diff": rep_err}}
        detail += [f"## {model}", "",
                   f"Checks: first-subtoken reconstruction {meta['checks']['max_first_subtoken_error']:.1e}, identity "
                   f"{meta['checks']['max_identity_error']:.1e} (tolerance {meta['checks'].get('tol', tol)}); max |logit| "
                   f"at the slot {meta['checks'].get('max_abs_logit', float('nan')):.1f}; round-5 reproduction "
                   f"({n_rep} rows in common) max |Δ lp| {rep_err:.1e}.", ""]
        for st in SETS:
            q, ctx = tabs[st]
            Bb, Bc = boots_for(pairs, pre, st, ctx, args.n_boot, rng)
            band = pairs.band.to_numpy()
            res = {}
            for sub_name, sub in (("all", allp), ("trainable", train), ("heldout", held)):
                x = {}
                for tag, W in (("", W4), ("3", W3)):
                    if sub_name == "heldout" and tag == "":
                        continue  # held-out Tail has no 3x3 pair: three-cell estimand only
                    x["L_tok" + tag], x["used" + tag] = tok_contrast(Bc, q["L"], pairs, sub, W)
                    x["dT_tok" + tag], _ = tok_contrast(Bc, q["dT"], pairs, sub, W)
                    x["dI_tok" + tag], _ = tok_contrast(Bc, q["dI"], pairs, sub, W)
                hb, xb, tb = sub & (band == "head"), sub & (band == "xtail"), sub & (band == "tail")
                x["dLhx"] = Bb.mean(q["L"], hb) - Bb.mean(q["L"], xb)
                x["dLht"] = Bb.mean(q["L"], hb) - Bb.mean(q["L"], tb)
                x["band_L"] = {b: Bb.mean(q["L"], sub & (band == b)) for b in BANDS if (sub & (band == b)).sum() >= 2}
                res[sub_name] = x
            r = res["all"]
            dHas = Bb.mean(q["Mhas"], band == "head") - Bb.mean(q["Mhas"], band == "xtail")
            dHas_tok, _ = tok_contrast(Bc, q["Mhas"], pairs, allp)
            sens_ols = ols_tail(Bc, q["L"], pairs, allp)
            m22t, m22x = (band == "tail") & (pairs.cell == "2×2").to_numpy(), (band == "xtail") & (pairs.cell == "2×2").to_numpy()
            sens_22 = Bc.mean(q["L"], m22t) - Bc.mean(q["L"], m22x)
            d = {"L_tok": [r["L_tok"][0], *ci(r["L_tok"])], "L_tok_reading": classify(r["L_tok"]),
                 "side_tok": side_reading(r["dT_tok"], r["dI_tok"], r["L_tok"]), "dT_tok": [r["dT_tok"][0], *ci(r["dT_tok"])],
                 "minus_dI_tok": [-r["dI_tok"][0], *ci(-r["dI_tok"])], "Mhas_tok": [dHas_tok[0], *ci(dHas_tok)],
                 "Mhas_tok_reading": classify(dHas_tok),
                 "reading_tok": reading(classify(r["L_tok"]), classify(dHas_tok)),
                 "dL_head_xtail": [r["dLhx"][0], *ci(r["dLhx"])], "dL_head_xtail_reading": classify(r["dLhx"]),
                 "dMhas_head_xtail": [dHas[0], *ci(dHas)], "reading_head_xtail": reading(classify(r["dLhx"]), classify(dHas)),
                 "dL_head_tail": [r["dLht"][0], *ci(r["dLht"])], "dL_head_tail_reading": classify(r["dLht"]),
                 "sens_ols": [sens_ols[0], *ci(sens_ols)], "sens_2x2": [sens_22[0], *ci(sens_22)], "cells": r["used"]}
            for sn in ("all", "trainable", "heldout"):
                x = res[sn]
                d["three_cell_" + sn] = {"L_tok3": [x["L_tok3"][0], *ci(x["L_tok3"])], "L_tok3_reading": classify(x["L_tok3"]),
                                         "side_tok3": side_reading(x["dT_tok3"], x["dI_tok3"], x["L_tok3"]), "cells": x["used3"],
                                         "dT_tok3": [x["dT_tok3"][0], *ci(x["dT_tok3"])],
                                         "minus_dI_tok3": [-x["dI_tok3"][0], *ci(-x["dI_tok3"])],
                                         "band_L": {b: [v[0], *ci(v)] for b, v in x["band_L"].items()}}
                if sn == "trainable":
                    d["trainable"] = {"L_tok": [x["L_tok"][0], *ci(x["L_tok"])], "L_tok_reading": classify(x["L_tok"]),
                                      "side_tok": side_reading(x["dT_tok"], x["dI_tok"], x["L_tok"])}
            dm[st] = d
            for k, v in d.items():
                if isinstance(v, list) and len(v) == 3 and all(isinstance(t, float) for t in v):
                    rows.append({"model": model, "set": st, "quantity": k, "est": v[0], "lo": v[1], "hi": v[2]})
            if st != "freq":
                continue
            # detailed freq tables
            detail += ["### freq (primary): band means", "",
                       "| Quantity | Head | Tail | XTail | Head − Tail | Head − XTail | Tail − XTail, token-standardized |",
                       "|---|---|---|---|---|---|---|"]
            for k in KEYS:
                m = {b: Bb.mean(q[k], band == b) for b in BANDS}
                tk, _ = tok_contrast(Bc, q[k], pairs, allp)
                detail.append(f"| {LAB[k]} | " + " | ".join(fmt(m[b]) for b in BANDS) +
                              f" | {fmt(m['head'] - m['tail'])} | {fmt(m['head'] - m['xtail'])} | {fmt(tk)} |")
                for b in BANDS:
                    rows.append({"model": model, "set": st, "quantity": f"{k}_{b}", "est": m[b][0], "lo": ci(m[b])[0], "hi": ci(m[b])[1]})
            detail += ["", "By token cell (Tail vs XTail; L, Δ_pass(T), Δ_pass(I)):", "",
                       "| Cell | Tail pairs | XTail pairs | L Tail | L XTail | Δ_pass(T) Tail | Δ_pass(T) XTail | Δ_pass(I) Tail | Δ_pass(I) XTail |",
                       "|---|---:|---:|---|---|---|---|---|---|"]
            for s in CELLS:
                mt, mx = (band == "tail") & (pairs.cell == s).to_numpy(), (band == "xtail") & (pairs.cell == s).to_numpy()
                detail.append(f"| {s} | {int(mt.sum())} | {int(mx.sum())} | " + " | ".join(
                    fmt(Bc.mean(q[k], mm)) for k in ("L", "dT", "dI") for mm in (mt, mx)) + " |")
            detail += ["", f"- **Primary ΔL_tok (Tail − XTail, token-standardized): {fmt(r['L_tok'])} → {d['L_tok_reading']}**; "
                       f"ΔΔ_pass(T) {fmt(r['dT_tok'])}, −ΔΔ_pass(I) {fmt(-r['dI_tok'])} → **{d['side_tok']}**. "
                       f"ΔM_has (token-standardized) {fmt(dHas_tok)} → {d['Mhas_tok_reading']}; reading {d['reading_tok']}.",
                       f"- Sensitivities: OLS (L ~ band + T tokens + I tokens, multi-multi) {fmt(sens_ols)}; 2×2 only "
                       f"{fmt(sens_22)}.",
                       f"- Head − XTail (token-confounded): ΔL {fmt(r['dLhx'])} → {d['dL_head_xtail_reading']}; ΔM_has "
                       f"{fmt(dHas)}; reading {d['reading_head_xtail']}. Head − Tail: ΔL {fmt(r['dLht'])} → "
                       f"{d['dL_head_tail_reading']}.",
                       "- Three-cell contrast (2×2, 2×3, 3×2; fixed weights 85/36/35): "
                       + "; ".join(f"{sn} {fmt(res[sn]['L_tok3'])} → {d['three_cell_' + sn]['L_tok3_reading']} "
                                   f"({d['three_cell_' + sn]['side_tok3']}; cells {', '.join(res[sn]['used3'])})"
                                   for sn in ("all", "trainable", "heldout")) + ".",
                       "- DAS-unexposed (held-out) band L: " + ", ".join(f"{NAME[b]} {fmt(v)}" for b, v in res["heldout"]["band_L"].items())
                       + f". Trainable pairs, four-cell ΔL_tok: {fmt(res['trainable']['L_tok'])} → {d['trainable']['L_tok_reading']}.", ""]
            a, auc, DT, DI, n = affinity(mdir, pre, V, Bb, rng, args.n_boot)
            DL = DT - DI
            dm["affinity"] = {"auc_head": auc, "D_L": [DL[0], *ci(DL)], "D_T": [DT[0], *ci(DT)], "D_I": [DI[0], *ci(DI)], "n": n}
            detail += [f"- Per verb (single-token): Head AUC(a_V, T vs I) {auc:.2f}; D_L = (ā_T,Head − ā_I,Head) − "
                       f"(ā_T,Tail − ā_I,Tail) {fmt(DL)} (D_T {fmt(DT)}, D_I {fmt(DI)}; n {n}).", ""]
            verbs_out.append(pd.DataFrame({"model": model, "verb": V.verb, "role": V.role, "band": V.band,
                                           "n_tok": V.n_tok, "a": a}))
        detail += ["Other prefix sets (ΔL_tok; reading; side):", "", "| set | ΔL_tok | reading | side | ΔL Head − XTail |",
                   "|---|---|---|---|---|"]
        for st in SETS:
            x = dm[st]
            detail.append(f"| {st} | {x['L_tok'][0]:+.2f} [{x['L_tok'][1]:+.2f}, {x['L_tok'][2]:+.2f}] | {x['L_tok_reading']} | "
                          f"{x['side_tok']} | {x['dL_head_xtail'][0]:+.2f} [{x['dL_head_xtail'][1]:+.2f}, {x['dL_head_xtail'][2]:+.2f}] |")
        detail.append("")
        f = dm["freq"]
        summ.append(f"| {model} | {f['L_tok'][0]:+.2f} [{f['L_tok'][1]:+.2f}, {f['L_tok'][2]:+.2f}] | **{f['L_tok_reading']}** | "
                    f"{f['dT_tok'][0]:+.2f} / {f['minus_dI_tok'][0]:+.2f} | {f['side_tok']} | "
                    f"{f['three_cell_heldout']['L_tok3'][0]:+.2f} [{f['three_cell_heldout']['L_tok3'][1]:+.2f}, "
                    f"{f['three_cell_heldout']['L_tok3'][2]:+.2f}] ({f['three_cell_heldout']['L_tok3_reading']}) | "
                    f"{f['dL_head_xtail'][0]:+.2f} [{f['dL_head_xtail'][1]:+.2f}, {f['dL_head_xtail'][2]:+.2f}] | "
                    f"{dm['affinity']['auc_head']:.2f} | {dm['affinity']['D_L'][0]:+.2f} [{dm['affinity']['D_L'][1]:+.2f}, "
                    f"{dm['affinity']['D_L'][2]:+.2f}] |")
    lines += ["## Summary (freq prefixes)", "",
              "| Model | ΔL_tok (Tail − XTail, token-standardized) | reading | ΔΔ_pass(T) / −ΔΔ_pass(I) | side | held-out three-cell ΔL | "
              "ΔL Head − XTail | Head AUC | singleton D_L |", "|---|---|---|---|---|---|---|---|---|"] + summ + [""] + detail
    out = Path(args.res_dir)
    Path(args.report).write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(out / "a2_summary.csv", index=False)
    (out / "a2_decisions.json").write_text(json.dumps(dec, indent=1, default=float))
    if verbs_out:
        pd.concat(verbs_out).to_csv(out / "a2_verbs.csv", index=False)
    print("\n".join(lines[:len(summ) + 12]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/round6/frame_d11")
    ap.add_argument("--res-dir", default="results/round6/frame_d11")
    ap.add_argument("--r5-frame", default="results/round5/frame")
    ap.add_argument("--r5-suite", default="results/round5/frame_suite")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--report", default="reports/round6/a2_suite_d11.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    main(ap.parse_args())
