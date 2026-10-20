#!/usr/bin/env python3
"""Round 6, A6 analysis: the D11 frame contrast over Pythia checkpoints, and exposure equivalence (reports/round6/plan.md,
A6). No model; needs scipy.

Inputs: results/round6/checkpoints/pythia-<size>/step<N>/{frame_scores.parquet, frame_h.npz, frame_meta.json};
data/round6/frame_d11_freq/{prefixes,verbs,pairs}.csv; A1's counts (data/round6/counts/pile_counts.csv); the
proportionality slices (data/round6/checkpoints/slice_counts.csv).
Per checkpoint (freq prefixes): band means of L, Delta_pass(T), Delta_pass(I); token-standardized Tail - XTail and
Head - XTail L contrasts with side parts and readings; Head AUC of a_V and singleton D_L. Exposure proxy
E = (c_total + 0.5) x step / 143000 (final-scaled). Predictive curve collapse on D11 I verbs x steps: M_E = f(log E),
M_Cg = q(log c) + step effects, M_2D = h(log c, log t) (natural-spline main effects and their tensor product), M_Eg =
M_E + step effects (diagnostic), M_g = step effects (baseline); log token count in every model; natural cubic splines
(ESL 5.4-5.5) with boundary knots at the 2.5 / 97.5 and interior knots at the 33.3 / 66.7 percentiles of the predictor
over all units (no outcome used). CV with A1's frozen fold map (by pair); bootstrap pairs within band x nouns within
animacy, 2,000 draws, identical across checkpoints. Stationarity diagnostic: quasi-Poisson log-linear trend of the slice
counts on log step (offset log slice length), pooled over Head and over Tail single-token participles.
Outputs: reports/round6/a6_checkpoints.md, results/round6/checkpoints/a6_summary.csv, a6_decisions.json, a6_cells.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_exposure import cv_mse, decide, fold_map
from analyze_frame_contrast import ci, classify, fmt, weights

W4 = {"2×2": 85, "2×3": 36, "3×2": 35, "3×3": 22}
BANDS = ("head", "tail", "xtail")
TOTAL = 143000


def ns_basis(x, knots):
    """Natural cubic spline basis (ESL 5.4-5.5) without the intercept: x and K - 2 nonlinear terms."""
    K = len(knots)
    d = lambda k: (np.maximum(x - knots[k], 0) ** 3 - np.maximum(x - knots[-1], 0) ** 3) / (knots[-1] - knots[k])
    return np.column_stack([x] + [d(k) - d(K - 2) for k in range(K - 2)])


def load_step(sdir, pre, P, wn):
    """[draw, pair] noun-weighted Delta_pass for T and I verbs; per-verb a_V point values."""
    S = pd.read_parquet(sdir / "frame_scores.parquet")
    ctx = list(pre.context_id)
    s = S[S.set == "freq"]
    lp = s.pivot_table(index=["verb", "frame"], columns="context_id", values="lp", aggfunc="first")[ctx]
    assert lp.notna().all().all() and len(lp) == 2 * (len(P) * 2), "missing scores"
    D = lp.xs("pass", level="frame") - lp.xs("act", level="frame")
    T = (wn @ D.loc[P.t_verb].to_numpy().T) / wn.sum(1, keepdims=True)
    I = (wn @ D.loc[P.i_verb].to_numpy().T) / wn.sum(1, keepdims=True)
    return T, I


def affinity(sdir, pre, V, wn, rng, n_boot):
    z = np.load(sdir / "frame_h.npz")
    keys = list(z["keys"])
    H, U = z["H"].astype(np.float64), z["U"].astype(np.float64)
    urow = {int(t): i for i, t in enumerate(z["U_ids"])}
    dh = np.stack([H[keys.index(f"freq|{c}|pass")] - H[keys.index(f"freq|{c}|act")] for c in pre.context_id])
    dbar = (wn @ dh) / wn.sum(1, keepdims=True)
    A = dbar @ np.stack([U[urow[int(t)]] for t in V.first_id]).T
    s1 = (V.n_tok == 1).to_numpy()
    hv = s1 & (V.band == "head").to_numpy()
    at, ai = A[0, hv & (V.role == "T").to_numpy()], A[0, hv & (V.role == "I").to_numpy()]
    auc = float(np.mean(at[:, None] > ai[None, :]) + 0.5 * np.mean(at[:, None] == ai[None, :]))
    idx = np.flatnonzero(s1 & V.band.isin(["head", "tail"]).to_numpy())
    grp = np.array([f"{r}_{b}" for r, b in zip(V.role.to_numpy()[idx], V.band.to_numpy()[idx])])
    w = weights(rng, grp, n_boot)
    gm = lambda g: (w[:, grp == g] * A[:, idx][:, grp == g]).sum(1) / w[:, grp == g].sum(1)
    DL = (gm("T_head") - gm("I_head")) - (gm("T_tail") - gm("I_tail"))
    return auc, DL


def main(args):
    data = Path(args.data_dir)
    pre = pd.read_csv(data / "prefixes.csv", keep_default_na=False)
    V = pd.read_csv(data / "verbs.csv", keep_default_na=False)
    P = pd.read_csv(data / "pairs.csv", keep_default_na=False)
    P["cell"] = [f"{a}×{b}" for a, b in zip(P.t_ntok, P.i_ntok)]
    band = P.band.to_numpy()
    lines = ["# Round 6, A6: the D11 frame contrast over Pythia checkpoints", "",
             "Spec: `plan.md`, A6. Runs: `run_frame_contrast.py --revision step<N>` (freq prefixes); analysis: "
             "`analyze_checkpoints.py`. Exposure E = c_total × step / 143000 (A1's c_total). 95% CIs: pairs within band "
             "(token contrast: band × cell) × nouns within animacy, 2,000 draws, identical across checkpoints. Readings as A2 "
             "(bound 0.25 nats).", ""]
    dec, rows, cells = {}, [], []
    # stationarity diagnostic (limited: lowercase single-token forms only)
    sc = Path(args.slices)
    if sc.exists():
        Sl = pd.read_csv(sc)

        def trend(g):
            """Quasi-Poisson fit of count ~ log step with offset log n_tokens (IRLS); returns the first-to-last rate ratio draws."""
            y, ls, off = g["count"].to_numpy(float), np.log(g.step.to_numpy(float)), np.log(g.n_tokens.to_numpy(float))
            X = np.column_stack([np.ones(len(y)), ls - ls.mean()])
            b = np.array([np.log(y.sum() / np.exp(off).sum()), 0.0])
            for _ in range(50):
                mu = np.exp(X @ b + off)
                b = b + np.linalg.solve(X.T @ (mu[:, None] * X), X.T @ (y - mu))
            mu = np.exp(X @ b + off)
            phi = max(1.0, float(((y - mu) ** 2 / mu).sum() / (len(y) - 2)))
            se = float(np.sqrt(phi * np.linalg.inv(X.T @ (mu[:, None] * X))[1, 1]))
            span = np.log(g.step.max() / g.step.min())
            rr = np.exp(np.array([b[1], b[1] - 1.96 * se, b[1] + 1.96 * se]) * span)
            return rr, phi

        lines += ["## Stationarity diagnostic (limited)", "",
                  f"{Sl.slice.nunique()} slices of {int(Sl.n_tokens.iloc[0]):,} stored tokens starting at steps "
                  f"{', '.join(str(x) for x in sorted(Sl.step.unique()))}; lowercase single-token D11 participles. Rate ratio "
                  "last-to-first slice from a quasi-Poisson trend on log step (95% CI):", "",
                  "| group | verbs | total count | rate ratio (95% CI) | dispersion |", "|---|---:|---:|---|---|"]
        dec["stationarity"] = {}
        for lab, mask in (("Head single-token", Sl.band == "head"), ("Tail single-token", Sl.band == "tail")):
            g = Sl[mask].groupby(["slice", "step"]).agg(count=("count", "sum"), n_tokens=("n_tokens", "first")).reset_index()
            rr, phi = trend(g)
            ok = bool(0.8 < rr[1] and rr[2] < 1.25)
            dec["stationarity"][lab] = {"rate_ratio": list(map(float, rr)), "dispersion": phi, "within_20pct": ok}
            lines.append(f"| {lab} | {Sl[mask].verb.nunique()} | {int(g['count'].sum()):,} | {rr[0]:.3f} [{rr[1]:.3f}, {rr[2]:.3f}] | {phi:.1f} |")
        top = Sl[Sl.band == "head"].groupby("verb")["count"].sum().nlargest(10).index
        for v in top:
            rr, phi = trend(Sl[Sl.verb == v])
            lines.append(f"| {v} | 1 | {int(Sl[Sl.verb == v]['count'].sum()):,} | {rr[0]:.3f} [{rr[1]:.3f}, {rr[2]:.3f}] | {phi:.1f} |")
        st_ok = all(v["within_20pct"] for v in dec["stationarity"].values())
        dec["stationarity_holds"] = st_ok
        lines += ["", f"**Stationary within ±20% (Head and Tail pooled): {st_ok}.** A diagnostic only: it does not cover "
                      "capitalized or unmarked forms, multi-token verbs, or the genre concentration of individual rare verbs.", ""]
    else:
        lines += ["## Stationarity diagnostic", "", "Slice counts not present.", ""]
    # exposure
    C = pd.read_csv(args.counts, keep_default_na=False)
    C["count"] = C["count"].astype(float)
    Wc = C.pivot(index="verb", columns="slot", values="count")
    ctot = Wc.overall + Wc.overall_cap + Wc.nl_low + Wc.nl_cap
    cmark = Wc.overall + Wc.overall_cap
    DECLARED = {"1.4b": [64, 128, 256, 512, 1000, 2000, 3000, 4000, 6000, 10000, 16000, 25000, 40000, 63000, 100000, 143000],
                "410m": [64, 128, 256, 512, 1000, 2000, 3000, 4000, 6000, 10000, 16000, 25000, 40000, 63000, 100000, 143000],
                "6.9b": [512, 2000, 8000, 23000, 63000, 143000]}
    split = P.split.to_numpy()
    for size in args.sizes:
        root = Path(args.res_dir) / f"pythia-{size}"
        steps = sorted(int(p.name[4:]) for p in root.glob("step*") if (p / "frame_scores.parquet").exists())
        missing = sorted(set(DECLARED[size]) - set(steps))
        dec.setdefault("missing_steps", {})[size] = missing
        if missing:
            lines += [f"**pythia-{size}: missing checkpoints {missing}** (results below use the available steps; "
                      + ("the primary analysis is incomplete)." if size == "1.4b" else "descriptive).") , ""]
        if not steps:
            continue
        rng = np.random.default_rng(args.seed)
        wn = weights(rng, pre.animacy.to_numpy(), args.n_boot)
        wv = weights(rng, np.array([f"{b}|{s_}" for b, s_ in zip(band, split)]), args.n_boot)  # as A2: band x split
        wc = weights(rng, np.array([f"{b}|{c}|{s_}" for b, c, s_ in zip(band, P.cell, split)]), args.n_boot)
        # curve collapse: pairs within band, as declared for A6's collapse (drawn after, so the trajectory draws are unchanged)
        wb = weights(rng, band, args.n_boot)
        pm = lambda w, Y, m: (w[:, m] * Y[:, m]).sum(1) / w[:, m].sum(1)

        def tok(Y):
            num = 0.0
            for s_, w_ in W4.items():
                mt, mx = (band == "tail") & (P.cell == s_).to_numpy(), (band == "xtail") & (P.cell == s_).to_numpy()
                num = num + w_ * (pm(wc, Y, mt) - pm(wc, Y, mx))
            return num / sum(W4.values())

        lines += [f"## pythia-{size}: trajectories (freq)", "",
                  "| step | checks (recon / identity) | L Head / Tail / XTail | ΔL Head − XTail | reading | side | ΔL_tok | reading | "
                  "side (token-std.) | Δ_pass(I) Head / XTail | Head AUC | D_L |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        YI, YT = {}, {}
        side_lines = []
        dsz = dec[f"pythia-{size}"] = {}
        for st in steps:
            sdir = root / f"step{st}"
            meta = json.loads((sdir / "frame_meta.json").read_text())
            T, I = load_step(sdir, pre, P, wn)
            YT[st], YI[st] = T, I
            L = T - I
            mb = {b: pm(wv, L, band == b) for b in BANDS}
            hx = mb["head"] - mb["xtail"]
            tk = tok(L)
            dT = pm(wv, T, band == "head") - pm(wv, T, band == "xtail")
            dI = pm(wv, I, band == "head") - pm(wv, I, band == "xtail")
            sd = ("no deficit" if classify(hx) != "deficit" else "intransitive-side" if ci(-dI)[0] > 0 and ci(-dI - dT)[0] > 0
                  else "transitive-side" if ci(dT)[0] > 0 and ci(dT + dI)[0] > 0 else "mixed")
            tT, tI = tok(T), tok(I)
            sd_tok = ("no deficit" if classify(tk) != "deficit" else "intransitive-side" if ci(-tI)[0] > 0 and ci(-tI - tT)[0] > 0
                      else "transitive-side" if ci(tT)[0] > 0 and ci(tT + tI)[0] > 0 else "mixed")
            auc, DL = affinity(sdir, pre, V, wn, np.random.default_rng(args.seed + 7), args.n_boot)
            dsz[st] = {"dL_head_xtail": [hx[0], *ci(hx)], "reading": classify(hx), "dL_tok": [tk[0], *ci(tk)],
                       "reading_tok": classify(tk), "side": sd, "side_tok": sd_tok, "tok_T_part": [tT[0], *ci(tT)],
                       "tok_I_part": [-tI[0], *ci(-tI)], "auc_head": auc, "D_L": [DL[0], *ci(DL)], "checks": meta["checks"]}
            lines.append(f"| {st} | {meta['checks']['max_first_subtoken_error']:.0e} / {meta['checks']['max_identity_error']:.0e} | "
                         + " / ".join(f"{mb[b][0]:+.2f}" for b in BANDS) + f" | {fmt(hx)} | {classify(hx)} | {sd} | {fmt(tk)} | "
                         f"{classify(tk)} | {sd_tok} | {pm(wv, I, band == 'head')[0]:+.2f} / {pm(wv, I, band == 'xtail')[0]:+.2f} | "
                         f"{auc:.2f} | {fmt(DL)} |")
            side_lines.append(f"| {st} | " + " / ".join(f"{pm(wv, T, band == b)[0]:+.2f}" for b in BANDS) + " | "
                              + " / ".join(f"{pm(wv, I, band == b)[0]:+.2f}" for b in BANDS)
                              + f" | {fmt(dT)} / {fmt(-dI)} | {fmt(tT)} / {fmt(-tI)} |")
            for nm, v in (("dL_head_xtail", hx), ("dL_tok", tk), ("D_L", DL)):
                rows.append({"size": size, "step": st, "stat": nm, "est": v[0], "lo": ci(v)[0], "hi": ci(v)[1]})
        for key in ("reading", "reading_tok"):
            reads = [dsz[st_][key] for st_ in steps]
            onset = next((steps[i] for i in range(len(steps)) if all(r == "deficit" for r in reads[i:])), None)
            dsz[f"onset_{key}"] = onset
        lines += ["", "Band means and side parts (positive part = contributes to the deficit):", "",
                  "| step | Δ_pass(T) Head / Tail / XTail | Δ_pass(I) Head / Tail / XTail | Head − XTail parts T / −I | "
                  "token-std. parts T / −I |", "|---|---|---|---|---|", *side_lines]
        lines += ["", f"Onset (first step after which every later checkpoint reads deficit; descriptive): Head − XTail "
                      f"{dsz['onset_reading']}, token-standardized {dsz['onset_reading_tok']}.", ""]
        if len(steps) < 4:
            continue
        # predictive curve collapse: units = I verb x step (step-major order)
        folds = fold_map(band, args.seed)
        nt = V.set_index("verb").n_tok
        ntok_i = np.log(nt.loc[P.t_verb if args.outcome == "T" else P.i_verb].to_numpy(float))
        if args.outcome == "L":  # descriptive: pair level, mean of the two verbs' log token counts
            ntok_i = 0.5 * (np.log(nt.loc[P.t_verb].to_numpy(float)) + np.log(nt.loc[P.i_verb].to_numpy(float)))

        def ns(x):
            k = np.percentile(x, [2.5, 100 / 3, 200 / 3, 97.5])
            return ns_basis(x, k)

        def rel_bounds(ma, mb):
            r = (ma - mb) / ma
            return r, np.percentile(r[1:], 2.5), np.percentile(r[1:], 97.5)

        for lab, cnt in (("c_total", ctot), ("c_marked", cmark)):
            c = cnt.loc[P.i_verb].to_numpy() + 0.5
            if args.outcome == "T":
                c = cnt.loc[P.t_verb].to_numpy() + 0.5
            if args.outcome == "L":  # pair exposure: geometric mean of the two verbs' smoothed counts
                c = np.sqrt((cnt.loc[P.t_verb].to_numpy() + 0.5) * (cnt.loc[P.i_verb].to_numpy() + 0.5))
            for sub_lab, keep in (("all steps", steps), ("steps ≥ 1000", [s_ for s_ in steps if s_ >= 1000]),
                                  ("steps ≥ 4000", [s_ for s_ in steps if s_ >= 4000])):
                if lab == "c_marked" and sub_lab != "all steps":
                    continue
                ks = np.array(keep)
                n_u = len(ks) * len(P)
                logc = np.tile(np.log10(c), len(ks))
                logt = np.repeat(np.log10(ks / TOTAL), len(P))
                logE = logc + logt
                tk = np.tile(ntok_i, len(ks))
                one = np.ones(n_u)
                stepd = np.column_stack([np.repeat(ks, len(P)) == s_ for s_ in ks[1:]]).astype(float)
                BE, Bc, Bt = ns(logE), ns(logc), ns(logt)
                tens = np.column_stack([Bc[:, i] * Bt[:, j] for i in range(Bc.shape[1]) for j in range(Bt.shape[1])])
                Xs = {"M_E": np.column_stack([one, tk, BE]), "M_Cg": np.column_stack([one, tk, Bc, stepd]),
                      "M_2D": np.column_stack([one, tk, Bc, Bt, tens]), "M_Eg": np.column_stack([one, tk, BE, stepd]),
                      "M_g": np.column_stack([one, tk, stepd])}
                Y = np.concatenate([YI[s_] for s_ in ks], axis=1)
                outcome = "Δ_pass(I)"
                if args.outcome == "T":
                    Y, outcome = np.concatenate([YT[s_] for s_ in ks], axis=1), "Δ_pass(T)"
                if args.outcome == "L":
                    Y, outcome = np.concatenate([YT[s_] - YI[s_] for s_ in ks], axis=1), "L"
                W = np.tile(wb, (1, len(ks)))
                F = np.tile(folds, len(ks))
                ms = {k: np.array([cv_mse(X, Y[b], W[b], F) for b in range(len(Y))]) for k, X in Xs.items()}
                r1, l1, h1 = rel_bounds(ms["M_E"], ms["M_2D"])
                cc1 = "collapse holds" if h1 < 0.02 else "collapse fails" if l1 > 0.02 else "unresolved"
                r2, l2, h2 = rel_bounds(ms["M_E"], ms["M_Cg"])
                cc2 = ("exposure as predictive as frequency + step" if h2 < 0.02 else
                       "frequency + step more predictive" if l2 > 0.02 else "unresolved")
                ru, lu, hu = rel_bounds(ms["M_g"], ms["M_E"])
                useful = bool(lu > 0.02)
                rd, ld, hd = rel_bounds(ms["M_E"], ms["M_Eg"])
                key = f"{lab}|{sub_lab}"
                dsz[key] = {"mse": {k: float(v[0]) for k, v in ms.items()}, "CC1": cc1, "CC1_rel": [r1[0], l1, h1],
                            "CC2": cc2, "CC2_rel": [r2[0], l2, h2], "useful_vs_step": useful, "useful_rel": [ru[0], lu, hu],
                            "diag_step_beyond_E_rel": [rd[0], ld, hd],
                            "collapse_supported": bool(cc1 == "collapse holds" and useful)}
                lines.append(f"- Curve collapse [{outcome}] ({lab}, {sub_lab}; {len(ks)} steps × {len(P)} verbs): CV MSE " +
                             ", ".join(f"{k} {v[0]:.3f}" for k, v in ms.items()) +
                             f". **CC1** M_E vs M_2D relative gain of M_2D {r1[0]:+.3f} [{l1:+.3f}, {h1:+.3f}] → **{cc1}**; "
                             f"**CC2** vs M_Cg {r2[0]:+.3f} [{l2:+.3f}, {h2:+.3f}] → **{cc2}**; M_E beats step-only by > 2%: "
                             f"**{useful}** ({ru[0]:+.3f} [{lu:+.3f}, {hu:+.3f}]); diagnostic step beyond E {rd[0]:+.3f} [{ld:+.3f}, {hd:+.3f}].")
                if lab == "c_total" and sub_lab == "all steps":
                    b0 = np.linalg.lstsq(Xs["M_Eg"], Y[0], rcond=None)[0]
                    g = b0[2 + BE.shape[1]:]
                    lines.append("  Step offsets in M_Eg (relative to the first step): " + ", ".join(f"{s_}: {x:+.2f}" for s_, x in zip(ks[1:], g)))
                    y0 = Y[0]
                    st_u = np.repeat(ks, len(P))
                    order = np.argsort(logE)
                    diffs = []
                    for a in range(len(order)):
                        for b in range(a + 1, len(order)):
                            i, j = order[a], order[b]
                            if logE[j] - logE[i] > 0.1:
                                break
                            if max(st_u[i], st_u[j]) >= 4 * min(st_u[i], st_u[j]):
                                late, early = (j, i) if st_u[j] > st_u[i] else (i, j)
                                diffs.append(y0[late] - y0[early])
                    if diffs:
                        lines.append(f"  Matched cells (|Δ log10 E| < 0.1, steps ≥ 4× apart): {len(diffs)} pairs; mean {outcome} later − "
                                     f"earlier {np.mean(diffs):+.3f} (median {np.median(diffs):+.3f}).")
                    for s_ in ks:
                        yv = {"I": YI[s_][0], "T": YT[s_][0], "L": YT[s_][0] - YI[s_][0]}[args.outcome]
                        cells.append(pd.DataFrame({"size": size, "step": s_, "t_verb": P.t_verb, "i_verb": P.i_verb, "band": band,
                                                   "logE": np.log10(c * s_ / TOTAL), "outcome": args.outcome, "y": yv}))
        lines.append("")
    out = Path(args.res_dir)
    sfx = "" if args.outcome == "I" else f"_{args.outcome}"  # descriptive T / L runs must not overwrite the primary outputs
    Path(args.report).write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(out / f"a6_summary{sfx}.csv", index=False)
    (out / f"a6_decisions{sfx}.json").write_text(json.dumps(dec, indent=1, default=float))
    if cells:
        pd.concat(cells).to_csv(out / f"a6_cells{sfx}.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/round6/frame_d11_freq")
    ap.add_argument("--res-dir", default="results/round6/checkpoints")
    ap.add_argument("--counts", default="data/round6/counts/pile_counts.csv")
    ap.add_argument("--slices", default="data/round6/checkpoints/slice_counts.csv")
    ap.add_argument("--sizes", nargs="+", default=["1.4b", "410m", "6.9b"])
    ap.add_argument("--outcome", default="I", choices=["I", "T", "L"],
                    help="primary: I (Delta_pass(I)); T, L: the descriptive sensitivities (separate reports; L uses the "
                         "pair's geometric-mean count and mean log token count)")
    ap.add_argument("--report", default="reports/round6/a6_checkpoints.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    main(ap.parse_args())
