#!/usr/bin/env python3
"""Round 5, A1-A2 analysis (reports/round5/plan.md, A1-A2). No model; numpy/pandas only.

Inputs: data/round5/frame/{prefixes,verbs,pairs}.csv, results/round5/frame/{frame_scores.parquet,frame_h.npz,
frame_meta.json}; the band-cross scores (reproduction check, whole-sentence and suffix margins) and the fit
ratings (fit-controlled accounting).
Outputs: reports/round5/a1_frame.md, reports/round5/a2_decomposition.md, results/round5/frame/a1_a2_summary.csv,
results/round5/frame/a2_verbs.csv, results/round5/frame/a1_a2_decisions.json.

Bootstrap: pairs resampled within band (per-verb pool analyses: verbs within role x band cells); contexts
resampled jointly for every band, frame and outcome of a prefix set (freq / been: nouns within animacy strata;
cur: within context band; they: one context). 2,000 draws, seed 17; draw 0 = the point estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_bad_fit import fit as wls_fit
from analyze_fit_ratings import mean_rating

BANDS = ("head", "tail", "xtail")
NAME = {"head": "Head", "tail": "Tail", "xtail": "XTail"}
SETS = ("freq", "been", "cur", "they")
BOUND = 0.25
LABELS = {"L": "L", "dT": "Δ_pass(T)", "dI": "Δ_pass(I)", "Mwas": "M_was", "Mhas": "M_has",
          "L1": "L₁ (first subtoken)", "Lrest": "L_rest (later subtokens)"}


def weights(rng, labels, n_boot):
    """Multinomial resampling weights within each label; row 0 = all ones (point estimate)."""
    labels = np.asarray(labels)
    w = np.ones((n_boot + 1, len(labels)))
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        w[1:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=n_boot)
    return w


def ci(d, lo=2.5, hi=97.5):
    d = np.asarray(d)[1:]
    d = d[np.isfinite(d)]
    return np.percentile(d, lo), np.percentile(d, hi)


def fmt(d, p=2):
    lo, hi = ci(d)
    return f"{d[0]:+.{p}f} [{lo:+.{p}f}, {hi:+.{p}f}]"


def classify(d, bound=BOUND):
    lo, hi = ci(d)
    lo90, hi90 = ci(d, 5, 95)
    if lo > 0 and d[0] >= bound:
        return "deficit"
    if hi < 0 and d[0] <= -bound:
        return "surplus"
    if -bound < lo90 and hi90 < bound:
        return "negligible"
    return "unresolved"


def wrong_sign(d):
    return float(np.mean(np.sign(d[1:]) != np.sign(d[0])))


class Boot:
    """Unit (pair or verb) x context bootstrap of [unit, context] matrices; one draw shared by all outcomes."""

    def __init__(self, unit_strata, ctx_strata, n_boot, rng):
        self.strata = np.asarray(unit_strata)
        self.wv = weights(rng, self.strata, n_boot)
        self.wc = weights(rng, ctx_strata, n_boot) if len(ctx_strata) > 1 else np.ones((n_boot + 1, 1))

    def subset(self, ctx_mask):
        """The same draws restricted to a subset of contexts (whole strata, so still a valid resample)."""
        b = object.__new__(Boot)
        b.strata, b.wv, b.wc = self.strata, self.wv, self.wc[:, ctx_mask]
        return b

    def unit_means(self, X):
        return (self.wc @ X.T) / self.wc.sum(1, keepdims=True)  # [draw, unit]

    def mean(self, X, mask):
        u = self.unit_means(X)
        w = self.wv[:, mask]
        return (w * u[:, mask]).sum(1) / w.sum(1)

    def slope(self, X, z, mask=None):
        u = self.unit_means(X)
        mask = np.ones(u.shape[1], bool) if mask is None else mask
        out = np.empty(len(u))
        for b in range(len(u)):
            w, zz, uu = self.wv[b, mask], z[mask], u[b, mask]
            zm, um = np.average(zz, weights=w), np.average(uu, weights=w)
            out[b] = np.sum(w * (zz - zm) * (uu - um)) / np.sum(w * (zz - zm) ** 2)
        return out


def pair_tables(S, pairs, pre):
    out = {}
    T, I = list(pairs.t_verb), list(pairs.i_verb)
    for st in SETS:
        s = S[S.set == st]
        ctx = list(pre[pre.set == st].context_id)
        lp = s.pivot_table(index=["verb", "frame"], columns="context_id", values="lp")[ctx]
        lp1 = s.pivot_table(index=["verb", "frame"], columns="context_id", values="lp1")[ctx]
        g = lambda tab, verbs, f: tab.loc[[(x, f) for x in verbs]].to_numpy()
        tw, ta, iw, ia = g(lp, T, "pass"), g(lp, T, "act"), g(lp, I, "pass"), g(lp, I, "act")
        t1w, t1a, i1w, i1a = g(lp1, T, "pass"), g(lp1, T, "act"), g(lp1, I, "pass"), g(lp1, I, "act")
        q = {"dT": tw - ta, "dI": iw - ia, "Mwas": tw - iw, "Mhas": ta - ia}
        q["L"] = q["dT"] - q["dI"]
        q["L1"] = (t1w - t1a) - (i1w - i1a)
        q["Lrest"] = q["L"] - q["L1"]
        assert np.allclose(q["Mwas"], q["L"] + q["Mhas"], atol=1e-6)
        out[st] = (q, ctx)
    return out


def ctx_strata(pre, st, ctx):
    p = pre.set_index("context_id").loc[ctx]
    if st == "cur":
        return list(p.context_band)
    if st in ("freq", "been"):
        return list(p.animacy)
    return ["all"] * len(ctx)


def band_table(B, q, keys, zp, zfor, lines, rows, st, tag=""):
    lines += ["| Quantity | Head | Tail | XTail | Head − Tail | Head − XTail | Zipf slope |",
              "|---|---|---|---|---|---|---|"]
    for k in keys:
        m = {b: B.mean(q[k], B.strata == b) for b in BANDS}
        sl = B.slope(q[k], zfor.get(k, zp))
        lines.append(f"| {LABELS[k]} | " + " | ".join(fmt(m[b]) for b in BANDS) + f" | {fmt(m['head'] - m['tail'])} | "
                     f"{fmt(m['head'] - m['xtail'])} | {fmt(sl)} |")
        for nm, d in [(b, m[b]) for b in BANDS] + [("head-tail", m["head"] - m["tail"]),
                                                     ("head-xtail", m["head"] - m["xtail"]), ("zipf_slope", sl)]:
            rows.append({"set": st + tag, "quantity": k, "stat": nm, "est": d[0], "lo": ci(d)[0], "hi": ci(d)[1]})


def reading(cL, cH):
    return {("deficit", "negligible"): "was–has-specific", ("negligible", "deficit"): "has-frame-general",
            ("deficit", "deficit"): "both", ("negligible", "negligible"): "neither"}.get((cL, cH), "unresolved")


def a1(args, pairs, pre, tabs, lines, rows, dec, rng):
    zp = ((pairs.t_zipf + pairs.i_zipf) / 2).to_numpy()
    zfor = {"dT": pairs.t_zipf.to_numpy(), "dI": pairs.i_zipf.to_numpy()}
    boots = {}
    keys = ("L", "dT", "dI", "Mwas", "Mhas", "L1", "Lrest")
    titles = {"freq": "freq (primary): \"The N was\" / \"The N has\"", "been": "been: \"The N has been\" / \"The N has\"",
              "cur": "cur: curated prefixes, was / has", "they": "they: \"They were\" / \"They have\""}
    for st in SETS:
        q, ctx = tabs[st]
        strata = ctx_strata(pre, st, ctx)
        B = Boot(pairs.band, strata, args.n_boot, rng)
        boots[st] = B
        lines += [f"### {titles[st]} ({len(ctx)} context{'s' if len(ctx) > 1 else ''})", ""]
        band_table(B, q, keys, zp, zfor, lines, rows, st)
        h, x = B.strata == "head", B.strata == "xtail"
        dL, dH = B.mean(q["L"], h) - B.mean(q["L"], x), B.mean(q["Mhas"], h) - B.mean(q["Mhas"], x)
        dT, dI = B.mean(q["dT"], h) - B.mean(q["dT"], x), B.mean(q["dI"], h) - B.mean(q["dI"], x)
        cL, cH = classify(dL), classify(dH)
        t = B.strata == "tail"
        tL, tH = (classify(B.mean(q[k], h) - B.mean(q[k], t)) for k in ("L", "Mhas"))
        dec[st] = {"L_head_xtail": cL, "Mhas_head_xtail": cH, "reading": reading(cL, cH),
                   "L_head_tail": tL, "Mhas_head_tail": tH, "reading_head_tail": reading(tL, tH),
                   "dL": [dL[0], *ci(dL)], "dMhas": [dH[0], *ci(dH)]}
        share = (f"; as shares of ΔL {dT[0] / dL[0]:.2f} / {-dI[0] / dL[0]:.2f}" if wrong_sign(dL) <= 0.025
                 else " (ΔL sign-unstable: no shares)")
        lines += ["", f"- Side contributions to ΔL (Head − XTail, additive): ΔΔ_pass(T) {fmt(dT)}, "
                      f"−ΔΔ_pass(I) {fmt(-dI)}{share}.",
                  f"- ΔL: **{cL}** ({fmt(dL)}; 90% CI [{ci(dL, 5, 95)[0]:+.2f}, {ci(dL, 5, 95)[1]:+.2f}]). "
                  f"ΔM_has: **{cH}** ({fmt(dH)}; 90% CI [{ci(dH, 5, 95)[0]:+.2f}, {ci(dH, 5, 95)[1]:+.2f}]).",
                  f"- **Reading (Head − XTail): {dec[st]['reading']}.** Head − Tail: L {tL}, M_has {tH} → "
                  f"{dec[st]['reading_head_tail']}.", ""]
        if st in ("freq", "been"):
            for an in ("animate", "inanimate"):
                cm = np.array([a == an for a in strata])
                qa = {k: v[:, cm] for k, v in q.items()}
                Ba = B.subset(cm)  # the set's own pair and context draws, animate / inanimate strata only
                lines += [f"{st}, {an} nouns only ({int(cm.sum())}):", ""]
                band_table(Ba, qa, ("L", "Mhas", "Mwas"), zp, zfor, lines, rows, st, f"_{an}")
                ha, xa = Ba.strata == "head", Ba.strata == "xtail"
                cLa = classify(Ba.mean(qa["L"], ha) - Ba.mean(qa["L"], xa))
                cHa = classify(Ba.mean(qa["Mhas"], ha) - Ba.mean(qa["Mhas"], xa))
                dec[f"{st}_{an}"] = {"L_head_xtail": cLa, "Mhas_head_xtail": cHa, "reading": reading(cLa, cHa)}
                lines += ["", f"- Reading ({an}): {reading(cLa, cHa)} (ΔL {cLa}, ΔM_has {cHa}).", ""]
    q, ctx = tabs["cur"]
    cix = {c: j for j, c in enumerate(ctx)}
    own = np.array([cix[c] for c in pairs.own_context_id])
    Bo = boots["cur"].subset(np.zeros(len(ctx), bool))
    Bo.wc = np.ones((boots["cur"].wv.shape[0], 1))  # one (own) context per pair; pair draws shared with cur
    lines += ["### own: each pair's own curated context (descriptive, paired)", "",
              "| Quantity | Head | Tail | XTail | Head − XTail |", "|---|---|---|---|---|"]
    for k in ("L", "dT", "dI", "Mwas", "Mhas"):
        v = q[k][np.arange(len(pairs)), own][:, None]
        m = {b: Bo.mean(v, Bo.strata == b) for b in BANDS}
        lines.append(f"| {LABELS[k]} | " + " | ".join(fmt(m[b]) for b in BANDS) + f" | {fmt(m['head'] - m['xtail'])} |")
    lines.append("")
    return boots


def spearman(x, y):
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def token_cells(pairs, tabs, V, boots, lines, rows):
    nt = V.set_index("verb").n_tok
    cell = np.array([f"{nt[t]}×{nt[i]}" for t, i in zip(pairs.t_verb, pairs.i_verb)])  # exact (T, I) token counts
    band = pairs.band.to_numpy()
    lines += ["### By token-count cell (exact T tokens × I tokens)", ""]
    for st in SETS:
        q, ctx = tabs[st]
        B = boots[st]
        lines += [f"**{st}**", "", "| Cell | Band | pairs | L | L₁ | L_rest |", "|---|---|---:|---|---|---|"]
        for c in sorted(set(cell)):
            for b in BANDS:
                m = (cell == c) & (band == b)
                if m.sum() == 0:
                    continue
                vals = [fmt(B.mean(q[k], m)) if m.sum() >= 5 else f"{B.mean(q[k], m)[0]:+.2f}" for k in ("L", "L1", "Lrest")]
                lines.append(f"| {c} | {NAME[b]} | {int(m.sum())} | " + " | ".join(vals) + " |")
        lines += ["", "Equal-count Tail − XTail comparisons (cells with ≥ 5 pairs in both bands):", ""]
        found = False
        for c in sorted(set(cell)):
            mt, mx = (cell == c) & (band == "tail"), (cell == c) & (band == "xtail")
            if mt.sum() >= 5 and mx.sum() >= 5:
                found = True
                parts = []
                for k in ("L", "L1", "Lrest"):
                    d = B.mean(q[k], mt) - B.mean(q[k], mx)
                    parts.append(f"{LABELS[k].split(' ')[0]} {fmt(d)}")
                    rows.append({"set": st, "quantity": k, "stat": f"tail-xtail_cell_{c}", "est": d[0],
                                 "lo": ci(d)[0], "hi": ci(d)[1]})
                lines.append(f"- {c} ({int(mt.sum())} Tail, {int(mx.sum())} XTail): " + "; ".join(parts) + ".")
        if not found:
            lines.append("- none")
        lines.append("")
    q, _ = tabs["freq"]
    gap = (pairs.t_zipf - pairs.i_zipf).to_numpy()
    lgap = (pairs.t_lemma_zipf - pairs.i_lemma_zipf).to_numpy()
    L = q["L"].mean(1)
    lines += ["### Within-pair Zipf gaps (T − I) and L (freq)", "", "| Band | form gap: mean [min, max] | lemma gap: mean | "
              "Spearman(L, form gap) |", "|---|---|---|---:|"]
    for b in BANDS:
        m = band == b
        lines.append(f"| {NAME[b]} | {gap[m].mean():+.2f} [{gap[m].min():+.2f}, {gap[m].max():+.2f}] | {lgap[m].mean():+.2f} | "
                     f"{spearman(L[m], gap[m]):+.2f} |")
    lines += ["", f"All pairs: Spearman(L, form gap) = {spearman(L, gap):+.2f}.", ""]


def accounting(args, pairs, tabs, boots, lines, rows, rng):
    q, ctx = tabs["cur"]
    B = boots["cur"]
    sc = pd.read_csv(args.scores, low_memory=False)
    sc = sc[(sc.context_set == "curated") & (sc.paradigm == "passive_1")]
    piv = lambda col: sc.pivot_table(index="verb_pair", columns="context_id", values=col).loc[pairs.verb_pair, ctx].to_numpy()
    vm, wm, sm = piv("verb_margin"), piv("whole_margin"), piv("suffix_margin")
    err = float(np.abs(vm - q["Mwas"]).max())
    lines += ["## Accounting for the earlier deficits (cur contexts)", "",
              f"Check: M_was reproduces the band-cross participle margin, max |difference| {err:.1e} nats.", ""]
    if err > 1e-3:
        raise SystemExit(f"M_was does not reproduce the band-cross verb margins (max {err:.2e}); stop")
    assert np.allclose(wm, vm + sm, atol=5e-4)  # prefix tokens: batch noise only
    lines += ["Whole-sentence margin, additive: ΔW = ΔL + ΔM_has + ΔM_suffix (ΔM_was = ΔL + ΔM_has).", "",
              "| Contrast | ΔW | ΔL | ΔM_has | ΔM_suffix | ΔM_was | ΔL / ΔM_was | ΔL / ΔW |", "|---|---|---|---|---|---|---|---|"]
    for b in ("tail", "xtail"):
        h, o = B.strata == "head", B.strata == b
        d = {k: B.mean(X, h) - B.mean(X, o) for k, X in (("W", wm), ("L", q["L"]), ("H", q["Mhas"]), ("S", sm), ("M", vm))}
        rat = []
        for num, den in (("L", "M"), ("L", "W")):
            r = d[num] / d[den]
            ws = wrong_sign(d[den])
            rat.append(fmt(r) if ws <= 0.025 else f"not interpreted ({ws:.1%} wrong-sign)")
            ok = ws <= 0.025
            rows.append({"set": "cur", "quantity": f"ratio_{num}_{den}", "stat": f"head-{b}",
                         "est": r[0] if ok else np.nan, "lo": ci(r)[0] if ok else np.nan, "hi": ci(r)[1] if ok else np.nan,
                         "denominator_wrong_sign": ws})
        lines.append(f"| Head − {NAME[b]} | {fmt(d['W'])} | {fmt(d['L'])} | {fmt(d['H'])} | {fmt(d['S'])} | {fmt(d['M'])} | "
                     + " | ".join(rat) + " |")
        for k in d:
            rows.append({"set": "cur", "quantity": f"acct_{k}", "stat": f"head-{b}", "est": d[k][0], "lo": ci(d[k])[0],
                         "hi": ci(d[k])[1]})
    _, good = mean_rating(args.good_ratings)
    _, badr = mean_rating(args.bad_ratings)
    links = pd.read_csv(args.bad_links, dtype={"item_id": str})
    bad = links.merge(badr[["item_id", "rating"]], on="item_id").pivot_table(index="pair_id", columns="role", values="rating")
    bad.columns = [f"bad_{c}" for c in bad.columns]
    gl = pd.read_csv(args.good_links, dtype={"item_id": str})[["pair_id", "item_id"]]
    cur = sc.merge(gl, on="pair_id").merge(good[["item_id", "rating"]].rename(columns={"rating": "good_fit"}), on="item_id") \
        .merge(bad, left_on="pair_id", right_index=True)
    for k in ("L", "Mhas", "Mwas"):
        cur = cur.merge(pd.DataFrame(q[k], index=pairs.verb_pair, columns=ctx).stack().rename(k),
                        left_on=["verb_pair", "context_id"], right_index=True)
    assert len(cur) == len(pairs) * len(ctx), len(cur)
    verbs = cur.drop_duplicates("verb_pair").sort_values("verb_pair")
    ctxs = cur.drop_duplicates("context_id").sort_values("context_id")
    vi = cur.verb_pair.map({v: i for i, v in enumerate(verbs.verb_pair)}).to_numpy()
    cx = cur.context_id.map({c: i for i, c in enumerate(ctxs.context_id)}).to_numpy()
    # the cur set's own pair and context draws, mapped into this table's order
    pcol = {v: j for j, v in enumerate(pairs.verb_pair)}
    ccol = {c: j for j, c in enumerate(ctx)}
    wv = B.wv[:, [pcol[v] for v in verbs.verb_pair]]
    wc = B.wc[:, [ccol[c] for c in ctxs.context_id]]
    X = np.column_stack([np.ones(len(cur)), cur.good_fit - 4, cur.bad_patient - 4,
                         (cur.verb_band == "tail").astype(float), (cur.verb_band == "xtail").astype(float)])
    beta = {k: wls_fit(cur, cur[k].to_numpy(float), X, vi, cx, wv, wc) for k in ("Mwas", "verb_margin", "L", "Mhas")}
    lines += ["", "Fit-controlled (`good+bad_patient` WLS refit per outcome on identical rows, design and draws; "
              "band − Head coefficients; β(M_was) = β(L) + β(M_has)):", "",
              "| Band | β(M_was) | β(L) | β(M_has) | β(L) / β(M_was) |", "|---|---|---|---|---|"]
    for j, b in ((3, "tail"), (4, "xtail")):
        bm, bl, bh = beta["Mwas"][:, j], beta["L"][:, j], beta["Mhas"][:, j]
        assert np.allclose(bm, bl + bh, atol=1e-6)
        assert abs(beta["verb_margin"][0, j] - bm[0]) < 1e-3  # the scorer's margin gives the same coefficient
        ws = wrong_sign(bm)
        r = bl / bm
        lines.append(f"| {NAME[b]} | {fmt(bm)} | {fmt(bl)} | {fmt(bh)} | "
                     + (fmt(r) if ws <= 0.025 else f"not interpreted ({ws:.1%} wrong-sign)") + " |")
        for nm, d in (("beta_Mwas", bm), ("beta_L", bl), ("beta_Mhas", bh)):
            rows.append({"set": "cur", "quantity": nm, "stat": b, "est": d[0], "lo": ci(d)[0], "hi": ci(d)[1]})
        ok = ws <= 0.025
        rows.append({"set": "cur", "quantity": "beta_ratio", "stat": b, "est": r[0] if ok else np.nan,
                     "lo": ci(r)[0] if ok else np.nan, "hi": ci(r)[1] if ok else np.nan, "denominator_wrong_sign": ws})
    qf, _ = tabs["freq"]
    Bf = boots["freq"]
    dLf = Bf.mean(qf["L"], Bf.strata == "head") - Bf.mean(qf["L"], Bf.strata == "xtail")
    lines += ["", f"Descriptive: ΔL (Head − XTail) on the freq contexts is {fmt(dLf)}, beside ΔL on cur above.", ""]


def a2(args, pairs, pre, V, tabs, boots, lines, rows, rng):
    z = np.load(args.h)
    keys = list(z["keys"])
    H, U = z["H"].astype(np.float64), z["U"].astype(np.float64)
    urow = {int(t): i for i, t in enumerate(z["U_ids"])}
    fctx = list(pre[pre.set == "freq"].context_id)
    an = list(pre.set_index("context_id").loc[fctx].animacy)
    dh = np.stack([H[keys.index(f"freq|{c}|pass")] - H[keys.index(f"freq|{c}|act")] for c in fctx])  # [24, d]
    B = boots["freq"]  # A1's freq pair and context draws; the same context draws give Delta-h-bar
    assert list(pre[pre.set == "freq"].context_id) == tabs["freq"][1]
    dbar = (B.wc @ dh) / B.wc.sum(1, keepdims=True)  # [draw, d]
    dev = np.mean(((dh - dbar[0]) ** 2).sum(1)) / np.mean((dh ** 2).sum(1))
    Uv = np.stack([U[urow[int(t)]] for t in V.first_id])  # [verb, d]
    A = dbar @ Uv.T  # [draw, verb]
    unorm = np.linalg.norm(Uv, axis=1)
    COS = A / (unorm[None] * np.linalg.norm(dbar, axis=1)[:, None])
    V = V.copy()
    V["a"], V["cos"], V["unorm"] = A[0], COS[0], unorm
    vix = {v: j for j, v in enumerate(V.verb)}
    q, ctx = tabs["freq"]
    ti, ii = np.array([vix[t] for t in pairs.t_verb]), np.array([vix[i] for i in pairs.i_verb])
    P = A[:, ti] - A[:, ii]  # [draw, pair]
    err = float(np.abs(P[0] - q["L1"].mean(1)).max())
    meta = json.loads((Path(args.res_dir) / "frame_meta.json").read_text())
    lines += ["## A2. Exact decomposition", "",
              f"- Identity L₁ = Δh · (u_T1 − u_I1): max error in the run {meta['checks']['max_identity_error']:.1e} "
              "(every pair and prefix, all sets; fail-fast ≤ 1e-3). This is algebraic; it checks the implementation.",
              f"- P = (u_T1 − u_I1) · Δh̄ equals the freq-mean L₁ for every pair: max |difference| {err:.1e}.",
              f"- ‖Δh̄‖ = {np.linalg.norm(dbar[0]):.2f}; per-context deviation share {dev:.2f} "
              "(mean ‖Δh_c − Δh̄‖² / mean ‖Δh_c‖²).", ""]
    wp = B.wv  # pair draws (T and I kept together), aligned with the context draws
    band = pairs.band.to_numpy()
    nt = V.set_index("verb").n_tok
    single = np.array([(nt[t] == 1) and (nt[i] == 1) for t, i in zip(pairs.t_verb, pairs.i_verb)])
    pm = lambda X, m: (wp[:, m] * X[:, m]).sum(1) / wp[:, m].sum(1)

    def single_cell(b):
        m = (band == b) & single
        if m.sum() >= 2:
            return fmt(pm(P, m))
        return f"{P[0, m].mean():+.2f} (n = 1)" if m.sum() == 1 else "—"

    lines += ["### Per pair: P = projection of u_T1 − u_I1 on Δh̄ (= freq-mean L₁)", "",
              "| | Head | Tail | XTail |", "|---|---|---|---|",
              "| P, all pairs | " + " | ".join(fmt(pm(P, band == b)) for b in BANDS) + " |",
              "| P, single-token pairs (= freq-mean L) | " + " | ".join(single_cell(b) for b in BANDS) + " |", "",
              f"Single-token pairs: {int(single.sum())} "
              f"({', '.join(f'{NAME[b]} {int((single & (band == b)).sum())}' for b in BANDS)}).", ""]
    multi = ~single
    lines += ["### Multi-token pairs: first vs later subtokens (freq)", "",
              "| Band | pairs | mean L | mean L₁ | mean L_rest | L₁ / L | L_rest / L |", "|---|---:|---|---|---|---|---|"]
    for b in ("tail", "xtail"):
        m = multi & (band == b)
        mL, m1, mr = (B.mean(q[k], m) for k in ("L", "L1", "Lrest"))
        ws = wrong_sign(mL)
        sh = [fmt(m1 / mL), fmt(mr / mL)] if ws <= 0.025 else ["not interpreted"] * 2
        lines.append(f"| {NAME[b]} | {int(m.sum())} | {fmt(mL)} | {fmt(m1)} | {fmt(mr)} | " + " | ".join(sh) + " |")
        ok = ws <= 0.025
        rows.append({"set": "freq", "quantity": "L1_share_multi", "stat": b, "est": (m1 / mL)[0] if ok else np.nan,
                     "lo": ci(m1 / mL)[0] if ok else np.nan, "hi": ci(m1 / mL)[1] if ok else np.nan,
                     "denominator_wrong_sign": ws})
    cell = np.array([f"{nt[t]}×{nt[i]}" for t, i in zip(pairs.t_verb, pairs.i_verb)])
    lines += ["", "By exact token-count cell (multi-token pairs with ≥ 5 pairs in the cell and band):", "",
              "| Cell | Band | pairs | mean L | mean L₁ | mean L_rest | L₁ / L |", "|---|---|---:|---|---|---|---|"]
    for c in sorted(set(cell[multi])):
        for b in ("tail", "xtail"):
            m = multi & (band == b) & (cell == c)
            if m.sum() < 5:
                continue
            mL, m1, mr = (B.mean(q[k], m) for k in ("L", "L1", "Lrest"))
            sh = fmt(m1 / mL) if wrong_sign(mL) <= 0.025 else "not interpreted"
            lines.append(f"| {c} | {NAME[b]} | {int(m.sum())} | {fmt(mL)} | {fmt(m1)} | {fmt(mr)} | {sh} |")
    Lh = B.mean(q["L"], band == "head")
    x1, xr = B.mean(q["L1"], multi & (band == "xtail")), B.mean(q["Lrest"], multi & (band == "xtail"))
    lines += ["", f"XTail shortfall: L̄_Head − L_XTail = (L̄_Head − L₁,XTail) − L_rest,XTail: {fmt(Lh - x1 - xr)} = "
              f"{fmt(Lh - x1)} − ({fmt(xr)}).", ""]
    V["bandkey"] = V.band.str.split("|").str[0]
    lines += ["### Per verb: passive-slot affinity a_V = u_V1 · Δh̄", "",
              "| Verbs | role | n | Zipf range | slope of a_V | slope of cos | slope of ‖u‖ |", "|---|---|---:|---|---|---|---|"]

    def slopes(idx, wverb):
        zz = V.zipf.to_numpy()[idx]
        res = []
        for Y in (A[:, idx], COS[:, idx], np.broadcast_to(unorm[idx], (len(A), len(idx)))):
            d = np.empty(len(A))
            for b in range(len(A)):
                w = wverb[b]
                zm, ym = np.average(zz, weights=w), np.average(Y[b], weights=w)
                d[b] = np.sum(w * (zz - zm) * (Y[b] - ym)) / np.sum(w * (zz - zm) ** 2)
            res.append(fmt(d, 3))
        return res

    pair_of = {v: k for k, v in enumerate(pairs.t_verb)} | {v: k for k, v in enumerate(pairs.i_verb)}
    for nm, base in (("single-token, curated", (V.n_tok == 1) & (V.curated == 1)),
                     ("first-subtoken fragments, curated", (V.n_tok > 1) & (V.curated == 1))):
        for role in ("T", "I"):
            idx = np.flatnonzero(base & (V.role == role))
            if len(idx) < 3:
                continue
            wverb = wp[:, [pair_of[v] for v in V.verb.to_numpy()[idx]]]
            zz = V.zipf.to_numpy()[idx]
            lines.append(f"| {nm} | {role} | {len(idx)} | {zz.min():.2f}–{zz.max():.2f} | " + " | ".join(slopes(idx, wverb)) + " |")
    for role in ("T", "I"):
        # curated verbs inherit the pair draws; pool-only verbs are resampled within role x band cells
        idx = np.flatnonzero((V.n_tok == 1) & (V.role == role))
        cur_ = V.curated.to_numpy()[idx] == 1
        wverb = np.empty((len(A), len(idx)))
        wverb[:, cur_] = wp[:, [pair_of[v] for v in V.verb.to_numpy()[idx][cur_]]]
        wverb[:, ~cur_] = weights(rng, V.bandkey.to_numpy()[idx][~cur_], args.n_boot)
        zz = V.zipf.to_numpy()[idx]
        lines.append(f"| single-token, curated + pool | {role} | {len(idx)} | {zz.min():.2f}–{zz.max():.2f} | "
                     + " | ".join(slopes(idx, wverb)) + " |")
    hv = (V.curated == 1) & (V.band == "head")
    at, ai = A[0, np.flatnonzero(hv & (V.role == "T"))], A[0, np.flatnonzero(hv & (V.role == "I"))]
    auc = float(np.mean(at[:, None] > ai[None, :]) + 0.5 * np.mean(at[:, None] == ai[None, :]))
    lines += ["", f"Head class separation (curated): AUC(a_V, T vs I) = {auc:.2f} ({len(at)} T, {len(ai)} I).", ""]
    lines += ["### Gate for A4 (single-token participles; verbs within role × band cells, Δh̄ re-drawn)", "",
              "| Population | Head T | rare T | Head I | rare I | rare bands | D_T | D_I | D_L = D_T − D_I |",
              "|---|---:|---:|---:|---:|---|---|---|---|"]
    gate = {}
    pops = {"curated + pool (gate)": V.n_tok == 1,
            "curated only": (V.n_tok == 1) & (V.curated == 1),
            "curated + pool, Head Zipf ≤ 4.0": (V.n_tok == 1) & ((V.bandkey != "head") | (V.zipf <= 4.0))}
    for nm, base in pops.items():
        idx = np.flatnonzero(base)
        role, bk = V.role.to_numpy()[idx], V.bandkey.to_numpy()[idx]
        grp = np.array([f"{r}_{'head' if b == 'head' else 'rare'}" for r, b in zip(role, bk)])
        w = weights(rng, np.array([f"{r}_{b}" for r, b in zip(role, bk)]), args.n_boot)
        Ab = A[:, idx]
        gm = lambda g: (w[:, grp == g] * Ab[:, grp == g]).sum(1) / w[:, grp == g].sum(1)
        DT, DI = gm("T_head") - gm("T_rare"), gm("I_head") - gm("I_rare")
        DL = DT - DI
        n = {g: int((grp == g).sum()) for g in ("T_head", "T_rare", "I_head", "I_rare")}
        rare_bands = sorted(set(bk[(grp == "T_rare") | (grp == "I_rare")]))
        lines.append(f"| {nm} | {n['T_head']} | {n['T_rare']} | {n['I_head']} | {n['I_rare']} | {', '.join(rare_bands)} | "
                     f"{fmt(DT, 3)} | {fmt(DI, 3)} | {fmt(DL, 3)} |")
        gate[nm] = {"D_L": [DL[0], *ci(DL)], "D_T": [DT[0], *ci(DT)], "D_I": [DI[0], *ci(DI)], "n": n, "rare_bands": rare_bands}
    V.to_csv(Path(args.res_dir) / "a2_verbs.csv", index=False)
    return gate, {"auc_head": auc, "deviation_share": dev, "P_err": err}


def main(args):
    rng = np.random.default_rng(args.seed)
    pre = pd.read_csv(args.data_dir / "prefixes.csv", keep_default_na=False)
    V = pd.read_csv(args.data_dir / "verbs.csv", keep_default_na=False)
    pairs = pd.read_csv(args.data_dir / "pairs.csv", keep_default_na=False)
    S = pd.read_parquet(args.res_dir / "frame_scores.parquet")
    tabs = pair_tables(S, pairs, pre)
    rows, dec = [], {}
    l1 = ["# Round 5, A1: frame contrast", "",
          "Spec: `plan.md`, A1. Run: `run_frame_contrast.py`; analysis: `analyze_frame_contrast.py`. "
          "Δ_pass(V) = log P(V | was-prefix) − log P(V | has-prefix) (been set: has been / has), full participle "
          "log-prob; L = Δ_pass(T) − Δ_pass(I); M_was / M_has = the T − I margin after was / has (M_was = L + M_has). "
          "126 curated pairs (26 Head, 50 Tail, 50 XTail). 95% CIs: pairs within band × contexts (one draw per set); "
          "Zipf slopes are secondary (pair-mean participle Zipf; Δ_pass(T) / Δ_pass(I) on their own participle's "
          "Zipf). Rule (bound 0.25 nats): deficit = 95% CI above 0 and estimate ≥ 0.25; surplus = the mirror; "
          "negligible = 90% CI within ±0.25; otherwise unresolved.", ""]
    boots = a1(args, pairs, pre, tabs, l1, rows, dec, rng)
    token_cells(pairs, tabs, V, boots, l1, rows)
    if not args.no_accounting:
        accounting(args, pairs, tabs, boots, l1, rows, rng)
    l2 = ["# Round 5, A2: exact decomposition", "", "Spec: `plan.md`, A2. Freq contexts unless stated; Δh̄ and "
          "affinities are recomputed in each context draw.", ""]
    gate, a2info = a2(args, pairs, pre, V, tabs, boots, l2, rows, rng)
    g = gate["curated + pool (gate)"]
    g1 = dec["freq"]["L_head_xtail"] == "deficit"
    g2 = g["D_L"][1] > 0
    g3 = g["n"]["T_rare"] >= 5 and g["n"]["I_rare"] >= 5
    if g1 and g2 and g3:
        verdict = "**A4 runs** (a Tail-singleton intervention; transfer to XTail untested)"
    else:
        verdict = "**A4 is not run**: " + ("no deficit in L (gate 1)" if not g1 else
                                           "no supported affinity gap in the eligible singleton population (gate 2)"
                                           if not g2 else "fewer than 5 rare single-token participles in a class (gate 3)")
    l2 += ["", f"Gate 1 (ΔL deficit, freq): {'met' if g1 else 'not met'}; gate 2 (D_L CI above 0): "
           f"{'met' if g2 else 'not met'}; gate 3: {'met' if g3 else 'not met'}. {verdict}.", ""]
    dec["a4_gate"] = {"gate1": g1, "gate2": g2, "gate3": g3, "runs": bool(g1 and g2 and g3), "detail": gate, **a2info}
    Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.report_dir) / "a1_frame.md").write_text("\n".join(l1) + "\n")
    (Path(args.report_dir) / "a2_decomposition.md").write_text("\n".join(l2) + "\n")
    pd.DataFrame(rows).to_csv(Path(args.res_dir) / "a1_a2_summary.csv", index=False)
    (Path(args.res_dir) / "a1_a2_decisions.json").write_text(json.dumps(dec, indent=1, default=float))
    print("\n".join(l1 + l2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/round5/frame"))
    ap.add_argument("--res-dir", type=Path, default=Path("results/round5/frame"))
    ap.add_argument("--h", default="results/round5/frame/frame_h.npz")
    ap.add_argument("--scores", default="results/passive_band_cross/pythia14b_scores.csv")
    ap.add_argument("--good-ratings", default="results/fit_ratings/gemma4_31b_it.csv")
    ap.add_argument("--good-links", default="data/fit_ratings/item_links.csv")
    ap.add_argument("--bad-ratings", default="results/fit_ratings/bad_gemma4_31b_it.csv")
    ap.add_argument("--bad-links", default="data/fit_ratings/bad_item_links.csv")
    ap.add_argument("--report-dir", default="reports/round5")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--no-accounting", action="store_true", help="skip the curated-score accounting (other models, D10b)")
    main(ap.parse_args())
