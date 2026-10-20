#!/usr/bin/env python3
"""Round 6, A1 analysis: Pile exposure, BE/HAVE preference and the D11 frame contrast (reports/round6/plan.md, A1).
No model. Needs scipy (Fisher-based strength).

Inputs: data/round6/counts/pile_counts.csv (fetch_pile_counts_r6.py); D11 verbs / pairs / prefixes
(data/round6/frame_d11) and the 1.4B D11 scores (results/round6/frame_d11/pythia-1.4b/frame_scores.parquet); curated
verbs / pairs (data/round5/frame) and round 5's 1.4B scores (results/round5/frame/frame_scores.parquet); round 5's
counts (data/round5/counts/pile_counts.csv; lowercase-only sensitivity); the D3 / precision audits
(data/round6/counts/diag_contexts.csv + diag_labels.csv), required before the outcome analysis.

Counts per participle: c_total = word-initial lowercase + capitalized + both forms after a newline (boundary-safe);
sensitivities: marked only, and an upper bound with every unmarked occurrence. c_be (7 be-forms), c_perf (4 have-forms),
c_comp (primary: "that/when/and" + he/she/they + simple past; secondary: broad pronoun adjacency). BE/HAVE preference S
(log-odds, Haldane +0.5, totals over the inventory); secondary S_fisher (signed -log10 p, saturating at 300), S_comp
(c_comp as the active column), S_washas ("was v" vs "has v").
Outcomes (1.4B, 24 freq frames): Delta_pass(T), Delta_pass(I), L. 10-fold CV with a frozen fold map (pairs shuffled
within band, seed 17, dealt in turn), repeated in each draw of a pairs-within-band x contexts-within-animacy bootstrap
(multiplicities as weights; 2,000 draws, seed 17). Comparison a -> b: Delta = MSE_a - MSE_b; b better = 95% CI above 0
(+ "within 2%" if the 90% CI of Delta / MSE_a is within +-2%); a better = below 0; equivalent = neither and the 90%
CI of Delta / MSE_a within +-2%; else unresolved. Direction: the added predictor's coefficient with its bootstrap CI.

Outputs: reports/round6/a1_exposure.md, results/round6/exposure/{a1_summary.csv, a1_decisions.json, a1_verbs.csv,
a1_diag_curated_I.csv}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BE = ("was", "were", "been", "is", "are", "be", "being")
PERF = ("has", "have", "had", "having")
COMP_BROAD = ("he", "she", "it", "they", "He", "She", "It", "They")
ANCHORS, ANCHOR_SUBJ = ("that", "when", "and"), ("he", "she", "they")
BANDS = ("head", "tail", "xtail")
NAME = {"head": "Head", "tail": "Tail", "xtail": "XTail"}
SIGN = {"I1": -1, "I3": -1, "T2": +1, "T3": +1}  # predicted sign of the added predictor's coefficient
PRIMARY = ("I1", "I3", "T2", "T3")
COMPARISONS = [  # id, outcome, base, alternative, question
    ("I1", "I", [], ["C"], "does total exposure predict intransitive rejection?"),
    ("I2", "I", ["C"], ["K"], "competing-frame count vs total count"),
    ("I3", "I", ["C"], ["C", "K"], "does the competing frame add to total exposure?"),
    ("I4", "I", ["C"], ["C", "B"], "does the passive slot add?"),
    ("I5", "I", ["Z"], ["C"], "Pile counts vs wordfreq Zipf"),
    ("T1", "T", [], ["C"], "does total exposure predict transitive acceptance?"),
    ("T2", "T", ["C"], ["C", "B"], "does the passive slot add?"),
    ("T3", "T", ["C"], ["C", "S"], "does the BE/HAVE preference add?"),
    ("T4", "T", ["Z"], ["C"], "Pile vs Zipf"),
    ("L1", "L", ["Z_T", "Z_I"], ["C_T", "C_I"], "Pile vs Zipf at pair level"),
    ("L2a", "L", ["C_T", "C_I"], ["C_T", "C_I", "S_T"], "does the T verb's preference add?"),
    ("L2b", "L", ["C_T", "C_I"], ["C_T", "C_I", "K_I"], "does the I verb's competing frame add?"),
    ("L2", "L", ["C_T", "C_I"], ["C_T", "C_I", "S_T", "K_I"], "both"),
]


def weights(rng, labels, n_boot):
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


def fmt(d, p=3, lo=2.5, hi=97.5):
    a, b = ci(d, lo, hi)
    return f"{d[0]:+.{p}f} [{a:+.{p}f}, {b:+.{p}f}]"


RANK_DEFICIENT = set()


def wls(X, y, w):
    sw = np.sqrt(w)
    beta, _, rank, _ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    if rank < X.shape[1]:
        RANK_DEFICIENT.add(X.shape[1])
    return beta


def r2(X, y):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def fold_map(bands, seed, folds=10):
    """Frozen fold map: units shuffled within band and dealt to folds in turn."""
    rng = np.random.default_rng(seed)
    f = np.empty(len(bands), int)
    k = 0
    for b in sorted(set(bands)):
        idx = rng.permutation(np.flatnonzero(np.asarray(bands) == b))
        f[idx] = (k + np.arange(len(idx))) % folds
        k += len(idx)
    return f


def cv_pred(X, y, w, folds_of, folds=10):
    pred = np.full(len(y), np.nan)
    for k in range(folds):
        te = folds_of == k
        tr = ~te & (w > 0)
        if te.sum() == 0:
            continue
        assert tr.sum() >= X.shape[1] + 1, "too few training units in a fold"
        pred[te] = X[te] @ wls(X[tr], y[tr], w[tr])
    assert np.isfinite(pred).all(), "a unit received no out-of-fold prediction"
    return pred


def cv_mse(X, y, w, folds_of):
    p = cv_pred(X, y, w, folds_of)
    m = w > 0
    return float((w[m] * (y[m] - p[m]) ** 2).sum() / w[m].sum())


def decide(ma, mb):
    d = ma - mb
    lo, hi = ci(d)
    lo90, hi90 = ci(d / ma, 5, 95)
    eq = -0.02 < lo90 and hi90 < 0.02
    if lo > 0:
        return ("b better (within 2%)" if eq else "b better"), d
    if hi < 0:
        return ("a better (within 2%)" if eq else "a better"), d
    return ("equivalent" if eq else "unresolved"), d


def build_counts(C, smooth):
    """Per-verb counts; every declared slot must be present exactly once per verb."""
    assert not C.duplicated(["verb", "slot"]).any(), "duplicate (verb, slot) rows"
    W = C.pivot(index="verb", columns="slot", values="count")
    need = (["overall", "overall_cap", "nl_low", "nl_cap", "bare_low", "bare_cap"] + [f"be_{x}" for x in BE]
            + [f"perf_{x}" for x in PERF] + [f"comp_{p}" for p in COMP_BROAD]
            + [f"compa_{a}_{p}" for a in ANCHORS for p in ANCHOR_SUBJ])
    missing = [c for c in need if c not in W.columns]
    assert not missing and W[need].notna().all().all(), f"missing count slots: {missing}"
    K = pd.DataFrame(index=W.index)
    K["c_low"], K["c_cap"], K["c_nl_low"], K["c_nl_cap"] = W.overall, W.overall_cap, W.nl_low, W.nl_cap
    K["c_bare_low"], K["c_bare_cap"] = W.bare_low, W.bare_cap
    K["c_total"] = K.c_low + K.c_cap + K.c_nl_low + K.c_nl_cap
    K["c_marked"] = K.c_low + K.c_cap
    K["c_upper"] = K.c_low + K.c_cap + K.c_bare_low + K.c_bare_cap
    K["c_be"] = W[[f"be_{x}" for x in BE]].sum(1)
    K["c_perf"] = W[[f"perf_{x}" for x in PERF]].sum(1)
    K["c_comp"] = W[[f"compa_{a}_{p}" for a in ANCHORS for p in ANCHOR_SUBJ]].sum(1)
    K["c_comp_broad"] = W[[f"comp_{p}" for p in COMP_BROAD]].sum(1)
    K["c_was"], K["c_has"] = W.be_was, W.perf_has
    from scipy.stats import fisher_exact

    s = smooth
    lo = lambda a, b, A, B: np.log((a + s) / (b + s)) - np.log((A - a + s) / (B - b + s))
    K["S"] = lo(K.c_be, K.c_perf, K.c_be.sum(), K.c_perf.sum())
    K["S_comp"] = lo(K.c_be, K.c_comp, K.c_be.sum(), K.c_comp.sum())
    K["S_washas"] = lo(K.c_was, K.c_has, K.c_was.sum(), K.c_has.sum())
    tb, tp = int(K.c_be.sum()), int(K.c_perf.sum())
    K["S_fisher"] = [np.sign(l) * -np.log10(max(fisher_exact([[a, b], [tb - a, tp - b]])[1], 1e-300))
                     for a, b, l in zip(K.c_be.astype(int), K.c_perf.astype(int), K.S)]
    return K


def outcome(S, pre, verbs, n_boot, seed):
    ctx = list(pre[pre.set == "freq"].context_id)
    s = S[S.set == "freq"]
    lp = s.pivot_table(index=["verb", "frame"], columns="context_id", values="lp")[ctx]
    D = (lp.xs("pass", level="frame") - lp.xs("act", level="frame")).loc[verbs].to_numpy()  # [verb, ctx]
    wc = weights(np.random.default_rng(seed), pre.set_index("context_id").loc[ctx].animacy.to_numpy(), n_boot)
    return (wc @ D.T) / wc.sum(1, keepdims=True)  # [draw, verb]


class Pop:
    """A verb population (D11 or curated): predictors per pair side, outcome draws, pair draws, frozen folds."""

    def __init__(self, name, pairs, K, YT, YI, wp, folds_of, smooth, ntok):
        self.name, self.pairs, self.wp, self.folds = name, pairs.reset_index(drop=True), wp, folds_of
        lg = lambda c: np.log10(c + smooth)
        self.X = {}
        for role in ("t", "i"):
            k = K.loc[self.pairs[f"{role}_verb"]]
            self.X[role.upper()] = pd.DataFrame({
                "Z": self.pairs[f"{role}_zipf"].to_numpy(float), "C": lg(k.c_total).to_numpy(),
                "C_marked": lg(k.c_marked).to_numpy(), "C_upper": lg(k.c_upper).to_numpy(),
                "B": lg(k.c_be).to_numpy(), "P": lg(k.c_perf).to_numpy(), "K": lg(k.c_comp).to_numpy(),
                "K_broad": lg(k.c_comp_broad).to_numpy(), "S": k.S.to_numpy(), "S_fisher": k.S_fisher.to_numpy(),
                "S_comp": k.S_comp.to_numpy(), "S_washas": k.S_washas.to_numpy(),
                "N": np.log(ntok.loc[self.pairs[f"{role}_verb"]].to_numpy(float))})
        self.Yd = {"T": YT, "I": YI, "L": YT - YI}

    def design(self, cls, preds, swap=None, add_tok=False):
        swap = swap or {}
        cols = []
        for p in preds:
            if cls == "L":
                nm, side = p.split("_")
                cols.append(self.X[side][swap.get(nm, nm)])
            else:
                cols.append(self.X[cls][swap.get(p, p)])
        if add_tok:
            cols += [self.X["T"]["N"], self.X["I"]["N"]] if cls == "L" else [self.X[cls]["N"]]
        return np.column_stack([np.ones(len(self.pairs))] + [np.asarray(c, float) for c in cols])

    def cv(self, cls, preds, mask=None, **kw):
        mask = np.ones(len(self.pairs), bool) if mask is None else mask
        X, Y = self.design(cls, preds, **kw)[mask], self.Yd[cls][:, mask]
        return np.array([cv_mse(X, Y[b], self.wp[b, mask], self.folds[mask]) for b in range(len(Y))])

    def coef(self, cls, preds, j, mask=None, **kw):
        """Bootstrap distribution of coefficient j (1 = first predictor) of the full-data weighted OLS."""
        mask = np.ones(len(self.pairs), bool) if mask is None else mask
        X, Y = self.design(cls, preds, **kw)[mask], self.Yd[cls][:, mask]
        return np.array([wls(X, Y[b], self.wp[b, mask])[j] for b in range(len(Y))])


def run_comparisons(pop, lines, rows, tag="", ids=None, **kw):
    out = {}
    lines += ["| ID | family | outcome | a → b | MSE a | MSE b | Δ = a − b (95% CI) | Δ / MSE a (90% CI) | decision | "
              "added coefficient (95% CI) | in-sample R² a / b | cond(b) | question |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for cid, cls, a, b, qtxt in COMPARISONS:
        if ids and cid not in ids:
            continue
        RANK_DEFICIENT.clear()
        ma, mb = pop.cv(cls, a, **kw), pop.cv(cls, b, **kw)
        rankflag = " (rank-deficient fits occurred)" if RANK_DEFICIENT else ""
        d, delta = decide(ma, mb)
        rel = delta / ma
        Xa, Xb = pop.design(cls, a, **{k: v for k, v in kw.items() if k != "mask"}), pop.design(cls, b, **{k: v for k, v in kw.items() if k != "mask"})
        y0 = pop.Yd[cls][0]
        r2a, r2b, cond = r2(Xa, y0), r2(Xb, y0), np.linalg.cond(Xb)
        added = [x for x in b if x not in a]
        cf = [pop.coef(cls, b, b.index(x) + 1, **kw) for x in added] if added else []
        cft = "; ".join(f"{x} {fmt(c)}" for x, c in zip(added, cf))
        fam = "primary" if cid in PRIMARY else "secondary"
        extra = f"; Bonferroni 98.75% {fmt(delta, 3, 0.625, 99.375)}" if cid in PRIMARY else ""
        lines.append(f"| {cid} | {fam} | {cls if cls == 'L' else f'Δ_pass({cls})'} | {' + '.join(a) or 'null'} → "
                     f"{' + '.join(b)} | {ma[0]:.3f} | {mb[0]:.3f} | {fmt(delta)}{extra} | {rel[0]:+.3f} "
                     f"[{ci(rel, 5, 95)[0]:+.3f}, {ci(rel, 5, 95)[1]:+.3f}] | **{d}**{rankflag} | {cft} | {r2a:.3f} / {r2b:.3f} | "
                     f"{cond:.0f} | {qtxt} |")
        rec = {"decision": d, "mse_a": ma[0], "mse_b": mb[0], "delta": [delta[0], *ci(delta)], "r2": [r2a, r2b],
               "cond_b": cond, "rank_deficient": bool(RANK_DEFICIENT),
               "coef": {x: [c[0], *ci(c)] for x, c in zip(added, cf)}}
        if cid in SIGN and len(added) == 1:
            c = cf[0]
            sign_ok = (ci(c)[1] < 0) if SIGN[cid] < 0 else (ci(c)[0] > 0)
            rec["direction_as_predicted"] = bool(sign_ok)
            rec["prediction_holds"] = bool(d.startswith("b better") and sign_ok)
        out[cid] = rec
        rows.append({"population": pop.name + tag, "id": cid, "mse_a": ma[0], "mse_b": mb[0], "delta": delta[0],
                     "lo": ci(delta)[0], "hi": ci(delta)[1], "decision": d})
    return out


def main(args):
    C = pd.read_csv(args.counts, keep_default_na=False)
    C["count"] = C["count"].astype(float)
    pre = pd.read_csv(Path(args.d11_dir) / "prefixes.csv", keep_default_na=False)
    Vd = pd.read_csv(Path(args.d11_dir) / "verbs.csv", keep_default_na=False)
    Pd = pd.read_csv(Path(args.d11_dir) / "pairs.csv", keep_default_na=False)
    Vc = pd.read_csv(Path(args.cur_dir) / "verbs.csv", keep_default_na=False)
    Vc = Vc[Vc.curated == 1]
    Pc = pd.read_csv(Path(args.cur_dir) / "pairs.csv", keep_default_na=False)
    Sd, Sc = pd.read_parquet(args.d11_scores), pd.read_parquet(args.cur_scores)
    # declared order: the D3 / precision audits are labelled before the outcome analysis
    dpath, lpath = Path(args.diag_contexts), Path(args.diag_labels)
    audit = None
    if dpath.exists() and lpath.exists():
        Dc = pd.read_csv(dpath, keep_default_na=False)
        Lb = pd.read_csv(lpath, keep_default_na=False)
        audit = Dc.merge(Lb[["audit", "verb", "variant", "k", "label"]], on=["audit", "verb", "variant", "k"], how="left")
        ok = {"D3": {"verbal", "adjectival", "nominal", "other"}, "precision": {"finite active", "other"}}
        bad = audit[[l not in ok[a] for a, l in zip(audit.audit, audit.label.fillna(""))]]
        assert bad.empty, f"{len(bad)} audit rows unlabelled or with an unknown label"
    else:
        assert args.allow_missing_audit, "D3 / precision audit not finished"
    m = Sd[Sd.set == "freq"].merge(Sc[Sc.set == "freq"], on=["context_id", "frame", "verb"], suffixes=("", "_r5"))
    rep = float((m.lp - m.lp_r5).abs().max())
    assert rep <= 1e-3, f"D11 vs round-5 scores differ by {rep}"
    lines = ["# Round 6, A1: Pile exposure and the frame contrast", "",
             "Spec: `plan.md`, A1. Counts: `fetch_pile_counts_r6.py` (infini-gram `v4_piletrain_llama`); analysis: "
             "`analyze_exposure.py`. Outcomes: Pythia-1.4B Δ_pass(V) (24 freq frames) and L. Reproduction of round-5 "
             f"scores by the D11 run (shared verbs, freq): max |Δ lp| {rep:.1e} ({len(m)} rows). CV: frozen fold map "
             "(pairs shuffled within band, seed 17); 95% CIs from pairs within band × contexts (2,000 draws). Decision: "
             "b better = 95% CI of Δ = MSE_a − MSE_b above 0; a better = below 0; equivalent = 90% CI of Δ / MSE_a within "
             "±2%; else unresolved. A direction prediction holds only with 'b better' and the predicted coefficient sign "
             "(95% CI excluding 0). Primary family: I1, I3, T2, T3 (Bonferroni 98.75% CIs beside).", ""]
    dec, rows = {"reproduction_max_abs": rep}, []
    Ks = {sm: build_counts(C, sm) for sm in (args.smooth, *args.smooth_sens)}
    K = Ks[args.smooth]
    allv = sorted(set(Vd.verb) | set(Vc.verb))
    assert set(allv) == set(K.index), "count verbs differ from the inventory"
    meta_v = pd.concat([Vd[["verb", "role", "band", "zipf", "n_tok"]], Vc[["verb", "role", "band", "zipf", "n_tok"]]]) \
        .drop_duplicates("verb").set_index("verb")
    K = K.join(meta_v)

    # counts overview
    lines += ["## Counts", "", "c_total = ▁lower + ▁Cap + after-newline lower + after-newline Cap (boundary-safe). Medians:", "",
              "| class | band | n | c_total | share ▁lower | share ▁Cap | share after newline | upper / c_total | c_be | c_perf | "
              "c_comp (anchored) | c_comp (broad) | share c_be = 0 | share c_comp = 0 |", "|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---|"]
    for (r, b), g in K.groupby(["role", "band"]):
        lines.append(f"| {r} | {NAME[b]} | {len(g)} | {g.c_total.median():,.0f} | {(g.c_low / g.c_total).median():.3f} | "
                     f"{(g.c_cap / g.c_total).median():.3f} | {((g.c_nl_low + g.c_nl_cap) / g.c_total).median():.3f} | "
                     f"{(g.c_upper / g.c_total).median():.2f} | {g.c_be.median():,.0f} | {g.c_perf.median():,.0f} | "
                     f"{g.c_comp.median():,.0f} | {g.c_comp_broad.median():,.0f} | {(g.c_be == 0).mean():.2f} | {(g.c_comp == 0).mean():.2f} |")
    inflate = K[K.c_upper / K.c_total > 1.5].index.tolist()
    lines += ["", f"Forms whose upper-bound count exceeds c_total by > 50% (likely word-internal matches of the unmarked "
              f"sequence): {len(inflate)}" + (f" ({', '.join(inflate[:40])}{'…' if len(inflate) > 40 else ''})" if inflate else "") + ".", ""]

    # populations
    folds = {"D11": fold_map(Pd.band.to_numpy(), args.seed), "curated": fold_map(Pc.band.to_numpy(), args.seed)}
    pops = {}
    ntok = meta_v.n_tok
    for name, P, S in (("D11", Pd, Sd), ("curated", Pc, Sc)):
        verbs = list(P.t_verb) + list(P.i_verb)
        Y = outcome(S, pre, verbs, args.n_boot, args.seed)
        wp = weights(np.random.default_rng(args.seed + 1), P.band.to_numpy(), args.n_boot)
        n = len(P)
        pops[name] = Pop(name, P, K, Y[:, :n], Y[:, n:], wp, folds[name], args.smooth, ntok)

    for name in ("D11", "curated"):
        pop = pops[name]
        lines += [f"## Model comparison: {name} ({len(pop.pairs)} pairs)", ""]
        for cls in ("T", "I"):
            X = pop.X[cls]
            r = {p: np.corrcoef(X[p], pop.Yd[cls][0])[0, 1] for p in ("Z", "C", "B", "P", "K", "K_broad", "S", "S_fisher", "S_washas", "N")}
            lines.append(f"- r(Δ_pass({cls}), ·): " + ", ".join(f"{p} {v:+.2f}" for p, v in r.items())
                         + f"; r(C, K) {np.corrcoef(X.C, X.K)[0, 1]:+.2f}; cond(C + K) "
                         f"{np.linalg.cond(pop.design(cls, ['C', 'K'])):.0f}")
        lines.append("")
        dec[name] = {"main": run_comparisons(pop, lines, rows)}
        lines.append("")
        sens_ids = ("I1", "I3", "I5", "T2", "T3", "L1", "L2a", "L2b", "L2")  # I5 on curated = D1's sensitivities
        variants = [("token count added (log participle tokens)", dict(add_tok=True)),
                    ("c_total marked only", dict(swap={"C": "C_marked"})),
                    ("c_total upper bound", dict(swap={"C": "C_upper"})),
                    ("K = broad pronoun adjacency", dict(swap={"K": "K_broad"})),
                    ("S = Fisher-based", dict(swap={"S": "S_fisher"})),
                    ("S = BE vs pronoun frame", dict(swap={"S": "S_comp"})),
                    ("S = was vs has", dict(swap={"S": "S_washas"}))]
        dec[name]["sensitivity"] = {}
        for lab, kw in variants:
            lines += [f"Sensitivity ({name}): {lab}", ""]
            dec[name]["sensitivity"][lab] = run_comparisons(pop, lines, rows, tag=f"|{lab}", ids=sens_ids, **kw)
            lines.append("")
        for sm in args.smooth_sens:
            ps = Pop(name, pop.pairs, Ks[sm].join(meta_v), pop.Yd["T"], pop.Yd["I"], pop.wp, pop.folds, sm, ntok)
            lines += [f"Sensitivity ({name}): smoothing {sm}", ""]
            dec[name]["sensitivity"][f"smoothing {sm}"] = run_comparisons(ps, lines, rows, tag=f"|s={sm}", ids=sens_ids)
            lines.append("")
        ma, mb = pop.cv("T", ["C"]), pop.cv("T", ["C", "B", "P"])
        d, delta = decide(ma, mb)
        cb, cp = pop.coef("T", ["C", "B", "P"], 2), pop.coef("T", ["C", "B", "P"], 3)
        lines += [f"- T3 decomposition ({name}): C → C + B + P: Δ {fmt(delta)} → {d}; B coefficient {fmt(cb)}, P coefficient "
                  f"{fmt(cp)} (a preference effect from more BE has B > 0; from less HAVE, P < 0).", ""]
        dec[name]["T3_decomposition"] = {"decision": d, "B": [cb[0], *ci(cb)], "P": [cp[0], *ci(cp)]}
        if name == "D11":
            lines += ["### Single-token subset (a_V = Δ_pass + a verb-independent constant; descriptive)", ""]
            dec[name]["single_token"] = {}
            for cid, cls, a, b, _ in COMPARISONS:
                if cid not in PRIMARY:
                    continue
                msk = (pop.pairs[f"{cls.lower()}_ntok"] == 1).to_numpy()
                ma, mb = pop.cv(cls, a, msk), pop.cv(cls, b, msk)
                d, delta = decide(ma, mb)
                c = pop.coef(cls, b, len(b), msk)
                dec[name]["single_token"][cid] = d
                lines.append(f"- {cid} ({int(msk.sum())} single-token {cls} verbs): MSE {ma[0]:.3f} → {mb[0]:.3f}, Δ {fmt(delta)} "
                             f"→ {d}; added coefficient {fmt(c)}.")
            lines.append("")

    # re-binning by counts (D11)
    pop = pops["D11"]
    lines += ["## Re-binning by Pile counts (D11)", "",
              "Within each class, verbs ranked by c_total (descending; ties alphabetical) and cut into bins of the band "
              "sizes (top 47 = bin 1, next 132 = bin 2, lowest 102 = bin 3); pairs by mean log c_total the same way. Means "
              "with 95% CIs (pairs within band × contexts). Rejection gradient in −Δ_pass(I) units.", ""]

    def bins_of(values, labels_):
        order = sorted(range(len(values)), key=lambda j: (-values[j], labels_[j]))
        rank = np.empty(len(values), int)
        rank[order] = np.arange(len(values))
        return np.where(rank < 47, "bin1", np.where(rank < 179, "bin2", "bin3"))

    bins = {cls: bins_of(pop.X[cls].C.to_numpy(), list(pop.pairs[f"{cls.lower()}_verb"])) for cls in ("T", "I")}
    bins["L"] = bins_of(((pop.X["T"].C + pop.X["I"].C) / 2).to_numpy(), list(pop.pairs.verb_pair))
    band = pop.pairs.band.to_numpy()
    wp = pop.wp
    pm = lambda Y, mk: (wp[:, mk] * Y[:, mk]).sum(1) / wp[:, mk].sum(1)
    lines += ["| outcome | grouping | 1 (Head / top) | 2 (Tail / middle) | 3 (XTail / bottom) | 1 − 3 |", "|---|---|---|---|---|---|"]
    grad = {}
    for cls in ("T", "I", "L"):
        Y = pop.Yd[cls]
        for gname, g, labs in (("Zipf band", band, BANDS), ("count bin", bins[cls], ("bin1", "bin2", "bin3"))):
            mm = [pm(Y, g == l) for l in labs]
            grad[(cls, gname)] = mm[0] - mm[2]
            lines.append(f"| {cls if cls == 'L' else f'Δ_pass({cls})'} | {gname} | " + " | ".join(fmt(x, 2) for x in mm)
                         + f" | {fmt(mm[0] - mm[2], 2)} |")
            rows.append({"population": "D11", "id": f"bins_{cls}_{gname}", "delta": (mm[0] - mm[2])[0],
                         "lo": ci(mm[0] - mm[2])[0], "hi": ci(mm[0] - mm[2])[1]})
    rej_bin, rej_band = -grad[("I", "count bin")], -grad[("I", "Zipf band")]
    lines += ["", f"Intransitive rejection gradient (top − bottom, −Δ_pass(I)): count bins {fmt(rej_bin, 2)}, Zipf bands "
              f"{fmt(rej_band, 2)}; difference {fmt(rej_bin - rej_band, 2)}."]
    dec["bins"] = {"rejection_count_bins": [rej_bin[0], *ci(rej_bin)], "rejection_zipf_bands": [rej_band[0], *ci(rej_band)],
                   "count_ge_zipf_point": bool(rej_bin[0] >= rej_band[0])}
    xt = pd.crosstab(pd.Series(band, name="band"), pd.Series(bins["L"], name="bin"))
    lines += ["", "Pairs, Zipf band × count bin: " + "; ".join(f"{NAME[b]} → " + ", ".join(f"{c} {int(xt.loc[b, c])}"
                                                                                      for c in xt.columns) for b in BANDS), "",
              "B1 (CV, secondary): Zipf-band dummies → count-bin dummies:", ""]
    dec["B1"] = {}
    for cls in ("T", "I", "L"):
        Y = pop.Yd[cls]
        Xa = np.column_stack([np.ones(len(band)), band == "tail", band == "xtail"]).astype(float)
        g = bins[cls]
        Xb = np.column_stack([np.ones(len(band)), g == "bin2", g == "bin3"]).astype(float)
        ma = np.array([cv_mse(Xa, Y[b], wp[b], pop.folds) for b in range(len(Y))])
        mb = np.array([cv_mse(Xb, Y[b], wp[b], pop.folds) for b in range(len(Y))])
        d, delta = decide(ma, mb)
        dec["B1"][cls] = d
        lines.append(f"- {cls if cls == 'L' else f'Δ_pass({cls})'}: MSE {ma[0]:.3f} → {mb[0]:.3f}, Δ {fmt(delta)} → **{d}**.")
    lines.append("")

    # diagnosis of the curated-I case
    cur = pops["curated"]
    lines += ["## Diagnosis: curated intransitives (round 5: Zipf beat Pile log c)", ""]
    r5 = pd.read_csv(args.r5_counts)
    r5c = np.log10(r5[r5.slot == "overall"].set_index("verb")["count"].loc[cur.pairs.i_verb].to_numpy() + args.smooth)
    Y = cur.Yd["I"]
    mse = lambda X: np.array([cv_mse(X, Y[b], cur.wp[b], cur.folds) for b in range(len(Y))])
    mz = mse(cur.design("I", ["Z"]))
    d1 = {}
    for lab, X in (("c_total", cur.design("I", ["C"])), ("marked only", cur.design("I", ["C_marked"])),
                   ("upper bound", cur.design("I", ["C_upper"])), ("round-5 lowercase only",
                                                                  np.column_stack([np.ones(len(r5c)), r5c]))):
        mc = mse(X)
        d, delta = decide(mz, mc)
        d1[lab] = {"mse": mc[0], "zipf_advantage": (mc - mz)[0], "decision": d}
        lines.append(f"- D1, {lab}: Pile MSE {mc[0]:.3f} vs Zipf {mz[0]:.3f}; Zipf advantage MSE_C − MSE_Z "
                     f"{fmt(mc - mz)} → {d} (a = Zipf, b = Pile).")
    shrink = d1["c_total"]["zipf_advantage"] / d1["round-5 lowercase only"]["zipf_advantage"]
    lines += [f"- Zipf advantage with c_total relative to round 5's count: {shrink:.2f} (prediction: ≤ 0.5 and ≥ 0).", ""]
    dec["D1"] = {**d1, "advantage_ratio": shrink}
    allz = meta_v.zipf.loc[allv]
    lc = np.log10(K.loc[allv].c_total + args.smooth)
    b = np.polyfit(allz.to_numpy(float), lc.to_numpy(), 1)
    resid = pd.Series(lc.to_numpy() - np.polyval(b, allz.to_numpy(float)), index=allv)
    iv = cur.pairs.i_verb.to_numpy()
    one = np.ones(len(iv))
    D2 = pd.DataFrame({"verb": iv, "band": cur.pairs.band, "zipf": cur.X["I"].Z, "log_c_total": cur.X["I"].C,
                       "resid": resid.loc[iv].to_numpy(), "delta_pass": Y[0],
                       "pred_zipf": cv_pred(cur.design("I", ["Z"]), Y[0], one, cur.folds),
                       "pred_pile": cv_pred(cur.design("I", ["C"]), Y[0], one, cur.folds)})
    D2["closer"] = np.where((D2.delta_pass - D2.pred_zipf).abs() < (D2.delta_pass - D2.pred_pile).abs(), "Zipf", "Pile")
    top = pd.concat([D2.nlargest(10, "resid").assign(direction="Pile > Zipf"),
                     D2.nsmallest(10, "resid").assign(direction="Pile < Zipf")])
    lines += [f"- D2: residual of log10 c_total on Zipf over the 608 forms (slope {b[0]:.2f}); the 10 curated I verbs "
              "with the largest positive and negative residuals:", "",
              "| direction | verb | band | Zipf | log10 c_total | residual | Δ_pass(I) | CV pred. Zipf | CV pred. Pile | closer |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in top.itertuples(index=False):
        lines.append(f"| {r.direction} | {r.verb} | {NAME[r.band]} | {r.zipf:.2f} | {r.log_c_total:.2f} | {r.resid:+.2f} | "
                     f"{r.delta_pass:+.2f} | {r.pred_zipf:+.2f} | {r.pred_pile:+.2f} | {r.closer} |")
    se = lambda sub, col: float(((sub.delta_pass - sub[col]) ** 2).mean())
    rest = D2[~D2.verb.isin(top.verb)]
    lines += ["", f"Mean squared error on these 20 verbs: Zipf {se(top, 'pred_zipf'):.2f}, Pile {se(top, 'pred_pile'):.2f}; on the "
              f"other {len(rest)}: Zipf {se(rest, 'pred_zipf'):.2f}, Pile {se(rest, 'pred_pile'):.2f}.", ""]
    if audit is not None:
        Dc = audit
        d3 = Dc[Dc.audit == "D3"]
        lines += ["- D3: Pile subsets and hand-labelled uses of sampled occurrences (variants in proportion to their counts):", "",
                  "| group | verbs | occurrences | top Pile subsets (share) | verbal | adjectival | nominal / name | other |",
                  "|---|---:|---:|---|---|---|---|---|"]
        for gname, g in d3.groupby("group"):
            subs = g.pile_set.value_counts(normalize=True).head(4)
            lab = g.label.value_counts(normalize=True)
            lines.append(f"| {gname} | {g.verb.nunique()} | {len(g)} | " + ", ".join(f"{k} {v:.2f}" for k, v in subs.items())
                         + " | " + " | ".join(f"{lab.get(x, 0):.2f}" for x in ("verbal", "adjectival", "nominal", "other")) + " |")
        pa = Dc[Dc.audit == "precision"]
        if len(pa):
            lines += ["", "- Precision of the broad pronoun proxy (share labelled finite active): "
                      + ", ".join(f"{v} ({r}) {g.label.eq('finite active').mean():.2f} of {len(g)}"
                                  for (v, r), g in pa.groupby(["verb", "group"])) + "."]
        lines.append("")
    else:
        lines += ["- D3: audit files not present.", ""]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    K.to_csv(out / "a1_verbs.csv")
    D2.to_csv(out / "a1_diag_curated_I.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "a1_summary.csv", index=False)
    (out / "a1_decisions.json").write_text(json.dumps(dec, indent=1, default=float))
    Path(args.report).write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--counts", default="data/round6/counts/pile_counts.csv")
    ap.add_argument("--d11-dir", default="data/round6/frame_d11")
    ap.add_argument("--cur-dir", default="data/round5/frame")
    ap.add_argument("--d11-scores", default="results/round6/frame_d11/pythia-1.4b/frame_scores.parquet")
    ap.add_argument("--cur-scores", default="results/round5/frame/frame_scores.parquet")
    ap.add_argument("--r5-counts", default="data/round5/counts/pile_counts.csv")
    ap.add_argument("--diag-contexts", default="data/round6/counts/diag_contexts.csv")
    ap.add_argument("--diag-labels", default="data/round6/counts/diag_labels.csv")
    ap.add_argument("--allow-missing-audit", action="store_true", help="testing only")
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--smooth-sens", type=float, nargs="*", default=[0.1, 1.0])
    ap.add_argument("--out-dir", default="results/round6/exposure")
    ap.add_argument("--report", default="reports/round6/a1_exposure.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    main(ap.parse_args())
