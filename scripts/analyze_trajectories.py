#!/usr/bin/env python3
"""Round 7, A8: per-verb training trajectories (reports/round7/plan.md, A8). A reanalysis of A6's checkpoint scores; no
model. Needs scipy (censored regressions) and matplotlib (figures).

Inputs: results/round6/checkpoints/pythia-<size>/step<N>/{frame_scores.parquet, frame_h.npz};
data/round6/frame_d11_freq/{prefixes,verbs,pairs}.csv; A1's counts (data/round6/counts/pile_counts.csv).
Per-verb Delta_pass at every checkpoint = mean over the 24 freq contexts. Reliability: 200 animacy-stratified 12/12 context
splits, Spearman-Brown; a checkpoint with r_SB < 0.7 (or undefined) is flagged.
1: onset = first checkpoint with Delta_pass(I) < 0 there and at the next two checkpoints (only checkpoints with two
followers can qualify), interpolated in log10 step; left / right censoring; log10 onset ~ log10 c_total + log n_tok on
crossers; sensitivities (lenient end, no persistence, threshold -1, no token covariate, Tobit, maturity-floor Tobit,
checkpoint-interval censoring). 2: late window (steps >= 16000): between-verb slope of the window mean on log10 c_total
vs within-verb slope on log10 E (verb fixed effects) and the log c x log step interaction; calibration against a common
nonlinear exposure curve (natural spline in log10 E, plug-in). 3: exposure-aligned bin curves (c_total quintiles; Zipf
bands) on a 0.1-decade log10 E grid, D(E) = rarer - more frequent bin, crossing test with pointwise and simultaneous
(sup-t) bands; per-verb own-exposure alignment as a qualifier; pooled D vs A6's matched-exposure diagnostic.
4: within-band T vs I AUC and mean L per checkpoint; a_V by band x class (single-token verbs).
Exposure E = (c_total + 0.5) x step / 143000 (A6). Bootstrap: A6's draws (pairs within band x nouns within animacy,
2,000 draws, seed 17, drawn in A6's order); draw 0 = the point estimate; fits that fail in a draw are dropped from that
fit's CI and counted.
Outputs: reports/round7/a8_trajectories.md; results/round7/trajectories/a8_{decisions.json,summary,onsets,reliability,
dgrid,class}.csv; reports/round7/figures/a8_f{1,2,3}_*.{png,pdf}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import norm

from analyze_checkpoints import ns_basis
from analyze_frame_contrast import weights

BANDS = ("head", "tail", "xtail")
NAME = {"head": "Head", "tail": "Tail", "xtail": "XTail"}
TOTAL = 143000
DECLARED = {"1.4b": [64, 128, 256, 512, 1000, 2000, 3000, 4000, 6000, 10000, 16000, 25000, 40000, 63000, 100000, 143000],
            "410m": [64, 128, 256, 512, 1000, 2000, 3000, 4000, 6000, 10000, 16000, 25000, 40000, 63000, 100000, 143000],
            "6.9b": [512, 2000, 8000, 23000, 63000, 143000]}
A6_DIAG = {"1.4b": 0.080, "410m": 0.133, "6.9b": 0.179}  # A6's reported matched-exposure diagnostic
FLAG_R = 0.7
A1_SLOPE = -0.728  # A1 I1, D11, final 1.4B: Delta_pass(I) on log10 c_total
# onset definitions: (threshold, persistence, strict end = only checkpoints with `persistence` followers can qualify)
ONSET_DEFS = {"primary": (0.0, 2, True), "lenient end": (0.0, 2, False), "no persistence": (0.0, 0, False),
              "threshold −1": (-1.0, 2, True)}
LATE_FROM = 16000
GRID = 0.1
MATURE = 512  # part 3's cut: before step 512 the Head AUC is 0.54-0.69
COVER = 0.5   # per-verb alignment: a grid point belongs to a bin if at least half its verbs cover it


# ---------------------------------------------------------------- loading

def load_ctx(sdir, pre, P):
    """Per-context Delta_pass [pair, context] for the T and the I verb (point values; A6's pivot)."""
    S = pd.read_parquet(sdir / "frame_scores.parquet")
    ctx = list(pre.context_id)
    s = S[S.set == "freq"]
    lp = s.pivot_table(index=["verb", "frame"], columns="context_id", values="lp", aggfunc="first")[ctx]
    assert lp.notna().all().all() and len(lp) == 4 * len(P), "missing scores"
    D = lp.xs("pass", level="frame") - lp.xs("act", level="frame")
    return D.loc[P.t_verb].to_numpy(float), D.loc[P.i_verb].to_numpy(float)


def load_av(sdir, pre, ids, wn):
    """[draw, verb] a_V = u_V1 . dh-bar (noun-weighted mean of h_pass - h_act over contexts)."""
    z = np.load(sdir / "frame_h.npz")
    keys = list(z["keys"])
    H, U = z["H"].astype(np.float64), z["U"].astype(np.float64)
    urow = {int(t): i for i, t in enumerate(z["U_ids"])}
    dh = np.stack([H[keys.index(f"freq|{c}|pass")] - H[keys.index(f"freq|{c}|act")] for c in pre.context_id])
    dbar = (wn @ dh) / wn.sum(1, keepdims=True)
    return dbar @ np.stack([U[urow[int(t)]] for t in ids]).T


# ---------------------------------------------------------------- helpers

def onset(Y, logt, thr=0.0, persist=2, strict=True):
    """Y [units, steps]. First step k with Y < thr there and at the next `persist` steps; with `strict`, only steps with
    `persist` followers can qualify (else the condition uses the followers that exist). log10 onset interpolated linearly
    in log10 step between steps k - 1 and k at Y = thr. cens: -1 left (k = 0), +1 right (no qualifying k). Returns
    (log10 onset, cens, k, right-censoring bound)."""
    neg = Y < thr
    S = Y.shape[1]
    q = neg.copy()
    for d in range(1, persist + 1):
        q[:, :S - d] &= neg[:, d:]
    if strict and persist:
        q[:, S - persist:] = False
    anyq = q.any(1)
    k = np.where(anyq, q.argmax(1), -1)
    cens = np.where(~anyq, 1, np.where(k == 0, -1, 0))
    kk = np.clip(k, 1, S - 1)
    r = np.arange(len(Y))
    y0, y1 = Y[r, kk - 1] - thr, Y[r, kk] - thr
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = logt[kk - 1] + y0 / (y0 - y1) * (logt[kk] - logt[kk - 1])
    bound = logt[S - 1 - persist] if strict and persist else logt[S - 1]
    return np.where(cens == 0, lo, np.nan), cens, k, bound


def wls(X, y, w):
    """Weighted least squares; NaN when the design is not identified on the units with positive weight."""
    m = w > 0
    if m.sum() < X.shape[1] + 2:
        return np.full(X.shape[1], np.nan)
    sw = np.sqrt(w[m])
    A = X[m] * sw[:, None]
    if np.linalg.matrix_rank(A) < X.shape[1]:
        return np.full(X.shape[1], np.nan)
    return np.linalg.lstsq(A, y[m] * sw, rcond=None)[0]


def _logdiff(a, b):
    """log(Phi(b) - Phi(a)) for b > a, via the upper tail when a > 0."""
    pos = a > 0
    la = np.where(pos, log_ndtr(-b), log_ndtr(a))
    lb = np.where(pos, log_ndtr(-a), log_ndtr(b))
    return lb + np.log1p(-np.exp(la - lb))


def cnreg(X, w, kind, y, l, u, start):
    """Weighted ML censored-normal regression. kind 0: exact y; 1: left-censored (latent <= u); 2: right-censored
    (latent > l); 3: interval (l < latent <= u). Returns (params [b..., log s], converged)."""
    k = X.shape[1]
    pos, info = w > 0, (w > 0) & ((kind == 0) | (kind == 3))
    if (pos.sum() < k + 2 or info.sum() < k + 1 or np.linalg.matrix_rank(X[pos]) < k
            or np.linalg.matrix_rank(X[info]) < k):
        return np.full(k + 1, np.nan), False  # not identified (too few uncensored units or a rank-deficient design)
    k0, k1, k2, k3 = (kind == i for i in range(4))
    y, l, u = np.nan_to_num(y), np.nan_to_num(l), np.nan_to_num(u)

    def f(p):
        b, ls = p[:-1], p[-1]
        s = np.exp(ls)
        mu = X @ b
        ll, gm, gs = np.zeros(len(mu)), np.zeros(len(mu)), np.zeros(len(mu))
        z = (y[k0] - mu[k0]) / s
        ll[k0], gm[k0], gs[k0] = -0.5 * z ** 2 - ls - 0.5 * np.log(2 * np.pi), z / s, z ** 2 - 1
        zu = (u[k1] - mu[k1]) / s
        lc = log_ndtr(zu)
        r = np.exp(norm.logpdf(zu) - lc)
        ll[k1], gm[k1], gs[k1] = lc, -r / s, -r * zu
        zl = (l[k2] - mu[k2]) / s
        lc = log_ndtr(-zl)
        r = np.exp(norm.logpdf(zl) - lc)
        ll[k2], gm[k2], gs[k2] = lc, r / s, r * zl
        zl, zu = (l[k3] - mu[k3]) / s, (u[k3] - mu[k3]) / s
        lp = _logdiff(zl, zu)
        pl, pu = np.exp(norm.logpdf(zl) - lp), np.exp(norm.logpdf(zu) - lp)
        ll[k3], gm[k3], gs[k3] = lp, (pl - pu) / s, pl * zl - pu * zu
        return -(w * ll).sum(), -np.append(X.T @ (w * gm), (w * gs).sum())

    for method in ("L-BFGS-B", "BFGS"):
        res = minimize(f, start, jac=True, method=method)
        ok = np.isfinite(res.fun) and np.all(np.isfinite(res.x)) and 1e-4 < np.exp(res.x[-1]) < 1e3 and np.all(np.abs(res.x[:-1]) < 1e3)
        if ok and (res.success or np.max(np.abs(res.jac)) < 1e-4):
            return res.x, True
    return np.full(len(start), np.nan), False


def ci(d, lo=2.5, hi=97.5):
    """Percentile interval over draws 1.. (non-finite draws dropped, as analyze_frame_contrast.ci); NaN if none is finite."""
    d = np.asarray(d, float)[1:]
    d = d[np.isfinite(d)]
    return (float(np.percentile(d, lo)), float(np.percentile(d, hi))) if len(d) else (np.nan, np.nan)


def onset_label(lo, hi):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "unestimable"
    if -1 < lo and hi < 0:
        return "held"
    if lo > 0:
        return "failed, reversed"
    if hi < -1:
        return "failed, steeper than count equivalence"
    inc1, inc0 = lo <= -1 <= hi, lo <= 0 <= hi
    if inc1 and inc0:
        return "unresolved"
    return "failed (consistent with count equivalence)" if inc1 else "no frequency effect on onset"


def f3(d, p=2):
    lo, hi = ci(d)
    return f"{d[0]:+.{p}f} [{lo:+.{p}f}, {hi:+.{p}f}]"


def wauc(Tv, Iv, wt, wi, chunk=250):
    """Weighted AUC per draw: sum_ij wt_i wi_j ([T_i > I_j] + 0.5 [T_i = I_j]) / (sum wt sum wi)."""
    out = np.empty(len(Tv))
    for a in range(0, len(Tv), chunk):
        T, I = Tv[a:a + chunk, :, None], Iv[a:a + chunk, None, :]
        c = (T > I).astype(np.float32) + 0.5 * (T == I).astype(np.float32)
        out[a:a + chunk] = np.einsum("di,dij,dj->d", wt[a:a + chunk], c, wi[a:a + chunk]) / (
            wt[a:a + chunk].sum(1) * wi[a:a + chunk].sum(1))
    return out


def stable_from(flags):
    """First index after which every later entry is True (None if the last is False)."""
    return next((i for i in range(len(flags)) if all(flags[i:])), None)


def late_stats(Y, w, x, lnt, sw_):
    """Late-window slopes for one draw. Y [verb, window step]; returns beta_b, beta_b (+ log n_tok), beta_w, gamma, and
    the time slope at the inventory mean of x. beta_w = weighted mean of per-verb slopes on log10 step (= the FE slope:
    every verb has the same steps); gamma = slope of the per-verb slopes on x (= the FE interaction)."""
    ybar = Y.mean(1)
    qv = ((Y - ybar[:, None]) * sw_).sum(1) / (sw_ ** 2).sum()
    n = len(x)
    bb = wls(np.column_stack([np.ones(n), x]), ybar, w)[1]
    bbt = wls(np.column_stack([np.ones(n), x, lnt]), ybar, w)[1]
    bw = (w * qv).sum() / w.sum()
    bt_, gam = wls(np.column_stack([np.ones(n), x - x.mean()]), qv, w)
    return bb, bbt, bw, gam, bt_


def crossing_reading(neg, pos):
    """(label, a crossing in the predicted order exists: a resolved negative point at lower E than a resolved positive)."""
    g = len(neg)
    po = any(neg[i] and pos[j] for i in range(g) for j in range(i + 1, g))
    ro = any(pos[i] and neg[j] for i in range(g) for j in range(i + 1, g))
    if neg.any() and pos.any():
        return "crossing (" + ("predicted order" if po and not ro else "reversed order" if ro and not po
                               else "non-monotonic: both orders") + ")", po
    return ("only negative where resolved" if neg.any() else "only positive where resolved" if pos.any()
            else "none resolved"), False


def compare(D, G):
    """Pointwise and simultaneous (sup-t) readings of D [draw, grid]."""
    Db = D[1:]
    with np.errstate(invalid="ignore"):
        lo_c, hi_c = np.nanpercentile(Db, 2.5, 0), np.nanpercentile(Db, 97.5, 0)
        sd = np.nanstd(Db, 0)
    full = np.isfinite(Db).all(1)  # sup-t over draws with every grid point defined
    dev = np.where(sd > 0, np.abs(Db[full] - D[0]) / np.where(sd > 0, sd, 1), 0.0)
    crit = float(np.percentile(dev.max(1), 95)) if full.any() else np.nan
    slo, shi = D[0] - crit * sd, D[0] + crit * sd
    rd, po = crossing_reading(hi_c < 0, lo_c > 0)
    rds, pos_ = crossing_reading(shi < 0, slo > 0)
    if len(G) == 1:
        rd, rds, po, pos_ = "insufficient overlap (one grid point)", "—", False, False
    d0 = D[0]
    nz = np.flatnonzero(np.isfinite(d0) & (d0 != 0))
    sc = []
    for i, j in zip(nz[:-1], nz[1:]):
        if np.sign(d0[i]) != np.sign(d0[j]):  # adjacent: interpolate; separated by exact zeros / undefined points: their midpoint
            sc.append(G[i] + d0[i] / (d0[i] - d0[j]) * (G[j] - G[i]) if j == i + 1 else float(np.mean(G[i + 1:j])))
    return dict(lo=lo_c, hi=hi_c, slo=slo, shi=shi, crit=crit, reading=rd, po=po, reading_sim=rds, po_sim=pos_, cross=sc,
                d0=d0, incomplete=int((~full).sum()))


# ---------------------------------------------------------------- main

def main(args):
    data = Path(args.data_dir)
    pre = pd.read_csv(data / "prefixes.csv", keep_default_na=False)
    V = pd.read_csv(data / "verbs.csv", keep_default_na=False)
    P = pd.read_csv(data / "pairs.csv", keep_default_na=False)
    P["cell"] = [f"{a}×{b}" for a, b in zip(P.t_ntok, P.i_ntok)]
    assert P.t_verb.is_unique and P.i_verb.is_unique and not set(P.t_verb) & set(P.i_verb)
    band, split, n = P.band.to_numpy(), P.split.to_numpy(), len(P)
    C = pd.read_csv(args.counts, keep_default_na=False)
    C["count"] = C["count"].astype(float)
    Wc = C.pivot(index="verb", columns="slot", values="count")
    ctot = Wc.overall + Wc.overall_cap + Wc.nl_low + Wc.nl_cap
    x = np.log10(ctot.loc[P.i_verb].to_numpy() + 0.5)
    lnt = np.log(V.set_index("verb").n_tok.loc[P.i_verb].to_numpy(float))
    # bins: c_total quintiles (x ascending, ties by form), fixed on the inventory; Zipf bands (rarer first)
    asc = sorted(range(n), key=lambda i: (x[i], P.i_verb[i]))
    quint = np.empty(n, int)
    for q, idx in enumerate(np.array_split(np.array(asc), 5)):
        quint[idx] = q
    binsets = {"quintile": ([f"Q{q + 1}" for q in range(5)], [quint == q for q in range(5)]),
               "zipf": (["XTail", "Tail", "Head"], [band == b for b in ("xtail", "tail", "head")])}
    # contexts: split halves (animacy-stratified 6 + 6), the same at every checkpoint and model
    an = pre.animacy.to_numpy()
    rs = np.random.default_rng(args.seed)
    halves = np.zeros((args.n_splits, len(pre)), bool)
    for k in range(args.n_splits):
        for a in np.unique(an):
            idx = np.flatnonzero(an == a)
            halves[k, rs.choice(idx, len(idx) // 2, replace=False)] = True
    # single-token verbs for a_V
    s1 = V[(V.n_tok == 1) & V.band.isin(["head", "tail"])].reset_index(drop=True)
    pidx = {v: i for i, v in enumerate(P.t_verb)} | {v: i for i, v in enumerate(P.i_verb)}
    s1_pair = np.array([pidx[v] for v in s1.verb])

    out, figd = Path(args.out_dir), Path(args.fig_dir)
    out.mkdir(parents=True, exist_ok=True)
    figd.mkdir(parents=True, exist_ok=True)
    lines = ["# Round 7, A8: per-verb training trajectories", "",
             "Spec: `plan.md`, A8. Reanalysis of A6's checkpoint scores (`results/round6/checkpoints/`, freq prefixes; no new "
             "runs); analysis `analyze_trajectories.py`. Per-verb Δ_pass = mean over the 24 freq contexts. Exposure E = "
             "(c_total + 0.5) × step / 143000 (A6). 95% CIs: A6's draws (pairs within band × nouns within animacy, 2,000, "
             "seed 17). † = checkpoint with split-half reliability r_SB < 0.7 (or undefined).", ""]
    dec, summ, onset_rows, rel_rows, grid_rows, class_rows = {}, [], [], [], [], []
    figdata = {}

    def srow(size, part, stat, d, label=""):
        lo, hi = ci(d)
        summ.append({"size": size, "part": part, "stat": stat, "est": float(d[0]), "lo": lo, "hi": hi, "label": label,
                     "valid_draws": int(np.isfinite(d[1:]).sum())})

    for size in args.sizes:
        root = Path(args.res_dir) / f"pythia-{size}"
        steps = np.array(DECLARED[size])
        miss = [s for s in steps if not (root / f"step{s}" / "frame_scores.parquet").exists()]
        assert not miss, f"pythia-{size}: missing checkpoints {miss}"
        S = len(steps)
        logt = np.log10(steps.astype(float))
        sv = np.log10(steps / TOTAL)
        # A6's draws, in A6's order (rng re-seeded per model; wv and wc are drawn only to keep the order)
        rng = np.random.default_rng(args.seed)
        wn = weights(rng, an, args.n_boot)
        weights(rng, np.array([f"{b}|{s_}" for b, s_ in zip(band, split)]), args.n_boot)
        weights(rng, np.array([f"{b}|{c}|{s_}" for b, c, s_ in zip(band, P.cell, split)]), args.n_boot)
        wb = weights(rng, band, args.n_boot)
        nd = len(wb)
        CT, CI_ = np.empty((S, n, len(pre))), np.empty((S, n, len(pre)))
        AV, idres = [], []
        for j, st in enumerate(steps):
            sdir = root / f"step{st}"
            CT[j], CI_[j] = load_ctx(sdir, pre, P)
            av = load_av(sdir, pre, s1.first_id, wn)
            AV.append(av)
            dp = np.where(s1.role.to_numpy() == "T", CT[j][s1_pair].mean(1), CI_[j][s1_pair].mean(1))
            dev = dp - av[0]
            idres.append(float(np.abs(dev - dev.mean()).max()))
        AV = np.stack(AV, 2)  # [draw, single-token verb, step]
        wsum = wn.sum(1)[:, None, None]
        YT = np.einsum("dc,snc->dns", wn, CT) / wsum
        YI = np.einsum("dc,snc->dns", wn, CI_) / wsum
        dz = dec[f"pythia-{size}"] = {"steps": steps.tolist()}
        fdz = figdata.setdefault(size, {})
        lines += [f"## pythia-{size}", ""]

        # ------------------------------------------------ reliability
        def sb_r(Cx, mask):
            A = np.einsum("snc,kc->ksn", Cx[:, mask], halves) / halves.sum(1)[:, None, None]
            B = np.einsum("snc,kc->ksn", Cx[:, mask], ~halves) / (~halves).sum(1)[:, None, None]
            A, B = A - A.mean(2, keepdims=True), B - B.mean(2, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = ((A * B).sum(2) / np.sqrt((A ** 2).sum(2) * (B ** 2).sum(2))).mean(0)  # NaN if a half is constant
            return r, 2 * r / (1 + r)

        allm = np.ones(n, bool)
        rI, rIsb = sb_r(CI_, allm)
        flag = ~(rIsb >= FLAG_R)  # undefined counts as flagged
        rT, rTsb = sb_r(CT, allm)
        rb = {b: sb_r(CI_, band == b)[1] for b in BANDS}
        # onset reliability: onsets from each half (primary definition)
        hA = np.einsum("snc,kc->kns", CI_, halves) / halves.sum(1)[:, None, None]
        hB = np.einsum("snc,kc->kns", CI_, ~halves) / (~halves).sum(1)[:, None, None]
        orel, ocnt, oagree = [], [], []
        for k in range(args.n_splits):
            la, ca, _, _ = onset(hA[k], logt)
            lb, cb, _, _ = onset(hB[k], logt)
            m = (ca == 0) & (cb == 0)
            oagree.append(float(np.mean(ca == cb)))
            if m.sum() >= 3 and np.std(la[m]) > 0 and np.std(lb[m]) > 0:
                orel.append(np.corrcoef(la[m], lb[m])[0, 1])
                ocnt.append(int(m.sum()))
        orel_m = float(np.mean(orel)) if orel else float("nan")
        dz["reliability"] = {"r_half_I": rI.tolist(), "r_SB_I": rIsb.tolist(), "flagged_steps": steps[flag].tolist(),
                             "r_SB_T": rTsb.tolist(), "r_SB_I_band": {b: rb[b].tolist() for b in BANDS},
                             "onset_r_half": orel_m, "onset_splits_finite": len(orel),
                             "onset_mean_verbs": float(np.mean(ocnt)) if ocnt else 0.0, "onset_censoring_agreement": float(np.mean(oagree))}
        mk = lambda j: f"{steps[j]}{' †' if flag[j] else ''}"
        lines += ["### Reliability (split-half, 200 animacy-stratified 12 / 12 context splits)", "",
                  "| step | Δ_pass(I) r_half | r_SB | flag | r_SB within Head / Tail / XTail | Δ_pass(T) r_SB |",
                  "|---|---|---|---|---|---|"]
        for j in range(S):
            lines.append(f"| {steps[j]} | {rI[j]:.2f} | {rIsb[j]:.2f} | {'†' if flag[j] else ''} | "
                         + " / ".join(f"{rb[b][j]:.2f}" for b in BANDS) + f" | {rTsb[j]:.2f} |")
            rel_rows += [{"size": size, "step": int(steps[j]), "outcome": "I", "group": "all", "r_half": rI[j], "r_SB": rIsb[j],
                          "flagged": bool(flag[j])},
                         {"size": size, "step": int(steps[j]), "outcome": "T", "group": "all", "r_half": rT[j], "r_SB": rTsb[j],
                          "flagged": bool(~(rTsb[j] >= FLAG_R))}]
            rel_rows += [{"size": size, "step": int(steps[j]), "outcome": "I", "group": b, "r_half": np.nan, "r_SB": rb[b][j],
                          "flagged": bool(~(rb[b][j] >= FLAG_R))} for b in BANDS]
        lines += ["", f"Flagged checkpoints (r_SB < {FLAG_R}): {', '.join(map(str, steps[flag])) or 'none'}. Onset reliability "
                      f"(primary definition; r of log10 onset between halves over verbs crossing in both): {orel_m:.2f} (mean over "
                      f"{len(orel)} splits with a finite r, {np.mean(ocnt) if ocnt else 0:.0f} verbs on average); the two halves give "
                      f"the same censoring status for {np.mean(oagree):.0%} of verbs.", ""]

        # ------------------------------------------------ 1. onset
        lines += ["### 1. Onset of Δ_pass(I) < 0", ""]
        X3 = np.column_stack([np.ones(n), x, lnt])
        on = {}
        for dname, (thr, per, strict) in ONSET_DEFS.items():
            lo_, ce, kk, bnd = onset(YI.reshape(nd * n, S), logt, thr, per, strict)
            on[dname] = (lo_.reshape(nd, n), ce.reshape(nd, n), kk.reshape(nd, n), bnd, per)
        cnt_lines = ["| definition | right-censoring bound | crossers Head / Tail / XTail | left-censored | right-censored | onsets "
                     "in the last qualifying interval | interpolation across a † step | persistence uses a † step |",
                     "|---|---|---|---|---|---|---|---|"]
        for dname, (lo_, ce, kk, bnd, per) in on.items():
            c0, k0 = ce[0], kk[0]
            byb = lambda m: " / ".join(str(int((m & (band == b)).sum())) for b in BANDS)
            fi = (c0 == 0) & (flag[np.clip(k0, 0, S - 1)] | flag[np.clip(k0 - 1, 0, S - 1)])
            fp = (c0 != 1) & np.array([flag[max(k_, 0):min(k_ + per + 1, S)].any() for k_ in k0])
            cnt_lines.append(f"| {dname} | {10 ** bnd:,.0f} | {byb(c0 == 0)} | {byb(c0 == -1)} | {byb(c0 == 1)} | "
                             f"{int(((c0 == 0) & (k0 == np.searchsorted(logt, bnd))).sum())} | {int(fi.sum())} | {int(fp.sum())} |")
            for i in range(n):
                onset_rows.append({"size": size, "definition": dname, "i_verb": P.i_verb[i], "band": band[i], "quintile": quint[i] + 1,
                                   "log10_c": x[i], "n_tok": int(np.exp(lnt[i]).round()), "cens": int(c0[i]),
                                   "onset_step": float(10 ** lo_[0, i]) if c0[i] == 0 else np.nan,
                                   "log10_E_onset": float(x[i] + lo_[0, i] - np.log10(TOTAL)) if c0[i] == 0 else np.nan,
                                   "interval_lo_step": int(steps[k0[i] - 1]) if c0[i] == 0 else np.nan,
                                   "interval_hi_step": int(steps[k0[i]]) if c0[i] == 0 else np.nan,
                                   "flag_interp": bool(fi[i]), "flag_persist": bool(fp[i])})
            if dname == "primary":
                fdz["onset_flag"] = fi
        neg_first = int(((YI[0, :, 0] < 0) & (on["primary"][1][0] != -1)).sum())
        lines += ["Counts (point estimate; Head / Tail / XTail). Right-censored verbs have no qualifying checkpoint up to the bound "
                  "(their onset is later than the bound).", "", *cnt_lines, "",
                  f"Verbs with Δ_pass(I) < 0 at the first step that are not left-censored (they return to ≥ 0 within the next two "
                  f"checkpoints): {neg_first}.", ""]
        fits, fails = {}, {}
        for fname, dname, cols in (("primary (crossers, + log n_tok)", "primary", [0, 1, 2]),
                                   ("(d) no token covariate", "primary", [0, 1]),
                                   ("lenient end (persistence shortened at the last two steps)", "lenient end", [0, 1, 2]),
                                   ("(a) no persistence", "no persistence", [0, 1, 2]),
                                   ("(b) threshold −1", "threshold −1", [0, 1, 2])):
            lo_, ce = on[dname][0], on[dname][1]
            b = np.full(nd, np.nan)
            for d in range(nd):
                m = ce[d] == 0
                b[d] = wls(X3[m][:, cols], lo_[d, m], wb[d, m])[1]
            fits[fname] = (b, int((ce[0] == 0).sum()))
        # censored regressions (primary definition, all verbs)
        lo_, ce, kk, bnd, _ = on["primary"]
        m0 = ce[0] == 0
        yfill = np.where(ce[0] == 0, lo_[0], np.where(ce[0] == -1, logt[0], bnd))  # censored verbs at their bounds
        b0 = wls(X3, yfill, np.ones(n))
        start = np.append(b0, np.log(np.std(yfill - X3 @ b0) + 1e-3))
        fl_b = np.log10(MATURE)
        ncr = {k_: np.full(nd, np.nan) for k_ in ("tobit", "floor", "interval")}
        nfail = {k_: 0 for k_ in ncr}
        for d in range(nd):
            c_, y_ = ce[d], lo_[d]
            kind = np.where(c_ == 0, 0, np.where(c_ == -1, 1, 2))
            lb = np.full(n, bnd)
            ub = np.full(n, logt[0])
            for key, kd, l_, u_ in (("tobit", kind, lb, ub),
                                    ("floor", np.where((c_ == -1) | ((c_ == 0) & (y_ <= fl_b)), 1, kind), lb, np.full(n, max(fl_b, logt[0]))),
                                    ("interval", np.where(c_ == 0, 3, kind),
                                     np.where(c_ == 0, logt[np.clip(kk[d] - 1, 0, S - 1)], bnd), np.where(c_ == 0, logt[np.clip(kk[d], 0, S - 1)], logt[0]))):
                p_, ok = cnreg(X3, wb[d], kd, y_, l_, u_, start)
                ncr[key][d] = p_[1]
                nfail[key] += int(not ok)
        fits["(c) Tobit (all verbs, + log n_tok)"] = (ncr["tobit"], n)
        fits[f"(e) Tobit, maturity floor (onsets ≤ {MATURE} left-censored)"] = (ncr["floor"], n)
        fits["(f) checkpoint-interval censored (all verbs, + log n_tok)"] = (ncr["interval"], n)
        fails.update({"(c) Tobit (all verbs, + log n_tok)": nfail["tobit"],
                      f"(e) Tobit, maturity floor (onsets ≤ {MATURE} left-censored)": nfail["floor"],
                      "(f) checkpoint-interval censored (all verbs, + log n_tok)": nfail["interval"]})
        lines += ["| fit | verbs (point) | β_on (95% CI) | label | valid draws |", "|---|---|---|---|---|"]
        dz["onset"] = {}
        for fname, (b, nv) in fits.items():
            lab = onset_label(*ci(b)) if np.isfinite(b[0]) else "unestimable"
            dz["onset"][fname] = {"beta_on": [float(b[0]), *ci(b)], "label": lab, "n": nv, "valid_draws": int(np.isfinite(b[1:]).sum()),
                                  "failed_fits": fails.get(fname, int((~np.isfinite(b[1:])).sum()))}
            lines.append(f"| {fname} | {nv} | {f3(b)} | {lab} | {int(np.isfinite(b[1:]).sum())} / {nd - 1} |")
            srow(size, "onset", f"beta_on|{fname}", b, lab)
        prim = dz["onset"]["primary (crossers, + log n_tok)"]["label"]
        qual = [nm for nm, key in (("Tobit", "(c) Tobit (all verbs, + log n_tok)"),
                                   ("maturity-floor Tobit", f"(e) Tobit, maturity floor (onsets ≤ {MATURE} left-censored)"))
                if dz["onset"][key]["label"] != "held"]
        verdict = f"held, not robust to censoring ({' and '.join(qual)} not 'held')" if prim == "held" and qual else prim
        dz["onset"]["verdict"] = verdict
        lines += ["", f"**Onset verdict (primary): {verdict}.** Count equivalence predicts β_on = −1; the prediction is "
                      "−1 < β_on < 0. A CI that includes a value is not evidence that the slope equals it.", ""]
        # onset exposure by band
        lo0, c0 = on["primary"][0][0], on["primary"][1][0]
        le = x + lo0 - np.log10(TOTAL)
        parts = []
        for b in BANDS:
            m = (c0 == 0) & (band == b)
            parts.append(f"{NAME[b]} (n = {int(m.sum())}) median {np.median(le[m]):.2f} [IQR {np.percentile(le[m], 25):.2f}, "
                         f"{np.percentile(le[m], 75):.2f}], onset step median {10 ** np.median(lo0[m]):,.0f}" if m.any() else f"{NAME[b]} no crossers")
        lines += ["Onset exposure log10 E_onset (crossers, primary): " + "; ".join(parts) + ". In the crossers sample and with the "
                  "same covariates, the slope of log10 E_onset on log10 c_total is β_on + 1.", ""]
        fdz["onset"] = (lo0, c0, x, band, logt, bnd, wls(X3[c0 == 0], lo0[c0 == 0], np.ones((c0 == 0).sum())))

        # ------------------------------------------------ 2. late window
        win = steps >= LATE_FROM
        Yw = YI[:, :, win]
        sw_ = sv[win] - sv[win].mean()
        st_ = np.array([late_stats(Yw[d], wb[d], x, lnt, sw_) for d in range(nd)])
        bb, bbt, bw, gam, bt_ = st_.T
        diff = bb - bw
        # calibration: plug-in values under a common nonlinear exposure curve fitted to the window (natural spline in log10 E)
        logEw = (x[:, None] + sv[win][None, :]).ravel()
        Bc = np.column_stack([np.ones(len(logEw)), ns_basis(logEw, np.percentile(logEw, [2.5, 100 / 3, 200 / 3, 97.5]))])
        nw = int(win.sum())
        nul = np.full((nd, 5), np.nan)
        for d in range(nd):
            cf = wls(Bc, Yw[d].ravel(), np.repeat(wb[d], nw))
            if np.all(np.isfinite(cf)):
                nul[d] = late_stats((Bc @ cf).reshape(n, nw), wb[d], x, lnt, sw_)
        diff_null, gam_null = nul[:, 0] - nul[:, 2], nul[:, 3]
        diff_cal, gam_cal = diff - diff_null, gam - gam_null
        lo90, hi90 = ci(bw, 5, 95)
        lab_main = "held" if ci(diff)[1] < 0 else "reversed" if ci(diff)[0] > 0 else "failed (consistent with count equivalence)"
        lab_cal = "below 0" if ci(diff_cal)[1] < 0 else "above 0" if ci(diff_cal)[0] > 0 else "includes 0"
        verdict_late = lab_main
        if lab_main == "held":
            verdict_late = ("held (also beyond the fitted common exposure curve)" if lab_cal == "below 0" else
                            "held, within what a common nonlinear exposure curve produces")
        lab_bb = "held" if (ci(bb)[1] < 0 or ci(bb)[0] > 0) and ci(bb)[0] <= A1_SLOPE <= ci(bb)[1] else "missed"
        lab_bw = "held" if -0.25 < lo90 and hi90 < 0.25 else "missed"
        lab_g = "positive" if ci(gam)[0] > 0 else "negative" if ci(gam)[1] < 0 else "unresolved"
        lab_gc = "positive" if ci(gam_cal)[0] > 0 else "negative" if ci(gam_cal)[1] < 0 else "unresolved"
        dz["late"] = {"window": steps[win].tolist(), "beta_b": [bb[0], *ci(bb)], "beta_b_ntok": [bbt[0], *ci(bbt)],
                      "beta_w": [bw[0], *ci(bw)], "beta_w_90": [lo90, hi90], "diff": [diff[0], *ci(diff)], "gamma": [gam[0], *ci(gam)],
                      "beta_t_at_mean_x": [bt_[0], *ci(bt_)], "diff_null": [diff_null[0], *ci(diff_null)],
                      "diff_calibrated": [diff_cal[0], *ci(diff_cal)], "gamma_null": [gam_null[0], *ci(gam_null)],
                      "gamma_calibrated": [gam_cal[0], *ci(gam_cal)], "main": lab_main, "verdict": verdict_late,
                      "calibrated_reading": lab_cal, "beta_b_near_A1": lab_bb, "beta_w_near_0": lab_bw, "gamma_label": lab_g,
                      "gamma_calibrated_label": lab_gc, "flagged_window_steps": steps[win & flag].tolist(),
                      "null_fit_valid_draws": int(np.isfinite(nul[1:, 0]).sum())}
        for nm, d_, lab in (("beta_b", bb, lab_bb), ("beta_b_ntok", bbt, ""), ("beta_w", bw, lab_bw), ("beta_b_minus_beta_w", diff, lab_main),
                            ("gamma", gam, lab_g), ("diff_null", diff_null, ""), ("diff_calibrated", diff_cal, lab_cal),
                            ("gamma_calibrated", gam_cal, lab_gc)):
            srow(size, "late", nm, d_, lab)
        fdz["late"] = (bb, bw, diff, diff_cal, bool((win & flag).any()))
        lines += [f"### 2. Late window (steps {', '.join(mk(j) for j in np.flatnonzero(win))})", "",
                  "Within a verb, log10 E varies only through log10 step (x is absorbed by the verb fixed effect), so β_w is the "
                  "shared late time slope (nats per decade of training).", "",
                  "| quantity | estimate (95% CI) | prediction | reading |", "|---|---|---|---|",
                  f"| β_b (window mean on log10 c_total) | {f3(bb)} | near A1's −0.728 (CI excludes 0, includes −0.728) | {lab_bb} |",
                  f"| β_b + log n_tok (sensitivity) | {f3(bbt)} | | |",
                  f"| β_w (verb fixed effects, on log10 E) | {f3(bw)}; 90% [{lo90:+.2f}, {hi90:+.2f}] | near 0 (90% CI within ±0.25) | {lab_bw} |",
                  f"| **β_b − β_w** | {f3(diff)} | **< 0 (CI below 0)** | **{lab_main}** |",
                  f"| γ (log10 c × log10 step) | {f3(gam)} | unresolved | {lab_g} |",
                  f"| time slope at mean log10 c | {f3(bt_)} | | |",
                  f"| calibration: β_b − β_w under the fitted common curve | {f3(diff_null)} | | |",
                  f"| calibrated β_b − β_w (observed − common curve) | {f3(diff_cal)} | | {lab_cal} |",
                  f"| calibrated γ (observed − common curve) | {f3(gam_cal)} | | {lab_gc} |", "",
                  f"**Late-window verdict: {verdict_late}.**", ""]

        # ------------------------------------------------ 3. exposure-aligned curves
        lines += ["### 3. Exposure-aligned curves", ""]
        dz["curves"] = {}
        hdr = ["| bins | steps | comparison | grid (log10 E) | points | resolved D < 0 at | resolved D > 0 at | reading "
               "(pointwise) | simultaneous band | crossing at log10 E (point estimate) | grid points using a † step | "
               "common-curve calibrated (pointwise) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        tabs = {"bin": list(hdr), "verb": list(hdr)}
        pooled = []
        rng_ = lambda G, m: (f"{G[m].min():.1f}–{G[m].max():.1f} ({int(m.sum())})" if m.any() else "—")
        for sub, keep in (("all", np.ones(S, bool)), (f"≥ {MATURE}", steps >= MATURE)):
            ks = np.flatnonzero(keep)
            sK, K = sv[ks], len(ks)
            Yk = YI[:, :, ks]
            # common-curve null: natural spline in log10 E fitted to every verb x step value of these steps, per draw
            lE = (x[:, None] + sK[None, :]).ravel()
            Bn = np.column_stack([np.ones(len(lE)), ns_basis(lE, np.percentile(lE, [2.5, 100 / 3, 200 / 3, 97.5]))])
            Yh = np.full_like(Yk, np.nan)
            for d in range(nd):
                cf = wls(Bn, Yk[d].ravel(), np.repeat(wb[d], K))
                if np.all(np.isfinite(cf)):
                    Yh[d] = (Bn @ cf).reshape(n, K)
            for bname, (labs, members) in binsets.items():
                nb = len(labs)
                shown_pairs = [(0, nb - 1)] + [(i, i + 1) for i in range(nb - 1)] if bname == "quintile" else [(0, 2), (0, 1), (1, 2)]
                xb = np.array([x[m].mean() for m in members])
                Xq = xb[:, None] + sK[None, :]
                binmean = lambda Ys: np.stack([np.einsum("dn,dns->ds", wb[:, m], Ys[:, m]) / wb[:, m].sum(1)[:, None] for m in members], 1)
                if bname == "quintile":
                    fdz[f"curves|{sub}"] = (Xq, binmean(Yk)[0], steps[ks], flag[ks], labs)
                pairs_ = list(shown_pairs)
                if bname == "quintile" and sub == "all":
                    pairs_ += [(i, j) for i in range(nb) for j in range(i + 1, nb) if (i, j) not in pairs_]
                for (qa, qb) in pairs_:
                    cname = f"{labs[qa]} − {labs[qb]}"
                    shown = (qa, qb) in shown_pairs
                    for mode in ("bin", "verb"):
                        if mode == "verb" and not shown:
                            continue
                        # grid, interpolation structure and flags: from counts and steps only
                        if mode == "bin":  # bin curves at the bin's inventory mean exposure
                            lo_e, hi_e = max(Xq[qa, 0], Xq[qb, 0]), min(Xq[qa, -1], Xq[qb, -1])
                        else:  # each verb on its own exposure axis; support where >= COVER of the bin's verbs are covered
                            lo_e, hi_e = min(Xq[qa, 0], Xq[qb, 0]) - 2, max(Xq[qa, -1], Xq[qb, -1]) + 2
                        G = np.round(np.arange(np.ceil(lo_e / GRID - 1e-9), np.floor(hi_e / GRID + 1e-9) + 1) * GRID, 1)
                        if mode == "verb":
                            ok_ = np.ones(len(G), bool)
                            for q in (qa, qb):
                                pos_ = G[None, :] - x[members[q]][:, None]
                                ok_ &= ((pos_ >= sK[0] - 1e-9) & (pos_ <= sK[-1] + 1e-9)).mean(0) >= COVER
                            G = G[ok_]
                        key = f"{bname}|{sub}|{mode}|{cname}"
                        if len(G) == 0:
                            dz["curves"][key] = {"reading": "no overlap"}
                            if shown:
                                tabs[mode].append(f"| {bname} | {sub} | {cname} | — | 0 | | | no overlap | | | | |")
                            continue
                        parts_, fl = [], np.zeros(len(G), bool)
                        for q in (qa, qb):
                            if mode == "bin":
                                i0 = np.clip(np.searchsorted(Xq[q], G, side="right") - 1, 0, K - 2)
                                fr = np.clip((G - Xq[q, i0]) / (Xq[q, i0 + 1] - Xq[q, i0]), 0, 1)
                                fl |= (flag[ks][i0] & (fr < 1)) | (flag[ks][i0 + 1] & (fr > 0))
                                parts_.append(("bin", q, i0, fr))
                            else:
                                mi = np.flatnonzero(members[q])
                                pos_ = G[None, :] - x[mi][:, None]  # [verb, grid] position on the log10(t / 143000) axis
                                cv = (pos_ >= sK[0] - 1e-9) & (pos_ <= sK[-1] + 1e-9)
                                i0 = np.clip(np.searchsorted(sK, pos_, side="right") - 1, 0, K - 2)
                                fr = np.clip((pos_ - sK[i0]) / (sK[i0 + 1] - sK[i0]), 0, 1)
                                fl |= (cv & ((flag[ks][i0] & (fr < 1)) | (flag[ks][i0 + 1] & (fr > 0)))).any(0)
                                parts_.append(("verb", mi, i0, fr, cv))

                        def apply(Ys):
                            vals = []
                            Mb = binmean(Ys) if mode == "bin" else None
                            for pt in parts_:
                                if pt[0] == "bin":
                                    _, q, i0, fr = pt
                                    vals.append(Mb[:, q, i0] * (1 - fr) + Mb[:, q, i0 + 1] * fr)
                                else:
                                    _, mi, i0, fr, cv = pt
                                    Yq = Ys[:, mi]
                                    r_ = np.arange(len(mi))[:, None]
                                    vv = Yq[:, r_, i0] * (1 - fr) + Yq[:, r_, i0 + 1] * fr
                                    ww = wb[:, mi][:, :, None] * cv[None]
                                    with np.errstate(invalid="ignore"):
                                        vals.append((ww * vv).sum(1) / ww.sum(1))
                            return vals[0] - vals[1]

                        D = apply(Yk)
                        if mode == "bin" and bname == "quintile" and sub == "all":
                            pooled.append(D)
                        if not shown:
                            continue
                        r = compare(D, G)
                        rc = compare(D - apply(Yh), G)
                        dz["curves"][key] = {"grid": [float(G[0]), float(G[-1])], "n": len(G), "reading": r["reading"],
                                             "simultaneous": r["reading_sim"], "predicted_order": r["po"],
                                             "predicted_order_sim": r["po_sim"], "crossing_log10_E": [float(v) for v in r["cross"]],
                                             "neg_points": G[r["hi"] < 0].tolist(), "pos_points": G[r["lo"] > 0].tolist(),
                                             "crit_sim": r["crit"], "incomplete_draws": r["incomplete"], "flagged_points": int(fl.sum()),
                                             "calibrated_reading": rc["reading"], "calibrated_predicted_order": rc["po"]}
                        tabs[mode].append(f"| {bname} | {sub} | {cname} | {G[0]:.1f} to {G[-1]:.1f} | {len(G)} | {rng_(G, r['hi'] < 0)} | "
                                          f"{rng_(G, r['lo'] > 0)} | {r['reading']} | {r['reading_sim']} | "
                                          f"{', '.join(f'{v:.2f}' for v in r['cross']) or '—'} | {int(fl.sum())} | {rc['reading']} |")
                        for i in range(len(G)):
                            grid_rows.append({"size": size, "alignment": mode, "bins": bname, "steps": sub, "comparison": cname,
                                              "log10_E": G[i], "D": D[0, i], "lo": r["lo"][i], "hi": r["hi"][i], "sim_lo": r["slo"][i],
                                              "sim_hi": r["shi"][i], "D_calibrated": rc["d0"][i], "cal_lo": rc["lo"][i],
                                              "cal_hi": rc["hi"][i], "flagged": bool(fl[i])})
        pr = dz["curves"].get("quintile|all|bin|Q1 − Q5", {})
        pv = dz["curves"].get("quintile|all|verb|Q1 − Q5", {})
        if pr.get("predicted_order"):
            q_ = [w_ for w_, ok in (("simultaneous inference", pr.get("predicted_order_sim")),
                                     ("within-bin spread (per-verb alignment)", pv.get("predicted_order")),
                                     ("the common-curve calibration", pr.get("calibrated_predicted_order"))) if not ok]
            verdict_c = "held" + (f" pointwise; not robust to {' or '.join(q_)}" if q_ else "")
        else:
            verdict_c = "missed"
        dz["curves"]["verdict"] = verdict_c
        lines += ["Bin curves at the bin's inventory mean exposure (declared construction):", "", *tabs["bin"], "",
                  f"Per-verb alignment (each verb interpolated on its own log10 E; grid points covered by ≥ {COVER:.0%} of each bin's "
                  "verbs; qualifier):", "", *tabs["verb"], "",
                  "Common-curve calibrated = D − D_null, where D_null passes a natural spline f̂(log10 E) fitted to all verbs and "
                  "steps of that comparison's step set through the same sampling, binning and interpolation (qualifier).", "",
                  f"**Primary (Q1 − Q5, all steps, bin curves): {pr.get('reading', 'no overlap')}; simultaneous band: "
                  f"{pr.get('simultaneous', '—')}; per-verb alignment: {pv.get('reading', 'no overlap')}; common-curve calibrated: "
                  f"{pr.get('calibrated_reading', '—')} → crossing verdict: {verdict_c}.** Matched exposure here compares frequent "
                  "verbs early in training with rare verbs late: an observational trajectory pattern, confounded with model age.", ""]
        # pooled D vs A6's matched-exposure diagnostic
        Dp = np.concatenate(pooled, 1).mean(1)
        logE_vs = x[:, None] + sv[None, :]  # [verb, step]
        mu_, mv_, ma_, mb_ = [], [], [], []
        for a in range(S):
            for b2 in range(S):
                if steps[b2] >= 4 * steps[a]:
                    u, v = np.nonzero(np.abs(logE_vs[:, a][:, None] - logE_vs[:, b2][None, :]) <= 0.1)
                    mu_.append(u), mv_.append(v), ma_.append(np.full(len(u), a)), mb_.append(np.full(len(u), b2))
        mu_, mv_, ma_, mb_ = (np.concatenate(z_) for z_ in (mu_, mv_, ma_, mb_))
        diag = np.empty(nd)
        for a in range(0, nd, 50):
            dd = YI[a:a + 50][:, mv_, mb_] - YI[a:a + 50][:, mu_, ma_]
            ww = wb[a:a + 50][:, mv_] * wb[a:a + 50][:, mu_]
            diag[a:a + 50] = (dd * ww).sum(1) / ww.sum(1)
        repro = bool(abs(float(diag[0]) - A6_DIAG.get(size, np.nan)) <= 0.0005)
        if not repro and not args.skip_a6_check:
            raise SystemExit(f"pythia-{size}: A6's matched-exposure diagnostic not reproduced ({diag[0]:+.4f} vs "
                             f"{A6_DIAG.get(size)}); stopping before any output is written")
        dz["curves"]["pooled_D"] = [Dp[0], *ci(Dp)]
        dz["curves"]["a6_diag"] = [diag[0], *ci(diag), int(len(mu_))]
        dz["curves"]["a6_diag_reproduced"] = repro
        dz["curves"]["pooled_minus_diag"] = [(Dp - diag)[0], *ci(Dp - diag)]
        srow(size, "curves", "pooled_D", Dp)
        srow(size, "curves", "a6_matched_diag", diag)
        srow(size, "curves", "pooled_D_minus_a6_diag", Dp - diag)
        lines += [f"Check against A6: A6's matched-exposure diagnostic recomputed from the same scores ({len(mu_)} cell pairs, "
                  f"|Δ log10 E| ≤ 0.1, steps ≥ 4× apart; later − earlier) {f3(diag, 3)} (A6 reported {A6_DIAG.get(size, float('nan')):+.3f}: "
                  f"{'reproduced' if repro else 'NOT reproduced'}). Pooled D̄ over the 10 quintile pairs (bin curves, all steps, grid "
                  f"cells equally weighted) {f3(Dp, 3)}; difference D̄ − diagnostic {f3(Dp - diag, 3)} (different weighting of "
                  "the same comparison; see the plan).", ""]

        # ------------------------------------------------ 4. class-level vs verb-specific
        lines += ["### 4. Class-level vs verb-specific (descriptive)", "",
                  "| step | AUC T vs I Head / Tail / XTail | mean L Head / Tail / XTail | single-token a_V T / I Head | Tail | "
                  "a_V AUC Head / Tail | identity max dev |", "|---|---|---|---|---|---|---|"]
        aucs = {b: [] for b in BANDS}
        sep = {}
        st_role, st_band = s1.role.to_numpy(), s1.band.to_numpy()
        ws1 = wb[:, s1_pair]
        for j in range(S):
            cells = []
            for b in BANDS:
                m = band == b
                A_ = wauc(YT[:, m, j], YI[:, m, j], wb[:, m], wb[:, m])
                Lm = (wb[:, m] * (YT[:, m, j] - YI[:, m, j])).sum(1) / wb[:, m].sum(1)
                aucs[b].append(A_)
                cells.append((A_, Lm))
                class_rows.append({"size": size, "step": int(steps[j]), "flagged": bool(flag[j]), "band": b, "measure": "auc_dpass",
                                   "est": A_[0], "lo": ci(A_)[0], "hi": ci(A_)[1]})
                class_rows.append({"size": size, "step": int(steps[j]), "flagged": bool(flag[j]), "band": b, "measure": "mean_L",
                                   "est": Lm[0], "lo": ci(Lm)[0], "hi": ci(Lm)[1]})
            avm, ava = {}, {}
            for b in ("head", "tail"):
                for r_ in ("T", "I"):
                    mm = (st_band == b) & (st_role == r_)
                    avm[b, r_] = (ws1[:, mm] * AV[:, mm, j]).sum(1) / ws1[:, mm].sum(1)
                    class_rows.append({"size": size, "step": int(steps[j]), "flagged": bool(flag[j]), "band": b,
                                       "measure": f"aV_mean_{r_}", "est": avm[b, r_][0], "lo": ci(avm[b, r_])[0], "hi": ci(avm[b, r_])[1]})
                mt, mi = (st_band == b) & (st_role == "T"), (st_band == b) & (st_role == "I")
                ava[b] = wauc(AV[:, mt, j], AV[:, mi, j], ws1[:, mt], ws1[:, mi])
                class_rows.append({"size": size, "step": int(steps[j]), "flagged": bool(flag[j]), "band": b, "measure": "auc_aV",
                                   "est": ava[b][0], "lo": ci(ava[b])[0], "hi": ci(ava[b])[1]})
            lines.append(f"| {mk(j)} | " + " / ".join(f"{a[0][0]:.2f} [{ci(a[0])[0]:.2f}, {ci(a[0])[1]:.2f}]" for a in cells) + " | "
                         + " / ".join(f"{a[1][0]:+.2f}" for a in cells) + " | "
                         + f"{avm['head', 'T'][0]:+.2f} / {avm['head', 'I'][0]:+.2f} | {avm['tail', 'T'][0]:+.2f} / {avm['tail', 'I'][0]:+.2f} | "
                         + f"{ava['head'][0]:.2f} / {ava['tail'][0]:.2f} | {idres[j]:.1e} |")
        for b in BANDS:
            k_ = stable_from([ci(a)[0] > 0.5 for a in aucs[b]])
            sep[b] = int(steps[k_]) if k_ is not None else None
        # XTail onset: median primary onset (left- / right-censored verbs below / above every crosser); band-mean onset
        mx = band == "xtail"
        key = np.sort(np.where(c0[mx] == -1, -np.inf, np.where(c0[mx] == 1, np.inf, lo0[mx])))
        med = key[(len(key) - 1) // 2] if len(key) % 2 else key[len(key) // 2 - 1:len(key) // 2 + 1].mean()
        med_s = ("≤ first step" if med == -np.inf else f"> {10 ** bnd:,.0f} (right-censored)" if not np.isfinite(med) else f"{10 ** med:,.0f}")
        bm = YI[0, mx].mean(0)
        kb = stable_from(list(bm < 0))
        dz["class"] = {"separation_step": sep, "xtail_median_onset": med_s,
                       "xtail_bandmean_onset": int(steps[kb]) if kb is not None else None, "identity_max_dev": max(idres)}
        lines += ["", "Separation step (AUC CI above 0.5 at every later checkpoint): " + ", ".join(
            f"{NAME[b]} {sep[b] if sep[b] is not None else 'none'}" for b in BANDS) + f". XTail I onset: median primary onset "
            f"{med_s}; band-mean Δ_pass(I) < 0 from step {dz['class']['xtail_bandmean_onset']}. a_V identity (single-token "
            f"a_V = Δ_pass + per-checkpoint constant): max deviation {max(idres):.1e} nats.", ""]

    # ------------------------------------------------ replication
    if "1.4b" in args.sizes:
        rows = [("onset (β_on verdict)", lambda z: z["onset"]["verdict"]),
                ("late window (β_b − β_w verdict)", lambda z: z["late"]["verdict"]),
                ("crossing (Q1 − Q5 verdict)", lambda z: z["curves"]["verdict"])]
        lines += ["## Replication (principal claims)", "",
                  "| claim | " + " | ".join(f"pythia-{s}" for s in args.sizes) + " | 410M replicates 1.4B |",
                  "|---|" + "---|" * (len(args.sizes) + 1)]
        for nm, get in rows:
            labs_ = {s: get(dec[f"pythia-{s}"]) for s in args.sizes}
            rep = ("yes" if labs_.get("410m") == labs_["1.4b"] else "no") if "410m" in labs_ else "—"
            lines.append(f"| {nm} | " + " | ".join(labs_[s] for s in args.sizes) + f" | {rep} |")
            dec.setdefault("replication", {})[nm] = {"labels": labs_, "replicates": rep}
        lines += ["", "Matching labels that are not 'held' (e.g. unresolved) are matching inconclusiveness, not confirmation.", ""]

    # ------------------------------------------------ outputs
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n")
    pd.DataFrame(summ).to_csv(out / "a8_summary.csv", index=False)
    pd.DataFrame(onset_rows).to_csv(out / "a8_onsets.csv", index=False)
    pd.DataFrame(rel_rows).to_csv(out / "a8_reliability.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(out / "a8_dgrid.csv", index=False)
    pd.DataFrame(class_rows).to_csv(out / "a8_class.csv", index=False)
    (out / "a8_decisions.json").write_text(json.dumps(dec, indent=1, default=float))
    if not args.no_figures:
        figures(figdata, figd)
    print("\n".join(lines))


# ---------------------------------------------------------------- figures

INK, INK2, MUTED, GRIDC, BASE, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
CAT = {"head": "#2a78d6", "tail": "#eb6834", "xtail": "#1baf7a"}
BLUE_ORD = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # Q1 (rarest) light -> Q5 dark (validated ordinal ramp)
ORANGE = ["#fbd9c9", "#f5b391", "#eb6834", "#b84a1c", "#7a2f10"]
MODEL = {"1.4b": "#2a78d6", "410m": "#eb6834", "6.9b": MUTED}


def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRIDC, lw=0.6)
    ax.set_axisbelow(True)


def figures(fd, figd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.labelcolor": INK2, "text.color": INK})
    cmap = LinearSegmentedColormap.from_list("steps", ORANGE)
    for size, d in fd.items():
        # F1: exposure-aligned bin curves
        Xq, M0, st, fl, labs = d["curves|all"]
        fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=SURF)
        _style(ax)
        ax.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)))
        nrm = LogNorm(st.min(), st.max())
        for q in range(len(labs)):
            ax.plot(Xq[q], M0[q], color=BLUE_ORD[q], lw=2, zorder=2)
            ax.scatter(Xq[q][~fl], M0[q][~fl], c=st[~fl], cmap=cmap, norm=nrm, s=26, edgecolors=BLUE_ORD[q], linewidths=1, zorder=3)
            ax.scatter(Xq[q][fl], M0[q][fl], facecolors=SURF, edgecolors=BLUE_ORD[q], s=26, linewidths=1, zorder=3)
            ax.plot([], [], color=BLUE_ORD[q], lw=2,
                    label=labs[q] + (" (rarest)" if q == 0 else " (most frequent)" if q == len(labs) - 1 else ""))
        ax.legend(frameon=False, fontsize=7, loc="lower left", title="c_total quintile", title_fontsize=7)
        sm = plt.cm.ScalarMappable(norm=nrm, cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, pad=0.03, fraction=0.04)
        cb.set_label("training step", color=INK2, fontsize=8)
        cb.ax.tick_params(labelsize=7, colors=INK2)
        cb.outline.set_edgecolor(BASE)
        ax.set_xlabel("log10 own exposure (bin mean; E = c_total × step / 143000)")
        ax.set_ylabel("Δ_pass(I), bin mean (nats)")
        ax.set_title(f"Pythia-{size}: intransitive passive preference by count quintile" + (" (hollow: r_SB < 0.7 or undefined)" if fl.any() else ""),
                     fontsize=9, color=INK, loc="left")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(figd / f"a8_f1_exposure_{size}.{ext}", dpi=200, facecolor=SURF)
        plt.close(fig)
        # F2: onset vs c_total
        lo0, c0, x, band, logt, bnd, bfit = d["onset"]
        fi = d["onset_flag"]
        fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=SURF)
        _style(ax)
        for b in BANDS:
            m = band == b
            ax.scatter(x[m & (c0 == 0) & ~fi], lo0[m & (c0 == 0) & ~fi], s=14, color=CAT[b], alpha=0.8, label=NAME[b], edgecolors="none")
            ax.scatter(x[m & (c0 == 0) & fi], lo0[m & (c0 == 0) & fi], s=14, facecolors=SURF, edgecolors=CAT[b], linewidths=0.8)
            ax.scatter(x[m & (c0 == -1)], np.full((m & (c0 == -1)).sum(), logt[0]), marker="v", s=22, facecolors="none",
                       edgecolors=CAT[b], linewidths=1)
            ax.scatter(x[m & (c0 == 1)], np.full((m & (c0 == 1)).sum(), bnd), marker="^", s=22, facecolors="none",
                       edgecolors=CAT[b], linewidths=1)
        cx = x[c0 == 0]
        xs = np.linspace(cx.min(), cx.max(), 50)
        xm, ym = cx.mean(), np.mean(lo0[c0 == 0])
        ax.plot(xs, ym - (xs - xm), color=MUTED, lw=1.2, ls=(0, (4, 3)), label="slope −1 (count equivalence)")
        ax.plot(xs, ym + bfit[1] * (xs - xm), color=INK, lw=1.5, label=f"OLS fit on crossers (β_on {bfit[1]:+.2f})")
        tk = sorted({int(round(10 ** logt[0]))} | {v for v in (256, 1000, 4000, 16000, 63000, 143000) if logt[0] < np.log10(v) <= logt[-1]})
        ax.set_yticks(np.log10(tk))
        ax.set_yticklabels([f"{v:,}" for v in tk])
        ax.set_ylim(logt[0] - 0.25, logt[-1] + 0.25)
        ax.set_xlabel("log10 c_total (Pile count)")
        ax.set_ylabel("onset step (log scale)")
        ax.set_title(f"Pythia-{size}: onset of Δ_pass(I) < 0 (▽ ≤ {int(10 ** logt[0])}, △ > {int(round(10 ** bnd)):,}"
                     + ("; hollow: interpolation uses a † step" if fi.any() else "") + ")", fontsize=9, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(figd / f"a8_f2_onset_{size}.{ext}", dpi=200, facecolor=SURF)
        plt.close(fig)
    # F3: late-window slopes
    fig, ax = plt.subplots(figsize=(6.4, 3.4), facecolor=SURF)
    _style(ax)
    names = ["β_b (between verbs)", "β_w (within verb)", "β_b − β_w", "β_b − β_w, calibrated"]
    sizes = [s for s in ("1.4b", "410m", "6.9b") if s in fd]
    for k, s in enumerate(sizes):
        flagged = fd[s]["late"][4]
        for i, arr in enumerate(fd[s]["late"][:4]):
            y = i + (k - (len(sizes) - 1) / 2) * 0.22
            lo, hi = ci(arr)
            ax.plot([lo, hi], [y, y], color=MODEL[s], lw=2, solid_capstyle="round")
            ax.scatter([arr[0]], [y], s=30, zorder=3, facecolors=SURF if flagged else MODEL[s], edgecolors=MODEL[s],
                       label=(f"Pythia-{s}" + (" (window has a † step)" if flagged else "")) if i == 0 else None)
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvline(A1_SLOPE, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_xlim(min(ax.get_xlim()[0], A1_SLOPE - 0.1), max(ax.get_xlim()[1], 0.1))
    ax.text(A1_SLOPE, len(names) - 0.45, " A1 −0.73", color=INK2, fontsize=7, ha="left", va="center")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_ylim(len(names) - 0.2, -0.45)
    ax.set_xlabel("nats per decade (log10 c_total or log10 E), 95% CI")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title("Late window (steps ≥ 16000): between- vs within-verb slope", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figd / f"a8_f3_late_slopes.{ext}", dpi=200, facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/round6/frame_d11_freq")
    ap.add_argument("--res-dir", default="results/round6/checkpoints")
    ap.add_argument("--counts", default="data/round6/counts/pile_counts.csv")
    ap.add_argument("--out-dir", default="results/round7/trajectories")
    ap.add_argument("--report", default="reports/round7/a8_trajectories.md")
    ap.add_argument("--fig-dir", default="reports/round7/figures")
    ap.add_argument("--sizes", nargs="+", default=["1.4b", "410m", "6.9b"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--skip-a6-check", action="store_true", help="synthetic tests only: do not stop when A6's diagnostic differs")
    main(ap.parse_args())
