#!/usr/bin/env python3
"""Round 4, D12: translation gain per pair (reports/round4/plan.md, D12): analysis.

From `run_translation_gain.py` (per-prompt direct effects on the centered " by" logit). Per item (pair x
context): TG = sum over MLPs 11-17 of (good - bad). Per pair: TG; gain = TG / site-8 z gap (pairs with z gap
>= 0.1); TG_res = TG - beta * z gap (beta: OLS across pairs). Analyses as round-3 A1 (`analyze_reliability`):
split-half reliability over contexts (1,000 splits), slope on participle Zipf (pair bootstrap), correlations
with the three behaviours with disattenuation and the A1 verdict rule. Decision rules: plan.md, D12.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_frequency import zipf_table
from analyze_reliability import BEH, Measure, corr, load_measures, med_rel, reliability, verdict

MLPS = tuple(range(11, 18))


class Ratio:
    """gain halves: TG half mean / z gap half mean on the same split (both Measures share items and strata)."""

    def __init__(self, num, den, keep):
        self.num, self.den, self.keep = num, den, keep
        self.n_pairs = num.n_pairs
        self.full = np.where(keep, num.full / den.full, np.nan)
        self.has = keep & num.has

    def halves(self, rng):
        st = rng.bit_generator.state
        a = self.num.halves(rng)
        rng.bit_generator.state = st  # the same split for the denominator
        b = self.den.halves(rng)
        return [np.where(self.keep, x / y, np.nan) for x, y in zip(a, b)]


def slope(x, y, B, rng):
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    est = np.polyfit(x, y, 1)[0]
    bs = [np.polyfit(x[ix], y[ix], 1)[0] for ix in (rng.integers(0, len(x), len(x)) for _ in range(B))]
    return {"est": float(est), "lo95": float(np.percentile(bs, 2.5)), "hi95": float(np.percentile(bs, 97.5)), "n": int(len(x))}


def coef_beyond(zipf, tg, zg, B, rng):
    """Zipf coefficient in TG ~ 1 + z gap + Zipf (pair bootstrap, refit per draw)."""
    def fit(ix):
        X = np.column_stack([np.ones(len(ix)), zg[ix], zipf[ix]])
        return np.linalg.lstsq(X, tg[ix], rcond=None)[0][2]
    n = len(tg)
    bs = [fit(rng.integers(0, n, n)) for _ in range(B)]
    return {"est": float(fit(np.arange(n))), "lo95": float(np.percentile(bs, 2.5)), "hi95": float(np.percentile(bs, 97.5))}


def partial(a, b, c):
    """Partial correlation of a and b given c."""
    ra, rb = a - np.polyval(np.polyfit(c, a, 1), c), b - np.polyval(np.polyfit(c, b, 1), c)
    return float(np.corrcoef(ra, rb)[0, 1])


def run(args):
    measures, prim, X = load_measures(args, [8])
    items = pd.read_csv(args.items)
    it = items[items.pair_id.isin(prim)]
    tg = pd.read_parquet(Path(args.dir) / "translation_gain.parquet").set_index("prompt")
    cols = [f"mlp{l}" for l in MLPS]
    item_tg = tg.loc[it.good_prompt, cols].to_numpy() - tg.loc[it.bad_prompt, cols].to_numpy()
    TG = Measure("TG", it.pair_id, it.context_band, item_tg.sum(1), prim)
    per_mlp = {l: Measure(f"TG_mlp{l}", it.pair_id, it.context_band, item_tg[:, j], prim) for j, l in enumerate(MLPS)}
    by_all = tg.loc[it.good_prompt, "by_actual"].to_numpy() - tg.loc[it.bad_prompt, "by_actual"].to_numpy()
    BYG = Measure("by_logit_gap", it.pair_id, it.context_band, by_all, prim)
    zg = measures["passive_gap_s8"]
    # the z gap Measure and TG must share items, order and strata for the ratio halves
    assert len(zg.v) == len(TG.v), "z gap and TG item sets differ"
    keep = zg.full >= 0.1
    gain = Ratio(TG, zg, keep)
    beta = float(np.polyfit(zg.full, TG.full, 1)[0])
    zitem = zg.v  # per-item z gap in the same item order as frame_z (checked below)
    TGres = Measure("TG_res", it.pair_id, it.context_band, item_tg.sum(1) - beta * zitem, prim)
    zt = zipf_table(args, prim)
    zipf = zt.part_zipf.loc[prim].to_numpy()
    rng = np.random.default_rng(args.seed)
    ms = {"TG": TG, "gain": gain, "TG_res": TGres, "by logit gap (all components)": BYG, "z gap (site 8)": zg}
    rel, sl, cr = [], [], []
    for name, m in ms.items():
        idx = np.flatnonzero(m.has)
        R, _ = med_rel(m, idx, args.seed, "pearson", args.n_splits)
        rel.append({"measure": name, "pairs": len(idx), "R": R})
        sl.append({"measure": name, **slope(zipf, m.full, args.n_boot, rng)})
        for b in BEH:
            mb = measures[b]
            ix = np.flatnonzero(m.has & mb.has)
            ro = corr(m.full[ix], mb.full[ix], "pearson")
            Rm, _ = med_rel(m, ix, args.seed, "pearson", args.n_splits // 4)
            Rb, _ = med_rel(mb, ix, args.seed, "pearson", args.n_splits // 4)
            rb = np.random.default_rng(args.seed + 2)
            draws, robs = [], []
            for _ in range(args.n_boot):
                bi = ix[rb.integers(0, len(ix), len(ix))]
                r1, r2 = reliability(m, bi, rb, "pearson"), reliability(mb, bi, rb, "pearson")
                o = corr(m.full[bi], mb.full[bi], "pearson")
                robs.append(o)
                if r1 > 0 and r2 > 0:
                    draws.append(o / np.sqrt(r1 * r2))
            lo, hi = np.percentile(draws, [2.5, 97.5])
            olo, ohi = np.percentile(robs, [2.5, 97.5])
            cr.append({"measure": name, "behaviour": b, "pairs": len(ix), "r_obs": ro, "r_lo95": olo, "r_hi95": ohi,
                       "R_measure": Rm, "R_beh": Rb, "r_disatt": ro / np.sqrt(Rm * Rb), "lo95": lo, "hi95": hi,
                       "verdict": verdict(Rm, Rb, olo, ohi, lo, hi)})
    beyond = coef_beyond(zipf, TG.full, zg.full, args.n_boot, rng)
    comp = []
    for b in BEH:
        mb = measures[b]
        ix = np.flatnonzero(TG.has & mb.has & zg.has)
        rt, rz = corr(TG.full[ix], mb.full[ix], "pearson"), corr(zg.full[ix], mb.full[ix], "pearson")
        pr_ = partial(TG.full[ix], mb.full[ix], zg.full[ix])
        dd, pp = [], []
        for _ in range(args.n_boot):
            bi = ix[rng.integers(0, len(ix), len(ix))]
            dd.append(corr(TG.full[bi], mb.full[bi], "pearson") - corr(zg.full[bi], mb.full[bi], "pearson"))
            pp.append(partial(TG.full[bi], mb.full[bi], zg.full[bi]))
        comp.append({"behaviour": b, "pairs": len(ix), "r_TG": rt, "r_zgap": rz, "diff": rt - rz,
                     "diff_lo95": float(np.percentile(dd, 2.5)), "diff_hi95": float(np.percentile(dd, 97.5)),
                     "partial_TG_given_z": pr_, "partial_lo95": float(np.percentile(pp, 2.5)),
                     "partial_hi95": float(np.percentile(pp, 97.5)), "part_whole": b == "by_pref"})
    per = pd.DataFrame({"pair_id": prim, "zipf": zipf, "TG": TG.full, "gain": gain.full, "TG_res": TGres.full,
                        "z_gap": zg.full, **{f"TG_mlp{l}": per_mlp[l].full for l in MLPS}})
    out = Path(args.dir)
    pd.DataFrame(rel).to_csv(out / "tg_reliability.csv", index=False)
    pd.DataFrame(sl).to_csv(out / "tg_slopes.csv", index=False)
    pd.DataFrame(cr).to_csv(out / "tg_correlations.csv", index=False)
    pd.DataFrame(comp).to_csv(out / "tg_vs_dgap.csv", index=False)
    per.to_csv(out / "tg_per_pair.csv", index=False)
    write_report(args, pd.DataFrame(rel), pd.DataFrame(sl), pd.DataFrame(cr), per, beta, beyond, pd.DataFrame(comp))


def write_report(args, rel, sl, cr, per, beta, beyond, comp):
    f = lambda r: f"{r.est:+.3f} [{r.lo95:+.3f}, {r.hi95:+.3f}]"
    L = ["# Round 4, D12. Translation gain per pair", "",
         "Spec: `plan.md`, D12. Run: `run_translation_gain.py` (fp32, natural pass); analysis: "
         "`analyze_translation_gain.py`. 64 primary pairs. TG = direct effect of MLPs 11–17 on the centered \" by\" logit "
         "(through the final-LN scale), good − bad, mean over contexts. gain = TG / site-8 z gap (pairs with z gap ≥ 0.1); "
         f"TG_res = TG − {beta:.3f} × z gap. CIs: pair bootstrap (2,000 draws).", "",
         f"Means: TG {per.TG.mean():+.3f}; z gap {per.z_gap.mean():+.3f}; gain {np.nanmean(per.gain):+.3f}. "
         "Per MLP (mean TG): " + ", ".join(f"MLP{l} {per[f'TG_mlp{l}'].mean():+.3f}" for l in MLPS) + ".", "",
         "| Measure | split-half R | slope on participle Zipf |", "|---|---:|---|"]
    for r in rel.itertuples():
        s = sl[sl.measure == r.measure].iloc[0]
        L.append(f"| {r.measure} | {r.R:.3f} | {f(s)} (n {s.n}) |")
    L += ["", "| Measure | Behaviour | pairs | r observed [95% CI] | r disattenuated [95% CI] | verdict |", "|---|---|---:|---|---|---|"]
    for r in cr.itertuples():
        L.append(f"| {r.measure} | {BEH[r.behaviour]} | {r.pairs} | {r.r_obs:+.2f} [{r.r_lo95:+.2f}, {r.r_hi95:+.2f}] | "
                 f"{r.r_disatt:+.2f} [{r.lo95:+.2f}, {r.hi95:+.2f}] | {r.verdict} |")
    L += ["", "TG against the d gap as a predictor of behaviour (paired bootstrap):", "",
          "| Behaviour | pairs | r(TG) | r(z gap) | r(TG) − r(z gap) [95% CI] | partial r(TG, behaviour | z gap) [95% CI] |",
          "|---|---:|---:|---:|---|---|"]
    for r in comp.itertuples():
        L.append(f"| {BEH[r.behaviour]}{' (part-whole: TG is a component of this readout)' if r.part_whole else ''} | "
                 f"{r.pairs} | {r.r_TG:+.2f} | {r.r_zgap:+.2f} | {r.diff:+.2f} [{r.diff_lo95:+.2f}, {r.diff_hi95:+.2f}] | "
                 f"{r.partial_TG_given_z:+.2f} [{r.partial_lo95:+.2f}, {r.partial_hi95:+.2f}] |")
    s_tg = sl[sl.measure == "TG"].iloc[0]
    tracks, specific = s_tg.lo95 > 0, beyond["lo95"] > 0
    better = comp[(~comp.part_whole) & (comp.diff_lo95 > 0) & (comp.partial_lo95 > 0)].behaviour.tolist()
    L += ["", "## Declared decisions", "",
          f"- **TG tracks frequency:** {'yes' if tracks else 'no'} (slope {f(s_tg)}).",
          f"- **Frequency dependence of TG beyond the d gap** (Zipf coefficient in TG ~ z gap + Zipf, refit per draw): "
          f"{'yes' if specific else 'no'} ({beyond['est']:+.3f} [{beyond['lo95']:+.3f}, {beyond['hi95']:+.3f}]).",
          f"- **TG predicts behaviour beyond the d gap** (paired Δr and partial r both above 0; part-whole readout "
          f"excluded): {better or 'none'}.",
          "- Reading (observational): " + ("rare verbs translate less per unit of d gap" if specific else
                                          "no evidence of a frequency dependence of translation beyond the d gap "
                                          "(this does not show there is none)") + ".", ""]
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/round4/translation")
    ap.add_argument("--root", default="results/das_round2")
    ap.add_argument("--items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--prompts", default="data/das_round2/passive_test/prompts.csv")
    ap.add_argument("--natural", default="results/das_round2/passive_test/natural.parquet")
    ap.add_argument("--passives", default="data/passive_das/passives.jsonl")
    ap.add_argument("--scores", default="results/passive_band_cross/pythia14b_scores.csv")
    ap.add_argument("--band-cross", default="data/passive_band_cross/pairs.jsonl")
    ap.add_argument("--eval-pairs", default="data/verb_expansion/eval_pairs.csv")
    ap.add_argument("--proj-dirs", default="results/das_round2/passive_test,results/das_round2/passive_test_fill")
    ap.add_argument("--report", default="reports/round4/d12_translation_gain.md")
    ap.add_argument("--n-splits", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
