#!/usr/bin/env python3
"""Split-half reliability of the per-pair measures (round3_plan.md, A1).

Per primary pair, each measure is a mean over contexts (subjects for the
active gap). Contexts are split at random into halves, stratified by context
band (alternating assignment within (pair, band) after a random shuffle), the
per-pair mean is taken in each half, and the halves are correlated across
pairs; Spearman-Brown R = 2r / (1 + r).

- R per measure: median over `--n-splits` random splits (and the 2.5-97.5%
  range over splits); pair-bootstrap 95% CI (`--n-boot` draws, each with a
  fresh split).
- Disattenuated correlation of the passive gap (each site) with each
  behavioural margin: r_obs / sqrt(R_gap R_beh), same bootstrap (each draw
  recomputes r_obs and both reliabilities on the resampled pairs).
- Pearson primary, Spearman secondary (rank correlation of halves, then SB).

Measures: passive gap and active gap at each site (`frame_z`), the single-
prompt " by" preference (`natural.parquet`), the released by margin (released
`passive_1` contexts, band-cross scores) and the curated passive_2 LP margin
(`passives.jsonl`).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_frequency import zipf_table
from analyze_passive_test import frame_z, populations

BEH = {"by_pref": 'single-prompt " by" preference', "by_margin_released": "released by margin (task 0)",
       "lp_margin": "curated passive_2 LP margin"}


class Measure:
    """Long table (pair, stratum, value) -> per-pair half means under random splits."""

    def __init__(self, name, pairs, strata, values, pair_order):
        self.name = name
        pix = {p: i for i, p in enumerate(pair_order)}
        keep = np.array([p in pix for p in pairs])
        self.p = np.array([pix[p] for p in np.asarray(pairs)[keep]])
        g = pd.Series(list(zip(self.p, np.asarray(strata)[keep]))).astype(str)
        self.g = pd.factorize(g)[0]
        self.v = np.asarray(values, float)[keep]
        self.n_pairs = len(pair_order)
        self.full = np.bincount(self.p, self.v, self.n_pairs) / np.bincount(self.p, minlength=self.n_pairs)
        self.has = np.bincount(self.p, minlength=self.n_pairs) > 0
        starts = np.r_[0, np.cumsum(np.bincount(self.g))[:-1]]
        self.starts = starts

    def halves(self, rng):
        order = np.lexsort((rng.random(len(self.v)), self.g))
        pos = np.arange(len(order)) - self.starts[self.g[order]]
        half = np.empty(len(order), int)
        # alternate within each stratum; randomise which half gets the odd item per stratum
        flip = rng.integers(0, 2, self.g.max() + 1)
        half[order] = (pos + flip[self.g[order]]) % 2
        out = []
        for h in (0, 1):
            m = half == h
            n = np.bincount(self.p[m], minlength=self.n_pairs)
            out.append(np.where(n > 0, np.bincount(self.p[m], self.v[m], self.n_pairs) / np.maximum(n, 1), np.nan))
        return out


def rank(x):
    return pd.Series(x).rank().to_numpy()


def corr(a, b, method):
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if method == "spearman":
        a, b = rank(a), rank(b)
    return float(np.corrcoef(a, b)[0, 1])


def sb(r):
    return 2 * r / (1 + r)


def residualize(x, X):
    beta, *_ = np.linalg.lstsq(X, x, rcond=None)
    return x - X @ beta


def reliability(m, idx, rng, method, X=None):
    """Spearman-Brown split-half reliability over pairs idx; X: optional design (rows = pairs) to residualize on."""
    a, b = m.halves(rng)
    a, b = a[idx], b[idx]
    if X is not None:
        Xi = X[idx]
        a, b = residualize(a, Xi), residualize(b, Xi)
    return sb(corr(a, b, method))


def med_rel(m, idx, seed, method, n, X=None):
    rng = np.random.default_rng(seed)
    s = np.array([reliability(m, idx, rng, method, X) for _ in range(n)])
    return float(np.median(s)), s


def load_measures(args, site_list):
    items = pd.read_csv(args.items)
    P = pd.read_csv(args.prompts)
    root = Path(args.root)
    pairs_folds = pd.read_csv(root / "final_strict" / "pairs_folds.csv")
    das_items = pd.read_csv(root / "final_strict" / "items.csv")
    pr, pops = populations(items)
    prim = list(pr.index[pops["primary (plain)"].to_numpy()])
    it = items[items.pair_id.isin(prim)]
    pidx = dict(zip(P.prompt, P.pid))
    measures = {}
    for site in site_list:
        f = next((Path(d) / f"projections_site{site}.parquet" for d in args.proj_dirs.split(",")
                  if (Path(d) / f"projections_site{site}.parquet").exists()))
        itz, at = frame_z(f, items, P, pairs_folds, das_items, pr)
        itz = itz[itz.pair_id.isin(prim)]
        at = at[at.pair_id.isin(prim)]
        g = Measure(f"passive_gap_s{site}", itz.pair_id, itz.context_band, itz["diff"], prim)
        g.by_split = []  # consistency across the three basis splits
        for k in range(3):
            itk, _ = frame_z(f, items, P, pairs_folds, das_items, pr, splits=(k,))
            g.by_split.append(itk.groupby("pair_id")["diff"].mean().reindex(prim).to_numpy())
        measures[g.name] = g
        measures[f"active_gap_s{site}"] = Measure(f"active_gap_s{site}", at.pair_id, np.zeros(len(at)), at["diff"], prim)
    nat = pd.read_parquet(Path(args.natural)).set_index("pid")
    byp = nat.by.loc[it.good_prompt.map(pidx)].to_numpy() - nat.by.loc[it.bad_prompt.map(pidx)].to_numpy()
    measures["by_pref"] = Measure("by_pref", it.pair_id, it.context_band, byp, prim)
    sc = pd.read_csv(args.scores, low_memory=False,
                     usecols=["verb_pair", "paradigm", "context_set", "context_band", "by_margin"])
    rel = sc[(sc.paradigm == "passive_1") & (sc.context_set == "released")]
    measures["by_margin_released"] = Measure("by_margin_released", rel.verb_pair, rel.context_band, rel.by_margin, prim)
    pas = pd.read_json(args.passives, lines=True, dtype={"item_id": str})
    measures["lp_margin"] = Measure("lp_margin", pas.verb_pair, pas.context_band, pas.lp_whole_margin_p2, prim)
    zt = zipf_table(args, prim)
    band = pr.band.loc[prim]
    X = np.column_stack([np.ones(len(prim)), zt.part_zipf.loc[prim].to_numpy(),
                         (band == "tail").to_numpy(float), (band == "xtail").to_numpy(float)])
    return measures, prim, X


def verdict(Rg, Rb, ro_lo, ro_hi, lo, hi):
    if min(Rg, Rb) < 0.6:
        return "measurement-limited"
    sig = "positive" if ro_lo > 0 else ("negative" if ro_hi < 0 else "")
    if sig:
        return f"significant {sig} link" + ("; no moderate link" if -0.3 < lo and hi < 0.3 else "")
    if -0.3 < lo and hi < 0.3:
        return "no moderate link"
    return "no detectable link; moderate not excluded"


def run(args):
    sites = [int(s) for s in args.sites.split(",")]
    measures, prim, X = load_measures(args, sites)
    n = len(prim)
    rel_rows, dis_rows, cons_rows = [], [], []
    for method in ("pearson", "spearman"):
        for name, m in measures.items():
            idx = np.flatnonzero(m.has)
            variants = [("raw", None)] + ([("zipf_band_residual", X)] if name.startswith("passive_gap") else [])
            for variant, Xv in variants:
                R, splits = med_rel(m, idx, args.seed, method, args.n_splits, Xv)
                rngb = np.random.default_rng(args.seed + 1)
                boots = np.array([reliability(m, idx[rngb.integers(0, len(idx), len(idx))], rngb, method,
                                              None if Xv is None else Xv) for _ in range(args.n_boot)])
                rel_rows.append({"measure": name, "variant": variant, "method": method, "n_pairs": len(idx),
                                 "units_per_pair": float(np.bincount(m.p).mean()), "R_median": R,
                                 "R_split_lo": float(np.percentile(splits, 2.5)),
                                 "R_split_hi": float(np.percentile(splits, 97.5)),
                                 "R_boot_lo": float(np.nanpercentile(boots, 2.5)),
                                 "R_boot_hi": float(np.nanpercentile(boots, 97.5)),
                                 "frac_boot_nonpositive": float((boots <= 0).mean())})
            if hasattr(m, "by_split"):
                s = m.by_split
                rs = [corr(s[a], s[b], method) for a, b in ((0, 1), (0, 2), (1, 2))]
                r = float(np.mean(rs))
                cons_rows.append({"measure": name, "method": method, "r_01": rs[0], "r_02": rs[1], "r_12": rs[2],
                                  "mean_r": r, "SB_3": 3 * r / (1 + 2 * r)})
        # disattenuated correlations, passive gap x behaviour; reliabilities on each test's exact population
        for site in sites:
            g = measures[f"passive_gap_s{site}"]
            for b in BEH:
                mb = measures[b]
                idx = np.flatnonzero(g.has & mb.has)
                r_obs = corr(g.full[idx], mb.full[idx], method)
                Rg, _ = med_rel(g, idx, args.seed, method, args.n_splits)
                Rb, _ = med_rel(mb, idx, args.seed, method, args.n_splits)
                rngb = np.random.default_rng(args.seed + 2)
                draws, robs, nonpos = [], [], 0
                for _ in range(args.n_boot):
                    bi = idx[rngb.integers(0, len(idx), len(idx))]
                    rg, rb = reliability(g, bi, rngb, method), reliability(mb, bi, rngb, method)
                    ro = corr(g.full[bi], mb.full[bi], method)
                    robs.append(ro)
                    if rg > 0 and rb > 0:
                        draws.append(ro / np.sqrt(rg * rb))
                    else:
                        nonpos += 1
                draws, robs = np.array(draws), np.array(robs)
                lo, hi = np.percentile(draws, [2.5, 97.5])
                ro_lo, ro_hi = np.percentile(robs, [2.5, 97.5])
                dis_rows.append({"site": site, "behaviour": b, "method": method, "n_pairs": len(idx),
                                 "r_obs": r_obs, "r_obs_lo95": ro_lo, "r_obs_hi95": ro_hi, "R_gap": Rg, "R_beh": Rb,
                                 "ceiling": float(np.sqrt(Rg * Rb)), "r_disatt": r_obs / np.sqrt(Rg * Rb),
                                 "lo95": lo, "hi95": hi, "frac_draws_nonpositive_R": nonpos / args.n_boot,
                                 "verdict": verdict(Rg, Rb, ro_lo, ro_hi, lo, hi)})
    rel, dis, cons = pd.DataFrame(rel_rows), pd.DataFrame(dis_rows), pd.DataFrame(cons_rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rel.to_csv(out / "reliability.csv", index=False)
    dis.to_csv(out / "reliability_disattenuated.csv", index=False)
    cons.to_csv(out / "reliability_across_bases.csv", index=False)
    write_report(args, rel, dis, cons, n)


def write_report(args, rel, dis, cons, n):
    L = ["# A1. Split-half reliability of per-pair measures", "",
         "Spec: `round3_plan.md`, A1. Script: `analyze_reliability.py`. "
         f"{n} primary pairs. Halves stratified by context band; R = Spearman-Brown of the half correlation; "
         f"median over {args.n_splits} random splits; 95% CI from a pair bootstrap ({args.n_boot} draws, fresh split "
         "per draw). R measures stability across sampled contexts.", "", "## Reliability", "",
         "| Measure | Variant | Pairs | Units per pair | R (Pearson) | range over splits | 95% CI (pairs) | R (Spearman) |",
         "|---|---|---:|---:|---:|---|---|---:|"]
    pe = rel[rel.method == "pearson"].set_index(["measure", "variant"])
    sp = rel[rel.method == "spearman"].set_index(["measure", "variant"])
    for key, r in pe.iterrows():
        L.append(f"| {key[0]} | {key[1]} | {int(r.n_pairs)} | {r.units_per_pair:.0f} | {r.R_median:.3f} | "
                 f"[{r.R_split_lo:.3f}, {r.R_split_hi:.3f}] | [{r.R_boot_lo:.3f}, {r.R_boot_hi:.3f}] | "
                 f"{sp.loc[key].R_median:.3f} |")
    L += ["", "`zipf_band_residual`: both halves residualized on participle Zipf and band across pairs before "
          "correlating (reliability of the part of the gap that frequency does not explain).", "",
          "## Consistency of the passive gap across basis splits", "",
          "Per-pair passive gap computed with one split seed's bases only (cross-fitted within the split).", "",
          "| Measure | r(0,1) | r(0,2) | r(1,2) | mean r | Spearman-Brown for 3 |", "|---|---:|---:|---:|---:|---:|"]
    for r in cons[cons.method == "pearson"].itertuples():
        L.append(f"| {r.measure} | {r.r_01:.3f} | {r.r_02:.3f} | {r.r_12:.3f} | {r.mean_r:.3f} | {r.SB_3:.3f} |")
    L += ["", "## Passive gap vs behaviour, disattenuated (Pearson primary)", "",
          "Reliabilities on each test's exact pair population. Declared rule: measurement-limited if either R < 0.6; "
          "significance from the CI of r observed; no moderate link if the whole disattenuated CI lies in "
          "[-0.3, 0.3]; otherwise no detectable link, moderate not excluded. The single-prompt \" by\" preference "
          "shares contexts with the gap, so its errors may correlate with the gap's.", "",
          "| Site | Behaviour | Pairs | r observed [95% CI] | R gap | R beh. | ceiling | r disattenuated [95% CI] | Verdict | "
          "Spearman disatt. [CI] |", "|---:|---|---:|---|---:|---:|---:|---|---|---|"]
    dp = dis[dis.method == "pearson"].set_index(["site", "behaviour"])
    ds = dis[dis.method == "spearman"].set_index(["site", "behaviour"])
    for (site, b), r in dp.iterrows():
        s = ds.loc[(site, b)]
        L.append(f"| {site} | {BEH[b]} | {int(r.n_pairs)} | {r.r_obs:+.2f} [{r.r_obs_lo95:+.2f}, {r.r_obs_hi95:+.2f}] | "
                 f"{r.R_gap:.2f} | {r.R_beh:.2f} | {r.ceiling:.2f} | {r.r_disatt:+.2f} [{r.lo95:+.2f}, {r.hi95:+.2f}] | "
                 f"{r.verdict} | {s.r_disatt:+.2f} [{s.lo95:+.2f}, {s.hi95:+.2f}] |")
    L += ["", "Verdict counts (Pearson): " + ", ".join(f"{k}: {v}" for k, v in dp.verdict.value_counts().items())]
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results/das_round2")
    ap.add_argument("--items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--prompts", default="data/das_round2/passive_test/prompts.csv")
    ap.add_argument("--natural", default="results/das_round2/passive_test/natural.parquet")
    ap.add_argument("--passives", default="data/passive_das/passives.jsonl")
    ap.add_argument("--scores", default="results/passive_band_cross/pythia14b_scores.csv")
    ap.add_argument("--band-cross", default="data/passive_band_cross/pairs.jsonl")
    ap.add_argument("--eval-pairs", default="data/verb_expansion/eval_pairs.csv")
    ap.add_argument("--proj-dirs", default="results/das_round2/passive_test,results/das_round2/passive_test_fill")
    ap.add_argument("--sites", default="4,6,8,10,12,14,16,17")
    ap.add_argument("--out-dir", default="results/das_round2/round3")
    ap.add_argument("--report", default="reports/passive_das_prep/round3_reliability.md")
    ap.add_argument("--n-splits", type=int, default=1000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
