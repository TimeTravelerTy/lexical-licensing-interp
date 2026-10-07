#!/usr/bin/env python3
"""Frequency and behaviour vs the passive-frame d gap (followup_plan.md, step 4).

Per primary pair (64; no verb repeats across pairs, so pair clusters are
verb clusters):
- passive z gap = mean over contexts of z(good passive) - z(bad passive);
- active z gap = mean over subjects of the same verbs in "<Subj> has <participle>";
- ratio = passive gap / active gap.
z is the cross-fitted, sign-aligned site coordinate of `analyze_passive_test.frame_z`.

Regressions: OLS of each outcome on the pair's participle Zipf (mean of the
two forms; secondary: lemma Zipf), slope per Zipf unit with a 95% pair
bootstrap CI (2,000 draws, seed 17).
Correlations (site 8 by default, every site reported): passive gap vs the
released by margin (task 0), the curated passive_2 whole-sentence LP margin
(mean over contexts) and the single-prompt " by" preference
log P(" by" | good) - log P(" by" | bad) from the unpatched readout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_passive_test import frame_z, populations


def zipf_table(args, pairs):
    bc = pd.read_json(args.band_cross, lines=True)
    bc = bc.drop_duplicates("verb_pair").set_index("verb_pair")
    ev = pd.read_csv(args.eval_pairs)
    ev["pair_id"] = "new/" + ev.band + "/" + ev.trans_lemma + "/" + ev.intrans_lemma
    ev = ev.set_index("pair_id")
    rows = []
    for p in pairs:
        if p in bc.index:
            r = bc.loc[p]
            rows.append((p, r.good_form_zipf, r.bad_form_zipf, r.good_lemma_zipf, r.bad_lemma_zipf))
        else:
            r = ev.loc[p]
            rows.append((p, r.trans_participle_zipf, r.intrans_participle_zipf, r.trans_lemma_zipf, r.intrans_lemma_zipf))
    z = pd.DataFrame(rows, columns=["pair_id", "good_part_zipf", "bad_part_zipf", "good_lemma_zipf", "bad_lemma_zipf"])
    z["part_zipf"] = (z.good_part_zipf + z.bad_part_zipf) / 2
    z["lemma_zipf"] = (z.good_lemma_zipf + z.bad_lemma_zipf) / 2
    return z.set_index("pair_id")


def ols_slope(x, y):
    x = x - x.mean()
    return float((x * (y - y.mean())).sum() / (x ** 2).sum())


def boot_pairs(df, fn, n, seed):
    rng = np.random.default_rng(seed)
    est = fn(df)
    draws = np.array([fn(df.iloc[rng.integers(0, len(df), len(df))]) for _ in range(n)])
    return est, np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def spearman(a, b):
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


def run(args):
    root = Path(args.root)
    items = pd.read_csv(args.items)
    P = pd.read_csv(args.prompts)
    pairs_folds = pd.read_csv(root / "final_strict" / "pairs_folds.csv")
    das_items = pd.read_csv(root / "final_strict" / "items.csv")
    pr, pops = populations(items)
    prim = pr.index[pops["primary (plain)"].to_numpy()]
    zt = zipf_table(args, prim)
    # behavioural margins
    nat = pd.read_parquet(Path(args.out_dir) / "natural.parquet").set_index("pid")
    pidx = dict(zip(P.prompt, P.pid))
    it = items[items.pair_id.isin(prim)]
    byp = pd.Series(nat.by.loc[it.good_prompt.map(pidx)].to_numpy() - nat.by.loc[it.bad_prompt.map(pidx)].to_numpy(),
                    index=it.pair_id.to_numpy()).groupby(level=0).mean()
    rel = pd.read_csv(args.reliability).set_index("verb_pair").by_margin
    pas = pd.read_json(args.passives, lines=True, dtype={"item_id": str})
    lp = pas.groupby("verb_pair").lp_whole_margin_p2.mean()

    sites = [int(s) for s in args.sites.split(",")]
    reg, cor = [], []
    per_pair = []
    for site in sites:
        f = next((Path(d) / f"projections_site{site}.parquet" for d in args.proj_dirs.split(",")
                  if (Path(d) / f"projections_site{site}.parquet").exists()), None)
        if f is None:
            continue
        itz, at = frame_z(f, items, P, pairs_folds, das_items, pr)
        g = pd.DataFrame({"passive_gap": itz.groupby("pair_id")["diff"].mean(),
                          "active_gap": at.groupby("pair_id")["diff"].mean()}).loc[prim]
        g["ratio"] = g.passive_gap / g.active_gap
        g = g.join(zt).join(byp.rename("by_pref")).join(rel.rename("by_margin_released")).join(lp.rename("lp_margin"))
        g["band"] = pr.band.loc[g.index]
        per_pair.append(g.assign(site=site))
        for y in ("passive_gap", "active_gap", "ratio"):
            for x in ("part_zipf", "lemma_zipf"):
                est, lo, hi = boot_pairs(g, lambda d: ols_slope(d[x].to_numpy(), d[y].to_numpy()), args.n_boot, args.seed)
                reg.append({"site": site, "outcome": y, "predictor": x, "slope": est, "lo95": lo, "hi95": hi,
                            "n_pairs": len(g), "mean_outcome": float(g[y].mean())})
        for b in ("by_pref", "by_margin_released", "lp_margin"):
            d = g.dropna(subset=[b])
            for name, fn in (("spearman", spearman), ("pearson", lambda a, c: float(np.corrcoef(a, c)[0, 1]))):
                est, lo, hi = boot_pairs(d, lambda dd: fn(dd.passive_gap.to_numpy(), dd[b].to_numpy()), args.n_boot,
                                         args.seed)
                cor.append({"site": site, "behaviour": b, "method": name, "r": est, "lo95": lo, "hi95": hi,
                            "n_pairs": len(d)})
    reg, cor = pd.DataFrame(reg), pd.DataFrame(cor)
    out = Path(args.out_dir)
    reg.to_csv(out / "frequency_regressions.csv", index=False)
    cor.to_csv(out / "frequency_behaviour_correlations.csv", index=False)
    pd.concat(per_pair).to_csv(out / "frequency_per_pair.csv")

    L = ["# Frequency and behaviour vs the passive d gap (step 4)", "",
         "Spec: `followup_plan.md`, step 4. Script: `analyze_frequency.py`. 64 primary pairs (one observation per pair; "
         "verbs do not repeat). 95% CIs: pair bootstrap, 2,000 draws, seed 17. z: 0 = held-out active intransitive "
         "level, 1 = held-out active transitive level, at each site.", "",
         "## Slope on participle Zipf (per Zipf unit)", "",
         "| Site | Passive gap: mean | slope | Active gap: mean | slope | Passive/active ratio: mean | slope |",
         "|---:|---:|---|---:|---|---:|---|"]
    fmt = lambda r: f"{r.slope:+.3f} [{r.lo95:+.3f}, {r.hi95:+.3f}]"
    for site in sorted(reg.site.unique()):
        x = reg[(reg.site == site) & (reg.predictor == "part_zipf")].set_index("outcome")
        L.append(f"| {site} | {x.mean_outcome['passive_gap']:.2f} | {fmt(x.loc['passive_gap'])} | "
                 f"{x.mean_outcome['active_gap']:.2f} | {fmt(x.loc['active_gap'])} | {x.mean_outcome['ratio']:.2f} | "
                 f"{fmt(x.loc['ratio'])} |")
    L += ["", "Lemma Zipf as the predictor (secondary):", "",
          "| Site | Passive gap slope | Active gap slope | Ratio slope |", "|---:|---|---|---|"]
    for site in sorted(reg.site.unique()):
        x = reg[(reg.site == site) & (reg.predictor == "lemma_zipf")].set_index("outcome")
        L.append(f"| {site} | {fmt(x.loc['passive_gap'])} | {fmt(x.loc['active_gap'])} | {fmt(x.loc['ratio'])} |")
    zr = zt.part_zipf
    L += ["", f"Participle Zipf range over the 64 pairs: {zr.min():.2f}–{zr.max():.2f} (mean {zr.mean():.2f}).", "",
          "## Passive gap vs behavioural passive margins", "",
          "| Site | Behaviour | Pairs | Spearman | Pearson |", "|---:|---|---:|---|---|"]
    names = {"by_pref": 'single-prompt " by" preference', "by_margin_released": "released by margin (task 0)",
             "lp_margin": "curated passive_2 whole-sentence LP margin"}
    for (site, b), x in cor.groupby(["site", "behaviour"]):
        x = x.set_index("method")
        L.append(f"| {site} | {names[b]} | {int(x.n_pairs.iloc[0])} | {x.r['spearman']:+.2f} "
                 f"[{x.lo95['spearman']:+.2f}, {x.hi95['spearman']:+.2f}] | {x.r['pearson']:+.2f} "
                 f"[{x.lo95['pearson']:+.2f}, {x.hi95['pearson']:+.2f}] |")
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results/das_round2")
    ap.add_argument("--items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--prompts", default="data/das_round2/passive_test/prompts.csv")
    ap.add_argument("--passives", default="data/passive_das/passives.jsonl")
    ap.add_argument("--band-cross", default="data/passive_band_cross/pairs.jsonl")
    ap.add_argument("--eval-pairs", default="data/verb_expansion/eval_pairs.csv")
    ap.add_argument("--reliability", default="reports/passive_das_prep/task0_by_reliability.csv")
    ap.add_argument("--out-dir", default="results/das_round2/passive_test")
    ap.add_argument("--proj-dirs", default="results/das_round2/passive_test,results/das_round2/passive_test_fill")
    ap.add_argument("--sites", default="4,6,8,10,12,14,16,17")
    ap.add_argument("--report", default="reports/passive_das_prep/frequency_results.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
