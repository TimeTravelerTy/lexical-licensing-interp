#!/usr/bin/env python3
"""Robustness of low early IIA (round3_plan.md, A4).

Token count: held-out cross-class swaps of each site's rank-1 run
(`heldout_swaps.csv.gz`), split by the pair's token count. Each base pair's
value is the mean over its swaps (IIA; gap fraction clipped to [-2, 3] as in
`run_das_round2.summarize`), per swap direction; pairs are then averaged
within group. Primary: expansion pairs only (single vs multi); secondary: all
pairs. CI: pair bootstrap within group (2,000 draws, seed 17). The natural
class margin |M| of the base items is reported per group.

Rank: `rank_sweep_site{4,6}/summary.csv` (ranks 1, 2, 4; same folds and
seeds). The declared rank rule as recorded in `final_meta.json`, and the
held-out cross-class IIA gain of ranks 2 and 4 over rank 1, paired by
fold x split (bootstrap over the 15 runs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DIRS = {0: "I←T (intransitive base)", 1: "T←I (transitive base)"}


def site_dir(root, site):
    return root / ("final_strict" if site == 17 else f"final_strict_site{site}")


def boot_mean(x, n, rng):
    x = np.asarray(x)
    return np.r_[x.mean(), x[rng.integers(0, len(x), (n, len(x)))].mean(1)]


def ci(d):
    return d[0], np.percentile(d[1:], 2.5), np.percentile(d[1:], 97.5)


def token_split(args, root, rng):
    rows = []
    for site in [int(s) for s in args.sites.split(",")]:
        f = site_dir(root, site) / "heldout_swaps.csv.gz"
        if not f.exists():
            print(f"missing {f}")
            continue
        h = pd.read_csv(f)
        h = h[(h["rank"] == 1) & (h.control == "das") & h.cross]
        h["frac_c"] = h.frac.clip(-2, 3)
        for cls, g in h.groupby("base_cls"):
            pp = g.groupby("base_pair_id").agg(iia=("iia", "mean"), frac=("frac_c", "mean"),
                                                margin=("M_base", lambda m: np.abs(m).mean()),
                                                tok=("base_tok", "first"), source=("base_source", "first"))
            for pop, sel in (("expansion", pp.source == "expansion"), ("all", pd.Series(True, index=pp.index))):
                q = pp[sel]
                draws = {}
                for tok in ("single", "multi"):
                    t = q[q.tok == tok]
                    draws[tok] = {m: boot_mean(t[m], args.n_boot, rng) for m in ("iia", "frac", "margin")}
                    rows.append({"site": site, "direction": DIRS[cls], "population": pop, "group": tok,
                                 "n_pairs": len(t), **{f"{m}_{k}": v for m in ("iia", "frac", "margin")
                                                       for k, v in zip(("est", "lo", "hi"), ci(draws[tok][m]))}})
                rows.append({"site": site, "direction": DIRS[cls], "population": pop, "group": "multi − single",
                             "n_pairs": len(q), **{f"{m}_{k}": v for m in ("iia", "frac", "margin")
                                                   for k, v in zip(("est", "lo", "hi"),
                                                                   ci(draws["multi"][m] - draws["single"][m]))}})
    return pd.DataFrame(rows)


def rank_sweep(args, root, rng):
    rows, rules = [], {}
    for site in (4, 6):
        d = root / f"rank_sweep_site{site}"
        if not (d / "summary.csv").exists():
            print(f"missing {d}")
            continue
        s = pd.read_csv(d / "summary.csv")
        s = s[s.control == "das"].set_index(["rank", "split", "fold"])
        meta = json.loads((d / "final_meta.json").read_text())
        rules[site] = {"chosen_rank": meta["chosen_rank"], "pc_frac_by_rank": meta["pc_frac_by_rank"]}
        for r in (2, 4):
            for m in ("iia_cross", "pc_frac_median"):
                gain = (s.loc[r][m] - s.loc[1][m]).to_numpy()
                e, lo, hi = ci(boot_mean(gain, args.n_boot, rng))
                rows.append({"site": site, "rank": r, "metric": m, "rank1_mean": float(s.loc[1][m].mean()),
                             "rank_mean": float(s.loc[r][m].mean()), "gain": e, "lo95": lo, "hi95": hi})
    return pd.DataFrame(rows), rules


def run(args):
    root = Path(args.root)
    rng = np.random.default_rng(args.seed)
    tok = token_split(args, root, rng)
    rk, rules = rank_sweep(args, root, rng)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok.to_csv(out / "token_iia.csv", index=False)
    rk.to_csv(out / "rank_sweep_gains.csv", index=False)
    f = lambda r, m, k=2: f"{r[m + '_est']:.{k}f} [{r[m + '_lo']:.{k}f}, {r[m + '_hi']:.{k}f}]"
    L = ["# A4. Robustness of low early IIA", "", "Spec: `round3_plan.md`, A4. Script: `analyze_token_iia.py`.", "",
         "## Rank sweep (sites 4 and 6)", ""]
    for site, r in rules.items():
        L.append(f"- Site {site}: mean held-out median pc fraction by rank "
                 + ", ".join(f"{k}: {v:.3f}" for k, v in r["pc_frac_by_rank"].items())
                 + f"; rank rule chooses **rank {r['chosen_rank']}**.")
    if len(rk):
        L += ["", "| Site | Rank | Metric | Rank 1 | This rank | Paired gain [95% CI] |", "|---:|---:|---|---:|---:|---|"]
        for r in rk.itertuples():
            L.append(f"| {r.site} | {r.rank} | {r.metric} | {r.rank1_mean:.3f} | {r.rank_mean:.3f} | "
                     f"{r.gain:+.3f} [{r.lo95:+.3f}, {r.hi95:+.3f}] |")
    for pop in ("expansion", "all"):
        L += ["", f"## Token count, {'expansion pairs only (primary)' if pop == 'expansion' else 'all pairs (secondary)'}",
              "", "| Site | Direction | Group | Pairs | IIA | Gap fraction | Natural \\|M\\| |", "|---:|---|---|---:|---|---|---|"]
        for r in tok[tok.population == pop].to_dict("records"):
            L.append(f"| {r['site']} | {r['direction']} | {r['group']} | {r['n_pairs']} | {f(r, 'iia')} | "
                     f"{f(r, 'frac')} | {f(r, 'margin')} |")
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results/das_round2")
    ap.add_argument("--sites", default="4,6,8,10,12,14,16,17")
    ap.add_argument("--out-dir", default="results/das_round2/round3")
    ap.add_argument("--report", default="reports/passive_das_prep/round3_token_rank.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
