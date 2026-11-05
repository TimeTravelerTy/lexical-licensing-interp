#!/usr/bin/env python3
"""Gemma replication: summarize held-out DAS IIA by participle token pattern (outputs of run_das_token_split.py).

Per model x task x site, cross-class held-out swaps (split 0, 5 folds pooled, the sweep's best epoch):
- all: every cross swap, pooled over folds (the sweep tables report the fold mean, so they can differ slightly);
- the model's own patterns: T1-I1 (both participles one token under that model's tokenizer; no length cue) and
  T1-I2+ (transitive one token, intransitive two or more);
- common single: both verbs one token under both tokenizers (Pythia-1.4B and Gemma-3-1B).
Matched comparison (Gemma − Pythia): common-single swaps restricted to the (base verb, source verb) pairs held out in
both models' runs (the active task's folds differ between models: Gemma's behaviour filter drops one pair, which
shifts run_das_round2.folds' round-robin deal), IIA averaged per verb pair, Pythia site p vs Gemma site
round(p * 26 / 24) (nearest relative depth); the bootstrap draws the same T / I verb weights for both models.
95% CIs: two-way bootstrap over transitive and intransitive verbs (each swap pairs one of each; swap weight =
product of the two verbs' multinomial counts), 2,000 draws, seed 17. Differences use the same draws.

Writes reports/gemma/das_token_split.md and results/gemma/das/token_split/summary.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = [("pythia-1.4b", "active", 24), ("gemma-3-1b-pt", "active", 26), ("pythia-1.4b", "passive", 24),
        ("gemma-3-1b-pt", "passive", 26)]
DIRS = {("pythia-1.4b", "active"): "pythia_active", ("gemma-3-1b-pt", "active"): "gemma_active",
        ("pythia-1.4b", "passive"): "pythia_passive", ("gemma-3-1b-pt", "passive"): "gemma_passive"}


class Boot:
    def __init__(self, t_verbs, i_verbs, n, rng):
        self.t = {v: k for k, v in enumerate(sorted(set(t_verbs)))}
        self.i = {v: k for k, v in enumerate(sorted(set(i_verbs)))}
        nt, ni = len(self.t), len(self.i)
        self.wt = np.vstack([np.ones(nt), rng.multinomial(nt, np.ones(nt) / nt, n)])
        self.wi = np.vstack([np.ones(ni), rng.multinomial(ni, np.ones(ni) / ni, n)])

    def mean(self, df):
        if not len(df):
            return np.full(self.wt.shape[0], np.nan)
        w = self.wt[:, df.t_verb.map(self.t).to_numpy()] * self.wi[:, df.i_verb.map(self.i).to_numpy()]
        y = df.iia.to_numpy(float)
        with np.errstate(invalid="ignore"):
            return (w * y).sum(1) / w.sum(1)


def ci(d):
    if np.isnan(d[0]):
        return (np.nan, np.nan, np.nan)
    x = d[1:][np.isfinite(d[1:])]
    return (float(d[0]), *np.percentile(x, [2.5, 97.5]))


def fmt(t, n=None):
    if np.isnan(t[0]):
        return "—"
    s = f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
    return s + (f" (n={n})" if n is not None else "")


def matched(swaps, rng, n_boot):
    out = []
    for task in ("active", "passive"):
        P, G = swaps.get(("pythia-1.4b", task)), swaps.get(("gemma-3-1b-pt", task))
        if P is None or G is None:
            continue
        P, G = P[P.common], G[G.common]
        keys = set(zip(P.base_verb, P.src_verb)) & set(zip(G.base_verb, G.src_verb))
        sel = lambda S: S[[k in keys for k in zip(S.base_verb, S.src_verb)]]
        agg = lambda S: sel(S).groupby(["site", "base_verb", "src_verb", "t_verb", "i_verb"]).iia.mean().reset_index()
        P, G = agg(P), agg(G)
        B = Boot(P.t_verb, P.i_verb, n_boot, rng)
        for ps in sorted(P.site.unique()):
            gs = int(round(ps * 26 / 24))
            if gs not in set(G.site):
                continue
            dp, dg = B.mean(P[P.site == ps]), B.mean(G[G.site == gs])
            r = {"task": task, "pythia_site": int(ps), "gemma_site": gs, "pythia_depth": ps / 24, "gemma_depth": gs / 26,
                 "n_pairs": len(keys), "nT": P.t_verb.nunique(), "nI": P.i_verb.nunique()}
            for k, d in (("pythia", dp), ("gemma", dg), ("diff", dg - dp)):
                r[f"{k}_est"], r[f"{k}_lo"], r[f"{k}_hi"] = ci(d)
            out.append(r)
    return pd.DataFrame(out)


def main(args):
    from transformers import AutoTokenizer

    toks = {m: AutoTokenizer.from_pretrained(h) for m, h in (("pythia-1.4b", "EleutherAI/pythia-1.4b"),
                                                             ("gemma-3-1b-pt", "google/gemma-3-1b-pt"))}
    pairs = pd.read_csv(args.pairs)
    part = dict(zip(pairs.trans, pairs.trans_part)) | dict(zip(pairs.intrans, pairs.intrans_part))
    single = {m: {lem for lem, p in part.items() if len(t.encode(" " + p, add_special_tokens=False)) == 1}
              for m, t in toks.items()}
    common = single["pythia-1.4b"] & single["gemma-3-1b-pt"]
    rng = np.random.default_rng(args.seed)
    rows, checks, swaps = [], [], {}
    for model, task, L in RUNS:
        d = Path(args.root) / DIRS[(model, task)]
        if not (d / "swaps.parquet").exists():
            continue
        S = pd.read_parquet(d / "swaps.parquet")
        S = S[S.cross].copy()
        S["t_verb"] = np.where(S.base_cls == 1, S.base_verb, S.src_verb)
        S["i_verb"] = np.where(S.base_cls == 0, S.base_verb, S.src_verb)
        S["common"] = S.t_verb.isin(common) & S.i_verb.isin(common)
        fc = pd.read_csv(d / "fold_checks.csv")
        checks.append({"model": model, "task": task, "max_abs_diff_vs_sweep": float(fc["diff"].max())})
        swaps[(model, task)] = S
        B = Boot(S.t_verb, S.i_verb, args.n_boot, rng)
        for site, g in S.groupby("site"):
            cells = {"all": g, "T1-I1": g[g.pattern == "T1-I1"], "T1-I2+": g[g.pattern == "T1-I2+"],
                     "common_single": g[g.common]}
            dr = {k: B.mean(v) for k, v in cells.items()}
            r = {"model": model, "task": task, "site": int(site), "depth": site / L, "epoch": int(g.epoch.iloc[0])}
            for k, v in cells.items():
                r[f"{k}_est"], r[f"{k}_lo"], r[f"{k}_hi"] = ci(dr[k])
                r[f"{k}_n"] = len(v)
                r[f"{k}_nT"], r[f"{k}_nI"] = v.t_verb.nunique(), v.i_verb.nunique()
            dd = ci(dr["T1-I1"] - dr["T1-I2+"])
            r["T1I1_minus_T1I2_est"], r["T1I1_minus_T1I2_lo"], r["T1I1_minus_T1I2_hi"] = dd
            rows.append(r)
    R = pd.DataFrame(rows)
    M = matched(swaps, rng, args.n_boot)
    M.to_csv(Path(args.csv).with_name("matched_common.csv"), index=False)
    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(out, index=False)

    lines = ["# DAS held-out IIA by participle token pattern: Gemma-3-1B-pt vs Pythia-1.4B", "",
             "Script `scripts/analyze_das_token_split.py` (definitions in its docstring); runs "
             "`run_das_token_split.py`. Cross-class held-out swaps, split 0, 5 folds pooled, each site at its "
             "sweep best epoch. T1-I1 = both participles one token under that model's tokenizer; T1-I2+ = "
             "transitive one token, intransitive two or more; common single = both verbs one "
             f"token under both tokenizers ({len(common)} verbs: "
             f"{sum(1 for v in common if v in set(pairs.trans))} T, {sum(1 for v in common if v in set(pairs.intrans))} I "
             "before each task's behaviour filter). 95% CIs: two-way bootstrap over T and I verbs.", "",
             "Retrained vs sweep (max |Δ fold IIA|): " + "; ".join(
                 f"{c['model']} {c['task']} {c['max_abs_diff_vs_sweep']:.1e}" for c in checks), ""]
    for task in ("active", "passive"):
        T = R[R.task == task]
        if T.empty:
            continue
        lines += [f"## {task} task", "",
                  "| model | site | depth | all | T1-I1 | T1-I2+ | T1-I1 − T1-I2+ | common single |",
                  "|---|---:|---:|---|---|---|---|---|"]
        for _, r in T.sort_values(["model", "site"], ascending=[False, True]).iterrows():
            g = lambda k: fmt((r[f"{k}_est"], r[f"{k}_lo"], r[f"{k}_hi"]), int(r[f"{k}_n"]))
            lines.append(f"| {r['model']} | {int(r['site'])} | {r['depth']:.2f} | {g('all')} | {g('T1-I1')} | "
                         f"{g('T1-I2+')} | "
                         f"{fmt((r['T1I1_minus_T1I2_est'], r['T1I1_minus_T1I2_lo'], r['T1I1_minus_T1I2_hi']))} | "
                         f"{g('common_single')} |")
        lines.append("")
    lines += ["## Matched comparison: common single-token verb pairs held out in both models", "",
              "Same (base, source) verb pairs for both models, IIA averaged per pair; Gemma site = nearest relative "
              "depth to the Pythia site; Gemma − Pythia with the same bootstrap draws.", "",
              "| task | Pythia site | Gemma site | depth (P / G) | n pairs (T, I verbs) | Pythia | Gemma | Gemma − Pythia |",
              "|---|---:|---:|---|---|---|---|---|"]
    for _, r in M.iterrows():
        f3 = lambda k: fmt((r[f"{k}_est"], r[f"{k}_lo"], r[f"{k}_hi"]))
        lines.append(f"| {r['task']} | {int(r['pythia_site'])} | {int(r['gemma_site'])} | {r['pythia_depth']:.2f} / "
                     f"{r['gemma_depth']:.2f} | {int(r['n_pairs'])} ({int(r['nT'])}, {int(r['nI'])}) | {f3('pythia')} | "
                     f"{f3('gemma')} | {f3('diff')} |")
    lines.append("")
    Path(args.report).write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:8]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results/gemma/das/token_split")
    ap.add_argument("--pairs", default="data/das_round2/train_pairs.csv")
    ap.add_argument("--csv", default="results/gemma/das/token_split/summary.csv")
    ap.add_argument("--report", default="reports/gemma/das_token_split.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    main(ap.parse_args())
