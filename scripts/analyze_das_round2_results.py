#!/usr/bin/env python3
"""Summarise the DAS round-2 final runs (held-out actives only; no passives).

Unit of analysis: the held-out verb pair. Each pair is held out once per
split seed; swap-level metrics are averaged within pair (cross-class swaps,
by base class), then across the 3 splits, and CIs come from a bootstrap over
pairs. Controls (random subspaces, shuffled labels) are compared per fold.
Directions: cosine of sign-aligned rank-1 bases within and between runs
(principal-angle cosines for rank > 1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def boot(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    d = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), len(x)


def f(t, k=2):
    return f"{t[0]:.{k}f} [{t[1]:.{k}f}, {t[2]:.{k}f}]"


def per_pair(det, rank):
    d = det[(det["rank"] == rank) & det.cross].copy()
    d["dM"] = d.M_patched - d.M_base
    for k in ("O", "det", "pron", "refl", "I"):
        d[f"d_{k}"] = d[f"lp_{k}_patched"] - d[f"lp_{k}_base"]
    d["frac_O"] = d.d_O / (d.lp_O_src - d.lp_O_base)
    cols = ["iia", "frac", "dM", "d_O", "d_det", "d_pron", "d_refl", "d_I", "frac_O"]
    g = d.groupby(["base_cls", "base_pair_id", "base_source", "base_tok", "split"])[cols].agg(
        lambda s: s.median() if s.name in ("frac", "frac_O") else s.mean()).reset_index()
    return g.groupby(["base_cls", "base_pair_id", "base_source", "base_tok"])[cols].mean().reset_index(), d


def directions(path):
    import torch

    b = torch.load(path)
    return {k: v.numpy() for k, v in b.items()}


def cos_stats(bases, rank):
    keys = [k for k in bases if k.startswith(f"r{rank}_")]
    mats = [bases[k] for k in keys]
    if rank == 1:
        v = np.stack([m[:, 0] / np.linalg.norm(m[:, 0]) for m in mats])
        ref = v[0]
        v = v * np.sign(v @ ref)[:, None]
        c = v @ v.T
        mean = v.mean(0)
        return v, mean / np.linalg.norm(mean), c[np.triu_indices(len(v), 1)]
    sims = []
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            sims.append(np.linalg.svd(mats[i].T @ mats[j], compute_uv=False).mean())
    return None, None, np.array(sims)


def run(args):
    out = []
    runs = {}
    for name, d in (("primary", args.primary), ("sensitivity", args.secondary)):
        if not d or not Path(d).exists():
            continue
        meta = json.loads((Path(d) / "final_meta.json").read_text())
        summ = pd.read_csv(Path(d) / "summary.csv")
        det = pd.read_csv(Path(d) / "heldout_swaps.csv.gz")
        runs[name] = {"meta": meta, "summ": summ, "det": det, "bases": directions(Path(d) / "bases.pt")}
    p = runs["primary"]
    rank = p["meta"]["chosen_rank"]
    L = ["# DAS round 2: held-out active results", "",
         f"Site {p['meta']['site']} (output of layer {p['meta']['site'] - 1}), {p['meta']['epochs']} epochs, "
         f"rank {rank} (rule: {p['meta']['rank_rule']}; mean held-out median positive-control fraction by rank "
         f"{ {int(k): round(v, 3) for k, v in p['meta']['pc_frac_by_rank'].items()} }). "
         f"Primary set: {p['meta']['pairs']} pairs ({p['meta']['dropped_pairs']} dropped by the behaviour filter). "
         "5-fold CV over pairs x 3 split seeds; held-out pairs only; CIs = bootstrap over pairs. No passive has "
         "been evaluated.", ""]
    for name, r in runs.items():
        pp, d = per_pair(r["det"], rank)
        intr, tr = pp[pp.base_cls == 0], pp[pp.base_cls == 1]
        L += [f"## {name.capitalize()} run ({r['meta']['pairs']} pairs)", "",
              "| Metric (held-out pairs) | Intransitive base <- transitive source | Transitive base <- intransitive source |",
              "|---|---|---|",
              f"| IIA (patched M on the source's side) | {f(boot(intr.iia))} | {f(boot(tr.iia))} |",
              f"| Fraction of the natural gap in M (median per pair) | {f(boot(intr.frac))} | {f(boot(tr.frac))} |",
              f"| Fraction of the natural gap in log P(O) | {f(boot(intr.frac_O))} | {f(boot(tr.frac_O))} |",
              f"| Delta log P(O) | {f(boot(intr.d_O))} | {f(boot(tr.d_O))} |",
              f"| Delta log P(determiners) | {f(boot(intr.d_det))} | {f(boot(tr.d_det))} |",
              f"| Delta log P(pronouns) | {f(boot(intr.d_pron))} | {f(boot(tr.d_pron))} |",
              f"| Delta log P(reflexives) | {f(boot(intr.d_refl))} | {f(boot(tr.d_refl))} |",
              f"| Delta log P(I) | {f(boot(intr.d_I))} | {f(boot(tr.d_I))} |", ""]
        s = r["summ"]
        das = s[(s.control == "das") & (s["rank"] == rank)]
        L += ["Per fold (mean over 15 fold x split runs):", "",
              "| Condition | IIA cross | IIA same-class | Positive-control fraction (median) | IIA cross, held-out subject David |",
              "|---|---|---|---|---|",
              f"| DAS | {das.iia_cross.mean():.3f} | {das.iia_same.mean():.3f} | {das.pc_frac_median.mean():.3f} | "
              f"{das.iia_cross_david.mean():.3f} |"]
        for c in ("random", "random_normmatched", "shuffled_labels"):
            x = s[s.control == c]
            if len(x):
                L.append(f"| {c} | {x.iia_cross.mean():.3f} | {x.iia_same.mean():.3f} | "
                         f"{x.pc_frac_median.mean():.3f} | {x.iia_cross_david.mean():.3f} |")
        rnd = s[s.control == "random_normmatched"]
        if len(rnd):
            q = rnd.groupby(["split", "fold"]).pc_frac_median.quantile(0.95)
            dd = das.set_index(["split", "fold"]).pc_frac_median
            L += ["", f"DAS beats the 95th percentile of norm-matched random subspaces on the positive-control "
                  f"fraction in {(dd > q.reindex(dd.index)).sum()} of {len(dd)} fold x split runs."]
        L += ["", "By source and token count (intransitive base <- transitive source; exploratory):", "",
              "| Source | Tokens | Pairs | IIA | Gap fraction |", "|---|---|---|---|---|"]
        for (src, tok), g in intr.groupby(["base_source", "base_tok"]):
            L.append(f"| {src} | {tok} | {len(g)} | {f(boot(g.iia))} | {f(boot(g.frac))} |")
        L.append("")
    # directions
    L += ["## Directions", ""]
    vecs = {}
    for name, r in runs.items():
        v, mean, c = cos_stats(r["bases"], rank)
        vecs[name] = mean
        L.append(f"- {name}: pairwise cosine across its {len([k for k in r['bases'] if k.startswith(f'r{rank}_')])} "
                 f"fold x split bases: median {np.median(c):.3f} (min {c.min():.3f})"
                 + (" (rank > 1: mean principal-angle cosine)" if rank > 1 else ""))
    if rank == 1 and len(vecs) == 2:
        cs = float(abs(vecs["primary"] @ vecs["sensitivity"]))
        vp, _, _ = cos_stats(runs["primary"]["bases"], 1)
        vs, _, _ = cos_stats(runs["sensitivity"]["bases"], 1)
        cross = np.abs(vp @ vs.T)
        L.append(f"- primary vs sensitivity: cosine of mean directions {cs:.3f}; all primary x sensitivity "
                 f"base pairs median {np.median(cross):.3f} (min {cross.min():.3f}).")
    if len(runs) == 2:
        a, b = (per_pair(runs[n]["det"], rank)[0] for n in ("primary", "sensitivity"))
        m = a.merge(b, on=["base_cls", "base_pair_id"], suffixes=("_p", "_s"))
        m = m[m.base_cls == 0]
        L.append(f"- held-out transfer agreement on the {len(m)} intransitive-base pairs present in both runs: "
                 f"gap fraction primary {m.frac_p.mean():.2f} vs sensitivity {m.frac_s.mean():.2f}, Pearson r "
                 f"{np.corrcoef(m.frac_p, m.frac_s)[0, 1]:.2f}.")
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--primary", default="results/das_round2/final_strict")
    ap.add_argument("--secondary", default="results/das_round2/final_named")
    ap.add_argument("--report", default="reports/passive_das_prep/das_round2_results.md")
    run(ap.parse_args())
