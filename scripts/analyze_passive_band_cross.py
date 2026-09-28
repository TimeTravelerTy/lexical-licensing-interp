#!/usr/bin/env python3
"""Separate verb-band and context-band effects in the crossed passive scores.

Verb pairs and contexts are both sampled units, so every interval comes from a
two-way cluster bootstrap: each draw resamples verb pairs within verb band and
contexts within context band independently. Cell means are computed from the
full verb x context matrix, so each verb pair and each context gets equal weight.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BANDS = ("head", "tail", "xtail")
PARADIGMS = ("passive_1", "passive_2")
METRICS = ("correct", "whole_margin", "verb_margin", "by_margin", "suffix_margin")
FRAME_COVARIATES = ("negated", "plural_aux", "present_aux", "patient_is_name")


def ci(draws: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def matrix(df: pd.DataFrame, metric: str, verbs: pd.Index, contexts: pd.Index) -> np.ndarray:
    m = df.pivot(index="verb_pair", columns="context_id", values=metric).reindex(index=verbs, columns=contexts)
    return m.to_numpy(dtype=float)


def band_weights(rng, bands: np.ndarray, n_boot: int) -> np.ndarray:
    """Multinomial resampling counts within each band; row 0 is the point estimate."""
    w = np.ones((n_boot + 1, len(bands)))
    for band in BANDS:
        idx = np.flatnonzero(bands == band)
        w[1:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=n_boot)
    return w


def weighted_cell(m, mask, wv, wc):
    """Weighted mean of m over rows/cols for every draw; mask excludes cells."""
    m = np.where(mask, m, 0.0)
    num = np.einsum("bv,vc,bc->b", wv, m, wc)
    den = np.einsum("bv,vc,bc->b", wv, mask.astype(float), wc)
    return num / den


def released_analysis(df, n_boot, rng):
    cells, contrasts = [], []
    for paradigm in PARADIGMS:
        sub = df[df.paradigm == paradigm]
        verbs = sub.drop_duplicates("verb_pair").sort_values(["verb_band", "verb_pair"])
        contexts = sub.drop_duplicates("context_id").sort_values(["context_band", "context_id"])
        vband, cband = verbs.verb_band.to_numpy(), contexts.context_band.to_numpy()
        wv_all, wc_all = band_weights(rng, vband, n_boot), band_weights(rng, cband, n_boot)
        metrics = [m for m in METRICS if not (m == "by_margin" and paradigm == "passive_2")]
        for metric in metrics:
            m = matrix(sub, metric, pd.Index(verbs.verb_pair), pd.Index(contexts.context_id))
            if np.isnan(m).any():
                raise ValueError(f"Incomplete verb x context matrix for {paradigm}/{metric}")
            draws = {}
            for vb in BANDS:
                for cb in BANDS:
                    rows, cols = vband == vb, cband == cb
                    d = weighted_cell(m[np.ix_(rows, cols)], np.ones((rows.sum(), cols.sum()), bool),
                                      wv_all[:, rows], wc_all[:, cols])
                    draws[vb, cb] = d
                    lo, hi = ci(d[1:])
                    cells.append({"paradigm": paradigm, "metric": metric, "verb_band": vb,
                                  "context_band": cb, "estimate": d[0], "ci_low": lo, "ci_high": hi,
                                  "n_verb_pairs": int(rows.sum()), "n_contexts": int(cols.sum())})

            def mean(keys):
                return np.mean([draws[k] for k in keys], axis=0)

            named = {
                # Head-to-XTail on the diagonal: what the released benchmark measures.
                "diagonal_xtail_minus_head": draws["xtail", "xtail"] - draws["head", "head"],
                "verb_xtail_minus_head": mean([("xtail", c) for c in BANDS]) - mean([("head", c) for c in BANDS]),
                "context_xtail_minus_head": mean([(v, "xtail") for v in BANDS]) - mean([(v, "head") for v in BANDS]),
                "interaction_xtail_head": (draws["xtail", "xtail"] - draws["xtail", "head"])
                                          - (draws["head", "xtail"] - draws["head", "head"]),
            }
            for cb in BANDS:
                named[f"verb_xtail_minus_head_in_{cb}_contexts"] = draws["xtail", cb] - draws["head", cb]
            for vb in BANDS:
                named[f"context_xtail_minus_head_for_{vb}_verbs"] = draws[vb, "xtail"] - draws[vb, "head"]
            for name, d in named.items():
                lo, hi = ci(d[1:])
                contrasts.append({"paradigm": paradigm, "metric": metric, "contrast": name,
                                  "estimate": d[0], "ci_low": lo, "ci_high": hi,
                                  "excludes_zero": int(lo > 0 or hi < 0)})
    return pd.DataFrame(cells), pd.DataFrame(contrasts)


def curated_analysis(curated, released, n_boot, rng):
    """Own curated context vs other verbs' curated contexts vs released contexts."""
    rows = []
    for paradigm in PARADIGMS:
        cur = curated[curated.paradigm == paradigm]
        rel = released[released.paradigm == paradigm]
        verbs = cur.drop_duplicates("verb_pair").sort_values(["verb_band", "verb_pair"])
        vidx = pd.Index(verbs.verb_pair)
        vband = verbs.verb_band.to_numpy()
        ccon = cur.drop_duplicates("context_id").sort_values(["context_band", "context_id"])
        rcon = rel.drop_duplicates("context_id").sort_values(["context_band", "context_id"])
        wv = band_weights(rng, vband, n_boot)
        wcc = band_weights(rng, ccon.context_band.to_numpy(), n_boot)
        wrc = band_weights(rng, rcon.context_band.to_numpy(), n_boot)
        own = cur[cur.own_context == 1].set_index("verb_pair")
        own_mask = cur.pivot(index="verb_pair", columns="context_id", values="own_context") \
            .reindex(index=vidx, columns=pd.Index(ccon.context_id)).to_numpy() == 1
        if own_mask.sum(axis=1).tolist() != [1] * len(vidx):
            raise ValueError("Each verb pair needs exactly one own curated context")
        for metric in METRICS:
            if metric == "by_margin" and paradigm == "passive_2":
                continue
            mc = matrix(cur, metric, vidx, pd.Index(ccon.context_id))
            mr = matrix(rel, metric, vidx, pd.Index(rcon.context_id))
            own_values = own[metric].reindex(vidx).to_numpy(float)
            for vb in BANDS:
                r = vband == vb
                d_own = (wv[:, r] @ own_values[r]) / wv[:, r].sum(axis=1)
                d_other = weighted_cell(mc[r], ~own_mask[r], wv[:, r], wcc)
                d_rel = weighted_cell(mr[r], np.ones(mr[r].shape, bool), wv[:, r], wrc)
                for condition, d in (("own_curated", d_own), ("other_curated", d_other),
                                     ("released_all_bands", d_rel),
                                     ("own_minus_other", d_own - d_other),
                                     ("other_curated_minus_released", d_other - d_rel)):
                    lo, hi = ci(d[1:])
                    rows.append({"paradigm": paradigm, "metric": metric, "verb_band": vb,
                                 "condition": condition, "estimate": d[0], "ci_low": lo, "ci_high": hi,
                                 "n_verb_pairs": int(r.sum())})
    return pd.DataFrame(rows)


def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else np.nan
    return beta, r2


def context_regression(df, n_boot, rng):
    """Context-level OLS: does the context-band effect survive frame covariates?

    The outcome is each context's margin averaged over the verb pairs of one
    band. Intervals bootstrap contexts within context band (verbs fixed).
    """
    rows = []
    for paradigm in PARADIGMS:
        sub = df[df.paradigm == paradigm]
        covars = list(FRAME_COVARIATES) + (["agent_is_name"] if paradigm == "passive_1" else [])
        zipfs = ["patient_zipf"] + (["agent_zipf"] if paradigm == "passive_1" else [])
        for verb_band in ("all",) + BANDS:
            part = sub if verb_band == "all" else sub[sub.verb_band == verb_band]
            # Equal weight per verb pair: average within verb pair, then over them.
            ctx = part.groupby("context_id").agg(
                y=("whole_margin", "mean"), acc=("correct", "mean"), context_band=("context_band", "first"),
                **{c: (c, "first") for c in covars + zipfs}).reset_index()
            ctx[covars + zipfs] = ctx[covars + zipfs].astype(float)
            dummies = pd.get_dummies(ctx.context_band)[["tail", "xtail"]].astype(float)
            designs = {
                "band_only": dummies,
                "band_plus_frame": pd.concat([dummies, ctx[covars]], axis=1),
                "noun_zipf_plus_frame": pd.concat([ctx[zipfs], ctx[covars]], axis=1),
                "band_plus_zipf_plus_frame": pd.concat([dummies, ctx[zipfs], ctx[covars]], axis=1),
            }
            bands = ctx.context_band.to_numpy()
            weights = band_weights(rng, bands, n_boot)
            for model, X in designs.items():
                names = ["intercept"] + list(X.columns)
                Xa = np.column_stack([np.ones(len(X)), X.to_numpy()])
                y = ctx.y.to_numpy()
                beta0, r2 = ols(y, Xa)
                draws = []
                for w in weights[1:]:
                    idx = np.repeat(np.arange(len(y)), w.astype(int))
                    b, _ = ols(y[idx], Xa[idx])
                    draws.append(b)
                draws = np.array(draws)
                for j, name in enumerate(names):
                    lo, hi = ci(draws[:, j])
                    rows.append({"paradigm": paradigm, "verb_band": verb_band, "model": model,
                                 "term": name, "estimate": beta0[j], "ci_low": lo, "ci_high": hi,
                                 "r2": r2, "n_contexts": len(y)})
    return pd.DataFrame(rows)


def fmt(row, pct=False):
    s = 100 if pct else 1
    return f"{row.estimate * s:.1f} [{row.ci_low * s:.1f}, {row.ci_high * s:.1f}]" if pct else \
        f"{row.estimate:.2f} [{row.ci_low:.2f}, {row.ci_high:.2f}]"


def report(cells, contrasts, curated, regression, scores, out_dir):
    lines = ["# Verb band x context band passive cross", "",
             f"{len(scores):,} scored pairs. Released contexts: 100 sampled per band and paradigm, "
             "each crossed with all 26/50/50 verb pairs. Intervals: two-way cluster bootstrap "
             "(verb pairs within verb band, contexts within context band).", ""]
    for paradigm in PARADIGMS:
        lines += [f"## `{paradigm}`", ""]
        for metric, pct in (("correct", True), ("whole_margin", False), ("verb_margin", False)):
            c = cells[(cells.paradigm == paradigm) & (cells.metric == metric)]
            label = "Accuracy (%)" if pct else f"Mean `{metric}`"
            lines += [f"**{label}**: rows = verb band, columns = context band", "",
                      "| Verb \\ Context | " + " | ".join(BANDS) + " |", "|---|" + "---:|" * len(BANDS)]
            for vb in BANDS:
                vals = [fmt(c[(c.verb_band == vb) & (c.context_band == cb)].iloc[0], pct) for cb in BANDS]
                lines.append(f"| {vb} | " + " | ".join(vals) + " |")
            lines.append("")
        k = contrasts[(contrasts.paradigm == paradigm) & contrasts.contrast.isin(
            ["diagonal_xtail_minus_head", "verb_xtail_minus_head", "context_xtail_minus_head",
             "interaction_xtail_head"])]
        lines += ["**Contrasts (XTail - Head)**", "", "| Contrast | Accuracy (pp) | Whole margin | Verb margin |",
                  "|---|---:|---:|---:|"]
        for name in ("diagonal_xtail_minus_head", "verb_xtail_minus_head", "context_xtail_minus_head",
                     "interaction_xtail_head"):
            vals = [fmt(k[(k.contrast == name) & (k.metric == m)].iloc[0], m == "correct")
                    for m in ("correct", "whole_margin", "verb_margin")]
            lines.append(f"| {name} | " + " | ".join(vals) + " |")
        lines.append("")
        cu = curated[(curated.paradigm == paradigm) & (curated.metric == "correct")]
        lines += ["**Curated contexts: own vs other verbs' vs released (accuracy %)**", "",
                  "| Verb band | Own curated | Other curated | Released (all bands) | Own - other | Other - released |",
                  "|---|---:|---:|---:|---:|---:|"]
        for vb in BANDS:
            vals = [fmt(cu[(cu.verb_band == vb) & (cu.condition == cond)].iloc[0], True)
                    for cond in ("own_curated", "other_curated", "released_all_bands",
                                 "own_minus_other", "other_curated_minus_released")]
            lines.append(f"| {vb} | " + " | ".join(vals) + " |")
        lines.append("")
        rg = regression[(regression.paradigm == paradigm) & (regression.verb_band == "all")
                        & (regression.term == "xtail")]
        lines += ["**Context-level regression, XTail-context coefficient on whole margin (all verbs)**", "",
                  "| Model | XTail vs Head context | R^2 |", "|---|---:|---:|"]
        for _, r in rg.iterrows():
            lines.append(f"| {r.model} | {fmt(r)} | {r.r2:.3f} |")
        lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    scores = pd.read_csv(args.scores, low_memory=False)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    released = scores[scores.context_set == "released"]
    curated = scores[scores.context_set == "curated"]
    cells, contrasts = released_analysis(released, args.n_boot, rng)
    cur = curated_analysis(curated, released, args.n_boot, rng)
    regression = context_regression(released, args.n_boot_regression, rng)
    per_verb = released.groupby(["paradigm", "verb_band", "verb_pair", "context_band"]).agg(
        accuracy=("correct", "mean"), whole_margin=("whole_margin", "mean"),
        verb_margin=("verb_margin", "mean"), n_contexts=("context_id", "nunique")).reset_index()
    cells.to_csv(out_dir / "cells.csv", index=False)
    contrasts.to_csv(out_dir / "contrasts.csv", index=False)
    cur.to_csv(out_dir / "curated_own_other.csv", index=False)
    regression.to_csv(out_dir / "context_regression.csv", index=False)
    per_verb.to_csv(out_dir / "per_verb_pair_by_context_band.csv", index=False)
    report(cells, contrasts, cur, regression, scores, out_dir)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default="results/passive_band_cross/pythia14b_scores.csv")
    ap.add_argument("--out-dir", default="reports/passive_band_cross")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-boot-regression", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
