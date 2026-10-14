#!/usr/bin/env python3
"""Round 5, D10c analysis: FreqBLiMP passive_1 Yes/No readout on Pythia 1.4B / 6.9B / 12B (reports/round5/plan.md,
D10c). No model.

Input: results/round5/yesno/yesno_pythia-{1.4b,6.9b,12b}.parquet (run_yesno.py).
Per model and band: accuracy (good scores higher; exact ties half credit) with an item bootstrap (2,000 draws, seed
17) and, as a sensitivity, a good-verb-cluster bootstrap (clusters = the good sentence's participle); the declared
readings (above chance: 95% CI above 0.5; practically at chance: 90% CI within 0.45-0.55; below chance: 95% CI below
0.5; otherwise unresolved); the mean margin yn_good - yn_bad; the median Yes + No probability mass. Model comparisons:
paired draws (the same items in each draw).
Output: reports/round5/d10c_yesno.md, results/round5/yesno/yesno_summary.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ("1.4b", "6.9b", "12b")
BANDS = ("head", "tail", "xtail")


def credit(d):
    return np.where(d.yn_good > d.yn_bad, 1.0, np.where(d.yn_good == d.yn_bad, 0.5, 0.0))


def reading(lo95, hi95, lo90, hi90):
    if lo95 > 0.5:
        return "above chance"
    if hi95 < 0.5:
        return "below chance"
    if lo90 >= 0.45 and hi90 <= 0.55:
        return "practically at chance"
    return "unresolved"


def participle(good, bad):
    g, b = good.rstrip(".").split(), bad.rstrip(".").split()
    diff = [x for x, y in zip(g, b) if x != y]
    assert len(g) == len(b) and len(diff) == 1, (good, bad)
    return diff[0].lower()


def main(args):
    D = {m: pd.read_parquet(Path(args.res_dir) / f"yesno_pythia-{m}.parquet") for m in MODELS}
    base = D[MODELS[0]][["band", "pairID", "sentence_good", "sentence_bad"]]
    for m in MODELS[1:]:
        assert D[m][["band", "pairID", "sentence_good", "sentence_bad"]].equals(base), m
    C = pd.DataFrame({m: credit(D[m]) for m in MODELS})
    Mg = pd.DataFrame({m: D[m].yn_good - D[m].yn_bad for m in MODELS})
    band = base.band.to_numpy()
    verb = np.array([participle(g, b) for g, b in zip(base.sentence_good, base.sentence_bad)])
    bverb = np.array([participle(b, g) for g, b in zip(base.sentence_good, base.sentence_bad)])
    rng = np.random.default_rng(args.seed)
    rows, draws = [], {}
    for b in BANDS:
        idx = np.flatnonzero(band == b)
        item = rng.integers(0, len(idx), (args.n_boot, len(idx)))
        vs = np.unique(verb[idx])
        members = {v: idx[verb[idx] == v] for v in vs}
        cl = []
        for _ in range(args.n_boot):
            pick = rng.choice(vs, len(vs))
            cl.append(np.concatenate([members[v] for v in pick]))
        bvs = np.unique(bverb[idx])
        bmembers = {v: idx[bverb[idx] == v] for v in bvs}
        bcl = [np.concatenate([bmembers[v] for v in rng.choice(bvs, len(bvs))]) for _ in range(args.n_boot)]
        for m in MODELS:
            c = C[m].to_numpy()
            bi = c[idx][item].mean(1)
            bb = np.array([c[x].mean() for x in bcl])
            bc = np.array([c[x].mean() for x in cl])
            draws[(m, b)] = bi
            lo95, hi95 = np.percentile(bi, [2.5, 97.5])
            lo90, hi90 = np.percentile(bi, [5, 95])
            clo95, chi95 = np.percentile(bc, [2.5, 97.5])
            clo90, chi90 = np.percentile(bc, [5, 95])
            mg = Mg[m].to_numpy()[idx]
            rows.append({"model": m, "band": b, "n": len(idx), "n_good_verbs": len(vs), "acc": c[idx].mean(),
                         "lo95": lo95, "hi95": hi95, "lo90": lo90, "hi90": hi90,
                         "reading": reading(lo95, hi95, lo90, hi90),
                         "cl_lo95": clo95, "cl_hi95": chi95, "cl_reading": reading(clo95, chi95, clo90, chi90),
                         "margin": mg.mean(), "margin_lo": np.percentile(mg[item].mean(1), 2.5),
                         "margin_hi": np.percentile(mg[item].mean(1), 97.5),
                         "mass_good_median": float(np.median(D[m].mass_good.to_numpy()[idx])),
                         "ties": int((c[idx] == 0.5).sum()), "n_bad_verbs": len(bvs),
                         "bcl_lo95": np.percentile(bb, 2.5), "bcl_hi95": np.percentile(bb, 97.5),
                         "yes_good": float((D[m].yn_good.to_numpy()[idx] > 0).mean()),
                         "yes_bad": float((D[m].yn_bad.to_numpy()[idx] > 0).mean())})
    S = pd.DataFrame(rows)
    S.to_csv(Path(args.res_dir) / "yesno_summary.csv", index=False)
    L = ["# Round 5, D10c: FreqBLiMP passive_1 Yes/No readout", "",
         "Spec: `plan.md`, D10c. Run: `run_yesno.py` (TSUBAME job 9009081, fp32). Analysis: `analyze_yesno.py`. Prompt: "
         "task 4's `YES_NO_BASE_PROMPT`; score = LSE(Yes, \" Yes\") − LSE(No, \" No\") at the answer position; a pair is "
         "correct if the good sentence scores higher (ties half). Items: released `passive_1`, 1,000 per band. CIs: item "
         "bootstrap (2,000 draws); sensitivity: good-verb clusters.", "",
         "| model | band | accuracy [95% CI] | 90% CI | reading | good-verb clusters: 95% CI, reading | margin [95% CI] | median Yes+No mass |",
         "|---|---|---|---|---|---|---|---:|"]
    for r in S.itertuples():
        L.append(f"| {r.model} | {r.band} | {r.acc:.3f} [{r.lo95:.3f}, {r.hi95:.3f}] | [{r.lo90:.3f}, {r.hi90:.3f}] | "
                 f"**{r.reading}** | [{r.cl_lo95:.3f}, {r.cl_hi95:.3f}], {r.cl_reading} | {r.margin:+.3f} "
                 f"[{r.margin_lo:+.3f}, {r.margin_hi:+.3f}] | {r.mass_good_median:.3f} |")
    L += ["", "**Model comparisons** (accuracy difference, paired item draws, 95% CI):", "",
          "| band | 6.9B − 1.4B | 12B − 1.4B | 12B − 6.9B |", "|---|---|---|---|"]
    for b in BANDS:
        cells = []
        for a, c in (("6.9b", "1.4b"), ("12b", "1.4b"), ("12b", "6.9b")):
            dd = draws[(a, b)] - draws[(c, b)]
            est = S[(S.model == a) & (S.band == b)].acc.item() - S[(S.model == c) & (S.band == b)].acc.item()
            cells.append(f"{est:+.3f} [{np.percentile(dd, 2.5):+.3f}, {np.percentile(dd, 97.5):+.3f}]")
        L.append(f"| {b} | " + " | ".join(cells) + " |")
    L += ["", "**Descriptive (not declared).** Share of sentences scored Yes > No, and the accuracy CI with the *bad* "
          "participle's clusters resampled (Head has only 20 bad verbs):", "",
          "| model | band | Yes > No: good / bad | bad verbs | bad-verb clusters, 95% CI |", "|---|---|---|---:|---|"]
    for r in S.itertuples():
        L.append(f"| {r.model} | {r.band} | {r.yes_good:.2f} / {r.yes_bad:.2f} | {r.n_bad_verbs} | "
                 f"[{r.bcl_lo95:.3f}, {r.bcl_hi95:.3f}] |")
    L.append("")
    Path(args.report).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res-dir", default="results/round5/yesno")
    ap.add_argument("--report", default="reports/round5/d10c_yesno.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    main(ap.parse_args())
