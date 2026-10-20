#!/usr/bin/env python3
"""Round 6, part 0: FreqBLiMP human validation, passive_1 / passive_2 by band (reports/round6/plan.md, part 0).

The human study (freq-blimp/human_validation_ui; three annotators, 15 pairs per paradigm x condition, every
annotator judged every pair) is forced choice: A, B or "can't decide" (with reasons), then a 1-5 naturalness rating of
the intended acceptable sentence only. There are no per-sentence ratings, so bad-item acceptance cannot be computed;
accuracy by band is reported, descriptively.

Input: the merged export freqblimp-all-annotators.jsonl (the corrected 2026-08-25 merge, outside the repo; its sha256
is recorded). Sentences come from the released FreqBLiMP files (Head / Tail / XTail) and the original BLiMP file;
Pythia-1.4B LP correctness on the same items comes from the unedited condition of round-5 A4
(results/round5/restore/restore_freqblimp*.parquet; Head / Tail / XTail only).

Per cell: correct among decided responses, can't-decide share, accuracy with can't-decide counted as half, the number
of items with 2-of-3 support for the intended sentence, mean naturalness. 95% CIs: items resampled within cell (an
item's three judgments stay together), 2,000 draws, seed 17. Descriptive only: 15 items per cell.

Outputs: data/round6/human/passive_items.csv (per item, no annotator-level data), reports/round6/human_passives.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PARADIGMS = ("passive_1", "passive_2")
CONDS = ("original", "head", "tail", "xtail")
NAME = {"original": "Original", "head": "Head", "tail": "Tail", "xtail": "XTail"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sentences(args):
    out = {}
    for c in CONDS:
        for p in PARADIGMS:
            f = Path(args.original) if c == "original" else Path(args.freqblimp) / c / f"{p}.jsonl"
            d = pd.read_json(f, lines=True, dtype={"pairID": str})
            if c == "original":
                d = d[d.UID == p]
            for r in d.itertuples(index=False):
                out[(c, p, str(r.pairID))] = (r.sentence_good, r.sentence_bad)
    return out


def main(args):
    D = pd.DataFrame([json.loads(l) for l in open(args.responses)])
    assert D.groupby(["annotator_id", "trial_id"]).size().max() == 1, "duplicate responses"
    D = D[D.paradigm.isin(PARADIGMS) & ~D.is_catch].copy()
    D["correct"] = D.response.ne("cant_decide") & D.selected_acceptable.eq(True)
    D["wrong"] = D.response.ne("cant_decide") & D.selected_acceptable.eq(False)
    D["cant"] = D.response.eq("cant_decide")
    assert (D.correct.astype(int) + D.wrong.astype(int) + D.cant.astype(int) == 1).all()
    reasons = D.explode("cant_decide_reasons")
    reasons = reasons[reasons.cant].groupby(["item_uid", "cant_decide_reasons"]).size().unstack(fill_value=0)
    it = D.groupby(["item_uid", "paradigm", "condition", "pair_id"]).agg(
        n=("response", "size"), n_correct=("correct", "sum"), n_wrong=("wrong", "sum"), n_cant=("cant", "sum"),
        naturalness=("naturalness", "mean")).reset_index()
    assert (it.n == 3).all() and len(it) == 15 * len(PARADIGMS) * len(CONDS)
    it = it.join(reasons.add_prefix("cant_"), on="item_uid").fillna({c: 0 for c in reasons.add_prefix("cant_").columns})
    it["supported"] = it.n_correct >= 2  # 2-of-3 for the intended acceptable sentence
    S = sentences(args)
    it["sentence_good"] = [S[(c, p, str(i))][0] for c, p, i in zip(it.condition, it.paradigm, it.pair_id)]
    it["sentence_bad"] = [S[(c, p, str(i))][1] for c, p, i in zip(it.condition, it.paradigm, it.pair_id)]
    # Pythia-1.4B LP on the same released items (round-5 A4, unedited)
    items = pd.read_parquet(args.lp_items)
    lp = pd.read_parquet(args.lp_scores)
    lp = lp[lp.cond == "unedited"].set_index("row")
    items = items.assign(pairID=items.pairID.astype(str)).set_index(["band", "uid", "pairID"]).row
    it["pythia14b_lp_correct"] = [float(lp.loc[items[(c, p, str(i))], "good_lp"] > lp.loc[items[(c, p, str(i))], "bad_lp"])
                                  if c != "original" else np.nan for c, p, i in zip(it.condition, it.paradigm, it.pair_id)]
    out = Path(args.out_items)
    out.parent.mkdir(parents=True, exist_ok=True)
    it.sort_values(["paradigm", "condition", "pair_id"]).to_csv(out, index=False)

    rng = np.random.default_rng(args.seed)
    lines = ["# Round 6, part 0: FreqBLiMP human validation, passive paradigms", "",
             "Spec: `plan.md`, part 0. Script: `analyze_human_passives.py`. Source: the merged three-annotator export "
             f"(corrected 2026-08-25; sha256 `{sha(args.responses)[:16]}…`, outside the repo); per-item table "
             f"`../../{args.out_items}`.", "",
             "**Format.** Forced choice between the two sentences of a pair (A / B / can't decide, with reasons), then a "
             "1–5 naturalness rating of the intended acceptable sentence only. There are no per-sentence acceptability "
             "ratings, so bad-item acceptance cannot be computed; per the round-6 rule, accuracy by band is reported, "
             "descriptively. 15 pairs per paradigm × condition, each judged by all three annotators (45 judgments). "
             "95% CIs: items resampled within cell (2,000 draws, seed 17); with 15 items per cell they are wide.", ""]
    for p in PARADIGMS:
        lines += [f"## `{p}`", "",
                  "| Condition | judgments | correct among decided (%) | can't decide (%) | accuracy, can't decide = ½ (%) | "
                  "items with 2-of-3 support | naturalness (good sentence) | Pythia-1.4B LP, same items (%) |",
                  "|---|---:|---|---|---|---:|---|---|"]
        for c in CONDS:
            x = it[(it.paradigm == p) & (it.condition == c)].reset_index(drop=True)
            w = np.vstack([np.ones(len(x)), rng.multinomial(len(x), np.full(len(x), 1 / len(x)), size=args.n_boot)])
            k, wr, ca, n = (w @ x[col].to_numpy(float) for col in ("n_correct", "n_wrong", "n_cant", "n"))
            stats = {"dec": 100 * k / (k + wr), "cant": 100 * ca / n, "half": 100 * (k + 0.5 * ca) / n,
                     "nat": (w @ x.naturalness.to_numpy()) / w.sum(1)}
            f = lambda d, q=1: f"{d[0]:.{q}f} [{np.nanpercentile(d[1:], 2.5):.{q}f}, {np.nanpercentile(d[1:], 97.5):.{q}f}]"
            lpc = "—" if c == "original" else f((w @ x.pythia14b_lp_correct.to_numpy()) / w.sum(1) * 100)
            lines.append(f"| {NAME[c]} | {int(x.n.sum())} | {f(stats['dec'])} | {f(stats['cant'])} | {f(stats['half'])} | "
                         f"{int(x.supported.sum())} of {len(x)} | {f(stats['nat'], 2)} | {lpc} |")
        lines.append("")
    cant = it.filter(like="cant_").drop(columns=["n_cant"], errors="ignore")
    rs = it[["paradigm", "condition"]].join(cant).groupby(["paradigm", "condition"]).sum().astype(int)
    lines += ["Can't-decide reasons (counts; an annotator could give several):", "",
              "| paradigm | condition | " + " | ".join(c.replace("cant_", "") for c in rs.columns) + " |",
              "|---|---|" + "---:|" * len(rs.columns)]
    for (p, c), r in rs.iterrows():
        lines.append(f"| {p} | {NAME[c]} | " + " | ".join(str(v) for v in r) + " |")
    wrong = it[it.n_wrong >= 2]
    lines += ["", f"Items where 2 of 3 chose the intended unacceptable sentence: {len(wrong)}"
              + ("" if wrong.empty else " (" + "; ".join(f"`{u}`: {g} / *{b}*" for u, g, b in
                                                   zip(wrong.item_uid, wrong.sentence_good, wrong.sentence_bad)) + ")") + "."]
    Path(args.report).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--responses", required=True, help="freqblimp-all-annotators.jsonl (merged export)")
    ap.add_argument("--freqblimp", default="/Users/tyronewhite/masters_research_code/freq-blimp/data/freqblimp")
    ap.add_argument("--original", default="/Users/tyronewhite/masters_research_code/freq-blimp/.archive/20260616-cleanup/"
                    "outputs/arr_submission_20260526/FreqBLiMP_data/datasets/blimp_original/blimp_original.jsonl")
    ap.add_argument("--lp-items", default="results/round5/restore/restore_freqblimp_items.parquet")
    ap.add_argument("--lp-scores", default="results/round5/restore/restore_freqblimp.parquet")
    ap.add_argument("--out-items", default="data/round6/human/passive_items.csv")
    ap.add_argument("--report", default="reports/round6/human_passives.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    main(ap.parse_args())
