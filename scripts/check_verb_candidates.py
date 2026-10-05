#!/usr/bin/env python3
"""Mechanical checks for verb candidates (passive DAS verb-set expansion).

Input CSV columns: `list, class, lemma, past, participle, justification,
round`. Lists: `transitive`, `intransitive`, `head_multi`, `tail_single`;
class: `trans` / `intrans`. Linguistic criteria (transitivity, alternation,
particles) are checked by hand, not here.

- Zipf: wordfreq English Zipf of the realised participle, as FreqBLiMP
  bands verbs (`freq-blimp/utils/frequency.py: zipf_for_expression`; see
  `data/matched_passives_v2/README.md`). Bands: head [3.5, 5.5],
  tail [2.4, 3.2], xtail [1.2, 2.2]. Lemma and past Zipf kept for matching.
- Summed-lemma Zipf: Zipf of the summed wordfreq frequency of all verb
  forms (lemma, -s, past, participle, -ing; lemminflect plus the given
  forms). Bare forms also count noun/adjective uses (e.g. "look").
- Tag (one per verb, in priority order): `head` (participle in the Head
  band), `near_head` (participle Zipf in (3.2, 3.5), or summed-lemma Zipf
  >= 3.5 with the participle below Head), then `tail` / `xtail`. `near_head`
  is a DAS-training-only pool, excluded from all band comparisons; band
  labels for eval stay participle-based.
- Tokens: Pythia-1.4B tokens of " participle" and " past".
- Duplicates against the 126 original pairs, earlier rows, and earlier
  rounds (`--seen`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import math

import pandas as pd
from lemminflect import getAllInflections
from wordfreq import word_frequency, zipf_frequency

BANDS = {"head": (3.5, 5.5), "tail": (2.4, 3.2), "xtail": (1.2, 2.2)}


def band_of(z):
    for b, (lo, hi) in BANDS.items():
        if lo <= z <= hi:
            return b
    return "gap" if 1.2 <= z <= 5.5 else "out"


NEAR_HEAD_LO = 3.2


def lemma_sum_zipf(lemma, *forms):
    allf = {lemma, *forms}
    for fs in getAllInflections(lemma, upos="VERB").values():
        allf.update(fs)
    total = sum(word_frequency(f, "en") for f in allf)
    return round(math.log10(total * 1e9), 2) if total > 0 else 0.0


def tag_of(band, part_zipf, sum_zipf):
    if band == "head":
        return "head"
    if NEAR_HEAD_LO < part_zipf < BANDS["head"][0] or (part_zipf < BANDS["head"][0] and sum_zipf >= BANDS["head"][0]):
        return "near_head"
    return band


def check(df, orig, tok, seen=None):
    rows, seen = [], set(seen or ())
    for row in df.to_dict("records"):
        lemma, past, part = (row[k].strip().lower() for k in ("lemma", "past", "participle"))
        row.update(lemma=lemma, past=past, participle=part)
        row["participle_zipf"] = zipf_frequency(part, "en")
        row["past_zipf"] = zipf_frequency(past, "en")
        row["lemma_zipf"] = zipf_frequency(lemma, "en")
        row["band"] = band_of(row["participle_zipf"])
        row["lemma_sum_zipf"] = lemma_sum_zipf(lemma, past, part)
        row["tag"] = tag_of(row["band"], row["participle_zipf"], row["lemma_sum_zipf"])
        for form in ("participle", "past"):
            ids = tok.encode(" " + row[form], add_special_tokens=False)
            row[f"{form}_ntok"] = len(ids)
            row[f"{form}_tokens"] = "|".join(tok.convert_ids_to_tokens(ids))
        multi = row["participle_ntok"] > 1 or row["past_ntok"] > 1
        reasons = []
        if lemma in orig:
            reasons.append("dup_original_126")
        if lemma in seen:
            reasons.append("dup_candidate")
        seen.add(lemma)
        if row["tag"] in ("gap", "out"):
            reasons.append(f"zipf_outside_bands({row['participle_zipf']:.2f})")
        if row["list"] == "head_multi" and (row["band"] != "head" or not multi):
            reasons.append("not_head_multi_token")
        if row["list"] == "tail_single" and (row["tag"] not in ("tail", "xtail", "near_head") or multi):
            reasons.append("not_tail_xtail_single_token")
        row["reasons"] = ";".join(reasons)
        row["decision"] = "reject" if reasons else "accept"
        rows.append(row)
    return pd.DataFrame(rows)


def run(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b", local_files_only=True)
    v2 = pd.DataFrame([json.loads(l) for l in Path(args.v2_pairs).open(encoding="utf-8")])
    orig = set(v2.good_lemma) | set(v2.bad_lemma)
    df = pd.read_csv(args.candidates, dtype=str).fillna("")
    seen = set()
    for path in args.seen:
        seen |= set(pd.read_csv(path, dtype=str).lemma.str.lower())
    out = check(df, orig, tok, seen)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out.groupby(["list", "decision"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--v2-pairs", default="data/matched_passives_v2/pairs.jsonl")
    ap.add_argument("--seen", nargs="*", default=[], help="checked CSVs from earlier rounds")
    ap.add_argument("--out", default="data/verb_expansion/work/checked.csv")
    run(ap.parse_args())
