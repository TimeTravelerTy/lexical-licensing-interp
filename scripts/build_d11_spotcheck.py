#!/usr/bin/env python3
"""Spot-check list for the R5-D11 intransitives not yet judged (paper plan, item 13; Nov 19).

D11 (`data/round5/verbs_expanded/verbs.csv`) accepted 306 intransitives by blind Claude / Codex rating (κ 0.88) under
looser criteria (rare transitive uses allowed, flag `rare_trans`). The ones not in the 126 original pairs, not in
`intrans_spotcheck.csv` (already judged) and not in v's 29 DAS training pairs are listed here with the same columns as
`intrans_spotcheck.csv`, for the same spot-check criteria. Run from scripts/ (needs wordfreq, lemminflect; tokens via
--tok-python as in build_intrans_spotcheck.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_intrans_spotcheck import tokenize


def run(a):
    from lemminflect import getInflection
    from wordfreq import zipf_frequency

    from check_verb_candidates import band_of, lemma_sum_zipf, tag_of

    V = pd.read_csv(a.d11_verbs)
    v2 = pd.DataFrame([json.loads(l) for l in Path(a.v2_pairs).open(encoding="utf-8")])
    S = pd.read_csv(a.spotcheck)
    T = pd.read_csv(a.das_pairs)
    seen = set(v2.good_lemma) | set(v2.bad_lemma) | set(S.lemma) | set(T.trans) | set(T.intrans)
    I = V[(V.accepted == "I") & ~V.lemma.isin(seen)].copy()
    C = pd.DataFrame({"source": "d11", "lemma": I.lemma, "participle": I.participle,
                      "past": [(getInflection(l, tag="VBD") or (p,))[0] for l, p in zip(I.lemma, I.participle)],
                      "d11_band": I.band, "d11_claude": I.claude_label + " " + I.claude_flag.fillna("") + ": " +
                      I.claude_reason.fillna(""), "d11_codex": I.codex_label + " " + I.codex_flag.fillna("") + ": " +
                      I.codex_reason.fillna("")})
    C["participle_zipf"] = [zipf_frequency(p, "en") for p in C.participle]
    C["past_zipf"] = [zipf_frequency(p, "en") for p in C.past]
    C["lemma_sum_zipf"] = [lemma_sum_zipf(l, pa, p) for l, pa, p in zip(C.lemma, C.past, C.participle)]
    C["band"] = C.participle_zipf.map(band_of)
    C["tag"] = [tag_of(b, z, s) for b, z, s in zip(C.band, C.participle_zipf, C.lemma_sum_zipf)]
    toks = tokenize(set(C.participle) | set(C.past), a.tokenizer, a.tok_python, Path(a.out).parent)
    for form in ("participle", "past"):
        C[f"{form}_ntok"] = C[form].map(lambda w: len(toks[w]))
    C["participle_tokens"] = C.participle.map(lambda w: "|".join(toks[w]))
    C["auto_flags"] = ["summed-lemma Zipf >> participle (noun/adjective homograph?)" if s - z >= 1.5 else ""
                       for s, z in zip(C.lemma_sum_zipf, C.participle_zipf)]
    C["keep"], C["note"] = "", ""
    C = C.sort_values("participle_zipf", ascending=False)
    C.to_csv(a.out, index=False)
    print(f"{len(C)} -> {a.out}")
    print(C.tag.value_counts().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d11-verbs", default="../data/round5/verbs_expanded/verbs.csv")
    ap.add_argument("--v2-pairs", default="../data/matched_passives_v2/pairs.jsonl")
    ap.add_argument("--spotcheck", default="../data/verb_expansion/intrans_spotcheck.csv")
    ap.add_argument("--das-pairs", default="../data/das_round2/train_pairs.csv")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tok-python", default="")
    ap.add_argument("--out", default="../data/verb_expansion/intrans_spotcheck_d11.csv")
    run(ap.parse_args())
