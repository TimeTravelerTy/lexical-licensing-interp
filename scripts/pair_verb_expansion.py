#!/usr/bin/env python3
"""Pair the kept intransitives of the verb-set expansion with unused transitives
(rules: data/verb_expansion/pairing_rules.md, declared before this was run).

Intransitives: intrans_spotcheck.csv, keep == y. Transitives: Codex-reviewed (verbs.csv, included) and FreqBLiMP
(vocabulary_overlay.csv: verb, en, passive, strict_trans; lemma / past by lemminflect). Exact Pythia participle token
count; Zipf caliper 0.25 on participle Zipf if either verb is homograph-flagged (summed-lemma − participle ≥ 1.5),
else on summed-lemma Zipf; greedy smallest gap first, reviewed transitives first on ties. --drop-trans removes
transitives that failed the post-pairing screen (their intransitives are re-paired under the same rules).
Run from scripts/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_intrans_spotcheck import tokenize

CAL, HOMO = 0.25, 1.5


def run(a):
    from lemminflect import getInflection, getLemma
    from wordfreq import zipf_frequency

    from check_verb_candidates import band_of, lemma_sum_zipf, tag_of

    v2 = pd.DataFrame([json.loads(l) for l in Path(a.v2_pairs).open(encoding="utf-8")])
    orig = set(v2.good_lemma) | set(v2.bad_lemma) | set(v2.good_verb) | set(v2.bad_verb)
    S = pd.read_csv(a.spotcheck)
    spot_lemmas = set(S.lemma) | set(S.participle)
    I = S[S.keep == "y"].copy()

    V = pd.read_csv(a.expansion)
    tr = V[(V["class"] == "trans") & V.included]
    T1 = pd.DataFrame({"lemma": tr.lemma, "past": tr.past, "participle": tr.participle, "t_source": "reviewed"})
    ov = pd.read_csv(a.overlay, low_memory=False, usecols=["expression", "verb", "en", "passive", "strict_trans"])
    ov = ov[(ov.verb == 1) & (ov.en == 1) & (ov.passive == 1) & (ov.strict_trans == 1)]
    parts = sorted(set(ov.expression.str.strip()))
    rows = []
    for p in parts:
        lem = (getLemma(p, upos="VERB") or (p,))[0]
        past = (getInflection(lem, tag="VBD") or (p,))[0]
        rows.append({"lemma": lem, "past": past, "participle": p, "t_source": "freqblimp"})
    T2 = pd.DataFrame(rows)
    T2 = T2[~T2.lemma.isin(set(T1.lemma)) & ~T2.participle.isin(set(T1.participle))]
    T = pd.concat([T1, T2], ignore_index=True).drop_duplicates("lemma")
    bad = T.lemma.isin(orig | spot_lemmas) | T.participle.isin(orig | spot_lemmas) | T.participle.isin(set(I.participle))
    T = T[~bad & ~T.lemma.isin(set(a.drop_trans))].copy()
    T["participle_zipf"] = [zipf_frequency(p, "en") for p in T.participle]
    T["lemma_sum_zipf"] = [lemma_sum_zipf(l, pa, p) for l, pa, p in zip(T.lemma, T.past, T.participle)]
    T = T[T.participle_zipf > 0]  # not in wordfreq at all
    toks = tokenize(set(T.participle), a.tokenizer, a.tok_python, Path(a.out).parent)
    T["participle_ntok"] = T.participle.map(lambda w: len(toks[w]))
    T["participle_tokens"] = T.participle.map(lambda w: "|".join(toks[w]))
    T["band"] = T.participle_zipf.map(band_of)
    T["tag"] = [tag_of(b, z, s) for b, z, s in zip(T.band, T.participle_zipf, T.lemma_sum_zipf)]
    for D in (I, T):
        D["homograph"] = D.lemma_sum_zipf - D.participle_zipf >= HOMO

    edges = []
    for i in I.itertuples():
        C = T[T.participle_ntok == i.participle_ntok]
        homo = i.homograph | C.homograph
        gap = (C.participle_zipf - i.participle_zipf).where(homo, C.lemma_sum_zipf - i.lemma_sum_zipf)
        ok = gap.abs() <= CAL
        for t, g, h in zip(C.index[ok], gap[ok], homo[ok]):
            edges.append((abs(g), T.at[t, "t_source"] != "reviewed", T.at[t, "lemma"], i.Index, t, g, h))
    edges.sort()
    used_i, used_t, pairs = set(), set(), []
    for ag, _, _, ii, tt, g, h in edges:
        if ii in used_i or tt in used_t:
            continue
        used_i.add(ii); used_t.add(tt)
        x, y = I.loc[ii], T.loc[tt]
        pairs.append({"pair_id": f"x/{y.lemma}/{x.lemma}", "trans_lemma": y.lemma, "trans_past": y.past,
                      "trans_participle": y.participle, "trans_source": y.t_source, "trans_tag": y.tag,
                      "intrans_lemma": x.lemma, "intrans_past": x.past, "intrans_participle": x.participle,
                      "intrans_source": x.source, "intrans_tag": x.tag, "intrans_selection": x.selection
                      if isinstance(x.selection, str) else "", "participle_ntok": int(x.participle_ntok),
                      "match_scale": "participle" if h else "summed_lemma", "match_gap": g,
                      "participle_zipf_gap": y.participle_zipf - x.participle_zipf,
                      "lemma_sum_zipf_gap": y.lemma_sum_zipf - x.lemma_sum_zipf,
                      "trans_participle_zipf": y.participle_zipf, "intrans_participle_zipf": x.participle_zipf,
                      "trans_lemma_sum_zipf": y.lemma_sum_zipf, "intrans_lemma_sum_zipf": x.lemma_sum_zipf,
                      "trans_tokens": y.participle_tokens, "intrans_tokens": x.participle_tokens,
                      "trans_screen": "reviewed" if y.t_source == "reviewed" else ""})
    P = pd.DataFrame(pairs).sort_values(["intrans_participle_zipf"], ascending=False)
    un = I.loc[sorted(set(I.index) - used_i)]
    out = Path(a.out)
    P.to_csv(out, index=False)
    un[["lemma", "participle", "tag", "participle_zipf", "lemma_sum_zipf", "participle_ntok", "homograph"]].to_csv(
        out.with_name("expanded_unpaired.csv"), index=False)
    print(f"{len(P)} pairs, {len(un)} intransitives unpaired -> {out}")
    print(pd.crosstab(P.intrans_tag, P.trans_source, margins=True).to_string())
    print(pd.crosstab(P.participle_ntok, P.trans_source).to_string())
    print(P[["match_gap", "participle_zipf_gap", "lemma_sum_zipf_gap"]].describe().round(3).to_string())
    if len(un):
        print("unpaired:", " ".join(un.participle))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2-pairs", default="../data/matched_passives_v2/pairs.jsonl")
    ap.add_argument("--spotcheck", default="../data/verb_expansion/intrans_spotcheck.csv")
    ap.add_argument("--expansion", default="../data/verb_expansion/verbs.csv")
    ap.add_argument("--overlay", default="../../freq-blimp/vocabulary_overlay.csv")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tok-python", default="")
    ap.add_argument("--drop-trans", nargs="*", default=[], help="transitive lemmas that failed the screen")
    ap.add_argument("--out", default="../data/verb_expansion/expanded_pairs.csv")
    run(ap.parse_args())
