#!/usr/bin/env python3
"""Pooled intransitive candidates for the verb-set expansion (paper plan, work item 13), for a manual spot-check.

Sources (verbs already in the 126 original pairs are dropped):
  expansion  data/verb_expansion/verbs.csv, intransitives included after Codex's adversarial review (with its notes)
  freqblimp  FreqBLiMP's strict-intransitive participles (freq-blimp/vocabulary_overlay.csv: verb, en,
             strict_intrans) not in the expansion; lemma from the overlay root, past from the same root's past row

Per verb: participle / past / lemma Zipf, summed-lemma Zipf, FreqBLiMP band and the expansion tag (both as
check_verb_candidates.py), Pythia-1.4B tokens of " participle" and " past", and Pile counts where round 6 fetched
them (results/round6/exposure/a1_verbs.csv: c_total, c_was; infini-gram v4_piletrain_llama). Empty `keep` / `note`
columns are for the spot-check.

Zipf needs wordfreq + lemminflect; tokens need the Pythia tokenizer. If this interpreter lacks `tokenizers`,
pass --tok-python (an interpreter that has it); the token step runs there.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

TOK_SNIPPET = r"""
import json, sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1])
words = json.load(open(sys.argv[2]))
out = {w: tok.encode(" " + w, add_special_tokens=False).tokens for w in words}
json.dump(out, open(sys.argv[3], "w"))
"""


def tokenize(words, tok_file, tok_python, tmp):
    wj, oj = tmp / "words.json", tmp / "tokens.json"
    wj.write_text(json.dumps(sorted(words)))
    cmd = [tok_python or sys.executable, "-c", TOK_SNIPPET, str(tok_file), str(wj), str(oj)]
    subprocess.run(cmd, check=True)
    return json.loads(oj.read_text())


def run(a):
    from wordfreq import zipf_frequency

    from check_verb_candidates import band_of, lemma_sum_zipf, tag_of

    v2 = pd.DataFrame([json.loads(l) for l in Path(a.v2_pairs).open(encoding="utf-8")])
    orig_lemmas = set(v2.good_lemma) | set(v2.bad_lemma)
    orig_forms = set(v2.good_verb) | set(v2.bad_verb)

    ve = pd.read_csv(a.expansion)
    ex = ve[(ve["class"] == "intrans") & ve.included].copy()
    ex = pd.DataFrame({"source": "expansion", "lemma": ex.lemma, "past": ex.past, "participle": ex.participle,
                       "expansion_use": ex.use, "review_note": ex.reason.fillna(""),
                       "codex_justification": ex.justification.fillna("")})

    ov = pd.read_csv(a.overlay, low_memory=False, usecols=["expression", "verb", "en", "past", "strict_intrans", "root"])
    ov = ov[ov.verb == 1]
    si = ov[(ov.en == 1) & (ov.strict_intrans == 1)].drop_duplicates("expression")
    past = ov[ov.past == 1].drop_duplicates("root").set_index("root").expression
    fb = pd.DataFrame({"source": "freqblimp", "participle": si.expression.str.strip().to_numpy(),
                       "lemma": si.root.str.split("_overlay").str[0].to_numpy(),
                       "past": si.root.map(past).fillna("").to_numpy(),
                       "freqblimp_template": si.root.str.split("_overlay_").str[1].fillna("").to_numpy()})
    fb = fb[~fb.participle.isin(set(ex.participle)) & ~fb.lemma.isin(set(ex.lemma))]
    bad_past = (fb.past == "") | (fb.past == fb.lemma) & (fb.participle != fb.lemma)
    fb.loc[bad_past, "past"] = fb.participle  # missing or bare-form past rows: regular verbs, past = participle

    C = pd.concat([ex, fb], ignore_index=True).fillna("")
    C = C[~C.lemma.isin(orig_lemmas) & ~C.participle.isin(orig_forms)].drop_duplicates("lemma")
    C["participle_zipf"] = [zipf_frequency(p, "en") for p in C.participle]
    C["past_zipf"] = [zipf_frequency(p, "en") for p in C.past]
    C["lemma_sum_zipf"] = [lemma_sum_zipf(l, pa, p) for l, pa, p in zip(C.lemma, C.past, C.participle)]
    C["band"] = C.participle_zipf.map(band_of)
    C["tag"] = [tag_of(b, z, s) for b, z, s in zip(C.band, C.participle_zipf, C.lemma_sum_zipf)]

    toks = tokenize(set(C.participle) | set(C.past), a.tokenizer, a.tok_python, Path(a.out).parent)
    for form in ("participle", "past"):
        C[f"{form}_ntok"] = C[form].map(lambda w: len(toks[w]))
        C[f"{form}_tokens"] = C[form].map(lambda w: "|".join(toks[w]))

    pile = pd.read_csv(a.pile).drop_duplicates("verb").set_index("verb")
    C["pile_c_total"] = C.participle.map(pile.c_total)
    C["pile_c_was"] = C.participle.map(pile.c_was)
    flags = []
    for r in C.itertuples():
        f = []
        if r.lemma_sum_zipf - r.participle_zipf >= 1.5:
            f.append("summed-lemma Zipf >> participle (noun/adjective homograph?)")
        if r.participle_zipf < 1.2:
            f.append("participle below FreqBLiMP xtail (Zipf < 1.2)")
        if r.past != r.participle:
            f.append("irregular past")
        flags.append("; ".join(f))
    C["auto_flags"] = flags
    C["keep"], C["note"] = "", ""
    order = {"head": 0, "near_head": 1, "tail": 2, "gap": 3, "xtail": 4, "out": 5}
    C = C.sort_values(["tag", "participle_zipf"], key=lambda s: s.map(order) if s.name == "tag" else -s)
    cols = ["source", "lemma", "past", "participle", "tag", "band", "participle_zipf", "lemma_sum_zipf", "past_zipf",
            "participle_ntok", "past_ntok", "participle_tokens", "pile_c_total", "pile_c_was", "expansion_use",
            "freqblimp_template", "auto_flags", "review_note", "codex_justification", "keep", "note"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    C[cols].to_csv(a.out, index=False)
    single = C.participle_ntok == 1
    print(f"{len(C)} candidates -> {a.out}")
    print(pd.crosstab([C.source], [C.tag]).to_string())
    print(pd.crosstab(C.tag, single.map({True: "single", False: "multi"})).to_string())
    print(f"Pile counts available: {C.pile_c_total.notna().sum()}; auto-flagged homograph: "
          f"{C.auto_flags.str.contains('homograph').sum()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2-pairs", default="data/matched_passives_v2/pairs.jsonl")
    ap.add_argument("--expansion", default="data/verb_expansion/verbs.csv")
    ap.add_argument("--overlay", default="../freq-blimp/vocabulary_overlay.csv")
    ap.add_argument("--pile", default="results/round6/exposure/a1_verbs.csv")
    ap.add_argument("--tokenizer", required=True, help="Pythia-1.4B tokenizer.json")
    ap.add_argument("--tok-python", default="", help="interpreter with `tokenizers`, if this one lacks it")
    ap.add_argument("--out", default="data/verb_expansion/intrans_spotcheck.csv")
    run(ap.parse_args())
