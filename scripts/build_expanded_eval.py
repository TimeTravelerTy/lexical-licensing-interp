#!/usr/bin/env python3
"""Inputs for the expanded eval set (rules: reports/expanded_eval_rules.md). No model.

New pairs = data/verb_expansion/expanded_pairs.csv minus das_train_overlap (142). Writes to --out-dir:
- eval_pairs_new.csv   for build_passive_test.py --eval-pairs (band = intransitive tag)
- leace_extra_pairs.csv for run_leace_necessity.py --extra-pairs
- classes.csv          bad-verb classes keyed (lemma, source): original rows of intrans_classes_final.csv, new rows from
                       expanded_intrans_class.csv (plain / prep_object)
- verb_forms.csv       data/das_round2/verb_forms.csv plus the new verbs' forms (for build_constructions.py)
- pairs_meta.csv       one row per new pair: class, tags, Zipfs, token counts, selection flag, transitive source
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def run(a):
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X = pd.read_csv(a.expanded)
    X = X[~X.das_train_overlap].reset_index(drop=True)
    cls = pd.read_csv(a.intrans_class).set_index("lemma").das_class
    X["bad_class"] = X.intrans_lemma.map(cls)
    assert X.bad_class.isin(["plain", "prep_object"]).all(), X[~X.bad_class.isin(["plain", "prep_object"])]
    X["pair_id"] = "new/" + X.intrans_tag + "/" + X.trans_lemma + "/" + X.intrans_lemma
    assert X.pair_id.is_unique and len(X) == a.expect, len(X)
    orig = pd.read_csv(a.classes)
    orig = orig[orig.source == "original"]
    clash = set(X.trans_lemma) | set(X.intrans_lemma)
    assert not clash & set(orig.lemma), "a new verb is an original verb"
    pd.DataFrame({"band": X.intrans_tag, "trans_lemma": X.trans_lemma, "intrans_lemma": X.intrans_lemma,
                  "trans_participle": X.trans_participle, "intrans_participle": X.intrans_participle}
                 ).to_csv(out / "eval_pairs_new.csv", index=False)
    pd.DataFrame({"verb_pair": X.pair_id, "verb_band": X.intrans_tag, "good_verb": X.trans_participle,
                  "bad_verb": X.intrans_participle}).to_csv(out / "leace_extra_pairs.csv", index=False)
    new_cls = pd.DataFrame({"lemma": X.intrans_lemma, "source": "new", "class": X.bad_class,
                            "prep_subtype": np.where(X.bad_class == "prep_object", "unclassified", None)})
    pd.concat([orig[["lemma", "source", "class", "prep_subtype"]], new_cls], ignore_index=True
              ).to_csv(out / "classes.csv", index=False)
    vf = pd.read_csv(a.verb_forms)
    add = pd.concat([pd.DataFrame({"lemma": X.trans_lemma, "participle": X.trans_participle, "past": X.trans_past}),
                     pd.DataFrame({"lemma": X.intrans_lemma, "participle": X.intrans_participle, "past": X.intrans_past})])
    add = add[~add.lemma.isin(vf.lemma)]
    pd.concat([vf, add], ignore_index=True).drop_duplicates("lemma").to_csv(out / "verb_forms.csv", index=False)
    meta = X[["pair_id", "trans_lemma", "intrans_lemma", "bad_class", "trans_tag", "intrans_tag", "batch",
              "intrans_selection", "trans_source", "participle_ntok", "trans_participle_zipf", "intrans_participle_zipf",
              "trans_lemma_sum_zipf", "intrans_lemma_sum_zipf", "trans_tokens", "intrans_tokens"]].copy()
    meta["single_token"] = meta.participle_ntok == 1
    meta.to_csv(out / "pairs_meta.csv", index=False)
    print(meta.groupby(["bad_class", "intrans_tag"]).size().to_string())
    print("new pairs", len(X), "plain", int((X.bad_class == "plain").sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expanded", default="data/verb_expansion/expanded_pairs.csv")
    ap.add_argument("--intrans-class", default="data/verb_expansion/expanded_intrans_class.csv")
    ap.add_argument("--classes", default="data/das_round2/intrans_classes_final.csv")
    ap.add_argument("--verb-forms", default="data/das_round2/verb_forms.csv")
    ap.add_argument("--expect", type=int, default=142)
    ap.add_argument("--out-dir", default="data/verb_expansion/eval")
    run(ap.parse_args())
