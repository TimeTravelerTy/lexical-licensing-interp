"""Loader for the round-5 D11 expanded verb list (data/round5/verbs_expanded/pairs.csv) with split checks.

load_pairs(split) returns the pairs of one split ("trainable", "heldout" or "all"). It asserts that splits are by
lemma (no lemma in two pairs or in both splits) and that no held-out lemma was used in any earlier DAS training or
selection run (das_historical()). Any DAS training or selection code must load with split="trainable".
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

PAIRS = Path("data/round5/verbs_expanded/pairs.csv")
# earlier DAS training / selection inventories: (path pattern, lemma columns, row filter)
DAS_SOURCES = [
    ("data/das_round2/train_pairs.csv", ("trans", "intrans"), None),
    ("data/das_round2/train_pairs_participle_matched.csv", ("trans", "intrans"), None),
    ("data/das_round2/sensitivity_named/train_pairs.csv", ("trans", "intrans"), None),
    ("data/das_round2/sensitivity_named/train_pairs_participle_matched.csv", ("trans", "intrans"), None),
    ("data/verb_expansion/das_train_pairs.csv", ("trans_lemma", "intrans_lemma"), None),
    ("results/das_v1/*.train_detail.csv", ("clean_verb", "corrupt_verb"), None),
    ("results/das_v1_transfer/*.train_detail.csv", ("clean_verb", "corrupt_verb"), None),
    ("results/das_v2_transfer/*.train_detail.csv", ("clean_verb", "corrupt_verb"), None),
    ("data/passive_das/actives.jsonl", ("verb_pair",), ("split", {"head_train", "head_dev"})),
]


def das_historical():
    """lemma -> sorted list of source files (earlier DAS training or selection)."""
    out = {}
    for pat, cols, filt in DAS_SOURCES:
        files = sorted(glob.glob(pat))
        assert files, pat
        for f in files:
            d = pd.read_json(f, lines=True) if f.endswith(".jsonl") else pd.read_csv(f, keep_default_na=False)
            if filt:
                d = d[d[filt[0]].isin(filt[1])]
            for c in cols:
                vals = d[c].astype(str).str.lower()
                if c == "verb_pair":  # band/trans_lemma/intrans_lemma
                    vals = pd.concat([vals.str.split("/").str[1], vals.str.split("/").str[2]])
                for v in set(vals):
                    out.setdefault(v, set()).add(f)
    return {k: sorted(v) for k, v in out.items()}


def load_pairs(split="trainable", path=PAIRS):
    P = pd.read_csv(path, keep_default_na=False)
    assert set(P.split) <= {"trainable", "heldout"}
    lem = pd.concat([P.t_lemma, P.i_lemma])
    assert lem.is_unique, "a lemma appears in two pairs"
    held = set(P[P.split == "heldout"].t_lemma) | set(P[P.split == "heldout"].i_lemma)
    train = set(P[P.split == "trainable"].t_lemma) | set(P[P.split == "trainable"].i_lemma)
    assert not held & train, "lemma in both splits"
    assert not held & set(das_historical()), "held-out lemma in an earlier DAS training / selection run"
    return P if split == "all" else P[P.split == split].reset_index(drop=True)
