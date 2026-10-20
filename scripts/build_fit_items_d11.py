#!/usr/bin/env python3
"""Round 6, A7: fit-rating items for the D11 pairs in the curated passive_1 contexts (reports/round6/plan.md, A7).

For every D11 pair and each of the 126 curated passive_1 contexts: the good sentence "<prefix><T participle><suffix>"
(as round P's cross) and the bad-patient sentence "The <patient> <I past>." (as build_bad_fit_items.py; the patient noun
of each context from data/matched_passives_v2/pairs.jsonl). Item ids are sha1(sentence)[:12] (round P's convention).
Sentences already rated by Gemma-4-31B-it with both prompts in round P (results/fit_ratings/gemma4_31b_it.csv,
bad_gemma4_31b_it.csv) are reused; only the rest are written for rating.
Outputs: data/round6/fit_d11/items.jsonl (to rate), links.csv (pair_id x context_id -> good / bad item ids), meta.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from verb_forms import PAST


def item_id(sentence):
    return hashlib.sha1(sentence.encode()).hexdigest()[:12]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    P = pd.read_csv(args.d11_pairs, keep_default_na=False)
    bca = pd.read_json(args.pairs, lines=True)
    cur = bca[(bca.context_set == "curated") & (bca.paradigm == "passive_1")].drop_duplicates("context_id")
    assert len(cur) == 126
    v2 = [json.loads(l) for l in open(args.v2_pairs)]
    patient = {f"curated/{r['paradigm']}/{r['band']}/{r['good_lemma']}": r["patient"] for r in v2 if r["frame_id"] == "was"}
    rated = set()
    for f in (args.good_ratings, args.bad_ratings):
        R = pd.read_csv(f, usecols=["sentence", "prompt"])
        both = R.groupby("sentence").prompt.nunique()
        rated |= set(both[both == 2].index)
    items, links = {}, []
    for c in cur.itertuples(index=False):
        assert c.prefix.endswith(" ") and c.suffix.startswith(" "), c.context_id
        for p in P.itertuples(index=False):
            good = f"{c.prefix}{p.t_participle}{c.suffix}"
            bad = f"The {patient[c.context_id]} {PAST.get(p.i_participle, p.i_participle)}."
            gi, bi = item_id(good), item_id(bad)
            for iid, s, kind, par in ((gi, good, "cross_d11", "passive_1"), (bi, bad, "bad_active", "active")):
                if s not in rated:
                    items.setdefault(iid, {"item_id": iid, "kind": kind, "paradigm": par, "sentence": s})
            links.append({"pair_id": p.pair_id, "band": p.band, "context_id": c.context_id, "context_band": c.context_band,
                          "good_item": gi, "good_sentence": good, "bad_item": bi, "bad_sentence": bad})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "items.jsonl").open("w") as fh:
        for it in sorted(items.values(), key=lambda x: x["item_id"]):
            fh.write(json.dumps(it) + "\n")
    L = pd.DataFrame(links)
    L.to_csv(out / "links.csv", index=False)
    meta = {"links": len(L), "unique_good": int(L.good_sentence.nunique()), "unique_bad": int(L.bad_sentence.nunique()),
            "to_rate": len(items), "reused": int(len(set(L.good_sentence) | set(L.bad_sentence)) - len(items)),
            "inputs": {p: sha(p) for p in (args.d11_pairs, args.pairs, args.v2_pairs, args.good_ratings, args.bad_ratings)},
            "outputs": {f: sha(out / f) for f in ("items.jsonl", "links.csv")}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("links", "unique_good", "unique_bad", "to_rate", "reused")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d11-pairs", default="data/round5/verbs_expanded/pairs.csv")
    ap.add_argument("--pairs", default="data/passive_band_cross/pairs.jsonl")
    ap.add_argument("--v2-pairs", default="data/matched_passives_v2/pairs.jsonl")
    ap.add_argument("--good-ratings", default="results/fit_ratings/gemma4_31b_it.csv")
    ap.add_argument("--bad-ratings", default="results/fit_ratings/bad_gemma4_31b_it.csv")
    ap.add_argument("--out-dir", default="data/round6/fit_d11")
    main(ap.parse_args())
