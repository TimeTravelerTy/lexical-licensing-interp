#!/usr/bin/env python3
"""How many frequent plain T / I pairs exist for DAS training? (paper plan, item 14 follow-up)

v is trained on 29 pairs (21 near-head expansion + 8 original head). This counts the frequent (head / near-head)
intransitives in the whole candidate pool, since intransitives are the binding class, and follows each through
the selection steps in the order they were applied:

  1. review / spot-check: dropped as not a usable intransitive (ordinary NP objects, clause-taking, copular, ...)
  2. class rule (round 2): prep_object (argument PP) or contaminated_bad -> excluded from training
  3. behavior filter: >= 4 of 7 "<Subj> has/have <participle>" items on the intransitive side of 0
     (run_das_round2 readout, recomputed here for every surviving verb, Pythia-1.4B bf16)

Sources (deduplicated by lemma; the latest judgment wins, so a D11 keep overrides an earlier expansion drop):
  original  the 126 original pairs, Head band (data/matched_passives_v2/pairs.jsonl)
  expansion data/verb_expansion/verbs.csv (Codex rounds 1-5 + adversarial review; round-2 pool)
  freqblimp data/verb_expansion/intrans_spotcheck.csv, FreqBLiMP strict intransitives (Nov 18 spot-check)
  d11       data/verb_expansion/intrans_spotcheck_d11.csv (R5-D11 list, Nov 19 spot-check)

Writes data/verb_expansion/train_pool_ceiling.csv and prints the counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

FREQ = ("head", "near_head")


def ledger(a):
    rows = []
    cls_r2 = pd.read_csv(a.classes).set_index("lemma")["class"]
    cls_x = pd.read_csv(a.expanded_class).set_index("lemma").das_class
    stored = pd.read_csv(a.stored_pairs)

    v2 = pd.DataFrame([json.loads(l) for l in Path(a.v2_pairs).open(encoding="utf-8")])
    h = v2[(v2.band == "head") & (v2.paradigm == "passive_1") & (v2.frame_id == "was")].drop_duplicates("bad_lemma")
    for r in h.itertuples():
        rows.append({"lemma": r.bad_lemma, "participle": r.bad_verb, "source": "original", "tag": "head",
                     "review": "kept", "class": cls_r2.get(r.bad_lemma, "?")})

    V = pd.read_csv(a.verbs)
    V = V[(V["class"] == "intrans") & V.tag.isin(FREQ)]
    for r in V.itertuples():
        rows.append({"lemma": r.lemma, "participle": r.participle, "source": "expansion", "tag": r.tag,
                     "review": "kept" if r.included else "dropped",
                     "class": cls_r2.get(r.lemma, cls_x.get(r.lemma, "?")) if r.included else ""})

    for src, path in (("freqblimp", a.spotcheck), ("d11", a.spotcheck_d11)):
        S = pd.read_csv(path)
        if "source" in S and src == "freqblimp":
            S = S[S.source == "freqblimp"]
        S = S[S.tag.isin(FREQ)]
        for r in S.itertuples():
            kept = str(r.keep).strip().lower() == "y"
            rows.append({"lemma": r.lemma, "participle": r.participle, "source": src, "tag": r.tag,
                         "review": "kept" if kept else "dropped",
                         "class": cls_x.get(r.lemma, cls_r2.get(r.lemma, "?")) if kept else ""})
    L = pd.DataFrame(rows).drop_duplicates("lemma", keep="last").reset_index(drop=True)
    # kept but never classified: homographs whose participle is unseen (Zipf 0), left unpaired by pairing_rules.md
    L.loc[(L.review == "kept") & (L["class"] == "?"), "class"] = "unpairable"
    L["in_stored_29"] = L.lemma.isin(set(stored.intrans))
    return L


def behavior(L, a):
    import run_das_round2 as d2
    from build_das_round2_prompts import SUBJECTS

    runner = d2.Runner(SimpleNamespace(model=a.model, device=a.device, dtype="bfloat16", bos=None, fp32_unembed=None))
    todo = L[(L.review == "kept") & L["class"].isin(["plain", "prep_object", "contaminated_bad"])]
    prompts, keys = [], []
    for r in todo.itertuples():
        for subj, aux in SUBJECTS:
            prompts.append(f"{subj} {aux} {r.participle}")
            keys.append(r.lemma)
    m, _, _ = runner.natural(prompts, batch=64)
    M = pd.DataFrame({"lemma": keys, "M": m.cpu().numpy()})
    g = M.groupby("lemma").M.agg(n_intrans_side=lambda x: int((x < 0).sum()), M_mean="mean")
    L = L.join(g, on="lemma")
    L["behavior_pass"] = L.n_intrans_side >= 4
    return L


def run(a):
    L = behavior(ledger(a), a)
    L.to_csv(a.out, index=False)
    kept = L[L.review == "kept"]
    plain = kept[kept["class"] == "plain"]
    print(f"frequent intransitive candidates (unique lemmas): {len(L)}")
    print(L.groupby(["source", "tag", "review"]).size().to_string(), "\n")
    print("kept after review, by class:\n" + kept.groupby(["source", "class"]).size().to_string(), "\n")
    print(f"plain: {len(plain)}; pass behavior filter: {int(plain.behavior_pass.sum())}")
    print(plain.groupby(["source", "tag"]).behavior_pass.agg(["size", "sum"]).to_string())
    print("plain, failing:", sorted(plain[~plain.behavior_pass].lemma))
    print(f"stored 29 intransitives found in ledger: {int(L.in_stored_29.sum())}")
    nov = plain[plain.behavior_pass & ~plain.in_stored_29]
    print(f"plain + passing, not in the stored 29: {len(nov)}: {sorted(nov.lemma)}")
    print("prep_object passing filter:", int(kept[kept['class'] == 'prep_object'].behavior_pass.sum()),
          "of", int((kept['class'] == 'prep_object').sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2-pairs", default="data/matched_passives_v2/pairs.jsonl")
    ap.add_argument("--verbs", default="data/verb_expansion/verbs.csv")
    ap.add_argument("--spotcheck", default="data/verb_expansion/intrans_spotcheck.csv")
    ap.add_argument("--spotcheck-d11", default="data/verb_expansion/intrans_spotcheck_d11.csv")
    ap.add_argument("--classes", default="data/das_round2/intrans_classes_final.csv")
    ap.add_argument("--expanded-class", default="data/verb_expansion/expanded_intrans_class.csv")
    ap.add_argument("--stored-pairs", default="data/das_round2/train_pairs.csv")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="data/verb_expansion/train_pool_ceiling.csv")
    run(ap.parse_args())
