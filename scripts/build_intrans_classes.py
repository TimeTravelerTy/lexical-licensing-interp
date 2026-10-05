#!/usr/bin/env python3
"""Final classes for the 225 intransitives (126 original bad verbs + 99 new).

Rule (round2.md): a verb is `prep_object` if its PP is an argument, i.e. a
reciprocal *with*-partner or a lexically selected, typical preposition;
locative / directional PPs leave it `plain`. `contaminated_bad` = original
bad verbs with an ordinary transitive use (jut, scram, crackle, resound,
pee, hibernate, reel).

Sources, in order:
1. FreqBLiMP verb inventory
   (`freq-blimp/generation_projects/blimp/verb_inventory.json`): a verb with
   `intr_pp` frames but no bare `intr` frame has an obligatory preposition,
   so it is `prep_object`. The inventory lists every natural verb+P
   combination, adjuncts included (die for/of/with, exist for), so beyond
   obligatoriness it cannot separate argument from adjunct PPs.
2. Otherwise the rule-based classification in `intrans_classes.csv` (round-2
   hand classification, Codex cross-check) plus the rule moves below.

prep_object subtype: `pseudo_passive_ok` if a natural pseudo-passive was
judged to exist (my first pass in `intrans_classes_mine.csv` or Codex's
blind pass), else `pseudo_passive_bad` (belong to, result in, succumb to).

Writes `intrans_classes_final.csv` and `class_disagreements.csv` (for review).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

NAMED = {"chat": "with", "quarrel": "with", "bicker": "with", "compete": "with"}
RULE = {"coincide": "with", "converse": "with", "differ": "from", "excel": "at", "feud": "with",
        "intervene": "in", "succumb": "to", "triumph": "over", "elope": "with", "hobnob": "with",
        "intercede": "for", "luxuriate": "in", "cohabit": "with", "fraternize": "with",
        "abstain": "from", "banter": "with", "bask": "in", "belong": "to", "clash": "with",
        "coexist": "with", "collude": "with", "cooperate": "with", "correspond": "with",
        "desist": "from", "diverge": "from", "eventuate": "in", "exult": "in", "result": "in",
        "teem": "with", "wallow": "in"}
CONTAMINATED = ["jut", "scram", "crackle", "resound", "pee", "hibernate", "reel"]
ARG_P = {"with", "to", "of", "about", "for", "against"}


def inventory(path):
    return {e["lemma"]: e["frames"] for e in json.loads(Path(path).read_text())}


def run(args):
    c = pd.read_csv(args.classes)
    mine1 = pd.read_csv(args.first_pass).set_index("lemma").mine_class
    inv = inventory(args.inventory)
    rows = []
    for r in c.to_dict("records"):
        lemma = r["lemma"]
        rule_cls, prep, why = r["class"], r["prep"] if isinstance(r["prep"], str) else "", ""
        if lemma in NAMED or lemma in RULE:
            rule_cls, prep, why = "prep_object", {**NAMED, **RULE}[lemma], "argument-PP rule"
        if lemma in CONTAMINATED:
            rule_cls, why = "contaminated_bad", "ordinary transitive use"
        fr = inv.get(lemma)
        types = {f["type"] for f in fr} if fr else set()
        preps = sorted({f["prep"] for f in fr if f["type"] == "intr_pp"}) if fr else []
        oblig = bool(fr) and "intr" not in types and bool(preps)
        heur = "not_in_inventory" if fr is None else ("prep_object" if oblig or set(preps) & ARG_P else "plain")
        cls = rule_cls
        if oblig and rule_cls == "plain":
            cls, why = "prep_object", "inventory: no bare intr frame (obligatory P)"
            prep = prep or "/".join(preps)
        ok = mine1.get(lemma) == "prep_object" or r.get("codex_label") == "prep_object"
        rows.append({"lemma": lemma, "source": r["source"], "class": cls,
                     "prep_subtype": ("pseudo_passive_ok" if ok else "pseudo_passive_bad") if cls == "prep_object" else "",
                     "prep": prep if cls == "prep_object" else "", "decided_by": why or "round-2 classification",
                     "rule_class": rule_cls, "inventory_obligatory_p": oblig, "inventory_p_heuristic": heur,
                     "inventory_frames": " ".join(f"{f['type']}{':' + f['prep'] if f.get('prep') else ''}"
                                                  for f in (fr or [])),
                     "codex_label": r.get("codex_label", ""), "codex_example": r.get("codex_example", ""),
                     "note": r.get("note", "")})
    out = pd.DataFrame(rows)
    od = Path(args.out_dir)
    out.to_csv(od / "intrans_classes_final.csv", index=False)
    dis = out[(out.inventory_p_heuristic != "not_in_inventory") & (out.inventory_p_heuristic != out.rule_class)
              & (out.rule_class != "contaminated_bad")].copy()
    dis["kind"] = [("inventory obligatory P overrides plain" if r.inventory_obligatory_p and r.rule_class == "plain"
                    else "inventory lists an argument-type P, rule says plain" if r.inventory_p_heuristic == "prep_object"
                    else "rule says prep_object, inventory lists spatial P only") for r in dis.itertuples()]
    dis[["lemma", "source", "class", "rule_class", "kind", "inventory_frames", "prep", "decided_by"]] \
        .sort_values(["kind", "lemma"]).to_csv(od / "class_disagreements.csv", index=False)
    print(out.groupby(["source", "class", "prep_subtype"]).size().to_string())
    print(dis.kind.value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--classes", default="data/das_round2/intrans_classes.csv")
    ap.add_argument("--first-pass", default="data/das_round2/intrans_classes_mine.csv")
    ap.add_argument("--inventory", default="../freq-blimp/generation_projects/blimp/verb_inventory.json")
    ap.add_argument("--out-dir", default="data/das_round2")
    run(ap.parse_args())
