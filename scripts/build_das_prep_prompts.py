#!/usr/bin/env python3
"""Build next-token readout prompts for the passive DAS groundwork (tasks 1-4).

Prompt sets (one row per unique prompt text within a set):

- `passive`:  curated `was` prefix + participle, "The hat was doffed", for
  every curated context x every verb (good and bad). `passive_1` and
  `passive_2` curated contexts share the patient, so this prefix covers both.
- `active`:   "The gentleman doffed" / "The gentleman salivated", every
  curated agent (both paradigms) x every verb in simple past.
- `released`: released `passive_1` prefix + participle, "The opinion wasn't
  insured", every released context x every verb (the *by* margin contexts).
- `yesno`:    the FreqBLiMP base-model Yes/No prompt
  (blimp-rare `src/acceptability_scoring.py`, YES_NO_BASE_PROMPT) around every
  curated good and bad sentence, both paradigms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

YES_NO_BASE_PROMPT = (
    "Your task is to evaluate the quality of given text.\n"
    "Is the following sentence grammatically acceptable?\n\n"
    "{sentence}\n"
    "Respond with Yes or No as your answer. Answer:"
)
# Simple past differs from the participle only here (bad verbs: all equal).
PAST = {"outdone": "outdid"}


def pid(kind: str, text: str) -> str:
    return f"{kind}:" + hashlib.sha1(text.encode()).hexdigest()[:12]


def run(args):
    scores = pd.read_csv(args.scores, low_memory=False)
    v2 = pd.DataFrame([json.loads(l) for l in Path(args.v2_pairs).open(encoding="utf-8")])
    v2 = v2[v2.frame_id == "was"]
    verbs = pd.concat([scores[["good_verb"]].rename(columns={"good_verb": "verb"}),
                       scores[["bad_verb"]].rename(columns={"bad_verb": "verb"})]).verb.unique()
    rows = {}

    def add(kind, prompt, **meta):
        key = pid(kind, prompt)
        rows.setdefault(key, {"prompt_id": key, "set": kind, "prompt": prompt, **meta})

    cur = scores[scores.context_set == "curated"]
    for prefix in sorted(cur.prefix.unique()):
        for verb in verbs:
            add("passive", prefix + verb, prefix=prefix, verb=verb)
    for agent in sorted(v2.agent.unique()):
        for verb in verbs:
            add("active", f"The {agent} {PAST.get(verb, verb)}", subject=agent, verb=verb)
    rel = scores[(scores.context_set == "released") & (scores.paradigm == "passive_1")]
    for prefix in sorted(rel.prefix.unique()):
        for verb in verbs:
            add("released", prefix + verb, prefix=prefix, verb=verb)
    for sentence in sorted(set(cur.sentence_good) | set(cur.sentence_bad)):
        add("yesno", YES_NO_BASE_PROMPT.format(sentence=sentence), sentence=sentence)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = pd.Series([r["set"] for r in rows.values()]).value_counts().to_dict()
    print(json.dumps({"out": str(out), "prompts": len(rows), "by_set": counts}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default="results/passive_band_cross/pythia14b_scores.csv")
    ap.add_argument("--v2-pairs", default="data/matched_passives_v2/pairs.jsonl")
    ap.add_argument("--out", default="data/passive_das_prep/prompts.jsonl")
    run(ap.parse_args())
