#!/usr/bin/env python3
"""Active-frame readout prompts for DAS round 2 (no training).

Verbs: every candidate in `data/verb_expansion/verbs.csv` (included or not)
plus the 252 verbs of the original 126 pairs.

Frames
- `perfect`: "<Subj> has/have <participle>", Subj in She (primary), He,
  They, We, I, Maria, David. The participle is the same string as in the
  passive test, and the auxiliary rules out a reduced-relative reading.
- `active`:  "The <agent> <past>" with the 90 curated `passive_1` agents
  (the round-1 frame, kept for comparison).

Also writes `extra_tokens.json`: tokens tracked on top of the readout
script's defaults (more object starts and intransitive continuations).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from build_das_prep_prompts import PAST, pid

SUBJECTS = [("She", "has"), ("He", "has"), ("They", "have"), ("We", "have"), ("I", "have"),
            ("Maria", "has"), ("David", "has")]
EXTRA_TOKENS = [" me", " us", " himself", " herself", " themselves", " itself", " myself", " yourself",
                " my", " our", " your", " these", " those", " every", " each", " both", " several",
                " about", " of", " upon", " against", " through", " after", " under", " around", " across",
                " toward", " back", " away", " here", " there", " again", " already", " never", " not",
                " so", " just", " well", " but", " or", " because", " when", " while", " since", " before",
                " until", " like", " without", " during", " behind", " near", " onto", " together",
                " alone", " too", " all", "!", "?", ";", ":"]


def run(args):
    verbs = pd.read_csv(args.verbs)
    rows = [(r.lemma, r.participle, r.past) for r in verbs.itertuples()]
    v2 = [json.loads(l) for l in Path(args.v2_pairs).open(encoding="utf-8")]
    for r in v2:
        for side in ("good", "bad"):
            part = r[f"{side}_verb"]
            rows.append((r[f"{side}_lemma"], part, PAST.get(part, part)))
    forms = pd.DataFrame(rows, columns=["lemma", "participle", "past"]).drop_duplicates("lemma")
    agents = sorted({r["agent"] for r in v2 if r["paradigm"] == "passive_1" and r["frame_id"] == "was"})
    out = {}
    for f in forms.itertuples():
        for subj, aux in SUBJECTS:
            p = f"{subj} {aux} {f.participle}"
            out.setdefault(pid("perfect", p), {"prompt_id": pid("perfect", p), "set": "perfect", "prompt": p,
                                               "lemma": f.lemma, "subject": subj})
        for a in agents:
            p = f"The {a} {f.past}"
            out.setdefault(pid("active", p), {"prompt_id": pid("active", p), "set": "active", "prompt": p,
                                              "lemma": f.lemma, "subject": a})
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    with (od / "prompts.jsonl").open("w", encoding="utf-8") as fh:
        for r in out.values():
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (od / "extra_tokens.json").write_text(json.dumps(EXTRA_TOKENS) + "\n", encoding="utf-8")
    forms.to_csv(od / "verb_forms.csv", index=False)
    print(json.dumps({"verbs": len(forms), "agents": len(agents), "prompts": len(out),
                      "by_set": pd.Series([r["set"] for r in out.values()]).value_counts().to_dict()}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbs", default="data/verb_expansion/verbs.csv")
    ap.add_argument("--v2-pairs", default="data/matched_passives_v2/pairs.jsonl")
    ap.add_argument("--out-dir", default="data/das_round2")
    run(ap.parse_args())
