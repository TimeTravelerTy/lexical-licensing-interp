#!/usr/bin/env python3
"""Nonce-verb passive prompts (round3_plan.md, C8). No model.

80 nonce lemmas (`data/nonce_passive/lemmas.csv`, the July selection) with their
regular past forms. Probe = a test sentence with lemma A: passive "The N was
A-ed" (main) or active "SUBJ has A-ed" (reference), 4 slots per lemma (nouns
and subjects counterbalanced over lemmas). Contexts (animate agents, July
lead schemas plus a fourth):
- matched_T / matched_I: 3 sentences with A, with / without objects;
- mismatched{1,2}_T / _I: the same with partner B under two fixed derangements;
- balanced_AB / balanced_BA: 4 sentences A, B, A, B; in AB, A takes objects and
  B none; in BA the roles swap (same words; only which verb has objects changes);
- none: the probe alone.
Real-verb reference: the good and bad passives of the 64 primary pairs after
`none` and after a fixed neutral context (a mismatched1_I context), 4 slots.
DAS active items are appended (kind `das`) for the projection scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_das_round2_prompts import SUBJECTS
from build_nonce_frames import LEAD_SCHEMAS, OBJECTS, regular_past

SCHEMAS = tuple(LEAD_SCHEMAS) + ("Before the meeting, the {subject} {past}",)
AGENTS = ("artist", "worker", "teacher", "singer")
NOUNS = ("house", "letter", "suspect", "king", "child", "horse", "statue", "thief", "baby", "ship", "suitcase",
         "wallet")
AUX = dict(SUBJECTS)
SUBJ = [s for s, _ in SUBJECTS]
N_SLOTS = 4


def sentences(verbs_objs):
    """[(past, obj or None)] -> context text, one sentence per entry (schema i, agent i)."""
    out = []
    for i, (past, obj) in enumerate(verbs_objs):
        core = SCHEMAS[i].format(subject=AGENTS[i], past=past)
        out.append(f"{core} the {obj}." if obj else f"{core}.")
    return " ".join(out)


def run(args):
    rng = np.random.default_rng(args.seed)
    lemmas = pd.read_csv(args.lemmas).lemma.tolist()
    partners = []
    for _ in range(2):
        order = rng.permutation(len(lemmas))
        partners.append({lemmas[order[i]]: lemmas[order[(i + 1) % len(order)]] for i in range(len(order))})
    rows = []
    neutral = []
    for li, lemma in enumerate(lemmas):
        a = regular_past(lemma)
        objs = [OBJECTS[j] for j in rng.choice(len(OBJECTS), 4, replace=False)]
        ctx = {"matched_T": sentences([(a, o) for o in objs[:3]]), "matched_I": sentences([(a, None)] * 3),
               "none": ""}
        for m, pm in enumerate(partners, start=1):
            b = regular_past(pm[lemma])
            ctx[f"mismatched{m}_T"] = sentences([(b, o) for o in objs[:3]])
            ctx[f"mismatched{m}_I"] = sentences([(b, None)] * 3)
        b = regular_past(partners[0][lemma])
        ctx["balanced_AB"] = sentences([(a, objs[0]), (b, None), (a, objs[1]), (b, None)])
        ctx["balanced_BA"] = sentences([(a, None), (b, objs[0]), (a, None), (b, objs[1])])
        neutral.append(ctx["mismatched1_I"])
        for slot in range(N_SLOTS):
            noun, s = NOUNS[(li + 3 * slot) % len(NOUNS)], SUBJ[(li + 2 * slot) % len(SUBJ)]
            for ptype, probe in (("passive", f"The {noun} was {a}"), ("active", f"{s} {AUX[s]} {a}")):
                for cond, c in ctx.items():
                    ctx_lemma = {"matched": lemma, "mismatched1": partners[0][lemma], "mismatched2": partners[1][lemma],
                                 "balanced": f"{lemma}+{partners[0][lemma]}", "none": ""}[cond.rsplit("_", 1)[0]
                                                                                       if cond != "none" else "none"]
                    rows.append({"kind": "nonce", "lemma": lemma, "past": a, "slot": slot, "probe_type": ptype,
                                 "cond": cond, "context_lemma": ctx_lemma, "probe": probe,
                                 "prompt": f"{c} {probe}" if c else probe})
    items = pd.read_csv(args.passive_items)
    pr = items[items.bad_class == "plain"].drop_duplicates("pair_id").reset_index(drop=True)
    for pi, p in enumerate(pr.itertuples()):
        for slot in range(N_SLOTS):
            noun = NOUNS[(pi + 3 * slot) % len(NOUNS)]
            for cond, c in (("none", ""), ("neutral", neutral[pi % len(neutral)])):
                for side in ("good", "bad"):
                    probe = f"The {noun} was {getattr(p, f'{side}_part')}"
                    rows.append({"kind": "real", "lemma": getattr(p, f"{side}_lemma"), "past": getattr(p, f"{side}_part"),
                                 "slot": slot, "probe_type": f"real_{side}", "cond": cond, "context_lemma": p.pair_id,
                                 "probe": probe, "prompt": f"{c} {probe}" if c else probe})
    das = pd.read_csv(args.das_items)
    for r in das.itertuples():
        rows.append({"kind": "das", "lemma": r.verb, "past": "", "slot": -1, "probe_type": "das", "cond": "",
                     "context_lemma": "", "probe": r.prompt, "prompt": r.prompt})
    P = pd.DataFrame(rows)
    P = P.drop_duplicates(["prompt", "kind", "probe_type", "cond", "lemma", "slot"]).reset_index(drop=True)
    P.insert(0, "pid", np.arange(len(P)))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    P.to_csv(out / "prompts.csv", index=False)
    meta = {"lemmas": len(lemmas), "slots": N_SLOTS, "prompts": len(P), "by_kind": P.kind.value_counts().to_dict(),
            "by_cond": P[P.kind == "nonce"].cond.value_counts().to_dict(), "seed": args.seed,
            "partners_sha256": hashlib.sha256(json.dumps(partners, sort_keys=True).encode()).hexdigest(),
            "example": P[(P.kind == "nonce") & (P.slot == 0) & (P.lemma == lemmas[0]) & (P.probe_type == "passive")]
            .prompt.tolist()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lemmas", default="data/nonce_passive/lemmas.csv")
    ap.add_argument("--passive-items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--das-items", default="results/das_round2/final_strict/items.csv")
    ap.add_argument("--out-dir", default="data/nonce_passive")
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
