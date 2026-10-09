#!/usr/bin/env python3
"""Round 4, Part C: nonce cue prompts (reports/round4/plan.md, C8 inflection mismatch, C9 cue 2 x 2). No model.

The round-3 C8 lemmas, slots, nouns, agents and passive probe ("The N was A-ed"). Contexts
(three sentences; balanced: four), lemma A, objects as round-3 C8:
- C8 forms: ed_T / ed_I (the round-3 matched contexts), ing_T / ing_I ("is A-ing (the obj)."),
  s_T / s_I ("A-s (the obj)."); balanced AB / BA in each form (A, B alternating; in AB only A takes
  objects; B = the round-3 first-derangement partner; ed_AB / ed_BA = the round-3 balanced
  contexts); mismatched-lemma T / I in the ing and s forms (B in place of A: ing_mT, ing_mI, s_mT,
  s_mI);
- C9 cells: ed_T (transitive + adjacent NP); rel_T ("the obj that the agent A-ed V2.") with its
  structure-matched intransitive rel_I ("the obj near which the agent A-ed V2."); q_T ("what did
  the agent A?") with q_I ("did the agent A?"); adj_I ("A-ed the whole night.") with the
  duration-PP control pp_I ("A-ed throughout the whole night."); ed_I (bare).
DAS active items are appended (kind `das`) for the z scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_nonce_frames import OBJECTS, regular_past
from build_nonce_passive import AGENTS, NOUNS, N_SLOTS, SCHEMAS

ADJUNCTS = ("the whole night", "the whole morning", "the whole afternoon", "the whole evening")
REL_VERBS = ("fell", "broke", "stayed there", "fell")
LEADS = ("In the lab,", "During the check,", "After the visit,", "Before the meeting,")


def ing(stem):
    return f"{stem[:-1]}ing" if stem.endswith("e") else f"{stem}ing"


def s_form(stem):
    assert not stem.endswith(("s", "x", "z", "sh", "ch")), stem
    return f"{stem}s"


def sent(i, core):
    return f"{LEADS[i]} {core}."


def contexts(a, b, objs):
    """Context texts for lemma a (partner b)."""
    pa, pb = regular_past(a), regular_past(b)
    c = {}
    c["ed_T"] = " ".join(sent(i, f"the {AGENTS[i]} {pa} the {objs[i]}") for i in range(3))
    c["ed_I"] = " ".join(sent(i, f"the {AGENTS[i]} {pa}") for i in range(3))
    for nm, fa, fb in (("ed", pa, pb), ("ing", f"is {ing(a)}", f"is {ing(b)}"), ("s", s_form(a), s_form(b))):
        if nm != "ed":
            c[f"{nm}_T"] = " ".join(sent(i, f"the {AGENTS[i]} {fa} the {objs[i]}") for i in range(3))
            c[f"{nm}_I"] = " ".join(sent(i, f"the {AGENTS[i]} {fa}") for i in range(3))
            c[f"{nm}_mT"] = " ".join(sent(i, f"the {AGENTS[i]} {fb} the {objs[i]}") for i in range(3))
            c[f"{nm}_mI"] = " ".join(sent(i, f"the {AGENTS[i]} {fb}") for i in range(3))
        c[f"{nm}_AB"] = " ".join(sent(i, f"the {AGENTS[i]} {fa if i % 2 == 0 else fb}" +
                                     (f" the {objs[i // 2]}" if i % 2 == 0 else "")) for i in range(4))
        c[f"{nm}_BA"] = " ".join(sent(i, f"the {AGENTS[i]} {fa if i % 2 == 0 else fb}" +
                                     (f" the {objs[i // 2]}" if i % 2 == 1 else "")) for i in range(4))
    c["rel_T"] = " ".join(sent(i, f"the {objs[i]} that the {AGENTS[i]} {pa} {REL_VERBS[i]}") for i in range(3))
    c["rel_I"] = " ".join(sent(i, f"the {objs[i]} near which the {AGENTS[i]} {pa} {REL_VERBS[i]}") for i in range(3))
    c["q_T"] = " ".join(f"{LEADS[i]} what did the {AGENTS[i]} {a}?" for i in range(3))
    c["q_I"] = " ".join(f"{LEADS[i]} did the {AGENTS[i]} {a}?" for i in range(3))
    c["adj_I"] = " ".join(sent(i, f"the {AGENTS[i]} {pa} {ADJUNCTS[i]}") for i in range(3))
    c["pp_I"] = " ".join(sent(i, f"the {AGENTS[i]} {pa} throughout {ADJUNCTS[i]}") for i in range(3))
    return c


def run(args):
    rng = np.random.default_rng(args.seed)
    lemmas = pd.read_csv(args.lemmas).lemma.tolist()
    # the round-3 C8 partners and objects: same seed and the same draw order as build_nonce_passive.py
    partners = []
    for _ in range(2):
        order = rng.permutation(len(lemmas))
        partners.append({lemmas[order[i]]: lemmas[order[(i + 1) % len(order)]] for i in range(len(order))})
    rows = []
    for li, lemma in enumerate(lemmas):
        objs = [OBJECTS[j] for j in rng.choice(len(OBJECTS), 4, replace=False)]
        a = regular_past(lemma)
        ctx = contexts(lemma, partners[0][lemma], objs)
        for slot in range(N_SLOTS):
            noun = NOUNS[(li + 3 * slot) % len(NOUNS)]
            probe = f"The {noun} was {a}"
            for cond, c in ctx.items():
                rows.append({"kind": "nonce", "lemma": lemma, "past": a, "slot": slot, "cond": cond,
                             "partner": partners[0][lemma], "context": c, "probe": probe, "prompt": f"{c} {probe}"})
    das = pd.read_csv(args.das_items)
    for r in das.itertuples():
        rows.append({"kind": "das", "lemma": r.verb, "past": "", "slot": -1, "cond": "", "partner": "", "context": "",
                     "probe": r.prompt, "prompt": r.prompt})
    P = pd.DataFrame(rows).drop_duplicates(["prompt", "kind", "cond", "lemma", "slot"]).reset_index(drop=True)
    P.insert(0, "pid", np.arange(len(P)))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    P.to_csv(out / "prompts.csv", index=False)
    ex = P[(P.kind == "nonce") & (P.lemma == lemmas[0]) & (P.slot == 0)].set_index("cond").prompt.to_dict()
    meta = {"lemmas": len(lemmas), "slots": N_SLOTS, "prompts": len(P),
            "by_cond": P[P.kind == "nonce"].cond.value_counts().to_dict(), "seed": args.seed, "example": ex}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lemmas", default="data/nonce_passive/lemmas.csv")
    ap.add_argument("--das-items", default="results/das_round2/final_strict/items.csv")
    ap.add_argument("--out-dir", default="data/round4/nonce_cues")
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
