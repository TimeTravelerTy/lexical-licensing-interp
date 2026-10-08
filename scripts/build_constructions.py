#!/usr/bin/env python3
"""Object relatives and tough constructions (round3_plan.md, C9). No model.

Items: the 64 primary pairs x 24 contexts per construction.
- OR: "The N that NAME <past>"   (12 nouns x {John, Mary}); past forms from verb_forms.csv.
- TC: "The N is ADJ to <base>"   (12 nouns x {easy, hard}).
The 12 nouns are the declared hand-picked list (`NOUNS`; 4 from contexts
curated for each verb band, which is not the noun's frequency). Plan (as `build_passive_test.py`): per item x split the
cross-fitted fold; T / I active donors from the fold's held-out DAS pairs
(one shared random subject per donor pair); `cswap` = the other verb of the
pair in the same construction and context. Both the bad and the good item are
bases.

Outputs (`--out-dir`): `items.csv`, `prompts.csv`, `plan.csv.gz`, `plan_meta.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_das_round2_prompts import SUBJECTS

AUX = dict(SUBJECTS)
SUBJ = [s for s, _ in SUBJECTS]
FRAMES = {"OR": ("John", "Mary"), "TC": ("easy", "hard")}
# Declared in round3_plan.md (Part C): hand-picked, 4 from contexts curated for each verb band.
# The band is the context's source verb band, not the noun's frequency.
NOUNS = (("house", "head"), ("letter", "head"), ("suspect", "head"), ("king", "head"),
         ("child", "tail"), ("horse", "tail"), ("statue", "tail"), ("thief", "tail"),
         ("baby", "xtail"), ("ship", "xtail"), ("suitcase", "xtail"), ("wallet", "xtail"))


def surface(cons, noun, slot, form):
    return f"The {noun} that {slot} {form}" if cons == "OR" else f"The {noun} is {slot} to {form}"


def build(args):
    rng = np.random.default_rng(args.seed)
    pit = pd.read_csv(args.passive_items)
    vf = pd.read_csv(args.verb_forms).set_index("lemma")
    pf = pd.read_csv(args.folds)
    nouns = list(NOUNS)
    have = set(pit.prefix.str.split().str[1])
    assert all(n in have for n, _ in nouns), "declared nouns must come from the curated contexts"
    pr = pit[pit.bad_class == "plain"].drop_duplicates("pair_id")
    prompts = {}

    def add(text, kind, lemma, subject="", cons=""):
        prompts.setdefault(text, {"prompt": text, "kind": kind, "lemma": lemma, "subject": subject,
                                  "construction": cons})
        return text

    das = pd.read_csv(args.das_items)
    for r in das.itertuples():
        add(r.prompt, "active", r.verb, r.subject)
    for p in pr.itertuples():  # actives of the primary verbs (projection reference)
        for s in SUBJ:
            add(f"{s} {AUX[s]} {p.good_part}", "active", p.good_lemma, s)
            add(f"{s} {AUX[s]} {p.bad_part}", "active", p.bad_lemma, s)
        for name in ("John", "Mary"):  # base-form reference for the TC gate
            add(f"{name} can {p.good_lemma}", "base_ref", p.good_lemma, name)
            add(f"{name} can {p.bad_lemma}", "base_ref", p.bad_lemma, name)
    items = []
    for cons, slots in FRAMES.items():
        for p in pr.itertuples():
            gf = vf.past[p.good_lemma] if cons == "OR" else p.good_lemma
            bf = vf.past[p.bad_lemma] if cons == "OR" else p.bad_lemma
            for noun, nband in nouns:
                for slot in slots:
                    cid = f"{cons}/{noun}/{slot}"
                    items.append({"item_id": f"{cons}|{p.pair_id}|{noun}|{slot}", "construction": cons,
                                  "pair_id": p.pair_id, "band": p.band, "source": p.source,
                                  "das_pair": p.das_pair if isinstance(p.das_pair, str) else "",
                                  "good_lemma": p.good_lemma, "bad_lemma": p.bad_lemma, "good_form": gf, "bad_form": bf,
                                  "good_part": p.good_part, "bad_part": p.bad_part, "context_id": cid,
                                  "context_band": nband, "noun": noun, "slot": slot,
                                  "good_prompt": add(surface(cons, noun, slot, gf), cons, p.good_lemma, "", cons),
                                  "bad_prompt": add(surface(cons, noun, slot, bf), cons, p.bad_lemma, "", cons),
                                  "bad_class": p.bad_class, "prep_subtype": p.prep_subtype,
                                  "bad_side_high": p.bad_side_high, "by_split": p.by_split, "primary": True})
    items = pd.DataFrame(items)
    n_folds = int(pf.fold_split0.max()) + 1
    verbs = {r.pair_id: (r.trans_part, r.intrans_part) for r in pf.itertuples()}
    plan = []
    for it in items.itertuples():
        for k in range(3):
            fold_of = dict(zip(pf.pair_id, pf[f"fold_split{k}"]))
            f = fold_of[it.das_pair] if it.das_pair else int(rng.integers(n_folds))
            held = sorted(q for q, ff in fold_of.items() if ff == f and q != it.das_pair)
            subj = {q: SUBJ[int(rng.integers(len(SUBJ)))] for q in held}
            for side, other in (("bad", "good"), ("good", "bad")):
                base = getattr(it, f"{side}_prompt")
                rows = []
                for q in held:
                    tp, ip = verbs[q]
                    s = subj[q]
                    rows.append(("T", f"{s} {AUX[s]} {tp}", 1, q))
                    rows.append(("I", f"{s} {AUX[s]} {ip}", 0, q))
                rows.append(("cswap", getattr(it, f"{other}_prompt"), 1 if other == "good" else 0, ""))
                for cond, donor, dcls, dpair in rows:
                    plan.append({"item_id": it.item_id, "split": k, "fold": f, "side": side, "base": base,
                                 "cond": cond, "donor": donor, "donor_cls": dcls, "donor_pair": dpair})
    P = pd.DataFrame(prompts.values())
    P.insert(0, "pid", np.arange(len(P)))
    idx = dict(zip(P.prompt, P.pid))
    plan = pd.DataFrame(plan)
    assert plan.donor.isin(idx).all()
    plan["base"], plan["donor"] = plan.base.map(idx), plan.donor.map(idx)
    plan.insert(0, "row", np.arange(len(plan)))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items.to_csv(out / "items.csv", index=False)
    P.to_csv(out / "prompts.csv", index=False)
    plan.to_csv(out / "plan.csv.gz", index=False)
    meta = {"items": len(items), "items_by_construction": items.construction.value_counts().to_dict(),
            "nouns": nouns, "prompts": len(P), "plan_rows": len(plan),
            "plan_rows_by_cond": plan.cond.value_counts().to_dict(), "seed": args.seed,
            "plan_sha256": hashlib.sha256(pd.util.hash_pandas_object(plan, index=False).values.tobytes()).hexdigest(),
            "examples": items.groupby("construction").head(1)[["good_prompt", "bad_prompt"]].values.tolist()}
    (out / "plan_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--passive-items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--verb-forms", default="data/das_round2/verb_forms.csv")
    ap.add_argument("--folds", default="results/das_round2/final_strict/pairs_folds.csv")
    ap.add_argument("--das-items", default="results/das_round2/final_strict/items.csv")
    ap.add_argument("--out-dir", default="data/constructions")
    ap.add_argument("--seed", type=int, default=17)
    build(ap.parse_args())
