#!/usr/bin/env python3
"""Build the passive test of the round-2 direction (no model; spec: passive_test_plan.md).

Outputs in `--out-dir` (default `data/das_round2/passive_test/`):
- `items.csv`: one row per (verb pair, context). The 126 original pairs come
  from `data/passive_das/passives.jsonl`; the 19 new eval pairs get all 126
  curated contexts. Each row has the bad verb's class, its groups, and the
  DAS training pair if the pair is one (`das_pair`).
- `prompts.csv`: every prompt run unpatched (passives, and actives
  "<Subj> has/have <participle>" for DAS, eval and participle-matched verbs).
- `plan.csv.gz` (not committed; SHA-256 in `plan_meta.json`): every patch.
  Per item x split: the basis (cross-fitted fold), and for both the good and
  the bad base the conditions T / I (held-out donor pairs of that fold,
  one shared random subject per donor pair), `same_verb`, `passive_swap`;
  and for the bad base `matched_T` / `matched_I` (participle-matched verbs
  not in the basis's training pairs).

Fold assignments are read from the frozen run (`pairs_folds.csv`); they are
deterministic in the split seed, so every site uses the same ones.
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


def active(subj, part):
    return f"{subj} {AUX[subj]} {part}"


def load_items(args):
    pas = pd.read_json(args.passives, lines=True, dtype={"item_id": str})
    pas["prefix"] = [g[: len(g) - len(v)] for g, v in zip(pas.good_prompt, pas.good_verb)]
    assert all(b == p + v for b, p, v in zip(pas.bad_prompt, pas.prefix, pas.bad_verb))
    lem = pas.verb_pair.str.split("/")
    orig = pd.DataFrame({
        "item_id": pas.item_id, "pair_id": pas.verb_pair, "source": "original", "band": pas.verb_band,
        "good_lemma": lem.str[1], "bad_lemma": lem.str[2], "good_part": pas.good_verb, "bad_part": pas.bad_verb,
        "context_id": pas.context_id, "context_band": pas.context_band, "prefix": pas.prefix,
        "by_split": pas.split})
    ctx = orig.drop_duplicates("context_id")[["context_id", "context_band", "prefix"]]
    ev = pd.read_csv(args.eval_pairs)
    rows = []
    for p in ev.itertuples():
        pair_id = f"new/{p.band}/{p.trans_lemma}/{p.intrans_lemma}"
        for c in ctx.itertuples():
            rows.append({"item_id": f"{pair_id}|{c.context_id}", "pair_id": pair_id, "source": "new",
                         "band": p.band, "good_lemma": p.trans_lemma, "bad_lemma": p.intrans_lemma,
                         "good_part": p.trans_participle, "bad_part": p.intrans_participle,
                         "context_id": c.context_id, "context_band": c.context_band, "prefix": c.prefix,
                         "by_split": "new"})
    items = pd.concat([orig, pd.DataFrame(rows)], ignore_index=True)
    cls = pd.read_csv(args.classes)
    bsh = pd.read_csv(args.groups).set_index(["lemma", "source"]).bad_side_high
    key = list(zip(items.bad_lemma, items.source))
    c = cls.set_index(["lemma", "source"])
    items["bad_class"] = [c["class"].get(k) for k in key]
    items["prep_subtype"] = [c.prep_subtype.get(k) for k in key]
    items["bad_side_high"] = [bool(bsh.get(k, False)) for k in key]
    if items.bad_class.isna().any():
        raise SystemExit(f"unclassified bad verbs: {sorted(items[items.bad_class.isna()].bad_lemma.unique())}")
    tp = pd.read_csv(args.folds)
    das = {(t, i): pid for t, i, pid in zip(tp.trans, tp.intrans, tp.pair_id)}
    items["das_pair"] = [das.get((g, b), "") for g, b in zip(items.good_lemma, items.bad_lemma)]
    items["primary"] = items.bad_class == "plain"
    items["good_prompt"] = items.prefix + items.good_part
    items["bad_prompt"] = items.prefix + items.bad_part
    return items, tp


def build(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items, tp = load_items(args)
    forms = pd.read_csv(args.verb_forms).set_index("lemma").participle
    matched = pd.read_csv(args.matched)
    matched["pair_id"] = matched.trans + "/" + matched.intrans

    # ---- prompts
    das_items = pd.read_csv(args.das_items)
    prompts = {}

    def add(text, kind, lemma, subject="", cls=-1):
        if text not in prompts:
            prompts[text] = {"prompt": text, "kind": kind, "lemma": lemma, "subject": subject, "cls": cls}
        return text

    for r in das_items.itertuples():
        add(r.prompt, "active", r.verb, r.subject, int(r.cls))
    for side, cls in (("good", 1), ("bad", 0)):
        for lemma, part in items[[f"{side}_lemma", f"{side}_part"]].drop_duplicates().itertuples(index=False):
            for s in SUBJ:
                add(active(s, part), "active", lemma, s, cls)
        for lemma, text in items[[f"{side}_lemma", f"{side}_prompt"]].drop_duplicates().itertuples(index=False):
            add(text, "passive", lemma, "", cls)
    for side, cls in (("trans", 1), ("intrans", 0)):
        for lemma in matched[side]:
            for s in SUBJ:
                add(active(s, forms[lemma]), "active", lemma, s, cls)
    P = pd.DataFrame(prompts.values())
    P.insert(0, "pid", np.arange(len(P)))
    idx = dict(zip(P.prompt, P.pid))
    das_part = {}
    for r in tp.itertuples():
        das_part[r.trans], das_part[r.intrans] = r.trans_part, r.intrans_part

    # ---- plan
    rng = np.random.default_rng(args.seed)
    n_folds = int(tp.fold_split0.max()) + 1
    pair_verbs = {r.pair_id: (r.trans, r.intrans) for r in tp.itertuples()}
    plan = []
    for it in items.itertuples():
        for k in range(3):
            fold_of = dict(zip(tp.pair_id, tp[f"fold_split{k}"]))
            f = fold_of[it.das_pair] if it.das_pair else int(rng.integers(n_folds))
            held = sorted(p for p, ff in fold_of.items() if ff == f and p != it.das_pair)
            train_verbs = {v for p, ff in fold_of.items() if ff != f for v in pair_verbs[p]}
            base = {"item_id": it.item_id, "split": k, "fold": f}
            subj = {p: SUBJ[int(rng.integers(len(SUBJ)))] for p in held}
            mpairs = list(matched.itertuples())
            msubj = {m.pair_id: SUBJ[int(rng.integers(len(SUBJ)))] for m in mpairs}
            own_subj = SUBJ[int(rng.integers(len(SUBJ)))]
            for side, other in (("good", "bad"), ("bad", "good")):
                b = idx[getattr(it, f"{side}_prompt")]
                rows = []
                for p in held:
                    t, i = pair_verbs[p]
                    rows.append(("T", idx[active(subj[p], das_part[t])], 1, p))
                    rows.append(("I", idx[active(subj[p], das_part[i])], 0, p))
                rows.append(("same_verb", idx[active(own_subj, getattr(it, f"{side}_part"))], 1 if side == "good" else 0, ""))
                rows.append(("passive_swap", idx[getattr(it, f"{other}_prompt")], 1 if other == "good" else 0, ""))
                if side == "bad":
                    for m in mpairs:
                        for v, c, cond in ((m.trans, 1, "matched_T"), (m.intrans, 0, "matched_I")):
                            if v not in train_verbs:
                                rows.append((cond, idx[active(msubj[m.pair_id], forms[v])], c, m.pair_id))
                for cond, donor, dcls, dpair in rows:
                    plan.append({**base, "side": side, "base": b, "cond": cond, "donor": donor, "donor_cls": dcls,
                                 "donor_pair": dpair})
    plan = pd.DataFrame(plan)
    plan.insert(0, "row", np.arange(len(plan)))

    items.to_csv(out / "items.csv", index=False)
    P.to_csv(out / "prompts.csv", index=False)
    plan.to_csv(out / "plan.csv.gz", index=False)
    meta = {"items": len(items), "pairs": int(items.pair_id.nunique()), "prompts": len(P),
            "plan_rows": len(plan), "plan_rows_by_cond": plan.cond.value_counts().to_dict(),
            "primary_pairs_by_band": items[items.primary].groupby("band").pair_id.nunique().to_dict(),
            "seed": args.seed, "folds_sha256": hashlib.sha256(Path(args.folds).read_bytes()).hexdigest(),
            "plan_sha256": hashlib.sha256(pd.util.hash_pandas_object(plan, index=False).values.tobytes()).hexdigest()}
    (out / "plan_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--passives", default="data/passive_das/passives.jsonl")
    ap.add_argument("--eval-pairs", default="data/verb_expansion/eval_pairs.csv")
    ap.add_argument("--classes", default="data/das_round2/intrans_classes_final.csv")
    ap.add_argument("--groups", default="data/das_round2/projection_groups.csv")
    ap.add_argument("--verb-forms", default="data/das_round2/verb_forms.csv")
    ap.add_argument("--matched", default="data/das_round2/train_pairs_participle_matched.csv")
    ap.add_argument("--das-items", default="results/das_round2/final_strict/items.csv")
    ap.add_argument("--folds", default="results/das_round2/final_strict/pairs_folds.csv")
    ap.add_argument("--out-dir", default="data/das_round2/passive_test")
    ap.add_argument("--seed", type=int, default=17)
    build(ap.parse_args())
