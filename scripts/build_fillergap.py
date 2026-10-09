#!/usr/bin/env python3
"""Round 4, Part B: filler-gap items and patch plans (reports/round4/plan.md, B6 / B7). No model.

Modes
- items: from `verbs.csv` (verbs both raters accepted), every verb x context x frame:
    B6 (exp `emb`): "I MATRIX that NAME PAST" (frame `nofill`) / "I MATRIX what NAME PAST" (`fill`);
    B7 (exp `mat`): "AUX NAME BASE" (`nofill`) / "What aux NAME BASE" (`fill`);
  plus base-form references "NAME can BASE" (John, Mary) for the B7 gate and the DAS active items
  (z scale) -> `items.csv` (with the context half: 0 = A selects I verbs, 1 = B tests), `prompts.csv`,
  `meta.json`.
- plan: for the kept I verbs of one experiment (`--kept`), per item x split the cross-fitted fold
  (DAS verbs: the fold where their pair is held out; others one random fold, seed 17) and the T / I
  active donors of that fold's held-out DAS pairs (one shared random subject per donor pair), plus
  two in-range rows (targets t_T / t_I filled in by the runner) -> `plan_{exp}.csv.gz`,
  `plan_{exp}_meta.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_das_round2_prompts import SUBJECTS

AUXD = dict(SUBJECTS)
SUBJ = [s for s, _ in SUBJECTS]
MATRIX = ("know", "remember", "forgot", "heard")
NAMES = ("John", "Mary", "Tom", "Sarah", "Peter", "Anna")
AUXES = ("Did", "Will", "Can", "Would")


def surface(exp, frame, ctx_a, name, form):
    if exp == "emb":
        return f"I {ctx_a} {'what' if frame == 'fill' else 'that'} {name} {form}"
    return f"What {ctx_a.lower()} {name} {form}" if frame == "fill" else f"{ctx_a} {name} {form}"


def das_pair_of(pf):
    out = {}
    for r in pf.itertuples():
        out[r.trans], out[r.intrans] = r.pair_id, r.pair_id
    return out


def halves(seed):
    """Context -> half (0 = A, 1 = B): per matrix verb / auxiliary, 3 of the 6 names in each half."""
    rng = np.random.default_rng(seed)
    out = {}
    for exp, ctxs in (("emb", MATRIX), ("mat", AUXES)):
        for a in ctxs:
            perm = rng.permutation(len(NAMES))
            for j, name in enumerate(NAMES):
                out[f"{exp}/{a}/{name}"] = int(perm[j] >= len(NAMES) // 2)
    return out


def build_items(args):
    v = pd.read_csv(args.verbs, comment="#")
    pf = pd.read_csv(args.folds)
    dp = das_pair_of(pf)
    prompts = {}

    def add(text, kind, lemma, extra=None):
        prompts.setdefault(text, {"prompt": text, "kind": kind, "lemma": lemma, **(extra or {})})
        return text

    for r in pd.read_csv(args.das_items).itertuples():
        add(r.prompt, "das", r.verb)
    items = []
    for exp, ctxs, formcol in (("emb", MATRIX, "past"), ("mat", AUXES, "lemma")):
        for w in v.itertuples():
            form = getattr(w, formcol)
            for a in ctxs:
                for name in NAMES:
                    cid = f"{exp}/{a}/{name}"
                    for frame in ("nofill", "fill"):
                        p = add(surface(exp, frame, a, name, form), exp, w.lemma)
                        items.append({"item_id": f"{exp}|{frame}|{w.lemma}|{a}|{name}", "exp": exp, "frame": frame,
                                      "lemma": w.lemma, "cls": w.pool, "form": form, "context_id": cid,
                                      "das_pair": dp.get(w.lemma, ""), "prompt": p})
        if exp == "mat":
            for w in v.itertuples():
                for name in ("John", "Mary"):
                    add(f"{name} can {w.lemma}", "base_ref", w.lemma)
    items = pd.DataFrame(items)
    items["half"] = items.context_id.map(halves(args.seed))
    P = pd.DataFrame(prompts.values())
    P.insert(0, "pid", np.arange(len(P)))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    items["pid"] = items.prompt.map(dict(zip(P.prompt, P.pid)))
    items.to_csv(out / "items.csv", index=False)
    P.to_csv(out / "prompts.csv", index=False)
    meta = {"verbs": v.pool.value_counts().to_dict(), "items": len(items), "prompts": len(P),
            "items_by_exp": items.exp.value_counts().to_dict(), "contexts": {"emb": len(MATRIX) * len(NAMES),
                                                                            "mat": len(AUXES) * len(NAMES)},
            "examples": items.groupby(["exp", "frame", "cls"]).prompt.first().to_dict()}
    meta["examples"] = {"/".join(k): s for k, s in meta["examples"].items()}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


def build_plan(args):
    out = Path(args.out_dir)
    items = pd.read_csv(out / "items.csv")
    P = pd.read_csv(out / "prompts.csv")
    pf = pd.read_csv(args.folds)
    kept = set(pd.read_csv(args.kept).lemma)
    exp = args.exp
    it = items[(items.exp == exp) & (items.cls == "I") & items.lemma.isin(kept)]
    it = it.assign(das_pair=it.das_pair.fillna(""))
    rng = np.random.default_rng(args.seed)
    n_folds = int(pf.fold_split0.max()) + 1
    verbs = {r.pair_id: (r.trans_part, r.intrans_part) for r in pf.itertuples()}
    pid = dict(zip(P.prompt, P.pid))
    extra = {}
    plan = []
    draw = {}  # fold and donor subjects per (verb, context, split), shared by the two frames
    for row in it.sort_values(["lemma", "context_id", "frame"]).itertuples():
        for k in range(3):
            key = (row.lemma, row.context_id, k)
            if key not in draw:
                fold_of = dict(zip(pf.pair_id, pf[f"fold_split{k}"]))
                f = fold_of[row.das_pair] if row.das_pair else int(rng.integers(n_folds))
                held = sorted(q for q, ff in fold_of.items() if ff == f and q != row.das_pair)
                draw[key] = (f, held, {q: SUBJ[int(rng.integers(len(SUBJ)))] for q in held})
            f, held, subj = draw[key]
            for q in held:
                tp, ip = verbs[q]
                s = subj[q]
                for cond, part in (("T", tp), ("I", ip)):
                    d = f"{s} {AUXD[s]} {part}"
                    if d not in pid:
                        extra.setdefault(d, len(P) + len(extra))
                    plan.append({"item_id": row.item_id, "split": k, "fold": f, "cond": cond,
                                 "base": row.pid, "donor": pid.get(d, extra.get(d)), "donor_pair": q})
            for cond in ("inT", "inI"):
                plan.append({"item_id": row.item_id, "split": k, "fold": f, "cond": cond, "base": row.pid,
                             "donor": -1, "donor_pair": ""})
    assert not extra, f"{len(extra)} donor prompts are not DAS items: {list(extra)[:3]}"
    plan = pd.DataFrame(plan)
    plan.insert(0, "row", np.arange(len(plan)))
    plan.to_csv(out / f"plan_{exp}.csv.gz", index=False)
    meta = {"exp": exp, "kept_I_verbs": sorted(kept & set(it.lemma)), "items": int(it.item_id.nunique()),
            "plan_rows": len(plan), "rows_by_cond": plan.cond.value_counts().to_dict(), "seed": args.seed,
            "plan_sha256": hashlib.sha256(pd.util.hash_pandas_object(plan, index=False).values.tobytes()).hexdigest()}
    (out / f"plan_{exp}_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "kept_I_verbs"}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("items", "plan"))
    ap.add_argument("--verbs", default="data/round4/fillergap/verbs.csv")
    ap.add_argument("--folds", default="results/das_round2/final_strict/pairs_folds.csv")
    ap.add_argument("--das-items", default="results/das_round2/final_strict/items.csv")
    ap.add_argument("--out-dir", default="data/round4/fillergap")
    ap.add_argument("--exp", choices=("emb", "mat"))
    ap.add_argument("--kept", help="csv with column lemma: the kept I verbs of this experiment")
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    build_items(a) if a.mode == "items" else build_plan(a)
