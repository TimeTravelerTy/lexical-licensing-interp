#!/usr/bin/env python3
"""Compare candidate DAS training targets on the active readout (task B).

For each active item ("The AGENT VERB-ed", transitive vs intransitive verb of
a pair), compute the target score at the verb's last token for:

- `the_vs_dot`:   log P(" the") - log P(".")                        (current)
- `obj_vs_rest`:  log P(O) - log(1 - P(O)), O = object-start set
- `obj_vs_intr`:  log P(O) - log P(I), I = intransitive continuations
- `pron_vs_intr`: same with O restricted to accusative pronouns

O = the, a, an, his, her, their, its, this, some, him, them, it (not "that":
after say/complain-type verbs it starts a clause).
I = ".", ",", "\\n", " and", and the prepositions to, in, with, on, at, for,
from, as, into, over. Particles (up, out, off, down) are left out of both
sets: they follow transitive and intransitive verbs alike. Only tracked
tokens are available (no " me", " us" or adverbs), so masses are lower
bounds.

Also lists, for intransitive items whose object-start mass beats the
intransitive set, the most frequent top continuations per verb.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# " that" is left out: after say/complain-type verbs it introduces a clause.
OBJ = (" the", " a", " an", " his", " her", " their", " its", " this", " some",
       " him", " them", " it")
PRON = (" him", " them", " it", " her")
INTR = (".", ",", "\\n", " and", " to", " in", " with", " on", " at", " for", " from", " as",
        " into", " over")


def lse(r, toks):
    return np.logaddexp.reduce(np.column_stack([r[f"lp[{t}]"] for t in toks]), axis=1)


def targets(r):
    o, i, p = lse(r, OBJ), lse(r, INTR), lse(r, PRON)
    return pd.DataFrame({
        "the_vs_dot": r["lp[ the]"] - r["lp[.]"],
        "obj_vs_rest": o - np.log1p(-np.exp(np.minimum(o, -1e-9))),
        "obj_vs_intr": o - i,
        "pron_vs_intr": p - i,
        "p_obj": np.exp(o), "p_intr": np.exp(i),
    }, index=r.index)


def run(args):
    act = pd.read_json(args.actives, lines=True)
    ro = pd.read_csv(args.readout, low_memory=False)
    ro = ro.set_index("prompt_id")
    from build_das_prep_prompts import pid

    t = targets(ro)
    for side in ("trans", "intrans"):
        ids = act[f"{side}_prompt"].map(lambda p: pid("active", p))
        for c in t.columns:
            act[f"{side}_{c}"] = t.loc[ids, c].to_numpy()
        act[f"{side}_topk"] = ro.loc[ids, "topk"].to_numpy()
    rows = []
    for split, g in act.groupby("split"):
        for c in ("the_vs_dot", "obj_vs_rest", "obj_vs_intr", "pron_vs_intr"):
            tr, it = g[f"trans_{c}"], g[f"intrans_{c}"]
            # Pair-level separation: AUC of trans vs intrans scores within subject.
            rows.append({"split": split, "target": c, "trans_mean": tr.mean(), "intrans_mean": it.mean(),
                         "gap": (tr - it).mean(), "trans_pos": (tr > 0).mean(), "intrans_neg": (it < 0).mean(),
                         "both": ((tr > 0) & (it < 0)).mean(), "paired_order": (tr > it).mean()})
    summ = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out / "target_comparison.csv", index=False)

    # Intransitive items where object-start beats the intransitive set.
    bad = act[act.intrans_obj_vs_intr > 0]
    per = []
    for (split, verb), g in bad.groupby(["split", "intrans_verb"]):
        acc = {}
        for s in g.intrans_topk:
            for tok, lp in json.loads(s)[:5]:
                acc[tok] = acc.get(tok, 0) + np.exp(lp)
        top = sorted(acc.items(), key=lambda x: -x[1])[:6]
        n_all = (act.intrans_verb == verb).sum()
        per.append({"split": split, "intrans_verb": verb, "n_items": len(g), "of": n_all,
                    "top_continuations": ", ".join(f"{json.dumps(k)} {v / len(g):.2f}" for k, v in top)})
    per = pd.DataFrame(per).sort_values(["split", "n_items"], ascending=[True, False])
    per.to_csv(out / "intrans_object_like.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(summ.round(3).to_string(index=False))
    print(per.head(40).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--actives", default="data/passive_das/actives.jsonl")
    ap.add_argument("--readout", default="results/passive_das_prep/pythia14b_readout.csv")
    ap.add_argument("--out-dir", default="reports/passive_das_prep/task_b")
    run(ap.parse_args())
