#!/usr/bin/env python3
"""Turn the checked verb candidates into simple sets for DAS.

Inputs: `work/round*_checked.csv` (from `check_verb_candidates.py`) and
`work/codex_review.csv` (Codex adversarial review: keep / doubtful / drop).

A verb is **included** if it passed the mechanical checks and the review
kept it, unless overridden in OVERRIDES (my own judgement). Everything else
is excluded with one short reason. Tags:
- `head` / `tail` / `xtail`: FreqBLiMP participle-Zipf bands (eval sets).
- `near_head`: participle Zipf in (3.2, 3.5), or summed-lemma Zipf >= 3.5
  with the participle below Head. DAS training only; never in band
  comparisons.

Outputs (in `data/verb_expansion/`):
- `verbs.csv`: every candidate with tag, frequencies, tokens, included, reason.
- `eval_pairs.csv`: new good/bad passive pairs within participle band and
  token count (single / multi), |lemma Zipf gap| <= 0.25 and |participle
  Zipf gap| <= 0.35 as in `build_diverse_passives.py`. Separate from the 126
  original pairs.
- `das_train_pairs.csv`: transitive / intransitive pairs from `head` and
  `near_head` verbs not used in eval pairs, same participle token count,
  summed-lemma Zipf within 0.25.
- `summary.md`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_TAGS = ("head", "tail", "xtail")
# lemma -> (include, reason): my calls on intransitives where I depart from
# the review verdict. Transitives follow the review (keep only).
# A prepositional complement (rely on, tamper with) does not disqualify an
# intransitive here: the bare passive is still bad and the active takes no
# NP object (the original 126 include listen, complain, object). Obsolete,
# technical or idiomatic senses (ensue = pursue, perish the thought) do not
# disqualify either.
_PREP_OR_RARE = """abound accede ache acquiesce allude assent bicker care chat cohabit coincide collaborate
comply converse cringe dally defect depend differ disagree dissent dwell encroach ensue equivocate excel
feud flinch fraternize frown gaze glance grovel hesitate hobnob impinge intercede intervene inveigh loom
luxuriate meditate menstruate overreact peek peer perish persist pontificate preside prevail prevaricate
quarrel react rebel reign relapse relent rely reside resort saunter sneeze sprint stare step strive stumble
subscribe succumb tamper tower triumph waver yearn""".split()
_EXCLUDE = {
    **{v: "main complement is a clause (it was hoped / insisted that)" for v in
       "comment contend hope insist remark speculate wonder".split()},
    **{v: "takes speech objects, like the dropped gasp / groan" for v in ("exclaim", "giggle")},
    **{v: "relational or stative verb; odd with animate subjects" for v in ("account", "amount", "consist")},
    "behave": "common reflexive object (behave yourself)",
    "look": "copular use (look tired)",
    **{v: "summed-lemma Zipf inflated by a noun/adjective homograph; distorts matching" for v in
       "consent deliberate long major moonlight party vacation".split()},
}
OVERRIDES: dict[str, tuple[bool, str]] = {
    **{v: (True, "prepositional complement or rare sense only") for v in _PREP_OR_RARE},
    **{v: (False, why) for v, why in _EXCLUDE.items()},
}


def greedy(cands):
    out, used_a, used_b = [], set(), set()
    for gap, i, j in sorted(cands):
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            out.append((gap, i, j))
    return out


def match(df, group_cols, gap_fn, ok_fn):
    rows = []
    for key, g in df.groupby(group_cols):
        good = g[g["class"] == "trans"].to_dict("records")
        bad = g[g["class"] == "intrans"].to_dict("records")
        cands = [(gap_fn(a, b), i, j) for i, a in enumerate(good) for j, b in enumerate(bad) if ok_fn(a, b)]
        for gap, i, j in greedy(cands):
            a, b = good[i], bad[j]
            row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
            row.update({"trans_lemma": a["lemma"], "trans_participle": a["participle"],
                        "intrans_lemma": b["lemma"], "intrans_participle": b["participle"],
                        "trans_tag": a["tag"], "intrans_tag": b["tag"],
                        "trans_participle_zipf": a["participle_zipf"], "intrans_participle_zipf": b["participle_zipf"],
                        "trans_lemma_zipf": a["lemma_zipf"], "intrans_lemma_zipf": b["lemma_zipf"],
                        "trans_lemma_sum_zipf": a["lemma_sum_zipf"], "intrans_lemma_sum_zipf": b["lemma_sum_zipf"],
                        "trans_ntok": a["participle_ntok"], "intrans_ntok": b["participle_ntok"],
                        "gap": round(gap, 3)})
            rows.append(row)
    return pd.DataFrame(rows)


def describe(x):
    x = pd.Series(x, dtype=float)
    return f"{x.mean():.2f} ({x.std():.2f}), range {x.min():.2f}-{x.max():.2f}"


def run(args):
    work, out = Path(args.work_dir), Path(args.out_dir)
    df = pd.concat([pd.read_csv(p, dtype={"round": str}) for p in sorted(work.glob("round*_checked.csv"))],
                   ignore_index=True)
    df["reasons"] = df.reasons.fillna("")
    df = df[df.reasons != "dup_candidate"].copy()
    rv = pd.read_csv(work / "codex_review.csv", dtype=str)
    df = df.merge(rv[["lemma", "class", "verdict", "reason"]].rename(columns={"reason": "review_reason"}),
                  on=["lemma", "class"], how="left")
    df["verdict"] = df.verdict.fillna("unreviewed")
    df["tok"] = np.where(df.participle_ntok > 1, "multi", "single")
    df["included"] = (df.decision == "accept") & (df.verdict == "keep")
    df["reason"] = np.where(df.decision != "accept", df.reasons,
                            np.where(df.verdict == "keep", "", df.verdict + ": " + df.review_reason.fillna("")))
    for lemma, (inc, why) in OVERRIDES.items():
        m = (df.lemma == lemma) & (df["class"] == "intrans") & (df.decision == "accept")
        df.loc[m, "included"] = inc
        df.loc[m, "reason"] = ("" if inc else "override: ") + why

    inc = df[df.included]
    ev = match(inc[inc.tag.isin(EVAL_TAGS)], ["band", "tok"],
               lambda a, b: abs(a["lemma_zipf"] - b["lemma_zipf"]) + abs(a["participle_zipf"] - b["participle_zipf"]),
               lambda a, b: abs(a["lemma_zipf"] - b["lemma_zipf"]) <= 0.25
               and abs(a["participle_zipf"] - b["participle_zipf"]) <= 0.35)
    in_eval = set(ev.trans_lemma) | set(ev.intrans_lemma) if len(ev) else set()
    pool = inc[inc.tag.isin(("head", "near_head")) & ~inc.lemma.isin(in_eval)]
    tr = match(pool, ["tok"], lambda a, b: abs(a["lemma_sum_zipf"] - b["lemma_sum_zipf"]),
               lambda a, b: abs(a["lemma_sum_zipf"] - b["lemma_sum_zipf"]) <= args.train_caliper)
    in_train = set(tr.trans_lemma) | set(tr.intrans_lemma) if len(tr) else set()
    df["use"] = np.select([df.lemma.isin(in_eval), df.lemma.isin(in_train), df.included],
                          ["eval", "das_train", "unpaired"], "excluded")

    cols = ["lemma", "class", "past", "participle", "tag", "band", "participle_zipf", "lemma_zipf",
            "lemma_sum_zipf", "participle_ntok", "participle_tokens", "past_ntok", "included", "use", "reason",
            "justification", "round"]
    df[cols].sort_values(["class", "tag", "lemma"]).to_csv(out / "verbs.csv", index=False)
    ev.to_csv(out / "eval_pairs.csv", index=False)
    tr.to_csv(out / "das_train_pairs.csv", index=False)

    cnt = df[df.included].groupby(["tag", "tok", "class"]).size().unstack(fill_value=0) \
        .reindex(columns=["trans", "intrans"], fill_value=0)
    lines = ["# Verb-set expansion", "",
             f"{len(df)} candidates from Codex over {df['round'].nunique()} generate-check rounds; "
             f"{int(df.included.sum())} included. A verb is included if it passed the mechanical checks "
             "(participle-Zipf band, Pythia tokens, no duplicate of the 126 original pairs) and Codex's "
             "adversarial review kept it" + (f"; for intransitives I override the review in {len(OVERRIDES)} cases (see OVERRIDES in "
             "`scripts/finalize_verb_expansion.py`)" if OVERRIDES else "")
             + ". Everything is still pending your spot-check.", "",
             "Tags: `head` / `tail` / `xtail` are FreqBLiMP participle-Zipf bands (used for eval). `near_head` "
             "(participle Zipf 3.2-3.5, or summed-lemma Zipf >= 3.5 with the participle below Head) is for DAS "
             "training only and is never used in band comparisons.", "",
             "## Included verbs", "", "| Tag | Participle tokens | Transitive | Intransitive |", "|---|---|---:|---:|"]
    lines += [f"| {t} | {k} | {r.trans} | {r.intrans} |" for (t, k), r in cnt.iterrows()]
    lines += ["", f"## Eval pairs (`eval_pairs.csv`): {len(ev)}", ""]
    if len(ev):
        lines += ["| Band | Tokens | Pairs |", "|---|---|---:|"] + \
                 [f"| {b} | {k} | {n} |" for (b, k), n in ev.groupby(["band", "tok"]).size().items()]
    lines += ["", f"## DAS training pairs (`das_train_pairs.csv`): {len(tr)}", "",
              "`head` + `near_head` verbs not used in eval pairs; each intransitive matched to one transitive "
              f"with the same participle token count and summed-lemma Zipf within {args.train_caliper}.", ""]
    if len(tr):
        lines += ["| Tokens | Pairs | Summed-lemma Zipf, trans | Intrans | Participle Zipf, trans | Intrans |",
                  "|---|---:|---|---|---|---|"]
        for k, g in list(tr.groupby("tok")) + [("all", tr)]:
            lines.append(f"| {k} | {len(g)} | {describe(g.trans_lemma_sum_zipf)} | {describe(g.intrans_lemma_sum_zipf)}"
                         f" | {describe(g.trans_participle_zipf)} | {describe(g.intrans_participle_zipf)} |")
        d = tr.trans_lemma_sum_zipf - tr.intrans_lemma_sum_zipf
        lines += ["", f"Summed-lemma Zipf gap (trans - intrans): mean {d.mean():+.3f}, max |gap| {d.abs().max():.2f}. "
                  f"Tags: transitive {tr.trans_tag.value_counts().to_dict()}, intransitive "
                  f"{tr.intrans_tag.value_counts().to_dict()}. Participle Zipf is not matched (only summed-lemma "
                  "Zipf and token count)."]
    unp = df[df.use == "unpaired"]
    src = df[~df.included].verdict.where(df.decision == "accept", "mechanical")
    lines += ["", f"Included but unpaired: {len(unp)} ({unp['class'].value_counts().to_dict()}).", "",
              "## Excluded", "", f"{int((~df.included).sum())} verbs; one-line reasons in `verbs.csv`. By source: "
              + ", ".join(f"{k} {v}" for k, v in src.value_counts().items()) + "."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", default="data/verb_expansion/work")
    ap.add_argument("--out-dir", default="data/verb_expansion")
    ap.add_argument("--train-caliper", type=float, default=0.25)
    run(ap.parse_args())
