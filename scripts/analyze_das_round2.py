#!/usr/bin/env python3
"""DAS round 2 (no training): active-frame readout, behaviour filter,
familiarity-matched subset, token-count split.

Target (option 3): M = log P(O) - log P(I) at the verb's last token; a
transitive item is correct if M > 0, an intransitive one if M < 0. " that",
" by" and particles (up, out, off, down, away, back) are in neither set.

Training pool: included expansion verbs tagged `head` / `near_head` (not in
eval pairs), intransitives restricted to class `plain` (prep_object and
contaminated verbs excluded), rematched one-to-one with transitives on
participle token count and summed-lemma Zipf (<= 0.25). The plain-intransitive
pairs of the original Head band are analysed alongside as `orig_head`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_das_prep_prompts import pid
from build_das_round2_prompts import SUBJECTS
from finalize_verb_expansion import greedy

O_ROUND1 = (" the", " a", " an", " his", " her", " their", " its", " this", " some", " him", " them", " it")
I_ROUND1 = (".", ",", "\\n", " and", " to", " in", " with", " on", " at", " for", " from", " as", " into", " over")
O_SET = O_ROUND1 + (" my", " our", " your", " these", " those", " every", " each", " several", " me", " us",
                    " himself", " herself", " themselves", " itself")
I_SET = I_ROUND1 + (" about", " of", " upon", " against", " through", " after", " under", " around", " across",
                    " toward", " here", " there", " again", " already", " never", " not", " so", " just",
                    " well", " but", " or", " because", " when", " while", " since", " before", " until",
                    " like", " without", " during", " behind", " near", " onto", " together", " alone", " too",
                    "!", "?", ";", ":")


def lse(r, toks):
    return np.logaddexp.reduce(np.column_stack([r[f"lp[{t}]"] for t in toks]), axis=1)


def load(args):
    ro = pd.read_csv(args.readout, low_memory=False).set_index("prompt_id")
    meta = pd.read_json(args.prompts, lines=True).set_index("prompt_id")
    ro = ro.join(meta[["set", "lemma", "subject"]])
    ro["M"] = lse(ro, O_SET) - lse(ro, I_SET)
    ro["M_r1"] = lse(ro, O_ROUND1) - lse(ro, I_ROUND1)
    ro["coverage"] = np.exp(np.logaddexp(lse(ro, O_SET), lse(ro, I_SET)))
    return ro


def build_pool(verbs, classes, args):
    inc = verbs[verbs.included & verbs.tag.isin(["head", "near_head"]) & (verbs.use != "eval")].copy()
    cls = classes.set_index("lemma")["class"]
    inc["iclass"] = np.where(inc["class"] == "trans", "trans", inc.lemma.map(cls).fillna("plain"))
    inc = inc[inc.iclass.isin(["trans", "plain"])]
    inc["tok"] = np.where(inc.participle_ntok > 1, "multi", "single")
    rows = []
    for tok, g in inc.groupby("tok"):
        good, bad = g[g.iclass == "trans"].to_dict("records"), g[g.iclass == "plain"].to_dict("records")
        cands = [(abs(a["lemma_sum_zipf"] - b["lemma_sum_zipf"]), i, j) for i, a in enumerate(good)
                 for j, b in enumerate(bad) if abs(a["lemma_sum_zipf"] - b["lemma_sum_zipf"]) <= args.caliper]
        for gap, i, j in greedy(cands):
            a, b = good[i], bad[j]
            rows.append({"source": "expansion", "tok": tok, "trans": a["lemma"], "intrans": b["lemma"],
                         "trans_tag": a["tag"], "intrans_tag": b["tag"],
                         "trans_part": a["participle"], "intrans_part": b["participle"],
                         "trans_part_zipf": a["participle_zipf"], "intrans_part_zipf": b["participle_zipf"],
                         "trans_sum_zipf": a["lemma_sum_zipf"], "intrans_sum_zipf": b["lemma_sum_zipf"],
                         "trans_ntok": a["participle_ntok"], "intrans_ntok": b["participle_ntok"]})
    return pd.DataFrame(rows)


def orig_head(classes, v2_path):
    cls = classes.set_index("lemma")["class"]
    v2 = pd.DataFrame([json.loads(l) for l in Path(v2_path).open(encoding="utf-8")])
    h = v2[(v2.band == "head") & (v2.paradigm == "passive_1") & (v2.frame_id == "was")]
    h = h[h.bad_lemma.map(cls).eq("plain") & (h.good_lemma != "bet")]
    from wordfreq import zipf_frequency
    from check_verb_candidates import lemma_sum_zipf
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b", local_files_only=True)
    ntok = lambda w: len(tok.encode(" " + w, add_special_tokens=False))
    rows = []
    for r in h.itertuples():
        rows.append({"source": "orig_head", "tok": "single" if max(ntok(r.good_verb), ntok(r.bad_verb)) == 1
                     else "multi", "trans": r.good_lemma, "intrans": r.bad_lemma, "trans_tag": "head",
                     "intrans_tag": "head", "trans_part": r.good_verb, "intrans_part": r.bad_verb,
                     "trans_part_zipf": zipf_frequency(r.good_verb, "en"),
                     "intrans_part_zipf": zipf_frequency(r.bad_verb, "en"),
                     "trans_sum_zipf": lemma_sum_zipf(r.good_lemma, r.good_verb),
                     "intrans_sum_zipf": lemma_sum_zipf(r.bad_lemma, r.bad_verb),
                     "trans_ntok": ntok(r.good_verb), "intrans_ntok": ntok(r.bad_verb)})
    return pd.DataFrame(rows)


def items(pairs, ro):
    """One row per (pair, side, frame/subject) with M."""
    out = []
    for p in pairs.itertuples():
        for side in ("trans", "intrans"):
            lemma = getattr(p, side)
            g = ro[ro.lemma == lemma]
            for r in g.itertuples():
                out.append({"source": p.source, "tok": p.tok, "band": getattr(p, f"{side}_tag"),
                            "pair": f"{p.trans}/{p.intrans}", "side": side, "lemma": lemma, "set": r.set,
                            "subject": r.subject, "M": r.M, "M_r1": r.M_r1, "coverage": r.coverage})
    it = pd.DataFrame(out)
    it["correct"] = np.where(it.side == "trans", it.M > 0, it.M < 0)
    it["correct_r1"] = np.where(it.side == "trans", it.M_r1 > 0, it.M_r1 < 0)
    return it


def pct(x):
    return f"{100 * np.mean(x):.0f}%"


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
        ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def run(args):
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    verbs = pd.read_csv(args.verbs)
    classes = pd.read_csv(args.classes)
    ro = load(args)
    pool = build_pool(verbs, classes, args)
    orig = orig_head(classes, args.v2_pairs)
    pairs = pd.concat([pool, orig], ignore_index=True)
    pairs["part_gap"] = (pairs.trans_part_zipf - pairs.intrans_part_zipf).round(2)
    pairs["sum_gap"] = (pairs.trans_sum_zipf - pairs.intrans_sum_zipf).round(2)
    pairs.to_csv(Path(args.data_dir) / "train_pairs.csv", index=False)
    it = items(pairs, ro)
    it.to_csv(Path(args.data_dir) / "train_items_readout.csv", index=False)
    she = it[(it.set == "perfect") & (it.subject == "She")]
    perf = it[it.set == "perfect"]
    old = it[it.set == "active"]
    L = ["# DAS round 2: active-frame readout, behaviour filter, matching", "",
         f"Readout: TSUBAME job {args.job} (Pythia-1.4B, bf16), `results/das_round2/pythia14b_readout.csv`. "
         "Target (option 3): M = log P(O) - log P(I) at the verb's last token. O = the, a, an, his, her, "
         "their, its, my, our, your, this, these, those, some, every, each, several, him, them, it, me, us, "
         "himself, herself, themselves, itself. I = . , newline, and, prepositions (to, in, with, on, at, "
         "for, from, as, into, over, about, of, upon, against, through, after, under, around, across, toward, "
         "behind, near, onto, during, without, like, until, before, since), adverbs and connectives (here, "
         "there, again, already, never, not, so, just, well, together, alone, too, but, or, because, when, "
         "while), ! ? ; :. In neither: \" that\", \" by\", particles (up, out, off, down, away, back). "
         "`r1 sets` = the round-1 sets (12 O / 14 I tokens) for comparison.", ""]

    # ---- task 1
    L += ["## 1. Readout on the new frame", "",
          f"Training pool: {len(pool)} expansion pairs (plain intransitives only, rematched) plus "
          f"{len(orig)} original Head pairs with a plain intransitive (`orig_head`, optional). New frame: "
          "\"She has <participle>\" (one item per verb) and the same with 7 subjects pooled (She, He, They, "
          "We, I, Maria, David). Old frame: \"The AGENT <past>\" (90 subjects). Accuracy = % items on the "
          "correct side of 0.", ""]
    rows = []
    for (src, band, tok), g in it.groupby(["source", "band", "tok"]):
        for side in ("trans", "intrans"):
            s = g[g.side == side]
            if not len(s):
                continue
            a, b, c = s[(s.set == "perfect") & (s.subject == "She")], s[s.set == "perfect"], s[s.set == "active"]
            rows.append([src, band, tok, side, s.lemma.nunique(), pct(a.correct), pct(b.correct), pct(c.correct),
                         pct(b.correct_r1), pct(c.correct_r1), f"{b.coverage.median():.2f}",
                         f"{c.coverage.median():.2f}"])
    L += table(["Source", "Band", "Tokens", "Class", "Verbs", "She has", "7 subj (new)", "Old frame",
                "7 subj, r1 sets", "Old, r1 sets", "Coverage new", "Coverage old"], rows)
    # pair-level ordering
    def order(df):
        w = df.pivot_table(index=["pair", "subject"], columns="side", values="M")
        return pct((w.trans > w.intrans).dropna())
    L += ["", f"Pair-level ordering (M_trans > M_intrans, same subject): new frame {order(perf)}, "
          f"old frame {order(old)}. Coverage = median P(O) + P(I).", ""]

    # ---- task 2: behaviour filter
    perf = perf.copy()
    verb = perf.groupby(["source", "band", "tok", "side", "lemma", "pair"]).agg(
        pass_rate=("correct", "mean"), n=("correct", "size")).reset_index()
    she_m = she.set_index("lemma").M
    verb["M_she"] = verb.lemma.map(she_m)
    verb["verb_pass"] = verb.pass_rate >= 0.5
    perf.to_csv(Path(args.data_dir) / "behavior_filter_items.csv", index=False)
    verb.to_csv(Path(args.data_dir) / "behavior_filter_verbs.csv", index=False)
    L += ["## 2. Behaviour filter (new frame, 7 subjects)", "",
          "Item = (verb, subject). An item passes if M is on its class's side of 0. A verb passes if at least "
          "half its 7 items pass. Rejects are kept (`behavior_filter_verbs.csv`).", ""]
    rows = []
    for (src, band, tok, side), g in verb.groupby(["source", "band", "tok", "side"]):
        gi = perf[(perf.source == src) & (perf.band == band) & (perf.tok == tok) & (perf.side == side)]
        rows.append([src, band, tok, side, len(g), pct(gi.correct), f"{int(g.verb_pass.sum())}/{len(g)}"])
    L += table(["Source", "Band", "Tokens", "Class", "Verbs", "Item pass", "Verbs passing"], rows)
    pp = pairs.copy()
    vp = verb.set_index("lemma").verb_pass
    pp["both_pass"] = pp.trans.map(vp) & pp.intrans.map(vp)
    L += ["", "Class balance after the verb-level filter:", ""]
    rows = []
    for src, g in pp.groupby("source"):
        rows.append([src, len(g), int(g.trans.map(vp).sum()), int(g.intrans.map(vp).sum()), int(g.both_pass.sum())])
    L += table(["Source", "Pairs", "Transitives passing", "Intransitives passing", "Pairs with both passing"], rows)
    rej = verb[~verb.verb_pass].sort_values(["side", "M_she"])
    L += ["", "Rejected verbs (She-has M; for intransitives, M > 0 means the model treats the verb as "
          "transitive in this frame):", ""]
    for side in ("intrans", "trans"):
        r = rej[rej.side == side]
        L.append(f"- {side}: " + (", ".join(f"{x.lemma} ({x.M_she:+.1f}, {x.pass_rate:.0%})" for x in r.itertuples())
                                  or "none"))
    # top continuations for rejected intransitives
    top = []
    for x in rej[rej.side == "intrans"].itertuples():
        t = ro.loc[pid("perfect", f"She has {pairs.set_index('intrans').intrans_part.get(x.lemma, x.lemma)}"), "topk"] \
            if x.lemma in set(pairs.intrans) else None
        if isinstance(t, str):
            top.append(f"  - She has {pairs.set_index('intrans').intrans_part.get(x.lemma, x.lemma)}: " + ", ".join(f"`{json.dumps(k)[1:-1]}` {np.exp(v):.2f}"
                                                                 for k, v in json.loads(t)[:6]))
    L += top + [""]

    # projection groups
    grp = classes.copy()
    she_all = ro[(ro.set == "perfect") & (ro.subject == "She")]
    forms = pd.read_csv(args.forms).set_index("lemma").participle
    grp["M_she"] = grp.lemma.map(lambda l: she_all.M.get(pid("perfect", f"She has {forms.get(l, l)}"), np.nan))
    bsh = pd.read_csv(args.by_split)
    bsh = bsh[bsh.driver == "bad side high"].verb_pair.str.split("/").str[2]
    grp["bad_side_high"] = grp.lemma.isin(set(bsh))
    grp.to_csv(Path(args.data_dir) / "projection_groups.csv", index=False)
    tr = she_all[she_all.lemma.isin(set(verbs[(verbs["class"] == "trans") & verbs.included].lemma))]
    L += ["### Projection groups (for later; She-has M)", "",
          "| Group | Verbs | Mean M | % M > 0 |", "|---|---|---|---|",
          f"| included transitives (all bands) | {len(tr)} | {tr.M.mean():+.2f} | {pct(tr.M > 0)} |"]
    for name, g in [("plain intransitive", grp[grp["class"] == "plain"]),
                    ("prep_object", grp[grp["class"] == "prep_object"]),
                    ("contaminated_bad", grp[grp["class"] == "contaminated_bad"]),
                    ("bad-side-high *by* (original)", grp[grp.bad_side_high])]:
        L.append(f"| {name} | {len(g)} | {g.M_she.mean():+.2f} | {pct(g.M_she > 0)} |")
    L += ["", "contaminated_bad: " + ", ".join(f"{r.lemma} (M {r.M_she:+.1f})" for r in
                                                grp[grp["class"] == "contaminated_bad"].itertuples()),
          ""]
    by = pd.read_csv(args.by_split)
    by["bad"] = by.verb_pair.str.split("/").str[2]
    c = by[by.bad.isin(["jut", "scram", "crackle", "resound"])]
    L += ["Their *by* margins (released contexts): " + "; ".join(
        f"{r.verb_pair.split('/', 1)[1]} {r.by_margin:+.2f} ({r.by_class}{', ' + r.driver if isinstance(r.driver, str) and r.driver else ''})"
        for r in c.itertuples()) + ". Overlap with the bad-side-high group: "
        + (", ".join(sorted(set(c.bad) & set(bsh))) or "none") + ".", ""]

    # ---- task 3: familiarity
    L += ["## 3. Familiarity: participle Zipf", ""]
    ex = pairs[pairs.source == "expansion"]
    surv = ex[ex.part_gap.abs() <= args.part_caliper]
    # rematch with both calipers
    inc = verbs[verbs.included & verbs.tag.isin(["head", "near_head"]) & (verbs.use != "eval")].copy()
    cls = classes.set_index("lemma")["class"]
    inc["iclass"] = np.where(inc["class"] == "trans", "trans", inc.lemma.map(cls).fillna("plain"))
    inc["tok"] = np.where(inc.participle_ntok > 1, "multi", "single")
    both = []
    for tok, g in inc[inc.iclass.isin(["trans", "plain"])].groupby("tok"):
        good, bad = g[g.iclass == "trans"].to_dict("records"), g[g.iclass == "plain"].to_dict("records")
        cands = [(abs(a["lemma_sum_zipf"] - b["lemma_sum_zipf"]) + abs(a["participle_zipf"] - b["participle_zipf"]), i, j)
                 for i, a in enumerate(good) for j, b in enumerate(bad)
                 if abs(a["lemma_sum_zipf"] - b["lemma_sum_zipf"]) <= args.caliper
                 and abs(a["participle_zipf"] - b["participle_zipf"]) <= args.part_caliper]
        for gap, i, j in greedy(cands):
            a, b = good[i], bad[j]
            both.append({"tok": tok, "trans": a["lemma"], "intrans": b["lemma"],
                         "trans_part_zipf": a["participle_zipf"], "intrans_part_zipf": b["participle_zipf"],
                         "trans_sum_zipf": a["lemma_sum_zipf"], "intrans_sum_zipf": b["lemma_sum_zipf"]})
    both = pd.DataFrame(both)
    both.to_csv(Path(args.data_dir) / "train_pairs_participle_matched.csv", index=False)
    d = lambda s: f"{s.mean():.2f} ({s.std():.2f})"
    L += [f"Expansion pairs: participle Zipf trans {d(ex.trans_part_zipf)} vs intrans {d(ex.intrans_part_zipf)}. "
          f"{len(surv)}/{len(ex)} existing pairs already have |participle gap| <= {args.part_caliper}. Rematching "
          f"the same pool under both calipers (summed-lemma <= {args.caliper}, participle <= {args.part_caliper}) "
          f"gives {len(both)} pairs ({(both.tok == 'single').sum() if len(both) else 0} single-token, "
          f"{(both.tok == 'multi').sum() if len(both) else 0} multi-token): "
          f"`data/das_round2/train_pairs_participle_matched.csv`."]
    if len(both):
        L += [f"Robustness set: participle Zipf trans {d(both.trans_part_zipf)} vs intrans {d(both.intrans_part_zipf)}; "
              f"summed-lemma {d(both.trans_sum_zipf)} vs {d(both.intrans_sum_zipf)}.",
              "Pairs: " + ", ".join(f"{r.trans}/{r.intrans}" for r in both.itertuples())]
    o = pairs[pairs.source == "orig_head"]
    L += ["", f"`orig_head` pairs are matched on lemma and participle Zipf by construction: participle Zipf "
          f"trans {d(o.trans_part_zipf)} vs intrans {d(o.intrans_part_zipf)}.", ""]

    # ---- task 4: token count in near_head
    L += ["## 4. Token count in the expansion pool (near_head intransitives; transitives near_head or Head)", ""]
    nh = verb[(verb.source == "expansion")]
    rows = []
    for (tok, side), g in nh.groupby(["tok", "side"]):
        rows.append([tok, side, len(g), int(g.verb_pass.sum())])
    L += table(["Participle tokens", "Class", "Verbs in pairs", "Passing filter"], rows)
    pt = pp[pp.source == "expansion"].groupby("tok").agg(pairs=("trans", "size"), both=("both_pass", "sum"))
    L += ["", "Pairs per token group (all / both verbs passing): " +
          ", ".join(f"{k} {int(r.pairs)} / {int(r.both)}" for k, r in pt.iterrows()) + "."]
    (Path(args.report)).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--readout", default="results/das_round2/pythia14b_readout.csv")
    ap.add_argument("--prompts", default="data/das_round2/prompts.jsonl")
    ap.add_argument("--forms", default="data/das_round2/verb_forms.csv")
    ap.add_argument("--verbs", default="data/verb_expansion/verbs.csv")
    ap.add_argument("--classes", default="data/das_round2/intrans_classes_final.csv")
    ap.add_argument("--v2-pairs", default="data/matched_passives_v2/pairs.jsonl")
    ap.add_argument("--by-split", default="reports/passive_das_prep/task2_by_split_per_verb.csv")
    ap.add_argument("--data-dir", default="data/das_round2")
    ap.add_argument("--report", default="reports/passive_das_prep/round2_analysis.md")
    ap.add_argument("--job", default="8900510")
    ap.add_argument("--caliper", type=float, default=0.25)
    ap.add_argument("--part-caliper", type=float, default=0.35)
    run(ap.parse_args())
