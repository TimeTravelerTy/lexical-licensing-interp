#!/usr/bin/env python3
"""Build DAS datasets for verb argument structure (no training).

Train (`actives.jsonl`): "The AGENT VERB-ed" with the transitive (good) vs
intransitive (bad) verb of each pair, same subject. Target after the verb:
object start " the" (transitive) vs "." (intransitive). Subjects are the
agents of all curated contexts. Splits: Head verb pairs -> `head_train` /
`head_dev` (held-out verb pairs); Tail and XTail pairs -> `transfer_tail`,
`transfer_xtail` for later transfer checks.

Passive eval (`passives.jsonl`): curated `was` prefix + participle, other
verbs' contexts only (contexts whose prefix equals the pair's own prefix are
dropped too). Splits by the released-context *by* margin of the verb pair
(task 0): `reliable` (CI > 0), `negative_by` (mean < 0), `ns_pos` (mean > 0,
CI includes 0). Each row records patient fit for both verbs: good-verb fit of
"The hat was doffed." (`passive_2`) and of the `passive_1` sentence with its
agent, and bad-verb patient fit of "The hat salivated.".

Token positions (0-based, no special tokens): `*_verb_tokens` = [first, last]
token of the verb; `*_readout_index` = last prompt token (logits there predict
the next token); `*_next_index` = index the next token would take.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_fit_ratings import mean_rating
from build_das_prep_prompts import PAST, pid

TRACK = {"lp_the": "lp[ the]", "lp_a": "lp[ a]", "lp_by": "lp[ by]", "lp_dot": "lp[.]"}


def positions(tokenizer, prefix, verb):
    text = prefix + verb
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    start, end = len(prefix), len(text)
    idx = [i for i, (s, e) in enumerate(enc.offset_mapping) if e > start and s < end]
    n = len(enc.input_ids)
    return {"prompt": text, "n_tokens": n, "verb_tokens": [idx[0], idx[-1]],
            "readout_index": n - 1, "next_index": n}


def side(tokenizer, prefix, verb, kind, readout, tag):
    p = positions(tokenizer, prefix, verb)
    out = {f"{tag}_{k}": v for k, v in p.items()}
    if readout is not None:
        r = readout.loc[pid(kind, p["prompt"])]
        out.update({f"{tag}_{k}": round(float(r[c]), 4) for k, c in TRACK.items()})
    return out


def run(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    readout = pd.read_csv(args.readout, low_memory=False).set_index("prompt_id") \
        if Path(args.readout).exists() else None
    scores = pd.read_csv(args.scores, low_memory=False)
    v2 = pd.DataFrame([json.loads(l) for l in Path(args.v2_pairs).open(encoding="utf-8")])
    v2 = v2[v2.frame_id == "was"]
    pairs = scores.drop_duplicates("verb_pair")[["verb_pair", "verb_band", "good_verb", "bad_verb"]] \
        .sort_values("verb_pair")

    # ---- actives
    rng = np.random.default_rng(args.seed)
    head = sorted(pairs[pairs.verb_band == "head"].verb_pair)
    dev = set(np.array(head)[rng.choice(len(head), args.n_dev, replace=False)])
    p1 = v2[v2.paradigm == "passive_1"]  # passive_2 contexts have no agent
    own_agents = {(r.band, r.good_lemma): r.agent for r in p1.itertuples()}
    agents = sorted(p1.agent.unique())
    n_act = 0
    with (out_dir / "actives.jsonl").open("w", encoding="utf-8") as f:
        for p in pairs.itertuples():
            split = ("head_dev" if p.verb_pair in dev else "head_train") if p.verb_band == "head" \
                else f"transfer_{p.verb_band}"
            band, lemma = p.verb_pair.split("/")[:2]
            for agent in agents:
                row = {"item_id": f"{p.verb_pair}|{agent}", "split": split, "verb_pair": p.verb_pair,
                       "verb_band": p.verb_band, "subject": agent,
                       "own_subject": int(agent == own_agents.get((band, lemma))),
                       "trans_verb": PAST.get(p.good_verb, p.good_verb), "intrans_verb": p.bad_verb,
                       "trans_target": " the", "intrans_target": "."}
                row.update(side(tok, f"The {agent} ", row["trans_verb"], "active", readout, "trans"))
                row.update(side(tok, f"The {agent} ", row["intrans_verb"], "active", readout, "intrans"))
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_act += 1

    # ---- passives
    _, good = mean_rating(args.good_ratings)
    _, badr = mean_rating(args.bad_ratings)
    gl = pd.read_csv(args.good_links, dtype={"item_id": str})[["pair_id", "item_id"]] \
        .merge(good[["item_id", "rating"]], on="item_id").set_index("pair_id").rating
    bl = pd.read_csv(args.bad_links, dtype={"item_id": str})
    bl = bl[bl.role == "patient"].merge(badr[["item_id", "rating"]], on="item_id").set_index("pair_id").rating
    rel = pd.read_csv(args.reliability).set_index("verb_pair")
    cls = rel.by_class.map({"reliable_pos": "reliable", "ns_pos": "ns_pos",
                            "ns_neg": "negative_by", "reliable_neg": "negative_by"})
    cur = scores[(scores.context_set == "curated") & (scores.paradigm == "passive_2")]
    own_prefix = cur[cur.own_context == 1].set_index("verb_pair").prefix
    p1_margin = scores[(scores.context_set == "curated") & (scores.paradigm == "passive_1")] \
        .set_index("pair_id")[["whole_margin", "by_margin"]]
    n_pas, dropped = {}, 0
    with (out_dir / "passives.jsonl").open("w", encoding="utf-8") as f:
        for r in cur[cur.own_context == 0].itertuples():
            if r.prefix == own_prefix[r.verb_pair]:
                dropped += 1
                continue
            p1_id = r.pair_id.replace("curated/passive_2/", "curated/passive_1/")
            row = {"item_id": r.pair_id, "split": cls[r.verb_pair], "verb_pair": r.verb_pair,
                   "verb_band": r.verb_band, "context_id": r.context_id, "context_band": r.context_band,
                   "patient": r.patient_head, "good_verb": r.good_verb, "bad_verb": r.bad_verb,
                   "pair_by_margin_released": round(float(rel.by_margin[r.verb_pair]), 4),
                   "good_fit_p2": round(float(gl[r.pair_id]), 4), "good_fit_p1": round(float(gl[p1_id]), 4),
                   "bad_patient_fit": round(float(bl[r.pair_id]), 4),
                   "lp_whole_margin_p2": round(float(r.whole_margin), 4),
                   "lp_whole_margin_p1": round(float(p1_margin.whole_margin[p1_id]), 4),
                   "lp_by_margin_p1": round(float(p1_margin.by_margin[p1_id]), 4)}
            row.update(side(tok, r.prefix, r.good_verb, "passive", readout, "good"))
            row.update(side(tok, r.prefix, r.bad_verb, "passive", readout, "bad"))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_pas[row["split"]] = n_pas.get(row["split"], 0) + 1
    manifest = {"actives": n_act, "head_dev_pairs": sorted(dev), "passives_by_split": n_pas,
                "passives_dropped_same_prefix_as_own": dropped, "readout_attached": readout is not None,
                "seed": args.seed, "model_tokenizer": args.model}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if readout is not None and args.report:
        summarize(out_dir, Path(args.report), manifest)


def summarize(out_dir, report, manifest):
    act = pd.read_json(out_dir / "actives.jsonl", lines=True, dtype={"item_id": str})
    pas = pd.read_json(out_dir / "passives.jsonl", lines=True, dtype={"item_id": str})
    # Logit difference " the" - "." at the readout position.
    act["trans_ld"] = act.trans_lp_the - act.trans_lp_dot
    act["intrans_ld"] = act.intrans_lp_the - act.intrans_lp_dot
    act["multi_token"] = (act.trans_verb_tokens.str[1] > act.trans_verb_tokens.str[0]) | \
                         (act.intrans_verb_tokens.str[1] > act.intrans_verb_tokens.str[0])
    lines = ["# Task 5: DAS datasets (built, not trained)", "",
             "Files in `data/passive_das/` (`actives.jsonl`, `passives.jsonl`, `manifest.json`), built by "
             "`scripts/build_das_passive_datasets.py`. Token indices are 0-based without special tokens: "
             "`*_verb_tokens` = [first, last] verb token, `*_readout_index` = last prompt token (its logits "
             "predict the next token), `*_next_index` = position of the next token. Baseline log-probs "
             "(`*_lp_the`, `*_lp_a`, `*_lp_by`, `*_lp_dot`) come from the task 1-4 readout.", "",
             "## Train: actives (\"The AGENT VERB-ed\" -> \" the\" vs \".\")", "",
             f"90 subjects (curated `passive_1` agents) x each verb pair. Head dev pairs (held out): "
             f"{', '.join(v.split('/', 1)[1] for v in manifest['head_dev_pairs'])}. `LD` = log P(\" the\") - "
             "log P(\".\"). Target behaviour: LD > 0 for the transitive verb, LD < 0 for the intransitive.", "",
             "| Split | Items | Pairs | Trans LD | Intrans LD | Trans LD > 0 | Intrans LD < 0 | Both | Multi-token verb |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split, g in act.groupby("split"):
        both = ((g.trans_ld > 0) & (g.intrans_ld < 0)).mean()
        lines.append(f"| {split} | {len(g):,} | {g.verb_pair.nunique()} | {g.trans_ld.mean():.2f} | "
                     f"{g.intrans_ld.mean():.2f} | {(g.trans_ld > 0).mean():.1%} | {(g.intrans_ld < 0).mean():.1%} | "
                     f"{both:.1%} | {g.multi_token.mean():.1%} |")
    lines += ["", "## Eval: passives (curated `was` prefix + participle, other contexts)", "",
              f"Contexts whose prefix equals the pair's own prefix are dropped "
              f"({manifest['passives_dropped_same_prefix_as_own']} items). Fit: Gemma-4-31B-it rating (1-7).", "",
              "| Split | Band | Items | Pairs | Good fit (p2) | Bad patient fit | lp by good | lp by bad | "
              "lp the good | lp the bad |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (split, band), g in pas.groupby(["split", "verb_band"]):
        lines.append(f"| {split} | {band} | {len(g):,} | {g.verb_pair.nunique()} | {g.good_fit_p2.mean():.2f} | "
                     f"{g.bad_patient_fit.mean():.2f} | {g.good_lp_by.mean():.2f} | {g.bad_lp_by.mean():.2f} | "
                     f"{g.good_lp_the.mean():.2f} | {g.bad_lp_the.mean():.2f} |")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default="results/passive_band_cross/pythia14b_scores.csv")
    ap.add_argument("--readout", default="results/passive_das_prep/pythia14b_readout.csv")
    ap.add_argument("--reliability", default="reports/passive_das_prep/task0_by_reliability.csv")
    ap.add_argument("--v2-pairs", default="data/matched_passives_v2/pairs.jsonl")
    ap.add_argument("--good-ratings", default="results/fit_ratings/gemma4_31b_it.csv")
    ap.add_argument("--good-links", default="data/fit_ratings/item_links.csv")
    ap.add_argument("--bad-ratings", default="results/fit_ratings/bad_gemma4_31b_it.csv")
    ap.add_argument("--bad-links", default="data/fit_ratings/bad_item_links.csv")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--out-dir", default="data/passive_das")
    ap.add_argument("--report", default="reports/passive_das_prep/task5_das_datasets.md")
    ap.add_argument("--n-dev", type=int, default=6)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
