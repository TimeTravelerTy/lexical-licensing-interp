#!/usr/bin/env python3
"""Gemma replication, step 1: single-token refilter of the existing verb inventories under the Gemma-3 tokenizer.

Tokenizer only (no model weights). Existing verb files are read, never written.

Inventories (both kept with their existing bands; nothing is re-binned):
- d11: the round-5 D11 pairs (281 T/I pairs; data/round6/frame_d11/verbs.csv, the A2 frame-contrast input whose
  n_tok / ids are the Pythia tokenization of " V"); the Delta_pass inventory.
- curated: the 126 curated band-cross pairs (data/round5/frame/verbs.csv, curated == 1); the *by*-readout inventory.

Form checked: the participle with a leading space, " V", exactly as run_frame_contrast.py scores it after
"The N was" / "The N has" (and as the curated passive prefixes + participle in the *by* readout). For Gemma the
check is (a) " V" encodes to one id without special tokens and (b) for every freq prefix (24 nouns x was/has) the
joined string "prefix V" encodes to prefix ids + that id (no merge across the boundary). The simple-past form
" past" (used in actives, not in step 2) is reported as an extra column only.

Writes to --out-dir: verbs.csv (one row per verb x inventory), counts.csv (class x band x set), meta.json.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
from pathlib import Path

import pandas as pd

BANDS = ("head", "tail", "xtail")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    from transformers import AutoTokenizer

    from build_das_prep_prompts import PAST as PAST_DAS
    from verb_forms import PAST as PAST_FORMS

    PAST = {**PAST_DAS, **PAST_FORMS}  # irregular simple pasts; otherwise past = participle

    gtok = AutoTokenizer.from_pretrained(args.gemma)
    ptok = AutoTokenizer.from_pretrained(args.pythia)
    d11 = pd.read_csv(args.d11, keep_default_na=False).assign(inventory="d11")
    r5 = pd.read_csv(args.round5_verbs, keep_default_na=False)
    cur = r5[r5.curated == 1].assign(inventory="curated")
    V = pd.concat([d11, cur], ignore_index=True)[["inventory", "verb", "role", "band", "verb_pair", "zipf", "n_tok", "ids"]]
    V = V.rename(columns={"n_tok": "pythia_ntok", "ids": "pythia_ids"})
    # the stored Pythia counts are the ones the existing analyses used; re-derive them as a check
    re_p = [len(ptok.encode(" " + v, add_special_tokens=False)) for v in V.verb]
    assert (pd.Series(re_p) == V.pythia_ntok.astype(int)).all(), "stored Pythia token counts do not reproduce"

    pre = pd.read_csv(args.prefixes, keep_default_na=False)
    pre = pre[pre["set"] == "freq"]
    prefixes = list(pre.pass_prefix) + list(pre.act_prefix)
    pids = {p: gtok.encode(p, add_special_tokens=False) for p in prefixes}

    g_ids, g_join, past_n = [], [], []
    for v in V.verb:
        ids = gtok.encode(" " + v, add_special_tokens=False)
        g_ids.append(ids)
        g_join.append(all(gtok.encode(p + " " + v, add_special_tokens=False) == pids[p] + ids for p in prefixes))
        past = PAST.get(v, v)
        past_n.append(len(gtok.encode(" " + past, add_special_tokens=False)))
    V["gemma_ntok"] = [len(x) for x in g_ids]
    V["gemma_ids"] = [" ".join(map(str, x)) for x in g_ids]
    V["gemma_tokens"] = ["|".join(gtok.convert_ids_to_tokens(x)) for x in g_ids]
    V["gemma_join_ok"] = g_join
    V["gemma_past_ntok"] = past_n
    V["pythia_single"] = (V.pythia_ntok.astype(int) == 1).astype(int)
    V["gemma_single"] = ((V.gemma_ntok == 1) & V.gemma_join_ok).astype(int)
    V["both_single"] = V.pythia_single * V.gemma_single

    rows = []
    for inv, g in V.groupby("inventory", sort=False):
        for role in ("T", "I"):
            for band in BANDS:
                c = g[(g.role == role) & (g.band == band)]
                rows.append({"inventory": inv, "class": role, "band": band, "all_verbs": len(c),
                             "pythia_single": int(c.pythia_single.sum()), "gemma_single": int(c.gemma_single.sum()),
                             "intersection": int(c.both_single.sum())})
    C = pd.DataFrame(rows)
    # pairs with both members single-token in both tokenizers
    pr = []
    for inv, g in V.groupby("inventory", sort=False):
        piv = g.pivot_table(index=["verb_pair", "band"], columns="role", values="both_single").reset_index()
        for band in BANDS:
            b = piv[piv.band == band]
            pr.append({"inventory": inv, "band": band, "pairs": len(b), "pairs_both_single_both_tok": int(((b["T"] == 1) & (b["I"] == 1)).sum())})
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    V.to_csv(out / "verbs.csv", index=False)
    C.to_csv(out / "counts.csv", index=False)
    pd.DataFrame(pr).to_csv(out / "pair_counts.csv", index=False)
    flag = C[(C["class"] == "I") & (C.band == "xtail") & (C.intersection < args.min_cell)]
    meta = {"gemma_tokenizer": args.gemma, "pythia_tokenizer": args.pythia,
            "versions": {k: md.version(k) for k in ("transformers", "tokenizers")},
            "inputs": {p: sha(p) for p in (args.d11, args.round5_verbs, args.prefixes)},
            "joined_prefixes_checked": len(prefixes), "n_join_failures_among_gemma_1tok": int(((V.gemma_ntok == 1) & ~V.gemma_join_ok).sum()),
            "min_cell": args.min_cell, "flag_I_xtail_below_min": flag[["inventory", "intersection"]].to_dict("records"),
            "outputs": {f: sha(out / f) for f in ("verbs.csv", "counts.csv", "pair_counts.csv")}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(C.to_string(index=False))
    print(pd.DataFrame(pr).to_string(index=False))
    print(json.dumps({k: meta[k] for k in ("versions", "n_join_failures_among_gemma_1tok", "flag_I_xtail_below_min")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gemma", default="google/gemma-3-1b-pt")
    ap.add_argument("--pythia", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--d11", default="data/round6/frame_d11/verbs.csv")
    ap.add_argument("--round5-verbs", default="data/round5/frame/verbs.csv")
    ap.add_argument("--prefixes", default="data/round6/frame_d11/prefixes.csv")
    ap.add_argument("--min-cell", type=int, default=15)
    ap.add_argument("--out-dir", default="data/gemma/verb_refilter")
    main(ap.parse_args())
