#!/usr/bin/env python3
"""Gemma replication, step 2 inputs for run_frame_contrast.py: freq prefixes x (D11 + curated) participles.

Prefixes: the freq set of data/round6/frame_d11/prefixes.csv (24 nouns x "The N was" / "The N has"; identical to
round 5's). Verbs: the union of the 562 D11 participles (data/round6/frame_d11/verbs.csv) and the 252 curated
participles (data/round5/frame/verbs.csv, curated == 1), with their existing role and band (a verb in both
inventories must have the same role and band). Pairs: the 281 D11 pairs and the 126 curated pairs, pair ids
prefixed with the inventory. No filter on token count. Every verb is scored by full participle log-prob.

Token ids are those of " V" under --model (no special tokens); the joined string "prefix V" must encode to
prefix ids + participle ids for every prefix and verb (BOS, when used, is prepended by the scorer). Both the
Pythia and the Gemma token counts are stored for every verb, so outputs can be split by either.

Writes prefixes.csv, verbs.csv, pairs.csv, meta.json to --out-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
from pathlib import Path

import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    ref = {m: AutoTokenizer.from_pretrained(m) for m in (args.pythia, args.gemma)}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    d11 = pd.read_csv(args.d11_verbs, keep_default_na=False)
    r5 = pd.read_csv(args.round5_verbs, keep_default_na=False)
    cur = r5[r5.curated == 1]
    # one row per participle form; role and band are per inventory (blasphemed is I in D11, T in a curated pair)
    V = pd.DataFrame({"verb": sorted(set(d11.verb) | set(cur.verb))})
    for inv, src in (("d11", d11), ("cur", cur)):
        m = src.set_index("verb")
        V[f"{inv}_role"] = V.verb.map(m.role).fillna("")
        V[f"{inv}_band"] = V.verb.map(m.band).fillna("")
    both = (V.d11_band != "") & (V.cur_band != "")
    assert (V.d11_band[both] == V.cur_band[both]).all(), "a verb is in different bands in D11 and curated"
    V["band"] = V.d11_band.where(V.d11_band != "", V.cur_band)
    V["in_d11"] = (V.d11_role != "").astype(int)
    V["in_curated"] = (V.cur_role != "").astype(int)
    V["curated"] = V.in_curated  # run_frame_contrast.py flag (only used for the cur set, not scored here)
    ids = [tok.encode(" " + v, add_special_tokens=False) for v in V.verb]
    V["n_tok"] = [len(x) for x in ids]
    V["ids"] = [" ".join(map(str, x)) for x in ids]
    V["first_id"] = [x[0] for x in ids]
    V["pythia_ntok"] = [len(ref[args.pythia].encode(" " + v, add_special_tokens=False)) for v in V.verb]
    V["gemma_ntok"] = [len(ref[args.gemma].encode(" " + v, add_special_tokens=False)) for v in V.verb]
    # stored Pythia counts must reproduce
    stored = pd.concat([d11.set_index("verb").n_tok, cur.set_index("verb").n_tok])
    stored = stored[~stored.index.duplicated()]
    assert (V.set_index("verb").pythia_ntok == stored.reindex(V.verb).astype(int)).all()

    pre = pd.read_csv(args.prefixes, keep_default_na=False)
    pre = pre[pre["set"] == "freq"].reset_index(drop=True)
    n_checked = 0
    for prefix in list(pre.pass_prefix) + list(pre.act_prefix):
        pid = tok.encode(prefix, add_special_tokens=False)
        for v, x in zip(V.verb, ids):
            assert tok.encode(prefix + " " + v, add_special_tokens=False) == pid + x, (prefix, v)
            n_checked += 1

    dp = pd.read_csv(args.d11_pairs, keep_default_na=False)
    cp = pd.read_csv(args.round5_pairs, keep_default_na=False)
    pairs = pd.concat([
        pd.DataFrame({"inventory": "d11", "verb_pair": "d11:" + dp.verb_pair, "band": dp.band, "t_verb": dp.t_verb,
                      "i_verb": dp.i_verb, "split": dp.split}),
        pd.DataFrame({"inventory": "curated", "verb_pair": "curated:" + cp.verb_pair, "band": cp.band,
                      "t_verb": cp.t_verb, "i_verb": cp.i_verb, "split": ""})], ignore_index=True)
    assert set(pairs.t_verb) | set(pairs.i_verb) == set(V.verb)
    assert len(pairs[pairs.inventory == "d11"]) == 281 and len(pairs[pairs.inventory == "curated"]) == 126

    pre.to_csv(out / "prefixes.csv", index=False)
    V.to_csv(out / "verbs.csv", index=False)
    pairs.to_csv(out / "pairs.csv", index=False)
    meta = {"model": args.model, "versions": {k: md.version(k) for k in ("transformers", "tokenizers")},
            "inputs": {p: sha(p) for p in (args.d11_verbs, args.round5_verbs, args.prefixes, args.d11_pairs, args.round5_pairs)},
            "joined_tokenization_checks": n_checked, "n_verbs": len(V), "n_prefixes": len(pre),
            "outputs": {f: sha(out / f) for f in ("prefixes.csv", "verbs.csv", "pairs.csv")}}
    meta["role_conflicts"] = V[both & (V.d11_role != V.cur_role)].verb.tolist()
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({k: meta[k] for k in ("model", "joined_tokenization_checks", "n_verbs", "n_prefixes", "role_conflicts")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--pythia", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--gemma", default="google/gemma-3-1b-pt")
    ap.add_argument("--d11-verbs", default="data/round6/frame_d11/verbs.csv")
    ap.add_argument("--d11-pairs", default="data/round6/frame_d11/pairs.csv")
    ap.add_argument("--round5-verbs", default="data/round5/frame/verbs.csv")
    ap.add_argument("--round5-pairs", default="data/round5/frame/pairs.csv")
    ap.add_argument("--prefixes", default="data/round6/frame_d11/prefixes.csv")
    ap.add_argument("--out-dir", required=True)
    main(ap.parse_args())
