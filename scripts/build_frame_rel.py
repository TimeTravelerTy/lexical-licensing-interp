#!/usr/bin/env python3
"""Round 6, A7 inputs: the D11 verbs in the 300 released FreqBLiMP passive_1 contexts (reports/round6/plan.md, A7).

The released contexts of round P's band cross (data/passive_band_cross/pairs.jsonl, context_set "released",
passive_1: 100 per band, nouns band-matched) give the noun-rarity bands. Each context's passive prefix ("The opinion
wasn't") is paired with its perfect counterpart, the auxiliary mapped was / is -> has, were / are -> have (with n't kept):
set "rel", context_band = the released band. Verbs and pairs: D11 (as data/round6/frame_d11).
Checks: prefix ids + participle ids = the joined string's ids for every verb and prefix.
Writes prefixes.csv, verbs.csv, pairs.csv, meta.json to --out-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

AUX = {"was": "has", "is": "has", "were": "have", "are": "have", "wasn't": "hasn't", "isn't": "hasn't",
       "weren't": "haven't", "aren't": "haven't"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    d = pd.read_json(args.pairs, lines=True)
    r = d[(d.context_set == "released") & (d.paradigm == "passive_1")].drop_duplicates("context_id").sort_values("context_id")
    assert len(r) == 300
    rows = []
    for c in r.itertuples(index=False):
        words = c.prefix.strip().split()
        assert words[-1] in AUX, c.prefix
        rows.append({"set": "rel", "context_id": c.context_id, "context_band": c.context_band, "noun": c.patient_head,
                     "animacy": "", "noun_zipf": c.patient_zipf, "pass_prefix": " ".join(words),
                     "act_prefix": " ".join(words[:-1] + [AUX[words[-1]]])})
    pre = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("verbs.csv", "pairs.csv"):
        shutil.copy(Path(args.d11) / f, out / f)
    V = pd.read_csv(out / "verbs.csv", keep_default_na=False)
    n = 0
    for p in list(pre.pass_prefix) + list(pre.act_prefix):
        pid = tok.encode(p, add_special_tokens=False)
        for v, ids in zip(V.verb, V.ids):
            assert tok.encode(p + " " + v, add_special_tokens=False) == pid + list(map(int, ids.split())), (p, v)
            n += 1
    pre.to_csv(out / "prefixes.csv", index=False)
    meta = {"inputs": {args.pairs: sha(args.pairs), **{f"{args.d11}/{f}": sha(Path(args.d11) / f) for f in ("verbs.csv", "pairs.csv")}},
            "contexts": pre.context_band.value_counts().to_dict(), "noun_zipf_median": pre.groupby("context_band").noun_zipf.median().to_dict(),
            "joined_tokenization_checks": n, "outputs": {f: sha(out / f) for f in ("prefixes.csv", "verbs.csv", "pairs.csv")}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="data/passive_band_cross/pairs.jsonl")
    ap.add_argument("--d11", default="data/round6/frame_d11")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--out-dir", default="data/round6/frame_d11_rel")
    main(ap.parse_args())
