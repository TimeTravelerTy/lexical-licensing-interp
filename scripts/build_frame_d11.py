#!/usr/bin/env python3
"""Round 6, A2 inputs: the round-5 frame-contrast prefixes with the D11 verbs (reports/round6/plan.md, A1-A2).

Prefixes: round 5's prefixes.csv unchanged (freq, been, cur, they). Verbs: the 562 participles of the D11 pairs
(data/round5/verbs_expanded/pairs.csv, loaded with the split checks of verbs_expanded.load_pairs). Every verb is
scored in every prefix set (run_frame_contrast.py --cur-all). The curated flag marks verbs that are also curated
band-cross verbs (for the reproduction check against round 5).

Checks: participle ids from " V" alone; the joined string's ids equal prefix ids + participle ids for every verb in
every freq / been / they prefix and in a sample of cur prefixes; token counts equal D11's t_ntok / i_ntok.

Writes prefixes.csv, verbs.csv, pairs.csv and meta.json to --out-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from verbs_expanded import load_pairs


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    P = load_pairs("all", Path(args.pairs))
    cur = set(pd.read_csv(Path(args.round5) / "verbs.csv", keep_default_na=False).query("curated == 1").verb)
    pairs = pd.DataFrame({"verb_pair": P.pair_id, "band": P.band, "t_verb": P.t_participle, "i_verb": P.i_participle,
                          "t_lemma": P.t_lemma, "i_lemma": P.i_lemma, "t_zipf": P.t_part_zipf, "i_zipf": P.i_part_zipf,
                          "t_lemma_zipf": P.t_lemma_zipf, "i_lemma_zipf": P.i_lemma_zipf, "t_ntok": P.t_ntok,
                          "i_ntok": P.i_ntok, "tok_cell": P.tok_cell, "split": P.split, "t_flag": P.t_flag,
                          "i_flag": P.i_flag})
    rows = []
    for r in pairs.itertuples(index=False):
        for role in ("t", "i"):
            v = getattr(r, f"{role}_verb")
            rows.append({"verb": v, "role": role.upper(), "curated": int(v in cur), "pool": 0, "band": r.band,
                         "verb_pair": r.verb_pair, "zipf": getattr(r, f"{role}_zipf"), "split": r.split,
                         "d11_ntok": getattr(r, f"{role}_ntok")})
    V = pd.DataFrame(rows)
    assert V.verb.is_unique, "a participle appears twice in D11"
    ids = [tok.encode(" " + v, add_special_tokens=False) for v in V.verb]
    V["n_tok"] = [len(x) for x in ids]
    V["ids"] = [" ".join(map(str, x)) for x in ids]
    V["first_id"] = [x[0] for x in ids]
    assert (V.n_tok == V.d11_ntok).all(), V[V.n_tok != V.d11_ntok]
    V = V.drop(columns="d11_ntok")

    shutil.copy(Path(args.round5) / "prefixes.csv", out / "prefixes.csv")
    pre = pd.read_csv(out / "prefixes.csv", keep_default_na=False)
    n_checked = 0
    for st, pp, ap_ in zip(pre["set"], pre.pass_prefix, pre.act_prefix):
        for prefix in (pp, ap_):
            pid = tok.encode(prefix, add_special_tokens=False)
            sample = V.index if st != "cur" else V.index[::7]
            for j in sample:
                assert tok.encode(prefix + " " + V.verb[j], add_special_tokens=False) == pid + ids[j], (prefix, V.verb[j])
                n_checked += 1
    V.to_csv(out / "verbs.csv", index=False)
    pairs.to_csv(out / "pairs.csv", index=False)
    import importlib.metadata as md
    meta = {"inputs": {args.pairs: sha(args.pairs), str(Path(args.round5) / "prefixes.csv"): sha(out / "prefixes.csv")},
            "versions": {k: md.version(k) for k in ("transformers", "tokenizers")}, "model": args.model,
            "joined_tokenization_checks": n_checked,
            "counts": {"pairs": pairs.groupby("band").size().to_dict(), "verbs": len(V),
                       "curated_verbs": int(V.curated.sum()),
                       "single_token": {f"{b}/{r}": int(n) for (b, r), n in V[V.n_tok == 1].groupby(["band", "role"]).size().items()}},
            "outputs": {f: sha(out / f) for f in ("prefixes.csv", "verbs.csv", "pairs.csv")}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps(meta, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="data/round5/verbs_expanded/pairs.csv")
    ap.add_argument("--round5", default="data/round5/frame")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--out-dir", default="data/round6/frame_d11")
    main(ap.parse_args())
