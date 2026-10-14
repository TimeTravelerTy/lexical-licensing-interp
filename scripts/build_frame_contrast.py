#!/usr/bin/env python3
"""Round 5, A1-A2 inputs: prefixes and verbs for the frame contrast (reports/round5/plan.md, A1).

Prefix sets (each with a passive "was" and an active "has" frame):
- freq: "The N was" / "The N has", 24 frequent nouns (declared list; reserves replace a noun that is not one
  Pythia token with a leading space or has wordfreq Zipf < 4.8);
- been: "The N has been" / "The N has", the same nouns (voice-matched sensitivity);
- cur: the 126 curated passive_1 prefixes of the band cross, " was" -> " has";
- they: "They were" / "They have".
Verbs: the T and I participles of the 126 curated pairs, plus every distinct participle of the released
FreqBLiMP passive files (pool verbs: good -> T pool, bad -> I pool; a form in both pools, or not a single
alphabetic word, is excluded).

Needs the Pythia tokenizer (local HF cache) and wordfreq. Writes `prefixes.csv`, `verbs.csv`, `pairs.csv`
and `meta.json` to --out-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ANIMATE = ["man", "woman", "child", "boy", "girl", "student", "teacher", "doctor", "officer", "player",
           "friend", "father"]
INANIMATE = ["house", "car", "book", "door", "letter", "report", "plan", "system", "building", "window",
             "road", "project"]
RESERVES = {"animate": ["mother", "king", "judge"], "inanimate": ["table", "phone", "film"]}
MIN_ZIPF = 4.8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def one_token(tok, text):
    return len(tok.encode(text, add_special_tokens=False)) == 1


def choose_nouns(tok, zipf):
    rows, log = [], []
    for animacy, nouns in (("animate", ANIMATE), ("inanimate", INANIMATE)):
        reserves = list(RESERVES[animacy])
        for noun in nouns:
            cand = noun
            while not (one_token(tok, " " + cand) and zipf(cand) >= MIN_ZIPF):
                log.append(f"{cand}: tokens {len(tok.encode(' ' + cand))}, zipf {zipf(cand):.2f}")
                if not reserves:
                    raise SystemExit(f"no reserve left for {noun}")
                cand = reserves.pop(0)
            rows.append({"noun": cand, "animacy": animacy, "zipf": zipf(cand)})
    return rows, log


def main(args):
    from transformers import AutoTokenizer
    from wordfreq import zipf_frequency

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    zipf = lambda w: zipf_frequency(w, "en")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # prefixes
    nouns, log = choose_nouns(tok, zipf)
    pre = [{"set": "freq", "context_id": f"freq/{n['noun']}", "context_band": "", "noun": n["noun"],
            "animacy": n["animacy"], "noun_zipf": round(n["zipf"], 2),
            "pass_prefix": f"The {n['noun']} was", "act_prefix": f"The {n['noun']} has"} for n in nouns]
    pre += [{"set": "been", "context_id": f"been/{n['noun']}", "context_band": "", "noun": n["noun"],
             "animacy": n["animacy"], "noun_zipf": round(n["zipf"], 2),
             "pass_prefix": f"The {n['noun']} has been", "act_prefix": f"The {n['noun']} has"} for n in nouns]
    bca = pd.read_json(args.pairs, lines=True)
    cur = bca[(bca.context_set == "curated") & (bca.paradigm == "passive_1")]
    ctx = cur.drop_duplicates("context_id").sort_values("context_id")
    for _, r in ctx.iterrows():
        p = r.prefix
        assert p.endswith(" was ") and p.count(" was ") == 1, p
        pre.append({"set": "cur", "context_id": r.context_id, "context_band": r.context_band,
                    "noun": r.patient_head, "animacy": "", "noun_zipf": r.patient_zipf,
                    "pass_prefix": p.rstrip(), "act_prefix": p[:-len(" was ")] + " has"})
    pre.append({"set": "they", "context_id": "they", "context_band": "", "noun": "", "animacy": "",
                "noun_zipf": float("nan"), "pass_prefix": "They were", "act_prefix": "They have"})
    pre = pd.DataFrame(pre)
    assert pre.groupby("set").size().to_dict() == {"freq": 24, "been": 24, "cur": 126, "they": 1}

    # curated pairs
    pairs = cur.drop_duplicates("verb_pair").sort_values("verb_pair")
    assert len(pairs) == 126
    own = cur[cur.own_context == 1].set_index("verb_pair").context_id
    pairs = pd.DataFrame({"verb_pair": pairs.verb_pair, "band": pairs.verb_band, "t_verb": pairs.good_verb,
                          "i_verb": pairs.bad_verb, "t_zipf": pairs.good_form_zipf, "i_zipf": pairs.bad_form_zipf,
                          "t_lemma_zipf": pairs.good_lemma_zipf, "i_lemma_zipf": pairs.bad_lemma_zipf,
                          "own_context_id": pairs.verb_pair.map(own)}).reset_index(drop=True)
    assert pairs.own_context_id.notna().all()

    # verbs
    verbs = {}
    for _, r in pairs.iterrows():
        for role in ("t", "i"):
            v = r[f"{role}_verb"]
            assert v not in verbs, f"curated verb repeats: {v}"
            verbs[v] = {"verb": v, "role": role.upper(), "curated": 1, "pool": 0, "band": r.band,
                        "verb_pair": r.verb_pair, "zipf": r[f"{role}_zipf"]}
    rel = pd.read_json(args.released, lines=True)
    pool = {}
    for side, role in (("good", "T"), ("bad", "I")):
        for band, v in rel[[ "band", f"{side}_verb"]].drop_duplicates().itertuples(index=False):
            pool.setdefault(v, set()).add((role, band))
    excluded = []
    for v, tags in sorted(pool.items()):
        roles = {t[0] for t in tags}
        if len(roles) > 1 or not v.isalpha():
            excluded.append(v)
            continue
        role = roles.pop()
        bands = sorted({t[1] for t in tags})
        if v in verbs:
            if verbs[v]["role"] != role:
                excluded.append(v)
                continue
            verbs[v]["pool"] = 1
            continue
        verbs[v] = {"verb": v, "role": role, "curated": 0, "pool": 1, "band": "|".join(bands),
                    "verb_pair": "", "zipf": round(zipf(v), 2)}
    V = pd.DataFrame(list(verbs.values()))
    ids = [tok.encode(" " + v, add_special_tokens=False) for v in V.verb]
    V["n_tok"] = [len(x) for x in ids]
    V["ids"] = [" ".join(map(str, x)) for x in ids]
    V["first_id"] = [x[0] for x in ids]
    V = V.sort_values(["curated", "verb"], ascending=[False, True]).reset_index(drop=True)

    # tokenization is compositional: joined string = prefix ids + participle ids
    for p in list(pre.pass_prefix) + list(pre.act_prefix):
        pid = tok.encode(p, add_special_tokens=False)
        for v, vid in zip(V.verb[:50], ids[:50]):
            assert tok.encode(p + " " + v, add_special_tokens=False) == pid + vid, (p, v)
    for v, vid in zip(V.verb, ids):
        assert tok.encode("The man was " + v, add_special_tokens=False) == \
            tok.encode("The man was", add_special_tokens=False) + vid, v

    pre.to_csv(out / "prefixes.csv", index=False)
    V.to_csv(out / "verbs.csv", index=False)
    pairs.to_csv(out / "pairs.csv", index=False)
    import importlib.metadata as md
    meta = {"inputs": {args.pairs: sha(args.pairs), args.released: sha(args.released)},
            "versions": {k: md.version(k) for k in ("wordfreq", "transformers", "tokenizers")},
            "model": args.model, "noun_replacements": log, "pool_excluded": excluded,
            "counts": {"prefixes": pre.groupby("set").size().to_dict(),
                       "verbs": {f"curated{c}_{r}": int(n) for (c, r), n in V.groupby(["curated", "role"]).size().items()},
                       "single_token_rare": V[(V.n_tok == 1) & (~V.band.str.fullmatch("head"))]
                       .groupby("role").size().to_dict()},
            "outputs": {f: sha(out / f) for f in ("prefixes.csv", "verbs.csv", "pairs.csv")}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps(meta["counts"], indent=1, default=str))
    print("noun replacements:", log or "none")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/passive_band_cross/pairs.jsonl")
    ap.add_argument("--released", default="data/freqblimp_original_passives/pairs.jsonl")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--out-dir", default="data/round5/frame")
    main(ap.parse_args())
