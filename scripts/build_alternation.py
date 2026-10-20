#!/usr/bin/env python3
"""Round 6, B3 inputs: object / clause-end frames for the D11 verbs (reports/round6/plan.md, B3).

Prefixes: "The N p" (simple past; primary) and "The N has v" (perfect; secondary) for the 562 D11 participles and the 24
A1 freq nouns. Token sets (Pythia, each must be one token): O = round 2's object-onset set (an object-onset proxy);
O_pron = pronoun objects; O_strict = him / them / me / us; END = clause-end punctuation; DOT = "." (primary clause-end
readout); PREP = the prepositions of round 2's I set.
Checks: every set member is one token; the prefix ids equal "The N" (or "The N has") ids + the verb's own " p" ids.
Writes prefixes.csv (id, verb, role, band, frame, noun, animacy, text, n_verb_tok), sets.json, meta.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from analyze_das_round2 import I_SET, O_SET
from verb_forms import PAST

O_PRON = (" him", " her", " them", " it", " me", " us", " himself", " herself", " themselves", " itself")
O_STRICT = (" him", " them", " me", " us")  # unambiguous objective pronouns (no possessive her, no resultative reflexive)
END = (".", ",", ";", ":", "!", "?", "\n")
PREP = tuple(t for t in I_SET if t.strip() in {"to", "in", "with", "on", "at", "for", "from", "into", "over", "about", "of",
                                               "upon", "against", "through", "after", "under", "around", "across", "toward",
                                               "behind", "near", "onto", "without", "during", "like"})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    V = pd.read_csv(Path(args.d11) / "verbs.csv", keep_default_na=False)
    pre = pd.read_csv(Path(args.d11) / "prefixes.csv", keep_default_na=False)
    nouns = pre[pre.set == "freq"][["noun", "animacy"]].to_numpy()
    sets = {"O": list(O_SET), "O_pron": list(O_PRON), "O_strict": list(O_STRICT), "END": list(END), "DOT": ["."],
            "PREP": list(PREP)}
    ids = {}
    for k, toks in sets.items():
        enc = [tok.encode(t, add_special_tokens=False) for t in toks]
        assert all(len(e) == 1 for e in enc), (k, [t for t, e in zip(toks, enc) if len(e) != 1])
        ids[k] = [e[0] for e in enc]
    rows = []
    for v in V.itertuples(index=False):
        p = PAST.get(v.verb, v.verb)
        for frame, form, aux in (("past", p, ""), ("perf", v.verb, " has")):
            vid = tok.encode(" " + form, add_special_tokens=False)
            for noun, an in nouns:
                text = f"The {noun}{aux} {form}"
                assert tok.encode(text, add_special_tokens=False) == tok.encode(f"The {noun}{aux}", add_special_tokens=False) + vid
                rows.append({"id": f"{frame}|{noun}|{v.verb}", "verb": v.verb, "role": v.role, "band": v.band,
                             "frame": frame, "noun": noun, "animacy": an, "text": text, "n_verb_tok": len(vid)})
    P = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    P.to_csv(out / "prefixes.csv", index=False)
    (out / "sets.json").write_text(json.dumps({"tokens": sets, "ids": ids}, indent=1))
    meta = {"rows": len(P), "verbs": int(P.verb.nunique()), "inputs": {f: sha(Path(args.d11) / f) for f in ("verbs.csv", "prefixes.csv")},
            "outputs": {f: sha(out / f) for f in ("prefixes.csv", "sets.json")}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d11", default="data/round6/frame_d11")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--out-dir", default="data/round6/alternation")
    main(ap.parse_args())
