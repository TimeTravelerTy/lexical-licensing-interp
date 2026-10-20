#!/usr/bin/env python3
"""Round 6, C5 inputs: in-context preemption contexts (reports/round6/plan.md, C5; revised after review).

Verbs: the 281 D11 intransitives (data/round6/preempt/i_frames.csv: subject class, plural flag, complement, fixed
subject, adjunct policy, bare_ok, dynamic, theme) and 80 rare D11 transitives (t_frames.csv: subject class, object, fixed
subject). k in {1, 2, 4, 8}; k = 0 is the probe alone.
Arms (I verbs): pre (simple-past intransitive use), exp (nominal -ing mentions), mention (exact probed form, mentioned
metalinguistically: "Someone wrote glared on the board."), fill (verbless fragments matched in token length to the pre
sentences), other (the pre sentences of another intransitive of the same band with the same complement status and the
closest context length), vs (present: V-s; base form with plural subjects), ing ("is / are V-ing"). Arms (T controls):
trans ("The S p the O adjunct.") and fill_t.
Each verb's 8 sentence slots (subject x adjunct x carrier template) are permuted by a verb-specific seed, and an arm's
k-sentence context is the first k slots in that order (the same order for every arm of the verb). Within a verb and k,
every arm's context is left-padded with verbless fragments to the longest arm's token length (+-1 token), so the arms of
a contrast have equal length and the arm's own sentences stay next to the probe.
Checks: no be- or have-form immediately before any form of the target lemma (except arm "ing"); no probe noun in any
context; joined tokenization = parts; lengths equal within (verb, k) up to 1 token.
Writes contexts.csv (cid, verb, lemma, role, band, arm, k, donor, context, n_tok_ctx, n_pad_tok, dist_last_mention),
meta.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from verb_forms import forms

POOLS = {
    "P": ["visitor", "clerk", "neighbor", "pilot", "nurse", "guard", "waiter", "farmer"],
    "A": ["dog", "cat", "horse", "goat", "pig", "cow", "puppy", "donkey"],
    "O": ["lamp", "fence", "bottle", "kettle", "wagon", "chair", "ladder", "bucket"],
    "E": ["accident", "crisis", "delay", "storm", "conflict", "dispute", "illness", "problem"],
    "Ew": ["storm", "fire", "war", "riot", "battle", "epidemic", "flood", "blizzard"],
    "Sh": ["show", "film", "play", "album", "product", "campaign", "sequel", "concert"],
    "Dr": ["soda", "champagne", "cider", "lemonade", "beer", "tonic", "cola", "wine"],
    "Ki": ["pan", "plate", "spoon", "tray", "lid", "cup", "kettle", "bucket"],
    "Cn": ["dog", "cat", "puppy", "fox", "kitten", "hound", "coyote", "jackal"],
    "Lg": ["horse", "cow", "donkey", "bear", "elephant", "ox", "buffalo", "camel"],
    "Q": ["price", "rate", "cost", "demand", "number", "output", "income", "value"],
    "S": ["water", "oil", "milk", "juice", "paint", "mud", "wine", "soup"],
    "Lt": ["flame", "light", "candle", "lantern", "torch", "star", "beam", "spark"],
    "Fi": ["fire", "ember", "log", "cigarette", "coal", "wick", "rubble", "fuse"],
    "Sd": ["bell", "alarm", "siren", "radio", "drum", "horn", "whistle", "gong"],
    "M": ["engine", "printer", "elevator", "pump", "motor", "scanner", "generator", "furnace"],
    "Vh": ["truck", "bus", "van", "bike", "boat", "cart", "plane", "tractor"],
    "Ob": ["apple", "log", "plank", "leaf", "stump", "carcass", "fruit", "tooth"],
    "Pt": ["nail", "pipe", "root", "rock", "bone", "beam", "wire", "ledge"],
    "G": ["path", "trail", "river", "stream", "tunnel", "track", "corridor", "canal"],
    "W": ["river", "stream", "lake", "pond", "canal", "bay", "harbor", "lagoon"],
    "Pl": ["town", "city", "suburb", "camp", "market", "estate", "village", "campus"],
    "F": ["rose", "tree", "plant", "flower", "tulip", "bush", "vine", "lily"],
    "D": ["contract", "license", "permit", "policy", "warranty", "lease", "deal", "treaty"],
    "T": ["day", "week", "month", "year", "hour", "minute", "season", "decade"],
    "Td": ["day", "morning", "era", "age", "spring", "summer", "century", "decade"],
}
PLURAL = {"crisis": "crises", "puppy": "puppies", "party": "parties", "ox": "oxen"}
ADJ = ["again", "at last", "once more", "suddenly", "today", "briefly", "in the end", "without warning"]
GERUND = ["Nobody noticed the {g}.", "The {g} continued for a while.", "They talked about the {g}.",
          "Everyone mentioned the {g}.", "The {g} stopped suddenly.", "We heard about the {g}.", "Nobody liked the {g}.",
          "The {g} began again."]
MENTION = ["The crossword answer: {v}.", "Someone wrote {v} on the board.", "The spelling list ended with {v}.",
           "Nobody could spell {v} correctly.", "The next word on the card: {v}.", "Her notes said only {v}.",
           "The puzzle needed the word {v}.", "The last entry on the page read {v}."]
FILLERS = ["A quiet morning in the old town.", "Rain over the hills.", "Coffee, bread and a little cheese.",
           "Another long day at the office.", "Bright lights along the river.", "Cold wind from the north.",
           "A small table near the wall.", "Green fields beyond the village.", "Two cups of tea on the counter.",
           "A red umbrella by the gate.", "Snow on the mountains again.", "Music from a distant radio.",
           "An empty bench in the park.", "Fresh flowers in a blue vase.", "A long queue outside the bakery.",
           "Silence in the narrow street.", "Grey clouds over the harbor.", "A stack of papers on the desk.",
           "Warm bread from the oven.", "Footprints in the wet sand.", "A faint smell of smoke.",
           "Lanterns along the old bridge.", "A map of the city on the wall.", "Three boats near the shore.",
           "Dust on every shelf.", "Laughter from the next room.", "A single light in the hallway.",
           "Mild weather for the season.", "Apples and pears in a basket.", "A wooden fence around the garden.",
           "Thin ice on the pond.", "A crowd near the entrance.", "Bells in the distance.", "Old photographs in a drawer.",
           "A strong smell of coffee.", "Heavy rain all night long.", "A narrow staircase to the attic.",
           "Two old chairs by the fireplace, a rug and a lamp.", "Breakfast on the terrace at dawn.",
           "A long, slow afternoon by the lake, with a volume of poems."]
PAD = ["Later.", "Quiet.", "Morning.", "A pause.", "Then silence.", "Rain again.", "Soft music.", "Distant thunder.",
       "A long silence.", "Grey skies outside.", "Cold coffee on the table.", "A light breeze from the sea.",
       "Sunlight on the old stone wall.", "Another quiet afternoon in the small garden.", "Evening.", "Outside, wind.",
       "Fog over the fields.", "A clock on the shelf.", "Footsteps upstairs.", "Bread and soup for lunch."]
BE_HAVE = {"am", "is", "are", "was", "were", "be", "been", "being", "has", "have", "had", "having"}
ARMS_I = ("pre", "exp", "mention", "fill", "other", "vs", "ing")
KS = (1, 2, 4, 8)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plural(n):
    if n in PLURAL:
        return PLURAL[n]
    if n.endswith(("s", "x", "ch", "sh")):
        return n + "es"
    if n.endswith("y") and n[-2] not in "aeiou":
        return n[:-1] + "ies"
    return n + "s"


def subj(cls, i, pl, override):
    n = override or POOLS[cls][i % 8]
    return f"The {plural(n) if pl else n}"


def verb_order(verb):
    return np.random.default_rng(zlib.crc32(verb.encode())).permutation(8)


def sentences_i(r, f, order):
    comp = f" {r.comp}" if r.comp else ""
    pl = bool(int(r.plural))
    out = {a: [] for a in ("pre", "vs", "ing", "exp", "mention")}
    for i in order:
        s = subj(r.subj_class, i, pl, r.subj_override)
        adj = "" if r.adj == "none" else f" {ADJ[i]}"
        out["pre"].append(f"{s} {f['past']}{comp}{adj}.")
        out["vs"].append(f"{s} {f['base'] if pl else f['3sg']}{comp}{adj}.")
        out["ing"].append(f"{s} {'are' if pl else 'is'} {f['ing']}{comp}{adj}.")
        out["exp"].append(GERUND[i].format(g=f["ing"]))
        out["mention"].append(MENTION[i].format(v=f["participle"]))
    return out


class Tok:
    def __init__(self, tok):
        self.tok, self.cache = tok, {}

    def n(self, s):
        if s not in self.cache:
            self.cache[s] = len(self.tok.encode(" " + s, add_special_tokens=False))
        return self.cache[s]


def match_fillers(T, sents):
    used, out = set(), []
    for s in sents:
        n = T.n(s)
        best = min((abs(T.n(x) - n), j) for j, x in enumerate(FILLERS) if j not in used)
        used.add(best[1])
        out.append(FILLERS[best[1]])
    return out


def pad_to(T, sents, target):
    """Prepend PAD fragments (largest first, no repeats) until the context is within 1 token of target."""
    cur = sum(T.n(s) for s in sents)
    pads = []
    for p in sorted(PAD, key=lambda x: -T.n(x)):
        if target - cur <= 1:
            break
        if T.n(p) <= target - cur:
            pads.append(p)
            cur += T.n(p)
    assert target - cur <= 1, (target, cur)
    return pads + sents, sum(T.n(p) for p in pads)


def main(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    T = Tok(tok)
    P = pd.read_csv(args.pairs, keep_default_na=False)
    IF = pd.read_csv(Path(args.data_dir) / "i_frames.csv", keep_default_na=False).set_index("lemma")
    TF = pd.read_csv(Path(args.data_dir) / "t_frames.csv", keep_default_na=False).set_index("lemma")
    pre = pd.read_csv(args.prefixes, keep_default_na=False)
    probe_nouns = set(pre[pre.set == "freq"].noun)
    pool_nouns = {x for v in POOLS.values() for n in v for x in (n, plural(n))}
    assert not probe_nouns & pool_nouns, "a subject-pool noun is a probe noun"
    I = {}
    for r in P.itertuples(index=False):
        f = IF.loc[r.i_lemma]
        fo = forms(r.i_lemma, r.i_participle)
        I[r.i_participle] = {"lemma": r.i_lemma, "band": r.band, "forms": fo, "comp": f.comp != "",
                             "sents": sentences_i(f, fo, verb_order(r.i_participle))}
    # other-verb donors: within band, sorted by (complement status, pre length at k = 8), each verb's donor is the next
    # verb in that order (cyclic), so donors share complement status where possible and have similar length
    donor = {}
    plen = {v: sum(T.n(s) for s in d["sents"]["pre"]) for v, d in I.items()}
    for v, d in I.items():  # nearest pre length (k = 8) within band x complement status; ties by a verb-specific hash
        grp = [u for u, e in I.items() if u != v and e["band"] == d["band"] and e["comp"] == d["comp"]]
        assert grp, f"no donor for {v}"
        donor[v] = min(grp, key=lambda u: (abs(plen[u] - plen[v]), zlib.crc32((v + "|" + u).encode())))
    rows = []
    for v, d in I.items():
        d["sents"]["fill"] = match_fillers(T, d["sents"]["pre"])
        d["sents"]["other"] = I[donor[v]]["sents"]["pre"]
        for k in KS:
            ctx = {arm: d["sents"][arm][:k] for arm in ARMS_I}
            target = max(sum(T.n(s) for s in c) for c in ctx.values())
            for arm in ARMS_I:
                sents, npad = pad_to(T, ctx[arm], target)
                rows.append({"verb": v, "lemma": d["lemma"], "role": "I", "band": d["band"], "arm": arm, "k": k,
                             "donor": donor[v] if arm == "other" else "", "context": " ".join(sents), "n_pad_tok": npad})
        rows.append({"verb": v, "lemma": d["lemma"], "role": "I", "band": d["band"], "arm": "none", "k": 0, "donor": "",
                     "context": "", "n_pad_tok": 0})
    Tp = P.set_index("t_lemma")
    for lemma, f in TF.iterrows():
        part, band = Tp.loc[lemma, "t_participle"], Tp.loc[lemma, "band"]
        fo = forms(lemma, part)
        order = verb_order(part)
        trans = [f"{subj(f.subj_class, i, False, f.subj_override)} {fo['past']} {f.obj} {ADJ[i]}." for i in order]
        fill = match_fillers(T, trans)
        for k in KS:
            ctx = {"trans": trans[:k], "fill_t": fill[:k]}
            target = max(sum(T.n(s) for s in c) for c in ctx.values())
            for arm, sents in ctx.items():
                sents, npad = pad_to(T, sents, target)
                rows.append({"verb": part, "lemma": lemma, "role": "T", "band": band, "arm": arm, "k": k, "donor": "",
                             "context": " ".join(sents), "n_pad_tok": npad})
        rows.append({"verb": part, "lemma": lemma, "role": "T", "band": band, "arm": "none", "k": 0, "donor": "",
                     "context": "", "n_pad_tok": 0})
    C = pd.DataFrame(rows)
    C.insert(0, "cid", [f"{v}|{a}|{k}" for v, a, k in zip(C.verb, C.arm, C.k)])
    assert C.cid.is_unique
    lemma_forms = {**{v: set(d["forms"].values()) for v, d in I.items()},
                   **{Tp.loc[l, "t_participle"]: set(forms(l, Tp.loc[l, "t_participle"]).values()) for l in TF.index}}
    n_viol = 0
    dist = []
    for r in C.itertuples(index=False):
        words = re.findall(r"[A-Za-z]+", r.context.lower())
        assert not probe_nouns & set(words), (r.cid, probe_nouns & set(words))
        fs = lemma_forms[r.verb]
        if r.arm != "ing":
            n_viol += sum(1 for a, b in zip(words, words[1:]) if b in fs and a in BE_HAVE)
        # tokens from the last mention of the probed participle to the end of the context
        part_ids = tok.encode(" " + r.verb, add_special_tokens=False)
        ids = tok.encode(r.context, add_special_tokens=False) if r.context else []
        last = max((i for i in range(len(ids) - len(part_ids) + 1) if ids[i:i + len(part_ids)] == part_ids), default=None)
        dist.append(len(ids) - last if last is not None else -1)
    assert n_viol == 0, f"{n_viol} be/have-form + target-form adjacencies"
    C["n_tok_ctx"] = [len(tok.encode(c, add_special_tokens=False)) if c else 0 for c in C.context]
    C["dist_last_mention"] = dist
    spread = C[C.k > 0].groupby(["verb", "k", C.arm.isin(["trans", "fill_t"])]).n_tok_ctx.agg(lambda x: x.max() - x.min())
    assert spread.max() <= 2, spread.max()
    smp = C[C.k > 0].sample(400, random_state=args.seed)
    for r in smp.itertuples(index=False):
        for pfx in ("The man was", "The window has been"):
            joined = tok.encode(f"{r.context} {pfx} {r.verb}", add_special_tokens=False)
            parts = (tok.encode(r.context, add_special_tokens=False) + tok.encode(" " + pfx, add_special_tokens=False)
                     + tok.encode(" " + r.verb, add_special_tokens=False))
            assert joined == parts, (r.cid, pfx)
    out = Path(args.data_dir)
    C.to_csv(out / "contexts.csv", index=False)
    k8 = C[C.k == 8]
    meta = {"contexts": len(C), "by_arm": C.arm.value_counts().to_dict(), "seed": args.seed,
            "length_spread_within_verb_k_max": int(spread.max()),
            "mean_pad_tokens_k8": k8.groupby("arm").n_pad_tok.mean().round(2).to_dict(),
            "dist_last_mention_k8_mean": k8[k8.dist_last_mention >= 0].groupby("arm").dist_last_mention.mean().round(2).to_dict(),
            "inputs": {p: sha(p) for p in (args.pairs, args.prefixes, str(out / "i_frames.csv"), str(out / "t_frames.csv"))},
            "outputs": {"contexts.csv": sha(out / "contexts.csv")}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="data/round5/verbs_expanded/pairs.csv")
    ap.add_argument("--prefixes", default="data/round6/frame_d11/prefixes.csv")
    ap.add_argument("--data-dir", default="data/round6/preempt")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--seed", type=int, default=17)
    main(ap.parse_args())
