#!/usr/bin/env python3
"""Round 5, A5: grammatical / perturbed sentence pairs for the grammaticality probe (reports/round5/plan.md, A5).

Adaptation of Wang et al. 2026 (ACL): Project Gutenberg text (jondurbin/gutenberg-dpo-v0.1, "chosen"; document =
book) and WikiText-103 raw test + validation (document = article; standing in for the Penn Treebank half).
Rule-based sentence splitting; regex tokens (words with contractions, numbers, single punctuation marks); originals
and perturbed sentences are rendered from tokens with one spacing rule, so no perturbation glues words together. Sentences of 6-40 words, ending in . ! ? (optionally followed by a closing quote).
One perturbation per sentence, uniform over insertion (k in 1..5 lowercase common word types from the corpus at random
token boundaries before the terminal punctuation), deletion (k in 1..5 alphabetic tokens) and a shuffle of a
contiguous 5-token window before the terminal punctuation (re-drawn until the order changes, up to 10 tries).
Shortcut controls: sentence-initial casing is restored after perturbation (the new first word capitalized, a moved-in
common sentence-initial word lowercased); originals with unbalanced brackets or quotes, or not starting with a capital,
are dropped; initials ("G.") do not end sentences; a perturbation is re-drawn (up to 10 times) until the perturbed
sentence starts like an original. Up to 25,000 sentences per source
(any WikiText shortfall filled from Gutenberg), seed 17; duplicates removed. Train / dev split 80 / 20 by document
for three split seeds.

Output: data/round5/probe/pairs.csv, audit_sample.txt, meta.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

TOK = re.compile(r"\w+(?:['’]\w+)*(?:[.,]\d+)*|[^\s\w]")
ABBR = ("Mr", "Mrs", "Ms", "Dr", "St", "Jr", "Sr", "Prof", "Gen", "Col", "Capt", "Lt", "Sgt", "Rev", "Mt", "No", "vs",
        "etc", "Co", "Inc", "Ltd", "U.S", "U.K", "i.e", "e.g")
END = re.compile(r"[.!?][\"'”’]?$")


def detok_wikitext(t):
    t = t.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    t = re.sub(r" ([,.;:!?%)\]])", r"\1", t)
    t = re.sub(r"([(\[$]) ", r"\1", t)
    t = re.sub(r" (['’](?:s|re|ve|ll|d|m|t)\b)", r"\1", t)
    t = re.sub(r" n't\b", "n't", t)
    t = re.sub(r'" ([^"]*?) "', r'"\1"', t)
    return re.sub(r"\s+", " ", t).strip()


def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?:(?<=[.!?])|(?<=[.!?][\"'”’]))\s+(?=[\"“‘']?[A-Z])", text)
    out, buf = [], ""
    for p in parts:
        buf = f"{buf} {p}".strip() if buf else p
        last = buf.split()[-1] if buf.split() else ""
        if (any(last.rstrip(".").rstrip("\"'”’").endswith(a) for a in ABBR) and last.endswith(".")) or \
                re.fullmatch(r"\(?[A-Z]\.", last):
            continue  # an abbreviation or an initial, not a sentence end
        if buf.count('"') % 2 == 1 or buf.count("“") > buf.count("”"):
            continue  # keep quotes together
        out.append(buf)
        buf = ""
    if buf:
        out.append(buf)
    return out


def tokens_ws(s):
    """[(token, trailing whitespace)] covering the sentence."""
    out = []
    for m in TOK.finditer(s):
        j = m.end()
        ws = ""
        while j < len(s) and s[j].isspace():
            ws += s[j]
            j += 1
        out.append([m.group(), ws])
    return out


NOSPACE_BEFORE = {",", ".", ";", ":", "!", "?", ")", "]", "}", "”", "’", "%", "-", "—"}
NOSPACE_AFTER = {"(", "[", "{", "“", "‘", "$", "-", "—"}


def render(toks):
    """One spacing convention for originals and perturbed sentences alike (no glued words)."""
    out, dq = "", 0
    for i, t in enumerate(toks):
        if i == 0:
            out = t
        else:
            prev = toks[i - 1]
            space = not (t in NOSPACE_BEFORE or prev in NOSPACE_AFTER or (prev == '"' and dq % 2 == 1)
                         or (t == '"' and dq % 2 == 1))
            out += (" " if space else "") + t
        if t == '"':
            dq += 1
    return out


def join(tw):
    return render([t for t, _ in tw])


def body_len(tw):
    """Number of tokens before the terminal punctuation group (e.g. '.', '!”')."""
    n = len(tw)
    while n > 0 and not tw[n - 1][0][0].isalnum():
        n -= 1
    return n


def fix_case(orig, tw, common):
    """Sentence-initial casing as in the original: capitalize the new first word; lowercase a moved-in common
    sentence-initial word."""
    first_o = orig[0][0]
    for i, (t, _) in enumerate(tw):
        if t[0].isalpha():
            if i == 0 or all(not x[0][0].isalnum() for x in tw[:i]):
                tw[i][0] = t[0].upper() + t[1:]
            break
    for i, (t, _) in enumerate(tw):
        if i > 0 and t == first_o and t[0].isupper() and t != "I" and t.lower() in common and any(x[0][0].isalnum() for x in tw[:i]):
            tw[i][0] = t.lower()
    return tw


def perturb(tw, kind, rng, vocab, common):
    orig = [list(x) for x in tw]
    tw = [list(x) for x in tw]
    nb = body_len(tw)
    if kind == "insertion":
        k = int(rng.integers(1, 6))
        for _ in range(k):
            pos = int(rng.integers(0, nb + 1))  # boundaries before the terminal punctuation
            tw.insert(pos, [vocab[int(rng.integers(len(vocab)))], " "])
            nb += 1
        return join(fix_case(orig, tw, common)), k
    if kind == "deletion":
        k = int(rng.integers(1, 6))
        alpha = [i for i, (t, _) in enumerate(tw) if t.isalpha()]
        if len(alpha) < k + 3:
            return None, k
        drop = set(rng.choice(alpha, k, replace=False).tolist())
        return join(fix_case(orig, [x for i, x in enumerate(tw) if i not in drop], common)), k
    if nb < 5:
        return None, 5
    st = int(rng.integers(0, nb - 4))  # the window stays before the terminal punctuation
    win = [tw[i][0] for i in range(st, st + 5)]
    for _ in range(10):
        perm = list(rng.permutation(5))
        if perm != list(range(5)) and [win[p] for p in perm] != win:
            for j, p in enumerate(perm):
                tw[st + j][0] = win[p]
            return join(fix_case(orig, tw, common)), 5
    return None, 5


def main(args):
    rng = np.random.default_rng(args.seed)
    g = pd.read_parquet(args.gutenberg)
    sents = []
    for src, text in zip(g.source, g.chosen):
        for s in split_sentences(text):
            sents.append(("gutenberg", src.replace("gutenberg ", ""), s))
    wt = pd.concat([pd.read_parquet(p) for p in args.wikitext], ignore_index=True).text.tolist()
    art, title = [], None
    for line in wt:
        m = re.match(r"^ = ([^=].*?) = \n?$", line)
        if m:
            title = m.group(1).strip()
            continue
        if line.strip() and not line.strip().startswith("=") and title:
            for s in split_sentences(detok_wikitext(line)):
                sents.append(("wikitext", title, s))
    df = pd.DataFrame(sents, columns=["source", "doc", "sentence"]).drop_duplicates("sentence")
    nw = df.sentence.str.split().str.len()
    df = df[(nw >= 6) & (nw <= 40) & df.sentence.str.contains(END) & ~df.sentence.str.contains(r"[=<>@|\[\]{}]")]
    df = df[df.sentence.str.match(r"[\"“‘']?[A-Z]")]
    bal = lambda x: (x.count("(") == x.count(")") and x.count("[") == x.count("]") and x.count('"') % 2 == 0
                     and x.count("“") == x.count("”"))
    df = df[df.sentence.map(bal)].reset_index(drop=True)
    low, cap = {}, {}
    for s_ in df.sentence:
        for i, (t, _) in enumerate(tokens_ws(s_)):
            if t.isalpha() and i > 0:
                d_ = low if t.islower() else cap
                d_[t.lower()] = d_.get(t.lower(), 0) + 1
    common = {w for w, c in low.items() if c > cap.get(w, 0)}  # usually lowercase mid-sentence
    vocab = sorted(common)  # inserted words: lowercase common word types
    wsel = df[df.source == "wikitext"].sample(frac=1, random_state=args.seed).head(args.per_source)
    gneed = args.per_source + (args.per_source - len(wsel))
    gsel = df[df.source == "gutenberg"].sample(frac=1, random_state=args.seed).head(gneed)
    sel = pd.concat([gsel, wsel]).reset_index(drop=True)
    kinds = np.array(["insertion", "deletion", "shuffle"])[rng.integers(0, 3, len(sel))]
    rows = []
    for (src, doc, s), kind in zip(sel[["source", "doc", "sentence"]].itertuples(index=False), kinds):
        tw = tokens_ws(s)
        orig = join(tw)  # the original, rendered with the same spacing rules
        for _ in range(10):  # re-draw until the perturbed sentence starts like an original (no punctuation-initial cue)
            p, k = perturb(tw, kind, rng, vocab, common)
            if p is not None and p != orig and re.match(r"[\"“‘']?[A-Z0-9]", p):
                break
        else:
            continue
        rows.append({"source": src, "doc": doc, "original": orig, "perturbed": p, "ptype": kind, "k": k})
    P = pd.DataFrame(rows)
    P = P[~P.perturbed.isin(set(P.original))].reset_index(drop=True)
    P.insert(0, "pair_id", np.arange(len(P)))
    for sd in range(3):  # 80 / 20 by document within each source
        r = np.random.default_rng(args.seed + sd)
        dev_docs = set()
        for src, sub in P.groupby("source"):
            docs = sub.doc.value_counts()
            order = r.permutation(docs.index.to_numpy())
            acc = 0
            for d in order:
                if acc >= 0.2 * len(sub):
                    break
                dev_docs.add((src, d))
                acc += docs[d]
        P[f"split{sd}"] = ["dev" if (s_, d_) in dev_docs else "train" for s_, d_ in zip(P.source, P.doc)]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    P.to_csv(out / "pairs.csv", index=False)
    audit = P.sample(200, random_state=args.seed)
    (out / "audit_sample.txt").write_text("\n".join(f"[{r.source} | {r.ptype}] {r.original}\n    -> {r.perturbed}"
                                                    for r in audit.itertuples()) + "\n")
    meta = {"inputs": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [args.gutenberg] + args.wikitext},
            "n_pairs": len(P), "by_source": P.source.value_counts().to_dict(), "by_ptype": P.ptype.value_counts().to_dict(),
            "dev_share": {f"split{sd}": float((P[f"split{sd}"] == "dev").mean()) for sd in range(3)},
            "n_docs": P.groupby("source").doc.nunique().to_dict(), "vocab": len(vocab),
            "candidates": df.source.value_counts().to_dict(),
            "pairs_sha256": hashlib.sha256((out / "pairs.csv").read_bytes()).hexdigest()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gutenberg", required=True)
    ap.add_argument("--wikitext", nargs="+", required=True)
    ap.add_argument("--out-dir", default="data/round5/probe")
    ap.add_argument("--per-source", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=17)
    main(ap.parse_args())
