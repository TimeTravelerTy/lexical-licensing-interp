#!/usr/bin/env python3
"""Round 5, B7 analysis: D11's switch neurons on Pile-10k text (reports/round5/plan.md, B7). No model.

Input: results/round5/wild/{wild_tokens.parquet, wild_meta.json}; data/round5/frame/verbs.csv (A1's 962 participles).

Mode `describe`: distributions, top-100 contexts (raw; signed and one-per-document sensitivities), surface proxies
against the baselines with document-resampling CIs (top list recomputed per draw), and the blind annotation items.
Outputs: reports/round5/b7_wild.md; results/round5/wild/{b7_top_contexts.csv, b7_summary.csv,
b7_annotation_items.csv (item, context only), b7_annotation_key.csv (item -> neuron; not shown to annotators)}.

Mode `decide` (after both frozen label files b7_annotation_{claude,codex}.csv exist): Cohen's kappa, per-neuron rates
and the declared readings; appended to the report; b7_decisions.json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BE = {"am", "is", "are", "was", "were", "be", "been", "being", "get", "gets", "got", "gotten", "getting"}
WH = {"what", "which", "who", "whom", "whose", "that", "where", "when", "why", "how"}
BY_ID = 407  # " by" (Pythia tokenizer)
CATS = ("verbal passive participle", "adjectival / copular participle", "active perfect or simple past",
        "object gap after the verb", "other", "uncertain")
PROX = ("participle", "be_get3", "gap_proxy", "by_next")
PROX_NAME = {"participle": "(a) participle", "be_get3": "(b) be/get in 3", "gap_proxy": "(c) wh/rel in 8",
             "by_next": "(d) \" by\" next"}


def proxies(D, participles):
    """(a)-(c) need the word a token ends (the run saved preceding words only at word-final tokens); other tokens fail them."""
    w = D.word.str.lower()
    prev = D.prev_words.str.lower().str.split()
    part = D.ends_word & (w.str.endswith("ed") | w.str.endswith("en") | w.isin(participles))
    be3 = D.ends_word & prev.map(lambda p: any(x in BE for x in p[-3:]))
    wh8 = D.ends_word & prev.map(lambda p: any(x in WH for x in p[-8:])) & ~be3
    return pd.DataFrame({"participle": part, "be_get3": be3, "gap_proxy": wh8, "by_next": D.next_id.eq(BY_ID)})


class Contexts:
    def __init__(self, D):
        self.g = {d: g.set_index("pos") for d, g in D[["doc", "pos", "tok", "start", "end"]].groupby("doc")}

    def __call__(self, doc, pos, before=24, after=4):
        g = self.g[doc]
        s = g.loc[max(0, pos - before):pos + after]
        text = ""
        for p, r in s.iterrows():
            if p == pos:
                left = text
            text += r.tok
            if p == pos:
                mid_end = len(text)
        # mark the whole word the token belongs to (pieces of the same word are glued to the token on the left)
        m = re.search(r"[A-Za-z'’]*$", left) if re.match(r"[A-Za-z]", text[len(left):mid_end]) else None
        a = m.start() if m else len(left)
        return (text[:a] + "[[" + text[a:mid_end] + "]]" + text[mid_end:]).replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ⏎ ")


def top_order(sc, D, k=100):
    return np.lexsort((D.pos.to_numpy(), D.doc.to_numpy(), -sc))[:k]


def boot_top(sc, doc_codes, n_docs, P, rng, n_boot, k=100, pool=20000):
    """Exact resampled top-k proxy means: documents drawn with replacement; the resampled top k lies in the global top
    `pool` whenever the drawn copies of that pool number >= k (asserted)."""
    o = np.argsort(-sc, kind="stable")[:pool]
    dc, Pm = doc_codes[o], P[o].astype(float)
    out = np.empty((n_boot, P.shape[1]))
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n_docs, n_docs), minlength=n_docs)[dc]
        cw = np.cumsum(w)
        assert cw[-1] >= k
        j = np.searchsorted(cw, k)  # first index where cumulative copies reach k
        ww = w[:j + 1].astype(float)
        ww[-1] -= cw[j] - k
        out[b] = ww @ Pm[:j + 1] / k
    return out


def describe(args):
    D = pd.read_parquet(args.tokens)
    meta = json.loads(Path(args.meta).read_text())
    neurons = meta["neurons"]
    participles = set(pd.read_csv(args.verbs, keep_default_na=False).verb.str.lower())
    assert len(participles) == 962, len(participles)
    Px = proxies(D, participles)
    P = Px[list(PROX)].to_numpy()
    cand, wf = Px.participle.to_numpy(), D.ends_word.to_numpy()
    doc_codes, docs = pd.factorize(D.doc)
    ctx = Contexts(D)
    rng = np.random.default_rng(args.seed)
    res = Path(args.res_dir)

    lines = ["# Round 5, B7: switch neurons on natural text (Pile-10k)", "",
             f"Spec: `plan.md`, B7. Run: `run_neurons_wild.py` (seed 17; {len(docs)} documents × ≤ 512 tokens = "
             f"{meta['tokens']:,} tokens; fp32 post-activations). Analysis: `analyze_neurons_wild.py describe` "
             f"(and `decide` for the annotation). Proxies are surface patterns, not grammatical classes: (a) the token ends "
             "a word in -ed/-en or one of A1's 962 participles; (b) a be/get-form among the 3 preceding words; (c) a "
             "wh-word or relativizer among the 8 preceding words and no be/get-form among the 3; (d) the next token is "
             "\" by\" (id 407). (a)–(c) are defined on word-final tokens (the run saved preceding words there only); "
             "other tokens fail them. CIs: 1,000 document resamples with the top 100 recomputed in each.", "",
             "## Activation distributions", "",
             "| rank | neuron | s_n | mean | SD | share > 0 | p99 | p99.9 | max | min |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, n in enumerate(neurons):
        a = D[f"a{k}"].to_numpy()
        lines.append(f"| {n['rank']} | {n['neuron']} | {int(np.sign(n['I_A'])):+d} | {a.mean():+.3f} | {a.std():.3f} | "
                     f"{(a > 0).mean():.3f} | {np.percentile(a, 99):.2f} | {np.percentile(a, 99.9):.2f} | {a.max():.2f} | "
                     f"{a.min():.2f} |")
    base = {"all tokens": Px.mean(), "word-final tokens": Px[wf].mean(), "participle candidates (a)": Px[cand].mean()}
    lines += ["", "**Baselines** (share of tokens meeting each proxy):", "",
              "| baseline | n | " + " | ".join(PROX_NAME[c] for c in PROX) + " |", "|---|---:|" + "---:|" * len(PROX)]
    for name, b in base.items():
        nn = {"all tokens": len(D), "word-final tokens": int(wf.sum()), "participle candidates (a)": int(cand.sum())}[name]
        lines.append(f"| {name} | {nn:,} | " + " | ".join(f"{b[c]:.3f}" for c in PROX) + " |")

    tops, rows = [], []
    lines += ["", "## Top 100 contexts by raw activation", "",
              "Each cell: share of the top 100 [95% CI]. Docs / words = unique documents / unique word forms in the top 100.",
              "", "| rank | neuron | docs | words | " + " | ".join(PROX_NAME[c] for c in PROX) + " | most frequent words |",
              "|---:|---|---:|---:|" + "---|" * (len(PROX) + 1)]
    sens = []
    for k, n in enumerate(neurons):
        a = D[f"a{k}"].to_numpy()
        s_n = int(np.sign(n["I_A"]))
        for mode in ("raw", "signed", "dedup", "dedup_signed"):
            sc = s_n * a if mode.endswith("signed") else a
            if mode.startswith("dedup"):
                best = pd.Series(sc).groupby(D.doc.to_numpy()).idxmax().to_numpy()
                order = best[np.lexsort((D.pos.to_numpy()[best], D.doc.to_numpy()[best], -sc[best]))][:100]
            else:
                order = top_order(sc, D)
            T, pr = D.iloc[order], Px.iloc[order]
            for i in range(len(order)):
                r = T.iloc[i]
                tops.append({"rank": n["rank"], "neuron": n["neuron"], "mode": mode, "k": i + 1, "doc": int(r.doc),
                             "pos": int(r.pos), "act": float(r[f"a{k}"]), "tok": r.tok.replace("\r", "␍").replace("\n", "⏎"), "word": r.word,
                             "context": ctx(r.doc, r.pos), **pr.iloc[i].to_dict()})
            if mode == "raw":
                bs = boot_top(a, doc_codes, len(docs), P, rng, args.n_boot)
                cells = []
                for j, c in enumerate(PROX):
                    lo, hi = np.percentile(bs[:, j], [2.5, 97.5])
                    cells.append(f"{pr[c].mean():.2f} [{lo:.2f}, {hi:.2f}]")
                    rows.append({"rank": n["rank"], "neuron": n["neuron"], "proxy": c, "top100": pr[c].mean(), "lo": lo,
                                 "hi": hi, **{f"base_{b}": base[b][c] for b in base}})
                tw = T.word.str.lower().where(T.word.ne(""), "«" + T.tok.str.strip() + "»").value_counts().head(6)
                lines.append(f"| {n['rank']} | {n['neuron']} | {T.doc.nunique()} | {T.word.str.lower().nunique()} | "
                             + " | ".join(cells) + " | " + ", ".join(f"{w} ({c})" for w, c in tw.items()) + " |")
            else:
                sens.append((n, mode, s_n, T, pr))
    TOP = pd.DataFrame(tops)
    TOP.to_csv(res / "b7_top_contexts.csv", index=False)
    pd.DataFrame(rows).to_csv(res / "b7_summary.csv", index=False)

    lines += ["", "**Sensitivities.** Signed ranking s_n · a (identical to raw where s_n = +1; for ranks 9–10 it is the "
                  "*lowest*-activation contexts) and one context per document (each document's maximum, top 100; for ranks 9–10 "
                  "also under the signed ranking):", "",
              "| rank | version | docs | words | " + " | ".join(PROX_NAME[c] for c in PROX) + " | most frequent words |",
              "|---:|---|---:|---:|" + "---:|" * len(PROX) + "---|"]
    for n, mode, s_n, T, pr in sens:
        if mode.endswith("signed") and s_n > 0:
            continue
        label = {"signed": "signed (lowest activation)", "dedup": "one per document",
                 "dedup_signed": "signed, one per document"}[mode]
        tw = T.word.str.lower().where(T.word.ne(""), "«" + T.tok.str.strip() + "»").value_counts().head(5)
        lines.append(f"| {n['rank']} | {label} | {T.doc.nunique()} | {T.word.str.lower().nunique()} | "
                     + " | ".join(f"{pr[c].mean():.2f}" for c in PROX)
                     + " | " + ", ".join(f"{w} ({c})" for w, c in tw.items()) + " |")

    # Examples (top 8 raw contexts per top-3 neuron)
    lines += ["", "## Example contexts (top 8 by raw activation, ranks 1–3)", ""]
    for r in (1, 2, 3):
        g = TOP[(TOP["rank"] == r) & (TOP["mode"] == "raw")].head(8)
        lines += [f"**Rank {r}, {g.neuron.iloc[0]}**", ""]
        lines += [f"- {row.act:.2f}: …{row.context.strip()}…".replace("|", "\\|") for row in g.itertuples()]
        lines.append("")

    ann = TOP[(TOP["mode"] == "raw") & (TOP["rank"] <= 3) & (TOP.k <= 40)].copy()
    ann = ann.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    ann.insert(0, "item", np.arange(len(ann)))
    ann[["item", "rank", "neuron", "k", "doc", "pos", "act", "word", "context"]].to_csv(res / "b7_annotation_key.csv",
                                                                                         index=False)
    ann[["item", "context"]].to_csv(res / "b7_annotation_items.csv", index=False)
    lines += [f"Annotation items: {len(ann)} (top 40 raw contexts of ranks 1–3, pooled and shuffled; the word holding the "
              "token is marked [[…]]; `b7_annotation_items.csv`).", ""]
    Path(args.report).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def decide(args):
    res = Path(args.res_dir)
    key = pd.read_csv(res / "b7_annotation_key.csv")
    for who in ("claude", "codex"):
        a = pd.read_csv(res / f"b7_annotation_{who}.csv")
        assert set(a.label) <= set(CATS), (who, set(a.label) - set(CATS))
        assert sorted(a.item) == sorted(key.item), who
        key = key.merge(a[["item", "label"]].rename(columns={"label": who}), on="item")
    po = (key.claude == key.codex).mean()
    pc = sum((key.claude == c).mean() * (key.codex == c).mean() for c in CATS)
    kappa = (po - pc) / (1 - pc)
    conf = pd.crosstab(key.claude, key.codex)
    lines = ["## Construction annotation (top 40 per top-3 neuron; Claude and Codex, independent)", "",
             f"{len(key)} items. Raw agreement {po:.2f}; Cohen's κ = {kappa:.2f} (6 categories).", "",
             "| rank | neuron | annotator | " + " | ".join(CATS) + " |", "|---:|---|---|" + "---:|" * len(CATS)]
    dec = {"kappa": float(kappa), "agreement": float(po), "neurons": {}}
    for r, g in key.groupby("rank"):
        rates = {}
        for who in ("claude", "codex"):
            share = g[who].value_counts(normalize=True).reindex(CATS, fill_value=0)
            rates[who] = share
            lines.append(f"| {r} | {g.neuron.iloc[0]} | {who} | " + " | ".join(f"{share[c]:.2f}" for c in CATS) + " |")
        vp = {w: s["verbal passive participle"] for w, s in rates.items()}
        gap = {w: s["object gap after the verb"] for w, s in rates.items()}
        if all(v >= 0.5 for v in vp.values()):
            reading = "passive-participle detector"
        elif all(vp[w] + gap[w] >= 0.5 and gap[w] >= 0.15 for w in rates):
            reading = "gap-sensitive participle reader"
        else:
            reading = "other"
        dec["neurons"][g.neuron.iloc[0]] = {"rank": int(r), "reading": reading, "vp": vp, "gap": gap}
    lines += ["", "| rank | neuron | verbal passive (Claude / Codex) | passive + gap | reading |", "|---:|---|---|---|---|"]
    for nrn, d in dec["neurons"].items():
        lines.append(f"| {d['rank']} | {nrn} | {d['vp']['claude']:.2f} / {d['vp']['codex']:.2f} | "
                     f"{d['vp']['claude'] + d['gap']['claude']:.2f} / {d['vp']['codex'] + d['gap']['codex']:.2f} | "
                     f"**{d['reading']}** |")
    lines += ["", "Confusion (rows Claude, columns Codex):", "", "| Claude \\ Codex | " + " | ".join(conf.columns) + " |",
              "|---|" + "---:|" * len(conf.columns)]
    lines += [f"| {i} | " + " | ".join(str(v) for v in r) + " |" for i, r in zip(conf.index, conf.to_numpy())]
    rep = Path(args.report).read_text().split("## Construction annotation")[0].rstrip("\n")
    Path(args.report).write_text(rep + "\n\n" + "\n".join(lines) + "\n")
    (res / "b7_decisions.json").write_text(json.dumps(dec, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("describe", "decide"))
    ap.add_argument("--tokens", default="results/round5/wild/wild_tokens.parquet")
    ap.add_argument("--meta", default="results/round5/wild/wild_meta.json")
    ap.add_argument("--verbs", default="data/round5/frame/verbs.csv")
    ap.add_argument("--res-dir", default="results/round5/wild")
    ap.add_argument("--report", default="reports/round5/b7_wild.md")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    describe(a) if a.mode == "describe" else decide(a)
