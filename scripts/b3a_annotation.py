#!/usr/bin/env python3
"""B3a annotation of core ranks 4-5 (reports/v2_controls_mech_b_rules.md): R5-B7's protocol on two more neurons.

Mode `items`: the top 40 raw Pile-10k contexts of MLP11.n4925 and MLP14.n4241 from the stored
results/round5/wild/b7_top_contexts.csv, pooled and shuffled (seed 17) -> b3a_annotation_items.csv (item, context) and
b3a_annotation_key.csv (item -> neuron; not shown to annotators).
Mode `decide` (after b3a_annotation_{claude,codex}.csv exist): Cohen's kappa, per-neuron rates and B7's reading rule
(as analyze_neurons_wild.decide) -> b3a_decisions.json, b3a_annotation.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_neurons_wild import CATS

NEURONS = ("MLP11.n4925", "MLP14.n4241")


def items(a):
    res = Path(a.res_dir)
    res.mkdir(parents=True, exist_ok=True)
    T = pd.read_csv(a.top)
    ann = T[(T["mode"] == "raw") & T.neuron.isin(NEURONS) & (T.k <= 40)].copy()
    assert ann.groupby("neuron").size().to_dict() == {n: 40 for n in NEURONS}
    ann = ann.sample(frac=1, random_state=a.seed).reset_index(drop=True)
    ann.insert(0, "item", np.arange(len(ann)))
    ann[["item", "rank", "neuron", "k", "doc", "pos", "act", "word", "context"]].to_csv(res / "b3a_annotation_key.csv", index=False)
    ann[["item", "context"]].to_csv(res / "b3a_annotation_items.csv", index=False)
    print(f"{len(ann)} items -> {res / 'b3a_annotation_items.csv'}")


def decide(a):
    res = Path(a.res_dir)
    key = pd.read_csv(res / "b3a_annotation_key.csv")
    for who in ("claude", "codex"):
        x = pd.read_csv(res / f"b3a_annotation_{who}.csv")
        assert set(x.label) <= set(CATS), (who, set(x.label) - set(CATS))
        assert sorted(x.item) == sorted(key.item), who
        key = key.merge(x[["item", "label"]].rename(columns={"label": who}), on="item")
    po = (key.claude == key.codex).mean()
    pc = sum((key.claude == c).mean() * (key.codex == c).mean() for c in CATS)
    kappa = (po - pc) / (1 - pc) if pc < 1 else 1.0
    dec = {"kappa": float(kappa), "agreement": float(po), "neurons": {}}
    L = ["# B3a: construction annotation of core ranks 4–5 (R5-B7 protocol; Claude and Codex, independent)", "",
         f"{len(key)} items. Raw agreement {po:.2f}; Cohen's κ = {kappa:.2f}.", "",
         "| neuron | annotator | " + " | ".join(CATS) + " |", "|---|---|" + "---:|" * len(CATS)]
    for n, g in key.groupby("neuron"):
        rates = {}
        for who in ("claude", "codex"):
            share = g[who].value_counts(normalize=True).reindex(CATS, fill_value=0)
            rates[who] = share
            L.append(f"| {n} | {who} | " + " | ".join(f"{share[c]:.2f}" for c in CATS) + " |")
        vp = {w: float(s["verbal passive participle"]) for w, s in rates.items()}
        gap = {w: float(s["object gap after the verb"]) for w, s in rates.items()}
        if all(v >= 0.5 for v in vp.values()):
            reading = "passive-participle detector"
        elif all(vp[w] + gap[w] >= 0.5 and gap[w] >= 0.15 for w in rates):
            reading = "gap-sensitive participle reader"
        else:
            reading = "other"
        dec["neurons"][n] = {"reading": reading, "vp": vp, "gap": gap}
    L += ["", "| neuron | verbal passive (Claude / Codex) | object gap (Claude / Codex) | reading |", "|---|---|---|---|"]
    for n, d in dec["neurons"].items():
        L.append(f"| {n} | {d['vp']['claude']:.2f} / {d['vp']['codex']:.2f} | {d['gap']['claude']:.2f} / {d['gap']['codex']:.2f} | **{d['reading']}** |")
    conf = pd.crosstab(key.claude, key.codex)
    L += ["", "Confusion (rows Claude, columns Codex):", "", "| Claude \\ Codex | " + " | ".join(conf.columns) + " |",
          "|---|" + "---:|" * len(conf.columns)]
    L += [f"| {i} | " + " | ".join(str(v) for v in r) + " |" for i, r in zip(conf.index, conf.to_numpy())]
    (res / "b3a_decisions.json").write_text(json.dumps(dec, indent=1) + "\n")
    Path(a.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("items", "decide"))
    ap.add_argument("--top", default="results/round5/wild/b7_top_contexts.csv")
    ap.add_argument("--res-dir", default="results/pythia/das_v2/mech_b")
    ap.add_argument("--report", default="reports/pythia/v2_mech/b3a_annotation.md")
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    items(a) if a.mode == "items" else decide(a)
