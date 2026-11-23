#!/usr/bin/env python3
"""Cross-fitted LEACE on the expanded eval set (rules: reports/expanded_eval_rules.md, part D).

Reads run_leace_necessity.py --extra-pairs output. Share of the natural *by* gap remaining (erased gap / natural gap),
with the stored estimator's two-way cluster bootstrap (analyze_das_prep.Boot: pairs within verb band x contexts within
context band; the same draws for every condition), for original / new / pooled pairs, each as all classes, plain
(primary) and prep_object (separate). Checks: the original items' natural log P(by) equals the stored necessity run's.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_das_prep import Boot

CONDS_MAIN = ("scrub_0_12", "scrub_0_8", "scrub_all", "single_8", "single_12")


def ci(d):
    d0, rest = d[0], d[1:][np.isfinite(d[1:])]
    return float(d0), float(np.percentile(rest, 2.5)), float(np.percentile(rest, 97.5))


def run(a):
    res = Path(a.results)
    P = pd.read_csv(res / "items.csv.gz")
    meta = pd.read_csv(a.pairs_meta).set_index("pair_id")
    cls = pd.read_csv(a.classes)
    cls = cls[cls.source == "original"].set_index("lemma")["class"]
    G, B = P[P.side == "good"].set_index("item"), P[P.side == "bad"].set_index("item")
    assert (G.index == B.index).all()
    it = G[["verb_pair", "verb_band", "context_id", "context_band"]].reset_index()
    it["source"] = np.where(it.verb_pair.str.startswith("new/"), "new", "original")
    pc = {}
    for vp in it.verb_pair.unique():
        pc[vp] = meta.bad_class.get(vp) if vp.startswith("new/") else cls.get(vp.split("/")[-1])
    it["bad_class"] = it.verb_pair.map(pc)
    assert it.bad_class.notna().all(), it[it.bad_class.isna()].verb_pair.unique()
    conds = [c[len("lp_by__"):] for c in P.columns if c.startswith("lp_by__")]
    for c in conds:
        it[f"gap__{c}"] = (G[f"lp_by__{c}"] - B[f"lp_by__{c}"]).to_numpy()
    boot = Boot(it, a.n_boot, np.random.default_rng(a.seed))
    checks = {}
    if a.stored and Path(a.stored).exists():  # natural readout of the original items vs the stored necessity run
        S = pd.read_csv(a.stored, usecols=["prompt", "lp_by__natural"]).set_index("prompt").lp_by__natural
        o = P[~P.verb_pair.str.startswith("new/")]
        checks["original_natural_vs_stored_max_abs"] = float((o.lp_by__natural.to_numpy()
                                                              - S.reindex(o.prompt).to_numpy()).__abs__().max())
        checks["n_original_prompts"] = int(len(o))
        # (a) same items, different fitting: stored run vs this run's original subset, one set of draws (paired)
        Q = pd.read_csv(a.stored)
        Gq, Bq = Q[Q.side == "good"].set_index("item"), Q[Q.side == "bad"].set_index("item")
        Go, Bo = G.loc[Gq.index], B.loc[Bq.index]  # original items keep their indices in the expanded run
        assert (Go.prompt.to_numpy() == Gq.prompt.to_numpy()).all() and (Bo.prompt.to_numpy() == Bq.prompt.to_numpy()).all()
        iq = Gq[["verb_pair", "verb_band", "context_id", "context_band"]].reset_index()
        bq = Boot(iq, a.n_boot, np.random.default_rng(a.seed))
        plain_q = iq.verb_pair.map(lambda vp: cls.get(vp.split("/")[-1])).eq("plain").to_numpy()
        comp = {}
        for sub, mk in (("all", np.ones(len(iq), bool)), ("plain", plain_q)):
            for c in CONDS_MAIN + ("scrub_0_23",):
                cq = "scrub_all" if c == "scrub_0_23" else c
                if f"lp_by__{cq}" not in Q.columns or f"lp_by__{cq}" not in P.columns:
                    continue
                sh_st = bq.means((Gq[f"lp_by__{cq}"] - Bq[f"lp_by__{cq}"]).to_numpy(), mk) / \
                    bq.means((Gq.lp_by__natural - Bq.lp_by__natural).to_numpy(), mk)
                sh_ex = bq.means((Go[f"lp_by__{cq}"] - Bo[f"lp_by__{cq}"]).to_numpy(), mk) / \
                    bq.means((Go.lp_by__natural - Bo.lp_by__natural).to_numpy(), mk)
                comp[f"{sub} {cq}"] = {"stored": ci(sh_st), "expanded_fit": ci(sh_ex), "difference": ci(sh_ex - sh_st)}
        checks["stored_scrub_0_12_share_recomputed"] = comp["all scrub_0_12"]["stored"]
        checks["comparison_a"] = comp
    rc = json.loads((res / "checks.json").read_text())
    checks["run_checks"] = rc
    masks = {}
    for src in ("original", "new", "pooled"):
        ms = np.ones(len(it), bool) if src == "pooled" else (it.source == src).to_numpy()
        for c_name, mc in (("all", np.ones(len(it), bool)), ("plain", (it.bad_class == "plain").to_numpy()),
                           ("prep", (it.bad_class == "prep_object").to_numpy())):
            masks[f"{src} {c_name}"] = ms & mc
    rows = []
    for m, mk in masks.items():
        nat = boot.means(it["gap__natural"].to_numpy(), mk)
        for c in conds:
            g = boot.means(it[f"gap__{c}"].to_numpy(), mk)
            e, lo, hi = ci(g / nat)
            ge, glo, ghi = ci(g)
            rows.append({"subset": m, "condition": c, "n_pairs": int(it.verb_pair[mk].nunique()), "n_items": int(mk.sum()),
                         "gap": ge, "gap_lo": glo, "gap_hi": ghi, "share_remaining": e, "share_lo": lo, "share_hi": hi})
    S = pd.DataFrame(rows)
    S.to_csv(res / "expanded_summary.csv", index=False)
    (res / "expanded_checks.json").write_text(json.dumps(checks, indent=1, default=float) + "\n")
    f3 = lambda r: f"{r.share_remaining:.2f} [{r.share_lo:.2f}, {r.share_hi:.2f}]"
    L = [f"# Cross-fitted LEACE on the expanded eval set ({a.model_name})", "",
         "Rules: `reports/expanded_eval_rules.md` (D). Scripts: `scripts/run_leace_necessity.py --extra-pairs`, "
         "`scripts/analyze_expanded_leace.py`. Erasers cross-fitted over all pairs (5 verb-level folds); share of the "
         "natural *by* gap remaining, 95% CIs (pairs within verb band x contexts).", "",
         f"Stored 126-pair reference (Settled 1): scrub 0–12 leaves {a.stored_ref}.", "",
         "| subset | pairs | natural gap | " + " | ".join(CONDS_MAIN) + " |", "|---|---:|---|" + "---|" * len(CONDS_MAIN)]
    for m in masks:
        q = S[S.subset == m].set_index("condition")
        nat = q.loc["natural"]
        L.append(f"| {m} | {nat.n_pairs} | {nat.gap:.2f} [{nat.gap_lo:.2f}, {nat.gap_hi:.2f}] | "
                 + " | ".join(f3(q.loc[c]) if c in q.index else "—" for c in CONDS_MAIN) + " |")
    pooled = S[(S.subset == "pooled all") & (S.condition == "scrub_0_12")].iloc[0]
    ref = float(a.stored_ref.split()[0])
    verdict = ("lower" if pooled.share_hi < ref else "higher" if pooled.share_lo > ref
               else "compatible with the stored point estimate")
    if "comparison_a" in checks:
        L += ["", "## (a) Same original items: stored 126-pair fit vs the expanded fit (paired)", "",
              "| subset / condition | stored fit | expanded fit | difference |", "|---|---|---|---|"]
        g = lambda t: f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"
        for k, v in checks["comparison_a"].items():
            L.append(f"| {k} | {g(v['stored'])} | {g(v['expanded_fit'])} | {g(v['difference'])} |")
    L += ["", f"**Declared comparison 2:** pooled (all classes, as the stored 126-pair number) scrub 0–12 leaves "
          f"{f3(pooled)}, against the stored {ref:.2f}: **{verdict}**.", "", "## Checks", "", "```",
          json.dumps(checks, indent=1, default=float), "```", ""]
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/pythia/expanded_eval/leace_cf")
    ap.add_argument("--pairs-meta", default="data/verb_expansion/eval/pairs_meta.csv")
    ap.add_argument("--classes", default="data/das_round2/intrans_classes_final.csv")
    ap.add_argument("--stored", default="/gs/fs/tga-sip_arase/tyrone/lli_leace/results/pythia/leace_necessity_v2/items.csv.gz")
    ap.add_argument("--stored-ref", default="0.59 [0.35, 0.70]")
    ap.add_argument("--model-name", default="Pythia-1.4B")
    ap.add_argument("--report", default="reports/pythia/expanded_eval_leace.md")
    ap.add_argument("--n-boot", type=int, default=1000, help="the stored run used 1000")
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
