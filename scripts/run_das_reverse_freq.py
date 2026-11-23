#!/usr/bin/env python3
"""Reverse frequency transfer: rank-1 DAS trained only on rare plain pairs (rules: reports/pythia/das_reverse_freq_rules.md).

Training pairs: plain pairs of the Nov 18 expansion batch whose intransitive is tagged xtail / gap / out (43), none
sharing a verb with v's training pairs. Conditions F (behavior-filtered, as v), U (unfiltered) and B (filtered, both
sides rare: the transitive also xtail / gap / out). Three seeds
(20000 + k); seed k's basis is written under r1_s{k}_f{0..4}, so run_passive_test.py runs the stored plan unchanged
(every donor is a frequent stored pair, none in training).

Per site and condition writes {C}_site{s}/ (bases.pt, items.csv = stored items, heldout_swaps.csv.gz = the rare basis
on each stored fold's held-out swaps, pairs_folds.csv) and v_site{s}_swaps.csv.gz (the stored v on the same swaps,
this pipeline); top level: pairs_rare.csv, filter.csv (per verb), cosines.csv, active_summary.csv, checks.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import run_das_round2 as d2

RARE_TAGS = ("xtail", "gap", "out")
SEED0 = 20000


def rare_pairs_csv(a, out):
    S = pd.read_csv(a.stored_pairs)
    X = pd.read_csv(a.expanded)
    cls = pd.read_csv(a.intrans_class).set_index("lemma").das_class
    X = X[(X.batch == "nov18") & ~X.das_train_overlap & X.intrans_tag.isin(RARE_TAGS)]
    X = X[X.intrans_lemma.map(cls) == "plain"].copy()
    stored = set(S.trans) | set(S.intrans)
    assert not (X.trans_lemma.isin(stored) | X.intrans_lemma.isin(stored)).any(), "rare pair shares a verb with v's"
    R = pd.DataFrame({"source": "rare", "tok": np.where(X.participle_ntok == 1, "single", "multi"),
                      "trans": X.trans_lemma, "intrans": X.intrans_lemma, "trans_tag": X.trans_tag,
                      "intrans_tag": X.intrans_tag, "trans_part": X.trans_participle,
                      "intrans_part": X.intrans_participle})
    assert R.trans.is_unique and R.intrans.is_unique
    R.to_csv(out, index=False)
    return R


def run(a):
    import torch

    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    site_dirs = {int(k): Path(v) for k, v in (x.split("=") for x in a.site_dirs.split(","))}
    sites = sorted(site_dirs)
    fm = {s: json.loads((site_dirs[s] / "final_meta.json").read_text()) for s in sites}
    sa = fm[sites[0]]["args"]
    stored_ref = json.loads(Path(a.stored_passive_meta).read_text())["checks"]

    def args_for(pairs):
        return SimpleNamespace(pairs=str(pairs), model=a.model, device=a.device, dtype=sa["dtype"], lr=sa["lr"],
                               batch_size=sa["batch_size"], n_cross=sa["n_cross"], n_same=sa["n_same"],
                               bos=sa.get("bos"), fp32_unembed=sa.get("fp32_unembed"))

    R = rare_pairs_csv(a, out / "pairs_rare.csv")
    plan = pd.read_csv(a.plan, usecols=["donor_pair"]).donor_pair.dropna().astype(str).unique()
    donors = {v for p in plan for v in p.split("/")}
    assert not ((set(R.trans) | set(R.intrans)) & donors), "a rare training verb is a passive-plan donor"
    runner, pairsS, itemsS, repsS, natS, droppedS = d2.setup(args_for(a.stored_pairs))
    assert droppedS == sorted(fm[sites[0]].get("dropped_pairs", []))

    pairsR, itemsR = d2.load_items(str(out / "pairs_rare.csv"))
    m, parts, repsR = runner.natural(itemsR.prompt.tolist())
    itemsR["M_nat"] = m.cpu().numpy()
    itemsR["correct"] = np.where(itemsR.cls == 1, itemsR.M_nat > 0, itemsR.M_nat < 0)
    vf = itemsR.groupby(["verb", "cls", "pair_id"]).agg(n_correct=("correct", "sum"), M_mean=("M_nat", "mean")).reset_index()
    vf["pass"] = vf.n_correct >= 4
    vf.to_csv(out / "filter.csv", index=False)
    vp = vf.set_index("verb")["pass"]
    keepF = set(pairsR[pairsR.trans.map(vp) & pairsR.intrans.map(vp)].pair_id)
    checks = {"runner_settings": runner.settings, "n_rare_pairs": len(pairsR), "n_kept_F": len(keepF),
              "dropped_F": sorted(set(pairsR.pair_id) - keepF),
              "failing_verbs": vf[~vf["pass"]][["verb", "cls", "n_correct"]].to_dict("records")}
    log(f"rare pairs {len(pairsR)}; filter keeps {len(keepF)}; failing verbs {checks['failing_verbs']}")
    both = set(pairsR[pairsR.trans_tag.isin(RARE_TAGS)].pair_id)
    checks["n_both_rare"], checks["n_kept_B"] = len(both), len(both & keepF)
    conds = {"F": itemsR.pair_id.isin(keepF).to_numpy(), "U": np.ones(len(itemsR), bool),
             "B": itemsR.pair_id.isin(keepF & both).to_numpy()}
    for c, sel in conds.items():
        P_c = pairsR[pairsR.pair_id.isin(set(itemsR.pair_id[sel]))]
        P_c.to_csv(out / f"pairs_{c}.csv", index=False)

    cos_rows, act_rows = [], []
    for s in sites:
        R0 = site_dirs[s]
        st_items = pd.read_csv(R0 / "items.csv")
        assert (st_items.prompt.to_numpy() == itemsS.prompt.to_numpy()).all(), "stored items differ"
        folds_df = pd.read_csv(R0 / "pairs_folds.csv")
        stored_b = {k: v.float().to(runner.device) for k, v in torch.load(R0 / "bases.pt").items() if k.startswith("r1_")}
        st_hs = pd.read_csv(R0 / "heldout_swaps.csv.gz", usecols=["base", "src", "M_patched", "rank", "split", "fold"])
        st_hs = st_hs[st_hs["rank"] == 1]
        epochs = fm[s]["epochs"]
        # stored v on each fold's held-out swaps, this pipeline (paired reference + recompute check)
        v_det, recompute = [], []
        for k in range(sa["n_splits"]):
            assign = dict(zip(folds_df.pair_id, folds_df[f"fold_split{k}"]))
            for f in range(sa["n_folds"]):
                _, teS = d2.fold_indices(itemsS, assign, f)
                dv = d2.evaluate(runner, itemsS, repsS, natS, teS, s, stored_b[f"r1_s{k}_f{f}"], detail=True)
                g = st_hs[(st_hs.split == k) & (st_hs.fold == f)]
                mg = dv.merge(g, on=["base", "src"], suffixes=("", "_stored"))
                assert len(mg) == len(g) == len(dv)
                recompute.append(np.abs(mg.M_patched - mg.M_patched_stored).to_numpy())
                dv["rank"], dv["split"], dv["fold"] = 1, k, f
                v_det.append(dv)
                act_rows.append({"site": s, "cond": "v", "split": k, "fold": f, **d2.summarize(dv, itemsS)})
        pd.concat(v_det, ignore_index=True).to_csv(out / f"v_site{s}_swaps.csv.gz", index=False)
        rc = np.concatenate(recompute)
        bar = max(1e-3, 1.5 * stored_ref.get(str(s), {}).get("heldout_recompute_median_abs_dM", 0.0))
        checks[f"stored_recompute_site{s}"] = {"max": float(rc.max()), "median": float(np.median(rc)),
                                               "median_bar": bar, "pass": bool(np.median(rc) <= bar and rc.max() <= 0.5)}
        for c, sel in conds.items():
            idx = np.flatnonzero(sel & itemsR.subject.isin(d2.TRAIN_SUBJECTS).to_numpy())
            bases, det = {}, []
            for k in range(sa["n_splits"]):
                b, _ = d2.train(runner, itemsR, repsR, idx, itemsR.cls.to_numpy(), s, 1, epochs, args_for(""), SEED0 + k)
                assign = dict(zip(folds_df.pair_id, folds_df[f"fold_split{k}"]))
                for f in range(sa["n_folds"]):
                    _, teS = d2.fold_indices(itemsS, assign, f)
                    df = d2.evaluate(runner, itemsS, repsS, natS, teS, s, b, detail=True)
                    df["rank"], df["split"], df["fold"] = 1, k, f
                    det.append(df)
                    bases[f"r1_s{k}_f{f}"] = b.cpu()
                    act_rows.append({"site": s, "cond": c, "split": k, "fold": f, "n_train_items": len(idx),
                                     **d2.summarize(df, itemsS)})
                for key, v in stored_b.items():
                    cos_rows.append({"site": s, "cond": c, "seed": k, "v_key": key,
                                     "abs_cos": float(abs(b[:, 0].float() @ v[:, 0]))})
                log(f"site {s} {c} seed {k}: median |cos| with v "
                    f"{np.median([r['abs_cos'] for r in cos_rows[-len(stored_b):]]):.3f}")
            d = out / f"{c}_site{s}"
            d.mkdir(exist_ok=True)
            torch.save(bases, d / "bases.pt")
            itemsS.drop(columns=[x for x in ("M_nat", "correct") if x in itemsS]).to_csv(d / "items.csv", index=False)
            pd.concat(det, ignore_index=True).to_csv(d / "heldout_swaps.csv.gz", index=False)
            folds_df.to_csv(d / "pairs_folds.csv", index=False)
    pd.DataFrame(cos_rows).to_csv(out / "cosines.csv", index=False)
    pd.DataFrame(act_rows).to_csv(out / "active_summary.csv", index=False)
    (out / "checks.json").write_text(json.dumps(checks, indent=2, default=str) + "\n")
    meta = {"args": vars(a), "stored_args": sa, "epochs": {s: fm[s]["epochs"] for s in sites},
            "minutes": (time.time() - t0) / 60}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps(checks, indent=2, default=str))
    A = pd.DataFrame(act_rows)
    print(A.groupby(["site", "cond"])[["iia_cross", "pc_frac_median"]].mean().round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site-dirs", default="8=results/das_round2/final_strict_site8,"
                    "12=results/das_round2/final_strict_site12,17=results/das_round2/final_strict")
    ap.add_argument("--stored-pairs", default="data/das_round2/train_pairs.csv")
    ap.add_argument("--stored-passive-meta", default="results/das_round2/passive_test/run_meta.json")
    ap.add_argument("--plan", default="data/das_round2/passive_test/plan.csv.gz")
    ap.add_argument("--expanded", default="data/verb_expansion/expanded_pairs.csv")
    ap.add_argument("--intrans-class", default="data/verb_expansion/expanded_intrans_class.csv")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="results/pythia/das_reverse_freq")
    run(ap.parse_args())
