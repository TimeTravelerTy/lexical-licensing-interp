#!/usr/bin/env python3
"""DAS stability, extended (rules: reports/das_stability_ext_rules.md; recompute bar: Amendment 2 of
reports/pythia/das_expanded_stability_rules.md).

Generalizes run_das_expanded_stability.py to any model and site set. For each site and each stored (split, fold):
  R      stored pairs, stored seed (10000 + 100*split + fold)
  S1-S5  stored pairs, seed + 5000, 6000, 7000, 8000, 9000 (seed band; S1 = the Nov 18 S)
  P      fold's stored training pairs + the new plain pairs of the Nov 18 batch (always in training)
  P2, P3 P with seed + 5000, 6000 (P's own seed band; training only)
  L      stored pairs trained for P's number of updates (epochs x P / stored training items, per fold; training only)
Writes {C}_site{s}/ laid out like a round-2 final (bases.pt r1_s{k}_f{f}, items.csv, heldout_swaps.csv.gz,
pairs_folds.csv) so run_passive_test.py runs unchanged; plus cosines.csv, active_summary.csv, checks.json, meta.json.
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

S_OFFSETS = {"S1": 5000, "S2": 6000, "S3": 7000, "S4": 8000, "S5": 9000}
P_OFFSETS = {"P": 0, "P2": 5000, "P3": 6000}


def plain_pairs_csv(stored_pairs, expanded, intrans_class, out, batch="nov18"):
    S = pd.read_csv(stored_pairs)
    X = pd.read_csv(expanded)
    cls = pd.read_csv(intrans_class).set_index("lemma").das_class
    X = X[(X.batch == batch) & ~X.das_train_overlap]
    X = X[X.intrans_lemma.map(cls) == "plain"].copy()
    stored_verbs = set(S.trans) | set(S.intrans)
    assert not (X.trans_lemma.isin(stored_verbs) | X.intrans_lemma.isin(stored_verbs)).any()
    new = pd.DataFrame({"source": "expanded", "tok": np.where(X.participle_ntok == 1, "single", "multi"),
                        "trans": X.trans_lemma, "intrans": X.intrans_lemma, "trans_tag": X.trans_tag,
                        "intrans_tag": X.intrans_tag, "trans_part": X.trans_participle,
                        "intrans_part": X.intrans_participle})
    C = pd.concat([S, new], ignore_index=True)
    assert C.trans.is_unique and C.intrans.is_unique
    C.to_csv(out, index=False)
    return len(S), len(new)


def filtered(runner, pairs_csv, stored_ids):
    import torch

    pairsX, itemsX = d2.load_items(pairs_csv)
    m, parts, repsX = runner.natural(itemsX.prompt.tolist())
    itemsX["M_nat"] = m.cpu().numpy()
    itemsX["correct"] = np.where(itemsX.cls == 1, itemsX.M_nat > 0, itemsX.M_nat < 0)
    vp = itemsX.groupby("verb").correct.sum() >= 4
    keep = set(pairsX[pairsX.trans.map(vp) & pairsX.intrans.map(vp)].pair_id)
    dropped = sorted(set(pairsX.pair_id) - keep)
    keep_new = keep - set(pairsX[pairsX.source != "expanded"].pair_id)
    keep = keep_new | set(stored_ids)  # stored pairs: exactly the stored final's kept set
    sel = itemsX.pair_id.isin(keep).to_numpy()
    itemsX = itemsX[sel].reset_index(drop=True)
    repsX = repsX[torch.tensor(sel, device=repsX.device)]
    nat = {"M": itemsX.M_nat.to_numpy(), **{k: v.cpu().numpy()[sel] for k, v in parts.items()}}
    return itemsX, repsX, nat, dropped, len(keep_new)


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
    for s in sites:
        for k in ("lr", "batch_size", "n_cross", "n_same", "dtype", "n_folds", "n_splits", "model", "pairs"):
            assert fm[s]["args"][k] == sa[k], (s, k)
        assert fm[s]["chosen_rank"] == 1 and fm[s]["site"] == s, s
    assert Path(sa["model"]).name == Path(a.model).name or sa["model"] == a.model, (sa["model"], a.model)
    stored_ref = json.loads(Path(a.stored_passive_meta).read_text())["checks"]
    n_old, n_new = plain_pairs_csv(a.stored_pairs, a.expanded, a.intrans_class, out / "pairs_P.csv")
    log(f"pairs: {n_old} stored + {n_new} new plain (P)")
    conds = ["R", *S_OFFSETS, *P_OFFSETS, "L"] if not a.conds else a.conds.split(",")

    def args_for(pairs):
        return SimpleNamespace(pairs=str(pairs), model=a.model, device=a.device, dtype=sa["dtype"], lr=sa["lr"],
                               batch_size=sa["batch_size"], n_cross=sa["n_cross"], n_same=sa["n_same"],
                               bos=sa.get("bos"), fp32_unembed=sa.get("fp32_unembed"))

    runner, pairsS, itemsS, repsS, natS, droppedS = d2.setup(args_for(a.stored_pairs))
    expect = sorted(fm[sites[0]].get("dropped_pairs", []))
    assert sorted(droppedS) == expect, f"stored drops {droppedS} differ from the stored final's {expect}"
    checks = {"runner_settings": runner.settings, "dropped_stored": droppedS}
    itemsP, repsP, natP, droppedP, n_keep = filtered(runner, str(out / "pairs_P.csv"), pairsS.pair_id)
    checks["dropped_P_new"] = [p for p in droppedP if p not in set(pairsS.pair_id) | set(droppedS)]
    checks["n_new_pairs_kept_P"] = n_keep
    log(f"P: behavior filter dropped {checks['dropped_P_new']}, kept {n_keep} new")

    cos_rows, act_rows = [], []
    pos = pd.Series(np.arange(len(itemsS)), index=itemsS.prompt)
    for s in sites:
        R0 = site_dirs[s]
        st_items = pd.read_csv(R0 / "items.csv")
        assert (st_items.prompt.to_numpy() == itemsS.prompt.to_numpy()).all(), "stored items differ"
        checks[f"stored_items_equal_site{s}"] = True
        folds_df = pd.read_csv(R0 / "pairs_folds.csv")
        stored_b = {k: v.float().to(runner.device) for k, v in torch.load(R0 / "bases.pt").items() if k.startswith("r1_")}
        st_hs = pd.read_csv(R0 / "heldout_swaps.csv.gz", usecols=["base", "src", "M_patched", "rank", "split", "fold"])
        st_hs = st_hs[st_hs["rank"] == 1]
        epochs = fm[s]["epochs"]
        new_b = {c: {} for c in conds}
        details = {c: [] for c in conds}
        recompute = []
        for k in range(sa["n_splits"]):
            assign = dict(zip(folds_df.pair_id, folds_df[f"fold_split{k}"]))
            assert assign == d2.folds(pairsS, sa["n_folds"], k), "stored fold assignment does not reproduce"
            for f in range(sa["n_folds"]):
                if a.smoke and (k, f) != (0, 0):
                    continue
                key = f"r1_s{k}_f{f}"
                v = stored_b[key]
                trS, teS = d2.fold_indices(itemsS, assign, f)
                trP, teP = d2.fold_indices(itemsP, assign, f)
                assert (itemsP.prompt.values[teP] == itemsS.prompt.values[teS]).all()
                dv = d2.evaluate(runner, itemsS, repsS, natS, teS, s, v)
                g = st_hs[(st_hs.split == k) & (st_hs.fold == f)]
                mg = dv.merge(g, on=["base", "src"], suffixes=("", "_stored"))
                assert len(mg) == len(g) == len(dv)
                recompute.append(np.abs(mg.M_patched - mg.M_patched_stored).to_numpy())
                seed = 10000 + 100 * k + f
                for c in conds:
                    ep = epochs
                    if c in P_OFFSETS:
                        items, reps, nat, tr, te, sd = itemsP, repsP, natP, trP, teP, seed + P_OFFSETS[c]
                    else:
                        items, reps, nat, tr, te = itemsS, repsS, natS, trS, teS
                        sd = seed + S_OFFSETS.get(c, 0)
                        if c == "L":
                            ep = int(round(epochs * len(trP) / len(trS)))
                    b, _ = d2.train(runner, items, reps, tr, items.cls.to_numpy(), s, 1, ep, args_for(""), sd)
                    df = d2.evaluate(runner, items, reps, nat, te, s, b, detail=True)
                    df["rank"], df["split"], df["fold"] = 1, k, f
                    if c in P_OFFSETS:  # re-index held-out swaps to the stored item order (prompts are unique)
                        df["base"] = pos[items.prompt.values[df.base]].to_numpy()
                        df["src"] = pos[items.prompt.values[df.src]].to_numpy()
                    details[c].append(df)
                    new_b[c][key] = b.cpu()
                    act_rows.append({"site": s, "cond": c, "split": k, "fold": f, "n_train_items": len(tr), "epochs": ep,
                                     **d2.summarize(df, itemsS)})
                    cos_rows.append({"site": s, "cond": c, "split": k, "fold": f,
                                     "abs_cos": float(abs(b[:, 0].float() @ v[:, 0]))})
                act_rows.append({"site": s, "cond": "stored", "split": k, "fold": f, "n_train_items": len(trS),
                                 **d2.summarize(dv, itemsS)})
                log(f"site {s} split {k} fold {f}: |cos| " +
                    " ".join(f"{r['cond']} {r['abs_cos']:.3f}" for r in cos_rows[-len(conds):]))
        rc = np.concatenate(recompute)
        ref = stored_ref.get(str(s), {})
        med_bar = max(1e-3, 1.5 * ref.get("heldout_recompute_median_abs_dM", 0.0))
        checks[f"stored_recompute_site{s}"] = {"max": float(rc.max()), "median": float(np.median(rc)),
                                               "median_bar": med_bar,
                                               "pass": bool(np.median(rc) <= med_bar and rc.max() <= 0.5)}
        for c in conds:
            d = out / f"{c}_site{s}"
            d.mkdir(exist_ok=True)
            torch.save(new_b[c], d / "bases.pt")
            itemsS.drop(columns=[x for x in ("M_nat", "correct") if x in itemsS]).to_csv(d / "items.csv", index=False)
            pd.concat(details[c], ignore_index=True).to_csv(d / "heldout_swaps.csv.gz", index=False)
            folds_df.to_csv(d / "pairs_folds.csv", index=False)
    C = pd.DataFrame(cos_rows)
    C.to_csv(out / "cosines.csv", index=False)
    pd.DataFrame(act_rows).to_csv(out / "active_summary.csv", index=False)
    for s in sites:
        checks[f"R_median_abs_cos_site{s}"] = float(C[(C.site == s) & (C.cond == "R")].abs_cos.median())
    (out / "checks.json").write_text(json.dumps(checks, indent=2, default=str) + "\n")
    meta = {"args": vars(a), "stored_args": sa, "epochs": {s: fm[s]["epochs"] for s in sites}, "conds": conds,
            "n_items_stored": len(itemsS), "n_items_P": len(itemsP), "minutes": (time.time() - t0) / 60}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps(checks, indent=2, default=str))
    print(C.groupby(["site", "cond"]).abs_cos.describe()[["min", "50%", "max"]].round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site-dirs", required=True, help="site=stored final dir,...")
    ap.add_argument("--stored-pairs", default="data/das_round2/train_pairs.csv")
    ap.add_argument("--stored-passive-meta", required=True, help="stored passive run_meta.json (recompute bar)")
    ap.add_argument("--expanded", default="data/verb_expansion/expanded_pairs.csv")
    ap.add_argument("--intrans-class", default="data/verb_expansion/expanded_intrans_class.csv")
    ap.add_argument("--conds", default="", help="subset of R,S1..S5,P,P2,P3,L (default all)")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smoke", action="store_true", help="split 0 fold 0 only (no passive test possible)")
    run(ap.parse_args())
