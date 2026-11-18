#!/usr/bin/env python3
"""DAS stability under more training verbs (rules: reports/pythia/das_expanded_stability_rules.md).

For each site and each stored (split, fold) of the round-2 DAS final, train a rank-1 basis in three conditions:
  R  stored 29 pairs, stored seed            (reproduction)
  S  stored 29 pairs, seed + 5000            (seed baseline)
  E  stored pairs of the fold's training set + all new expanded pairs (always in training), stored seed
  P  as E, new pairs restricted to plain intransitives (the round-2 class rule; primary)
with the stored recipe (run_das_round2.train / evaluate). Writes, per condition and site, a directory laid out like a
round-2 final (bases.pt with keys r1_s{split}_f{fold}, items.csv, heldout_swaps.csv.gz, pairs_folds.csv = the stored
assignment of the 29 pairs), so run_passive_test.py can use it unchanged. Also cosines.csv, active_summary.csv,
checks.json and meta.json at the top level.
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

SEED_OFFSET_S = 5000
CONDS = "RSEP"


def expanded_pairs_csv(stored_pairs, expanded, out, plain_only=None):
    S = pd.read_csv(stored_pairs)
    X = pd.read_csv(expanded)
    X = X[~X.das_train_overlap].copy()
    if plain_only is not None:  # round-2 class rule: plain intransitives only
        X = X[X.intrans_lemma.map(plain_only) == "plain"].copy()
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


def run(a):
    import torch

    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sites = [int(s) for s in a.sites.split(",")]
    ref = {s: Path(a.stored_root) / f"final_strict_site{s}" for s in sites}
    fm = {s: json.loads((ref[s] / "final_meta.json").read_text()) for s in sites}
    sa = fm[sites[0]]["args"]
    for s in sites:
        for k in ("lr", "batch_size", "n_cross", "n_same", "dtype", "n_folds", "n_splits"):
            assert fm[s]["args"][k] == sa[k], (s, k)
        assert fm[s]["chosen_rank"] == 1
    cls = pd.read_csv(a.intrans_class).set_index("lemma").das_class
    n_old, n_new = expanded_pairs_csv(a.stored_pairs, a.expanded, out / "pairs_E.csv")
    _, n_plain = expanded_pairs_csv(a.stored_pairs, a.expanded, out / "pairs_P.csv", plain_only=cls)
    log(f"pairs: {n_old} stored + {n_new} new (E), {n_plain} new plain (P)")

    def args_for(pairs):
        return SimpleNamespace(pairs=str(pairs), model=a.model, device=a.device, dtype=sa["dtype"], lr=sa["lr"],
                               batch_size=sa["batch_size"], n_cross=sa["n_cross"], n_same=sa["n_same"],
                               bos=None, fp32_unembed=None)

    # stored-pair setup (R, S) and expanded setups (E all, P plain); one model load
    argsS = args_for(a.stored_pairs)
    runner, pairsS, itemsS, repsS, natS, droppedS = d2.setup(argsS)
    assert not droppedS, "a stored pair failed the behavior filter"
    checks = {"dropped_stored": droppedS}
    X = {}
    for c in "EP":
        pairsX, itemsX = d2.load_items(str(out / f"pairs_{c}.csv"))
        m, parts, repsX = runner.natural(itemsX.prompt.tolist())
        itemsX["M_nat"] = m.cpu().numpy()
        itemsX["correct"] = np.where(itemsX.cls == 1, itemsX.M_nat > 0, itemsX.M_nat < 0)
        vp = itemsX.groupby("verb").correct.sum() >= 4
        keep = set(pairsX[pairsX.trans.map(vp) & pairsX.intrans.map(vp)].pair_id)
        assert set(pairsS.pair_id) <= keep
        sel = itemsX.pair_id.isin(keep).to_numpy()
        itemsX = itemsX[sel].reset_index(drop=True)
        repsX = repsX[torch.tensor(sel, device=repsX.device)]
        natX = {"M": itemsX.M_nat.to_numpy(), **{k: v.cpu().numpy()[sel] for k, v in parts.items()}}
        X[c] = (itemsX, repsX, natX)
        checks[f"dropped_{c}"] = sorted(set(pairsX.pair_id) - keep)
        checks[f"n_new_pairs_kept_{c}"] = int(len(keep) - len(pairsS))
        log(f"{c}: behavior filter dropped {len(checks[f'dropped_{c}'])}, kept {checks[f'n_new_pairs_kept_{c}']} new")

    cos_rows, act_rows = [], []
    for s in sites:
        R0 = ref[s]
        st_items = pd.read_csv(R0 / "items.csv")
        assert (st_items.prompt.to_numpy() == itemsS.prompt.to_numpy()).all(), "stored items differ"
        checks[f"stored_items_equal_site{s}"] = True
        folds_df = pd.read_csv(R0 / "pairs_folds.csv")
        stored_b = {k: v.float().to(runner.device) for k, v in torch.load(R0 / "bases.pt").items() if k.startswith("r1_")}
        st_hs = pd.read_csv(R0 / "heldout_swaps.csv.gz", usecols=["base", "src", "M_patched", "rank", "split", "fold"])
        st_hs = st_hs[st_hs["rank"] == 1]
        epochs = fm[s]["epochs"]
        new_b = {c: {} for c in CONDS}
        details = {c: [] for c in CONDS}
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
                trX, teX = {}, {}
                for c in "EP":
                    trX[c], teX[c] = d2.fold_indices(X[c][0], assign, f)
                    assert (X[c][0].prompt.values[teX[c]] == itemsS.prompt.values[teS]).all()
                # stored basis through this pipeline vs the stored held-out swaps
                dv = d2.evaluate(runner, itemsS, repsS, natS, teS, s, v)
                g = st_hs[(st_hs.split == k) & (st_hs.fold == f)]
                mg = dv.merge(g, on=["base", "src"], suffixes=("", "_stored"))
                assert len(mg) == len(g) == len(dv)
                recompute.append(float(np.abs(mg.M_patched - mg.M_patched_stored).max()))
                seed = 10000 * 1 + 100 * k + f
                for c, (items, reps, nat, tr, te, sd) in {
                        "R": (itemsS, repsS, natS, trS, teS, seed),
                        "S": (itemsS, repsS, natS, trS, teS, seed + SEED_OFFSET_S),
                        "E": (*X["E"], trX["E"], teX["E"], seed),
                        "P": (*X["P"], trX["P"], teX["P"], seed)}.items():
                    b, _ = d2.train(runner, items, reps, tr, items.cls.to_numpy(), s, 1, epochs, args_for(""), sd)
                    df = d2.evaluate(runner, items, reps, nat, te, s, b, detail=True)
                    df["rank"], df["split"], df["fold"] = 1, k, f
                    if c in "EP":  # re-index held-out swaps to the stored item order (prompts are unique)
                        pos = pd.Series(np.arange(len(itemsS)), index=itemsS.prompt)
                        df["base"] = pos[items.prompt.values[df.base]].to_numpy()
                        df["src"] = pos[items.prompt.values[df.src]].to_numpy()
                    details[c].append(df)
                    new_b[c][key] = b.cpu()
                    act_rows.append({"site": s, "cond": c, "split": k, "fold": f, "n_train_items": len(tr),
                                     **d2.summarize(df, itemsS)})
                    cos_rows.append({"site": s, "cond": c, "split": k, "fold": f,
                                     "abs_cos": float(abs(b[:, 0].float() @ v[:, 0]))})
                act_rows.append({"site": s, "cond": "stored", "split": k, "fold": f, "n_train_items": len(trS),
                                 **d2.summarize(dv, itemsS)})
                log(f"site {s} split {k} fold {f}: |cos| " +
                    " ".join(f"{r['cond']} {r['abs_cos']:.3f}" for r in cos_rows[-len(CONDS):]))
        checks[f"stored_heldout_recompute_max_abs_site{s}"] = max(recompute)
        for c in CONDS:
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
    (out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    meta = {"args": vars(a), "stored_args": sa, "epochs": {s: fm[s]["epochs"] for s in sites},
            "n_items_stored": len(itemsS), "n_items": {c: len(X[c][0]) for c in "EP"}, "minutes": (time.time() - t0) / 60}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps(checks, indent=2))
    print(C.groupby(["site", "cond"]).abs_cos.describe()[["min", "50%", "max"]].round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stored-root", default="../results/das_round2")
    ap.add_argument("--stored-pairs", default="../data/das_round2/train_pairs.csv")
    ap.add_argument("--expanded", default="../data/verb_expansion/expanded_pairs.csv")
    ap.add_argument("--intrans-class", default="../data/verb_expansion/expanded_intrans_class.csv")
    ap.add_argument("--sites", default="8,12")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="../results/pythia/das_expanded_stability")
    ap.add_argument("--smoke", action="store_true", help="split 0 fold 0 only (no passive test possible)")
    run(ap.parse_args())
