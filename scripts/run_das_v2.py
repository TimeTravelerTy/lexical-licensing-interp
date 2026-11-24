#!/usr/bin/env python3
"""Train v2, the replacement DAS direction (rules: reports/v2_retrain_rules.md).

v2 is trained at every site that has a stored round-2 final:
  training pairs = the fold's stored training pairs (stored fold assignment of the 29 stored pairs)
                   + the 44 new plain pairs of data/das_v2/train_pairs_v2.csv (fixed file, sha256 in meta.json;
                   pairs whose verbs occur in an evaluation file are already excluded) that pass this model's
                   behavior filter (always in training; none is evaluated as held out);
  seed 10000 + 100*split + fold, the stored epochs, lr, batch size, swaps, dtype, BOS and unembedding settings of
  that site's stored final; rank 1; 3 splits x 5 folds.
Held-out active swaps are the stored fold's held-out items, evaluated in the stored item order with the stored
natural readouts, so items.csv, pairs_folds.csv and every downstream plan
stay byte-identical to the stored final. Each output directory is laid out like a round-2 final (final_meta.json,
bases.pt r1_s{k}_f{f}, items.csv, pairs_folds.csv, heldout_swaps.csv.gz, summary.csv with the random, norm-matched
and shuffled-label controls), so every downstream script runs on the v2 root unchanged.

Checks (checks_{sites}.json; a failed hard check stops the run):
  pairs file sha256 equals data/das_v2/meta.json (hard);
  stored items equal, stored folds reproduce (hard);
  stored held-out recompute along the stored v, patched M and natural M_base / M_src: median |d| <= 0.01, max <= 0.5 (hard);
  disjoint: no lemma or participle of a new training pair is a cell value of any --eval-files file (hard; required);
  layout: 15 keys, one-to-one finite held-out swaps per (split, fold) (hard);
  descriptive: |cos| with the stored v and, where --p-ref is given, with stability-ext P.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import run_das_round2 as d2
from run_das_stability_ext import filtered

RUNNER_KEYS = ("lr", "batch_size", "n_cross", "n_same", "dtype", "n_folds", "n_splits", "n_random", "model", "pairs")


def disjoint_check(new_pairs, eval_csvs):
    forms = set(new_pairs.trans) | set(new_pairs.intrans) | set(new_pairs.trans_part) | set(new_pairs.intrans_part)
    hits = {}
    for f in eval_csvs:
        E = pd.read_csv(f, low_memory=False)
        vals = set()
        for c in E.select_dtypes(exclude="number").columns:
            vals |= set(E[c].dropna().astype(str))
        hits[f] = sorted(forms & vals)
    return hits


def run(a):
    import torch

    t0 = time.time()
    log = lambda m: print(f"[{time.time() - t0:7.0f}s] {m}", flush=True)
    out = Path(a.out_root)
    out.mkdir(parents=True, exist_ok=True)
    site_dirs = {int(k): Path(v) for k, v in (x.split("=") for x in a.site_dirs.split(","))}
    sites = sorted(site_dirs)
    tag = "_".join(map(str, sites)) + ("_smoke" if a.smoke else "")
    fm = {s: json.loads((site_dirs[s] / "final_meta.json").read_text()) for s in sites}
    sa = fm[sites[0]]["args"]
    for s in sites:
        assert fm[s]["site"] == s and fm[s]["chosen_rank"] == 1, s
        for k in RUNNER_KEYS:
            assert fm[s]["args"][k] == sa[k], (s, k, fm[s]["args"][k], sa[k])
        for k in ("bos", "fp32_unembed"):
            assert fm[s]["args"].get(k) == sa.get(k), (s, k)
    assert Path(sa["model"]).name == Path(a.model).name or sa["model"] == a.model, (sa["model"], a.model)
    if a.smoke:
        assert out.name.endswith("_smoke"), "smoke runs write to their own root (*_smoke)"
    import hashlib
    pm = json.loads((Path(a.pairs).parent / "meta.json").read_text())
    assert hashlib.sha256(Path(a.pairs).read_bytes()).hexdigest() == pm["sha256"], "pairs file differs from meta.json"
    eval_files = [x for x in a.eval_files.split(",") if x]
    assert eval_files, "--eval-files is required"

    args_for = lambda pairs: SimpleNamespace(pairs=str(pairs), model=a.model, device=a.device, dtype=sa["dtype"],
                                             lr=sa["lr"], batch_size=sa["batch_size"], n_cross=sa["n_cross"],
                                             n_same=sa["n_same"], bos=sa.get("bos"), fp32_unembed=sa.get("fp32_unembed"))
    runner, pairsS, itemsS, repsS, natS, droppedS = d2.setup(args_for(a.stored_pairs))
    expect = sorted(fm[sites[0]].get("dropped_pairs", []))
    assert sorted(droppedS) == expect, f"stored drops {droppedS} differ from the stored final's {expect}"
    itemsP, repsP, natP, droppedP, n_keep = filtered(runner, a.pairs, pairsS.pair_id)
    allP = pd.read_csv(a.pairs).assign(pair_id=lambda d: d.trans + "/" + d.intrans)
    stored_only = pd.read_csv(a.stored_pairs)
    mine = allP[allP.source != "expanded"][stored_only.columns].reset_index(drop=True)
    assert len(mine) == len(stored_only)
    for c in stored_only.columns:
        x, y = mine[c], stored_only[c]
        same = (np.allclose(pd.to_numeric(x), y, equal_nan=True) if pd.api.types.is_numeric_dtype(y)
                else x.fillna("").astype(str).equals(y.fillna("").astype(str)))
        assert same, f"stored rows of the v2 pairs file differ from the stored pairs in column {c}"
    newP = allP[(allP.source == "expanded") & allP.pair_id.isin(set(itemsP.pair_id))]
    assert len(newP) == n_keep
    newP.to_csv(out / f"new_pairs_trained_{tag}.csv", index=False)
    checks = {"runner_settings": runner.settings, "dropped_stored": droppedS, "n_new_pairs_kept": n_keep,
              "dropped_new": [p for p in droppedP if p not in set(pairsS.pair_id) | set(droppedS)]}
    checks["pairs_sha256"] = pm["sha256"]
    checks["disjoint"] = disjoint_check(allP[allP.source == "expanded"], eval_files)
    assert not any(checks["disjoint"].values()), checks["disjoint"]
    log(f"v2 trains on {n_keep} new plain pairs; disjoint from {len(checks['disjoint'])} eval item files")
    p_ref = {int(k): Path(v) for k, v in (x.split("=") for x in a.p_ref.split(",") if x)}

    labelsP = itemsP.cls.to_numpy()
    for s in sites:
        R0 = site_dirs[s]
        st_items = pd.read_csv(R0 / "items.csv")
        assert (st_items.prompt.to_numpy() == itemsS.prompt.to_numpy()).all(), f"stored items differ at {s}"
        folds_df = pd.read_csv(R0 / "pairs_folds.csv")
        stored_b = {k: v.float().to(runner.device) for k, v in torch.load(R0 / "bases.pt").items() if k.startswith("r1_")}
        st_hs = pd.read_csv(R0 / "heldout_swaps.csv.gz",
                            usecols=["base", "src", "M_patched", "M_base", "M_src", "rank", "split", "fold"])
        st_hs = st_hs[st_hs["rank"] == 1]
        ref_b = torch.load(p_ref[s] / "bases.pt") if s in p_ref else None
        epochs = fm[s]["epochs"]
        bases, shuf_b, details, summ, recompute, nat_diff, cos_v, cos_p = {}, {}, [], [], [], [], [], []
        for k in range(sa["n_splits"]):
            assign = dict(zip(folds_df.pair_id, folds_df[f"fold_split{k}"]))
            assert assign == d2.folds(pairsS, sa["n_folds"], k), f"stored folds do not reproduce at {s}"
            for f in range(sa["n_folds"]):
                if a.smoke and (k, f) != (0, 0):
                    continue
                key = f"r1_s{k}_f{f}"
                trS, teS = d2.fold_indices(itemsS, assign, f)
                trP, teP = d2.fold_indices(itemsP, assign, f)
                assert (itemsP.prompt.values[teP] == itemsS.prompt.values[teS]).all()
                dv = d2.evaluate(runner, itemsS, repsS, natS, teS, s, stored_b[key])
                g = st_hs[(st_hs.split == k) & (st_hs.fold == f)]
                mg = dv.merge(g, on=["base", "src"], suffixes=("", "_stored"))
                assert len(mg) == len(g) == len(dv)
                recompute.append(np.abs(mg.M_patched - mg.M_patched_stored).to_numpy())
                nat_diff.append(np.abs(np.r_[mg.M_base - mg.M_base_stored, mg.M_src - mg.M_src_stored]))
                b, _ = d2.train(runner, itemsP, repsP, trP, labelsP, s, 1, epochs, args_for(""), 10000 + 100 * k + f)
                # held-out swaps on the stored items, stored order and stored natural readouts (= teP's prompts)
                df = d2.evaluate(runner, itemsS, repsS, natS, teS, s, b, detail=True)
                df["rank"], df["split"], df["fold"], df["control"] = 1, k, f, "das"
                details.append(df)
                summ.append({"rank": 1, "split": k, "fold": f, "control": "das", "n_train_items": len(trP),
                             **d2.summarize(df, itemsS)})
                iia = summ[-1]["iia_cross"]
                bases[key] = b.cpu()
                cos_v.append(float(abs(b[:, 0].float() @ stored_b[key][:, 0])))
                if ref_b is not None:
                    cos_p.append(float(abs(b[:, 0].float().cpu() @ ref_b[key][:, 0].float())))
                # controls, as run_das_round2.run_final (held-out stored fold; shuffled labels on v2's training set)
                gen = torch.Generator(device="cpu").manual_seed(900007 + 1000 * (100 * k + f))  # not round 2's seed-7 stream
                for r in range(sa["n_random"] if not a.smoke else 2):
                    rnd = torch.linalg.qr(torch.randn(b.shape, generator=gen))[0].to(runner.device)
                    for name, mn in (("random", None), ("random_normmatched", b)):
                        dr = d2.evaluate(runner, itemsS, repsS, natS, teS, s, rnd, match_norm=mn)
                        summ.append({"rank": 1, "split": k, "fold": f, "control": name, "draw": r, **d2.summarize(dr, itemsS)})
                vcls = itemsP.groupby("verb").cls.first()
                tv = sorted(set(itemsP.verb.values[trP]))
                shuf = dict(zip(tv, np.random.default_rng(100 * k + f).permutation(vcls[tv].to_numpy())))
                perm = np.array([shuf.get(v, c) for v, c in zip(itemsP.verb, labelsP)])
                sb, _ = d2.train(runner, itemsP, repsP, trP, perm, s, 1, epochs, args_for(""), 777 + 100 * k + f)
                ds = d2.evaluate(runner, itemsS, repsS, natS, teS, s, sb)
                summ.append({"rank": 1, "split": k, "fold": f, "control": "shuffled_labels", **d2.summarize(ds, itemsS)})
                shuf_b[key] = sb.cpu()
                log(f"site {s} split {k} fold {f}: iia_cross {iia:.3f} "
                    f"|cos v| {cos_v[-1]:.3f}" + (f" |cos P| {cos_p[-1]:.4f}" if cos_p else ""))
        rc, nd = np.concatenate(recompute), np.concatenate(nat_diff)
        checks[f"site{s}"] = {"recompute_median": float(np.median(rc)), "recompute_max": float(rc.max()),
                              "natural_M_median": float(np.median(nd)), "natural_M_max": float(nd.max()),
                              "recompute_pass": bool(np.median(rc) <= 0.01 and rc.max() <= 0.5
                                                     and np.median(nd) <= 0.01 and nd.max() <= 0.5),
                              "abs_cos_stored_v": [min(cos_v), float(np.median(cos_v)), max(cos_v)],
                              "abs_cos_P_ref": [min(cos_p), float(np.median(cos_p))] if cos_p else None,
                              }
        (out / f"checks_{tag}.json").write_text(json.dumps(checks, indent=2, default=str) + "\n")
        assert checks[f"site{s}"]["recompute_pass"], (s, checks[f"site{s}"])
        det = pd.concat(details, ignore_index=True)
        want = [(k, f) for k in range(sa["n_splits"]) for f in range(sa["n_folds"]) if not a.smoke or (k, f) == (0, 0)]
        assert sorted(bases) == sorted(f"r1_s{k}_f{f}" for k, f in want), sorted(bases)
        for (k, f), g in det.groupby(["split", "fold"]):
            assert not g.duplicated(["base", "src"]).any() and np.isfinite(g.M_patched).all(), (s, k, f)
        assert sorted(det.groupby(["split", "fold"]).groups) == want
        name = "final_strict" if s == a.primary else f"final_strict_site{s}"
        import os
        d = out / f".{name}.{os.getpid()}.tmp"
        d.mkdir()
        torch.save(bases, d / "bases.pt")
        torch.save(shuf_b, d / "shuffled_bases.pt")
        for col in ("verb", "pair_id", "subject", "source", "tok"):
            det[f"base_{col}"] = itemsS[col].values[det.base]
            det[f"src_{col}"] = itemsS[col].values[det.src]
        for kk in ("O", "I", "det", "pron", "refl"):
            det[f"lp_{kk}_base"] = natS[kk][det.base]
            det[f"lp_{kk}_src"] = natS[kk][det.src]
        det.to_csv(d / "heldout_swaps.csv.gz", index=False)
        pd.DataFrame(summ).to_csv(d / "summary.csv", index=False)
        shutil.copy(R0 / "items.csv", d / "items.csv")
        shutil.copy(R0 / "pairs_folds.csv", d / "pairs_folds.csv")
        S_ = pd.DataFrame(summ)
        das = S_[S_.control == "das"]
        meta = {"site": s, "epochs": epochs, "ranks_trained": [1], "chosen_rank": 1, "rank_rule": "rank 1 (v2)",
                "pc_frac_by_rank": {1: float(das.pc_frac_median.mean())}, "pairs": int(itemsP.pair_id.nunique()),
                "dropped_pairs": droppedS, "items": len(itemsS), "config": fm[s].get("config"),
                "args": {**sa, "pairs": a.pairs, "out_dir": str(out / name), "mode": "v2"},
                "v2": {"recipe": "stored fold training pairs + data/das_v2 new plain pairs, always in training",
                       "pairs_sha256": pm["sha256"], "n_new_pairs": n_keep, "rules": "reports/v2_retrain_rules.md",
                       "smoke": bool(a.smoke), "runner_settings": runner.settings},
                "stored_final": {"dir": str(R0), "meta": fm[s]}}
        (d / "final_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        final = out / name
        assert not final.exists() or a.overwrite, f"{final} exists (pass --overwrite)"
        old = out / f".{name}.{os.getpid()}.old"
        if final.exists():
            final.rename(old)
        d.rename(final)
        shutil.rmtree(old, ignore_errors=True)
        log(f"site {s} written to {final}: {checks[f'site{s}']}")
    (out / f"meta_{tag}.json").write_text(json.dumps(
        {"args": vars(a), "stored_args": sa, "minutes": (time.time() - t0) / 60}, indent=2, default=str) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site-dirs", required=True, help="site=stored final dir,...")
    ap.add_argument("--primary", type=int, required=True, help="site written to final_strict/")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--p-ref", default="", help="site=stability_ext P_site dir,... (descriptive |cos|)")
    ap.add_argument("--eval-files", required=True, help="comma list of evaluation CSVs for the disjointness check")
    ap.add_argument("--pairs", default="data/das_v2/train_pairs_v2.csv")
    ap.add_argument("--stored-pairs", default="data/das_round2/train_pairs.csv")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="split 0 fold 0 only, 2 random draws")
    run(ap.parse_args())
