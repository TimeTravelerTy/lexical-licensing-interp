#!/usr/bin/env python3
"""Gemma replication: held-out DAS IIA split by the participles' token counts (does a length cue drive IIA?).

For each site, retrains the rank-1 direction exactly as the sweep did (split 0, 5 folds, seed 1000 * site + fold,
same swaps) for the sweep's best epoch at that site (mean held-out cross-class IIA over folds), then evaluates
every held-out swap. Because the sweep evaluated the same training run after each epoch, the per-fold IIA here
must equal the sweep's row at (site, fold, epoch); with --assert-sweep this is required to 1e-12 (same environment);
otherwise the difference is reported (Pythia: the stored sweeps were run in October).

Tasks (the sweep's own arguments are read from its sweep_meta.json, so items, filters and folds are identical):
- active:  run_das_round2 (perfect-frame actives, object-vs-other readout M);
- passive: run_passive_das (curated was-passives, *by*-vs-rest readout minus the fold threshold).

Token pattern of a cross-class swap: T{n}-I{m}, n / m = the model's token count of the transitive / intransitive
participle (" V", no special tokens) among the swap's two verbs, binned 1 / 2+. T1-I1 swaps carry no length cue.

Writes swaps.parquet (one row per held-out swap), check.json and by_pattern.csv to --out-dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def best_epochs(sweep):
    g = sweep.groupby(["site", "epoch"]).iia_cross.mean().reset_index()
    return g.loc[g.groupby("site").iia_cross.idxmax()].set_index("site").epoch.astype(int).to_dict()


def main(cli):
    import run_das_round2 as r2

    meta = json.loads((Path(cli.sweep_dir) / "sweep_meta.json").read_text())
    args = argparse.Namespace(**meta["args"])
    args.device = cli.device
    sweep = pd.read_csv(Path(cli.sweep_dir) / "sweep.csv")
    ep = best_epochs(sweep)
    sites = [int(s) for s in cli.sites.split(",")] if cli.sites else sorted(ep)
    out = Path(cli.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if cli.task == "active":
        runner, pairs, items, reps, nat, dropped = r2.setup(args)
        assign = r2.folds(pairs, args.n_folds, 0)
        labels = items.cls.to_numpy()

        def fold(f):
            tr, te = r2.fold_indices(items, assign, f)
            return tr, te, nat
    else:
        import run_passive_das as rp

        runner, pf, items, reps, keep, filt, ctx = rp.setup(args)
        labels = items.cls.to_numpy()

        def fold(f):
            tr, te = rp.fold_sets(items, pf, keep, 0, f)
            tau = rp.set_tau(runner, items, tr)
            return tr, te, {"M": items.Mby_nat.to_numpy() - tau}

    part = items.prompt.str.rsplit(" ", n=1).str[1]
    ntok = np.array([len(runner.tok.encode(" " + p, add_special_tokens=False)) for p in part])
    rows, checks = [], []
    for site in sites:
        for f in range(args.n_folds):
            tr, te, nat_f = fold(f)
            basis, _ = r2.train(runner, items, reps, tr, labels, site, 1, ep[site], args, seed=1000 * site + f)
            df = r2.evaluate(runner, items, reps, nat_f, te, site, basis)
            cross = df[df.cross]
            ref = sweep[(sweep.site == site) & (sweep.fold == f) & (sweep.epoch == ep[site])].iia_cross
            assert len(ref) == 1
            checks.append({"site": site, "fold": f, "epoch": ep[site], "iia_cross": float(cross.iia.mean()),
                           "sweep_iia_cross": float(ref.iloc[0])})
            df = df.assign(site=site, fold=f, epoch=ep[site])
            rows.append(df[["site", "fold", "epoch", "base", "src", "base_cls", "src_cls", "cross", "iia",
                            "M_patched", "M_base", "M_src"]])
        print(f"site {site} epoch {ep[site]} done", flush=True)
    S = pd.concat(rows, ignore_index=True)
    t_idx = np.where(S.base_cls == 1, S.base, S.src)
    i_idx = np.where(S.base_cls == 1, S.src, S.base)
    S["t_ntok"], S["i_ntok"] = ntok[t_idx], ntok[i_idx]
    S["base_verb"], S["src_verb"] = items.verb.values[S.base], items.verb.values[S.src]
    S["pattern"] = np.where(S.cross, "T" + np.where(S.t_ntok == 1, "1", "2+") + "-I" + np.where(S.i_ntok == 1, "1", "2+"),
                            "same")
    S["direction"] = np.where(S.base_cls == 0, "I<-T", "T<-I")
    S.to_parquet(out / "swaps.parquet", index=False)
    C = pd.DataFrame(checks)
    C["diff"] = (C.iia_cross - C.sweep_iia_cross).abs()
    res = {"task": cli.task, "model": args.model, "settings": runner.settings, "sites": sites,
           "best_epochs": {int(s): int(ep[s]) for s in sites}, "max_abs_diff_vs_sweep": float(C["diff"].max()),
           "n_items": len(items), "item_token_counts": pd.Series(ntok).value_counts().sort_index().to_dict()}
    C.to_csv(out / "fold_checks.csv", index=False)
    # per site x pattern: pooled IIA, fold-mean IIA and fold range, n swaps, n distinct T / I verbs
    cr = S[S.cross]
    g = cr.groupby(["site", "pattern", "fold"]).iia.mean().unstack("fold")
    cr = cr.assign(t_verb=np.where(cr.base_cls == 1, cr.base_verb, cr.src_verb),
                   i_verb=np.where(cr.base_cls == 0, cr.base_verb, cr.src_verb))
    P = cr.groupby(["site", "pattern"]).agg(iia_pooled=("iia", "mean"), n_swaps=("iia", "size"),
                                            n_t_verbs=("t_verb", "nunique"), n_i_verbs=("i_verb", "nunique")).reset_index()
    P = P.merge(g.mean(1).rename("iia_fold_mean").reset_index(), on=["site", "pattern"])
    P = P.merge(g.min(1).rename("iia_fold_min").reset_index(), on=["site", "pattern"])
    P = P.merge(g.max(1).rename("iia_fold_max").reset_index(), on=["site", "pattern"])
    P.insert(0, "model", args.model)
    P.insert(1, "task", cli.task)
    P.to_csv(out / "by_pattern.csv", index=False)
    (out / "check.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    if cli.assert_sweep and res["max_abs_diff_vs_sweep"] > 1e-12:  # CSV round-trip of the sweep's float means
        raise SystemExit("retrained IIA differs from the sweep")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", choices=("active", "passive"))
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--sites", default="", help="default: every site in the sweep")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--assert-sweep", action="store_true")
    ap.add_argument("--out-dir", required=True)
    main(ap.parse_args())
