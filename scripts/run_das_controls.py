#!/usr/bin/env python3
"""Gemma replication: controls for the early-layer DAS gap (random rank-1 directions, linear probe).

Items, behaviour filter and folds come from the sweep's own arguments (its sweep_meta.json), so both controls use
the sweep's held-out sets: split 0, 5 folds, the same train / held-out items as run_das_token_split.py.

- random: per site, --n-random unit directions, drawn as the final runs draw their random controls
  (torch.linalg.qr of a standard normal (d, 1); CPU generator seeded --seed + site), no training. Each direction
  is evaluated on every fold's held-out swaps with the unchanged interchange (run_das_round2.evaluate), against
  that fold's natural readout (passive: minus the fold's tau). One row per site x draw x fold; also the
  unpatched IIA of the same swaps (base readout already on the source's side), the floor a null patch gives.
- probe: per site and fold, logistic regression for T vs I (item class) on the verb-position states (the
  natural-pass states at the participle's last subtoken, the DAS source states), fit on the fold's DAS training
  items and scored on its held-out items. Features standardized with the training mean / sd; sklearn
  LogisticRegression (L2, C = 1, lbfgs, max_iter 5000).

The last site's states are post-final-norm (output_hidden_states), as in the sweep: reported, not used for picks.
Writes random.csv, probe.csv and meta.json to --out-dir.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main(cli):
    import torch
    from sklearn.linear_model import LogisticRegression

    import run_das_round2 as r2

    meta = json.loads((Path(cli.sweep_dir) / "sweep_meta.json").read_text())
    args = argparse.Namespace(**meta["args"])
    args.device = cli.device
    out = Path(cli.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if cli.task == "active":
        runner, pairs, items, reps, nat, dropped = r2.setup(args)
        assign = r2.folds(pairs, args.n_folds, 0)

        def fold(f):
            tr, te = r2.fold_indices(items, assign, f)
            return tr, te, nat, None
    else:
        import run_passive_das as rp

        runner, pf, items, reps, keep, filt, ctx = rp.setup(args)

        def fold(f):
            tr, te = rp.fold_sets(items, pf, keep, 0, f)
            tau = rp.set_tau(runner, items, tr)
            return tr, te, {"M": items.Mby_nat.to_numpy() - tau}, tau

    sites = [int(s) for s in cli.sites.split(",")] if cli.sites else list(range(runner.n_sites))
    labels = items.cls.to_numpy()
    d = reps.shape[-1]
    folds = [fold(f) for f in range(args.n_folds)]
    rnd_rows, probe_rows = [], []
    for site in sites:
        t0 = time.time()
        gen = torch.Generator(device="cpu").manual_seed(cli.seed + site)
        for r in range(cli.n_random):
            basis = torch.linalg.qr(torch.randn((d, 1), generator=gen))[0].to(runner.device)
            for f, (tr, te, nat_f, tau) in enumerate(folds):
                if tau is not None:
                    runner.tau = tau  # the passive readout subtracts the fold's threshold
                df = r2.evaluate(runner, items, reps, nat_f, te, site, basis)
                cross = df[df.cross]
                rnd_rows.append({"site": site, "draw": r, "fold": f, "iia_cross": float(cross.iia.mean()),
                                 "n_cross": int(len(cross)), "iia_same": float(df[~df.cross].iia.mean()),
                                 "iia_cross_unpatched": float(((cross.M_base > 0) == (cross.src_cls == 1)).mean())})
        X = reps[:, site].double().cpu().numpy()
        for f, (tr, te, _, _) in enumerate(folds):
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd[sd == 0] = 1.0
            clf = LogisticRegression(C=1.0, max_iter=5000).fit((X[tr] - mu) / sd, labels[tr])
            pred = clf.predict((X[te] - mu) / sd)
            probe_rows.append({"site": site, "fold": f, "acc": float((pred == labels[te]).mean()), "n_test": len(te),
                               "n_train": len(tr), "train_acc": float(clf.score((X[tr] - mu) / sd, labels[tr])),
                               "n_iter": int(clf.n_iter_[0])})
        pd.DataFrame(rnd_rows).to_csv(out / "random.csv", index=False)
        pd.DataFrame(probe_rows).to_csv(out / "probe.csv", index=False)
        R = pd.DataFrame(rnd_rows)
        g = R[R.site == site].groupby("draw").iia_cross.mean()
        P = pd.DataFrame(probe_rows)
        print(f"site {site}: random IIA mean {g.mean():.3f} p95 {g.quantile(0.95):.3f}; probe acc "
              f"{P[P.site == site].acc.mean():.3f} ({time.time() - t0:.0f}s)", flush=True)
    res = {"task": cli.task, "model": args.model, "settings": runner.settings, "sites": sites,
           "n_random": cli.n_random, "seed": cli.seed, "n_items": len(items), "sweep_dir": cli.sweep_dir,
           "d_model": int(d)}
    (out / "meta.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", choices=("active", "passive"))
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--sites", default="", help="default: every site 0..n_layers")
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", required=True)
    main(ap.parse_args())
