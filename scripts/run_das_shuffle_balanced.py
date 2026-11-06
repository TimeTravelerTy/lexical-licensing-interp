#!/usr/bin/env python3
"""Gemma replication, exploratory: balanced shuffled-label control for a final DAS run (run_das_round2.py final).

The final runs' shuffled-label control (run_das_round2.run_final) trains one DAS direction per fold on a random
permutation of the training verbs' classes. A random permutation can correlate with the true classes, in either sign,
and a rank-1 interchange then still finds the class axis. Here every permutation is constrained to be uncorrelated
with the true classes:
- per (draw, split, fold), among the fold's training verbs, exactly half of the transitive verbs are relabelled
  intransitive and exactly half of the intransitive verbs transitive. When a class has an odd count n, the number
  flipped is floor(n / 2) or ceil(n / 2), chosen at random per class (so the verb-level correlation is 0 in
  expectation and at most about 1 / n in size; recorded per fold as phi_verbs);
- verbs outside the training set keep their labels (they are not used in training);
- training is the final run's own (`run_das_round2.train`, same rank, epochs, lr, swaps, batch), seed
  50000 + 1000 * draw + 100 * split + fold;
- evaluation is the final run's own (`run_das_round2.evaluate` / `summarize`) on the fold's held-out swaps, with
  the true classes.
All 3 splits x 5 folds of the final run, same items and folds (checked against the final run's items.csv and
pairs_folds.csv). The run's settings come from the final run's final_meta.json args (model, dtype, BOS, fp32
unembedding, lr, ...).

Writes <out-dir>/shuffle_balanced.csv (one row per draw x split x fold) and meta.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_das_round2 as r2


def balanced_flip(verbs, cls, rng):
    """verbs: training verbs; cls: their true classes. Returns {verb: permuted class}, k per class, phi."""
    out = {}
    ks = {}
    for c in (0, 1):
        vs = [v for v, k in zip(verbs, cls) if k == c]
        n = len(vs)
        k = n // 2 if n % 2 == 0 else n // 2 + int(rng.integers(2))
        flip = set(rng.choice(vs, size=k, replace=False).tolist()) if k else set()
        ks[c] = (n, k)
        for v in vs:
            out[v] = 1 - c if v in flip else c
    t = np.array(cls, float)
    p = np.array([out[v] for v in verbs], float)
    phi = float(np.corrcoef(t, p)[0, 1]) if p.std() > 0 else float("nan")
    return out, ks, phi


def main(cli):
    fdir = Path(cli.final_dir)
    meta = json.loads((fdir / "final_meta.json").read_text())
    args = argparse.Namespace(**meta["args"])
    args.device = cli.device
    if cli.pairs:
        args.pairs = cli.pairs
    site, epochs, rank = int(meta["site"]), int(meta["epochs"]), int(meta["chosen_rank"])
    runner, pairs, items, reps, nat, dropped = r2.setup(args)
    ref = pd.read_csv(fdir / "items.csv")
    for c in ("verb", "cls", "subject", "pair_id"):
        assert (ref[c].to_numpy() == items[c].to_numpy()).all(), f"items differ from the final run ({c})"
    pf = pd.read_csv(fdir / "pairs_folds.csv").set_index("pair_id")
    labels = items.cls.to_numpy()
    vcls = items.groupby("verb").cls.first()
    out = Path(cli.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for split in range(args.n_splits):
        assign = r2.folds(pairs, args.n_folds, split)
        assert all(pf.loc[p, f"fold_split{split}"] == a for p, a in assign.items()), "folds differ from the final run"
        for f in range(args.n_folds):
            tr, te = r2.fold_indices(items, assign, f)
            tv = sorted(set(items.verb.values[tr]))
            for draw in range(cli.n_perm):
                rng = np.random.default_rng([2026, draw, split, f])
                shuffled, ks, phi = balanced_flip(tv, vcls[tv].tolist(), rng)
                perm = np.array([shuffled.get(v, c) for v, c in zip(items.verb, labels)])
                phi_items = float(np.corrcoef(labels[tr], perm[tr])[0, 1])
                basis, _ = r2.train(runner, items, reps, tr, perm, site, rank, epochs, args,
                                    seed=50000 + 1000 * draw + 100 * split + f)
                df = r2.evaluate(runner, items, reps, nat, te, site, basis)
                rows.append({"site": site, "draw": draw, "split": split, "fold": f, "n_T": ks[1][0], "k_T": ks[1][1],
                             "n_I": ks[0][0], "k_I": ks[0][1], "phi_verbs": phi, "phi_items": phi_items,
                             **r2.summarize(df, items)})
            pd.DataFrame(rows).to_csv(out / "shuffle_balanced.csv", index=False)
            print(f"split {split} fold {f}: {cli.n_perm} draws, mean iia_cross "
                  f"{np.mean([r['iia_cross'] for r in rows[-cli.n_perm:]]):.3f} ({time.time() - t0:.0f}s)", flush=True)
    (out / "meta.json").write_text(json.dumps(
        {"final_dir": str(fdir), "site": site, "epochs": epochs, "rank": rank, "n_perm": cli.n_perm,
         "runner_settings": runner.settings, "dropped_pairs": dropped, "args": vars(args)}, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final-dir", required=True, help="final_strict* directory of the run (final_meta.json, items.csv)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pairs", default="", help="override the final run's --pairs path (relative to the cwd)")
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    main(ap.parse_args())
