#!/usr/bin/env python3
"""Reverse DAS, training side: a rank-1 direction learned on passives (round3_plan.md, B5).

Items: "The N was <participle>" for both verbs of the 29 strict DAS pairs, in
15 curated passive_2 contexts drawn once (seed 17; 5 per context band, of
which 4 are training contexts and 1 a held-out context). Fold assignments are
the frozen ones of the active run (`pairs_folds.csv`), so fold f's passive and
active directions are trained on the same verbs.

Read-out ("by vs rest"): M_p = log P(" by") - log(1 - P(" by")) at the
participle's last token; each fold subtracts its threshold tau = midpoint of
the mean M_p of its training good and bad items (training contexts).
Training is `run_das_round2.train` (BCE on sigma(M_p - tau), label = source
class; base and source share the context; 4 cross-class and 2 same-class swaps
per base). A pair is kept only if its good passive has the higher M_p in >= 8
of the 12 training contexts.

Modes
- sweep: rank 1, every site in --sites, split 0, 5 folds, held-out passive
  metrics per epoch; then the frozen epoch rule per site ->
  `frozen_config_site{s}.json` in --out-dir.
- final --site S: 3 splits x 5 folds at the frozen epochs; controls (100
  random rank-1 directions raw and norm-matched; shuffled-label DAS); saves
  `bases.pt` (r1_s{split}_f{fold}), `summary.csv`, `heldout_swaps.csv.gz`,
  `items.csv`, `final_meta.json`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_das_round2 as r2

N_TRAIN_CTX, N_HELD_CTX = 4, 1  # per context band


class ByRunner(r2.Runner):
    def __init__(self, args):
        super().__init__(args)
        enc = self.tok.encode(" by", add_special_tokens=False)
        assert len(enc) == 1
        self.by_id = enc[0]
        self.tau = 0.0

    def readout(self, logits, anchors, full=False):
        torch = self.torch
        rows = torch.arange(logits.shape[0], device=logits.device)
        z = logits[rows, anchors].float()
        rest = z.clone()
        rest[:, self.by_id] = float("-inf")
        m = z[:, self.by_id] - torch.logsumexp(rest, -1) - self.tau  # log-odds of " by", minus the fold threshold
        if not full:
            return m, None
        lp = torch.log_softmax(z, -1)
        parts = {k: torch.logsumexp(lp[:, self.ids[k]], -1) for k in ("O", "I", "det", "pron", "refl")}
        return m, parts


def contexts(args):
    pas = pd.read_json(args.passives, lines=True, dtype={"item_id": str})
    pas["prefix"] = [g[: len(g) - len(v)] for g, v in zip(pas.good_prompt, pas.good_verb)]
    ctx = pas.drop_duplicates("context_id")[["context_id", "context_band", "prefix"]].sort_values("context_id")
    rng = np.random.default_rng(args.seed)
    rows = []
    for band, g in ctx.groupby("context_band"):
        pick = g.iloc[rng.permutation(len(g))[:N_TRAIN_CTX + N_HELD_CTX]]
        for i, r in enumerate(pick.itertuples()):
            rows.append({"context_id": r.context_id, "context_band": band, "prefix": r.prefix,
                         "role": "train" if i < N_TRAIN_CTX else "heldout"})
    return pd.DataFrame(rows), ctx


def load_items(args):
    pf = pd.read_csv(args.folds)
    ctx, _ = contexts(args)
    items = []
    for p in pf.itertuples():
        for side, cls in (("trans", 1), ("intrans", 0)):
            for c in ctx.itertuples():
                items.append({"verb": getattr(p, side), "cls": cls, "pair_id": p.pair_id, "source": p.source,
                              "tok": p.tok, "subject": c.context_id, "ctx_role": c.role,
                              "context_band": c.context_band, "prompt": c.prefix + getattr(p, f"{side}_part")})
    return pf, pd.DataFrame(items), ctx


def natural(runner, items, batch=256):
    """Unpatched logit-by (tau = 0), set log-probs and last-token states at every site."""
    torch = runner.torch
    runner.tau = 0.0
    m, parts, reps = runner.natural(items.prompt.tolist(), batch=batch)
    return m.cpu().numpy(), {k: v.cpu().numpy() for k, v in parts.items()}, reps


def setup(args):
    runner = ByRunner(args)
    pf, items, ctx = load_items(args)
    mnat, parts, reps = natural(runner, items)
    items["Mby_nat"] = mnat
    w_all = items.pivot_table(index=["pair_id", "subject"], columns="cls", values="Mby_nat")
    items["pair_gap"] = items.pair_id.map((w_all[1] - w_all[0]).groupby(level=0).mean())
    for k, v in parts.items():
        items[f"lp_{k}_nat"] = v
    # behaviour filter: good > bad in >= 8 of the 12 training contexts
    tr = items[items.ctx_role == "train"]
    w = tr.pivot_table(index=["pair_id", "subject"], columns="cls", values="Mby_nat")
    wins = (w[1] > w[0]).groupby(level=0).sum()
    keep = set(wins[wins >= args.min_wins].index)
    filt = pd.DataFrame({"pair_id": wins.index, "wins": wins.to_numpy(), "kept": wins.index.isin(keep)})
    return runner, pf, items, reps, keep, filt, ctx


def fold_sets(items, pf, keep, split, f):
    fo = items.pair_id.map(dict(zip(pf.pair_id, pf[f"fold_split{split}"]))).to_numpy()
    kept = items.pair_id.isin(keep).to_numpy()
    tr = np.flatnonzero((fo != f) & kept & (items.ctx_role == "train").to_numpy())
    te = np.flatnonzero((fo == f) & kept)
    return tr, te


def set_tau(runner, items, tr):
    t = items.iloc[tr]
    runner.tau = float((t[t.cls == 1].Mby_nat.mean() + t[t.cls == 0].Mby_nat.mean()) / 2)
    return runner.tau


def metrics(df, items):
    """Held-out swap metrics. Primary: Delta M_p = M_patched - M_base for bad base <- good source (pc)."""
    gap = items.pair_gap.values[df.base]  # the base pair's natural good - bad gap
    df = df.assign(dM=df.M_patched - df.M_base, held=items.ctx_role.values[df.base] == "heldout",
                   gap_ok=gap >= 0.2, fpair=(df.M_patched - df.M_base) / gap)
    cross = df[df.cross]
    pc, rev = cross[cross.base_cls == 0], cross[cross.base_cls == 1]
    same = df[~df.cross]
    return {"pc_dM": float(pc.dM.mean()), "pc_dM_heldout_ctx": float(pc[pc.held].dM.mean()),
            "rev_dM": float(rev.dM.mean()), "same_abs_dM": float(same.dM.abs().mean()),
            "pc_frac_median": float(pc[pc.gap_ok].fpair.median()),  # Delta M_p / base pair's gap (declared)
            "pc_frac_mean": float(pc[pc.gap_ok].fpair.mean()),
            "rev_frac_median": float(-rev[rev.gap_ok].fpair.median()),
            "pc_swapfrac_median": float(pc.frac.median()),  # run_das_round2's (patched - base) / (source - base)
            "iia_cross": float(cross.iia.mean()), "iia_same": float(same.iia.mean()),
            "iia_cross_heldout_ctx": float(cross[cross.held].iia.mean()), "n_swaps": int(len(df))}


def nat_accuracy(items, te, tau):
    t = items.iloc[te]
    return float(((t.Mby_nat - tau > 0) == (t.cls == 1)).mean())


def run_sweep(args):
    runner, pf, items, reps, keep, filt, ctx = setup(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for site in [int(s) for s in args.sites.split(",")]:
        t0 = time.time()
        for f in range(args.n_folds):
            tr, te = fold_sets(items, pf, keep, 0, f)
            tau = set_tau(runner, items, tr)
            nat = {"M": items.Mby_nat.to_numpy() - tau}
            fn = lambda basis: metrics(r2.evaluate(runner, items, reps, nat, te, site, basis), items)
            _, hist = r2.train(runner, items, reps, tr, items.cls.to_numpy(), site, 1, args.epochs, args,
                               seed=1000 * site + f, eval_fn=fn)
            for h in hist:
                rows.append({"site": site, "split": 0, "fold": f, "tau": tau, "nat_acc": nat_accuracy(items, te, tau), **h})
        pd.DataFrame(rows).to_csv(out / "sweep.csv", index=False)
        print(f"site {site} done in {time.time() - t0:.0f}s", flush=True)
    sw = pd.DataFrame(rows)
    for site, g in sw.groupby("site"):
        curve = g.groupby("epoch").iia_cross.mean()
        best = curve.max()
        ep = int(curve[curve >= best - 0.01].index.min())
        cfg = {"site": int(site), "layer_output": int(site) - 1, "epochs": ep, "rank": 1,
               "role": "B5 reverse DAS (round3_plan.md), trained on passives",
               "epoch_rule": "smallest epoch whose mean held-out cross-class passive IIA is within 0.01 of the site's best",
               "best_epoch_iia": float(best), "chosen_epoch_iia": float(curve[ep]),
               "declared": "written by run_passive_das.py sweep, before any active evaluation"}
        (out / f"frozen_config_site{site}.json").write_text(json.dumps(cfg, indent=2) + "\n")
    filt.to_csv(out / "pair_filter.csv", index=False)
    ctx.to_csv(out / "contexts.csv", index=False)
    (out / "sweep_meta.json").write_text(json.dumps({"args": vars(args), "kept_pairs": len(keep),
                                                    "dropped": sorted(set(pf.pair_id) - keep)}, indent=2))


def run_final(args):
    import torch

    cfg = json.loads((Path(args.out_dir) / f"frozen_config_site{args.site}.json").read_text())
    site, epochs = cfg["site"], cfg["epochs"]
    runner, pf, items, reps, keep, filt, ctx = setup(args)
    out = Path(args.out_dir) / f"final_site{site}"
    out.mkdir(parents=True, exist_ok=True)
    labels = items.cls.to_numpy()
    summ, bases, details = [], {}, []
    gen = torch.Generator(device="cpu").manual_seed(7)
    for split in range(args.n_splits):
        for f in range(args.n_folds):
            tr, te = fold_sets(items, pf, keep, split, f)
            tau = set_tau(runner, items, tr)
            nat = {"M": items.Mby_nat.to_numpy() - tau}
            basis, _ = r2.train(runner, items, reps, tr, labels, site, 1, epochs, args, seed=10000 + 100 * split + f)
            df = r2.evaluate(runner, items, reps, nat, te, site, basis, detail=True)
            df["split"], df["fold"], df["control"], df["tau"] = split, f, "das", tau
            details.append(df)
            base_row = {"rank": 1, "split": split, "fold": f, "tau": tau, "nat_acc": nat_accuracy(items, te, tau)}
            summ.append({**base_row, "control": "das", **metrics(df, items)})
            bases[(split, f)] = basis.cpu()
            for r in range(args.n_random):
                rnd = torch.linalg.qr(torch.randn(basis.shape, generator=gen))[0].to(runner.device)
                for name, mn in (("random", None), ("random_normmatched", basis)):
                    d = r2.evaluate(runner, items, reps, nat, te, site, rnd, match_norm=mn)
                    summ.append({**base_row, "control": name, "draw": r, **metrics(d, items)})
            vcls = items.groupby("verb").cls.first()
            tv = sorted(set(items.verb.values[tr]))
            shuffled = dict(zip(tv, np.random.default_rng(100 * split + f).permutation(vcls[tv].to_numpy())))
            perm = np.array([shuffled.get(v, c) for v, c in zip(items.verb, labels)])
            sb, _ = r2.train(runner, items, reps, tr, perm, site, 1, epochs, args, seed=777 + 100 * split + f)
            d = r2.evaluate(runner, items, reps, nat, te, site, sb)
            summ.append({**base_row, "control": "shuffled_labels", **metrics(d, items)})
            bases[("shuf", split, f)] = sb.cpu()
            print(f"site {site} split {split} fold {f}: {summ[-2 * args.n_random - 2]}", flush=True)
    pd.DataFrame(summ).to_csv(out / "summary.csv", index=False)
    det = pd.concat(details, ignore_index=True)
    for col in ("verb", "pair_id", "subject", "ctx_role"):
        det[f"base_{col}"] = items[col].values[det.base]
        det[f"src_{col}"] = items[col].values[det.src]
    det.to_csv(out / "heldout_swaps.csv.gz", index=False)
    torch.save({(f"r1_s{k[0]}_f{k[1]}" if k[0] != "shuf" else f"shuf_s{k[1]}_f{k[2]}"): v for k, v in bases.items()},
               out / "bases.pt")
    items.to_csv(out / "items.csv", index=False)
    pf.to_csv(out / "pairs_folds.csv", index=False)
    meta = {"site": site, "epochs": epochs, "config": cfg, "kept_pairs": sorted(keep),
            "dropped_pairs": sorted(set(pf.pair_id) - keep), "items": len(items), "args": vars(args)}
    (out / "final_meta.json").write_text(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("sweep", "final"))
    ap.add_argument("--folds", default="results/das_round2/final_strict/pairs_folds.csv")
    ap.add_argument("--passives", default="data/passive_das/passives.jsonl")
    ap.add_argument("--out-dir", default="results/das_round2/reverse")
    ap.add_argument("--sites", default="4,6,8,10,12,14,16,17")
    ap.add_argument("--site", type=int, default=8)
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--n-random", type=int, default=100)
    ap.add_argument("--n-cross", type=int, default=4)
    ap.add_argument("--n-same", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--min-wins", type=int, default=8)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    run_sweep(args) if args.mode == "sweep" else run_final(args)
