#!/usr/bin/env python3
"""LEACE learning curve (Pythia-1.4B fp32): erasers fit on n training verb pairs, evaluated on held-out pairs.
Rules: reports/pythia/leace_learning_curve_rules.md (Part A), committed before any run.

Conditions, each fit on the training pairs only (both prompts, all contexts; LeaceFitter defaults, float64, rank 1):
- single_8: LEACE at site 8 alone, fit on natural states.
- scrub_0_12: sequential scrub over sites 0..12 (eraser L fit on site-L states with erasers 0..L-1 applied).

Modes:
- subsets: n in --ns (default 25,50,75,100), --reps random subsets each (rng default_rng(10_000 + 100 n + r));
  held out = every other pair. Per fit: per-prompt log P(by) and KL on the held-out prompts (natural recomputed in the
  same batch) and per-prompt linear-probe predictions on the held-out erased states (single_8 at site 8, scrub_0_12 at
  sites 8 and 12) and natural states (sites 8, 12): held-out pairs dealt into 5 groups (inner_splits, seed 1 + fit
  index), train on 4, test on the 5th. --with-full also fits on all pairs and evaluates on all prompts (check: must
  reproduce the in-sample run, --reference-items).
- loo: fits for pair indices --loo a-b (sorted pair list): train on all other pairs, evaluate on the held-out pair;
  writes the held-out erased states (states.npz) for loo_probe.
- loo_probe: assembles the LOO erased states of every pair (--loo-dirs) and probes them with 5 pair groups over all
  pairs (seed 1). Flagged as leaky in the rules: each probe-training item's eraser saw the probe's test pairs.

Outputs (--out-dir): held.csv.gz (one row per fit x held-out prompt), prompts.csv.gz (prompt metadata, row = global
prompt index), fits.csv, fit_diagnostics.csv, checks.json, meta.json; loo: states.npz; loo_probe: loo_probes.csv.gz.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_leace_necessity import (DAS_SITE, Model, Stop, apply_erasers, fit_stats, inner_splits, load_items,
                                 probe_cv, sha)

TOP = 12  # scrub 0..TOP
PROBES = [("single_8", DAS_SITE), ("scrub_0_12", DAS_SITE), ("scrub_0_12", TOP)]
META_COLS = ["item", "side", "cls", "verb", "verb_pair", "verb_band", "context_id", "context_band", "ntok_part",
             "n_tokens", "fold", "prompt"]


def probe_pred(X, y, splits, torch, dev, seed=0):
    """probe_cv's linear probe, returning per-row predictions (-1 where a row is in no test group)."""
    pred = np.full(len(y), -1, dtype=np.int8)
    for g, k, tr, te in splits:
        Xtr = torch.as_tensor(X[tr], device=dev)
        mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
        Xtr = (Xtr - mu) / sd
        Xte = (torch.as_tensor(X[te], device=dev) - mu) / sd
        ytr = torch.as_tensor(y[tr], device=dev, dtype=torch.float32)
        torch.manual_seed(seed + 10 * g + k)
        net = torch.nn.Linear(X.shape[1], 1).to(dev)
        opt = torch.optim.LBFGS(net.parameters(), max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xtr)[:, 0], ytr) \
                + 1e-3 * net.weight.pow(2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            pred[te] = (net(Xte)[:, 0] > 0).long().cpu().numpy()
    return pred


class Ctx:
    """Model, items and natural site-8 / site-12 states shared by every fit."""

    def __init__(self, args):
        import torch
        from concept_erasure import LeaceFitter

        args.dtype = "float32"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch, self.LeaceFitter, self.args = torch, LeaceFitter, args
        self.t0 = time.time()
        self.M = Model(args)
        self.dev, self.d = self.M.dev, self.M.d
        _, self.P, self.pairs = load_items(args, self.M.R.tok)
        self.N = len(self.P)
        self.prompts = self.P.prompt.tolist()
        self.ntok = self.P.n_tokens.to_numpy()
        self.cls = self.P.cls.to_numpy()
        self.z_t = torch.as_tensor(self.cls, device=self.dev, dtype=torch.float64)
        self.pair_list = sorted(self.P.verb_pair.unique())
        self.pair_np = self.P.verb_pair.to_numpy()
        self.order = np.argsort(self.ntok, kind="stable")
        self.nat = {s: torch.zeros((self.N, self.d), dtype=torch.float32, device=self.dev) for s in (DAS_SITE, TOP)}
        for b in self.batches(np.arange(self.N)):
            cap = {}
            hooks = {s: (lambda v, s=s: cap.__setitem__(s, v.clone())) for s in self.nat}
            self.M.forward([self.prompts[i] for i in b], hooks, logits=False)
            bt = torch.as_tensor(b, device=self.dev)
            for s in self.nat:
                self.nat[s][bt] = cap[s]
        self.log(f"{self.N} prompts, {len(self.pair_list)} pairs; natural site {DAS_SITE} / {TOP} states captured")

    def log(self, msg):
        print(f"[{time.time() - self.t0:7.0f}s] {msg}", flush=True)

    def batches(self, idx):
        idx = idx[np.argsort(self.ntok[idx], kind="stable")]
        return [idx[i:i + self.args.batch_size] for i in range(0, len(idx), self.args.batch_size)]

    def rows_of(self, pairs):
        return np.flatnonzero(np.isin(self.pair_np, list(pairs)))

    def fit(self, train, fit_id):
        """single_8 and scrub 0..TOP erasers on the training prompts (the in-sample run's fitting code path)."""
        torch, dev, d = self.torch, self.dev, self.d
        fitter = self.LeaceFitter(d, 1, device=dev, dtype=torch.float64)
        for i in range(0, len(train), 4096):
            j = torch.as_tensor(train[i:i + 4096], device=dev)
            fitter.update(self.nat[DAS_SITE][j].double(), self.z_t[j])
        E, st = fit_stats(fitter, torch)
        single = {0: E}
        diag = [{"fit_id": fit_id, "condition": "single", "site": DAS_SITE, **st}]
        scrub = {}
        bidx = self.batches(train)
        for L in range(TOP + 1):
            fitter = self.LeaceFitter(d, 1, device=dev, dtype=torch.float64)
            for b in bidx:
                cap = {}
                ef = torch.zeros(len(b), dtype=torch.long, device=dev)
                hooks = {s: (lambda v, s=s: apply_erasers(scrub[s], ef, v)) for s in range(L)}

                def capture(v):
                    cap["x"] = v.clone()  # site L, before any erasure at L
                    raise Stop
                hooks[L] = capture
                self.M.forward([self.prompts[i] for i in b], hooks)
                fitter.update(cap["x"].double(), self.z_t[torch.as_tensor(b, device=dev)])
            E, st = fit_stats(fitter, torch)
            scrub[L] = {0: E}
            diag.append({"fit_id": fit_id, "condition": "scrub", "site": L, **st})
        return single, scrub, diag

    def evaluate(self, held, single, scrub):
        """Per held-out prompt: log P(by) natural / single_8 / scrub_0_12, KL, and erased states for the probes."""
        torch, dev = self.torch, self.dev
        pos = {g: i for i, g in enumerate(held)}
        H = len(held)
        cols = {k: np.zeros(H) for k in ("lp_by__natural", "lp_by__single_8", "lp_by__scrub_0_12",
                                          "kl__single_8", "kl__scrub_0_12")}
        states = {(c, s): np.zeros((H, self.d), dtype=np.float32) for c, s in PROBES}
        for b in self.batches(held):
            loc = np.array([pos[g] for g in b])
            bp = [self.prompts[i] for i in b]
            ef = torch.zeros(len(b), dtype=torch.long, device=dev)
            lp0, _ = self.M.forward(bp, {})
            p0 = lp0.exp()
            cap = {}

            def h8(v):
                new = apply_erasers(single, ef, v)
                cap[("single_8", DAS_SITE)] = new.clone()
                return new
            lp1, _ = self.M.forward(bp, {DAS_SITE: h8})

            def hs(v, s):
                new = apply_erasers(scrub[s], ef, v)
                if ("scrub_0_12", s) in states:
                    cap[("scrub_0_12", s)] = new.clone()
                return new
            lp2, _ = self.M.forward(bp, {s: (lambda v, s=s: hs(v, s)) for s in range(TOP + 1)})
            cols["lp_by__natural"][loc] = lp0[:, self.M.by].cpu().numpy()
            for c, lp in (("single_8", lp1), ("scrub_0_12", lp2)):
                cols[f"lp_by__{c}"][loc] = lp[:, self.M.by].cpu().numpy()
                cols[f"kl__{c}"][loc] = (p0 * (lp0 - lp)).sum(-1).cpu().numpy()
            for k in states:
                states[k][loc] = cap[k].cpu().numpy()
        return cols, states

    def probes(self, held, states, seed):
        """Linear probes on the held-out set (5 pair groups), erased and natural; per-row predictions."""
        torch = self.torch
        splits = inner_splits(np.zeros(len(held), dtype=np.int64), self.pair_np[held], 5, seed=seed)
        y = self.cls[held]
        out = {f"pred__{c}__s{s}": probe_pred(states[(c, s)], y, splits, torch, self.dev) for c, s in PROBES}
        for s in (DAS_SITE, TOP):
            out[f"pred__natural__s{s}"] = probe_pred(self.nat[s][torch.as_tensor(held, device=self.dev)].cpu().numpy(),
                                                     y, splits, torch, self.dev)
        return out, splits


def write_common(ctx, out, fits, diag, checks, extra_meta=None):
    P = ctx.P[META_COLS].copy()
    P.insert(0, "row", np.arange(ctx.N))
    P.to_csv(out / "prompts.csv.gz", index=False)
    pd.DataFrame(fits).to_csv(out / "fits.csv", index=False)
    pd.DataFrame(diag).to_csv(out / "fit_diagnostics.csv", index=False)
    (out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    meta = {"args": vars(ctx.args), "runner_settings": ctx.M.R.settings, "n_prompts": ctx.N,
            "n_pairs": len(ctx.pair_list), "scores_sha256": sha(ctx.args.scores),
            "minutes": (time.time() - ctx.t0) / 60, **(extra_meta or {})}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


def run_subsets(ctx, out):
    args = ctx.args
    fits, diag, frames, checks = [], [], [], {}
    plan = [("full", 0, 0, ctx.pair_list)] if args.with_full else []
    for n in [int(x) for x in args.ns.split(",")]:
        for r in range(args.reps):
            rng = np.random.default_rng(10_000 + 100 * n + r)
            pick = rng.choice(len(ctx.pair_list), size=n, replace=False)
            plan.append(("subset", n, r, sorted(ctx.pair_list[i] for i in pick)))
    k = 0  # fit index among the subset fits (probe split seed 1 + k)
    for kind, n, r, train_pairs in plan:
        fit_id = "full" if kind == "full" else f"n{n}_r{r}"
        train = ctx.rows_of(train_pairs)
        held = np.arange(ctx.N) if kind == "full" else np.setdiff1d(np.arange(ctx.N), train)
        single, scrub, dg = ctx.fit(train, fit_id)
        diag += dg
        cols, states = ctx.evaluate(held, single, scrub)
        F = pd.DataFrame({"fit_id": fit_id, "row": held, **cols})
        if kind == "subset":
            preds, splits = ctx.probes(held, states, seed=1 + k)
            for c, v in preds.items():
                F[c] = v
            if k == 0:  # probe_pred must equal probe_cv's linear probe on the same split set
                X = ctx.nat[DAS_SITE][ctx.torch.as_tensor(held, device=ctx.dev)].cpu().numpy()
                acc_cv = probe_cv(X, ctx.cls[held], splits, "linear", ctx.torch, ctx.dev)
                acc_pp = float((F[f"pred__natural__s{DAS_SITE}"].to_numpy() == ctx.cls[held]).mean())
                checks["probe_pred_vs_probe_cv"] = {"probe_cv": acc_cv, "probe_pred": acc_pp}
            k += 1
        frames.append(F)
        fits.append({"fit_id": fit_id, "kind": kind, "n": len(train_pairs), "rep": r,
                     "n_held_pairs": int(len(np.unique(ctx.pair_np[held]))), "n_train_prompts": len(train),
                     "n_held_prompts": len(held), "train_pairs": ";".join(map(str, train_pairs))})
        ctx.log(f"{fit_id}: share remaining (held-out) single_8 "
                f"{_share(ctx, F, 'single_8'):.3f}, scrub_0_12 {_share(ctx, F, 'scrub_0_12'):.3f}"
                + (f"; probe acc single_8@8 {_acc(ctx, F, 'pred__single_8__s8'):.3f}, scrub@12 "
                   f"{_acc(ctx, F, 'pred__scrub_0_12__s12'):.3f}, natural@8 {_acc(ctx, F, 'pred__natural__s8'):.3f}"
                   if kind == "subset" else ""))
    H = pd.concat(frames, ignore_index=True)
    if args.reference_items and Path(args.reference_items).exists():
        R = pd.read_csv(args.reference_items)
        assert (R.prompt.to_numpy() == ctx.P.prompt.to_numpy()).all()
        checks["reference_items_sha256"] = sha(args.reference_items)
        if args.with_full:
            Fh = H[H.fit_id == "full"].sort_values("row")
            checks["full_vs_insample_max_abs"] = {
                c: float(np.abs(Fh[f"lp_by__{c}"].to_numpy() - R[f"lp_by__{c}"].to_numpy()).max())
                for c in ("natural", "single_8", "scrub_0_12")}
        nat = H.groupby("row").lp_by__natural.first()
        checks["natural_vs_reference_max_abs"] = float(np.abs(nat.to_numpy() - R.lp_by__natural.to_numpy()[nat.index]).max())
    checks["erased_rank"] = sorted({int(x["erased_rank"]) for x in diag})
    H.to_csv(out / "held.csv.gz", index=False)
    write_common(ctx, out, fits, diag, checks)
    ctx.log(f"checks {checks}")


def _share(ctx, F, c):
    m = F.row.to_numpy()
    s = np.where(ctx.cls[m] == 1, 1.0, -1.0)
    n_items = len(m) / 2
    return float((s * F[f"lp_by__{c}"]).sum() / n_items / ((s * F["lp_by__natural"]).sum() / n_items))


def _acc(ctx, F, col):
    return float((F[col].to_numpy() == ctx.cls[F.row.to_numpy()]).mean())


def run_loo(ctx, out):
    a, b = (int(x) for x in ctx.args.loo.split("-"))
    fits, diag, frames = [], [], []
    st_rows, st = [], {k: [] for k in PROBES}
    for i in range(a, min(b, len(ctx.pair_list) - 1) + 1):
        p = ctx.pair_list[i]
        held = ctx.rows_of([p])
        train = np.setdiff1d(np.arange(ctx.N), held)
        fit_id = f"loo_{i}"
        single, scrub, dg = ctx.fit(train, fit_id)
        diag += dg
        cols, states = ctx.evaluate(held, single, scrub)
        frames.append(pd.DataFrame({"fit_id": fit_id, "row": held, **cols}))
        st_rows.append(held)
        for k in PROBES:
            st[k].append(states[k])
        fits.append({"fit_id": fit_id, "kind": "loo", "n": len(ctx.pair_list) - 1, "rep": i, "held_pair": p,
                     "n_held_pairs": 1, "n_train_prompts": len(train), "n_held_prompts": len(held)})
        ctx.log(f"{fit_id} ({p}) done")
    H = pd.concat(frames, ignore_index=True)
    H.to_csv(out / "held.csv.gz", index=False)
    np.savez(out / "states.npz", row=np.concatenate(st_rows),
             **{f"{c}__s{s}": np.concatenate(st[(c, s)]) for c, s in PROBES})
    checks = {"erased_rank": sorted({int(x["erased_rank"]) for x in diag})}
    write_common(ctx, out, fits, diag, checks)


def run_loo_probe(ctx, out):
    dirs = [Path(x) for x in ctx.args.loo_dirs.split(",")]
    rows, X = [], {f"{c}__s{s}": [] for c, s in PROBES}
    for d_ in dirs:
        z = np.load(d_ / "states.npz")
        rows.append(z["row"])
        for k in X:
            X[k].append(z[k])
    rows = np.concatenate(rows)
    assert len(rows) == ctx.N and (np.sort(rows) == np.arange(ctx.N)).all(), "LOO parts do not cover every prompt once"
    inv = np.argsort(rows)
    splits = inner_splits(np.zeros(ctx.N, dtype=np.int64), ctx.pair_np, 5, seed=1)
    res = pd.DataFrame({"row": np.arange(ctx.N)})
    for k in X:
        Xa = np.concatenate(X[k])[inv]
        res[f"pred__{k}"] = probe_pred(Xa, ctx.cls, splits, ctx.torch, ctx.dev)
        ctx.log(f"LOO probe {k}: {(res[f'pred__{k}'] == ctx.cls).mean():.3f}")
    for s in (DAS_SITE, TOP):
        res[f"pred__natural__s{s}"] = probe_pred(ctx.nat[s].cpu().numpy(), ctx.cls, splits, ctx.torch, ctx.dev)
        ctx.log(f"LOO probe natural s{s}: {(res[f'pred__natural__s{s}'] == ctx.cls).mean():.3f}")
    res.to_csv(out / "loo_probes.csv.gz", index=False)
    write_common(ctx, out, [], [], {"loo_dirs": [str(d_) for d_ in dirs]})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["subsets", "loo", "loo_probe"], required=True)
    ap.add_argument("--scores", default="results/passive_band_cross/pythia14b_scores.csv")
    ap.add_argument("--v2-pairs", default="data/matched_passives_v2/pairs.jsonl")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-folds", type=int, default=5, help="only for load_items' fold column")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--ns", default="25,50,75,100")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--with-full", action="store_true", help="also fit on all pairs (reproduction check)")
    ap.add_argument("--loo", default="0-125", help="LOO pair indices a-b (inclusive) in the sorted pair list")
    ap.add_argument("--loo-dirs", default="", help="comma-separated LOO part directories (loo_probe)")
    ap.add_argument("--reference-items", default="", help="items.csv.gz of the in-sample run")
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--max-contexts", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    o = Path(a.out_dir)
    o.mkdir(parents=True, exist_ok=True)
    c = Ctx(a)
    {"subsets": run_subsets, "loo": run_loo, "loo_probe": run_loo_probe}[a.mode](c, o)
    c.log(f"wrote {o}")
