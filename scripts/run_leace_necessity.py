#!/usr/bin/env python3
"""LEACE necessity test: is verb-class (T vs I) information at the verb position needed for the *by* preference?
Pythia-1.4B, fp32 (TF32 off).

Items and readout: exactly task 1 of analyze_das_prep.py (the natural passive baseline of run_next_token_readout.py):
curated `was` prefixes + participle, other verbs' contexts only (126 verb pairs x 125 contexts = 15,750 items, a good
(T) and a bad (I) prompt each), log P(" by") at the participle's last token. That token is the last prompt token, so it
is both the erasure position and the readout position (the DAS Runner's rule: attention_mask.sum - 1, no BOS).

Sites: DAS convention, site L = residual stream entering layer L (site 0 = embedding output, site L >= 1 = output of
layer L - 1), L = 0..23. The DAS site is 8.

Split: the DAS fold procedure (run_das_round2.folds: pairs shuffled within strata, dealt round-robin; 5 folds, seed 0)
applied to the 126 pairs, stratified by verb band x token pattern (both participles single-token or not). Every verb
belongs to exactly one pair, so this is a verb-level split. Erasers for fold f are fit on the items of the other
folds and applied to fold f's items (cross-fitting), so every item is erased by an eraser that never saw its verb.

Concept z = class (T = 1, I = 0). Fitting data per fold: the verb-position states of both prompts of every training
item (all 125 contexts). LeaceFitter defaults (method "leace", affine, constrain_cov_trace, shrinkage, svd_tol 0.01),
statistics in float64.

Conditions (erasure on both T and I prompts, at the verb position only):
- single_L: LEACE at site L alone, fit on natural states (L = 0..23).
- scrub_all: concept scrubbing at sites 0..23. Per fold, sequentially: eraser L is fit on the site-L states of the
  training items computed with erasers 0..L-1 (of the same fold) applied and eraser L not applied.
- scrub_0_8: erasers 0..8 of scrub_all only (sequential fitting makes them identical to the first nine).

Measures per item (written per condition): log P(" by"), KL(P_natural || P_erased) over the full vocabulary at the
verb position; the natural readout is recomputed in the same batch. Probes (nested, see inner_splits): for each
condition, site and outer fold g, the erased verb-position states of fold g's items (erased by fold g's erasers, fit
on the other folds) -> logistic regression and a 1-hidden-layer MLP trained on the fold-g items of 4 of 5 inner pair
groups and tested on the 5th; accuracy pooled over all test items. Natural states get the same splits.

Checks (fit_diagnostics.csv, checks.json): fitting n and distinct verbs; top singular value of W Sigma_xz (erasure is
skipped where it is below svd_tol); eraser rank; training cross-covariance residual ||P Sigma_xz|| / ||Sigma_xz||;
held-out class-mean difference after/before erasure; identity-hook self-patch; erasure leaves non-anchor positions
untouched; in-model erased states equal the offline erasure of the natural states (single_L); scrub replay: a pass over
the training items with all 24 erasers reproduces, at each site, the pre-erasure statistics the eraser was fit on.

In-sample variant (--insample): a single eraser set is fit on all items (all verbs, both prompts) and applied to those
same items, so the evaluated items are the fitting data and the linear erasure is exact on them. The fold structure
collapses to one "fold" (no cross-fitting); the same conditions, measures and checks are computed. The probes' inner
split becomes a 5-group verb-held-out split over all pairs (the eraser has seen the test verbs), and a linear probe is
also trained and scored on all items ("linear_fitset": the fitting distribution, where LEACE's guarantee applies).

--cumulative adds the scrub prefixes scrub_0_k (erasers 0..k of scrub_all, k = 0..23; scrub_0_8 and scrub_all are
k = 8 and 23), evaluated for log P(by) and KL only.

Outputs (--out-dir): items.csv.gz (one row per prompt, lp_by and kl per condition), probes.csv, fit_diagnostics.csv,
pairs_folds.csv, checks.json, meta.json, erasers.npz (per condition, fold and site: the rank-1 eraser's edit
direction (proj_left column: x' = x - <read, x - bias> edit), read direction (proj_right row) and bias).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_das_round2 as r2
from analyze_das_prep import curated_items

N_SITES = 24
DAS_SITE = 8


class Stop(Exception):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- items and folds

def load_items(args, tok):
    scores = pd.read_csv(args.scores, low_memory=False)
    v2 = pd.DataFrame([json.loads(l) for l in Path(args.v2_pairs).open(encoding="utf-8")])
    v2 = v2[v2.frame_id == "was"]
    pas = curated_items(scores, v2, "passive_2")
    pas = pas[pas.own_context == 0].reset_index(drop=True)
    pas = pas[["verb_pair", "verb_band", "context_id", "context_band", "prefix", "good_verb", "bad_verb"]].copy()
    if args.max_pairs:
        keep = []
        for band, g in pas.drop_duplicates("verb_pair").groupby("verb_band"):
            keep += sorted(g.verb_pair)[: args.max_pairs]
        pas = pas[pas.verb_pair.isin(keep)]
    if args.max_contexts:
        ctx = sorted(pas.context_id.unique())
        pas = pas[pas.context_id.isin(ctx[:: max(1, len(ctx) // args.max_contexts)][: args.max_contexts])]
    pas = pas.reset_index(drop=True)
    pas["item"] = np.arange(len(pas))
    rows = []
    for r in pas.itertuples():
        pre = tok.encode(r.prefix.rstrip(), add_special_tokens=False)
        for side, cls, verb in (("good", 1, r.good_verb), ("bad", 0, r.bad_verb)):
            prompt = r.prefix + verb
            ids = tok.encode(prompt, add_special_tokens=False)
            assert ids[: len(pre)] == pre, prompt
            rows.append({"item": r.item, "side": side, "cls": cls, "verb": verb, "verb_pair": r.verb_pair,
                         "verb_band": r.verb_band, "context_id": r.context_id, "context_band": r.context_band,
                         "prompt": prompt, "n_tokens": len(ids), "ntok_part": len(ids) - len(pre)})
    P = pd.DataFrame(rows)
    nt = P.groupby("verb").ntok_part.nunique()
    assert (nt == 1).all(), f"participle token count varies with context: {list(nt[nt > 1].index)}"
    pairs = P.drop_duplicates("verb_pair")[["verb_pair", "verb_band"]].copy()
    vt = P.groupby(["verb_pair", "cls"]).ntok_part.first().unstack()
    pairs["ntok_T"] = pairs.verb_pair.map(vt[1])
    pairs["ntok_I"] = pairs.verb_pair.map(vt[0])
    pairs["pair_id"] = pairs.verb_pair
    pairs["source"] = pairs.verb_band
    pairs["tok"] = np.where((pairs.ntok_T == 1) & (pairs.ntok_I == 1), "single", "multi")
    assign = r2.folds(pairs, args.n_folds, args.split_seed)
    pairs["fold"] = pairs.pair_id.map(assign)
    P["fold"] = P.verb_pair.map(assign)
    return pas, P, pairs.drop(columns=["pair_id", "source"]).sort_values("verb_pair").reset_index(drop=True)


# ---------------------------------------------------------------- model plumbing

class Model:
    def __init__(self, args):
        import torch

        self.torch = torch
        self.R = r2.Runner(args)  # loads the model, checks that hooked sites equal output_hidden_states
        self.dev = self.R.device
        by = self.R.tok.encode(" by", add_special_tokens=False)
        assert len(by) == 1
        self.by = by[0]
        self.d = self.R.model.config.hidden_size
        self.track_outside, self.outside_max = False, 0.0

    def module(self, site):
        return self.R.embed if site == 0 else self.R.layers[site - 1]

    def forward(self, prompts, hooks, logits=True):
        """hooks: {site: fn(v [B, d] fp32 anchor states) -> new v or None}. A hook may raise Stop.
        Returns log-softmax at the anchor (or None if stopped), and anchors."""
        torch = self.torch
        enc, anc = self.R.encode(prompts)
        rows = torch.arange(len(anc), device=self.dev)
        handles = []

        def make(site, fn):
            def hook(m, i, o):
                h = r2.normalize_output(o)
                new = fn(h[rows, anc].float())
                if new is None:
                    return None
                hn = h.clone()
                hn[rows, anc] = new.to(h.dtype)
                if self.track_outside:  # erasure must leave every non-anchor position untouched
                    diff = (hn - h).abs()
                    diff[rows, anc] = 0
                    self.outside_max = max(self.outside_max, float(diff.max()))
                return r2.replace_output(o, hn)
            return hook

        for s, fn in hooks.items():
            handles.append(self.module(s).register_forward_hook(make(s, fn)))
        try:
            with torch.no_grad():
                out = self.R.model(**enc, use_cache=False)
            lp = torch.log_softmax(out.logits[rows, anc].float(), -1) if logits else None
        except Stop:
            lp = None
        finally:
            for h in handles:
                h.remove()
        return lp, anc


def apply_erasers(erasers, efold, v):
    """Row r of v erased by erasers[efold[r]] (float64 maths)."""
    out = v.clone()
    for f in efold.unique().tolist():
        m = efold == f
        out[m] = erasers[f](v[m].double()).to(v.dtype)
    return out


def compress(E, torch):
    """Store the eraser at its actual rank (the trace-constraint branch returns d x d factors)."""
    from concept_erasure import LeaceEraser

    if E.proj_left.shape[1] <= 4:  # numerical rank: columns zeroed by svd_tol erase nothing
        return E, int((E.proj_left.norm(dim=0) > 0).sum()), False
    M = E.proj_left @ E.proj_right
    u, s, vh = torch.linalg.svd(M)
    k = int((s > s[0] * 1e-10).sum()) if s[0] > 0 else 0
    return LeaceEraser(u[:, :k] * s[:k], vh[:k], E.bias), k, True


def rank1(E):
    """Edit and read vectors of a rank-1 eraser (x' = x - (read . (x - bias)) edit)."""
    nl = E.proj_left.norm(dim=0)
    j = int(nl.argmax())
    assert int((nl > 0).sum()) == 1, f"eraser rank {int((nl > 0).sum())} != 1"
    return E.proj_left[:, j], E.proj_right[j], E.bias


def fit_stats(fitter, torch):
    """Top singular value of W Sigma_xz (as in LeaceFitter.eraser) and the eraser."""
    sigma = fitter.sigma_xx
    L, V = torch.linalg.eigh(sigma)
    mask = L > (L[-1] * sigma.shape[-1] * torch.finfo(L.dtype).eps)
    L = L.clamp_min(0.0)
    W = V * torch.where(mask, L.rsqrt(), 0.0) @ V.mH
    s = torch.linalg.svdvals(W @ fitter.sigma_xz)
    E = fitter.eraser
    E, rank, constrained = compress(E, torch)
    sxz = fitter.sigma_xz
    resid = (sxz - E.proj_left @ (E.proj_right @ sxz)).norm() / sxz.norm()
    return E, {"sv_top": float(s[0]), "erased_rank": rank, "trace_branch": constrained,
               "train_xcov_resid": float(resid), "n_fit": int(fitter.n)}


# ---------------------------------------------------------------- probes

def inner_splits(folds, pairs, n_inner, seed):
    """Nested probe splits: within each outer fold g, its pairs are dealt into n_inner groups; a probe is trained on
    the fold-g items of n_inner - 1 groups and tested on the last. All items involved were erased by fold g's eraser,
    which was fit on the other outer folds only, so neither the probe's training nor its test items were seen by the
    eraser. (Training on other outer folds instead leaks: their erasers were fit on data that include fold g, which
    pushes held-out accuracy below chance.)"""
    assert n_inner >= 2
    out = []
    for g in np.unique(folds):
        in_g = folds == g
        pg = np.random.default_rng(seed + int(g)).permutation(np.unique(pairs[in_g]))
        assert len(pg) >= 2, f"outer fold {g} has {len(pg)} pair(s); the nested probe needs at least 2"
        grp = {p: i % n_inner for i, p in enumerate(pg)}
        gi = np.array([grp.get(p, -1) for p in pairs])
        for k in range(min(n_inner, len(pg))):
            tr, te = in_g & (gi != k), in_g & (gi == k)
            if tr.any() and te.any():
                out.append((int(g), k, tr, te))
    return out


def probe_cv(X, y, splits, kind, torch, dev, seed=0):
    """Accuracy of a probe over the nested splits, pooled over test items."""
    correct, n = 0, 0
    for g, k, tr, te in splits:
        Xtr = torch.as_tensor(X[tr], device=dev)
        mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
        Xtr = (Xtr - mu) / sd
        Xte = (torch.as_tensor(X[te], device=dev) - mu) / sd
        ytr = torch.as_tensor(y[tr], device=dev, dtype=torch.float32)
        yte = torch.as_tensor(y[te], device=dev)
        torch.manual_seed(seed + 10 * g + k)
        if kind == "linear":
            net = torch.nn.Linear(X.shape[1], 1).to(dev)
            opt = torch.optim.LBFGS(net.parameters(), max_iter=200, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xtr)[:, 0], ytr) \
                    + 1e-3 * net.weight.pow(2).sum()
                loss.backward()
                return loss
            opt.step(closure)
        else:
            net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(),
                                      torch.nn.Linear(256, 1)).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
            gen = torch.Generator(device="cpu").manual_seed(seed + 10 * g + k)
            for _ in range(10):
                perm = torch.randperm(len(Xtr), generator=gen).to(dev)
                for i in range(0, len(Xtr), 512):
                    b = perm[i:i + 512]
                    opt.zero_grad()
                    torch.nn.functional.binary_cross_entropy_with_logits(net(Xtr[b])[:, 0], ytr[b]).backward()
                    opt.step()
        with torch.no_grad():
            pred = (net(Xte)[:, 0] > 0).long()
        correct += int((pred == yte).sum())
        n += int(te.sum())
    return correct / n


# ---------------------------------------------------------------- main

def run(args):
    import torch
    from concept_erasure import LeaceFitter

    args.dtype = "float32"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    M = Model(args)
    dev, d, tok = M.dev, M.d, M.R.tok
    pas, P, pairs = load_items(args, tok)
    pairs.to_csv(out / "pairs_folds.csv", index=False)
    # eraser "fold" of each prompt: its outer fold (cross-fitted) or 0 for every prompt (in-sample: one eraser set)
    fold_np = np.zeros(len(P), dtype=np.int64) if args.insample else P.fold.to_numpy()
    N, F = len(P), 1 if args.insample else args.n_folds
    sites = list(range(N_SITES))
    order = np.argsort(P.n_tokens.to_numpy(), kind="stable")
    batches = [order[i:i + args.batch_size] for i in range(0, N, args.batch_size)]
    fold_t = torch.as_tensor(fold_np, device=dev)
    cls_np = P.cls.to_numpy()
    z_t = torch.as_tensor(cls_np, device=dev, dtype=torch.float64)
    prompts = P.prompt.tolist()
    checks = {}
    log = lambda msg: print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)
    log(f"{N} prompts, {P.verb_pair.nunique()} pairs, {P.context_id.nunique()} contexts, folds "
        f"{pairs.fold.value_counts().sort_index().tolist()}")

    # 1. natural pass: anchor states at sites 0..23, log P(by)
    nat = np.zeros((N, N_SITES, d), dtype=np.float32)
    lp_nat_pass = np.zeros(N)
    for b in batches:
        cap = {}
        hooks = {s: (lambda v, s=s: cap.__setitem__(s, v.clone())) for s in sites}
        lp, _ = M.forward([prompts[i] for i in b], hooks)
        for s in sites:
            nat[b, s] = cap[s].cpu().numpy()
        lp_nat_pass[b] = lp[:, M.by].cpu().numpy()
    log("natural pass done")

    diag = []
    if args.insample:  # fit on everything, evaluate on everything
        train_idx = held_idx = {0: np.arange(N)}
    else:
        train_idx = {f: np.flatnonzero(fold_np != f) for f in range(F)}
        held_idx = {f: np.flatnonzero(fold_np == f) for f in range(F)}

    def held_meandiff(X, idx):
        c = cls_np[idx]
        return float(np.linalg.norm(X[idx][c == 1].mean(0) - X[idx][c == 0].mean(0)))

    # 2. single-layer erasers, fit on natural states of the training folds
    single = {s: {} for s in sites}
    for s in sites:
        for f in range(F):
            idx = train_idx[f]
            fitter = LeaceFitter(d, 1, device=dev, dtype=torch.float64)
            for i in range(0, len(idx), 4096):
                j = idx[i:i + 4096]
                fitter.update(torch.as_tensor(nat[j, s], device=dev).double(), z_t[j])
            E, st = fit_stats(fitter, torch)
            single[s][f] = E
            h = held_idx[f]
            Xh = torch.as_tensor(nat[h, s], device=dev)
            Xe = E(Xh.double()).float().cpu().numpy()
            pre = held_meandiff(nat[:, s], h)
            post = float(np.linalg.norm(Xe[cls_np[h] == 1].mean(0) - Xe[cls_np[h] == 0].mean(0)))
            diag.append({"condition": "single", "site": s, "fold": f, "n_verbs": int(P.verb.iloc[idx].nunique()),
                         **st, "held_meandiff_pre": pre, "held_meandiff_post": post})
    log("single-layer erasers fit")

    # 3. concept scrubbing, sequential per fold
    scrub = {s: {} for s in sites}
    fit_moments = {}
    for f in range(F):
        idx = train_idx[f]
        bidx = [idx[np.argsort(P.n_tokens.to_numpy()[idx], kind="stable")][i:i + args.batch_size]
                for i in range(0, len(idx), args.batch_size)]
        for L in sites:
            fitter = LeaceFitter(d, 1, device=dev, dtype=torch.float64)
            for b in bidx:
                cap = {}
                ef = torch.full((len(b),), f, device=dev)
                hooks = {s: (lambda v, s=s: apply_erasers(scrub[s], ef, v)) for s in range(L)}  # earlier sites

                def capture(v):
                    cap["x"] = v.clone()  # site L, before any erasure at L
                    raise Stop
                hooks[L] = capture
                M.forward([prompts[i] for i in b], hooks)
                fitter.update(cap["x"].double(), z_t[b])
            E, st = fit_stats(fitter, torch)
            scrub[L][f] = E
            fit_moments[(f, L)] = (fitter.mean_x.clone(), fitter.sigma_xz.clone())
            diag.append({"condition": "scrub", "site": L, "fold": f, "n_verbs": int(P.verb.iloc[idx].nunique()), **st})
        log(f"scrub fold {f} fit")

    # 3b. replay: all 24 erasers on the training items; the pre-erasure moments at each site must equal those fit on
    # (catches fitting on the wrong distribution, e.g. natural states or states already erased at L)
    rep = {"mean_maxabs": 0.0, "xcov_rel": 0.0, "post_xcov_resid_max": 0.0}
    for f in range(F):
        idx = train_idx[f]
        pre_f = {s: LeaceFitter(d, 1, device=dev, dtype=torch.float64, shrinkage=False) for s in sites}
        post_f = {s: LeaceFitter(d, 1, device=dev, dtype=torch.float64, shrinkage=False) for s in sites}
        for i in range(0, len(idx), args.batch_size):
            b = idx[i:i + args.batch_size]
            ef = torch.full((len(b),), f, device=dev)
            zb = z_t[b]

            def hk(v, s, zb=zb, ef=ef):
                pre_f[s].update(v.double(), zb)
                new = apply_erasers(scrub[s], ef, v)
                post_f[s].update(new.double(), zb)
                return new
            M.forward([prompts[i] for i in b], {s: (lambda v, s=s: hk(v, s)) for s in sites}, logits=False)
        for s in sites:
            mx, sxz = fit_moments[(f, s)]
            rep["mean_maxabs"] = max(rep["mean_maxabs"], float((pre_f[s].mean_x - mx).abs().max()))
            rep["xcov_rel"] = max(rep["xcov_rel"], float((pre_f[s].sigma_xz - sxz).norm() / sxz.norm()))
            rep["post_xcov_resid_max"] = max(rep["post_xcov_resid_max"],
                                             float(post_f[s].sigma_xz.norm() / pre_f[s].sigma_xz.norm()))
    checks["scrub_replay"] = rep
    log(f"scrub replay {rep}")

    # 4. evaluation: every condition, natural recomputed in the same batch
    # erasers.npz: rank-1 vectors of every eraser
    vec = {}
    for cname, ers in (("single", single), ("scrub", scrub)):
        for part, i in (("edit", 0), ("read", 1), ("bias", 2)):
            vec[f"{cname}_{part}"] = np.stack([np.stack([rank1(ers[s][f])[i].double().cpu().numpy() for s in sites])
                                               for f in range(F)])  # [F, 24, d]
    np.savez_compressed(out / "erasers.npz", **vec)

    conds = [f"single_{s}" for s in sites] + ["scrub_all", "scrub_0_8"]
    if args.cumulative:
        conds += [f"scrub_0_{k}" for k in sites if k not in (DAS_SITE, N_SITES - 1)]
    lp_by = {c: np.zeros(N) for c in ["natural"] + conds}
    kl = {c: np.zeros(N) for c in conds}
    scrub_states = {c: np.zeros((N, N_SITES, d), dtype=np.float32) for c in ("scrub_all", "scrub_0_8")}
    chk = {"self_patch_max_abs": 0.0, "outside_anchor_max_abs": 0.0, "single_offline_max_abs": 0.0,
           "natural_vs_pass_max_abs": 0.0}
    for bi, b in enumerate(batches):
        bp = [prompts[i] for i in b]
        ef = fold_t[torch.as_tensor(b, device=dev)]
        lp0, _ = M.forward(bp, {})
        lp_by["natural"][b] = lp0[:, M.by].cpu().numpy()
        chk["natural_vs_pass_max_abs"] = max(chk["natural_vs_pass_max_abs"],
                                             float(np.abs(lp_by["natural"][b] - lp_nat_pass[b]).max()))
        lpi, _ = M.forward(bp, {DAS_SITE: lambda v: v.clone()})
        chk["self_patch_max_abs"] = max(chk["self_patch_max_abs"], float((lpi - lp0).abs().max()))
        p0 = lp0.exp()
        for c in conds:
            cap = {}
            if c.startswith("single_"):
                s = int(c.split("_")[1])

                def h1(v, s=s):
                    new = apply_erasers(single[s], ef, v)
                    cap[s] = new.clone()
                    return new
                hooks = {s: h1}
            else:
                top = N_SITES - 1 if c == "scrub_all" else int(c.split("_")[2])

                def h2(v, s):
                    new = apply_erasers(scrub[s], ef, v) if s <= top else v
                    cap[s] = new.clone()
                    return None if s > top else new
                hooks = {s: (lambda v, s=s: h2(v, s)) for s in sites}
            lp, _ = M.forward(bp, hooks)
            lp_by[c][b] = lp[:, M.by].cpu().numpy()
            kl[c][b] = (p0 * (lp0 - lp)).sum(-1).cpu().numpy()
            if c.startswith("single_"):
                off = apply_erasers(single[s], ef, torch.as_tensor(nat[b, s], device=dev))
                chk["single_offline_max_abs"] = max(chk["single_offline_max_abs"], float((cap[s] - off).abs().max()))
            elif c in scrub_states:
                for s in sites:
                    scrub_states[c][b, s] = cap[s].cpu().numpy()
        if bi == 0:  # non-anchor positions untouched by the erasure hooks (scrub_all, every site)
            M.track_outside = True
            M.forward(bp, {s: (lambda v, s=s: apply_erasers(scrub[s], ef, v)) for s in sites})
            M.track_outside = False
            chk["outside_anchor_max_abs"] = M.outside_max
        if bi % 20 == 0:
            log(f"eval batch {bi + 1}/{len(batches)}")
    checks.update(chk)
    log(f"checks {chk}")

    # held-out class-mean difference under scrubbing (all items, their own fold's erasers)
    for c, X in scrub_states.items():
        for s in sites:
            for f in range(F):
                h = held_idx[f]
                diag.append({"condition": f"{c}_heldout", "site": s, "fold": f,
                             "held_meandiff_pre": held_meandiff(nat[:, s], h),
                             "held_meandiff_post": held_meandiff(X[:, s], h)})

    # 5. probes
    splits = inner_splits(fold_np, P.verb_pair.to_numpy(), args.n_inner, seed=1)
    fitset = [(0, 0, np.ones(N, bool), np.ones(N, bool))]  # in-sample only: train and score on all items
    prow = []
    for s in sites:
        Xs = {"natural": nat[:, s], "scrub_all": scrub_states["scrub_all"][:, s],
              "scrub_0_8": scrub_states["scrub_0_8"][:, s]}
        er = np.zeros((N, d), dtype=np.float32)
        for f in range(F):
            h = held_idx[f]
            er[h] = single[s][f](torch.as_tensor(nat[h, s], device=dev).double()).float().cpu().numpy()
        Xs["single"] = er
        for c, X in Xs.items():
            here = c == "single" or c == "scrub_all" or (c == "scrub_0_8" and s <= DAS_SITE)
            for kind in ("linear", "mlp"):
                prow.append({"condition": c, "site": s, "probe": kind, "erased_here": here,
                             "acc": probe_cv(X, cls_np, splits, kind, torch, dev)})
            if args.insample:
                prow.append({"condition": c, "site": s, "probe": "linear_fitset", "erased_here": here,
                             "acc": probe_cv(X, cls_np, fitset, "linear", torch, dev)})
        k = 12 if args.insample else 8
        log(f"probes site {s}: " + ", ".join(f"{r['condition']}/{r['probe']} {r['acc']:.3f}" for r in prow[-k:]))
    pd.DataFrame(prow).to_csv(out / "probes.csv", index=False)
    pd.DataFrame(diag).to_csv(out / "fit_diagnostics.csv", index=False)

    res = P.drop(columns=["prompt"]).copy()
    res["prompt"] = P.prompt
    res["lp_by_natural_pass"] = lp_nat_pass
    for c, v in lp_by.items():
        res[f"lp_by__{c}"] = v
    for c, v in kl.items():
        res[f"kl__{c}"] = v
    res.to_csv(out / "items.csv.gz", index=False)
    (out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    meta = {"args": vars(args), "runner_settings": M.R.settings, "n_prompts": N, "n_items": len(pas),
            "n_pairs": int(P.verb_pair.nunique()), "sites": sites, "das_site": DAS_SITE,
            "scores_sha256": sha(args.scores), "minutes": (time.time() - t0) / 60}
    import concept_erasure
    meta["concept_erasure_version"] = getattr(concept_erasure, "__version__", "0.2.4 (wheel)")
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    log(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default="results/passive_band_cross/pythia14b_scores.csv")
    ap.add_argument("--v2-pairs", default="data/matched_passives_v2/pairs.jsonl")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--n-inner", type=int, default=5, help="inner pair groups per outer fold for the probes")
    ap.add_argument("--max-pairs", type=int, default=0, help="smoke test: pairs per verb band")
    ap.add_argument("--max-contexts", type=int, default=0, help="smoke test: contexts kept")
    ap.add_argument("--cumulative", action="store_true", help="also evaluate every scrub prefix 0..k")
    ap.add_argument("--insample", action="store_true", help="one eraser set fit on all items, evaluated on them")
    ap.add_argument("--out-dir", default=None,
                    help="default results/pythia/leace_necessity (results/pythia/leace_insample with --insample)")
    a = ap.parse_args()
    a.out_dir = a.out_dir or ("results/pythia/leace_insample" if a.insample else "results/pythia/leace_necessity")
    run(a)
