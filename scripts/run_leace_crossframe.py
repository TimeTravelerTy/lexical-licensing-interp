#!/usr/bin/env python3
"""Cross-frame LEACE (rules: reports/pythia/leace_crossframe_rules.md, amended 2026-11-18): erasers fit on ACTIVE
verb-position states, applied to the passive items of the necessity run. Pythia-1.4B, fp32 (TF32 off).

Passive items, position, readout, sites, folds and fitter are run_leace_necessity.py's (imported). Active fitting
items: for each passive item, "The AGENT VERB-ed" with the agent of the same curated context (passive_1, same lemma)
and the verb's past form (PAST, else the participle), as task 1 of analyze_das_prep.py. Position: the verb's last
token (= last prompt token). Class T = 1, I = 0.

Fittings: "in" = one eraser set fit on all active prompts (applied to every passive item); "cf" = erasers for fold f
fit on the actives of the other folds (applied to fold f's passives). Each: single-site LEACE at sites 0..23 (natural
active states) and sequential scrubbing (eraser L fit on active site-L states with erasers 0..L-1 applied).

Variants on the passives (an active-fit eraser x' = x - e r.(x - mu_A) shifts every passive state by the constant
-e r.(mu_P - mu_A), which can move the readout without touching the class contrast):
  c (primary)  centered: bias replaced by the label-free passive mean of the prompts the eraser is applied to (per
               fold for cf), at that site; for scrubbing, the mean under the centered erasers of the earlier sites
               (computed sequentially on the passives)
  u            uncentered: the eraser as fit (active mean)
  t            translation only: x + (E_u(mu_P) - mu_P), the constant shift alone (scrub: mu_P from the c sequence)
Every variant at single sites 0..23 and scrub prefixes 0..k (k = 0..23).

Permutation control: "in" fittings with permuted verb labels (run_leace_permutation.permuted_labels), variants c and
u, scrub 0..12 and single site 8. Active-side check: in / cf erasers as fit (single 8, scrub 0..12) on the actives
(cf: each fold's held-out actives), readout log P(object start) over analyze_das_prep.OBJ.

Outputs (--out-dir): items_passive.csv.gz (lp_by, kl per condition), items_active.csv.gz (lp_obj per condition),
erasers.npz, meandiff.csv (passive class-mean difference per condition, site and fold), edit_norms.csv,
labels_perm.csv, fit_diagnostics.csv, pairs_folds.csv, checks.json, meta.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_leace_necessity as nec
from analyze_das_prep import OBJ, curated_items
from build_das_prep_prompts import PAST
from run_leace_permutation import permuted_labels

N_SITES, DAS_SITE, NEC_TOP = 24, 8, 12


def active_items(P, args):
    """One active prompt per passive prompt (same row order)."""
    scores = pd.read_csv(args.scores, low_memory=False)
    v2 = pd.DataFrame([json.loads(l) for l in Path(args.v2_pairs).open(encoding="utf-8")])
    v2 = v2[v2.frame_id == "was"]
    act = curated_items(scores, v2, "passive_1")
    act = act[act.own_context == 0].drop_duplicates("context_id")
    key = lambda c: c.split("/", 2)[2]  # curated/passive_k/<band>/<lemma> -> <band>/<lemma>
    agent = pd.Series(act.agent.to_numpy(), index=act.context_id.map(key).to_numpy())
    assert agent.index.is_unique
    k = P.context_id.map(key)
    missing = sorted(set(k) - set(agent.index))
    assert not missing, f"no passive_1 agent for contexts {missing[:5]}"
    A = P.copy()
    A["agent"] = k.map(agent).to_numpy()
    assert A.agent.notna().all() and (A.agent.str.len() > 0).all()
    A["verb_past"] = A.verb.map(lambda v: PAST.get(v, v))
    A["prompt"] = "The " + A.agent + " " + A.verb_past
    return A


def run(args):
    import torch
    from concept_erasure import LeaceEraser, LeaceFitter

    args.dtype = "float32"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    M = nec.Model(args)
    dev, d, tok = M.dev, M.d, M.R.tok
    pas, P, pairs = nec.load_items(args, tok)
    pairs.to_csv(out / "pairs_folds.csv", index=False)
    checks = {}
    if args.ref_folds:  # same verb split as the stored necessity run
        R = pd.read_csv(args.ref_folds).set_index("verb_pair").fold  # fails loudly if the path is wrong
        checks["folds_equal_stored"] = bool(len(R) == len(pairs) and
                                            (pairs.set_index("verb_pair").fold.reindex(R.index) == R).all())
        assert checks["folds_equal_stored"]
    elif not args.max_pairs:
        raise SystemExit("--ref-folds is required for a full run")
    A = active_items(P, args)
    A_tok = [len(tok.encode(p, add_special_tokens=False)) for p in A.prompt]
    N, sites, F = len(P), list(range(N_SITES)), args.n_folds
    cls_np, fold_np = P.cls.to_numpy().copy(), P.fold.to_numpy().copy()
    z_true = torch.as_tensor(cls_np, device=dev, dtype=torch.float64)
    pprompts, aprompts = P.prompt.tolist(), A.prompt.tolist()
    obj_ids = []
    for w in OBJ:
        t = tok.encode(w, add_special_tokens=False)
        assert len(t) == 1, (w, t)
        obj_ids.append(t[0])
    obj = torch.tensor(obj_ids, device=dev)
    log(f"{N} prompts per frame, {P.verb_pair.nunique()} pairs; active e.g. {aprompts[:2]}; OBJ {len(obj_ids)} ids")

    def batches_of(idx, ntok):
        idx = np.asarray(idx)
        idx = idx[np.argsort(np.asarray(ntok)[idx], kind="stable")]
        return [idx[i:i + args.batch_size] for i in range(0, len(idx), args.batch_size)]
    a_batches = batches_of(np.arange(N), A_tok)
    p_batches = batches_of(np.arange(N), P.n_tokens.to_numpy())
    efold = {"in": np.zeros(N, dtype=np.int64), "cf": fold_np}  # eraser index per prompt (both frames)
    nfit = {"in": 1, "cf": F}

    # 1. natural active pass: states at every site, log P(object start)
    natA = np.zeros((N, N_SITES, d), dtype=np.float32)
    lpobj_nat = np.zeros(N)
    for b in a_batches:
        cap = {}
        lp, _ = M.forward([aprompts[i] for i in b], {s: (lambda v, s=s: cap.__setitem__(s, v.clone())) for s in sites})
        for s in sites:
            natA[b, s] = cap[s].cpu().numpy()
        lpobj_nat[b] = torch.logsumexp(lp[:, obj], -1).cpu().numpy()
    # natural passive sums per fold (label-free centering means)
    psum = np.zeros((F, N_SITES, d))
    for b in p_batches:
        cap = {}
        M.forward([pprompts[i] for i in b], {s: (lambda v, s=s: cap.__setitem__(s, v.clone())) for s in sites},
                  logits=False)
        for s in sites:
            x = cap[s].double().cpu().numpy()
            for f in range(F):
                psum[f, s] += x[fold_np[b] == f].sum(0)
    pcount = np.bincount(fold_np, minlength=F).astype(float)
    log("natural passes done")

    def passive_mean(name, f, s):
        """Label-free mean of the natural passive states the eraser (name, f) is applied to."""
        if nfit.get(name, 1) == 1:
            return psum[:, s].sum(0) / pcount.sum()
        return psum[f, s] / pcount[f]

    diag, ers, shifts, fit_moments = [], {}, {}, {}

    def fit_single(name, Fn, train, z, site_list):
        ers[(name, "u", "single")] = {s: {} for s in site_list}
        for s in site_list:
            for f in range(Fn):
                idx = train[f]
                fitter = LeaceFitter(d, 1, device=dev, dtype=torch.float64)
                for i in range(0, len(idx), 4096):
                    j = idx[i:i + 4096]
                    fitter.update(torch.as_tensor(natA[j, s], device=dev).double(), z[j])
                E, st = nec.fit_stats(fitter, torch)
                ers[(name, "u", "single")][s][f] = E
                diag.append({"fitting": name, "condition": "single", "site": s, "fold": f,
                             "n_verbs": int(P.verb.iloc[idx].nunique()), **st})

    def fit_scrub(name, Fn, train, z, top):
        ers[(name, "u", "scrub")] = sc = {s: {} for s in range(top + 1)}
        for f in range(Fn):
            bl = batches_of(train[f], A_tok)
            for L in range(top + 1):
                fitter = LeaceFitter(d, 1, device=dev, dtype=torch.float64)
                for b in bl:
                    cap = {}
                    ef = torch.full((len(b),), f, device=dev)
                    hooks = {s: (lambda v, s=s: nec.apply_erasers(sc[s], ef, v)) for s in range(L)}

                    def capture(v):
                        cap["x"] = v.clone()
                        raise nec.Stop
                    hooks[L] = capture
                    M.forward([aprompts[i] for i in b], hooks)
                    fitter.update(cap["x"].double(), z[torch.as_tensor(b, device=dev)])
                E, st = nec.fit_stats(fitter, torch)
                sc[L][f] = E
                fit_moments[(name, f, L)] = (fitter.mean_x.clone(), fitter.sigma_xz.clone())
                diag.append({"fitting": name, "condition": "scrub", "site": L, "fold": f,
                             "n_verbs": int(P.verb.iloc[train[f]].nunique()), **st})
        log(f"scrub {name} fit (sites 0..{top})")

    def replay(name, Fn, train, z, top):
        """All erasers 0..top on the active training items: pre-erasure moments equal those fit on."""
        sc = ers[(name, "u", "scrub")]
        rep = {"mean_maxabs": 0.0, "xcov_rel": 0.0, "post_xcov_resid_max": 0.0}
        for f in range(Fn):
            pre = {s: LeaceFitter(d, 1, device=dev, dtype=torch.float64, shrinkage=False) for s in range(top + 1)}
            post = {s: LeaceFitter(d, 1, device=dev, dtype=torch.float64, shrinkage=False) for s in range(top + 1)}
            for b in batches_of(train[f], A_tok):
                ef = torch.full((len(b),), f, device=dev)
                zb = z[torch.as_tensor(b, device=dev)]

                def hk(v, s, zb=zb, ef=ef):
                    pre[s].update(v.double(), zb)
                    new = nec.apply_erasers(sc[s], ef, v)
                    post[s].update(new.double(), zb)
                    return new
                M.forward([aprompts[i] for i in b], {s: (lambda v, s=s: hk(v, s)) for s in range(top + 1)},
                          logits=False)
            for s in range(top + 1):
                mx, sxz = fit_moments[(name, f, s)]
                rep["mean_maxabs"] = max(rep["mean_maxabs"], float((pre[s].mean_x - mx).abs().max()))
                rep["xcov_rel"] = max(rep["xcov_rel"], float((pre[s].sigma_xz - sxz).norm() / sxz.norm()))
                rep["post_xcov_resid_max"] = max(rep["post_xcov_resid_max"],
                                                 float(post[s].sigma_xz.norm() / pre[s].sigma_xz.norm()))
        return rep

    def recenter(E, mu):
        return LeaceEraser(E.proj_left, E.proj_right, torch.as_tensor(mu, device=dev, dtype=E.bias.dtype))

    def shift_of(E, mu):
        m = torch.as_tensor(mu, device=dev, dtype=torch.float64)
        return (E(m[None]) - m[None])[0]

    def center(name, Fn, top_single, top_scrub):
        """Variants c and t from the fitted (u) erasers; scrub means computed sequentially on the passives."""
        eu, su = ers[(name, "u", "single")], ers[(name, "u", "scrub")]
        ers[(name, "c", "single")] = {s: {f: recenter(eu[s][f], passive_mean(name, f, s)) for f in range(Fn)}
                                      for s in top_single}
        shifts[(name, "single")] = {s: torch.stack([shift_of(eu[s][f], passive_mean(name, f, s)) for f in range(Fn)])
                                    for s in top_single}
        ef_all = efold[name] if name in efold else np.zeros(N, dtype=np.int64)
        ers[(name, "c", "scrub")] = sc = {}
        shifts[(name, "scrub")] = {}
        for L in range(top_scrub + 1):
            msum, mcnt = np.zeros((Fn, d)), np.zeros(Fn)
            for b in p_batches:
                cap = {}
                ef = torch.as_tensor(ef_all[b], device=dev)
                hooks = {s: (lambda v, s=s: nec.apply_erasers(sc[s], ef, v)) for s in range(L)}

                def capture(v):
                    cap["x"] = v.clone()
                    raise nec.Stop
                hooks[L] = capture
                M.forward([pprompts[i] for i in b], hooks)
                x = cap["x"].double().cpu().numpy()
                for f in range(Fn):
                    m = ef_all[b] == f
                    msum[f] += x[m].sum(0)
                    mcnt[f] += m.sum()
            mu = msum / mcnt[:, None]
            sc[L] = {f: recenter(su[L][f], mu[f]) for f in range(Fn)}
            shifts[(name, "scrub")][L] = torch.stack([shift_of(su[L][f], mu[f]) for f in range(Fn)])
        log(f"{name}: centered variants built")

    for name in ("in", "cf"):
        Fn = nfit[name]
        train = {f: np.flatnonzero(fold_np != f) for f in range(Fn)} if Fn > 1 else {0: np.arange(N)}
        fit_single(name, Fn, train, z_true, sites)
        fit_scrub(name, Fn, train, z_true, N_SITES - 1)
        checks[f"scrub_replay_{name}"] = replay(name, Fn, train, z_true, N_SITES - 1)
        log(f"{name}: replay {checks[f'scrub_replay_{name}']}")
        center(name, Fn, sites, N_SITES - 1)
    lab_rows, perm_r = [], []
    for j in range(args.n_perm):
        name = f"perm{j}"
        verbs, zp = permuted_labels(P, args.perm_seed_base + j)
        perm_r.append(float(abs(np.corrcoef(zp, cls_np)[0, 1])))
        lab_rows += [{"perm": j, "verb": v, "cls": c, "label": l} for v, c, l in verbs.itertuples(index=False)]
        zt = torch.as_tensor(zp, device=dev, dtype=torch.float64)
        fit_single(name, 1, {0: np.arange(N)}, zt, [DAS_SITE])
        fit_scrub(name, 1, {0: np.arange(N)}, zt, NEC_TOP)
        center(name, 1, [DAS_SITE], NEC_TOP)
    checks["perm_max_abs_r"] = max(perm_r) if perm_r else None
    # with all 126 pairs the flips are exactly balanced (r = 0); smoke subsets have too few verbs per class
    assert not perm_r or args.max_pairs or max(perm_r) < 0.05
    pd.DataFrame(lab_rows).to_csv(out / "labels_perm.csv", index=False)

    vec = {}
    for name in ("in", "cf"):
        Fn = nfit[name]
        for kind in ("single", "scrub"):
            for part, i in (("edit", 0), ("read", 1), ("bias", 2)):
                vec[f"{name}_{kind}_{part}"] = np.stack([np.stack([nec.rank1(ers[(name, "u", kind)][s][f])[i]
                                                                   .double().cpu().numpy() for s in sites])
                                                         for f in range(Fn)])
            vec[f"{name}_{kind}_bias_centered"] = np.stack([np.stack([ers[(name, "c", kind)][s][f].bias.double()
                                                                      .cpu().numpy() for s in sites])
                                                            for f in range(Fn)])
    np.savez_compressed(out / "erasers.npz", **vec)

    # 2. passive evaluation
    conds = []  # (column, fitting, variant, kind, site or top)
    for name in ("in", "cf"):
        for var in ("c", "u", "t"):
            conds += [(f"{name}_{var}_single_{s}", name, var, "single", s) for s in sites]
            conds += [(f"{name}_{var}_scrub_0_{k}", name, var, "scrub", k) for k in sites]
    for j in range(args.n_perm):
        for var in ("c", "u"):
            conds += [(f"perm{j}_{var}_single_{DAS_SITE}", f"perm{j}", var, "single", DAS_SITE),
                      (f"perm{j}_{var}_scrub_0_{NEC_TOP}", f"perm{j}", var, "scrub", NEC_TOP)]
    lp_by = {c[0]: np.zeros(N) for c in conds}
    lp_by["natural"] = np.zeros(N)
    kl = {c[0]: np.zeros(N) for c in conds}
    edit = {c[0]: 0.0 for c in conds}
    n_edit = {c[0]: (1 if c[3] == "single" else c[4] + 1) * N for c in conds}
    md_keys = [(c, s) for c, nm, var, kd, k in conds if nm in ("in", "cf") and var in ("c", "u")
               for s in ([k] if kd == "single" else range(k + 1)) if kd == "single" or k in (NEC_TOP, N_SITES - 1)]
    md = {key: np.zeros((F, 2, d)) for key in md_keys + [("natural", s) for s in sites]}
    ncls = np.array([[((fold_np == f) & (cls_np == c)).sum() for c in (0, 1)] for f in range(F)], dtype=float)
    chk = {"self_patch_max_abs": 0.0, "outside_anchor_max_abs": 0.0, "single_offline_max_abs": 0.0}

    def add_md(key, x, cb, fb):
        for f in np.unique(fb):
            for c in (0, 1):
                md[key][f, c] += x[(fb == f) & (cb == c)].sum(0)

    for bi, b in enumerate(p_batches):
        bp = [pprompts[i] for i in b]
        cb, fb = cls_np[b], fold_np[b]
        capn = {}
        lp0, _ = M.forward(bp, {s: (lambda v, s=s: capn.__setitem__(s, v.clone())) for s in sites})
        lp_by["natural"][b] = lp0[:, M.by].cpu().numpy()
        for s in sites:
            add_md(("natural", s), capn[s].double().cpu().numpy(), cb, fb)
        lpi, _ = M.forward(bp, {DAS_SITE: lambda v: v.clone()})
        chk["self_patch_max_abs"] = max(chk["self_patch_max_abs"], float((lpi - lp0).abs().max()))
        p0 = lp0.exp()
        for col, name, var, kind, k in conds:
            ef = torch.as_tensor(efold[name][b] if name in efold else np.zeros(len(b), dtype=np.int64), device=dev)
            cap, en = {}, [0.0]
            site_list = [k] if kind == "single" else list(range(k + 1))

            def hk(v, s):
                if var == "t":
                    new = v + shifts[(name, kind)][s][ef].to(v.dtype)
                else:
                    new = nec.apply_erasers(ers[(name, var, kind)][s], ef, v)
                en[0] += float((new - v).norm(dim=1).sum())
                cap[s] = new.clone()
                return new
            lp, _ = M.forward(bp, {s: (lambda v, s=s: hk(v, s)) for s in site_list})
            lp_by[col][b] = lp[:, M.by].cpu().numpy()
            kl[col][b] = (p0 * (lp0 - lp)).sum(-1).cpu().numpy()
            edit[col] += en[0]
            if kind == "single" and var != "t":
                off = nec.apply_erasers(ers[(name, var, kind)][k], ef, capn[k])
                chk["single_offline_max_abs"] = max(chk["single_offline_max_abs"], float((cap[k] - off).abs().max()))
            for s in cap:
                if (col, s) in md:
                    add_md((col, s), cap[s].double().cpu().numpy(), cb, fb)
        if bi == 0:
            M.track_outside = True
            ef = torch.as_tensor(fold_np[b], device=dev)
            M.forward(bp, {s: (lambda v, s=s: nec.apply_erasers(ers[("cf", "c", "scrub")][s], ef, v)) for s in sites})
            M.track_outside = False
            chk["outside_anchor_max_abs"] = M.outside_max
        if bi % 10 == 0:
            log(f"passive eval batch {bi + 1}/{len(p_batches)}")
    checks.update(chk)
    log(f"checks {chk}")
    mrows = []
    for (col, s), v in md.items():
        nat = md[("natural", s)]
        for f in list(range(F)) + ["pooled"]:
            if f == "pooled":
                diff = v[:, 1].sum(0) / ncls[:, 1].sum() - v[:, 0].sum(0) / ncls[:, 0].sum()
                d0 = nat[:, 1].sum(0) / ncls[:, 1].sum() - nat[:, 0].sum(0) / ncls[:, 0].sum()
            else:
                diff = v[f, 1] / ncls[f, 1] - v[f, 0] / ncls[f, 0]
                d0 = nat[f, 1] / ncls[f, 1] - nat[f, 0] / ncls[f, 0]
            mrows.append({"condition": col, "site": s, "fold": f, "meandiff_norm": float(np.linalg.norm(diff)),
                          "natural_norm": float(np.linalg.norm(d0)),
                          "ratio": float(np.linalg.norm(diff) / np.linalg.norm(d0)),
                          "cos_with_natural": float(diff @ d0 / (np.linalg.norm(diff) * np.linalg.norm(d0) + 1e-30))})
    pd.DataFrame(mrows).to_csv(out / "meandiff.csv", index=False)
    pd.DataFrame([{"condition": c, "mean_edit_norm_per_site": edit[c] / n_edit[c]} for c in edit]).to_csv(
        out / "edit_norms.csv", index=False)

    # 3. active-side check: erasers as fit, on the actives (cf: each fold's held-out actives)
    aconds = [(f"{name}_u_{kind}_{k if kind == 'single' else '0_' + str(k)}", name, kind, k)
              for name in ("in", "cf") for kind, k in (("single", DAS_SITE), ("scrub", NEC_TOP))]
    lpobj = {c[0]: np.zeros(N) for c in aconds}
    lpobj_nat2 = np.zeros(N)
    for b in a_batches:
        ba = [aprompts[i] for i in b]
        lp0, _ = M.forward(ba, {})
        lpobj_nat2[b] = torch.logsumexp(lp0[:, obj], -1).cpu().numpy()
        for col, name, kind, k in aconds:
            ef = torch.as_tensor(efold[name][b], device=dev)
            e_ = ers[(name, "u", kind)]
            site_list = [k] if kind == "single" else list(range(k + 1))
            lp, _ = M.forward(ba, {s: (lambda v, s=s: nec.apply_erasers(e_[s], ef, v)) for s in site_list})
            lpobj[col][b] = torch.logsumexp(lp[:, obj], -1).cpu().numpy()
    checks["active_natural_vs_pass_max_abs"] = float(np.abs(lpobj_nat2 - lpobj_nat).max())
    log("active check done")

    cols = {**{f"lp_by__{c}": v for c, v in lp_by.items()}, **{f"kl__{c}": v for c, v in kl.items()}}
    pd.concat([P, pd.DataFrame(cols, index=P.index)], axis=1).to_csv(out / "items_passive.csv.gz", index=False)
    cols = {"lp_obj__natural": lpobj_nat, **{f"lp_obj__{c}": v for c, v in lpobj.items()}}
    pd.concat([A, pd.DataFrame(cols, index=A.index)], axis=1).to_csv(out / "items_active.csv.gz", index=False)
    pd.DataFrame(diag).to_csv(out / "fit_diagnostics.csv", index=False)
    (out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    import concept_erasure
    meta = {"args": vars(args), "runner_settings": M.R.settings, "n_prompts": N, "n_pairs": int(P.verb_pair.nunique()),
            "sites": sites, "das_site": DAS_SITE, "scores_sha256": nec.sha(args.scores),
            "obj_tokens": list(OBJ), "obj_ids": obj_ids, "smoke": bool(args.max_pairs or args.max_contexts),
            "concept_erasure_version": getattr(concept_erasure, "__version__", "0.2.4 (wheel)"),
            "minutes": (time.time() - t0) / 60}
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
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--perm-seed-base", type=int, default=0)
    ap.add_argument("--ref-folds", default="", help="stored pairs_folds.csv of the necessity run (required, full run)")
    ap.add_argument("--max-pairs", type=int, default=0, help="smoke test: pairs per verb band")
    ap.add_argument("--max-contexts", type=int, default=0, help="smoke test: contexts kept")
    ap.add_argument("--out-dir", default="results/pythia/leace_crossframe")
    run(ap.parse_args())
