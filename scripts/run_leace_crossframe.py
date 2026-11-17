#!/usr/bin/env python3
"""Cross-frame LEACE (rules: reports/pythia/leace_crossframe_rules.md): erasers fit on ACTIVE verb-position states,
applied to the passive items of the necessity run. Pythia-1.4B, fp32 (TF32 off).

Passive items, position, readout, sites, folds and fitter are run_leace_necessity.py's (imported). Active fitting
items: for each passive item, "The AGENT VERB-ed" with the agent of the same curated context (passive_1, same lemma)
and the verb's past form (PAST, else the participle), as task 1 of analyze_das_prep.py. Position: the verb's last
token (= last prompt token). Class T = 1, I = 0.

Fittings: "in" = one eraser set fit on all active prompts (applied to every passive item); "cf" = erasers for fold f
fit on the actives of the other folds (applied to fold f's passives). Each: single-site LEACE at sites 0..23 (natural
active states) and sequential scrubbing (eraser L fit on active site-L states with erasers 0..L-1 applied), evaluated
for every prefix 0..k. Permutation control: "in" with permuted verb labels (run_leace_permutation.permuted_labels),
scrub 0..12 and single site 8. Active-side check: in / cf erasers (single 8, scrub 0..12) on the actives, readout
log P(object start) (the DAS Runner's O set).

Outputs (--out-dir): items_passive.csv.gz (lp_by, kl per condition), items_active.csv.gz (lp_obj per condition),
erasers.npz, meandiff.csv (passive class-mean difference before / after, per condition and site), labels_perm.csv,
fit_diagnostics.csv, pairs_folds.csv, checks.json, meta.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_leace_necessity as nec
from analyze_das_prep import curated_items
from build_das_prep_prompts import PAST
from run_leace_permutation import permuted_labels

N_SITES, DAS_SITE, NEC_TOP = 24, 8, 12


def active_items(P, args):
    """One active prompt per passive prompt (same row order)."""
    scores = pd.read_csv(args.scores, low_memory=False)
    v2 = pd.DataFrame([json.loads(l) for l in Path(args.v2_pairs).open(encoding="utf-8")])
    v2 = v2[v2.frame_id == "was"]
    act = curated_items(scores, v2, "passive_1")
    act = act[act.own_context == 0]
    key = lambda c: c.split("/", 2)[2]  # curated/passive_k/<band>/<lemma> -> <band>/<lemma>
    agent = act.drop_duplicates("context_id").set_index(act.drop_duplicates("context_id").context_id.map(key)).agent
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
    from concept_erasure import LeaceFitter

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
    if args.ref_folds and Path(args.ref_folds).exists():  # same verb split as the stored necessity run
        R = pd.read_csv(args.ref_folds).set_index("verb_pair").fold
        checks["folds_equal_stored"] = bool((pairs.set_index("verb_pair").fold.reindex(R.index) == R).all()
                                            and len(R) == len(pairs))
        assert checks["folds_equal_stored"]
    A = active_items(P, args)
    A_tok = [len(tok.encode(p, add_special_tokens=False)) for p in A.prompt]
    N, sites = len(P), list(range(N_SITES))
    cls_np, fold_np = P.cls.to_numpy(), P.fold.to_numpy()
    z_true = torch.as_tensor(cls_np, device=dev, dtype=torch.float64)
    pprompts, aprompts = P.prompt.tolist(), A.prompt.tolist()
    obj = M.R.ids["O"]
    log(f"{N} prompts per frame, {P.verb_pair.nunique()} pairs; active e.g. {aprompts[:2]}; O set {len(obj)} ids")

    def batches_of(idx, ntok):
        idx = np.asarray(idx)
        idx = idx[np.argsort(np.asarray(ntok)[idx], kind="stable")]
        return [idx[i:i + args.batch_size] for i in range(0, len(idx), args.batch_size)]
    a_batches = batches_of(np.arange(N), A_tok)
    p_batches = batches_of(np.arange(N), P.n_tokens.to_numpy())

    # 1. natural active pass: states at every site, log P(object start)
    natA = np.zeros((N, N_SITES, d), dtype=np.float32)
    lpobj_nat = np.zeros(N)
    for b in a_batches:
        cap = {}
        lp, _ = M.forward([aprompts[i] for i in b], {s: (lambda v, s=s: cap.__setitem__(s, v.clone())) for s in sites})
        for s in sites:
            natA[b, s] = cap[s].cpu().numpy()
        lpobj_nat[b] = torch.logsumexp(lp[:, obj], -1).cpu().numpy()
    log("natural active pass done")

    modes = {"in": (np.zeros(N, dtype=np.int64), 1), "cf": (fold_np, args.n_folds)}
    diag, single, scrub, fit_moments = [], {}, {}, {}

    def fit_single(name, F, train, z, site_list):
        single[name] = {s: {} for s in site_list}
        for s in site_list:
            for f in range(F):
                idx = train[f]
                fitter = LeaceFitter(d, 1, device=dev, dtype=torch.float64)
                for i in range(0, len(idx), 4096):
                    j = idx[i:i + 4096]
                    fitter.update(torch.as_tensor(natA[j, s], device=dev).double(), z[j])
                E, st = nec.fit_stats(fitter, torch)
                single[name][s][f] = E
                diag.append({"fitting": name, "condition": "single", "site": s, "fold": f,
                             "n_verbs": int(P.verb.iloc[idx].nunique()), **st})

    def fit_scrub(name, F, train, z, top):
        scrub[name] = {s: {} for s in range(top + 1)}
        for f in range(F):
            bl = batches_of(train[f], A_tok)
            for L in range(top + 1):
                fitter = LeaceFitter(d, 1, device=dev, dtype=torch.float64)
                for b in bl:
                    cap = {}
                    ef = torch.full((len(b),), f, device=dev)
                    hooks = {s: (lambda v, s=s: nec.apply_erasers(scrub[name][s], ef, v)) for s in range(L)}

                    def capture(v):
                        cap["x"] = v.clone()
                        raise nec.Stop
                    hooks[L] = capture
                    M.forward([aprompts[i] for i in b], hooks)
                    fitter.update(cap["x"].double(), z[torch.as_tensor(b, device=dev)])
                E, st = nec.fit_stats(fitter, torch)
                scrub[name][L][f] = E
                fit_moments[(name, f, L)] = (fitter.mean_x.clone(), fitter.sigma_xz.clone())
                diag.append({"fitting": name, "condition": "scrub", "site": L, "fold": f,
                             "n_verbs": int(P.verb.iloc[train[f]].nunique()), **st})
        log(f"scrub {name} fit (sites 0..{top})")

    def replay(name, F, train, z, top):
        """All erasers 0..top on the active training items: pre-erasure moments equal those fit on."""
        rep = {"mean_maxabs": 0.0, "xcov_rel": 0.0, "post_xcov_resid_max": 0.0}
        for f in range(F):
            pre = {s: LeaceFitter(d, 1, device=dev, dtype=torch.float64, shrinkage=False) for s in range(top + 1)}
            post = {s: LeaceFitter(d, 1, device=dev, dtype=torch.float64, shrinkage=False) for s in range(top + 1)}
            for b in batches_of(train[f], A_tok):
                ef = torch.full((len(b),), f, device=dev)
                zb = z[torch.as_tensor(b, device=dev)]

                def hk(v, s, zb=zb, ef=ef):
                    pre[s].update(v.double(), zb)
                    new = nec.apply_erasers(scrub[name][s], ef, v)
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

    for name, (fnp, F) in modes.items():
        train = {f: np.flatnonzero(fnp != f) for f in range(F)} if F > 1 else {0: np.arange(N)}
        fit_single(name, F, train, z_true, sites)
        fit_scrub(name, F, train, z_true, N_SITES - 1)
        checks[f"scrub_replay_{name}"] = replay(name, F, train, z_true, N_SITES - 1)
        log(f"{name}: replay {checks[f'scrub_replay_{name}']}")
    lab_rows, perm_r = [], []
    for j in range(args.n_perm):
        verbs, zp = permuted_labels(P, args.perm_seed_base + j)
        perm_r.append(float(abs(np.corrcoef(zp, cls_np)[0, 1])))
        lab_rows += [{"perm": j, "verb": v, "cls": c, "label": l} for v, c, l in verbs.itertuples(index=False)]
        zt = torch.as_tensor(zp, device=dev, dtype=torch.float64)
        fit_single(f"perm{j}", 1, {0: np.arange(N)}, zt, [DAS_SITE])
        fit_scrub(f"perm{j}", 1, {0: np.arange(N)}, zt, NEC_TOP)
    checks["perm_max_abs_r"] = max(perm_r) if perm_r else None
    # with all 126 pairs the flips are exactly balanced (r = 0); smoke subsets have too few verbs per class
    assert not perm_r or args.max_pairs or max(perm_r) < 0.05
    pd.DataFrame(lab_rows).to_csv(out / "labels_perm.csv", index=False)

    # erasers.npz: rank-1 vectors (true-label fittings)
    vec = {}
    for name, (fnp, F) in modes.items():
        for cname, ers in (("single", single[name]), ("scrub", scrub[name])):
            for part, i in (("edit", 0), ("read", 1), ("bias", 2)):
                vec[f"{name}_{cname}_{part}"] = np.stack([np.stack([nec.rank1(ers[s][f])[i].double().cpu().numpy()
                                                                    for s in sites]) for f in range(F)])
    np.savez_compressed(out / "erasers.npz", **vec)

    # 2. passive evaluation
    conds = []  # (column name, fitting, kind, site or top)
    for name in modes:
        conds += [(f"{name}_single_{s}", name, "single", s) for s in sites]
        conds += [(f"{name}_scrub_0_{k}", name, "scrub", k) for k in sites]
    for j in range(args.n_perm):
        conds += [(f"perm{j}_single_{DAS_SITE}", f"perm{j}", "single", DAS_SITE),
                  (f"perm{j}_scrub_0_{NEC_TOP}", f"perm{j}", "scrub", NEC_TOP)]
    efold = {name: torch.as_tensor(fnp, device=dev) for name, (fnp, F) in modes.items()}
    lp_by = {c[0]: np.zeros(N) for c in conds}
    lp_by["natural"] = np.zeros(N)
    kl = {c[0]: np.zeros(N) for c in conds}
    # class sums of passive states for the mean-difference ratio: natural and per condition at the erased sites
    md_keys = [(c, s) for c, nm, kd, k in conds if not nm.startswith("perm")
               for s in ([k] if kd == "single" else range(k + 1)) if kd == "single" or k in (NEC_TOP, N_SITES - 1)]
    md = {key: np.zeros((2, d)) for key in md_keys + [("natural", s) for s in sites]}
    ncls = np.array([(cls_np == 0).sum(), (cls_np == 1).sum()])
    chk = {"self_patch_max_abs": 0.0, "outside_anchor_max_abs": 0.0, "single_offline_max_abs": 0.0}
    for bi, b in enumerate(p_batches):
        bp = [pprompts[i] for i in b]
        bt = torch.as_tensor(b, device=dev)
        cb = cls_np[b]
        capn = {}
        lp0, _ = M.forward(bp, {s: (lambda v, s=s: capn.__setitem__(s, v.clone())) for s in sites})
        lp_by["natural"][b] = lp0[:, M.by].cpu().numpy()
        for s in sites:
            x = capn[s].double().cpu().numpy()
            for c in (0, 1):
                md[("natural", s)][c] += x[cb == c].sum(0)
        lpi, _ = M.forward(bp, {DAS_SITE: lambda v: v.clone()})
        chk["self_patch_max_abs"] = max(chk["self_patch_max_abs"], float((lpi - lp0).abs().max()))
        p0 = lp0.exp()
        for col, name, kind, k in conds:
            ef = efold[name][bt] if name in efold else torch.zeros(len(b), dtype=torch.long, device=dev)
            cap = {}
            if kind == "single":
                def h1(v, s=k):
                    new = nec.apply_erasers(single[name][s], ef, v)
                    cap[s] = new.clone()
                    return new
                hooks = {k: h1}
            else:
                def h2(v, s):
                    new = nec.apply_erasers(scrub[name][s], ef, v)
                    cap[s] = new.clone()
                    return new
                hooks = {s: (lambda v, s=s: h2(v, s)) for s in range(k + 1)}
            lp, _ = M.forward(bp, hooks)
            lp_by[col][b] = lp[:, M.by].cpu().numpy()
            kl[col][b] = (p0 * (lp0 - lp)).sum(-1).cpu().numpy()
            if kind == "single":
                off = nec.apply_erasers(single[name][k], ef, capn[k])
                chk["single_offline_max_abs"] = max(chk["single_offline_max_abs"], float((cap[k] - off).abs().max()))
            for s in cap:
                if (col, s) in md:
                    x = cap[s].double().cpu().numpy()
                    for c in (0, 1):
                        md[(col, s)][c] += x[cb == c].sum(0)
        if bi == 0:
            M.track_outside = True
            ef = efold["cf"][bt]
            M.forward(bp, {s: (lambda v, s=s: nec.apply_erasers(scrub["cf"][s], ef, v)) for s in sites})
            M.track_outside = False
            chk["outside_anchor_max_abs"] = M.outside_max
        if bi % 10 == 0:
            log(f"passive eval batch {bi + 1}/{len(p_batches)}")
    checks.update(chk)
    log(f"checks {chk}")
    mrows = []
    for (col, s), v in md.items():
        diff = v[1] / ncls[1] - v[0] / ncls[0]
        nat = md[("natural", s)][1] / ncls[1] - md[("natural", s)][0] / ncls[0]
        mrows.append({"condition": col, "site": s, "meandiff_norm": float(np.linalg.norm(diff)),
                      "natural_norm": float(np.linalg.norm(nat)),
                      "ratio": float(np.linalg.norm(diff) / np.linalg.norm(nat)),
                      "cos_with_natural": float(diff @ nat / (np.linalg.norm(diff) * np.linalg.norm(nat) + 1e-30))})
    pd.DataFrame(mrows).to_csv(out / "meandiff.csv", index=False)

    # 3. active-side check: in / cf erasers on the actives (cf: each fold's held-out actives)
    aconds = [(f"{name}_{kind}_{k if kind == 'single' else '0_' + str(k)}", name, kind, k)
              for name in modes for kind, k in (("single", DAS_SITE), ("scrub", NEC_TOP))]
    lpobj = {c[0]: np.zeros(N) for c in aconds}
    lpobj_nat2 = np.zeros(N)
    for b in a_batches:
        ba = [aprompts[i] for i in b]
        bt = torch.as_tensor(b, device=dev)
        lp0, _ = M.forward(ba, {})
        lpobj_nat2[b] = torch.logsumexp(lp0[:, obj], -1).cpu().numpy()
        for col, name, kind, k in aconds:
            ef = efold[name][bt]
            if kind == "single":
                hooks = {k: (lambda v: nec.apply_erasers(single[name][k], ef, v))}
            else:
                hooks = {s: (lambda v, s=s: nec.apply_erasers(scrub[name][s], ef, v)) for s in range(k + 1)}
            lp, _ = M.forward(ba, hooks)
            lpobj[col][b] = torch.logsumexp(lp[:, obj], -1).cpu().numpy()
    checks["active_natural_vs_pass_max_abs"] = float(np.abs(lpobj_nat2 - lpobj_nat).max())
    log("active check done")

    res = P.copy()
    for c, v in lp_by.items():
        res[f"lp_by__{c}"] = v
    for c, v in kl.items():
        res[f"kl__{c}"] = v
    res.to_csv(out / "items_passive.csv.gz", index=False)
    ra = A.copy()
    ra["lp_obj__natural"] = lpobj_nat
    for c, v in lpobj.items():
        ra[f"lp_obj__{c}"] = v
    ra.to_csv(out / "items_active.csv.gz", index=False)
    pd.DataFrame(diag).to_csv(out / "fit_diagnostics.csv", index=False)
    (out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    import concept_erasure
    meta = {"args": vars(args), "runner_settings": M.R.settings, "n_prompts": N, "n_pairs": int(P.verb_pair.nunique()),
            "sites": sites, "das_site": DAS_SITE, "scores_sha256": nec.sha(args.scores),
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
    ap.add_argument("--n-perm", type=int, default=10)
    ap.add_argument("--perm-seed-base", type=int, default=0)
    ap.add_argument("--ref-folds", default="", help="stored pairs_folds.csv of the necessity run (must match)")
    ap.add_argument("--max-pairs", type=int, default=0, help="smoke test: pairs per verb band")
    ap.add_argument("--max-contexts", type=int, default=0, help="smoke test: contexts kept")
    ap.add_argument("--out-dir", default="results/pythia/leace_crossframe")
    run(ap.parse_args())
