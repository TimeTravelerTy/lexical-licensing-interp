#!/usr/bin/env python3
"""Split-edit test, frame invariance and passive-axis patching (Pythia-1.4B; rules: reports/pythia/split_edit_plan.md).

A. Per site (4, 6, 8, 12, 17) and final-run basis: v = a u + w (u = u_md of the basis's training items, w ⊥ u).
   The v interchange h <- h + delta v (delta = v.(h_src - h_base), in-batch) is split into its on-axis piece
   (write a u) and off-axis piece (write w); conditions full / on / off on the direction check's passive plan rows
   and on the held-out active swaps. Push sizes against the natural spread along u and w-hat.
B. Per-verb mean verb-position states at sites 0-23 for the LEACE items (passive `was` frame) and the active DAS
   items (class_states.npz; cosines are computed in the analysis).
C. Interchange along the passive class axis u_pas, cross-fitted (passives: 5 groups of primary pairs, each direction
   fit without the group's verbs and without the DAS verbs; actives: fit without the DAS verbs).

Outputs in --out-dir (see the rules). Run from scripts/ (paths default to ../).
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
from run_das_direction_check import FOLD_COLS, load_bases
from run_passive_controls import upgrade
from run_passive_test import READ

N_SITES_B = 24


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patched_rw(runner, prompts, src_h, site, read, write, full=True):
    """Runner.patched with the displacement read along `read` and written along `write` ([d, 1] each):
    h <- h + ((h_src - h) @ read) @ write.T at the last token. write = read reproduces Runner.patched."""
    torch = runner.torch
    enc, anchors = runner.encode(prompts)
    rows = torch.arange(len(anchors), device=runner.device)

    def patch(hidden):
        h = hidden.float()
        base = h[rows, anchors]
        delta = ((src_h - base) @ read) @ write.T
        h = h.clone()
        h[rows, anchors] = base + delta
        return h.to(hidden.dtype)

    if site == 0:
        handle = runner.embed.register_forward_hook(lambda m, i, o: patch(o))
    else:
        handle = runner.layers[site - 1].register_forward_hook(
            lambda m, i, o: r2.replace_output(o, patch(r2.normalize_output(o))))
    try:
        with torch.no_grad():
            out = runner.model(**enc, use_cache=False)
    finally:
        handle.remove()
    return runner.readout(out.logits, anchors, full=full)


def unit(x):
    return x / np.linalg.norm(x)


def smd(x, y):
    """Standardized mean difference (pooled sd)."""
    sp = np.sqrt((x.var(ddof=1) * (len(x) - 1) + y.var(ddof=1) * (len(y) - 1)) / (len(x) + len(y) - 2))
    return float((x.mean() - y.mean()) / sp)


def leace_items(args, tok):
    from run_leace_necessity import load_items
    la = argparse.Namespace(scores=args.scores, v2_pairs=args.v2_pairs, max_pairs=args.max_pairs,
                            max_contexts=args.max_contexts, n_folds=5, split_seed=0)
    _, P, pairs = load_items(la, tok)
    return P, pairs


def run(args):
    import torch

    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sites = [int(s) for s in args.sites.split(",")]
    root = Path(args.root)
    smoke = bool(args.max_items or args.max_pairs or args.max_contexts)

    runner, pairs, items, reps, nat, dropped = r2.setup(args)
    runner = upgrade(runner)
    dev, d = runner.device, reps.shape[-1]
    stored_items = pd.read_csv(root / "final_strict" / "items.csv")
    assert stored_items.prompt.tolist() == items.prompt.tolist(), "active items differ from the final runs'"
    labels = items.cls.to_numpy()
    prompts_a = items.prompt.tolist()
    ref_folds = pd.read_csv(root / "final_strict" / "pairs_folds.csv")[FOLD_COLS] \
        .sort_values("pair_id").reset_index(drop=True)
    assigns = {k: r2.folds(pairs, args.n_folds, k) for k in range(3)}
    for k in range(3):
        assert all(ref_folds.set_index("pair_id")[f"fold_split{k}"].to_dict()[p] == f for p, f in assigns[k].items())
    all_pairs = pd.read_csv(args.pairs)  # every DAS pair, including any dropped by the behaviour filter
    das_parts = set(all_pairs.trans_part) | set(all_pairs.intrans_part)
    checks = {"items": len(items), "pairs": len(pairs), "dropped_pairs": dropped, "smoke": smoke, "sites": {}}
    meta = {"args": vars(args), "runner_settings": runner.settings, "bases_sha256": {},
            "leace_insample_sha256": sha(args.leace_insample), "plan_sha256": sha(args.plan)}

    # ---------------------------------------------------------------- B. per-verb mean states, both frames
    P, lpairs = leace_items(args, runner.tok)
    pverbs = P.drop_duplicates("verb")[["verb", "cls", "verb_pair"]].reset_index(drop=True)
    vix = {v: i for i, v in enumerate(pverbs.verb)}
    cnt = P.groupby("verb").size()
    checks["leace_items"] = {"prompts": len(P), "pairs": int(P.verb_pair.nunique()),
                             "contexts": int(P.context_id.nunique()), "prompts_per_verb": sorted(set(cnt.tolist()))}
    if not smoke:
        assert len(P) == 31500 and P.verb_pair.nunique() == 126 and set(cnt) == {125}, checks["leace_items"]  # 125 other-verb contexts per verb
    sums = torch.zeros(len(pverbs), N_SITES_B, d, dtype=torch.float64, device=dev)
    vidx = torch.as_tensor(P.verb.map(vix).to_numpy(), device=dev)
    pl = P.prompt.tolist()
    with torch.no_grad():
        for i in range(0, len(pl), args.batch):
            enc, anchors = runner.encode(pl[i:i + args.batch])
            o = runner.model(**enc, output_hidden_states=True, use_cache=False)
            rows = torch.arange(len(anchors), device=dev)
            H = torch.stack([o.hidden_states[s][rows, anchors].double() for s in range(N_SITES_B)], 1)
            sums.index_add_(0, vidx[i:i + args.batch], H)
    pas_mean = (sums / torch.as_tensor(cnt[pverbs.verb].to_numpy(), device=dev)[:, None, None]).cpu().numpy()
    averbs = items.drop_duplicates("verb")[["verb", "cls", "pair_id"]].reset_index(drop=True)
    act_mean = np.stack([reps[torch.as_tensor(np.flatnonzero(items.verb.to_numpy() == v), device=dev), :N_SITES_B]
                         .double().mean(0).cpu().numpy() for v in averbs.verb])
    np.savez_compressed(out / "class_states.npz", pas_mean=pas_mean.astype(np.float32),
                        pas_verb=pverbs.verb.to_numpy(), pas_cls=pverbs.cls.to_numpy(),
                        pas_pair=pverbs.verb_pair.to_numpy(), act_mean=act_mean.astype(np.float32),
                        act_verb=averbs.verb.to_numpy(), act_cls=averbs.cls.to_numpy(),
                        act_pair=averbs.pair_id.to_numpy())
    log(f"class states: {len(pverbs)} passive verbs ({len(P)} prompts), {len(averbs)} active verbs")

    def u_pas_fit(exclude_parts):
        """Unit T - I mean difference over LEACE pairs with neither participle excluded; [24, d]."""
        keep_pairs = [vp for vp, g in pverbs.groupby("verb_pair") if not (set(g.verb) & exclude_parts)]
        bad = sorted(set(pverbs.verb_pair) - set(keep_pairs))
        k = pverbs.verb_pair.isin(keep_pairs).to_numpy()
        T = pas_mean[k & (pverbs.cls.to_numpy() == 1)].mean(0)
        I = pas_mean[k & (pverbs.cls.to_numpy() == 0)].mean(0)
        md = T - I
        assert not (set(pverbs.verb[k]) & exclude_parts)
        return md / np.linalg.norm(md, axis=-1, keepdims=True), len(keep_pairs), bad

    # ---------------------------------------------------------------- passive plan and natural states
    it = pd.read_csv(args.items)
    prim = it[it.bad_class == "plain"]
    if args.max_items:
        prim = prim.groupby("pair_id").head(max(1, args.max_items // prim.pair_id.nunique()))
    prim_ids = set(prim.item_id)
    plan = pd.read_csv(args.plan)
    plan = plan[plan.item_id.isin(prim_ids)]
    pat_plan = plan[(plan.side == "bad") & plan.cond.isin(["T", "I"])].reset_index(drop=True)
    side_rows = plan.drop_duplicates(["item_id", "side"])[["item_id", "side", "base"]]
    Pp = pd.read_csv(args.prompts)
    assert (Pp.pid.to_numpy() == np.arange(len(Pp))).all()
    prompts_p = Pp.prompt.tolist()
    need = np.unique(np.concatenate([pat_plan.base, pat_plan.donor, side_rows.base]))
    meta.update({"passive_patch_rows": len(pat_plan), "passive_items": len(prim_ids)})
    P_states = torch.zeros(len(Pp), len(sites), d, device=dev)
    pnat = {k: np.full(len(Pp), np.nan, np.float32) for k in ("M",) + READ}
    with torch.no_grad():
        for i in range(0, len(need), args.batch):
            ids = need[i:i + args.batch]
            enc, anchors = runner.encode([prompts_p[j] for j in ids])
            o = runner.model(**enc, output_hidden_states=True, use_cache=False)
            m, parts = runner.readout(o.logits, anchors, full=True)
            rows = torch.arange(len(anchors), device=dev)
            ix = torch.as_tensor(ids, device=dev)
            P_states[ix] = torch.stack([o.hidden_states[s][rows, anchors].float() for s in sites], 1)
            pnat["M"][ids] = m.cpu().numpy()
            for k in READ:
                pnat[k][ids] = parts[k].cpu().numpy()
    pd.DataFrame({"pid": need, **{k: v[need] for k, v in pnat.items()}}) \
        .to_parquet(out / "passive_natural.parquet", index=False)
    log(f"passive: {len(pat_plan)} patch rows, natural pass done")
    good_pids = side_rows[side_rows.side == "good"].base.to_numpy()
    bad_pids = side_rows[side_rows.side == "bad"].base.to_numpy()

    # C: groups of primary pairs, fit exclusions
    ppairs = sorted(prim.pair_id.unique())
    order = [ppairs[i] for i in np.random.default_rng(2).permutation(len(ppairs))]
    group = {p: j % 5 for j, p in enumerate(order)}
    pparts = it.drop_duplicates("pair_id").set_index("pair_id")[["good_part", "bad_part"]]
    u_pas_g, c_fit = {}, {}
    for g in range(5):
        ex = das_parts | {x for p in ppairs if group[p] == g for x in pparts.loc[p]}
        u_pas_g[g], n_keep, bad = u_pas_fit(ex)
        c_fit[g] = {"pairs_kept": n_keep, "pairs_excluded": len(bad)}
        assert smoke or n_keep >= 40, c_fit[g]
    u_pas_act, n_keep_a, _ = u_pas_fit(das_parts)
    c_fit["active"] = {"pairs_kept": n_keep_a}
    assert smoke or n_keep_a >= 40, c_fit
    checks["c_fit"] = c_fit
    item_group = it.set_index("item_id").pair_id.map(group)

    # ---------------------------------------------------------------- A + C per site
    act_summ, act_swaps, push_rows = [], [], []
    for si, site in enumerate(sites):
        ts = time.time()
        B, meta["bases_sha256"][str(site)] = load_bases(root, site, ref_folds, torch, dev)
        Xn = reps[:, site].double().cpu().numpy()
        keys = sorted(B)
        sc = {"decomp_max_err": 0.0, "w_dot_u_max": 0.0, "write_eq_read_max_abs_dM": 0.0}
        stored_u = None
        if args.dirchk_dir and (Path(args.dirchk_dir) / f"directions_site{site}.npz").exists():
            z = np.load(Path(args.dirchk_dir) / f"directions_site{site}.npz")
            stored_u = {tuple(int(t) for t in k): z["u_md"][i] for i, k in enumerate(z["keys"])}
            sc["u_vs_dirchk_max_abs"] = 0.0
        W = {}
        dirs = {"keys": np.array(keys), "v": [], "u": [], "a": [], "w": []}
        S = P_states[:, si]
        Sn = S.double().cpu().numpy()
        for (k, f) in keys:
            v = B[(k, f)][:, 0].double().cpu().numpy()
            tr, te = r2.fold_indices(items, assigns[k], f)
            lab = labels[tr]
            u = unit(Xn[tr][lab == 1].mean(0) - Xn[tr][lab == 0].mean(0))
            if stored_u is not None:
                sc["u_vs_dirchk_max_abs"] = max(sc["u_vs_dirchk_max_abs"], float(np.abs(u - stored_u[(k, f)]).max()))
            a = float(v @ u)
            w = v - a * u
            sc["decomp_max_err"] = max(sc["decomp_max_err"], float(np.abs(a * u + w - v).max()))
            sc["w_dot_u_max"] = max(sc["w_dot_u_max"], abs(float(w @ u)))
            b = float(np.linalg.norm(w))
            wh = w / b
            T = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)[:, None]
            W[(k, f)] = {"full": B[(k, f)], "on": T(a * u), "off": T(w)}
            for n, x in (("v", v), ("u", u), ("a", a), ("w", w)):
                dirs[n].append(x)
            # push sizes and class separation
            sd = {"act_u": (Xn @ u).std(ddof=1), "act_w": (Xn @ wh).std(ddof=1)}
            pp_ids = np.concatenate([good_pids, bad_pids])
            sd["pas_u"], sd["pas_w"] = (Sn[pp_ids] @ u).std(ddof=1), (Sn[pp_ids] @ wh).std(ddof=1)
            sw = [(bb, jj) for bb, jj in r2.eval_swaps(items, te) if labels[bb] == 0 and labels[jj] == 1]
            dl_a = np.array([v @ (Xn[jj] - Xn[bb]) for bb, jj in sw])
            g_ = pat_plan[(pat_plan.split == k) & (pat_plan.fold == f)]
            dl_p = (Sn[g_.donor.to_numpy()] - Sn[g_.base.to_numpy()]) @ v
            row = {"site": site, "split": k, "fold": f, "a": a, "b": b, **{f"sd_{x}": y for x, y in sd.items()},
                   "sep_act_u": smd(Xn[labels == 1] @ u, Xn[labels == 0] @ u),
                   "sep_act_w": smd(Xn[labels == 1] @ wh, Xn[labels == 0] @ wh),
                   "sep_pas_u": smd(Sn[good_pids] @ u, Sn[bad_pids] @ u),
                   "sep_pas_w": smd(Sn[good_pids] @ wh, Sn[bad_pids] @ wh), "n_act_swaps": len(dl_a),
                   "n_pas_rows": len(dl_p)}
            for fr, dl in (("act", dl_a), ("pas", dl_p)):
                if len(dl):
                    row[f"{fr}_on_med"] = float(np.median(np.abs(dl * a)) / sd[f"{fr}_u"])
                    row[f"{fr}_off_med"] = float(np.median(np.abs(dl * b)) / sd[f"{fr}_w"])
                    row[f"{fr}_off_gt2sd"] = float(np.mean(np.abs(dl * b) > 2 * sd[f"{fr}_w"]))
            push_rows.append(row)
        np.savez_compressed(out / f"directions_site{site}.npz", **{k: np.array(v) for k, v in dirs.items()})

        # check 2: write = read reproduces Runner.patched bitwise
        g0 = pat_plan.iloc[:min(256, len(pat_plan))]
        key0 = (int(g0.split.iloc[0]), int(g0.fold.iloc[0]))
        g0 = g0[(g0.split == key0[0]) & (g0.fold == key0[1])]
        src0 = S[torch.as_tensor(g0.donor.to_numpy(), device=dev)]
        pr0 = [prompts_p[j] for j in g0.base]
        with torch.no_grad():
            m_ref, _ = runner.patched(pr0, src0, site, B[key0], full=True)
        m_new, _ = patched_rw(runner, pr0, src0, site, B[key0], B[key0])
        sc["write_eq_read_max_abs_dM"] = float((m_ref - m_new).abs().max())

        # ---- A. held-out actives: full / on / off (evaluate with the written direction replaced), C: u_pas
        for (k, f) in keys:
            _, te = r2.fold_indices(items, assigns[k], f)
            for cond in ("full", "on", "off"):
                wr = W[(k, f)][cond]
                runner.patched = (lambda prompts, src, s_, basis, full=False, match_norm=None, wr=wr:
                                  patched_rw(runner, prompts, src, s_, basis, wr, full))
                try:
                    df = r2.evaluate(runner, items, reps, nat, te, site, B[(k, f)], detail=True)
                finally:
                    del runner.patched
                df["rank"], df["split"], df["fold"], df["condition"], df["site"] = 1, k, f, cond, site
                act_swaps.append(df)
                act_summ.append({"site": site, "condition": cond, "split": k, "fold": f,
                                 **r2.summarize(df, items)})
            up = torch.as_tensor(u_pas_act[site], dtype=torch.float32, device=dev)[:, None]
            df = r2.evaluate(runner, items, reps, nat, te, site, up, detail=True)
            df["rank"], df["split"], df["fold"], df["condition"], df["site"] = 1, k, f, "u_pas", site
            act_swaps.append(df)
            act_summ.append({"site": site, "condition": "u_pas", "split": k, "fold": f, **r2.summarize(df, items)})
        log(f"site {site}: actives done")

        # ---- A. passives: full / on / off; C: u_pas (group-cross-fitted)
        res = {c: {kk: np.empty(len(pat_plan), np.float32) for kk in ("M",) + READ} for c in ("full", "on", "off")}
        delta = np.empty(len(pat_plan), np.float32)
        for key, g in pat_plan.groupby(["split", "fold"]):
            ix = g.index.to_numpy()
            base, donor = g.base.to_numpy(), g.donor.to_numpy()
            v_t = B[key]
            for i in range(0, len(ix), args.batch):
                sl = slice(i, i + args.batch)
                src = S[torch.as_tensor(donor[sl], device=dev)]
                pr = [prompts_p[j] for j in base[sl]]
                delta[ix[sl]] = ((src - S[torch.as_tensor(base[sl], device=dev)]) @ v_t)[:, 0].cpu().numpy()
                for cond in ("full", "on", "off"):
                    m, parts = patched_rw(runner, pr, src, site, v_t, W[key][cond])
                    res[cond]["M"][ix[sl]] = m.cpu().numpy()
                    for kk in READ:
                        res[cond][kk][ix[sl]] = parts[kk].cpu().numpy()
        pd.concat([pd.DataFrame({"row": pat_plan.row, "condition": c, "delta": delta, **res[c]})
                   for c in ("full", "on", "off")], ignore_index=True) \
            .to_parquet(out / f"split_patches_site{site}.parquet", index=False)
        log(f"site {site}: passive split edit done ({len(pat_plan)} rows x 3)")
        res_c = {kk: np.empty(len(pat_plan), np.float32) for kk in ("M",) + READ}
        grp = pat_plan.item_id.map(item_group).to_numpy()
        assert not np.isnan(grp.astype(float)).any(), "plan item without a C group"
        for gk, g in pat_plan.assign(grp=grp).groupby("grp"):
            ix = g.index.to_numpy()
            up = torch.as_tensor(u_pas_g[int(gk)][site], dtype=torch.float32, device=dev)[:, None]
            base, donor = g.base.to_numpy(), g.donor.to_numpy()
            for i in range(0, len(ix), args.batch):
                sl = slice(i, i + args.batch)
                with torch.no_grad():
                    m, parts = runner.patched([prompts_p[j] for j in base[sl]],
                                              S[torch.as_tensor(donor[sl], device=dev)], site, up, full=True)
                res_c["M"][ix[sl]] = m.cpu().numpy()
                for kk in READ:
                    res_c[kk][ix[sl]] = parts[kk].cpu().numpy()
        pd.DataFrame({"row": pat_plan.row, "group": grp, **res_c}) \
            .to_parquet(out / f"axis_patches_site{site}.parquet", index=False)
        checks["sites"][str(site)] = sc
        log(f"site {site}: C done; checks {sc} ({time.time() - ts:.0f}s)")
        pd.DataFrame(act_summ).to_csv(out / "active_summary.csv", index=False)
        pd.DataFrame(push_rows).to_csv(out / "push.csv", index=False)

    det = pd.concat(act_swaps, ignore_index=True)
    for col in ("verb", "pair_id", "subject", "source", "tok"):
        det[f"base_{col}"] = items[col].values[det.base]
        det[f"src_{col}"] = items[col].values[det.src]
    for k in ("O", "I", "det", "pron", "refl"):
        det[f"lp_{k}_base"] = nat[k][det.base]
        det[f"lp_{k}_src"] = nat[k][det.src]
    det.to_csv(out / "active_swaps.csv.gz", index=False)
    (out / "checks.json").write_text(json.dumps(checks, indent=2, default=str) + "\n")
    meta["minutes"] = (time.time() - t0) / 60
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    log(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", default="4,6,8,12,17")
    ap.add_argument("--root", default="../results/das_round2")
    ap.add_argument("--pairs", default="../data/das_round2/train_pairs.csv")
    ap.add_argument("--items", default="../data/das_round2/passive_test/items.csv")
    ap.add_argument("--prompts", default="../data/das_round2/passive_test/prompts.csv")
    ap.add_argument("--plan", default="../data/das_round2/passive_test/plan.csv.gz")
    ap.add_argument("--scores", default="../results/passive_band_cross/pythia14b_scores.csv")
    ap.add_argument("--v2-pairs", default="../data/matched_passives_v2/pairs.jsonl")
    ap.add_argument("--leace-insample", default="../results/pythia/leace_insample/erasers.npz")
    ap.add_argument("--dirchk-dir", default="", help="direction-check outputs (stored u_md, for check 3)")
    ap.add_argument("--out-dir", default="../results/pythia/split_edit")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--max-items", type=int, default=0, help="smoke test: about this many primary passive items")
    ap.add_argument("--max-pairs", type=int, default=0, help="smoke test: LEACE pairs per verb band")
    ap.add_argument("--max-contexts", type=int, default=0, help="smoke test: LEACE contexts kept")
    ap.add_argument("--bos", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--fp32-unembed", action=argparse.BooleanOptionalAction, default=None)
    run(ap.parse_args())
