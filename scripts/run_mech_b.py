#!/usr/bin/env python3
"""C4 phase 1b runs (reports/v2_controls_mech_b_rules.md, part B). Pythia-1.4B, fp32. Readout log P(" by") at the
participle's last token.

--mode components (B1): phase 1's mediation keys (half B; rebuilt as run_conjunction_mediation.py and asserted equal
  to its stored mediation_keys.parquet). Per key: TT, II (full v2 edits) and, for each component set C, T<-I:C (T edit,
  the outputs of C at the last token set to their II values) and I<-T:C. Components: attention output (attention.dense)
  and MLP output of layers 8-23; groups attn_all, mlp_all, all, mlp11_14.
  Checks: mlp11_14 equals phase 1's ALL-neuron clamp; TT / II equal phase 1's; under T<-I:all the final-norm input
  minus II's equals the site-8 state difference.
--mode natural_halfA (B2 controls): the half-A pairs' sampled items, good and bad prompts, unclamped; per neuron of
  MLPs 11 / 14 the pair-averaged good - bad post-activation at the last token. Output natural_halfA.npz.
--mode natural (B2, B3b, B3c): the half-B pairs' sampled items. Per item: good, bad;
  g<-b:S, b<-g:S (S's post-activations set to the other prompt's) for every set of the --sets files;
  g<-g2:S (good prompt, S set to a same-class donor's: another eval plain pair's good prompt in the same context with a
  different good lemma, same token count where available, else nearest, ties at random, seed 17);
  gswap / bswap: the v2 coordinate at site 8 swapped with the paired prompt's (split-0 basis of the item's plan fold);
  gswap|S / bswap|S: the swap with S held at the unswapped run's values (--swap-sets).
  Also the core neurons' post-activations in the unclamped good / bad runs.
Checks: empty-set clamps equal their references; entries outside a clamp unchanged at the hook (0); entries inside
equal their targets; the unclamped readouts against the stored natural passive run.
Outputs (--out-dir): {mode}_keys.parquet, {mode}.npz (by [rows, conds], conds), {mode}_meta.json; natural also
natural_core_acts.parquet and natural_donors.parquet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_das_round2 import normalize_output, replace_output
from run_passive_test import PassiveRunner
from run_path_patch_frames import build_frames
from run_site8_mechanism import SITE, load_bases

NL = 24
D = 8192
NEU = (11, 14)
KCOLS = ["item_id", "donor_pair", "pair", "fold", "base", "donor_T", "donor_I", "plan_row_T", "plan_row_I", "len", "rid"]


def sample_items(items, contexts, seed):
    """Item ids as run_conjunction.py (same rng draws): per plain pair, `contexts` items without replacement."""
    prim = items[items.bad_class == "plain"]
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in prim.groupby("pair_id"):
        ids = sorted(g.item_id)
        keep += list(rng.choice(ids, min(contexts, len(ids)), replace=False))
    return prim, keep


def halves(args):
    sel = np.load(args.selection, allow_pickle=True)
    p, h = np.asarray(sel["pairs"]), np.asarray(sel["halves"])
    return set(p[h == 0]), set(p[h == 1])


def edit_keys(args, prompts, items, tok):
    """Phase 1's mediation keys, rebuilt as run_conjunction_mediation.py and asserted equal to the stored ones."""
    plan = pd.read_csv(args.plan, usecols=["row", "item_id", "split", "fold", "side", "cond", "base", "donor", "donor_pair"])
    _, half_b = halves(args)
    prim, keep = sample_items(items, args.contexts, args.seed)
    plan = plan.rename(columns={"row": "plan_row"})
    rows = plan[(plan.split == 0) & (plan.side == "bad") & plan.cond.isin(["T", "I"]) & plan.item_id.isin(set(keep))].copy()
    it = prim.set_index("item_id")
    rows["pair"] = rows.item_id.map(it.pair_id)
    rows = rows[rows.pair.isin(half_b)].copy()
    T_ = rows[rows.cond == "T"].set_index(["item_id", "donor_pair"])
    I_ = rows[rows.cond == "I"].set_index(["item_id", "donor_pair"])
    assert T_.index.is_unique and I_.index.is_unique and T_.index.sort_values().equals(I_.index.sort_values())
    keys = T_.join(I_[["donor", "plan_row", "fold", "base"]], rsuffix="_I").reset_index()
    assert (keys.fold == keys.fold_I).all() and (keys.base == keys.base_I).all()
    keys = keys.rename(columns={"donor": "donor_T", "plan_row": "plan_row_T"})
    keys["was_plain"] = [prompts[b] for b in keys.base]
    ids_by_frame, fm = build_frames(keys, it, tok, [("was", (" was",))])
    keys = keys.join(fm)
    keys["rid"] = np.arange(len(keys))
    st = pd.read_parquet(Path(args.phase1_dir) / "mediation_keys.parquet")
    assert keys[KCOLS].reset_index(drop=True).equals(st[KCOLS].reset_index(drop=True)), "keys differ from phase 1's"
    return keys, ids_by_frame["was"]


def run(args):
    import torch

    args.dtype = "float32"
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runner = PassiveRunner(args)
    tok, dev = runner.tok, runner.device
    L = runner.model.gpt_neox.layers
    assert len(L) == NL
    by_id = runner.ids["by"]
    prompts = pd.read_csv(args.prompts).prompt.tolist()
    items = pd.read_csv(args.items, low_memory=False)
    checks = {"hook_out_max": 0.0, "hook_in_max": 0.0}

    def forward(enc_ids, mask, anc, patch=None, comp_set=None, comp_tgt=None, neu_set=None, neu_tgt=None, fin=False):
        """One forward. patch: site-8 edit fn on [B, d]. comp_set: (kind, l) whose outputs at the anchor are set to
        comp_tgt[(kind, l)]. neu_set: {l: mask [B, D]} with targets neu_tgt[l]. Returns by [B] and a capture dict:
        "s8" site-8 state at the anchor (before any patch), ("attn"|"mlp", l) component outputs, l neuron
        post-activations (after any setting), "fin" final-norm input at the anchor (if fin)."""
        B = enc_ids.shape[0]
        r = torch.arange(B, device=dev)
        cap, hs = {}, []
        comp_set = set(comp_set or [])

        def at8(m, i, o):
            h = normalize_output(o)
            cap["s8"] = h[r, anc].float()
            if patch is None:
                return None
            h = h.clone()
            h[r, anc] = patch(h[r, anc].float()).to(h.dtype)
            return replace_output(o, h)

        hs.append(L[SITE - 1].register_forward_hook(at8))
        if comp_set or comp_tgt is not None or args.mode == "components":
            for l in range(SITE, NL):
                for kind, mod in (("attn", L[l].attention.dense), ("mlp", L[l].mlp)):
                    def post(m, i, o, key=(kind, l)):
                        h = normalize_output(o)
                        if key in comp_set:
                            h = h.clone()
                            h[r, anc] = comp_tgt[key].to(h.dtype)
                            b = h[r, anc].float()
                            checks["hook_in_max"] = max(checks["hook_in_max"], float((b - comp_tgt[key]).abs().max()))
                            cap[key] = b
                            return replace_output(o, h)
                        cap[key] = h[r, anc].float()
                        return None
                    hs.append(mod.register_forward_hook(post))
        for l in NEU:
            def pre(m, i, l=l):
                x = i[0]
                a = x[r, anc].float()
                if neu_set is not None:
                    msk, tgt = neu_set[l], neu_tgt[l]
                    x = x.clone()
                    x[r, anc] = torch.where(msk, tgt, a).to(x.dtype)
                    b = x[r, anc].float()
                    if (~msk).any():
                        checks["hook_out_max"] = max(checks["hook_out_max"], float((b - a)[~msk].abs().max()))
                    if msk.any():
                        checks["hook_in_max"] = max(checks["hook_in_max"], float((b - tgt)[msk].abs().max()))
                    cap[l] = b
                    return (x,) + tuple(i[1:])
                cap[l] = a
                return None
            hs.append(L[l].mlp.dense_4h_to_h.register_forward_pre_hook(pre))
        if fin:
            def fpre(m, i):
                cap["fin"] = i[0][r, anc].float()
            hs.append(runner.model.gpt_neox.final_layer_norm.register_forward_pre_hook(fpre))
        try:
            with torch.no_grad():
                logits = runner.model(input_ids=enc_ids, attention_mask=mask, use_cache=False).logits
        finally:
            for h in hs:
                h.remove()
        lp = torch.log_softmax(logits[r, anc].float(), -1)
        return lp[:, by_id].logsumexp(-1), cap

    def neuron_masks(names_files):
        names, masks = ["empty"], [np.zeros(2 * D, bool)]
        for f in names_files:
            S = np.load(f, allow_pickle=True)
            for n, m in zip(S["names"], S["masks"]):
                n = str(n)
                if n in names:
                    assert (masks[names.index(n)] == m.astype(bool)).all(), f"set {n} differs between files"
                    continue
                names.append(n)
                masks.append(m.astype(bool))
        return names, torch.as_tensor(np.stack(masks), device=dev)

    def lay(mk, k, B):
        return {l: mk[k, j * D:(j + 1) * D][None].expand(B, D) for j, l in enumerate(NEU)}

    if args.mode == "components":
        keys, ids_was = edit_keys(args, prompts, items, tok)
        bases = load_bases(Path(args.site_dir) / "bases.pt", dev)
        groups = {"empty": []}
        for l in range(SITE, NL):
            groups[f"attn{l}"] = [("attn", l)]
            groups[f"mlp{l}"] = [("mlp", l)]
        groups["attn_all"] = [("attn", l) for l in range(SITE, NL)]
        groups["mlp_all"] = [("mlp", l) for l in range(SITE, NL)]
        groups["all"] = groups["attn_all"] + groups["mlp_all"]
        groups["mlp11_14"] = [("mlp", 11), ("mlp", 14)]
        conds = ["TT", "II"] + [f"{c}:{g}" for c in ("T<-I", "I<-T") for g in groups]
        res = np.full((len(keys), len(conds)), np.nan, np.float32)
        ci = {c: k for k, c in enumerate(conds)}
        donors = np.unique(np.r_[keys.donor_T.to_numpy(), keys.donor_I.to_numpy()])
        dstate = {}
        with torch.no_grad():
            for i in range(0, len(donors), args.batch):
                ids = donors[i:i + args.batch]
                enc, anchors = runner.encode([prompts[j] for j in ids])
                o = runner.model(**enc, output_hidden_states=True, use_cache=False)
                rr = torch.arange(len(anchors), device=dev)
                for j, v in zip(ids, o.hidden_states[SITE][rr, anchors].float()):
                    dstate[int(j)] = v
        checks["resid_all_rel_max"] = 0.0
        for ln, g in keys.groupby("len"):
            for i in range(0, len(g), args.batch):
                gg = g.iloc[i:i + args.batch]
                x = torch.as_tensor([ids_was[j] for j in gg.rid], device=dev)
                msk = torch.ones_like(x)
                anc = torch.full((len(gg),), x.shape[1] - 1, device=dev)
                dd = torch.stack([bases[(0, int(f))] for f in gg.fold])

                def edit(col):
                    src = torch.stack([dstate[int(j)] for j in gg[col]])
                    return lambda h: h + ((src - h) * dd).sum(-1, keepdim=True) * dd

                pT, pI = edit("donor_T"), edit("donor_I")
                yT, cT = forward(x, msk, anc, pT, fin=True)
                yI, cI = forward(x, msk, anc, pI, fin=True)
                s8T, s8I = pT(cT["s8"]), pI(cI["s8"])  # site-8 states after each edit
                rix = gg.rid.to_numpy()
                res[rix, ci["TT"]], res[rix, ci["II"]] = yT.cpu().numpy(), yI.cpu().numpy()
                for gname, comps in groups.items():
                    for c, p_, tg in (("T<-I", pT, cI), ("I<-T", pI, cT)):
                        y, cc = forward(x, msk, anc, p_, comps, tg, fin=(gname == "all" and c == "T<-I"))
                        res[rix, ci[f"{c}:{gname}"]] = y.cpu().numpy()
                        if gname == "all" and c == "T<-I":
                            ref = s8T - s8I
                            dev_ = (cc["fin"] - cI["fin"] - ref).abs().max() / ref.abs().max()
                            checks["resid_all_rel_max"] = max(checks["resid_all_rel_max"], float(dev_))
            print(f"len {ln}: {len(g)} keys, {time.time() - t0:.0f}s", flush=True)
        z = np.load(Path(args.phase1_dir) / "mediation.npz", allow_pickle=True)
        zc = [str(c) for c in z["conds"]]
        for c, ref in (("T<-I", "TT"), ("I<-T", "II")):
            checks[f"{c}:empty_vs_{ref}_max"] = float(np.nanmax(np.abs(res[:, ci[f"{c}:empty"]] - res[:, ci[ref]])))
            checks[f"{c}:mlp11_14_vs_phase1_ALL_max"] = float(np.nanmax(np.abs(
                res[:, ci[f"{c}:mlp11_14"]] - z["by"][:, zc.index(f"{c}:ALL")].astype(float))))
        for c in ("TT", "II"):
            checks[f"{c}_vs_phase1_max"] = float(np.nanmax(np.abs(res[:, ci[c]] - z["by"][:, zc.index(c)].astype(float))))
        keys[KCOLS].to_parquet(out / "components_keys.parquet")

    elif args.mode == "natural_halfA":
        half_a, _ = halves(args)
        prim, keep = sample_items(items, args.contexts, args.seed)
        keys = prim[prim.item_id.isin(set(keep)) & prim.pair_id.isin(half_a)].reset_index(drop=True).rename(columns={"pair_id": "pair"})
        keys["rid"] = np.arange(len(keys))
        diff = np.zeros((len(keys), 2 * D), np.float32)
        res = np.full((len(keys), 2), np.nan, np.float32)
        conds = ["good", "bad"]
        for i in range(0, len(keys), args.batch):
            gg = keys.iloc[i:i + args.batch]
            eg, ag = runner.encode(gg.good_prompt.tolist())
            eb, ab = runner.encode(gg.bad_prompt.tolist())
            yg, cg = forward(eg.input_ids, eg.attention_mask, ag)
            yb, cb = forward(eb.input_ids, eb.attention_mask, ab)
            rix = gg.rid.to_numpy()
            res[rix, 0], res[rix, 1] = yg.cpu().numpy(), yb.cpu().numpy()
            diff[rix] = torch.cat([cg[l] - cb[l] for l in NEU], -1).cpu().numpy()
        pairs = np.array(sorted(keys.pair.unique()))
        pc = keys.pair.map({p: j for j, p in enumerate(pairs)}).to_numpy()
        s = np.zeros((len(pairs), 2 * D))
        np.add.at(s, pc, diff)
        s /= np.bincount(pc)[:, None]
        np.savez_compressed(out / "natural_halfA.npz", pair_diff=s.astype(np.float32), pairs=pairs,
                            mean_diff=s.mean(0).astype(np.float32))
        keys = keys[["item_id", "pair", "context_id", "band", "rid"]]
        keys.to_parquet(out / "natural_halfA_keys.parquet")

    else:  # natural
        _, half_b = halves(args)
        prim, keep = sample_items(items, args.contexts, args.seed)
        keys = prim[prim.item_id.isin(set(keep)) & prim.pair_id.isin(half_b)][
            ["item_id", "pair_id", "context_id", "band", "good_lemma", "good_prompt", "bad_prompt"]].reset_index(drop=True)
        keys = keys.rename(columns={"pair_id": "pair"})
        keys["rid"] = np.arange(len(keys))
        plan = pd.read_csv(args.plan, usecols=["item_id", "split", "fold"])
        pf = plan[(plan.split == 0) & plan.item_id.isin(set(keys.item_id))].drop_duplicates(["item_id", "fold"])
        assert pf.item_id.is_unique and set(pf.item_id) == set(keys.item_id), "one split-0 fold per item"
        keys["fold"] = keys.item_id.map(pf.set_index("item_id").fold).astype(int)
        # same-class donors: another eval plain pair's good prompt, same context, different good lemma
        ntok = lambda ps: [len(tok(p, add_special_tokens=False).input_ids) for p in ps]
        keys["ntok"] = ntok(keys.good_prompt)
        cand = prim[["item_id", "pair_id", "context_id", "good_lemma", "good_prompt"]].copy()
        cand["ntok"] = ntok(cand.good_prompt)
        by_ctx = {c: g for c, g in cand.groupby("context_id")}
        rng = np.random.default_rng(args.seed)
        don = []
        for r_ in keys.itertuples():
            g = by_ctx[r_.context_id]
            g = g[(g.pair_id != r_.pair) & (g.good_lemma != r_.good_lemma)]
            assert len(g), f"no same-class donor for {r_.item_id}"
            dist = (g.ntok - r_.ntok).abs()
            g = g[dist == dist.min()]
            pick = g.iloc[int(rng.integers(len(g)))]
            don.append({"rid": r_.rid, "donor_item": pick.item_id, "donor_pair": pick.pair_id,
                        "donor_prompt": pick.good_prompt, "ntok_diff": int(pick.ntok - r_.ntok)})
        don = pd.DataFrame(don)
        keys["donor_prompt"] = don.donor_prompt.to_numpy()
        names, mk = neuron_masks(args.sets.split(","))
        swap_sets = ["empty"] + [n for n in names if n in set(args.swap_sets.split(",")) or
                                 any(n.startswith(p + "_") and n[len(p) + 1:].isdigit() for p in args.swap_families.split(","))]
        bases = load_bases(Path(args.site_dir) / "bases.pt", dev)
        core = pd.read_csv(args.switch).head(5)
        conds = (["good", "bad", "gswap", "bswap"] + [f"{c}:{n}" for c in ("g<-b", "b<-g", "g<-g2") for n in names]
                 + [f"{c}|{n}" for c in ("gswap", "bswap") for n in swap_sets])
        res = np.full((len(keys), len(conds)), np.nan, np.float32)
        ci = {c: k for k, c in enumerate(conds)}
        acts = []
        for i in range(0, len(keys), args.batch):
            gg = keys.iloc[i:i + args.batch]
            B = len(gg)
            eg, ag = runner.encode(gg.good_prompt.tolist())
            eb, ab = runner.encode(gg.bad_prompt.tolist())
            ed, ad = runner.encode(gg.donor_prompt.tolist())
            yg, cg = forward(eg.input_ids, eg.attention_mask, ag)
            yb, cb = forward(eb.input_ids, eb.attention_mask, ab)
            _, cd = forward(ed.input_ids, ed.attention_mask, ad)
            dd = torch.stack([bases[(0, int(f))] for f in gg.fold])
            sw = lambda src: (lambda h: h + ((src - h) * dd).sum(-1, keepdim=True) * dd)
            pg, pb = sw(cb["s8"]), sw(cg["s8"])  # good gets bad's coordinate, and the converse
            ygs, _ = forward(eg.input_ids, eg.attention_mask, ag, pg)
            ybs, _ = forward(eb.input_ids, eb.attention_mask, ab, pb)
            rix = gg.rid.to_numpy()
            for c, y in (("good", yg), ("bad", yb), ("gswap", ygs), ("bswap", ybs)):
                res[rix, ci[c]] = y.cpu().numpy()
            a = {"rid": rix}
            for r_ in core.itertuples():
                a[f"good_{r_.neuron}"] = cg[int(r_.layer)][:, int(r_.index)].cpu().numpy()
                a[f"bad_{r_.neuron}"] = cb[int(r_.layer)][:, int(r_.index)].cpu().numpy()
            acts.append(pd.DataFrame(a))
            for k, n in enumerate(names):
                m = lay(mk, k, B)
                for c, e, an, tg in (("g<-b", eg, ag, cb), ("b<-g", eb, ab, cg), ("g<-g2", eg, ag, cd)):
                    y, _ = forward(e.input_ids, e.attention_mask, an, neu_set=m, neu_tgt=tg)
                    res[rix, ci[f"{c}:{n}"]] = y.cpu().numpy()
                if n in swap_sets:
                    for c, e, an, p_, tg in (("gswap", eg, ag, pg, cg), ("bswap", eb, ab, pb, cb)):
                        y, _ = forward(e.input_ids, e.attention_mask, an, p_, neu_set=m, neu_tgt=tg)
                        res[rix, ci[f"{c}|{n}"]] = y.cpu().numpy()
            print(f"{i + B} / {len(keys)} items, {time.time() - t0:.0f}s", flush=True)
        for c, ref in (("g<-b", "good"), ("b<-g", "bad"), ("g<-g2", "good")):
            checks[f"{c}:empty_vs_{ref}_max"] = float(np.nanmax(np.abs(res[:, ci[f"{c}:empty"]] - res[:, ci[ref]])))
        for c in ("gswap", "bswap"):
            checks[f"{c}|empty_vs_{c}_max"] = float(np.nanmax(np.abs(res[:, ci[f"{c}|empty"]] - res[:, ci[c]])))
        nat = pd.read_parquet(args.stored_natural)
        P = pd.read_csv(args.prompts)
        pid = dict(zip(P.prompt, P.pid))
        sn = nat.set_index("pid").by
        tg_, tb_ = sn.reindex(keys.good_prompt.map(pid)).to_numpy(), sn.reindex(keys.bad_prompt.map(pid)).to_numpy()
        dg, db = np.abs(res[:, ci["good"]] - tg_), np.abs(res[:, ci["bad"]] - tb_)
        checks["stored_natural_missing"] = int(np.isnan(dg).sum() + np.isnan(db).sum())
        checks["stored_natural_median_abs"] = float(np.nanmedian(np.r_[dg, db]))
        checks["stored_natural_max_abs"] = float(np.nanmax(np.r_[dg, db]))
        gp = pd.DataFrame({"pair": keys.pair, "mine": res[:, ci["good"]] - res[:, ci["bad"]], "stored": tg_ - tb_}).groupby("pair").mean()
        checks["stored_natural_pair_gap_mean_diff"] = float((gp.mine - gp.stored).mean())
        checks["core"] = core.neuron.tolist()
        checks["same_class_donor_ntok_diff_abs_mean"] = float(don.ntok_diff.abs().mean())
        checks["same_class_donor_exact_len_share"] = float((don.ntok_diff == 0).mean())
        pd.concat(acts).sort_values("rid").to_parquet(out / "natural_core_acts.parquet")
        don.to_parquet(out / "natural_donors.parquet")
        keys.drop(columns=["good_prompt", "bad_prompt", "donor_prompt"]).to_parquet(out / "natural_keys.parquet")
    checks["nan_cells"] = int(np.isnan(res).sum())
    np.savez_compressed(out / f"{args.mode}.npz", by=res, conds=np.array(conds))
    meta = {"args": vars(args), "rows": len(keys), "pairs": int(keys.pair.nunique()), "checks": checks,
            "seconds": round(time.time() - t0, 1)}
    if args.site_dir:
        meta["bases_sha256"] = hashlib.sha256((Path(args.site_dir) / "bases.pt").read_bytes()).hexdigest()
    (out / f"{args.mode}_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps(meta, default=str), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("components", "natural_halfA", "natural"))
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--site-dir", default=None)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--phase1-dir", default="results/pythia/das_v2/mech/med_v2")
    ap.add_argument("--sets", default=None, help="natural: comma-separated set npz files")
    ap.add_argument("--swap-sets", default="P,ALL", help="natural: sets held fixed during the coordinate swap")
    ap.add_argument("--swap-families", default="mag", help="natural: control families also held fixed (name_k)")
    ap.add_argument("--switch", default="results/pythia/das_v2/mech/conj_v2/conjunction_switch_neurons.csv")
    ap.add_argument("--stored-natural", default="results/pythia/das_v2/expanded_eval/passive_rule/natural.parquet")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--contexts", type=int, default=32)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=128)
    run(ap.parse_args())
