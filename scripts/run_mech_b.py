#!/usr/bin/env python3
"""C4 phase 1b runs (reports/v2_controls_mech_b_rules.md, part B). Pythia-1.4B, fp32. Readout log P(" by") at the
participle's last token.

--mode components (B1): phase 1's mediation keys (half B; rebuilt exactly as run_conjunction_mediation.py and asserted
  equal to its stored mediation_keys.parquet). Per key: TT, II (full v2 edits) and, for each component set C,
  T<-I:C (T edit, the outputs of C at the last token set to their II values) and I<-T:C (the converse). Components:
  attention output (attention.dense) and MLP output of layers 8-23; groups attn_all, mlp_all, all, mlp11_14.
  Check: mlp11_14 equals phase 1's ALL-neuron clamp (stored mediation.npz), max |diff| <= 1e-4.
--mode natural (B2): the half-B pairs' natural passives (contexts drawn as run_conjunction.py), good (T verb) and bad
  (I verb) prompt of each item. Per item: good, bad, and for each neuron set S (masks over MLP 11 then MLP 14 from the
  --sets npz files) g<-b:S (good prompt, S's post-activations at the last token set to the bad prompt's) and b<-g:S.
  Also the core neurons' post-activations in the unclamped good and bad runs (B3c).
Checks: empty-set clamps equal their references; entries outside a clamp unchanged at the hook (0); entries inside equal
their targets.
Outputs (--out-dir): {mode}_keys.parquet, {mode}.npz (by [rows, conds], conds), {mode}_meta.json; natural also
natural_core_acts.parquet.
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


def sample_items(items, contexts, seed):
    """Item ids as run_conjunction.py (same rng draws): per plain pair, `contexts` items without replacement."""
    prim = items[items.bad_class == "plain"]
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in prim.groupby("pair_id"):
        ids = sorted(g.item_id)
        keep += list(rng.choice(ids, min(contexts, len(ids)), replace=False))
    return prim, keep


def edit_keys(args, prompts, items, tok):
    """Phase 1's mediation keys, rebuilt as run_conjunction_mediation.py."""
    plan = pd.read_csv(args.plan, usecols=["row", "item_id", "split", "fold", "side", "cond", "base", "donor", "donor_pair"])
    sel = np.load(args.selection, allow_pickle=True)
    half_b = set(np.asarray(sel["pairs"])[np.asarray(sel["halves"]) == 1])
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
    cols = ["item_id", "donor_pair", "pair", "fold", "base", "donor_T", "donor_I", "plan_row_T", "plan_row_I", "len", "rid"]
    st = pd.read_parquet(Path(args.phase1_dir) / "mediation_keys.parquet")
    assert keys[cols].reset_index(drop=True).equals(st[cols].reset_index(drop=True)), "keys differ from phase 1's"
    return keys, ids_by_frame["was"], half_b


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

    def forward(enc_ids, mask, anc, patch=None, comp_set=None, comp_tgt=None, neu_set=None, neu_tgt=None):
        """One forward. patch: site-8 edit fn on [B, d]. comp_set: list of (kind, l) whose outputs at the anchor are
        set to comp_tgt[(kind, l)]. neu_set: {l: mask [B, D]} with targets neu_tgt[l]. Returns by [B], captured
        component outputs {(kind, l): [B, d]} and neuron post-activations {l: [B, D]} (after any setting)."""
        B = enc_ids.shape[0]
        r = torch.arange(B, device=dev)
        cap_c, cap_n, hs = {}, {}, []
        comp_set = set(comp_set or [])

        def at8(m, i, o):
            h = normalize_output(o).clone()
            h[r, anc] = patch(h[r, anc].float()).to(h.dtype)
            return replace_output(o, h)

        if patch is not None:
            hs.append(L[SITE - 1].register_forward_hook(at8))
        for l in range(SITE, NL):
            for kind, mod in (("attn", L[l].attention.dense), ("mlp", L[l].mlp)):
                def post(m, i, o, key=(kind, l)):
                    h = normalize_output(o)
                    a = h[r, anc].float()
                    if key in comp_set:
                        h = h.clone()
                        h[r, anc] = comp_tgt[key].to(h.dtype)
                        b = h[r, anc].float()
                        checks["hook_in_max"] = max(checks["hook_in_max"], float((b - comp_tgt[key]).abs().max()))
                        cap_c[key] = b
                        return replace_output(o, h)
                    cap_c[key] = a
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
                    cap_n[l] = b
                    return (x,) + tuple(i[1:])
                cap_n[l] = a
                return None
            hs.append(L[l].mlp.dense_4h_to_h.register_forward_pre_hook(pre))
        try:
            with torch.no_grad():
                logits = runner.model(input_ids=enc_ids, attention_mask=mask, use_cache=False).logits
        finally:
            for h in hs:
                h.remove()
        lp = torch.log_softmax(logits[r, anc].float(), -1)
        return lp[:, by_id].logsumexp(-1), cap_c, cap_n

    if args.mode == "components":
        keys, ids_was, _ = edit_keys(args, prompts, items, tok)
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
                yT, cT, _ = forward(x, msk, anc, pT)
                yI, cI, _ = forward(x, msk, anc, pI)
                rix = gg.rid.to_numpy()
                res[rix, ci["TT"]], res[rix, ci["II"]] = yT.cpu().numpy(), yI.cpu().numpy()
                for gname, comps in groups.items():
                    for c, p_, tg in (("T<-I", pT, cI), ("I<-T", pI, cT)):
                        y, _, _ = forward(x, msk, anc, p_, comps, tg)
                        res[rix, ci[f"{c}:{gname}"]] = y.cpu().numpy()
            print(f"len {ln}: {len(g)} keys, {time.time() - t0:.0f}s", flush=True)
        z = np.load(Path(args.phase1_dir) / "mediation.npz", allow_pickle=True)
        zc = [str(c) for c in z["conds"]]
        for c, ref in (("T<-I", "TT"), ("I<-T", "II")):
            checks[f"{c}:empty_vs_{ref}_max"] = float(np.nanmax(np.abs(res[:, ci[f"{c}:empty"]] - res[:, ci[ref]])))
            checks[f"{c}:mlp11_14_vs_phase1_ALL_max"] = float(np.nanmax(np.abs(
                res[:, ci[f"{c}:mlp11_14"]] - z["by"][:, zc.index(f"{c}:ALL")].astype(float))))
        checks["TT_vs_phase1_max"] = float(np.nanmax(np.abs(res[:, ci["TT"]] - z["by"][:, zc.index("TT")].astype(float))))
        checks["II_vs_phase1_max"] = float(np.nanmax(np.abs(res[:, ci["II"]] - z["by"][:, zc.index("II")].astype(float))))
        kcols = ["item_id", "donor_pair", "pair", "fold", "base", "donor_T", "donor_I", "plan_row_T", "plan_row_I", "len", "rid"]
        keys[kcols].to_parquet(out / "components_keys.parquet")
    else:
        sel = np.load(args.selection, allow_pickle=True)
        half_b = set(np.asarray(sel["pairs"])[np.asarray(sel["halves"]) == 1])
        prim, keep = sample_items(items, args.contexts, args.seed)
        keys = prim[prim.item_id.isin(set(keep)) & prim.pair_id.isin(half_b)][
            ["item_id", "pair_id", "context_id", "band", "good_prompt", "bad_prompt"]].reset_index(drop=True)
        keys = keys.rename(columns={"pair_id": "pair"})
        keys["rid"] = np.arange(len(keys))
        names, masks = ["empty"], [np.zeros(2 * D, bool)]
        for f in args.sets.split(","):
            S = np.load(f, allow_pickle=True)
            for n, m in zip(S["names"], S["masks"]):
                n = str(n)
                if n in names:
                    assert (masks[names.index(n)] == m.astype(bool)).all(), f"set {n} differs between files"
                    continue
                names.append(n)
                masks.append(m.astype(bool))
        mk = torch.as_tensor(np.stack(masks), device=dev)
        core = pd.read_csv(args.switch).head(5)
        conds = ["good", "bad"] + [f"{c}:{n}" for c in ("g<-b", "b<-g") for n in names]
        res = np.full((len(keys), len(conds)), np.nan, np.float32)
        ci = {c: k for k, c in enumerate(conds)}
        acts = []
        for i in range(0, len(keys), args.batch):
            gg = keys.iloc[i:i + args.batch]
            eg, ag = runner.encode(gg.good_prompt.tolist())
            eb, ab = runner.encode(gg.bad_prompt.tolist())
            yg, _, ng = forward(eg.input_ids, eg.attention_mask, ag)
            yb, _, nb = forward(eb.input_ids, eb.attention_mask, ab)
            rix = gg.rid.to_numpy()
            res[rix, ci["good"]], res[rix, ci["bad"]] = yg.cpu().numpy(), yb.cpu().numpy()
            a = {"rid": rix}
            for r_ in core.itertuples():
                a[f"good_{r_.neuron}"] = ng[int(r_.layer)][:, int(r_.index)].cpu().numpy()
                a[f"bad_{r_.neuron}"] = nb[int(r_.layer)][:, int(r_.index)].cpu().numpy()
            acts.append(pd.DataFrame(a))
            for k, n in enumerate(names):
                m = {l: mk[k, j * D:(j + 1) * D][None].expand(len(gg), D) for j, l in enumerate(NEU)}
                for c, e, an, tg in (("g<-b", eg, ag, nb), ("b<-g", eb, ab, ng)):
                    y, _, _ = forward(e.input_ids, e.attention_mask, an, neu_set=m, neu_tgt=tg)
                    res[rix, ci[f"{c}:{n}"]] = y.cpu().numpy()
            print(f"{i + len(gg)} / {len(keys)} items, {time.time() - t0:.0f}s", flush=True)
        checks["g<-b:empty_vs_good_max"] = float(np.nanmax(np.abs(res[:, ci["g<-b:empty"]] - res[:, ci["good"]])))
        checks["b<-g:empty_vs_bad_max"] = float(np.nanmax(np.abs(res[:, ci["b<-g:empty"]] - res[:, ci["bad"]])))
        nat = pd.read_parquet(args.stored_natural)
        P = pd.read_csv(args.prompts)
        pid = dict(zip(P.prompt, P.pid))
        sn = nat.set_index("pid").by
        dg = np.abs(res[:, ci["good"]] - sn.reindex(keys.good_prompt.map(pid)).to_numpy())
        db = np.abs(res[:, ci["bad"]] - sn.reindex(keys.bad_prompt.map(pid)).to_numpy())
        checks["stored_natural_missing"] = int(np.isnan(dg).sum() + np.isnan(db).sum())
        checks["stored_natural_median_abs"] = float(np.nanmedian(np.r_[dg, db]))
        checks["stored_natural_max_abs"] = float(np.nanmax(np.r_[dg, db]))
        checks["core"] = core.neuron.tolist()
        pd.concat(acts).sort_values("rid").to_parquet(out / "natural_core_acts.parquet")
        keys.drop(columns=["good_prompt", "bad_prompt"]).to_parquet(out / "natural_keys.parquet")
    checks["nan_cells"] = int(np.isnan(res).sum())
    np.savez_compressed(out / f"{args.mode}.npz", by=res, conds=np.array(conds))
    meta = {"args": vars(args), "rows": len(keys), "pairs": int(keys.pair.nunique()), "checks": checks,
            "seconds": round(time.time() - t0, 1)}
    if args.mode == "components":
        meta["bases_sha256"] = hashlib.sha256((Path(args.site_dir) / "bases.pt").read_bytes()).hexdigest()
    (out / f"{args.mode}_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps(meta, default=str), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True, choices=("components", "natural"))
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--site-dir", default=None)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--phase1-dir", default="results/pythia/das_v2/mech/med_v2")
    ap.add_argument("--sets", default=None, help="natural: comma-separated set npz files")
    ap.add_argument("--switch", default="results/pythia/das_v2/mech/conj_v2/conjunction_switch_neurons.csv")
    ap.add_argument("--stored-natural", default="results/pythia/das_v2/expanded_eval/passive_rule/natural.parquet")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--contexts", type=int, default=32)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=128)
    run(ap.parse_args())
