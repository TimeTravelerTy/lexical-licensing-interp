#!/usr/bin/env python3
"""Mediation of the site-8 edit's effect on " by" by MLP 11 / 14 neuron sets (reports/pythia/mechanism_v2_rules.md,
stage 6). fp32.

Rows: as `run_conjunction.py` (bad passives of the plain pairs, 32 contexts per pair, seed 17, split 0, T and I donors
from the plan), restricted to the half-B pairs of a stage-4 selection (`conjunction_selection.npz`), in the "was" frame
(token ids as `run_path_patch_frames.build_frames`). Each plan key (base item, donor pair) has one T row and one I row.
Instrument: the rank-1 interchange at site 8 with the bases of `--site-dir` (fold from the plan),
h <- h + ((h_src - h).d) d at the participle's last token.

Readout log P(" by") at the participle's last token, per key, Y[edit arm, clamp source] with the post-activations
(input of dense_4h_to_h) of a set S at that token clamped to their values in another run of the same key:
- nat (no edit), TT = full T edit, II = full I edit;
- T<-I(S): T edit, S clamped to the I run's values; I<-T(S): I edit, S clamped to the T run's values;
- T<-nat(S), I<-nat(S): either edit, S clamped to the unedited base's values (secondary).
Sets S (masks over the 2 x 8192 neurons of MLP 11 then MLP 14, `--sets` npz) come from `build_mediation_sets.py`; the
empty set is always added (checks: every clamp with the empty set equals its unclamped run). Hook checks: entries
outside S unchanged at the hook, entries in S equal their targets (max |diff|).

Output (--out-dir): mediation_keys.parquet (key meta with the plan rows of both arms), mediation.npz (by[keys, cond],
cond names), mediation_meta.json.
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

LAYERS = (11, 14)
D = 8192


def run(args):
    import torch

    args.dtype = "float32"
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runner = PassiveRunner(args)
    tok, dev = runner.tok, runner.device
    layers = runner.model.gpt_neox.layers
    by_id = runner.ids["by"]
    prompts = pd.read_csv(args.prompts).prompt.tolist()
    items = pd.read_csv(args.items, low_memory=False)
    plan = pd.read_csv(args.plan, usecols=["row", "item_id", "split", "fold", "side", "cond", "base", "donor", "donor_pair"])
    bases = load_bases(Path(args.site_dir) / "bases.pt", dev)
    sel = np.load(args.selection, allow_pickle=True)
    half_b = set(np.asarray(sel["pairs"])[np.asarray(sel["halves"]) == 1])
    S = np.load(args.sets, allow_pickle=True)
    set_names = ["empty"] + [str(x) for x in S["names"]]
    masks = np.concatenate([np.zeros((1, 2 * D), bool), S["masks"].astype(bool)])
    mk = torch.as_tensor(masks, device=dev)  # [nS, 2D]

    # rows exactly as run_conjunction.py (same rng draws), then half B
    prim = items[items.bad_class == "plain"]
    rng = np.random.default_rng(args.seed)
    keep = []
    for _, g in prim.groupby("pair_id"):
        ids = sorted(g.item_id)
        keep += list(rng.choice(ids, min(args.contexts, len(ids)), replace=False))
    plan = plan.rename(columns={"row": "plan_row"}) if "row" in plan.columns else plan
    rows = plan[(plan.split == 0) & (plan.side == "bad") & plan.cond.isin(["T", "I"]) & plan.item_id.isin(set(keep))].copy()
    it = prim.set_index("item_id")
    rows["pair"] = rows.item_id.map(it.pair_id)
    rows = rows[rows.pair.isin(half_b)].copy()
    T_ = rows[rows.cond == "T"].set_index(["item_id", "donor_pair"])
    I_ = rows[rows.cond == "I"].set_index(["item_id", "donor_pair"])
    assert T_.index.is_unique and I_.index.is_unique and T_.index.sort_values().equals(I_.index.sort_values()), "T / I keys"
    keys = T_.join(I_[["donor", "plan_row", "fold", "base"]], rsuffix="_I").reset_index()
    assert (keys.fold == keys.fold_I).all() and (keys.base == keys.base_I).all()
    keys = keys.rename(columns={"donor": "donor_T", "plan_row": "plan_row_T"})
    keys["was_plain"] = [prompts[b] for b in keys.base]
    spec = [("was", (" was",))]
    ids_by_frame, fm = build_frames(keys, it, tok, spec)
    keys = keys.join(fm)
    keys["rid"] = np.arange(len(keys))

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

    def forward(ids, patch=None, setter=None):
        """One forward at the last token. setter(l) -> (mask [B, D] bool, target [B, D]) or None. Returns
        by [B], acts {l: [B, D]} (after any setting), check max |diff| outside / inside the set."""
        B = ids.shape[0]
        r = torch.arange(B, device=dev)
        anc = torch.full((B,), ids.shape[1] - 1, device=dev)
        cap, chk, hs = {}, {"out": 0.0, "in": 0.0}, []

        def at_site8(m, i, o):
            h = normalize_output(o).clone()
            h[r, anc] = patch(h[r, anc].float()).to(h.dtype)
            return replace_output(o, h)

        if patch is not None:
            hs.append(layers[SITE - 1].register_forward_hook(at_site8))
        for l in LAYERS:
            def pre(m, i, l=l):
                x = i[0]
                a = x[r, anc].float()
                if setter is not None:
                    msk, tgt = setter(l)
                    new = torch.where(msk, tgt, a)
                    x = x.clone()
                    x[r, anc] = new.to(x.dtype)
                    b = x[r, anc].float()
                    chk["out"] = max(chk["out"], float((b - a)[~msk].abs().max()) if (~msk).any() else 0.0)
                    chk["in"] = max(chk["in"], float((b - tgt)[msk].abs().max()) if msk.any() else 0.0)
                    cap[l] = b
                    return (x,) + tuple(i[1:])
                cap[l] = a
                return None
            hs.append(layers[l].mlp.dense_4h_to_h.register_forward_pre_hook(pre))
        try:
            with torch.no_grad():
                logits = runner.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        finally:
            for h in hs:
                h.remove()
        lp = torch.log_softmax(logits[r, anc].float(), -1)
        return lp[:, by_id].logsumexp(-1), cap, chk

    conds = ["nat", "TT", "II"] + [f"{c}:{n}" for c in ("T<-I", "I<-T", "T<-nat", "I<-nat") for n in set_names]
    res = np.full((len(keys), len(conds)), np.nan, np.float32)
    ci = {c: k for k, c in enumerate(conds)}
    checks = {"hook_out_max": 0.0, "hook_in_max": 0.0}
    for ln, g in keys.groupby("len"):
        for i in range(0, len(g), args.batch):
            gg = g.iloc[i:i + args.batch]
            ids = torch.as_tensor([ids_by_frame["was"][j] for j in gg.rid], device=dev)
            dd = torch.stack([bases[(0, int(f))] for f in gg.fold])  # [B, d]

            def edit(col):
                src = torch.stack([dstate[int(j)] for j in gg[col]])
                return lambda h: h + ((src - h) * dd).sum(-1, keepdim=True) * dd

            pT, pI = edit("donor_T"), edit("donor_I")
            y_nat, a_nat, _ = forward(ids)
            y_T, a_T, _ = forward(ids, pT)
            y_I, a_I, _ = forward(ids, pI)
            rix = gg.rid.to_numpy()
            for c, y in (("nat", y_nat), ("TT", y_T), ("II", y_I)):
                res[rix, ci[c]] = y.cpu().numpy()
            for k, n in enumerate(set_names):
                m = {l: mk[k, j * D:(j + 1) * D][None].expand(len(gg), D) for j, l in enumerate(LAYERS)}
                for c, p_, tg in (("T<-I", pT, a_I), ("I<-T", pI, a_T), ("T<-nat", pT, a_nat), ("I<-nat", pI, a_nat)):
                    b, _, chk = forward(ids, p_, lambda l: (m[l], tg[l]))
                    res[rix, ci[f"{c}:{n}"]] = b.cpu().numpy()
                    checks["hook_out_max"] = max(checks["hook_out_max"], chk["out"])
                    checks["hook_in_max"] = max(checks["hook_in_max"], chk["in"])
        print(f"len {ln}: {len(g)} keys, {time.time() - t0:.0f}s", flush=True)
    for c, ref in (("T<-I", "TT"), ("T<-nat", "TT"), ("I<-T", "II"), ("I<-nat", "II")):
        checks[f"{c}:empty_vs_{ref}_max"] = float(np.nanmax(np.abs(res[:, ci[f"{c}:empty"]] - res[:, ci[ref]])))
    checks["nan_cells"] = int(np.isnan(res).sum())
    keys[["item_id", "donor_pair", "pair", "fold", "base", "donor_T", "donor_I", "plan_row_T", "plan_row_I", "len", "rid"]
         ].to_parquet(out / "mediation_keys.parquet")
    np.savez_compressed(out / "mediation.npz", by=res, conds=np.array(conds), sets=np.array(set_names))
    meta = {"args": vars(args), "keys": len(keys), "pairs": int(keys.pair.nunique()), "checks": checks,
            "bases_sha256": hashlib.sha256((Path(args.site_dir) / "bases.pt").read_bytes()).hexdigest(),
            "sets_sha256": hashlib.sha256(Path(args.sets).read_bytes()).hexdigest(),
            "seconds": round(time.time() - t0, 1)}
    (out / "mediation_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps(meta, default=str), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--site-dir", required=True)
    ap.add_argument("--selection", required=True, help="stage-4 conjunction_selection.npz (pairs, halves)")
    ap.add_argument("--sets", required=True, help="npz: names [nS], masks [nS, 16384] (MLP11 then MLP14)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--contexts", type=int, default=32)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=128)
    run(ap.parse_args())
