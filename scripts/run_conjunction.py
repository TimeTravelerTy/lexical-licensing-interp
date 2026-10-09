#!/usr/bin/env python3
"""Round 4, D11: conjunction neurons in MLPs 11 and 14 (reports/round4/plan.md, D11). fp32.

Rows: the B7 items (primary bad passives, 32 contexts per pair, seed 17; split-0 basis) with both T
and I donors, in the "was" and "has" frames (token ids as `run_path_patch_frames.py`); the site-8
interchange h8 <- h8 + ((h_src - h8).d) d at the participle's last token.
Captured at that token, for every neuron of MLPs 11 and 14: the post-activation a (input of
dense_4h_to_h); plus the final-LN scale sigma of each run. Sums per half (A / B, by context, as B7) x pair
x frame x donor class. The analysis turns activations into direct effects on " by" with one reference
scale, a * (W_out[:, n] . r_by) / sigma_ref (r_by = the centered " by" readout of
`run_site8_mechanism.Mech`; w_by saved).

Generalization passes (no patching):
- natural: every primary passive-test item, good and bad prompt ("The N was V"), sums per pair;
- nonce: the round-3 C8 passive probes in the matched_T / matched_I / balanced_AB / balanced_BA
  contexts, sums per lemma.

Output (`--out-dir`): `conjunction.npz`, `conjunction_meta.json`.
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
from run_path_patch_frames import FRAME_SETS, build_frames
from run_site8_mechanism import SITE, Mech, load_bases

LAYERS = (11, 14)
NONCE_CONDS = ("matched_T", "matched_I", "balanced_AB", "balanced_BA")


class Acts:
    def __init__(self, runner, mech):
        self.r, self.m = runner, mech
        self.layers = runner.model.gpt_neox.layers
        r_by = mech.R[0]
        self.w_by = {l: (r_by @ self.layers[l].mlp.dense_4h_to_h.weight.float()) for l in LAYERS}  # [8192]

    def run(self, ids, anchors, attn, patch=None):
        """One forward; returns {l: a [B, 8192]}, sigma [B]."""
        torch, r = self.r.torch, self.r
        B = ids.shape[0]
        rows = torch.arange(B, device=r.device)
        cap, hs = {}, []

        def at_site8(m, i, o):
            if patch is None:
                return None
            h = normalize_output(o).clone()
            h[rows, anchors] = patch(h[rows, anchors].float()).to(h.dtype)
            return replace_output(o, h)

        hs.append(self.layers[SITE - 1].register_forward_hook(at_site8))
        for l in LAYERS:
            hs.append(self.layers[l].mlp.dense_4h_to_h.register_forward_pre_hook(
                lambda m, i, l=l: cap.__setitem__(l, i[0][rows, anchors].float())))

        def fin(m, i):
            x = i[0][rows, anchors].float()
            cap["sigma"] = torch.sqrt(x.var(-1, unbiased=False) + self.m.eps)
        hs.append(r.model.gpt_neox.final_layer_norm.register_forward_pre_hook(fin))
        try:
            with torch.no_grad():
                r.model(input_ids=ids, attention_mask=attn, use_cache=False)
        finally:
            for h in hs:
                h.remove()
        return cap


def run(args):
    import torch

    args.dtype = "float32"
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runner = PassiveRunner(args)
    mech = Mech(runner, args)
    acts = Acts(runner, mech)
    tok = runner.tok
    P = pd.read_csv(args.prompts)
    prompts = P.prompt.tolist()
    items = pd.read_csv(args.items)
    plan = pd.read_csv(args.plan, usecols=["item_id", "split", "fold", "side", "cond", "base", "donor", "donor_pair"])
    bases = load_bases(Path(args.site_dir) / "bases.pt", runner.device)
    prim = items[items.bad_class == "plain"]
    rng = np.random.default_rng(args.seed)  # the B7 row selection and halves
    keep = []
    for _, g in prim.groupby("pair_id"):
        ids = sorted(g.item_id)
        keep += list(rng.choice(ids, min(args.contexts, len(ids)), replace=False))
    ctxs = sorted(prim.context_id.unique())
    half_of = dict(zip(ctxs, rng.permutation(len(ctxs)) % 2))
    rows = plan[(plan.split == 0) & (plan.side == "bad") & plan.cond.isin(["T", "I"]) & plan.item_id.isin(set(keep))].copy()
    it = prim.set_index("item_id")
    rows["pair"] = rows.item_id.map(it.pair_id)
    rows["half"] = rows.item_id.map(it.context_id).map(half_of)
    rows["was_plain"] = [prompts[b] for b in rows.base]
    spec = [fs for fs in FRAME_SETS["plain"] if fs[0] in ("was", "has")]
    ids_by_frame, fm = build_frames(rows, it, tok, spec)
    rows = rows.join(fm)
    rows["rid"] = np.arange(len(rows))
    pairs = sorted(prim.pair_id.unique())
    pix = {p: i for i, p in enumerate(pairs)}
    rows["pix"] = rows.pair.map(pix)
    donors = np.unique(rows.donor.to_numpy())
    dstate = {}
    with torch.no_grad():
        for i in range(0, len(donors), args.batch):
            ids = donors[i:i + args.batch]
            enc, anchors = runner.encode([prompts[j] for j in ids])
            o = runner.model(**enc, output_hidden_states=True, use_cache=False)
            rr = torch.arange(len(anchors), device=runner.device)
            for j, v in zip(ids, o.hidden_states[SITE][rr, anchors].float()):
                dstate[int(j)] = v
    nP, D = len(pairs), 8192
    acc = {f"a_{l}": torch.zeros(2, nP, 2, 2, D, dtype=torch.float64, device=runner.device)
           for l in LAYERS}  # [half, pair, frame (was, has), cls (T, I)]
    cnt = torch.zeros(2, nP, 2, 2, dtype=torch.float64, device=runner.device)
    sig = torch.zeros(2, nP, 2, 2, dtype=torch.float64, device=runner.device)
    for (f, ln), g in rows.groupby(["fold", "len"]):
        d8 = bases[(0, f)]
        for i in range(0, len(g), args.batch):
            gg = g.iloc[i:i + args.batch]
            src = torch.stack([dstate[int(j)] for j in gg.donor])
            patch = lambda h: h + ((src - h) @ d8)[:, None] * d8[None]
            hh = torch.as_tensor(gg.half.to_numpy(), device=runner.device)
            pp = torch.as_tensor(gg.pix.to_numpy(), device=runner.device)
            cc = torch.as_tensor((gg.cond == "I").astype(int).to_numpy(), device=runner.device)
            for fi, (fr, _) in enumerate(spec):
                ids = torch.as_tensor([ids_by_frame[fr][j] for j in gg.rid], device=runner.device)
                anchors = torch.full((len(gg),), ids.shape[1] - 1, device=runner.device)
                cap = acts.run(ids, anchors, torch.ones_like(ids), patch)
                ff = torch.full_like(hh, fi)
                cnt.index_put_((hh, pp, ff, cc), torch.ones(len(gg), dtype=torch.float64, device=runner.device),
                               accumulate=True)
                sig.index_put_((hh, pp, ff, cc), cap["sigma"].double(), accumulate=True)
                for l in LAYERS:
                    acc[f"a_{l}"].index_put_((hh, pp, ff, cc), cap[l].double(), accumulate=True)
    print(f"patched rows: {len(rows)} x 2 frames in {time.time() - t0:.0f}s", flush=True)
    # natural passives, good and bad, per pair
    nat = {f"nat_{l}": torch.zeros(nP, 2, D, dtype=torch.float64, device=runner.device) for l in LAYERS}
    ncnt = torch.zeros(nP, 2, dtype=torch.float64, device=runner.device)
    sides = pd.concat([pd.DataFrame({"prompt": prim.good_prompt, "pix": prim.pair_id.map(pix), "s": 0}),
                       pd.DataFrame({"prompt": prim.bad_prompt, "pix": prim.pair_id.map(pix), "s": 1})], ignore_index=True)
    for i in range(0, len(sides), args.batch):
        gg = sides.iloc[i:i + args.batch]
        enc, anchors = runner.encode(gg.prompt.tolist())
        cap = acts.run(enc.input_ids, anchors, enc.attention_mask)
        k = (torch.as_tensor(gg.pix.to_numpy(), device=runner.device), torch.as_tensor(gg.s.to_numpy(), device=runner.device))
        ncnt.index_put_(k, torch.ones(len(gg), dtype=torch.float64, device=runner.device), accumulate=True)
        for l in LAYERS:
            nat[f"nat_{l}"].index_put_(k, cap[l].double(), accumulate=True)
    # nonce probes, per lemma and condition
    NP_ = pd.read_csv(args.nonce_prompts)
    NP_ = NP_[(NP_.kind == "nonce") & (NP_.probe_type == "passive") & NP_.cond.isin(NONCE_CONDS)]
    lemmas = sorted(NP_.lemma.unique())
    lix = {x: i for i, x in enumerate(lemmas)}
    non = {f"non_{l}": torch.zeros(len(lemmas), len(NONCE_CONDS), D, dtype=torch.float64, device=runner.device)
           for l in LAYERS}
    ccnt = torch.zeros(len(lemmas), len(NONCE_CONDS), dtype=torch.float64, device=runner.device)
    for i in range(0, len(NP_), args.batch):
        gg = NP_.iloc[i:i + args.batch]
        enc, anchors = runner.encode(gg.prompt.tolist())
        cap = acts.run(enc.input_ids, anchors, enc.attention_mask)
        k = (torch.as_tensor(gg.lemma.map(lix).to_numpy(), device=runner.device),
             torch.as_tensor(gg.cond.map({c: j for j, c in enumerate(NONCE_CONDS)}).to_numpy(), device=runner.device))
        ccnt.index_put_(k, torch.ones(len(gg), dtype=torch.float64, device=runner.device), accumulate=True)
        for l in LAYERS:
            non[f"non_{l}"].index_put_(k, cap[l].double(), accumulate=True)
    save = {"pairs": np.array(pairs), "bands": prim.drop_duplicates("pair_id").set_index("pair_id").band.loc[pairs].to_numpy(),
            "lemmas": np.array(lemmas), "nonce_conds": np.array(NONCE_CONDS), "layers": np.array(LAYERS),
            "cnt": cnt.cpu().numpy(), "sigma": sig.cpu().numpy(), "ncnt": ncnt.cpu().numpy(), "ccnt": ccnt.cpu().numpy(),
            **{f"w_by_{l}": acts.w_by[l].cpu().numpy() for l in LAYERS}}
    for dct in (acc, nat, non):
        for k, v in dct.items():
            save[k] = v.float().cpu().numpy()
    np.savez_compressed(out / "conjunction.npz", **save)
    meta = {"args": vars(args), "rows": len(rows), "pairs": nP, "nonce_lemmas": len(lemmas),
            "bases_sha256": hashlib.sha256((Path(args.site_dir) / "bases.pt").read_bytes()).hexdigest(),
            "seconds": round(time.time() - t0, 1)}
    (out / "conjunction_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps(meta, default=str), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", default="data/das_round2/passive_test/prompts.csv")
    ap.add_argument("--items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--plan", default="data/das_round2/passive_test/plan.csv.gz")
    ap.add_argument("--site-dir", default="results/das_round2/final_strict_site8")
    ap.add_argument("--nonce-prompts", default="data/nonce_passive/prompts.csv")
    ap.add_argument("--out-dir", default="results/round4/translation")
    ap.add_argument("--contexts", type=int, default=32)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    run(ap.parse_args())
