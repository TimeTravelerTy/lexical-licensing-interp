#!/usr/bin/env python3
"""Path patching of the frame switch in the flagged MLPs (round3_plan.md, B7). fp32.

Frames: "The N was V" (passive base), "The N has V" (counterfactual; same
tokens except the auxiliary) and "The N had V" (tense-matched sensitivity,
used for S_l only). In every frame the passive test's site-8 T-donor
interchange is applied (same donor, same split-0 basis):
    h8 <- h8 + ((h_src - h8) . d) d   at the participle's last token.

F_l(x) = MLP_l(LN_l(x)) on the pre-LN residual x before layer l (Pythia's
parallel block). For each flagged MLP l with readout r (O-bar for 15, 18,
21, 22; " by" for 11, 13, 14, 16, 17; r as in `run_site8_mechanism.Mech`):
    Delta_l^f = r . (F_l(x^{f,p}) - F_l(x^{f,u})) / sigma_ref,
sigma_ref = final-LN scale of the was-frame patched run; S_l = Delta_l^has - Delta_l^was.

Path patching C -> MLP_l: add (C^{has,c} - C^{was,c}) to x^{was,c} for both
conditions c in {patched p, unpatched u}, recompute F_l:
    PE_l(C) = r . (F_l(x^{w,p} + dC^p) - F_l(x^{w,u} + dC^u)) / sigma_ref - Delta_l^was.
Components at the last position: the carry (site-8 interchange displacement;
zero in unpatched runs), heads (l', h) and MLPs l' for l' < l. Singles, and
joint groups (all heads, all MLPs, carry, everything = exactness check).

Contexts are split into halves A / B (seed 17). After half A, the top 5
heads are ranked by sum_l sign(S_l) PE_l(h) / |S_l| over eligible MLPs
(|S_l| >= 0.05 on half A). On half B, these 5 are also replaced jointly, and
three head-internal replacements of the 5 are computed (per condition c):
- aux value only:  W_O a^{w,c}_aux (v^{h,c}_aux - v^{w,c}_aux)
- all values:      W_O sum_t a^{w,c}_t (v^{h,c}_t - v^{w,c}_t)
- pattern only:    W_O sum_t (a^{h,c}_t - a^{w,c}_t) v^{w,c}_t
The was-run attention from the participle's last token to the auxiliary is
saved for every head.

Outputs (`--out-dir`): `path_patch_b7.npz` (per half x pair sums, counts),
`path_patch_rows_b7.parquet`, `path_patch_meta_b7.json`.
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
from run_site8_mechanism import SITE, Mech, load_bases

FLAGGED = {11: "by", 13: "by", 14: "by", 16: "by", 17: "by", 15: "Obar", 18: "Obar", 21: "Obar", 22: "Obar"}
RIX = {"by": 0, "Obar": 4}
FRAMES = ("was", "has", "had")
TOPK = 5


class Capture:
    def __init__(self, runner, mech):
        self.r, self.m = runner, mech
        model = runner.model
        try:
            model.set_attn_implementation("eager")
        except Exception:
            model.config._attn_implementation = "eager"
        self.layers = model.gpt_neox.layers
        self.L, self.H, self.hs = mech.L, mech.H, mech.dh

    def run(self, prompts, aux, patch=None):
        """One forward; last-position captures (values and attention at every position)."""
        torch, r = self.r.torch, self.r
        enc, anchors = r.encode(prompts)
        B, T = enc.input_ids.shape
        rows = torch.arange(B, device=r.device)
        cap = {"z": torch.empty(B, self.L, self.H, self.hs, device=r.device),
               "mlp": torch.empty(B, self.L, r.model.config.hidden_size, device=r.device),
               "v": torch.empty(B, self.L, self.H, T, self.hs, device=r.device)}
        hs = []

        def at_site8(m, i, o):
            h = normalize_output(o)
            x = h[rows, anchors].float()
            cap["x8_pre"] = x
            if patch is None:
                return None
            h = h.clone()
            h[rows, anchors] = patch(x).to(h.dtype)
            return replace_output(o, h)

        hs.append(self.layers[SITE - 1].register_forward_hook(at_site8))
        for l in range(self.L):
            att = self.layers[l].attention

            def pre(m, i, l=l):
                cap["z"][:, l] = i[0][rows, anchors].float().view(-1, self.H, self.hs)

            def qkv(m, i, o, l=l):
                cap["v"][:, l] = o.view(B, T, self.H, 3 * self.hs)[..., 2 * self.hs:].float().permute(0, 2, 1, 3)

            def mlp(m, i, o, l=l):
                cap["mlp"][:, l] = normalize_output(o)[rows, anchors].float()

            hs.append(att.dense.register_forward_pre_hook(pre))
            hs.append(att.query_key_value.register_forward_hook(qkv))
            hs.append(self.layers[l].mlp.register_forward_hook(mlp))

        def fin(m, i):
            x = i[0][rows, anchors].float()
            cap["sigma"] = torch.sqrt(x.var(-1, unbiased=False) + self.m.eps)
        hs.append(r.model.gpt_neox.final_layer_norm.register_forward_pre_hook(fin))
        try:
            with torch.no_grad():
                out = r.model(**enc, output_hidden_states=True, output_attentions=True, use_cache=False)
        finally:
            for h in hs:
                h.remove()
        cap["x"] = torch.stack([out.hidden_states[l][rows, anchors].float() for l in range(self.L)], 1)
        cap["att"] = torch.stack([a[rows, :, anchors, :].float() for a in out.attentions], 1)  # [B, L, H, T]
        cap["att_aux"] = cap["att"][rows, :, :, aux]  # [B, L, H]
        return cap


def run(args):
    import torch

    args.dtype = "float32"
    t0 = time.time()
    runner = PassiveRunner(args)
    mech = Mech(runner, args)
    cap = Capture(runner, mech)
    tok = runner.tok
    L, H, hs_ = cap.L, cap.H, cap.hs
    WOs = torch.stack([cap.layers[l].attention.dense.weight.float().view(-1, H, hs_).permute(1, 0, 2)
                       for l in range(L)])  # [L, H, d, hs]
    P = pd.read_csv(args.prompts)
    prompts = P.prompt.tolist()
    items = pd.read_csv(args.items)
    plan = pd.read_csv(args.plan, usecols=["item_id", "split", "fold", "side", "cond", "base", "donor"])
    bases = load_bases(Path(args.site_dir) / "bases.pt", runner.device)

    prim = items[items.bad_class == "plain"]
    rng = np.random.default_rng(args.seed)
    keep = []
    for _, g in prim.groupby("pair_id"):
        ids = sorted(g.item_id)
        keep += list(rng.choice(ids, min(args.contexts, len(ids)), replace=False))
    ctxs = sorted(prim.context_id.unique())
    half_of = dict(zip(ctxs, rng.permutation(len(ctxs)) % 2))  # 0 = A, 1 = B
    rows = plan[(plan.split == 0) & (plan.side == "bad") & (plan.cond == "T") & plan.item_id.isin(set(keep))].copy()
    it = prim.set_index("item_id")
    rows["pair"] = rows.item_id.map(it.pair_id)
    rows["half"] = rows.item_id.map(it.context_id).map(half_of)
    rows["was"] = [prompts[b] for b in rows.base]
    assert all(s.count(" was ") == 1 for s in rows.was)
    for fr in FRAMES[1:]:
        rows[fr] = rows.was.str.replace(" was ", f" {fr} ", regex=False)
    prefix = rows.item_id.map(it.prefix).str.rstrip()
    aux_ix, aux_id = {}, {fr: tok.encode(f" {fr}", add_special_tokens=False) for fr in FRAMES}
    assert all(len(v) == 1 for v in aux_id.values())
    for pre, full in zip(prefix, rows.was):
        if pre not in aux_ix:
            a, b = tok.encode(pre, add_special_tokens=False), tok.encode(full, add_special_tokens=False)
            assert b[:len(a)] == a and a[-1] == aux_id["was"][0], (pre, full)
            for fr in FRAMES[1:]:
                bb = tok.encode(full.replace(" was ", f" {fr} "), add_special_tokens=False)
                assert len(bb) == len(b) and bb[len(a) - 1] == aux_id[fr][0] and bb[len(a):] == b[len(a):]
            aux_ix[pre] = len(a) - 1
    rows["aux"] = [aux_ix[p] for p in prefix]
    pairs = sorted(rows.pair.unique())
    rows["pix"] = rows.pair.map({p: i for i, p in enumerate(pairs)})
    print(f"setup {time.time() - t0:.0f}s; {len(rows)} rows, {rows.item_id.nunique()} items, {len(pairs)} pairs; "
          f"half sizes {rows.half.value_counts().to_dict()}", flush=True)

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

    nP = len(pairs)
    acc = {}

    def add(name, idx, vals):
        """idx = (half index [B], pair index [B]); vals [B, ...] summed into acc[name][half, pair]."""
        if name not in acc:
            acc[name] = torch.zeros((2, nP) + tuple(vals.shape[1:]), dtype=torch.float64, device=runner.device)
        acc[name].index_put_(idx, vals.double(), accumulate=True)

    def head_out(z, layers):  # z [B, l, H, hs] -> [l, H, B, d]
        return torch.einsum("blhk,lhdk->lhbd", z, WOs[:layers])

    top5 = None
    per_row = []
    t0 = time.time()
    for half in (0, 1):
        if half == 1:  # select the top heads on half A
            ok = acc["n"][0] > 0  # pair means first, then the mean over pairs (as in the analysis)
            pm = lambda k: (acc[k][0][ok] / acc["n"][0][ok].view((-1,) + (1,) * (acc[k][0].dim() - 1))).mean(0)
            S = {l: pm(f"dhas_{l}") - pm(f"dwas_{l}") for l in FLAGGED}
            score = torch.zeros(L * H, dtype=torch.float64, device=runner.device)
            elig = [l for l in FLAGGED if abs(float(S[l])) >= 0.05]
            for l in elig:
                pe = pm(f"pe_{l}")  # [N_C]
                score[: l * H] += torch.sign(S[l]) * pe[1:1 + l * H] / abs(S[l])
            top5 = [int(i) for i in torch.argsort(score, descending=True)[:TOPK].cpu()]
            print(f"half A: S = { {l: round(float(v), 3) for l, v in S.items()} }; eligible {elig}; "
                  f"top heads {[f'L{i // H}H{i % H}' for i in top5]}", flush=True)
        for f, g in rows[rows.half == half].groupby("fold"):
            d8 = bases[(0, f)]
            for i in range(0, len(g), args.batch):
                gg = g.iloc[i:i + args.batch]
                n = len(gg)
                src = torch.stack([dstate[int(j)] for j in gg.donor])
                aux = torch.as_tensor(gg.aux.to_numpy(), device=runner.device)
                patch = lambda h: h + ((src - h) @ d8)[:, None] * d8[None]
                C = {}
                for fr in FRAMES:
                    C[f"{fr}u"] = cap.run(gg[fr].tolist(), aux)
                    C[f"{fr}p"] = cap.run(gg[fr].tolist(), aux, patch)
                for k, c in C.items():
                    c["carry"] = c["x"][:, SITE] - c["x8_pre"] if k.endswith("p") else torch.zeros_like(c["x8_pre"])
                sref = C["wasp"]["sigma"]
                hidx = torch.full((n,), half, device=runner.device, dtype=torch.long)
                p_ix = torch.as_tensor(gg.pix.to_numpy(), device=runner.device)
                key = (hidx, p_ix)
                add("n", key, torch.ones(n, device=runner.device))
                add("att_aux_was", key, C["wasu"]["att_aux"])
                rec = {}
                for l, rname in FLAGGED.items():
                    r = mech.R[RIX[rname]]
                    lay = cap.layers[l]
                    F = lambda x: lay.mlp(lay.post_attention_layernorm(x)) @ r
                    with torch.no_grad():
                        xw = {c: C[f"was{c}"]["x"][:, l] for c in "pu"}
                        dwas = (F(xw["p"]) - F(xw["u"])) / sref
                        dfr = {fr: (F(C[f"{fr}p"]["x"][:, l]) - F(C[f"{fr}u"]["x"][:, l])) / sref for fr in FRAMES[1:]}
                        comp = {}
                        for c in "pu":
                            w, h = C[f"was{c}"], C[f"has{c}"]
                            heads = head_out(h["z"][:, :l] - w["z"][:, :l], l).reshape(l * H, n, -1)
                            mlps = (h["mlp"][:, :l] - w["mlp"][:, :l]).permute(1, 0, 2)
                            carry = (h["carry"] - w["carry"])[None]
                            comp[c] = {"singles": torch.cat([carry, heads, mlps], 0),
                                       "heads": heads.sum(0), "mlps": mlps.sum(0), "carry": carry[0]}
                            comp[c]["all"] = comp[c]["heads"] + comp[c]["mlps"] + comp[c]["carry"]
                            if top5 is not None:
                                sel = [j for j in top5 if j // H < l]
                                lh = torch.as_tensor([j // H for j in sel], device=runner.device)
                                hh = torch.as_tensor([j % H for j in sel], device=runner.device)
                                comp[c]["top5"] = heads[torch.as_tensor(sel, device=runner.device)].sum(0) if sel \
                                    else torch.zeros_like(carry[0])
                                if sel:
                                    aw, ah = w["att"][:, lh, hh], h["att"][:, lh, hh]  # [B, k, T]
                                    vw, vh = w["v"][:, lh, hh], h["v"][:, lh, hh]  # [B, k, T, hs]
                                    ar = torch.arange(n, device=runner.device)
                                    dz_aux = aw[ar, :, aux][..., None] * (vh[ar, :, aux] - vw[ar, :, aux])  # [B, k, hs]
                                    dz_val = torch.einsum("bkt,bkth->bkh", aw, vh - vw)
                                    dz_pat = torch.einsum("bkt,bkth->bkh", ah - aw, vw)
                                    Wk = WOs[lh, hh]  # [k, d, hs]
                                    for nm, dz in (("aux", dz_aux), ("val", dz_val), ("pat", dz_pat)):
                                        comp[c][f"top5_{nm}"] = torch.einsum("bkh,kdh->bd", dz, Wk)
                                else:
                                    for nm in ("aux", "val", "pat"):
                                        comp[c][f"top5_{nm}"] = torch.zeros_like(carry[0])

                        def pe(dp, du):
                            return (F(xw["p"] + dp) - F(xw["u"] + du)) / sref - dwas

                        singles = []
                        for j in range(0, comp["p"]["singles"].shape[0], args.comp_chunk):
                            sl = slice(j, j + args.comp_chunk)
                            singles.append((F(xw["p"][None] + comp["p"]["singles"][sl])
                                            - F(xw["u"][None] + comp["u"]["singles"][sl])) / sref[None] - dwas[None])
                        singles = torch.cat(singles, 0)  # [N_C, B]
                        groups = {g_: pe(comp["p"][g_], comp["u"][g_]) for g_ in comp["p"] if g_ != "singles"}
                    add(f"dwas_{l}", key, dwas)
                    for fr in FRAMES[1:]:
                        add(f"d{fr}_{l}", key, dfr[fr])
                    add(f"pe_{l}", key, singles.T)
                    for g_, v in groups.items():
                        add(f"grp_{g_}_{l}", key, v)
                    rec[f"S_{l}"] = (dfr["has"] - dwas).cpu().numpy()
                    rec[f"Shad_{l}"] = (dfr["had"] - dwas).cpu().numpy()
                    rec[f"exact_err_{l}"] = (groups["all"] - (dfr["has"] - dwas)).abs().cpu().numpy()
                    rec[f"mlp_check_{l}"] = (F(xw["p"]) - C["wasp"]["mlp"][:, l] @ r).abs().cpu().numpy()
                per_row.append(pd.DataFrame({"item_id": gg.item_id.to_numpy(), "pair": gg.pair.to_numpy(),
                                             "half": half, "donor": gg.donor.to_numpy(), **rec}))
            print(f"half {half} fold {f} done ({time.time() - t0:.0f}s)", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pr = pd.concat(per_row, ignore_index=True)
    pr.to_parquet(out / "path_patch_rows_b7.parquet", index=False)
    np.savez_compressed(out / "path_patch_b7.npz", pairs=np.array(pairs), flagged=np.array(list(FLAGGED)),
                        readouts=np.array(list(FLAGGED.values())), top5=np.array(top5),
                        **{k: v.cpu().numpy() for k, v in acc.items()})
    meta = {"args": vars(args), "rows": len(rows), "items": int(rows.item_id.nunique()), "pairs": len(pairs),
            "flagged": FLAGGED, "top5_heads": [f"L{i // H}H{i % H}" for i in top5],
            "component_order": "carry, then heads (layer-major, 16 per layer) for layers < l, then MLPs for layers < l",
            "max_exactness_error": {l: float(pr[f"exact_err_{l}"].max()) for l in FLAGGED},
            "max_mlp_check": {l: float(pr[f"mlp_check_{l}"].max()) for l in FLAGGED},
            "bases_sha256": hashlib.sha256((Path(args.site_dir) / "bases.pt").read_bytes()).hexdigest()}
    (out / "path_patch_meta_b7.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps({k: meta[k] for k in ("top5_heads", "max_exactness_error", "max_mlp_check")}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", default="data/das_round2/passive_test/prompts.csv")
    ap.add_argument("--items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--plan", default="data/das_round2/passive_test/plan.csv.gz")
    ap.add_argument("--site-dir", default="results/das_round2/final_strict_site8")
    ap.add_argument("--out-dir", default="results/das_round2/round3")
    ap.add_argument("--contexts", type=int, default=32)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--comp-chunk", type=int, default=64)
    run(ap.parse_args())
