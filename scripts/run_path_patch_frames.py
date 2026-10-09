#!/usr/bin/env python3
"""Round 4, A5(b): B7 path patching in new frames (reports/round4/plan.md, A5). fp32.

Frame sets (base / counterfactual / sensitivity), built from the B7 rows (primary bad passives,
32 contexts per pair, seed 17, split-0 basis, all T donors):
- adverb: "The N was quickly V" / "The N has quickly V" / "The N had quickly V";
- got:    "The N got V"         / "The N has V"         / "The N had V";
- been:   "The N has been V"    / "The N has V"         / "The N was V" (round 4 D10; the frames differ
          in length, so only joint head-set replacements are computed);
- (plain: "The N was V" / "has" / "had", the B7 frames, for checking the code against B7.)
In every frame the passive test's site-8 T-donor interchange is applied (same donor and basis).
Quantities as `run_path_patch.py` (B7): Delta_l per frame, S_l = Delta_l^cf - Delta_l^base, path
effects of the carry, every head and every MLP below l (singles and joint groups; replacing all
reproduces S_l exactly). Heads are ranked on half A. On half B two head sets are replaced jointly:
the fixed B7 top 5 (L9H7, L10H2, L3H3, L1H6, L8H4) and the newly selected top 5, each with the
head-internal replacements: value at the auxiliary only, value at the adverb only (adverb frame),
all values, pattern only.

Prompts are built and checked on token ids: context tokens, auxiliary (and adverb) position and
participle tokens are verified in every frame; base and counterfactual differ only at the
auxiliary. Sums are kept per half x pair x participle token group (single / multi-token).

Outputs (`--out-dir`): `path_patch_{frames}.npz`, `path_patch_rows_{frames}.parquet`,
`path_patch_meta_{frames}.json`.
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
TOPK = 5
B7_TOP = ("L9H7", "L10H2", "L3H3", "L1H6", "L8H4")
# frame set -> [(frame name, tokens between the subject and the participle)]; the first frame is the base,
# the second the counterfactual, the third the sensitivity frame. "aux" = the first of those tokens and
# "adv" = the second (the adverb, or "been") in the base frame.
FRAME_SETS = {"adverb": [("was", (" was", " quickly")), ("has", (" has", " quickly")), ("had", (" had", " quickly"))],
              "got": [("got", (" got",)), ("has", (" has",)), ("had", (" had",))],
              "plain": [("was", (" was",)), ("has", (" has",)), ("had", (" had",))],
              "been": [("hasbeen", (" has", " been")), ("has", (" has",)), ("was", (" was",))]}


def head_index(name, H):
    l, h = name[1:].split("H")
    return int(l) * H + int(h)


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

    def run(self, ids, pos, patch=None):
        """One forward on token ids [B, T] (equal lengths); pos = {name: index tensor [B]} for attention."""
        torch, r = self.r.torch, self.r
        B, T = ids.shape
        anchors = torch.full((B,), T - 1, device=r.device)
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
                out = r.model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True,
                              output_attentions=True, use_cache=False)
        finally:
            for h in hs:
                h.remove()
        cap["x"] = torch.stack([out.hidden_states[l][rows, anchors].float() for l in range(self.L)], 1)
        cap["att"] = torch.stack([a[rows, :, anchors, :].float() for a in out.attentions], 1)  # [B, L, H, T]
        for k, p in pos.items():
            cap[f"att_{k}"] = cap["att"][rows, :, :, p]  # [B, L, H]
        return cap


def build_frames(rows, it, tok, spec):
    """Token ids per frame for each row, with auxiliary / second-token / previous-token positions (base frame)
    and the participle token count. All checks on ids."""
    toks = {fr: [tok.encode(t, add_special_tokens=False) for t in ts] for fr, ts in spec}
    assert all(len(x) == 1 for v in toks.values() for x in v)
    toks = {fr: [x[0] for x in v] for fr, v in toks.items()}
    was = tok.encode(" was", add_special_tokens=False)[0]
    base = spec[0][0]
    out = {fr: [] for fr, _ in spec}
    meta = []
    for r in rows.itertuples():
        pre = it.prefix[r.item_id].rstrip()  # "The N was"
        assert pre.endswith(" was")
        ctx = tok.encode(pre[: -len(" was")], add_special_tokens=False)  # "The N"
        full = tok.encode(r.was_plain, add_special_tokens=False)  # "The N was V"
        assert full[:len(ctx)] == ctx and full[len(ctx)] == was
        verb = full[len(ctx) + 1:]
        for fr, _ in spec:
            ids = ctx + toks[fr] + verb
            assert tok.encode(tok.decode(ids), add_special_tokens=False) == ids  # round trip
            out[fr].append(ids)
        nb = len(toks[base])
        meta.append({"aux": len(ctx), "adv": len(ctx) + 1 if nb > 1 else -1, "ntok": len(verb),
                     "len": len(ctx) + nb + len(verb)})
    return out, pd.DataFrame(meta, index=rows.index)


def run(args):
    import torch

    args.dtype = "float32"
    t0 = time.time()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    sfx = "" if args.donor_cond == "T" else "_I"
    spec = FRAME_SETS[args.frames]
    frames = tuple(fr for fr, _ in spec)
    adverb = len(spec[0][1]) > 1  # a second token between the subject and the participle in the base frame
    aligned = len({len(ts) for _, ts in spec}) == 1  # all frames the same length: head-internal replacements
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

    # the B7 rows (same seed and selection as run_path_patch.py)
    prim = items[items.bad_class == "plain"]
    rng = np.random.default_rng(args.seed)
    keep = []
    for _, g in prim.groupby("pair_id"):
        ids = sorted(g.item_id)
        keep += list(rng.choice(ids, min(args.contexts, len(ids)), replace=False))
    ctxs = sorted(prim.context_id.unique())
    half_of = dict(zip(ctxs, rng.permutation(len(ctxs)) % 2))  # 0 = A, 1 = B
    rows = plan[(plan.split == 0) & (plan.side == "bad") & (plan.cond == args.donor_cond)
                & plan.item_id.isin(set(keep))].copy()
    it = prim.set_index("item_id")
    rows["pair"] = rows.item_id.map(it.pair_id)
    rows["half"] = rows.item_id.map(it.context_id).map(half_of)
    rows["was_plain"] = [prompts[b] for b in rows.base]
    ids_by_frame, fm = build_frames(rows, it, tok, spec)
    rows = rows.join(fm)
    rows["grp"] = (rows.ntok > 1).astype(int)  # 0 = single-token participle, 1 = multi-token
    for fr in frames[1:]:  # same-length frames differ from the base only at the auxiliary
        for a, b, x, nt in zip(ids_by_frame[frames[0]], ids_by_frame[fr], rows.aux, rows.ntok):
            if len(a) == len(b):
                assert all((u == v) or j == x for j, (u, v) in enumerate(zip(a, b)))
            else:
                assert not aligned and a[-nt:] == b[-nt:] and a[:x] == b[:x]  # same context and participle tokens
    pairs = sorted(rows.pair.unique())
    rows["pix"] = rows.pair.map({p: i for i, p in enumerate(pairs)})
    rows["rid"] = np.arange(len(rows))
    print(f"setup {time.time() - t0:.0f}s; {len(rows)} rows, {rows.item_id.nunique()} items, {len(pairs)} pairs; "
          f"half sizes {rows.half.value_counts().to_dict()}; token groups {rows.grp.value_counts().to_dict()}",
          flush=True)

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
        """idx = (half [B], pair [B], token group [B]); vals [B, ...] summed into acc[name][half, pair, group]."""
        if name not in acc:
            acc[name] = torch.zeros((2, nP, 2) + tuple(vals.shape[1:]), dtype=torch.float64, device=runner.device)
        acc[name].index_put_(idx, vals.double(), accumulate=True)

    def head_out(z, layers):  # z [B, l, H, hs] -> [l, H, B, d]
        return torch.einsum("blhk,lhdk->lhbd", z, WOs[:layers])

    head_sets = {"fixed": [head_index(n, H) for n in B7_TOP]}
    base_fr, cf_fr = frames[0], frames[1]
    per_row = []
    t0 = time.time()
    for half in (0, 1):
        if half == 1:  # select the new top heads on half A (pair means, then the mean over pairs)
            n_ = acc["n"][0].sum(-1)
            ok = n_ > 0
            pm = lambda k: (acc[k][0].sum(1)[ok] / n_[ok].view((-1,) + (1,) * (acc[k][0].dim() - 2))).mean(0)
            S = {l: pm(f"d{cf_fr}_{l}") - pm(f"d{base_fr}_{l}") for l in FLAGGED}
            score = torch.zeros(L * H, dtype=torch.float64, device=runner.device)
            elig = [l for l in FLAGGED if abs(float(S[l])) >= 0.05]
            for l in elig:
                pe = pm(f"pe_{l}")
                score[: l * H] += torch.sign(S[l]) * pe[1:1 + l * H] / abs(S[l])
            head_sets["new"] = [int(i) for i in torch.argsort(score, descending=True)[:TOPK].cpu()]
            np.save(Path(args.out_dir) / f"head_scores_halfA_{args.frames}{sfx}.npy", score.cpu().numpy())
            print(f"half A: S = { {l: round(float(v), 3) for l, v in S.items()} }; eligible {elig}; new top "
                  f"{[f'L{i // H}H{i % H}' for i in head_sets['new']]}", flush=True)
        for f, g in rows[rows.half == half].groupby("fold"):
            d8 = bases[(0, f)]
            g = g.sort_values("len")
            for _, gl in g.groupby("len"):  # equal lengths within a batch (no padding)
                for i in range(0, len(gl), args.batch):
                    gg = gl.iloc[i:i + args.batch]
                    n = len(gg)
                    src = torch.stack([dstate[int(j)] for j in gg.donor])
                    pos = {"aux": torch.as_tensor(gg.aux.to_numpy(), device=runner.device),
                           "prev": torch.as_tensor(gg.len.to_numpy() - 2, device=runner.device)}
                    if adverb:
                        pos["adv"] = torch.as_tensor(gg.adv.to_numpy(), device=runner.device)
                    patch = lambda h: h + ((src - h) @ d8)[:, None] * d8[None]
                    C = {}
                    for fr in frames:
                        ids = torch.as_tensor([ids_by_frame[fr][j] for j in gg.rid], device=runner.device)
                        C[f"{fr}u"] = cap.run(ids, pos)
                        C[f"{fr}p"] = cap.run(ids, pos, patch)
                    for k, c in C.items():
                        c["carry"] = c["x"][:, SITE] - c["x8_pre"] if k.endswith("p") else torch.zeros_like(c["x8_pre"])
                    sref = C[f"{base_fr}p"]["sigma"]
                    key = (torch.full((n,), half, device=runner.device, dtype=torch.long),
                           torch.as_tensor(gg.pix.to_numpy(), device=runner.device),
                           torch.as_tensor(gg.grp.to_numpy(), device=runner.device))
                    add("n", key, torch.ones(n, device=runner.device))
                    for pk in pos:
                        add(f"att_{pk}_base", key, C[f"{base_fr}u"][f"att_{pk}"])
                    aux_ix = pos["aux"]
                    rec = {}
                    for l, rname in FLAGGED.items():
                        r = mech.R[RIX[rname]]
                        lay = cap.layers[l]
                        F = lambda x: lay.mlp(lay.post_attention_layernorm(x)) @ r
                        with torch.no_grad():
                            xw = {c: C[f"{base_fr}{c}"]["x"][:, l] for c in "pu"}
                            dbase = (F(xw["p"]) - F(xw["u"])) / sref
                            dfr = {fr: (F(C[f"{fr}p"]["x"][:, l]) - F(C[f"{fr}u"]["x"][:, l])) / sref for fr in frames[1:]}
                            comp = {}
                            for c in "pu":
                                w, h = C[f"{base_fr}{c}"], C[f"{cf_fr}{c}"]
                                heads = head_out(h["z"][:, :l] - w["z"][:, :l], l).reshape(l * H, n, -1)
                                mlps = (h["mlp"][:, :l] - w["mlp"][:, :l]).permute(1, 0, 2)
                                carry = (h["carry"] - w["carry"])[None]
                                comp[c] = {"singles": torch.cat([carry, heads, mlps], 0),
                                           "heads": heads.sum(0), "mlps": mlps.sum(0), "carry": carry[0]}
                                comp[c]["all"] = comp[c]["heads"] + comp[c]["mlps"] + comp[c]["carry"]
                                if half == 1:
                                    for sname, hset in head_sets.items():
                                        sel = [j for j in hset if j // H < l]
                                        zero = torch.zeros_like(carry[0])
                                        if not sel:
                                            for nm in ("", "_aux", "_adv", "_val", "_pat"):
                                                comp[c][f"{sname}{nm}"] = zero
                                            continue
                                        if not aligned:  # joint replacement only; positions differ across frames
                                            comp[c][sname] = heads[torch.as_tensor(sel, device=runner.device)].sum(0)
                                            for nm in ("_aux", "_adv", "_val", "_pat"):
                                                comp[c][f"{sname}{nm}"] = zero
                                            continue
                                        lh = torch.as_tensor([j // H for j in sel], device=runner.device)
                                        hh = torch.as_tensor([j % H for j in sel], device=runner.device)
                                        comp[c][sname] = heads[torch.as_tensor(sel, device=runner.device)].sum(0)
                                        aw, ah = w["att"][:, lh, hh], h["att"][:, lh, hh]  # [B, k, T]
                                        vw, vh = w["v"][:, lh, hh], h["v"][:, lh, hh]  # [B, k, T, hs]
                                        ar = torch.arange(n, device=runner.device)
                                        Wk = WOs[lh, hh]  # [k, d, hs]
                                        dz = {"_aux": aw[ar, :, aux_ix][..., None] * (vh[ar, :, aux_ix] - vw[ar, :, aux_ix]),
                                              "_val": torch.einsum("bkt,bkth->bkh", aw, vh - vw),
                                              "_pat": torch.einsum("bkt,bkth->bkh", ah - aw, vw)}
                                        if adverb:
                                            a_ = pos["adv"]
                                            dz["_adv"] = aw[ar, :, a_][..., None] * (vh[ar, :, a_] - vw[ar, :, a_])
                                        for nm, v in dz.items():
                                            comp[c][f"{sname}{nm}"] = torch.einsum("bkh,kdh->bd", v, Wk)
                                        if not adverb:
                                            comp[c][f"{sname}_adv"] = zero

                            def pe(dp, du):
                                return (F(xw["p"] + dp) - F(xw["u"] + du)) / sref - dbase

                            singles = []
                            for j in range(0, comp["p"]["singles"].shape[0], args.comp_chunk):
                                sl = slice(j, j + args.comp_chunk)
                                singles.append((F(xw["p"][None] + comp["p"]["singles"][sl])
                                                - F(xw["u"][None] + comp["u"]["singles"][sl])) / sref[None] - dbase[None])
                            singles = torch.cat(singles, 0)  # [N_C, B]
                            groups = {g_: pe(comp["p"][g_], comp["u"][g_]) for g_ in comp["p"] if g_ != "singles"}
                        add(f"d{base_fr}_{l}", key, dbase)
                        for fr in frames[1:]:
                            add(f"d{fr}_{l}", key, dfr[fr])
                        add(f"pe_{l}", key, singles.T)
                        for g_, v in groups.items():
                            add(f"grp_{g_}_{l}", key, v)
                        rec[f"S_{l}"] = (dfr[cf_fr] - dbase).cpu().numpy()
                        rec[f"exact_err_{l}"] = (groups["all"] - (dfr[cf_fr] - dbase)).abs().cpu().numpy()
                        rec[f"mlp_check_{l}"] = (F(xw["p"]) - C[f"{base_fr}p"]["mlp"][:, l] @ r).abs().cpu().numpy()
                    per_row.append(pd.DataFrame({"item_id": gg.item_id.to_numpy(), "pair": gg.pair.to_numpy(),
                                                 "half": half, "ntok": gg.ntok.to_numpy(), "donor": gg.donor.to_numpy(),
                                                 **rec}))
            print(f"half {half} fold {f} done ({time.time() - t0:.0f}s)", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pr = pd.concat(per_row, ignore_index=True)
    pr.to_parquet(out / f"path_patch_rows_{args.frames}{sfx}.parquet", index=False)
    np.savez_compressed(out / f"path_patch_{args.frames}{sfx}.npz", pairs=np.array(pairs), flagged=np.array(list(FLAGGED)),
                        readouts=np.array(list(FLAGGED.values())), frames=np.array(frames),
                        fixed=np.array(head_sets["fixed"]), new=np.array(head_sets["new"]),
                        **{k: v.cpu().numpy() for k, v in acc.items()})
    meta = {"args": vars(args), "frames": frames, "spec": spec, "aligned": aligned, "donor_cond": args.donor_cond,
            "rows": len(rows),
            "items": int(rows.item_id.nunique()), "pairs": len(pairs), "flagged": FLAGGED,
            "fixed_heads": list(B7_TOP), "new_heads": [f"L{i // H}H{i % H}" for i in head_sets["new"]],
            "token_groups": rows.grp.value_counts().to_dict(),
            "component_order": "carry, then heads (layer-major, 16 per layer) for layers < l, then MLPs for layers < l",
            "max_exactness_error": {l: float(pr[f"exact_err_{l}"].max()) for l in FLAGGED},
            "max_mlp_check": {l: float(pr[f"mlp_check_{l}"].max()) for l in FLAGGED},
            "bases_sha256": hashlib.sha256((Path(args.site_dir) / "bases.pt").read_bytes()).hexdigest()}
    (out / f"path_patch_meta_{args.frames}{sfx}.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps({k: meta[k] for k in ("new_heads", "max_exactness_error", "max_mlp_check")}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames", choices=tuple(FRAME_SETS))
    ap.add_argument("--prompts", default="data/das_round2/passive_test/prompts.csv")
    ap.add_argument("--items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--plan", default="data/das_round2/passive_test/plan.csv.gz")
    ap.add_argument("--site-dir", default="results/das_round2/final_strict_site8")
    ap.add_argument("--out-dir", default="results/round4/aux_heads")
    ap.add_argument("--contexts", type=int, default=32)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--comp-chunk", type=int, default=64)
    ap.add_argument("--donor-cond", choices=("T", "I"), default="T",
                    help="donor class of the site-8 interchange (B7: T); outputs get an _I suffix for I")
    run(ap.parse_args())
