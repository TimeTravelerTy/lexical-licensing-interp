#!/usr/bin/env python3
"""Site-8 mechanism (followup_plan.md, steps 1 and 2). fp32.

Step 1, signed direct-logit decomposition of the site-8 interchange
(h <- h + ((h_src - h) d) d at the last token, the passive-test patch):
  Delta x_24 = carry (Delta x_8) + sum_{l=8..23} (sum_h Delta head_{l,h} + Delta mlp_l)
exactly (Pythia: parallel attention / MLP; the attention output bias does
not change). Each term is read through the final LayerNorm with the scale
frozen at the PATCHED run, on the centered unembedding:
  contribution_t = r_t . Delta term / sigma_patched,  r_t = center(W_Uc[t] * gamma).
The LN-scale term r_t . x_unpatched (1/sigma_p - 1/sigma_u) is kept separately; actual
Delta(centered logit) - sum(terms) - LN term is the residual (numerical error).
Readouts (centered logits): " by", ".", " the", " him", O-bar (mean over the
27 object-start tokens). Frames: primary bad passive bases (plan rows, T and
I donors, all splits) and held-out intransitive active bases (held-out
transitive / intransitive donors, same subject; all 15 fold bases).
Also: site-12 / site-17 d-coordinates (z units of the same split/fold bases)
before and after the patch.

Step 2, dose-response: the site-8 coordinate set to z in [-0.5, 2.0]
(step 0.05) on primary bad and good passives (split-0 basis) and on DAS
active items (split-0 held-out basis); readouts as in the passive test, plus
the site-12 / site-17 coordinates.

Outputs (`--out-dir`): `decomp_site8.npz` (per frame x condition x pair sums
and counts of every term), `decomp_rows_site8.parquet` (per patch: totals,
LN term, residual, coordinates, Delta log P), `dose_site8.parquet`,
`mechanism_meta_site8.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_das_round2 import O_TOKENS, normalize_output, replace_output
from run_passive_test import READ, PassiveRunner

SITE, DOWN = 8, (12, 17)
RNAMES = ("by", "dot", "the", "him", "Obar")


def load_bases(path, device):
    import torch

    return {(int(k.split("_s")[1].split("_")[0]), int(k.split("_f")[1])): v[:, 0].float().to(device)
            for k, v in torch.load(path).items() if k.startswith("r1_")}


class Mech:
    def __init__(self, runner, args):
        import torch

        self.torch, self.r = torch, runner
        m = runner.model
        self.layers = m.gpt_neox.layers
        self.L, self.H = m.config.num_hidden_layers, m.config.num_attention_heads
        self.dh = m.config.hidden_size // self.H
        self.eps = m.config.layer_norm_eps
        WU = m.embed_out.weight.float()
        WUc = WU - WU.mean(0, keepdim=True)
        gamma = m.gpt_neox.final_layer_norm.weight.float()

        def rvec(ids):
            a = WUc[ids] * gamma
            return a - a.mean(-1, keepdim=True)

        tok = runner.tok
        one = lambda t: tok.encode(t, add_special_tokens=False)[0]
        o_ids = [one(t) for t in O_TOKENS]
        self.R = torch.stack([rvec([one(" by")])[0], rvec([one(".")])[0], rvec([one(" the")])[0],
                              rvec([one(" him")])[0], rvec(o_ids).mean(0)])  # [5, d]
        self.read_ids = {"by": [one(" by")], "dot": [one(".")], "the": [one(" the")], "him": [one(" him")],
                         "Obar": o_ids}
        # per-layer head projections: dense.weight [out, in]; head h uses input columns h*dh:(h+1)*dh
        self.P = {l: (self.layers[l].attention.dense.weight.float().T @ self.R.T).view(self.H, self.dh, 5)
                  for l in range(SITE, self.L)}

    def run(self, prompts, patch=None, decomp=True):
        """One forward. patch(h_last [B,d]) -> new h_last at site 8. Returns dict of last-position captures."""
        torch, r = self.torch, self.r
        enc, anchors = r.encode(prompts)
        rows = torch.arange(len(anchors), device=r.device)
        cap, hs = {}, []

        def at_site8(m, i, o):
            h = normalize_output(o)
            x = h[rows, anchors].float()
            if patch is not None:
                x = patch(x)
                h = h.clone()
                h[rows, anchors] = x.to(h.dtype)
            cap["x8"] = x
            return replace_output(o, h) if patch is not None else None

        hs.append(self.layers[SITE - 1].register_forward_hook(at_site8))
        for s in DOWN:
            hs.append(self.layers[s - 1].register_forward_hook(
                lambda m, i, o, s=s: cap.__setitem__(f"x{s}", normalize_output(o)[rows, anchors].float())))
        if decomp:
            cap["heads"] = torch.empty(len(prompts), self.L - SITE, self.H, 5, device=r.device)
            cap["mlp"] = torch.empty(len(prompts), self.L - SITE, 5, device=r.device)
            for l in range(SITE, self.L):
                def pre(m, i, l=l):
                    z = i[0][rows, anchors].float().view(-1, self.H, self.dh)
                    cap["heads"][:, l - SITE] = torch.einsum("bhd,hdr->bhr", z, self.P[l])
                def post(m, i, o, l=l):
                    cap["mlp"][:, l - SITE] = normalize_output(o)[rows, anchors].float() @ self.R.T
                hs.append(self.layers[l].attention.dense.register_forward_pre_hook(pre))
                hs.append(self.layers[l].mlp.register_forward_hook(post))

            def fin(m, i):
                x = i[0][rows, anchors].float()
                cap["raw_final"] = x @ self.R.T
                cap["sigma"] = torch.sqrt(x.var(-1, unbiased=False) + self.eps)
            hs.append(r.model.gpt_neox.final_layer_norm.register_forward_pre_hook(fin))
        try:
            with torch.no_grad():
                out = r.model(**enc, use_cache=False)
        finally:
            for h in hs:
                h.remove()
        z = out.logits[rows, anchors].float()
        zc = z - z.mean(-1, keepdim=True)
        cap["zc"] = torch.stack([zc[:, self.read_ids[k]].mean(-1) for k in RNAMES], -1)
        # log P(O) = LSE over the object set - LSE over the vocabulary (centered logits)
        cap["lse"] = torch.stack([torch.logsumexp(zc[:, self.read_ids["Obar"]], -1), torch.logsumexp(zc, -1)], -1)
        m, parts = r.readout(out.logits, anchors, full=True)
        cap["lp"] = torch.stack([m] + [parts[k] for k in READ], -1)  # M + READ
        return cap


def scaling(states, das, pairs_folds, bases):
    """Per basis (split, fold): sign (training items), mu_I and mu_T (held-out items), as in the analysis."""
    out = {}
    for (k, f), d in bases.items():
        fo = das.pair_id.map(dict(zip(pairs_folds.pair_id, pairs_folds[f"fold_split{k}"]))).to_numpy()
        x = (states[das.pid.to_numpy()] @ d).cpu().numpy()
        tr = (fo != f) & (das.subject != "David").to_numpy()
        te = fo == f
        cls = das.cls.to_numpy()
        sign = np.sign(x[tr & (cls == 1)].mean() - x[tr & (cls == 0)].mean())
        mt, mi = (sign * x[te & (cls == 1)]).mean(), (sign * x[te & (cls == 0)]).mean()
        out[(k, f)] = (float(sign), float(mi), float(mt))
    return out


def run(args):
    import torch

    args.dtype = "float32"
    t0 = time.time()
    runner = PassiveRunner(args)
    mech = Mech(runner, args)
    P = pd.read_csv(args.prompts)
    prompts = P.prompt.tolist()
    items = pd.read_csv(args.items)
    plan = pd.read_csv(args.plan)
    pairs_folds = pd.read_csv(Path(args.site_dirs["8"]) / "pairs_folds.csv")
    das = pd.read_csv(Path(args.site_dirs["8"]) / "items.csv")
    das["pid"] = das.prompt.map(dict(zip(P.prompt, P.pid)))
    bases = {s: load_bases(Path(args.site_dirs[str(s)]) / "bases.pt", runner.device) for s in (SITE,) + DOWN}

    # unpatched states of every prompt at sites 8, 12, 17 (donor values and coordinate scaling)
    states = {s: torch.empty(len(P), runner.model.config.hidden_size, device=runner.device) for s in (SITE,) + DOWN}
    with torch.no_grad():
        for i in range(0, len(P), args.batch):
            enc, anchors = runner.encode(prompts[i:i + args.batch])
            o = runner.model(**enc, output_hidden_states=True, use_cache=False)
            rows = torch.arange(len(anchors), device=runner.device)
            for s in states:
                states[s][i:i + len(anchors)] = o.hidden_states[s][rows, anchors].float()
    sc = {s: scaling(states[s], das, pairs_folds, bases[s]) for s in states}
    print(f"setup {time.time() - t0:.0f}s", flush=True)

    def zcoord(x, s, key):
        sign, mi, mt = sc[s][key]
        return (sign * (x @ bases[s][key]) - mi) / (mt - mi)

    # ---------------- step 1 rows
    prim = set(items[items.bad_class == "plain"].item_id)
    pas = plan[plan.item_id.isin(prim) & (plan.side == "bad") & plan.cond.isin(["T", "I"])].copy()
    pas["frame"] = "passive"
    pas["pair"] = pas.item_id.map(items.set_index("item_id").pair_id)
    act = []
    for k in range(3):
        fo = dict(zip(pairs_folds.pair_id, pairs_folds[f"fold_split{k}"]))
        das_f = das.assign(fold=das.pair_id.map(fo))
        for f in range(5):
            held = das_f[das_f.fold == f]
            for b in held[held.cls == 0].itertuples():
                for s in held[(held.subject == b.subject) & (held.verb != b.verb)].itertuples():
                    act.append({"frame": "active", "split": k, "fold": f, "base": b.pid, "donor": s.pid,
                                "cond": "T" if s.cls == 1 else "I", "pair": b.pair_id, "item_id": f"{b.verb}|{b.subject}"})
    rows_df = pd.concat([pas[["frame", "split", "fold", "base", "donor", "cond", "pair", "item_id"]],
                         pd.DataFrame(act)], ignore_index=True)
    pair_ix = {p: i for i, p in enumerate(sorted(rows_df.pair.unique()))}
    rows_df["pix"] = rows_df.pair.map(pair_ix)

    nl = mech.L - SITE
    acc = {}  # (frame, cond) -> dict of [n_pairs, ...] sums

    def acc_add(key, pix, **arrs):
        a = acc.setdefault(key, {"n": torch.zeros(len(pair_ix), device=runner.device)})
        a["n"].index_add_(0, pix, torch.ones(len(pix), device=runner.device))
        for k, v in arrs.items():
            if k not in a:
                a[k] = torch.zeros((len(pair_ix),) + tuple(v.shape[1:]), device=runner.device)
            a[k].index_add_(0, pix, v)

    # unpatched captures, once per base prompt
    t0 = time.time()
    base_ids = np.unique(rows_df.base.to_numpy())
    chunks = [mech.run([prompts[j] for j in base_ids[i:i + args.batch]]) for i in range(0, len(base_ids), args.batch)]
    U = {k: torch.cat([c[k] for c in chunks]) for k in chunks[0]}
    upos = {int(b): i for i, b in enumerate(base_ids)}
    del chunks
    print(f"unpatched captures: {len(base_ids)} bases in {time.time() - t0:.0f}s", flush=True)

    t0 = time.time()
    per_row = []
    for (frame, k, f), g in rows_df.groupby(["frame", "split", "fold"]):
        d8 = bases[SITE][(k, f)]
        for i in range(0, len(g), args.batch):
            gg = g.iloc[i:i + args.batch]
            src = states[SITE][torch.as_tensor(gg.donor.to_numpy(), device=runner.device)]
            p = mech.run([prompts[j] for j in gg.base], patch=lambda h: h + ((src - h) @ d8)[:, None] * d8[None])
            ui = torch.as_tensor([upos[int(b)] for b in gg.base], device=runner.device)
            u = {key: U[key][ui] for key in p}
            sp = p["sigma"]
            heads = (p["heads"] - u["heads"]) / sp[:, None, None, None]
            mlp = (p["mlp"] - u["mlp"]) / sp[:, None, None]
            carry = ((p["x8"] - u["x8"]) @ mech.R.T) / sp[:, None]
            ln = u["raw_final"] * (1 / sp - 1 / u["sigma"])[:, None]
            actual = p["zc"] - u["zc"]
            resid = actual - carry - heads.sum((1, 2)) - mlp.sum(1) - ln
            coords = torch.stack([zcoord(u["x8"], 8, (k, f)), zcoord(p["x8"], 8, (k, f)),
                                  zcoord(u["x12"], 12, (k, f)), zcoord(p["x12"], 12, (k, f)),
                                  zcoord(u["x17"], 17, (k, f)), zcoord(p["x17"], 17, (k, f))], -1)
            dlp = p["lp"] - u["lp"]
            for cond in ("T", "I"):
                msk = torch.as_tensor((gg.cond == cond).to_numpy(), device=runner.device)
                if msk.any():
                    acc_add((frame, cond), torch.as_tensor(gg.pix.to_numpy(), device=runner.device)[msk],
                            heads=heads[msk], mlp=mlp[msk], carry=carry[msk], ln=ln[msk], actual=actual[msk],
                            resid=resid[msk], coords=coords[msk], dlp=dlp[msk])
            per_row.append(pd.DataFrame({
                "frame": frame, "split": k, "fold": f, "cond": gg.cond.to_numpy(), "pair": gg.pair.to_numpy(),
                "item_id": gg.item_id.to_numpy(), "base": gg.base.to_numpy(), "donor": gg.donor.to_numpy(),
                **{f"actual_{n}": actual[:, j].cpu().numpy() for j, n in enumerate(RNAMES)},
                **{f"ln_{n}": ln[:, j].cpu().numpy() for j, n in enumerate(RNAMES)},
                **{f"resid_{n}": resid[:, j].cpu().numpy() for j, n in enumerate(RNAMES)},
                **{f"carry_{n}": carry[:, j].cpu().numpy() for j, n in enumerate(RNAMES)},
                **{c: coords[:, j].cpu().numpy() for j, c in enumerate(
                    ("z8_u", "z8_p", "z12_u", "z12_p", "z17_u", "z17_p"))},
                **{f"dlp_{n}": dlp[:, j].cpu().numpy() for j, n in enumerate(("M",) + READ)}}))
    print(f"decomposition: {len(rows_df)} patches in {time.time() - t0:.0f}s", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(per_row, ignore_index=True).to_parquet(out / "decomp_rows_site8.parquet", index=False)
    np.savez_compressed(out / "decomp_site8.npz", pairs=np.array(sorted(pair_ix, key=pair_ix.get)),
                        readouts=np.array(RNAMES), layers=np.arange(SITE, mech.L),
                        **{f"{fr}_{c}_{k}": v.cpu().numpy() for (fr, c), a in acc.items() for k, v in a.items()})

    # ---------------- step 2: dose-response
    t0 = time.time()
    grid = np.round(np.arange(-0.5, 2.0001, 0.05), 2)
    fold0 = plan[plan.split == 0].drop_duplicates("item_id").set_index("item_id").fold
    it = items[items.bad_class == "plain"]
    pid = dict(zip(P.prompt, P.pid))
    dose_bases = []
    for side in ("bad", "good"):
        dose_bases.append(pd.DataFrame({"frame": f"{side}_passive", "base": it[f"{side}_prompt"].map(pid).to_numpy(),
                                        "item_id": it.item_id.to_numpy(), "pair": it.pair_id.to_numpy(),
                                        "fold": it.item_id.map(fold0).to_numpy()}))
    fo0 = dict(zip(pairs_folds.pair_id, pairs_folds.fold_split0))
    dose_bases.append(pd.DataFrame({"frame": np.where(das.cls == 1, "trans_active", "intrans_active"),
                                    "base": das.pid.to_numpy(), "item_id": (das.verb + "|" + das.subject).to_numpy(),
                                    "pair": das.pair_id.to_numpy(), "fold": das.pair_id.map(fo0).to_numpy()}))
    db = pd.concat(dose_bases, ignore_index=True)
    res = []
    for f, g in db.groupby("fold"):
        d8 = bases[SITE][(0, f)]
        sign, mi, mt = sc[SITE][(0, f)]
        for zv in grid:
            target = sign * (mi + zv * (mt - mi))  # raw projection value
            for i in range(0, len(g), args.batch):
                gg = g.iloc[i:i + args.batch]
                c = mech.run([prompts[j] for j in gg.base], patch=lambda h: h + (target - h @ d8)[:, None] * d8[None],
                             decomp=False)
                res.append(pd.DataFrame({
                    "frame": gg.frame.to_numpy(), "item_id": gg.item_id.to_numpy(), "pair": gg.pair.to_numpy(),
                    "z": zv, "z8_check": zcoord(c["x8"], 8, (0, f)).cpu().numpy(),
                    "z12": zcoord(c["x12"], 12, (0, f)).cpu().numpy(), "z17": zcoord(c["x17"], 17, (0, f)).cpu().numpy(),
                    **{n: c["lp"][:, j].cpu().numpy() for j, n in enumerate(("M",) + READ)}}))
    pd.concat(res, ignore_index=True).to_parquet(out / "dose_site8.parquet", index=False)
    print(f"dose-response: {len(db)} bases x {len(grid)} z in {time.time() - t0:.0f}s", flush=True)
    meta = {"args": vars(args), "grid": grid.tolist(), "scaling": {str(s): {f"{k}_{f}": v for (k, f), v in d.items()}
                                                                  for s, d in sc.items()},
            "bases_sha256": {s: hashlib.sha256((Path(p) / "bases.pt").read_bytes()).hexdigest()
                             for s, p in args.site_dirs.items()},
            "n_patches": len(rows_df), "ln_scale": "frozen at the patched run", "dtype": "float32"}
    (out / "mechanism_meta_site8.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", default="data/das_round2/passive_test/prompts.csv")
    ap.add_argument("--items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--plan", default="data/das_round2/passive_test/plan.csv.gz")
    ap.add_argument("--site-dirs", default="8=results/das_round2/final_strict_site8,"
                                           "12=results/das_round2/final_strict_site12,17=results/das_round2/final_strict")
    ap.add_argument("--out-dir", default="results/das_round2/mechanism")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()
    a.site_dirs = dict(x.split("=") for x in a.site_dirs.split(","))
    run(a)
