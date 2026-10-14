#!/usr/bin/env python3
"""Round 5, C9: shared readers (reports/round5/plan.md, C9). fp32.

1. Natural pass: the top-50 switch neurons' post-activations (MLPs 11, 14) at the verb's last token for
   - every filler-gap prompt of the C8 items (both experiments, both frames; `data/round4/fillergap`);
   - the B7 passive items' good and bad prompts ("The N was V").
2. In-range patches at site 8 (split-0 basis of each item's fold), coordinate set to t_T and to t_I:
   - filler-gap: the kept I bases (all kept I verbs; the analysis selects populations), C8's non-DAS targets
     (`results/round5/fillergap/targets_{exp}_nondas.parquet`, the plan rows of split 0);
   - passives: the B7 bad items, B5's was-frame targets (`results/round5/necessity/necessity_natural.parquet`).
   In each patched run: the top-50 activations and a direct-effect decomposition of the construction's readout at
   the anchor: per layer 8-23 the MLP output and the attention output (heads summed) projected on r, the site-8
   carry and the final-LN input (raw) and scale sigma. r = centered-logit readout vectors (as Mech.rvec): " by"
   (passive), "." / "?" minus the mean over the PREP set (filler-gap, linear L~). Also the exact log-probs (L with
   logsumexp over PREP) and the PREP logsumexp - mean correction.
Checks: the decomposition of zc_T - zc_I (carry + attention + MLPs + LN term) reproduces the actual centered-logit
change (<= 1e-3); the coordinate equals its target after the patch.

Outputs (--out-dir): sr_natural_fg.parquet, sr_natural_pass.parquet, sr_patch_fg.parquet, sr_patch_pass.parquet,
sr_meta.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_construction_test import SETS
from run_das_round2 import normalize_output, replace_output
from run_passive_test import PassiveRunner
from run_site8_mechanism import SITE, load_bases

NEU_LAYERS = (11, 14)
KEYS = [(k, f) for k in range(3) for f in range(5)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Reader:
    def __init__(self, runner, top):
        import torch

        self.t, self.r = torch, runner
        m = runner.model
        self.layers = m.gpt_neox.layers
        self.L, self.H = m.config.num_hidden_layers, m.config.num_attention_heads
        self.dh = m.config.hidden_size // self.H
        self.eps = m.config.layer_norm_eps
        tok = runner.tok
        one = lambda s: tok.encode(s, add_special_tokens=False)
        WU = m.embed_out.weight.float()
        WUc = WU - WU.mean(0, keepdim=True)
        gamma = m.gpt_neox.final_layer_norm.weight.float()

        def rvec(ids):
            a = WUc[ids] * gamma
            return a - a.mean(-1, keepdim=True)

        prep = [one(s) for s in SETS["PREP"]]
        assert all(len(p) == 1 for p in prep)
        self.prep_ids = torch.tensor([p[0] for p in prep], device=runner.device)
        self.ids = {"by": one(" by")[0], "dot": one(".")[0], "q": one("?")[0]}
        # readout rows: by, dot - PREPmean, ? - PREPmean
        pm = rvec(self.prep_ids).mean(0)
        self.R = torch.stack([rvec([self.ids["by"]])[0], rvec([self.ids["dot"]])[0] - pm, rvec([self.ids["q"]])[0] - pm])
        self.P = {l: (self.layers[l].attention.dense.weight.float().T @ self.R.T).view(self.H, self.dh, 3)
                  for l in range(SITE, self.L)}
        self.nidx = {l: torch.as_tensor(top[top.layer == l]["index"].to_numpy().copy(), device=runner.device)
                     for l in NEU_LAYERS}

    def run(self, prompts, d=None, tcoord=None, decomp=False):
        t, r = self.t, self.r
        enc, anc = r.encode(prompts)
        rows = t.arange(len(anc), device=r.device)
        cap, hs = {}, []

        def at8(m, i, o):
            h = normalize_output(o)
            x = h[rows, anc].float()
            if tcoord is not None:
                x = x + (tcoord - (x * d).sum(-1))[:, None] * d
                h = h.clone()
                h[rows, anc] = x.to(h.dtype)
            cap["x8"] = x
            if d is not None:
                cap["coord"] = (x * d).sum(-1)
            return replace_output(o, h) if tcoord is not None else None

        hs.append(self.layers[SITE - 1].register_forward_hook(at8))
        for l in NEU_LAYERS:
            hs.append(self.layers[l].mlp.dense_4h_to_h.register_forward_pre_hook(
                lambda m, i, l=l: cap.__setitem__(("neu", l), i[0][rows, anc][:, self.nidx[l]].float())))
        if decomp:
            cap["attn"] = t.empty(len(prompts), self.L - SITE, 3, device=r.device)
            cap["mlp"] = t.empty(len(prompts), self.L - SITE, 3, device=r.device)
            for l in range(SITE, self.L):
                def pre(m, i, l=l):
                    z = i[0][rows, anc].float().view(-1, self.H, self.dh)
                    cap["attn"][:, l - SITE] = t.einsum("bhd,hdr->br", z, self.P[l])
                def post(m, i, o, l=l):
                    cap["mlp"][:, l - SITE] = normalize_output(o)[rows, anc].float() @ self.R.T
                hs.append(self.layers[l].attention.dense.register_forward_pre_hook(pre))
                hs.append(self.layers[l].mlp.register_forward_hook(post))

            def fin(m, i):
                x = i[0][rows, anc].float()
                cap["raw"] = x @ self.R.T
                cap["sigma"] = t.sqrt(x.var(-1, unbiased=False) + self.eps)
            hs.append(r.model.gpt_neox.final_layer_norm.register_forward_pre_hook(fin))
        try:
            with t.no_grad():
                out = r.model(**enc, use_cache=False)
        finally:
            for h in hs:
                h.remove()
        z = out.logits[rows, anc].float()
        zc = z - z.mean(-1, keepdim=True)
        lp = t.log_softmax(z, -1)
        prep_lse = t.logsumexp(lp[:, self.prep_ids], -1)
        cap["lp_by"], cap["lp_dot"], cap["lp_q"] = lp[:, self.ids["by"]], lp[:, self.ids["dot"]], lp[:, self.ids["q"]]
        cap["lp_prep"] = prep_lse
        # linear readouts and the logsumexp - mean correction over PREP
        pz = zc[:, self.prep_ids]
        cap["zc"] = t.stack([zc[:, self.ids["by"]], zc[:, self.ids["dot"]] - pz.mean(-1), zc[:, self.ids["q"]] - pz.mean(-1)], -1)
        cap["prep_corr"] = t.logsumexp(pz, -1) - pz.mean(-1)
        cap["neu"] = t.cat([cap[("neu", l)] for l in NEU_LAYERS], -1)
        return cap


def run(args):
    import torch

    args.dtype = "float32"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runner = PassiveRunner(args)
    dev = runner.device
    sw = pd.read_csv(args.switch)
    top = sw[sw["rank"] <= 50].sort_values(["layer", "rank"]).reset_index(drop=True)
    order = pd.concat([top[top.layer == l] for l in NEU_LAYERS])  # column order of cap["neu"]
    RD = Reader(runner, top)
    meta = {"args": vars(args), "neuron_columns": [f"MLP{l}.n{i}" for l, i in zip(order.layer, order["index"])],
            "neuron_rank": order["rank"].tolist(), "neuron_sign_IA": np.sign(order.I_A).tolist(), "checks": {}}
    ncols = [f"a{j}" for j in range(len(order))]

    # 1. natural pass: filler-gap prompts and B7 passive items
    fdir = Path(args.fg_data)
    P = pd.read_csv(fdir / "prompts.csv")
    items = pd.read_csv(fdir / "items.csv")
    acts = np.empty((len(P), len(order)), np.float32)
    for i in range(0, len(P), args.batch):
        c = RD.run(P.prompt.tolist()[i:i + args.batch])
        acts[i:i + args.batch] = c["neu"].cpu().numpy()
    nat_fg = items[["item_id", "exp", "frame", "lemma", "cls", "context_id", "das_pair", "half", "pid"]].copy()
    nat_fg[ncols] = acts[items.pid.to_numpy()]
    nat_fg.to_parquet(out / "sr_natural_fg.parquet", index=False)
    pit = pd.read_csv(args.passive_items)
    prim = pit[pit.bad_class == "plain"].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    keep = []
    for _, g in prim.groupby("pair_id"):
        ids = sorted(g.item_id)
        keep += list(rng.choice(ids, min(args.contexts, len(ids)), replace=False))
    b7 = prim[prim.item_id.isin(set(keep))].reset_index(drop=True)
    recs = []
    for side in ("good", "bad"):
        pr = b7[f"{side}_prompt"].tolist()
        a = np.empty((len(pr), len(order)), np.float32)
        for i in range(0, len(pr), args.batch):
            a[i:i + args.batch] = RD.run(pr[i:i + args.batch])["neu"].cpu().numpy()
        d_ = b7[["item_id", "pair_id", "band", "context_id"]].copy()
        d_["side"] = side
        d_[ncols] = a
        recs.append(d_)
    pd.concat(recs, ignore_index=True).to_parquet(out / "sr_natural_pass.parquet", index=False)
    print(f"natural passes {time.time() - t0:.0f}s", flush=True)

    # 2. in-range patches with decomposition
    npz = np.load(args.bases)
    bases = {kf: torch.tensor(npz[f"das_site{SITE}_s{kf[0]}_f{kf[1]}"].reshape(-1), dtype=torch.float32, device=dev)
             for kf in KEYS}
    pbases = load_bases(Path(args.site_dir) / "bases.pt", dev)
    for f in range(5):  # the split-0 bases of the two files are the same directions
        cs = float(torch.dot(bases[(0, f)], pbases[(0, f)]))
        meta["checks"][f"basis_cos_f{f}"] = cs
        assert abs(abs(cs) - 1) < 1e-4, cs

    def patched_pair(prompts, d, tT, tI, readout_ix):
        cT = RD.run(prompts, d=d, tcoord=tT, decomp=True)
        cI = RD.run(prompts, d=d, tcoord=tI, decomp=True)
        sT, sI = cT["sigma"], cI["sigma"]
        j = readout_ix
        carry = ((cT["x8"] - cI["x8"]) @ RD.R.T)[:, j] / sT
        attn = (cT["attn"][:, :, j] - cI["attn"][:, :, j]) / sT[:, None]
        mlp = (cT["mlp"][:, :, j] - cI["mlp"][:, :, j]) / sT[:, None]
        ln = cI["raw"][:, j] * (1 / sT - 1 / sI)
        actual = cT["zc"][:, j] - cI["zc"][:, j]
        resid = actual - carry - attn.sum(1) - mlp.sum(1) - ln
        meta["checks"]["decomp_resid_max"] = max(meta["checks"].get("decomp_resid_max", 0.0), float(resid.abs().max()))
        meta["checks"]["coord_err_max"] = max(meta["checks"].get("coord_err_max", 0.0),
                                              float((cT["coord"] - tT).abs().max()), float((cI["coord"] - tI).abs().max()))
        o = {"carry": carry, "ln": ln, "actual_lin": actual, "resid": resid,
             "dprep_corr": cT["prep_corr"] - cI["prep_corr"]}
        for k in ("lp_by", "lp_dot", "lp_q", "lp_prep"):
            o[f"d_{k}"] = cT[k] - cI[k]
        o = {k: v.cpu().numpy() for k, v in o.items()}
        o.update({f"attn{SITE + l}": attn[:, l].cpu().numpy() for l in range(attn.shape[1])})
        o.update({f"mlp{SITE + l}": mlp[:, l].cpu().numpy() for l in range(mlp.shape[1])})
        o["neuT"], o["neuI"] = cT["neu"].cpu().numpy(), cI["neu"].cpu().numpy()
        return o

    def collect(df, prompts, ds, tT, tI, readout_ix, extra):
        outs = []
        for i in range(0, len(df), args.batch):
            sl = slice(i, i + args.batch)
            o = patched_pair(prompts[sl], ds[sl], tT[sl], tI[sl], readout_ix)
            rec = {k: np.asarray(v)[sl] for k, v in extra.items()}
            for k, v in o.items():
                if k in ("neuT", "neuI"):
                    for jj in range(v.shape[1]):
                        rec[f"{k}_{jj}"] = v[:, jj]
                else:
                    rec[k] = v
            outs.append(pd.DataFrame(rec))
        return pd.concat(outs, ignore_index=True)

    # filler-gap: split-0 in-range rows of the C8 plans, non-DAS targets (only experiments whose C8 gate passed)
    gates = json.loads((Path(args.fg_results) / "gates_r5.json").read_text())
    fg_out = []
    for exp in [e for e in ("emb", "mat") if gates[e]["passes"]]:
        plan = pd.read_csv(fdir / f"plan_{exp}.csv.gz")
        tg = pd.read_parquet(Path(args.fg_results) / f"targets_{exp}_nondas.parquet")
        assert (tg.row.to_numpy() == plan.row.to_numpy()).all()
        sub = plan[(plan.split == 0) & plan.cond.isin(["inT", "inI"])].copy()
        sub["t"] = tg.loc[sub.index, f"t_site{SITE}"].to_numpy()
        w = sub.pivot_table(index=["item_id", "fold", "base"], columns="cond", values="t").reset_index()
        it = items.set_index("item_id")
        prompts = [P.prompt[b] for b in w.base]
        ds = torch.stack([bases[(0, int(f))] for f in w.fold])
        tT = torch.as_tensor(w.inT.to_numpy(), dtype=torch.float32, device=dev)
        tI = torch.as_tensor(w.inI.to_numpy(), dtype=torch.float32, device=dev)
        extra = {"item_id": w.item_id.to_numpy(), "exp": np.full(len(w), exp), "fold": w.fold.to_numpy(),
                 "lemma": it.lemma.reindex(w.item_id).to_numpy(), "frame": it.frame.reindex(w.item_id).to_numpy(),
                 "context_id": it.context_id.reindex(w.item_id).to_numpy(), "half": it.half.reindex(w.item_id).to_numpy(),
                 "das_pair": it.das_pair.reindex(w.item_id).fillna("").to_numpy(), "tT": w.inT.to_numpy(), "tI": w.inI.to_numpy()}
        fg_out.append(collect(w, prompts, ds, tT, tI, 1 if exp == "emb" else 2, extra))
    pd.concat(fg_out, ignore_index=True).to_parquet(out / "sr_patch_fg.parquet", index=False)
    print(f"filler-gap patches {time.time() - t0:.0f}s; {meta['checks']}", flush=True)

    # passives: B7 bad items, B5 was-frame targets (split-0 fold basis)
    nn = pd.read_parquet(args.b5_natural)
    nn = nn[(nn.frame == "was") & (nn.side == "bad")].set_index("item_id")
    bb = b7.join(nn[["fold", "tT", "tI"]], on="item_id")
    assert bb.tT.notna().all()
    ds = torch.stack([pbases[(0, int(f))] for f in bb.fold])
    tT = torch.as_tensor(bb.tT.to_numpy(), dtype=torch.float32, device=dev)
    tI = torch.as_tensor(bb.tI.to_numpy(), dtype=torch.float32, device=dev)
    extra = {"item_id": bb.item_id.to_numpy(), "pair_id": bb.pair_id.to_numpy(), "band": bb.band.to_numpy(),
             "context_id": bb.context_id.to_numpy(), "fold": bb.fold.to_numpy(), "tT": bb.tT.to_numpy(), "tI": bb.tI.to_numpy()}
    pp = collect(bb, bb.bad_prompt.tolist(), ds, tT, tI, 0, extra)
    pp.to_parquet(out / "sr_patch_pass.parquet", index=False)
    assert meta["checks"]["decomp_resid_max"] <= 1e-3 and meta["checks"]["coord_err_max"] <= 1e-3, meta["checks"]
    meta["seconds"] = round(time.time() - t0, 1)
    meta["inputs"] = {k: sha(v) for k, v in (("switch", args.switch), ("bases", args.bases),
                                              ("b5_natural", args.b5_natural))}
    (out / "sr_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps(meta["checks"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fg-data", default="data/round4/fillergap")
    ap.add_argument("--fg-results", default="results/round5/fillergap")
    ap.add_argument("--passive-items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--b5-natural", default="results/round5/necessity/necessity_natural.parquet")
    ap.add_argument("--bases", default="results/das_round2/bases_rank1.npz")
    ap.add_argument("--site-dir", default="results/das_round2/final_strict_site8")
    ap.add_argument("--switch", default="results/round4/translation/conjunction_switch_neurons.csv")
    ap.add_argument("--out-dir", default="results/round5/shared")
    ap.add_argument("--contexts", type=int, default=32)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=256)
    run(ap.parse_args())
