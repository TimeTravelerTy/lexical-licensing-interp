#!/usr/bin/env python3
"""Round 5, B6: serial chain into the late object MLPs (reports/round5/plan.md, B6). fp32.

Rows and frames as round 4 D10 (`run_path_patch_frames.py been`): the B7 items (primary bad passives, 32
contexts per pair, seed 17), all split-0 T-donor rows, site-8 interchange in every patched run; base frame b =
"The N has been V", counterfactual h = "The N has V". Targets t in {15, 18, 21, 22}, readout r = the centered
object-start readout (Mech.R[4]); sigma = the final-LN scale of the (b, p) run for every quantity.
Per row and target (notation of plan.md B6; q = p / u patched / unpatched):
  S_t        = Delta_t(h) - Delta_t(b)
  one_mid    : e^q = sum_{m in M_t} (mlp_m(h, q) - mlp_m(b, q))
  all        : every upstream component (carry, heads < t, MLPs < t) -> must reproduce S_t
  two_par    : v_m = G_m(x_b,m + H_m) - G_m(x_b,m); e = sum_m v_m        (parallel two-hop)
  two_seq    : v_m = G_m(x_b,m + H_m + sum_{k<m} v_k) - G_m(x_b,m)       (sequential)
  single_m   : e = v_m (parallel), for each m in M_t
  two_1114   : e = v_11 + v_14 (parallel)
  neu50      : e = sum over the top-50 switch neurons of (a_n(x_b,m + H_m) - a_n(x_b,m)) W_out[:, n]
  freeze     : has-frame forward with M_t's anchor outputs held at their (b, q) values (11-14 for t = 15,
               11-17 otherwise); remaining = Delta_t in the frozen runs - Delta_t(b)
M_t = MLPs 11..min(17, t-1). Path effects PE_t(e) = r.[G_t(x^p_b,t + e^p) - G_t(x^u_b,t + e^u)] / sigma - Delta_t(b).
Checks: `all` reproduces S_t (<= 1e-5); H = 0 gives zero two-hop effects; a self-freeze (has-frame values)
reproduces Delta_t(h) on the first batch.

Outputs (--out-dir): serial_rows.parquet (one row per patched row), serial_meta.json.
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
from run_path_patch_frames import FRAME_SETS, Capture, build_frames
from run_site8_mechanism import SITE, Mech, load_bases

TARGETS = (15, 18, 21, 22)
MID = tuple(range(11, 18))
NEU_LAYERS = (11, 14)


def mids(t):
    return [m for m in MID if m < t]


def run(args):
    import torch

    args.dtype = "float32"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = FRAME_SETS["been"][:2]  # (hasbeen, has)
    runner = PassiveRunner(args)
    mech = Mech(runner, args)
    cap = Capture(runner, mech)
    dev = runner.device
    tok = runner.tok
    L, H, hs_ = cap.L, cap.H, cap.hs
    layers = cap.layers
    WOs = torch.stack([layers[l].attention.dense.weight.float().view(-1, H, hs_).permute(1, 0, 2) for l in range(L)])
    r_O = mech.R[4]
    sw = pd.read_csv(args.switch)
    top = sw[sw["rank"] <= 50]
    nidx = {l: torch.as_tensor(top[top.layer == l]["index"].to_numpy(), device=dev) for l in NEU_LAYERS}
    assert sum(len(v) for v in nidx.values()) == 50

    P = pd.read_csv(args.prompts)
    prompts = P.prompt.tolist()
    items = pd.read_csv(args.items)
    plan = pd.read_csv(args.plan, usecols=["item_id", "split", "fold", "side", "cond", "base", "donor"])
    bases = load_bases(Path(args.site_dir) / "bases.pt", dev)
    prim = items[items.bad_class == "plain"]
    rng = np.random.default_rng(args.seed)
    keep = []
    for _, g in prim.groupby("pair_id"):
        ids = sorted(g.item_id)
        keep += list(rng.choice(ids, min(args.contexts, len(ids)), replace=False))
    rows = plan[(plan.split == 0) & (plan.side == "bad") & (plan.cond == "T") & plan.item_id.isin(set(keep))].copy()
    it = prim.set_index("item_id")
    rows["pair"] = rows.item_id.map(it.pair_id)
    rows["was_plain"] = [prompts[b] for b in rows.base]
    ids_by_frame, fm = build_frames(rows, it, tok, spec)
    rows = rows.join(fm)
    rows["rid"] = np.arange(len(rows))
    print(f"setup {time.time() - t0:.0f}s; {len(rows)} rows, {rows.item_id.nunique()} items, "
          f"{rows.pair.nunique()} pairs", flush=True)
    donors = np.unique(rows.donor.to_numpy())
    dstate = {}
    with torch.no_grad():
        for i in range(0, len(donors), args.batch):
            idx = donors[i:i + args.batch]
            enc, anchors = runner.encode([prompts[j] for j in idx])
            o = runner.model(**enc, output_hidden_states=True, use_cache=False)
            rr = torch.arange(len(anchors), device=dev)
            for j, v in zip(idx, o.hidden_states[SITE][rr, anchors].float()):
                dstate[int(j)] = v

    def G(l, x):
        lay = layers[l]
        return lay.mlp(lay.post_attention_layernorm(x))

    def acts(l, x):
        lay = layers[l]
        return lay.mlp.act(lay.mlp.dense_h_to_4h(lay.post_attention_layernorm(x)))

    def frozen(ids, patch, frz, vals):
        """Forward with the anchor outputs of MLPs `frz` replaced by vals[:, m]; returns x at every target."""
        B, T = ids.shape
        rr = torch.arange(B, device=dev)
        anc = torch.full((B,), T - 1, device=dev)
        hs, capx = [], {}

        def at8(m, i, o):
            h = normalize_output(o).clone()
            h[rr, anc] = patch(h[rr, anc].float()).to(h.dtype)
            return replace_output(o, h)

        if patch is not None:
            hs.append(layers[SITE - 1].register_forward_hook(at8))
        for m in frz:
            def mh(mod, i, o, m=m):
                y = normalize_output(o).clone()
                y[rr, anc] = vals[:, m].to(y.dtype)
                return replace_output(o, y)
            hs.append(layers[m].mlp.register_forward_hook(mh))
        try:
            with torch.no_grad():
                o = runner.model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True,
                                 use_cache=False)
        finally:
            for h in hs:
                h.remove()
        return {t: o.hidden_states[t][rr, anc].float() for t in TARGETS}

    recs = []
    meta = {"args": vars(args), "rows": len(rows), "pairs": int(rows.pair.nunique()), "checks": {}}
    t0 = time.time()
    first = True
    for f, g in rows.groupby("fold"):
        d8 = bases[(0, f)]
        for _, gl in g.sort_values("len").groupby("len"):
            for i in range(0, len(gl), args.batch):
                gg = gl.iloc[i:i + args.batch]
                n = len(gg)
                src = torch.stack([dstate[int(j)] for j in gg.donor])
                patch = lambda h: h + ((src - h) @ d8)[:, None] * d8[None]
                C, ids_t = {}, {}
                for fr, _ in spec:
                    ids_t[fr] = torch.as_tensor([ids_by_frame[fr][j] for j in gg.rid], device=dev)
                    C[f"{fr}u"] = cap.run(ids_t[fr], {})
                    C[f"{fr}p"] = cap.run(ids_t[fr], {}, patch)
                for k, c in C.items():
                    c["carry"] = c["x"][:, SITE] - c["x8_pre"] if k.endswith("p") else torch.zeros_like(c["x8_pre"])
                b, h = spec[0][0], spec[1][0]
                sref = C[f"{b}p"]["sigma"]
                rec = {"item_id": gg.item_id.to_numpy(), "pair": gg.pair.to_numpy(), "donor": gg.donor.to_numpy(),
                       "ntok": gg.ntok.to_numpy()}
                # head-output differences per layer (has minus has-been) and mid MLP differences, per q
                Hcum, mlpd, x = {}, {}, {}
                for q in "pu":
                    zb, zh = C[f"{b}{q}"]["z"], C[f"{h}{q}"]["z"]
                    per_layer = torch.einsum("blhk,lhdk->lbd", zh - zb, WOs)  # [L, B, d]
                    Hcum[q] = torch.cumsum(per_layer, 0)  # Hcum[q][l] = heads of layers <= l
                    mlpd[q] = C[f"{h}{q}"]["mlp"] - C[f"{b}{q}"]["mlp"]  # [B, L, d]
                    x[q] = C[f"{b}{q}"]["x"]  # [B, L, d] residual inputs (base frame)
                Hbelow = lambda q, m: Hcum[q][m - 1]  # heads of layers < m
                with torch.no_grad():
                    # step 1 (shared by all targets): parallel and sequential v_m for m in 11..17; neuron deltas
                    vpar, vseq, vneu = {q: {} for q in "pu"}, {q: {} for q in "pu"}, {q: {} for q in "pu"}
                    for q in "pu":
                        acc = torch.zeros_like(x[q][:, 0])
                        for m in MID:
                            xm = x[q][:, m]
                            g0 = G(m, xm)
                            vpar[q][m] = G(m, xm + Hbelow(q, m)) - g0
                            vseq[q][m] = G(m, xm + Hbelow(q, m) + acc) - g0
                            acc = acc + vseq[q][m]
                            if m in NEU_LAYERS:
                                da = (acts(m, xm + Hbelow(q, m)) - acts(m, xm))[:, nidx[m]]  # [B, k]
                                Wn = layers[m].mlp.dense_4h_to_h.weight.float()[:, nidx[m]]  # [d, k]
                                vneu[q][m] = da @ Wn.T
                    if first:  # zero source change: the full parallel and sequential paths give zero PE at every target
                        zp, zs = {}, {}
                        for q in "pu":
                            acc = torch.zeros_like(x[q][:, 0])
                            zp[q], zs[q] = {}, {}
                            for m in MID:
                                xm = x[q][:, m]
                                g0 = G(m, xm)
                                zp[q][m] = G(m, xm + 0 * Hbelow(q, m)) - g0
                                zs[q][m] = G(m, xm + 0 * Hbelow(q, m) + acc) - g0
                                acc = acc + zs[q][m]
                        for t in TARGETS:
                            Ft = lambda xx: G(t, xx) @ r_O
                            db = (Ft(x["p"][:, t]) - Ft(x["u"][:, t])) / sref
                            for zz in (zp, zs):
                                ep, eu = (sum(zz[q][m] for m in mids(t)) for q in "pu")
                                v = (Ft(x["p"][:, t] + ep) - Ft(x["u"][:, t] + eu)) / sref - db
                                meta["checks"]["zero_source_max"] = max(meta["checks"].get("zero_source_max", 0.0),
                                                                         float(v.abs().max()))
                    for t in TARGETS:
                        Ft = lambda xx: G(t, xx) @ r_O
                        dbase = (Ft(x["p"][:, t]) - Ft(x["u"][:, t])) / sref
                        dcf = (Ft(C[f"{h}p"]["x"][:, t]) - Ft(C[f"{h}u"]["x"][:, t])) / sref
                        S = dcf - dbase
                        pe = lambda ep, eu: (Ft(x["p"][:, t] + ep) - Ft(x["u"][:, t] + eu)) / sref - dbase
                        M = mids(t)
                        rec[f"S_{t}"] = S.cpu().numpy()
                        rec[f"dbase_{t}"] = dbase.cpu().numpy()
                        e = {}
                        for q in "pu":
                            allc = Hcum[q][t - 1] + mlpd[q][:, :t].sum(1) + (C[f"{h}{q}"]["carry"] - C[f"{b}{q}"]["carry"])
                            e[("all", q)] = allc
                            e[("one_mid", q)] = mlpd[q][:, M].sum(1)
                            e[("one_mlps", q)] = mlpd[q][:, :t].sum(1)
                            e[("two_par", q)] = sum(vpar[q][m] for m in M)
                            e[("two_seq", q)] = sum(vseq[q][m] for m in M)
                            e[("two_1114", q)] = vpar[q][11] + vpar[q][14]
                            e[("neu50", q)] = vneu[q][11] + vneu[q][14]
                            for m in M:
                                e[(f"single_{m}", q)] = vpar[q][m]
                                e[(f"seqsingle_{m}", q)] = vseq[q][m]
                        names = sorted({k for k, _ in e})
                        for nm in names:
                            rec[f"{nm}_{t}"] = pe(e[(nm, "p")], e[(nm, "u")]).cpu().numpy()
                        err = float((torch.as_tensor(rec[f"all_{t}"]) - torch.as_tensor(rec[f"S_{t}"])).abs().max())
                        meta["checks"]["exact_all_max"] = max(meta["checks"].get("exact_all_max", 0.0), err)
                    # (b) freeze runs in the has frame
                    for tag, frz in (("f1114", (11, 12, 13, 14)), ("f1117", MID)):
                        xs = {q: frozen(ids_t[h], patch if q == "p" else None, frz, C[f"{b}{q}"]["mlp"]) for q in "pu"}
                        for t in TARGETS:
                            if (tag == "f1114") != (t == 15):
                                continue
                            Ft = lambda xx: G(t, xx) @ r_O
                            rem = (Ft(xs["p"][t]) - Ft(xs["u"][t])) / sref - torch.as_tensor(rec[f"dbase_{t}"], device=dev)
                            rec[f"freeze_{t}"] = rem.cpu().numpy()
                    if first:  # self-freeze: the has frame's own anchor outputs reproduce Delta_t(h)
                        xs = {q: frozen(ids_t[h], patch if q == "p" else None, MID, C[f"{h}{q}"]["mlp"]) for q in "pu"}
                        for t in TARGETS:
                            Ft = lambda xx: G(t, xx) @ r_O
                            got = (Ft(xs["p"][t]) - Ft(xs["u"][t])) / sref
                            want = (Ft(C[f"{h}p"]["x"][:, t]) - Ft(C[f"{h}u"]["x"][:, t])) / sref
                            meta["checks"]["self_freeze_max"] = max(meta["checks"].get("self_freeze_max", 0.0),
                                                                     float((got - want).abs().max()))
                        first = False
                recs.append(pd.DataFrame(rec))
        print(f"fold {f} done ({time.time() - t0:.0f}s); checks {meta['checks']}", flush=True)
    R = pd.concat(recs, ignore_index=True)
    assert meta["checks"]["exact_all_max"] <= 1e-5, meta["checks"]
    assert meta["checks"]["zero_source_max"] == 0.0 and meta["checks"]["self_freeze_max"] <= 1e-4, meta["checks"]
    R.to_parquet(out / "serial_rows.parquet", index=False)
    meta["bases_sha256"] = hashlib.sha256((Path(args.site_dir) / "bases.pt").read_bytes()).hexdigest()
    meta["switch_sha256"] = hashlib.sha256(Path(args.switch).read_bytes()).hexdigest()
    (out / "serial_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps(meta["checks"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", default="data/das_round2/passive_test/prompts.csv")
    ap.add_argument("--items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--plan", default="data/das_round2/passive_test/plan.csv.gz")
    ap.add_argument("--site-dir", default="results/das_round2/final_strict_site8")
    ap.add_argument("--switch", default="results/round4/translation/conjunction_switch_neurons.csv")
    ap.add_argument("--out-dir", default="results/round5/serial")
    ap.add_argument("--contexts", type=int, default=32)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=32)
    run(ap.parse_args())
