#!/usr/bin/env python3
"""Round 4, E13-T: oracle at the translation stage (reports/round4/plan.md, E13-T). fp32.

Items: the curated passive_1 ("The N was V by the X.") and passive_2 ("The N was V.") sentences of all 126
band-cross pairs (`data/passive_band_cross/pairs.jsonl`; contexts x pairs fully crossed).
1. Natural pass over every sentence: whole / verb / suffix / first-suffix-token log-probs (as
   `run_oracle.py`), and, at the participle's last token, the output vectors of MLPs 11-17 and the
   post-activations (input of dense_4h_to_h) of the neuron sets of `--neurons` (`build_oracle_neurons.py`:
   `neurons` = D11's top 50, `random1`..`random5`, `wmatched`) in MLPs 11 and 14. Sums per band (Head,
   XTail) x context x side; per-pair values of the Head pairs (for leave-own-pair-out).
2. Targets: the Head mean of the same context and side (true class), own pair excluded; Head items also get
   the XTail mean (reverse control).
3. Every sentence is scored under each intervention, one at a time: `mlp` (MLP 11-17 outputs set jointly)
   and one per neuron set (its post-activations set).
Checks: self-patch (own natural values as targets reproduce the natural scores); replacement (with distinct
targets, the patched entries equal the targets and nothing else in the hooked tensors changes); after each
intervention, the verb's log-prob and whole - suffix are unchanged.

Output (`--out-dir`): `oracle_t_scores.parquet`, `oracle_t_meta.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_das_round2 import Runner

LAYERS = tuple(range(11, 18))
NEURON_LAYERS = (11, 14)
BANDS = ("head", "xtail")  # target bands


def build_sentences(args, tok):
    bca = pd.read_json(args.pairs, lines=True)
    bca = bca[bca.context_set == "curated"].reset_index(drop=True)
    n_pairs, n_ctx = bca.verb_pair.nunique(), bca.context_id.nunique()
    assert len(bca) == n_pairs * n_ctx == len(bca.drop_duplicates(["verb_pair", "context_id"])), "contexts x pairs not crossed"
    parts = []
    for side in ("good", "bad"):
        parts.append(pd.DataFrame({"paradigm": bca.paradigm, "context_id": bca.context_id, "verb_pair": bca.verb_pair,
                                   "band": bca.verb_band, "side": side, "sent": bca[f"sentence_{side}"],
                                   "pv": bca.prefix + bca[f"{side}_verb"], "prefix": bca.prefix}))
    S = pd.concat(parts, ignore_index=True)
    enc_pv = [tok.encode(x, add_special_tokens=False) for x in S.pv]
    enc_pre = [tok.encode(x.rstrip(), add_special_tokens=False) for x in S.prefix]
    enc_full = [tok.encode(x, add_special_tokens=False) for x in S.sent]
    assert all(f[:len(p)] == p for f, p in zip(enc_full, enc_pv)), "participle prompt is not a token prefix"
    assert all(p[:len(q)] == q for p, q in zip(enc_pv, enc_pre)), "prefix is not a token prefix"
    S["npv"], S["npre"], S["ntok"] = [len(p) for p in enc_pv], [len(q) for q in enc_pre], [len(f) for f in enc_full]
    by_id = tok.encode(" by", add_special_tokens=False)[0]
    p1 = (S.paradigm == "passive_1").to_numpy()
    assert all(enc_full[i][S.npv[i]] == by_id for i in np.flatnonzero(p1)), "first suffix token is not ' by'"
    S["ctx"] = S.context_id.map({c: i for i, c in enumerate(sorted(S.context_id.unique()))})
    S["s"] = (S.side == "bad").astype(int)
    heads = sorted(S[S.band == "head"].verb_pair.unique())
    S["h"] = S.verb_pair.map({p: i for i, p in enumerate(heads)}).fillna(-1).astype(int)
    return S.sort_values(["ntok", "ctx", "verb_pair", "s"]).reset_index(drop=True), heads


class Oracle:
    def __init__(self, runner, nd):
        self.r, self.torch = runner, runner.torch
        self.layers = runner.model.gpt_neox.layers
        t, dev = self.torch, runner.device
        nd = nd.reset_index(drop=True)
        self.sets = list(dict.fromkeys(nd.set))
        self.n_neu = len(nd)
        # per neuron layer: the neuron indices captured (all sets) and their columns in the captured vector
        self.nidx = {l: t.as_tensor(nd[nd.layer == l]["index"].to_numpy(), device=dev) for l in NEURON_LAYERS}
        self.ncol = {l: t.as_tensor(np.flatnonzero((nd.layer == l).to_numpy()), device=dev) for l in NEURON_LAYERS}
        # per set and layer: (neuron indices, columns), aligned
        self.set_idx = {}
        for s in self.sets:
            for l in NEURON_LAYERS:
                m = ((nd.layer == l) & (nd.set == s)).to_numpy()
                self.set_idx[(s, l)] = (t.as_tensor(nd["index"].to_numpy()[m], device=dev),
                                        t.as_tensor(np.flatnonzero(m), device=dev))

    def run(self, sents, npv, npre, capture=False, cond=None, tgt=None, verify=None):
        """Scores [B, 4] (whole, verb, suffix, first); with capture also mlp [B, 7, H] and neu [B, n_neu] at the
        participle's last token (natural runs only); with cond the targets tgt (mlp: [B, 7, H]; a neuron set:
        [B, n_neu], that set's columns used) are written there; verify (a dict) receives the replacement check:
        max |patched entries - target| and max |change outside the patched entries| over the hooked tensors."""
        t, r = self.torch, self.r
        enc = r.tok(sents, return_tensors="pt", padding=True, add_special_tokens=False).to(r.device)
        B, T = enc.input_ids.shape
        rows = t.arange(B, device=r.device)
        pos = t.as_tensor(npv - 1, device=r.device)
        cap, hs = {}, []

        def check(before, after, mask, patched, target):
            if verify is None:
                return
            verify["target"] = max(verify.get("target", 0.0), float((patched - target).abs().max()))
            verify["outside"] = max(verify.get("outside", 0.0), float((after - before)[~mask].abs().max()))

        for j, l in enumerate(LAYERS):
            def mlp_hook(m, i, o, j=j):
                if capture:
                    cap[("mlp", j)] = o[rows, pos].float()
                if cond != "mlp":
                    return None
                x = o.clone()
                x[rows, pos] = tgt[:, j].to(x.dtype)
                mask = t.zeros(x.shape[:2], dtype=t.bool, device=x.device)
                mask[rows, pos] = True
                check(o, x, mask[..., None].expand_as(x), x[rows, pos].float(), tgt[:, j])
                return x
            hs.append(self.layers[l].mlp.register_forward_hook(mlp_hook))
        for l in NEURON_LAYERS:
            def pre_hook(m, i, l=l):
                if capture:
                    cap[("neu", l)] = i[0][rows, pos][:, self.nidx[l]].float()
                if cond is None or cond == "mlp":
                    return None
                idx, cols = self.set_idx[(cond, l)]
                if len(idx) == 0:
                    return None
                x = i[0].clone()
                x[rows[:, None], pos[:, None], idx[None]] = tgt[:, cols].to(x.dtype)
                if verify is not None:
                    mask = t.zeros_like(x, dtype=t.bool)
                    mask[rows[:, None], pos[:, None], idx[None]] = True
                    check(i[0], x, mask, x[rows[:, None], pos[:, None], idx[None]].float(), tgt[:, cols])
                return (x,) + tuple(i[1:])
            hs.append(self.layers[l].mlp.dense_4h_to_h.register_forward_pre_hook(pre_hook))
        try:
            with t.no_grad():
                lg = r.model(**enc, use_cache=False).logits.float()
        finally:
            for h in hs:
                h.remove()
        lp = t.log_softmax(lg[:, :-1], -1).gather(-1, enc.input_ids[:, 1:, None])[..., 0] * enc.attention_mask[:, 1:]
        j = t.arange(1, T, device=r.device)[None]
        a, b = t.as_tensor(npv, device=r.device)[:, None], t.as_tensor(npre, device=r.device)[:, None]
        sc = t.stack([lp.sum(1), (lp * ((j >= b) & (j < a))).sum(1), (lp * (j >= a)).sum(1), lp[rows, a[:, 0] - 1]], 1)
        if not capture:
            return sc
        neu = t.zeros(B, self.n_neu, device=r.device)
        for l in NEURON_LAYERS:
            neu[:, self.ncol[l]] = cap[("neu", l)]
        return sc, t.stack([cap[("mlp", j)] for j in range(len(LAYERS))], 1), neu


def run(args):
    import torch

    args.dtype = "float32"
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runner = Runner(args)
    dev = runner.device
    S, heads = build_sentences(args, runner.tok)
    nd = pd.read_csv(args.neurons)
    assert set(nd.layer) <= set(NEURON_LAYERS) and not nd.duplicated(["set", "layer", "index"]).any()
    orc = Oracle(runner, nd)
    conds = ["mlp"] + orc.sets
    C, H, NN = S.ctx.max() + 1, runner.model.config.hidden_size, orc.n_neu
    sum_mlp = torch.zeros(2, C, 2, len(LAYERS), H, dtype=torch.float64, device=dev)
    sum_neu = torch.zeros(2, C, 2, NN, dtype=torch.float64, device=dev)
    cnt = torch.zeros(2, C, 2, dtype=torch.float64, device=dev)
    own_mlp = torch.zeros(len(heads), C, 2, len(LAYERS), H, dtype=torch.float32, device=dev)
    own_neu = torch.zeros(len(heads), C, 2, NN, dtype=torch.float32, device=dev)
    nat = np.full((len(S), 4), np.nan, np.float32)
    bix = S.band.map({"head": 0, "xtail": 1}).fillna(-1).astype(int).to_numpy()
    for i in range(0, len(S), args.batch):
        g = S.iloc[i:i + args.batch]
        sc, m, n = orc.run(g.sent.tolist(), g.npv.to_numpy(), g.npre.to_numpy(), capture=True)
        nat[i:i + len(g)] = sc.cpu().numpy()
        c, s, b, h = (torch.as_tensor(x, device=dev) for x in (g.ctx.to_numpy(), g.s.to_numpy(), bix[i:i + len(g)], g.h.to_numpy()))
        k = b >= 0
        sum_mlp.index_put_((b[k], c[k], s[k]), m[k].double(), accumulate=True)
        sum_neu.index_put_((b[k], c[k], s[k]), n[k].double(), accumulate=True)
        cnt.index_put_((b[k], c[k], s[k]), torch.ones(int(k.sum()), dtype=torch.float64, device=dev), accumulate=True)
        kh = h >= 0
        own_mlp[h[kh], c[kh], s[kh]] = m[kh]
        own_neu[h[kh], c[kh], s[kh]] = n[kh]
    print(f"natural pass: {len(S)} sentences in {time.time() - t0:.0f}s", flush=True)
    assert int(cnt[0].min().item()) == int(cnt[0].max().item()) == len(heads), "a context lacks some Head pairs"
    meta = {"args": vars(args), "sentences": int(len(S)), "pairs": int(S.verb_pair.nunique()),
            "bands": S.drop_duplicates("verb_pair").band.value_counts().to_dict(), "conds": conds,
            "target_pairs": {"head_leave_one_out": len(heads) - 1, "head": len(heads), "xtail": int(cnt[1].min().item())},
            "neurons_sha256": hashlib.sha256(Path(args.neurons).read_bytes()).hexdigest(),
            "self_patch_max_abs": {}, "replacement_check": {}, "unchanged_check": {}}

    def targets(g, tb, cond):
        c, s, h = (torch.as_tensor(x, device=dev) for x in (g.ctx.to_numpy(), g.s.to_numpy(), g.h.to_numpy()))
        bi = BANDS.index(tb)
        sm, own = (sum_mlp, own_mlp) if cond == "mlp" else (sum_neu, own_neu)
        tot, n = sm[bi, c, s], cnt[bi, c, s]
        shape = (-1,) + (1,) * (tot.dim() - 1)
        if tb == "head":  # leave own pair out
            is_h = h >= 0
            tot = tot - torch.where(is_h.view(shape), own[h.clamp(min=0), c, s].double(), 0.0)
            n = n - is_h.double()
        return (tot / n.view(shape)).float()

    # checks on a sample spanning paradigms, lengths and bands
    gs = S[S.h >= 0].sample(n=min(args.batch, int((S.h >= 0).sum())), random_state=args.seed)
    ga = S.sample(n=min(args.batch, len(S)), random_state=args.seed + 1)
    for cond in conds:
        c, s, h = (torch.as_tensor(x, device=dev) for x in (gs.ctx.to_numpy(), gs.s.to_numpy(), gs.h.to_numpy()))
        tg = own_mlp[h, c, s] if cond == "mlp" else own_neu[h, c, s]
        sc = orc.run(gs.sent.tolist(), gs.npv.to_numpy(), gs.npre.to_numpy(), cond=cond, tgt=tg).cpu().numpy()
        meta["self_patch_max_abs"][cond] = float(np.abs(sc - nat[gs.index.to_numpy()]).max())
        v = {}
        sc = orc.run(ga.sent.tolist(), ga.npv.to_numpy(), ga.npre.to_numpy(), cond=cond,
                     tgt=targets(ga, "head", cond) + 0.37, verify=v).cpu().numpy()
        v["score_change"] = float(np.abs(sc - nat[ga.index.to_numpy()]).max())  # > 0: the patch reaches the scores
        meta["replacement_check"][cond] = v
    print("self-patch:", meta["self_patch_max_abs"], "\nreplacement:", meta["replacement_check"], flush=True)
    assert max(meta["self_patch_max_abs"].values()) < 1e-3, "self-patch does not reproduce the natural scores"
    assert all(v["target"] == 0.0 and v["outside"] == 0.0 and v["score_change"] > 1e-3
               for v in meta["replacement_check"].values()), "replacement check"
    res = []
    for cond in conds:
        for tb in BANDS:
            W = S if tb == "head" else S[S.band == "head"]
            sc = np.full((len(W), 4), np.nan, np.float32)
            for i in range(0, len(W), args.batch):
                g = W.iloc[i:i + args.batch]
                sc[i:i + len(g)] = orc.run(g.sent.tolist(), g.npv.to_numpy(), g.npre.to_numpy(), cond=cond,
                                           tgt=targets(g, tb, cond)).cpu().numpy()
            o = W[["paradigm", "context_id", "verb_pair", "band", "side"]].copy()
            o[["whole", "verb", "suffix", "first"]] = sc
            o[["nat_whole", "nat_verb", "nat_suffix", "nat_first"]] = nat[W.index.to_numpy()]
            o["t_n"] = (cnt[BANDS.index(tb)][W.ctx.to_numpy(), W.s.to_numpy()].cpu().numpy()
                        - ((W.h >= 0) & (tb == "head")).to_numpy())
            dv = float((o.verb - o.nat_verb).abs().max())
            dws = float(((o.whole - o.suffix) - (o.nat_whole - o.nat_suffix)).abs().max())
            meta["unchanged_check"][f"{cond}@{tb}"] = {"verb": dv, "whole_minus_suffix": dws}
            assert dv < 1e-3 and dws < 1e-3, f"{cond}@{tb}: the patch changed the prefix or verb log-probs"
            res.append(o.assign(cond=cond, target=tb, split=0))
            print(f"{cond} @ {tb}: {len(W)} sentences ({time.time() - t0:.0f}s)", flush=True)
    R = pd.concat(res, ignore_index=True)
    assert not R[["whole", "verb", "suffix", "first"]].isna().any().any()
    R.to_parquet(out / "oracle_t_scores.parquet", index=False)
    meta["seconds"] = round(time.time() - t0, 1)
    (out / "oracle_t_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps(meta, default=str), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="data/passive_band_cross/pairs.jsonl")
    ap.add_argument("--neurons", default="data/round4/oracle_translation/neurons.csv")
    ap.add_argument("--out-dir", default="results/round4/oracle_translation")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
