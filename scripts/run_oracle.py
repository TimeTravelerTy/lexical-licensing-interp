#!/usr/bin/env python3
"""Round 4, E13: oracle at d (reports/round4/plan.md, E13).

Items: the curated passive_1 ("The N was V by the X.") and passive_2 ("The N was V.") sentences of the 64
primary pairs (`data/passive_band_cross/pairs.jsonl`). Per site (6, 8):
1. natural pass over the participle prompt (prefix + verb) of every curated pair (all 126 original pairs):
   raw projection onto the site's 15 fold bases;
2. targets under the recipient's own basis (the item's cross-fitted fold for that split: DAS pairs the fold
   where they are held out, others one random fold, seed 17), per paradigm x context x side (good / bad):
   the mean raw projection of the Head-band pairs' participles -> `head`, or of the XTail-band pairs ->
   `xtail` (reverse control on Head items), over pairs that are neither the item's own pair nor training
   pairs of that basis (>= 2 pairs required);
3. every primary sentence (good and bad, both paradigms) x split is scored unpatched and with the coordinate
   set to the target at the participle's last token (h <- h + (t - h.d) d); Head items get both targets,
   others the Head target.
Scores as `score_matched_passives.py`: whole (tokens 1..), verb (participle tokens), suffix (after the
participle) and by (the " by" token, passive_1) log-probs.

Output (`--out-dir`): `oracle_scores.parquet`, `oracle_meta.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_passive_test import PassiveRunner
from run_das_round2 import normalize_output, replace_output

KEYS = [(k, f) for k in range(3) for f in range(5)]


def sentence_lp(runner, sents, n_pv, n_pre, site=None, basis=None, target_raw=None):
    """Per sentence: whole, verb, suffix, first-suffix-token log-probs; optional coordinate setting at the
    participle's last token (index n_pv - 1)."""
    torch = runner.torch
    enc = runner.tok(sents, return_tensors="pt", padding=True, add_special_tokens=False).to(runner.device)
    B, T = enc.input_ids.shape
    rows = torch.arange(B, device=runner.device)
    pos = torch.as_tensor(n_pv - 1, device=runner.device)
    handle = None
    if site is not None:
        tr = torch.as_tensor(target_raw, dtype=torch.float32, device=runner.device)

        def hook(m, i, o):
            h = normalize_output(o).clone()
            x = h[rows, pos].float()
            h[rows, pos] = (x + (tr - x @ basis)[:, None] * basis[None]).to(h.dtype)
            return replace_output(o, h)
        handle = runner.model.gpt_neox.layers[site - 1].register_forward_hook(hook)
    try:
        with torch.no_grad():
            lg = runner.model(**enc, use_cache=False).logits.float()
    finally:
        if handle is not None:
            handle.remove()
    lp = torch.log_softmax(lg[:, :-1], -1).gather(-1, enc.input_ids[:, 1:, None])[..., 0]  # lp of token j at j-1
    lp = lp * enc.attention_mask[:, 1:]
    j = torch.arange(1, T, device=runner.device)[None]
    npv, npre = torch.as_tensor(n_pv, device=runner.device)[:, None], torch.as_tensor(n_pre, device=runner.device)[:, None]
    whole = lp.sum(1)
    verb = (lp * ((j >= npre) & (j < npv))).sum(1)
    suffix = (lp * (j >= npv)).sum(1)
    first = lp[rows, (npv[:, 0] - 1)]
    return torch.stack([whole, verb, suffix, first], 1).cpu().numpy()


def run(args):
    import torch

    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runner = PassiveRunner(args)
    tok = runner.tok
    items = pd.read_csv(args.passive_items)
    allp = items.drop_duplicates("pair_id").set_index("pair_id")
    prim = set(items[items.bad_class == "plain"].pair_id)
    bca = pd.read_json(args.pairs, lines=True)
    bca = bca[(bca.context_set == "curated") & bca.verb_pair.isin(allp.index)].reset_index(drop=True)
    bca["band"] = bca.verb_pair.map(allp.band)
    bca["das_pair"] = bca.verb_pair.map(allp.das_pair).fillna("")
    pf = pd.read_csv(args.folds)
    bc = bca[bca.verb_pair.isin(prim)].reset_index(drop=True)  # scored sentences: primary pairs
    for side in ("good", "bad"):
        bca[f"{side}_pv"] = bca.prefix + bca[f"{side}_verb"]
        pv = (bc.prefix + bc[f"{side}_verb"]).tolist()
        bc[f"{side}_pv"] = pv
        enc_pv = [tok.encode(x, add_special_tokens=False) for x in pv]
        enc_pre = [tok.encode(x.rstrip(), add_special_tokens=False) for x in bc.prefix]
        enc_full = [tok.encode(x, add_special_tokens=False) for x in bc[f"sentence_{side}"]]
        assert all(f[:len(p)] == p for f, p in zip(enc_full, enc_pv)), "participle prompt is not a token prefix"
        assert all(p[:len(q)] == q for p, q in zip(enc_pv, enc_pre)), "prefix is not a token prefix"
        bc[f"{side}_npv"] = [len(p) for p in enc_pv]
        bc[f"{side}_npre"] = [len(q) for q in enc_pre]
    by_id = tok.encode(" by", add_special_tokens=False)[0]
    p1 = bc.paradigm == "passive_1"
    assert all(tok.encode(x, add_special_tokens=False)[n] == by_id
               for x, n in zip(bc.sentence_good[p1], bc.good_npv[p1])), "first suffix token is not ' by'"
    # natural projections of every participle prompt (all pairs) on each site's 15 bases
    pv_prompts = sorted(set(bca.good_pv) | set(bca.bad_pv))
    pid = {p: i for i, p in enumerate(pv_prompts)}
    npz = np.load(args.bases)
    sites = [int(x) for x in args.sites.split(",")]
    B = {st: torch.tensor(np.stack([npz[f"das_site{st}_s{k}_f{f}"].reshape(-1) for k, f in KEYS], 1),
                          dtype=torch.float32, device=runner.device) for st in sites}
    proj = {st: np.empty((len(pv_prompts), 15), np.float32) for st in sites}
    with torch.no_grad():
        for i in range(0, len(pv_prompts), args.batch):
            enc, anchors = runner.encode(pv_prompts[i:i + args.batch])
            o = runner.model(**enc, output_hidden_states=True, use_cache=False)
            rr = torch.arange(len(anchors), device=runner.device)
            for st in sites:
                proj[st][i:i + len(anchors)] = (o.hidden_states[st][rr, anchors].float() @ B[st]).cpu().numpy()
    # fold per (pair, context, split): DAS pairs the fold where held out, others random (seed 17)
    rng = np.random.default_rng(args.seed)
    fold = {}
    pfi = pf.set_index("pair_id")
    for key in sorted(set(zip(bc.verb_pair, bc.context_id))):
        dp = allp.das_pair.get(key[0])
        for k in range(3):
            fold[(key, k)] = int(pfi.loc[dp, f"fold_split{k}"]) if isinstance(dp, str) and dp else int(rng.integers(5))
    train = {(k, f): set(pfi.index[pfi[f"fold_split{k}"] != f]) for k, f in KEYS}
    rows_out, tg_out = [], []
    meta = {"args": vars(args), "sentences": int(len(bc)), "min_target_pairs": {}}
    for st in sites:
        X = proj[st]
        # target sums per basis x paradigm x context x side x band, over pairs that are not training pairs of the basis
        T = []
        for side in ("good", "bad"):
            T.append(bca[["paradigm", "context_id", "verb_pair", "band", "das_pair"]].assign(
                side=side, ix=bca[f"{side}_pv"].map(pid).to_numpy()))
        T = pd.concat(T, ignore_index=True)
        T = T[T.band.isin(["head", "xtail"])]
        sums = {}
        for j, kf in enumerate(KEYS):
            ok = ~T.das_pair.isin(train[kf])
            t = T[ok].assign(x=X[T[ok].ix.to_numpy(), j])
            sums[kf] = t.groupby(["band", "paradigm", "context_id", "side"]).x.agg(["sum", "count"])
            sums[(kf, "own")] = t.set_index(["band", "paradigm", "context_id", "verb_pair", "side"]).x
        work = []
        for side in ("good", "bad"):
            for k in range(3):
                for tb in ("head", "xtail"):
                    sel = bc if tb == "head" else bc[bc.band == "head"]
                    work.append(sel.assign(side=side, split=k, target=tb))
        W = pd.concat(work, ignore_index=True)
        W["fold"] = [fold[((r.verb_pair, r.context_id), r.split)] for r in W.itertuples()]
        W["sent"] = np.where(W.side == "good", W.sentence_good, W.sentence_bad)
        W["npv"] = np.where(W.side == "good", W.good_npv, W.bad_npv)
        W["npre"] = np.where(W.side == "good", W.good_npre, W.bad_npre)
        W["t_raw"], W["t_n"] = np.nan, np.nan
        for (k, f), g in W.groupby(["split", "fold"]):
            sm, own = sums[(k, f)], sums[((k, f), "own")]
            key = list(zip(g.target, g.paradigm, g.context_id, g.side))
            sv = sm.reindex(key)
            ov = own.reindex(list(zip(g.target, g.paradigm, g.context_id, g.verb_pair, g.side)))
            has_own = ov.notna().to_numpy()
            n = sv["count"].to_numpy() - has_own
            W.loc[g.index, "t_raw"] = (sv["sum"].to_numpy() - np.where(has_own, ov.to_numpy(), 0)) / n
            W.loc[g.index, "t_n"] = n
        assert not W.t_raw.isna().any() and W.t_n.min() >= 2, f"site {st}: a target has fewer than 2 eligible pairs"
        meta["min_target_pairs"][str(st)] = {tb: int(W[W.target == tb].t_n.min()) for tb in ("head", "xtail")}
        res = np.full((len(W), 4), np.nan, np.float32)
        for (k, f), g in W.groupby(["split", "fold"]):
            basis = B[st][:, KEYS.index((k, f))]
            for i in range(0, len(g), args.batch):
                gg = g.iloc[i:i + args.batch]
                res[gg.index.to_numpy()] = sentence_lp(runner, gg.sent.tolist(), gg.npv.to_numpy(), gg.npre.to_numpy(),
                                                       st, basis, gg.t_raw.to_numpy())
        Wn = W.drop_duplicates(["paradigm", "context_id", "verb_pair", "side"])
        nat = np.full((len(Wn), 4), np.nan, np.float32)
        for i in range(0, len(Wn), args.batch):
            gg = Wn.iloc[i:i + args.batch]
            nat[i:i + len(gg)] = sentence_lp(runner, gg.sent.tolist(), gg.npv.to_numpy(), gg.npre.to_numpy())
        natd = pd.DataFrame(nat, columns=["nat_whole", "nat_verb", "nat_suffix", "nat_first"])
        natd[["paradigm", "context_id", "verb_pair", "side"]] = Wn[["paradigm", "context_id", "verb_pair", "side"]].to_numpy()
        o = W[["paradigm", "context_id", "verb_pair", "band", "side", "split", "fold", "target", "t_raw", "t_n"]].copy()
        o[["whole", "verb", "suffix", "first"]] = res
        o = o.merge(natd, on=["paradigm", "context_id", "verb_pair", "side"])
        o["site"] = st
        rows_out.append(o)
        print(f"site {st}: {len(W)} patched sentences in {time.time() - t0:.0f}s; {meta['min_target_pairs'][str(st)]}",
              flush=True)
    pd.concat(rows_out, ignore_index=True).to_parquet(out / "oracle_scores.parquet", index=False)
    meta["bases_sha256"] = hashlib.sha256(Path(args.bases).read_bytes()).hexdigest()
    meta["seconds"] = round(time.time() - t0, 1)
    (out / "oracle_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="data/passive_band_cross/pairs.jsonl")
    ap.add_argument("--passive-items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--folds", default="results/das_round2/final_strict/pairs_folds.csv")
    ap.add_argument("--das-items", default="results/das_round2/final_strict/items.csv")
    ap.add_argument("--bases", default="results/das_round2/bases_rank1.npz")
    ap.add_argument("--sites", default="6,8")
    ap.add_argument("--out-dir", default="results/round4/oracle")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
