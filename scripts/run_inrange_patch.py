#!/usr/bin/env python3
"""Round 4, A2: in-range coordinate-setting patch (reports/round4/plan.md, A2).

Constructions: passive (passive-test items, primary pairs), OR and TC (C9 items). For every
bad item x split, with the item's cross-fitted fold basis d of that split (the existing plans'
fold assignments), the coordinate on d at site s is set to two targets (raw projection units):
  t_T = mean projection of the construction's good items in the same context,
  t_I = the same over bad items,
both over primary pairs other than the item's own and outside the basis's DAS training pairs.
Patch: h <- h + (t - h.d) d at the verb's last token (Runner.patched with source t d).

Outputs (`--out-dir`): `natural.parquet` (readouts of every good / bad prompt), `projections.npz`
(every prompt on each site's 15 bases), `rows.parquet` (one row per bad item x split x target,
with the target and the basis), `patches_site{s}.parquet` (readouts per row), `run_meta.json`.
Readouts as C9 (`run_construction_test.ConsRunner`) plus PREP without " by".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_construction_test import READ as CREAD, SETS, ConsRunner

SITES = (4, 6, 8, 10, 12, 14, 16, 17)
KEYS = [(k, f) for k in range(3) for f in range(5)]
READ = tuple(CREAD) + ("prep_noby",)  # prep_noby: the declared PREP set without " by"


class InRangeRunner(ConsRunner):
    def __init__(self, args):
        super().__init__(args)
        ids = [self.tok.encode(t, add_special_tokens=False) for t in SETS["PREP"] if t != " by"]
        assert all(len(i) == 1 for i in ids)
        self.ids["prep_noby"] = self.torch.tensor([i[0] for i in ids], device=self.device)

    def readout(self, logits, anchors, full=False):
        m, parts = super().readout(logits, anchors, full=full)
        if full:
            torch = self.torch
            rows = torch.arange(logits.shape[0], device=logits.device)
            lp = torch.log_softmax(logits[rows, anchors].float(), -1)
            parts["prep_noby"] = torch.logsumexp(lp[:, self.ids["prep_noby"]], -1)
        return m, parts


def load_items(args):
    """Primary items of the three constructions with fold per split: item_id, construction, pair_id, context_id,
    das_pair, good_prompt, bad_prompt, fold_s0..2."""
    pt = Path(args.passive_dir)
    pit = pd.read_csv(pt / "items.csv")
    pit = pit[pit.bad_class == "plain"].assign(construction="passive")
    pplan = pd.read_csv(pt / "plan.csv.gz", usecols=["item_id", "split", "fold"]).drop_duplicates(["item_id", "split"])
    cit = pd.read_csv(Path(args.cons_dir) / "items.csv")
    cplan = pd.read_csv(Path(args.cons_dir) / "plan.csv.gz", usecols=["item_id", "split", "fold"]).drop_duplicates(
        ["item_id", "split"])
    cols = ["item_id", "construction", "pair_id", "band", "context_id", "das_pair", "good_prompt", "bad_prompt"]
    items = pd.concat([pit[cols], cit[cols]], ignore_index=True)
    folds = pd.concat([pplan, cplan]).pivot(index="item_id", columns="split", values="fold")
    folds.columns = [f"fold_s{k}" for k in folds.columns]
    items = items.join(folds, on="item_id")
    assert items[[f"fold_s{k}" for k in range(3)]].notna().all().all(), "every item needs a fold per split"
    items["das_pair"] = items.das_pair.fillna("")
    return items


def targets(items, proj_s, pidx, pf):
    """t_T, t_I per (item, split) at one site. proj_s: [N_prompts, 15] raw projections."""
    fo = pf.set_index("pair_id")
    out = {}
    for k in range(3):
        tT = np.full(len(items), np.nan)
        tI = np.full(len(items), np.nan)
        for f in range(5):
            j = KEYS.index((k, f))
            train = set(fo.index[fo[f"fold_split{k}"] != f])  # DAS training pairs of basis (k, f)
            elig = ~items.das_pair.isin(train).to_numpy()
            g = proj_s[items.good_prompt.map(pidx).to_numpy(), j]
            b = proj_s[items.bad_prompt.map(pidx).to_numpy(), j]
            df = pd.DataFrame({"key": items.construction + "|" + items.context_id, "g": g * elig, "b": b * elig,
                               "n": elig.astype(float)})
            s = df.groupby("key")[["g", "b", "n"]].transform("sum")
            n = s.n - df.n  # leave the item's own pair out (one item per pair and context)
            use = (items[f"fold_s{k}"] == f).to_numpy()
            tT[use] = ((s.g - df.g) / n).to_numpy()[use]
            tI[use] = ((s.b - df.b) / n).to_numpy()[use]
        assert not np.isnan(tT).any() and not np.isnan(tI).any()
        out[k] = (tT, tI)
    return out


def run(args):
    import torch

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runner = InRangeRunner(args)
    items = load_items(args)
    pf = pd.read_csv(args.folds)
    prompts = sorted(set(items.good_prompt) | set(items.bad_prompt))
    pidx = {p: i for i, p in enumerate(prompts)}
    npz = np.load(args.bases)
    bases = {s: {kf: torch.tensor(npz[f"das_site{s}_s{kf[0]}_f{kf[1]}"].reshape(-1), dtype=torch.float32,
                                  device=runner.device) for kf in KEYS} for s in SITES}
    norms = [float(b.norm()) for s in SITES for b in bases[s].values()]
    assert max(abs(n - 1) for n in norms) < 1e-3, "bases must be unit norm for coordinate setting"
    meta = {"args": vars(args), "prompts": len(prompts), "items": items.construction.value_counts().to_dict(),
            "bases_sha256": hashlib.sha256(Path(args.bases).read_bytes()).hexdigest(), "checks": {}}
    t0 = time.time()
    nat = {k: np.empty(len(prompts), np.float32) for k in READ}
    proj = np.empty((len(prompts), len(SITES), 15), np.float32)
    Bst = torch.stack([torch.stack([bases[s][kf] for kf in KEYS], 1) for s in SITES])  # [S, d, 15]
    with torch.no_grad():
        for i in range(0, len(prompts), args.batch):
            enc, anchors = runner.encode(prompts[i:i + args.batch])
            o = runner.model(**enc, output_hidden_states=True, use_cache=False)
            m, parts = runner.readout(o.logits, anchors, full=True)
            rows = torch.arange(len(anchors), device=runner.device)
            x = torch.stack([o.hidden_states[s][rows, anchors].float() for s in SITES], 1)
            proj[i:i + len(m)] = torch.einsum("bsd,sdk->bsk", x, Bst).cpu().numpy()
            nat["M"][i:i + len(m)] = m.cpu().numpy()
            for k in READ[1:]:
                nat[k][i:i + len(m)] = parts[k].cpu().numpy()
    pd.DataFrame({"pid": np.arange(len(prompts)), "prompt": prompts, **nat}).to_parquet(out / "natural.parquet",
                                                                                        index=False)
    np.savez_compressed(out / "projections.npz", sites=np.array(SITES), keys=np.array([f"s{k}_f{f}" for k, f in KEYS]),
                        proj=proj)
    print(f"natural pass: {len(prompts)} prompts in {time.time() - t0:.0f}s", flush=True)
    base_rows = []
    for k in range(3):
        for tgt in ("T", "I"):
            base_rows.append(pd.DataFrame({"item_id": items.item_id, "construction": items.construction,
                                           "split": k, "fold": items[f"fold_s{k}"].astype(int), "target": tgt,
                                           "base": items.bad_prompt.map(pidx)}))
    R = pd.concat(base_rows, ignore_index=True)
    R.insert(0, "row", np.arange(len(R)))
    tcols = {}
    for si, site in enumerate(SITES):
        tg = targets(items, proj[:, si], pidx, pf)
        t = np.empty(len(R))
        for k in range(3):
            for tgt, arr in zip(("T", "I"), tg[k]):
                sel = ((R.split == k) & (R.target == tgt)).to_numpy()
                t[sel] = arr
        tcols[f"t_site{site}"] = t
        res = {key: np.full(len(R), np.nan, np.float32) for key in READ}
        t1 = time.time()
        with torch.no_grad():
            for (k, f), g in R.groupby(["split", "fold"]):
                d = bases[site][(k, f)]
                rows, base = g.row.to_numpy(), g.base.to_numpy()
                for i in range(0, len(g), args.batch):
                    sl = slice(i, i + args.batch)
                    src = torch.as_tensor(t[rows[sl]], dtype=torch.float32, device=runner.device)[:, None] * d[None]
                    m, parts = runner.patched([prompts[j] for j in base[sl]], src, site, d[:, None], full=True)
                    res["M"][rows[sl]] = m.cpu().numpy()
                    for key in READ[1:]:
                        res[key][rows[sl]] = parts[key].cpu().numpy()
        assert not np.isnan(res["M"]).any()
        # check: setting the coordinate to the item's own projection reproduces the unpatched readout
        pick = np.random.default_rng(0).choice(len(prompts), 256, replace=False)
        d = bases[site][(0, 0)]
        with torch.no_grad():
            own = torch.as_tensor(proj[pick, si, 0], device=runner.device)[:, None] * d[None]
            m, _ = runner.patched([prompts[j] for j in pick], own, site, d[:, None])
        meta["checks"][str(site)] = {"self_set_max_abs_dM": float(np.abs(m.cpu().numpy() - nat["M"][pick]).max())}
        pd.DataFrame({"row": R.row, **res}).to_parquet(out / f"patches_site{site}.parquet", index=False)
        print(f"site {site}: {len(R)} patches in {time.time() - t1:.0f}s; {meta['checks'][str(site)]}", flush=True)
    R.assign(**tcols).to_parquet(out / "rows.parquet", index=False)
    items.to_csv(out / "items.csv", index=False)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--passive-dir", default="data/das_round2/passive_test")
    ap.add_argument("--cons-dir", default="data/constructions")
    ap.add_argument("--folds", default="results/das_round2/final_strict/pairs_folds.csv")
    ap.add_argument("--bases", default="results/das_round2/bases_rank1.npz")
    ap.add_argument("--out-dir", default="results/round4/inrange")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=1024)
    run(ap.parse_args())
