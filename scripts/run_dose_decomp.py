#!/usr/bin/env python3
"""Decompose the site-8 dose-response by interval (round3_plan.md, A3). fp32.

The site-8 coordinate is set to each grid value z (split-0 basis, as in the
dose-response of `run_site8_mechanism.py`): h <- h + (c(z) - h.d) d. For each
consecutive pair of grid points (z_a, z_b) the change in centered logits is
decomposed exactly as in step 1, with z_a as the reference run and z_b as the
patched run (LN scale frozen at the z_b run):
  Delta x_24 = carry (x8_b - x8_a) + sum_{l=8..23} (sum_h Delta head_{l,h} + Delta mlp_l),
each term read through r_t / sigma_b; LN-scale term r_t . x_a (1/sigma_b - 1/sigma_a);
residual = actual - terms - LN term.

Frames: primary bad passives (every item) and, as a reference, intransitive
DAS actives (split-0 held-out basis). Readouts as in step 1 (" by", ".",
" the", " him", O-bar) plus the log-prob readouts of the passive test.

Output (`--out-dir`): `dose_decomp_site8.npz` with per frame x interval x pair
sums of every term and the pair counts; `dose_decomp_meta_site8.json`.
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
from run_site8_mechanism import DOWN, SITE, Mech, load_bases, scaling

GRID = (0.10, 0.275, 0.45, 0.675, 0.90, 1.20, 1.50)


def run(args):
    import torch

    args.dtype = "float32"
    t0 = time.time()
    runner = PassiveRunner(args)
    mech = Mech(runner, args)
    P = pd.read_csv(args.prompts)
    prompts = P.prompt.tolist()
    pid = dict(zip(P.prompt, P.pid))
    items = pd.read_csv(args.items)
    plan = pd.read_csv(args.plan, usecols=["item_id", "split", "fold"])
    pairs_folds = pd.read_csv(Path(args.site_dirs["8"]) / "pairs_folds.csv")
    das = pd.read_csv(Path(args.site_dirs["8"]) / "items.csv")
    das["pid"] = das.prompt.map(pid)
    bases = load_bases(Path(args.site_dirs["8"]) / "bases.pt", runner.device)

    # scaling of the split-0 bases from unpatched site-8 states of the DAS items (as in run_site8_mechanism)
    st = torch.empty(len(P), runner.model.config.hidden_size, device=runner.device)
    need = np.unique(das.pid.to_numpy())
    with torch.no_grad():
        for i in range(0, len(need), args.batch):
            ids = need[i:i + args.batch]
            enc, anchors = runner.encode([prompts[j] for j in ids])
            o = runner.model(**enc, output_hidden_states=True, use_cache=False)
            rows = torch.arange(len(anchors), device=runner.device)
            st[torch.as_tensor(ids, device=runner.device)] = o.hidden_states[SITE][rows, anchors].float()
    sc = scaling(st, das, pairs_folds, bases)

    fold0 = plan[plan.split == 0].drop_duplicates("item_id").set_index("item_id").fold
    it = items[items.bad_class == "plain"]
    fo0 = dict(zip(pairs_folds.pair_id, pairs_folds.fold_split0))
    ia = das[das.cls == 0]
    db = pd.concat([
        pd.DataFrame({"frame": "bad_passive", "base": it.bad_prompt.map(pid).to_numpy(), "pair": it.pair_id.to_numpy(),
                      "fold": it.item_id.map(fold0).to_numpy()}),
        pd.DataFrame({"frame": "intrans_active", "base": ia.pid.to_numpy(), "pair": ia.pair_id.to_numpy(),
                      "fold": ia.pair_id.map(fo0).to_numpy()})], ignore_index=True)
    pair_ix = {fr: {p: i for i, p in enumerate(sorted(g.pair.unique()))} for fr, g in db.groupby("frame")}
    db["pix"] = [pair_ix[f][p] for f, p in zip(db.frame, db.pair)]
    print(f"setup {time.time() - t0:.0f}s; {len(db)} bases", flush=True)

    acc = {}

    def add(key, n_pairs, pix, **arrs):
        a = acc.setdefault(key, {"n": torch.zeros(n_pairs, device=runner.device)})
        a["n"].index_add_(0, pix, torch.ones(len(pix), device=runner.device))
        for k, v in arrs.items():
            if k not in a:
                a[k] = torch.zeros((n_pairs,) + tuple(v.shape[1:]), device=runner.device)
            a[k].index_add_(0, pix, v)

    t0 = time.time()
    for (frame, f), g in db.groupby(["frame", "fold"]):
        d8 = bases[(0, f)]
        sign, mi, mt = sc[(0, f)]
        for i in range(0, len(g), args.batch):
            gg = g.iloc[i:i + args.batch]
            ps = [prompts[j] for j in gg.base]
            pix = torch.as_tensor(gg.pix.to_numpy(), device=runner.device)
            caps = []
            for zv in GRID:
                target = sign * (mi + zv * (mt - mi))
                caps.append(mech.run(ps, patch=lambda h: h + (target - h @ d8)[:, None] * d8[None]))
            s_ref = caps[0]["sigma"]  # fixed per-item scale (z = 0.10) for the second version
            for j in range(len(GRID) - 1):
                a, b = caps[j], caps[j + 1]
                sb = b["sigma"]
                dh, dm = b["heads"] - a["heads"], b["mlp"] - a["mlp"]
                dc = (b["x8"] - a["x8"]) @ mech.R.T
                heads = dh / sb[:, None, None, None]
                mlp = dm / sb[:, None, None]
                carry = dc / sb[:, None]
                ln = a["raw_final"] * (1 / sb - 1 / a["sigma"])[:, None]
                actual = b["zc"] - a["zc"]
                resid = actual - carry - heads.sum((1, 2)) - mlp.sum(1) - ln
                add((frame, j), len(pair_ix[frame]), pix, heads=heads, mlp=mlp, carry=carry, ln=ln, actual=actual,
                    resid=resid, dlp=b["lp"] - a["lp"], dlse=b["lse"] - a["lse"],
                    heads_fx=dh / s_ref[:, None, None, None], mlp_fx=dm / s_ref[:, None, None],
                    carry_fx=dc / s_ref[:, None])
            # absolute levels at each grid point (for curves and z checks)
            for j, c in enumerate(caps):
                zchk = (sign * (c["x8"] @ d8) - mi) / (mt - mi)
                add((frame, f"level{j}"), len(pair_ix[frame]), pix, zc=c["zc"], lp=c["lp"], lse=c["lse"],
                    z8=zchk[:, None])
    print(f"decomposition: {len(db)} bases x {len(GRID)} grid points in {time.time() - t0:.0f}s", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save = {"grid": np.array(GRID), "layers": np.arange(SITE, mech.L)}
    for fr, ix in pair_ix.items():
        save[f"{fr}_pairs"] = np.array(sorted(ix, key=ix.get))
    for (fr, j), a in acc.items():
        for k, v in a.items():
            save[f"{fr}_{j}_{k}"] = v.cpu().numpy()
    np.savez_compressed(out / "dose_decomp_site8.npz", **save)
    meta = {"args": vars(args), "grid": GRID, "scaling_split0": {str(f): sc[(0, f)] for f in range(5)},
            "bases_sha256": hashlib.sha256((Path(args.site_dirs["8"]) / "bases.pt").read_bytes()).hexdigest(),
            "n_bases": {fr: int((db.frame == fr).sum()) for fr in pair_ix}, "dtype": "float32",
            "ln_scale": "frozen at the upper (z_b) run of each interval", "down_sites_unused": list(DOWN)}
    (out / "dose_decomp_meta_site8.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", default="data/das_round2/passive_test/prompts.csv")
    ap.add_argument("--items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--plan", default="data/das_round2/passive_test/plan.csv.gz")
    ap.add_argument("--site-dirs", default="8=results/das_round2/final_strict_site8")
    ap.add_argument("--out-dir", default="results/das_round2/round3")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    a.site_dirs = dict(x.split("=") for x in a.site_dirs.split(","))
    run(a)
