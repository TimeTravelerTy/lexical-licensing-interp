#!/usr/bin/env python3
"""Gemma replication: checks of the ported DAS runner (run_das_round2.Runner) before any Gemma DAS training.

1. --reproduce (Pythia, default flags): natural M of every training item vs the stored
   results/das_round2/final_strict_site8/items.csv, and the stored site-8 rank-1 basis (split 0, fold 0) evaluated on
   its held-out swaps vs the stored heldout_swaps.csv.gz (M_patched).
2. Self-patch at every site: patching each item with its own natural state (any rank-1 basis) must leave M unchanged
   if the hooked layer output equals output_hidden_states at that site. Reported per site (max |dM|).
3. Batched vs one-at-a-time: natural M and patched M (one site, a random basis) on --n-check items.
4. Behaviour filter (the runner's own rule: a verb is kept if >= 4 of its 7 items have M on its class's side):
   per-verb correct counts, dropped pairs, M_nat by class, and the model's token counts of the participles.

Writes check.json and items_natural.csv to --out-dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_das_round2 import Runner, eval_swaps, fold_indices, folds, load_items


def main(args):
    import torch

    runner = Runner(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs, items = load_items(args.pairs)
    prompts = items.prompt.tolist()
    m, parts, reps = runner.natural(prompts)
    items["M_nat"] = m.cpu().numpy()
    items["correct"] = np.where(items.cls == 1, items.M_nat > 0, items.M_nat < 0)
    items["n_tok_model"] = [len(runner.tok.encode(" " + p.split(" ", 2)[2], add_special_tokens=False)) for p in prompts]
    res = {"model": args.model, "dtype": args.dtype, **runner.settings, "n_sites": runner.n_sites, "n_items": len(items)}

    # 4. behaviour filter
    vc = items.groupby("verb").correct.sum()
    keep = pairs[pairs.trans.map(vc >= 4) & pairs.intrans.map(vc >= 4)].pair_id
    res["dropped_pairs"] = sorted(set(pairs.pair_id) - set(keep))
    res["verbs_below_4_of_7"] = {v: int(n) for v, n in vc[vc < 4].items()}
    res["item_accuracy_by_class"] = items.groupby("cls").correct.mean().round(3).to_dict()
    res["M_nat_mean_by_class"] = items.groupby("cls").M_nat.mean().round(3).to_dict()
    tokpat = items.groupby("pair_id").apply(lambda g: f"T{g[g.cls == 1].n_tok_model.iloc[0]}-I{g[g.cls == 0].n_tok_model.iloc[0]}")
    res["model_token_pattern_counts"] = tokpat.value_counts().to_dict()
    items.drop(columns="prompt").assign(prompt=items.prompt, tok_pattern=items.pair_id.map(tokpat)).to_csv(
        out / "items_natural.csv", index=False)

    # 3 (first half). natural pass on a subset in one batch; its states are the self-patch sources
    gen = torch.Generator(device="cpu").manual_seed(0)
    d = reps.shape[-1]
    basis = torch.linalg.qr(torch.randn(d, 1, generator=gen))[0].to(runner.device)
    idx = list(range(0, len(items), max(1, len(items) // args.n_check)))[:args.n_check]
    sub = [prompts[i] for i in idx]
    mb, _, rb = runner.natural(sub, batch=256)

    # 2. self-patch at every site (same batch as the reference, so a zero displacement must change nothing)
    self_patch = {}
    with torch.no_grad():
        for site in range(runner.n_sites):
            mp, _ = runner.patched(sub, rb[:, site], site, basis)
            self_patch[site] = float((mp - mb).abs().max())
    res["self_patch_max_abs_dM_by_site"] = self_patch

    # 3. batched vs one-at-a-time
    m1 = torch.cat([runner.natural([p], batch=1)[0] for p in sub])
    res["natural_batched_vs_single_max_abs"] = float((mb - m1).abs().max())
    site = args.check_site
    src = rb[list(reversed(range(len(sub)))), site]
    with torch.no_grad():
        pb, _ = runner.patched(sub, src, site, basis)
        p1 = torch.cat([runner.patched([p], src[k:k + 1], site, basis)[0] for k, p in enumerate(sub)])
    res["patched_batched_vs_single_max_abs"] = {"site": site, "max_abs": float((pb - p1).abs().max())}

    # 1. Pythia reproduction of stored outputs
    if args.reproduce:
        st = pd.read_csv(Path(args.stored) / "items.csv", keep_default_na=False)
        mm = items.merge(st[["prompt", "M_nat"]], on="prompt", suffixes=("", "_stored"))
        assert len(mm) == len(items) == len(st)
        res["reproduce_M_nat_max_abs"] = float((mm.M_nat - mm.M_nat_stored).abs().max())
        bases = torch.load(Path(args.stored) / "bases.pt")
        meta = json.loads((Path(args.stored) / "final_meta.json").read_text())
        s8 = meta["site"]
        # the stored items are those kept by the filter (none dropped at 1.4B); same order as load_items
        assert (st.prompt.to_numpy() == items.prompt.to_numpy()).all()
        assign = folds(pairs, 5, 0)
        _, te = fold_indices(items, assign, 0)
        sw = eval_swaps(items, te)
        b = bases["r1_s0_f0"].to(runner.device)
        rows = []
        with torch.no_grad():
            for i in range(0, len(sw), 512):
                ch = sw[i:i + 512]
                mp, _ = runner.patched([prompts[x[0]] for x in ch], reps[[x[1] for x in ch], s8], s8, b)
                rows += [(x[0], x[1], float(v)) for x, v in zip(ch, mp.cpu().numpy())]
        new = pd.DataFrame(rows, columns=["base", "src", "M_new"])
        sd = pd.read_csv(Path(args.stored) / "heldout_swaps.csv.gz")
        sd = sd[(sd["rank"] == 1) & (sd.split == 0) & (sd.fold == 0) & (sd.control == "das")]
        j = new.merge(sd[["base", "src", "M_patched"]], on=["base", "src"])
        assert len(j) == len(new) == len(sd), (len(j), len(new), len(sd))
        res["reproduce_patched_site8_s0f0"] = {"n_swaps": len(j), "max_abs": float((j.M_new - j.M_patched).abs().max())}
    (out / "check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="data/das_round2/train_pairs.csv")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--bos", action=argparse.BooleanOptionalAction, default=None, help="default: model_arch.ARCH")
    ap.add_argument("--fp32-unembed", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--reproduce", action="store_true")
    ap.add_argument("--stored", default="results/das_round2/final_strict_site8")
    ap.add_argument("--n-check", type=int, default=20)
    ap.add_argument("--check-site", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    main(ap.parse_args())
