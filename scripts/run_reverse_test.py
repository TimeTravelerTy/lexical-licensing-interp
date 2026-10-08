#!/usr/bin/env python3
"""Reverse DAS, test side: passive-trained directions patched into actives (round3_plan.md, B5).

Mirror of the passive test (`build_passive_test.py`, `run_passive_test.py`).

build (no model) -> `--data-dir`: `items.csv` (primary pair x subject; base
  prompts "<Subj> has <participle>"), `prompts.csv`, `plan.csv.gz`, `plan_meta.json`.
  Per item x split: the fold (the 8 orig_head pairs use the fold where they are
  held out; other items one random fold, seed 17); donors = the kept held-out
  passive-DAS pairs of that fold (excluding the item's own pair), each with its
  good (G) and bad (B) passive in one shared random curated context outside the
  15 passive-DAS contexts, and its transitive (T) and intransitive (I) active
  with one shared random subject (dose control); plus `same_verb` (the base verb's own passive, random
  such context) and `active_swap` (the other verb's active, same subject). Both
  the bad and the good active are bases.
run (GPU) -> `--out-dir`: `natural.parquet`, `projections_site{s}.parquet`
  (every prompt on the 15 passive-trained bases), `patches_site{s}.parquet`,
  `patches_da_site{s}.parquet` (the same rows patched with the active-trained
  basis of the same site, split and fold), `random_site{s}.npz` (100
  norm-matched random rank-1 draws on the split-0 G/B rows of bad bases:
  point-estimate D per draw), `run_meta.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from build_das_round2_prompts import SUBJECTS

READ = ("M", "O", "I", "det", "pron", "refl", "by", "dot", "the", "him")
AUX = dict(SUBJECTS)
SUBJ = [s for s, _ in SUBJECTS]


def build(args):
    out = Path(args.data_dir)
    out.mkdir(parents=True, exist_ok=True)
    pit = pd.read_csv(args.passive_items)
    pf = pd.read_csv(args.folds)
    rev = Path(args.reverse_dir)
    keep = set(pd.read_csv(rev / "pair_filter.csv").query("kept").pair_id)
    das_ctx = set(pd.read_csv(rev / "contexts.csv").context_id)
    pas = pd.read_json(args.passives, lines=True, dtype={"item_id": str})
    pas["prefix"] = [g[: len(g) - len(v)] for g, v in zip(pas.good_prompt, pas.good_verb)]
    ctx = pas.drop_duplicates("context_id").sort_values("context_id")
    ctx = ctx[~ctx.context_id.isin(das_ctx)][["context_id", "prefix"]].reset_index(drop=True)
    pr = pit[pit.bad_class == "plain"].drop_duplicates("pair_id")
    prompts = {}

    def add(text, kind, lemma, subject=""):
        prompts.setdefault(text, {"prompt": text, "kind": kind, "lemma": lemma, "subject": subject})
        return text

    items = []
    for p in pr.itertuples():
        for s in SUBJ:
            g = add(f"{s} {AUX[s]} {p.good_part}", "active", p.good_lemma, s)
            b = add(f"{s} {AUX[s]} {p.bad_part}", "active", p.bad_lemma, s)
            items.append({"item_id": f"{p.pair_id}|{s}", "pair_id": p.pair_id, "band": p.band, "source": p.source,
                          "das_pair": p.das_pair if isinstance(p.das_pair, str) else "", "context_id": s,
                          "context_band": "subject", "good_lemma": p.good_lemma, "bad_lemma": p.bad_lemma,
                          "good_part": p.good_part, "bad_part": p.bad_part, "good_prompt": g, "bad_prompt": b,
                          "bad_class": p.bad_class, "prep_subtype": p.prep_subtype, "bad_side_high": p.bad_side_high,
                          "by_split": p.by_split, "primary": True})
    items = pd.DataFrame(items)
    rng = np.random.default_rng(args.seed)
    n_folds = int(pf.fold_split0.max()) + 1
    verbs = {r.pair_id: (r.trans, r.trans_part, r.intrans, r.intrans_part) for r in pf.itertuples()}
    plan = []
    for it in items.itertuples():
        for k in range(3):
            fold_of = dict(zip(pf.pair_id, pf[f"fold_split{k}"]))
            f = fold_of[it.das_pair] if it.das_pair else int(rng.integers(n_folds))
            held = sorted(p for p, ff in fold_of.items() if ff == f and p != it.das_pair and p in keep)
            cpick = {p: ctx.iloc[int(rng.integers(len(ctx)))] for p in held}
            spick = {p: SUBJ[int(rng.integers(len(SUBJ)))] for p in held}
            own = ctx.iloc[int(rng.integers(len(ctx)))]
            for side, other in (("bad", "good"), ("good", "bad")):
                base = getattr(it, f"{side}_prompt")
                rows = []
                for p in held:
                    t, tp, i, ip = verbs[p]
                    rows.append(("G", add(cpick[p].prefix + tp, "passive", t), 1, p))
                    rows.append(("B", add(cpick[p].prefix + ip, "passive", i), 0, p))
                    sj = spick[p]
                    rows.append(("T", add(f"{sj} {AUX[sj]} {tp}", "active", t, sj), 1, p))
                    rows.append(("I", add(f"{sj} {AUX[sj]} {ip}", "active", i, sj), 0, p))
                rows.append(("same_verb", add(own.prefix + getattr(it, f"{side}_part"), "passive",
                                              getattr(it, f"{side}_lemma")), 1 if side == "good" else 0, ""))
                rows.append(("active_swap", getattr(it, f"{other}_prompt"), 1 if other == "good" else 0, ""))
                for cond, donor, dcls, dpair in rows:
                    plan.append({"item_id": it.item_id, "split": k, "fold": f, "side": side, "base": base,
                                 "cond": cond, "donor": donor, "donor_cls": dcls, "donor_pair": dpair})
    P = pd.DataFrame(prompts.values())
    P.insert(0, "pid", np.arange(len(P)))
    idx = dict(zip(P.prompt, P.pid))
    plan = pd.DataFrame(plan)
    plan["base"], plan["donor"] = plan.base.map(idx), plan.donor.map(idx)
    plan.insert(0, "row", np.arange(len(plan)))
    items.to_csv(out / "items.csv", index=False)
    P.to_csv(out / "prompts.csv", index=False)
    plan.to_csv(out / "plan.csv.gz", index=False)
    meta = {"items": len(items), "pairs": int(items.pair_id.nunique()), "prompts": len(P), "plan_rows": len(plan),
            "plan_rows_by_cond": plan.cond.value_counts().to_dict(), "kept_das_pairs": len(keep),
            "donor_contexts": len(ctx), "seed": args.seed,
            "plan_sha256": hashlib.sha256(pd.util.hash_pandas_object(plan, index=False).values.tobytes()).hexdigest()}
    (out / "plan_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


def run(args):
    import torch

    from run_passive_test import READ as PREAD, PassiveRunner

    assert tuple(PREAD) == READ[1:]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runner = PassiveRunner(args)
    P = pd.read_csv(Path(args.data_dir) / "prompts.csv")
    plan = pd.read_csv(Path(args.data_dir) / "plan.csv.gz")
    items = pd.read_csv(Path(args.data_dir) / "items.csv")
    prompts = P.prompt.tolist()
    sites = [int(s) for s in args.sites.split(",")]
    states = torch.empty(len(P), len(sites), runner.model.config.hidden_size, device=runner.device)
    nat = {k: np.empty(len(P), np.float32) for k in READ}
    with torch.no_grad():
        for i in range(0, len(P), args.batch):
            enc, anchors = runner.encode(prompts[i:i + args.batch])
            o = runner.model(**enc, output_hidden_states=True, use_cache=False)
            m, parts = runner.readout(o.logits, anchors, full=True)
            rows = torch.arange(len(anchors), device=runner.device)
            states[i:i + len(m)] = torch.stack([o.hidden_states[s][rows, anchors].float() for s in sites], 1)
            nat["M"][i:i + len(m)] = m.cpu().numpy()
            for k in READ[1:]:
                nat[k][i:i + len(m)] = parts[k].cpu().numpy()
    pd.DataFrame({"pid": P.pid, **nat}).to_parquet(out / "natural.parquet", index=False)
    meta = {"args": vars(args), "prompts": len(P), "plan_rows": len(plan), "bases_sha256": {}, "checks": {}}
    groups = {k: g for k, g in plan.groupby(["split", "fold"])}
    # point-estimate D (G - B on bad bases), pair means of item means, for the random null
    r0 = plan[(plan.split == 0) & (plan.side == "bad") & plan.cond.isin(["G", "B"])]
    pair_of = dict(zip(items.item_id, items.pair_id))
    natb = pd.DataFrame(nat).loc[r0.base.to_numpy()].to_numpy()

    def point_d(vals):
        d = pd.DataFrame(vals - natb, columns=READ)
        d["item"], d["cond"] = r0.item_id.to_numpy(), r0.cond.to_numpy()
        m = d.groupby(["item", "cond"])[list(READ)].mean()
        dd = (m.xs("G", level="cond") - m.xs("B", level="cond"))
        dd["pair"] = dd.index.map(pair_of)
        return dd.groupby("pair")[list(READ)].mean().mean().to_numpy()

    for si, site in enumerate(sites):
        d = Path(args.reverse_dir) / f"final_site{site}"
        b = torch.load(d / "bases.pt")
        bases = {(int(k.split("_s")[1].split("_")[0]), int(k.split("_f")[1])): v.float().to(runner.device)
                 for k, v in b.items() if k.startswith("r1_")}
        meta["bases_sha256"][str(site)] = hashlib.sha256((d / "bases.pt").read_bytes()).hexdigest()
        cols = {f"s{k[0]}_f{k[1]}": (states[:, si] @ v[:, 0]).cpu().numpy() for k, v in sorted(bases.items())}
        pd.DataFrame({"pid": P.pid, **cols}).to_parquet(out / f"projections_site{site}.parquet", index=False)
        t0 = time.time()
        res = {k: np.full(len(plan), np.nan, np.float32) for k in READ}
        with torch.no_grad():
            for key, g in groups.items():
                rows, base, donor = g.row.to_numpy(), g.base.to_numpy(), g.donor.to_numpy()
                for i in range(0, len(g), args.batch):
                    sl = slice(i, i + args.batch)
                    m, parts = runner.patched([prompts[j] for j in base[sl]],
                                              states[torch.as_tensor(donor[sl], device=runner.device), si],
                                              site, bases[key], full=True)
                    res["M"][rows[sl]] = m.cpu().numpy()
                    for k in READ[1:]:
                        res[k][rows[sl]] = parts[k].cpu().numpy()
        if np.isnan(res["M"]).any():
            raise SystemExit(f"site {site}: unfilled plan rows")
        pd.DataFrame({"row": plan.row, **res}).to_parquet(out / f"patches_site{site}.parquet", index=False)
        # arm 2: the active-trained basis of the same site, split and fold, on the same rows
        act = np.load(args.active_bases)
        abases = {k: torch.tensor(act[f"das_site{site}_s{k[0]}_f{k[1]}"].reshape(-1, 1), dtype=torch.float32,
                                  device=runner.device) for k in bases}
        resa = {k: np.full(len(plan), np.nan, np.float32) for k in READ}
        with torch.no_grad():
            for key, g in groups.items():
                rows, base, donor = g.row.to_numpy(), g.base.to_numpy(), g.donor.to_numpy()
                for i in range(0, len(g), args.batch):
                    sl = slice(i, i + args.batch)
                    m, parts = runner.patched([prompts[j] for j in base[sl]],
                                              states[torch.as_tensor(donor[sl], device=runner.device), si],
                                              site, abases[key], full=True)
                    resa["M"][rows[sl]] = m.cpu().numpy()
                    for k in READ[1:]:
                        resa[k][rows[sl]] = parts[k].cpu().numpy()
        pd.DataFrame({"row": plan.row, **resa}).to_parquet(out / f"patches_da_site{site}.parquet", index=False)
        das_d = point_d(np.stack([res[k][r0.row.to_numpy()] for k in READ], 1))
        # self-patch check
        with torch.no_grad():
            pick = np.random.default_rng(0).choice(len(P), 256, replace=False)
            m, _ = runner.patched([prompts[j] for j in pick], states[torch.as_tensor(pick, device=runner.device), si],
                                  site, bases[(0, 0)])
        meta["checks"][str(site)] = {"self_patch_max_abs_dM": float(np.abs(m.cpu().numpy() - nat["M"][pick]).max())}
        # random null, norm-matched to the DAS displacement, split-0 G/B rows on bad bases
        gen = torch.Generator(device="cpu").manual_seed(7)
        draws = []
        with torch.no_grad():
            for r in range(args.n_random):
                vals = np.empty((len(r0), len(READ)), np.float32)
                for f, g in r0.assign(pos=np.arange(len(r0))).groupby("fold"):
                    rnd = torch.linalg.qr(torch.randn(bases[(0, f)].shape, generator=gen))[0].to(runner.device)
                    for i in range(0, len(g), args.batch):
                        gg = g.iloc[i:i + args.batch]
                        m, parts = runner.patched([prompts[j] for j in gg.base],
                                                  states[torch.as_tensor(gg.donor.to_numpy(), device=runner.device), si],
                                                  site, rnd, full=True, match_norm=bases[(0, f)])
                        vals[gg.pos.to_numpy()] = np.stack([m.cpu().numpy()] + [parts[k].cpu().numpy()
                                                                                for k in READ[1:]], 1)
                draws.append(point_d(vals))
        np.savez_compressed(out / f"random_site{site}.npz", readouts=np.array(READ), das=das_d, draws=np.array(draws))
        print(f"site {site}: {len(plan)} patches + {args.n_random} random draws in {time.time() - t0:.0f}s", flush=True)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("build", "run"))
    ap.add_argument("--passive-items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--passives", default="data/passive_das/passives.jsonl")
    ap.add_argument("--folds", default="results/das_round2/final_strict/pairs_folds.csv")
    ap.add_argument("--reverse-dir", default="results/das_round2/reverse")
    ap.add_argument("--data-dir", default="data/das_round2/reverse_test")
    ap.add_argument("--out-dir", default="results/das_round2/reverse_test")
    ap.add_argument("--sites", default="4,6,8,10,12,14,16,17")
    ap.add_argument("--active-bases", default="results/das_round2/bases_rank1.npz")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--n-random", type=int, default=100)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    build(a) if a.mode == "build" else run(a)
