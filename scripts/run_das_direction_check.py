#!/usr/bin/env python3
"""DAS direction check (Pythia-1.4B; rules: reports/pythia/das_direction_check_plan.md).

Per site (4, 6, 8, 12, 17) and final-run basis (3 splits x 5 folds, rank 1):
1. directions: v (DAS), u_md (class mean difference of the basis's training items, natural states), saved LEACE
   eraser vectors; |cos| and references (random isotropic, shuffled-label mean difference); projection variance.
2. component patching along v and along u_md: held-out active swaps (run_das_round2.evaluate) and the stored
   passive-test plan rows (primary population, bad bases, T / I donors, all splits).
3. mean ablation (h <- h + (mu - h.u) u, mu = grand mean of h.u over the frame's items) of v, u_md and random
   directions in natural runs: active items (cross-fitted bases) and primary passive prompts (plan's basis per
   item x split).

Outputs in --out-dir: cosines.csv, directions_site{s}.npz, active_summary.csv, active_swaps.csv.gz,
ablation_active.csv.gz, passive_natural.parquet, passive_patches_site{s}.parquet, ablation_passive_site{s}.parquet,
checks.json, meta.json. Run from scripts/ (paths default to ../).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import run_das_round2 as r2
from run_passive_controls import upgrade
from run_passive_test import READ

SITE_DIRS = {4: "final_strict_site4", 6: "final_strict_site6", 8: "final_strict_site8",
             12: "final_strict_site12", 17: "final_strict"}
FOLD_COLS = ["pair_id", "fold_split0", "fold_split1", "fold_split2"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bases(root, site, ref_folds, torch, dev):
    d = Path(root) / SITE_DIRS[site]
    pf = pd.read_csv(d / "pairs_folds.csv")[FOLD_COLS].sort_values("pair_id").reset_index(drop=True)
    if not pf.equals(ref_folds):
        raise SystemExit(f"site {site}: fold assignment differs from the frozen one")
    b = torch.load(d / "bases.pt")
    out = {}
    for k, v in b.items():
        if k.startswith("r1_"):
            split, fold = int(k.split("_s")[1].split("_")[0]), int(k.split("_f")[1])
            out[(split, fold)] = v.float().to(dev)
    assert sorted(out) == [(s, f) for s in range(3) for f in range(5)], sorted(out)
    return out, sha(d / "bases.pt")


def unit(x):
    return x / np.linalg.norm(x)


def run(args):
    import torch

    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sites = [int(s) for s in args.sites.split(",")]
    root = Path(args.root)

    runner, pairs, items, reps, nat, dropped = r2.setup(args)
    runner = upgrade(runner)
    dev, d = runner.device, reps.shape[-1]
    stored_items = pd.read_csv(root / "final_strict" / "items.csv")
    assert stored_items.prompt.tolist() == items.prompt.tolist(), "active items differ from the final runs'"
    labels = items.cls.to_numpy()
    prompts_a = items.prompt.tolist()
    N = len(items)
    ref_folds = pd.read_csv(root / "final_strict" / "pairs_folds.csv")[FOLD_COLS] \
        .sort_values("pair_id").reset_index(drop=True)
    assigns = {k: r2.folds(pairs, args.n_folds, k) for k in range(3)}
    for k in range(3):  # the frozen fold assignment is the one r2.folds gives
        assert all(ref_folds.set_index("pair_id")[f"fold_split{k}"].to_dict()[p] == f for p, f in assigns[k].items())
    leace_in = np.load(args.leace_insample)
    leace_cf = np.load(args.leace_crossfit)
    checks = {"items": N, "pairs": len(pairs), "dropped_pairs": dropped, "sites": {}}
    meta = {"args": vars(args), "runner_settings": runner.settings, "bases_sha256": {},
            "leace_insample_sha256": sha(args.leace_insample), "leace_crossfit_sha256": sha(args.leace_crossfit)}
    log(f"{N} active items, {len(pairs)} pairs")

    # ---- passive plan subset (primary population) and states
    do_passive = not args.skip_passive
    if do_passive:
        it = pd.read_csv(args.items)
        prim = it[it.bad_class == "plain"]
        if args.max_items:
            prim = prim.groupby("pair_id").head(max(1, args.max_items // prim.pair_id.nunique()))
        prim_ids = set(prim.item_id)
        plan = pd.read_csv(args.plan)
        meta["plan_rows_total"] = len(plan)
        plan = plan[plan.item_id.isin(prim_ids)]
        pat_plan = plan[(plan.side == "bad") & plan.cond.isin(["T", "I"])].reset_index(drop=True)
        abl_plan = plan.drop_duplicates(["item_id", "split", "side"])[["item_id", "split", "fold", "side", "base"]] \
            .reset_index(drop=True)
        assert len(abl_plan) == 2 * plan.drop_duplicates(["item_id", "split"]).shape[0]
        P = pd.read_csv(args.prompts)
        assert (P.pid.to_numpy() == np.arange(len(P))).all()
        prompts_p = P.prompt.tolist()
        need = np.unique(np.concatenate([pat_plan.base, pat_plan.donor, abl_plan.base]))
        abl_pids = np.unique(abl_plan.base.to_numpy())
        meta.update({"passive_patch_rows": len(pat_plan), "passive_ablation_rows": len(abl_plan),
                     "passive_items": len(prim_ids), "passive_prompts_needed": len(need)})
        log(f"passive: {len(pat_plan)} patch rows, {len(abl_plan)} ablation rows, {len(need)} prompts")
        # natural states at every requested site, natural readouts
        P_states = torch.zeros(len(P), len(sites), d, device=dev)
        pnat = {k: np.full(len(P), np.nan, np.float32) for k in ("M",) + READ}
        with torch.no_grad():
            for i in range(0, len(need), args.batch):
                ids = need[i:i + args.batch]
                enc, anchors = runner.encode([prompts_p[j] for j in ids])
                o = runner.model(**enc, output_hidden_states=True, use_cache=False)
                m, parts = runner.readout(o.logits, anchors, full=True)
                rows = torch.arange(len(anchors), device=dev)
                ix = torch.as_tensor(ids, device=dev)
                P_states[ix] = torch.stack([o.hidden_states[s][rows, anchors].float() for s in sites], 1)
                pnat["M"][ids] = m.cpu().numpy()
                for k in READ:
                    pnat[k][ids] = parts[k].cpu().numpy()
        pd.DataFrame({"pid": need, **{k: v[need] for k, v in pnat.items()}}) \
            .to_parquet(out / "passive_natural.parquet", index=False)
        log("passive natural pass done")

    def patch(prompts, src, site, basis):
        m, parts = runner.patched(prompts, src, site, basis, full=True)
        return {"M": m.cpu().numpy(), **{k: parts[k].cpu().numpy() for k in READ}}

    cos_rows, act_summ, act_swaps, act_abl = [], [], [], []
    for si, site in enumerate(sites):
        ts = time.time()
        B, meta["bases_sha256"][str(site)] = load_bases(root, site, ref_folds, torch, dev)
        X = reps[:, site]  # [N, d] fp32, natural
        Xn = X.double().cpu().numpy()
        tot_var = Xn.var(0).sum() / d
        le = {p: unit(leace_in[f"single_{p}"][0, site]) for p in ("edit", "read")}
        lcf = {p: np.stack([unit(leace_cf[f"single_{p}"][f, site]) for f in range(leace_cf[f"single_{p}"].shape[0])])
               for p in ("edit", "read")}
        rng0 = np.random.default_rng(0)
        R = rng0.standard_normal((args.n_rand_cos, d))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        keys = sorted(B)
        U = {}
        V_np, U_np, shuf_np, rand_np = [], [], [], []
        sc = {"unit_norm_max_dev": 0.0, "u_md_self_patch_max_abs_dM": 0.0}
        for (k, f) in keys:
            v = B[(k, f)][:, 0].double().cpu().numpy()
            sc["unit_norm_max_dev"] = max(sc["unit_norm_max_dev"], abs(np.linalg.norm(v) - 1))
            tr, te = r2.fold_indices(items, assigns[k], f)
            lab = labels[tr]
            md = unit(Xn[tr][lab == 1].mean(0) - Xn[tr][lab == 0].mean(0))
            U[(k, f)] = torch.as_tensor(md, dtype=torch.float32, device=dev)[:, None]
            # shuffled-label mean differences (verb-level permutations of the training verbs' classes)
            vcls = items.iloc[tr].groupby("verb").cls.first()
            tv = sorted(vcls.index)
            verbs_tr = items.verb.values[tr]
            sh = []
            for p in range(args.n_perm):
                perm = dict(zip(tv, np.random.default_rng(1000 * (5 * k + f) + p).permutation(vcls[tv].to_numpy())))
                lp = np.array([perm[x] for x in verbs_tr])
                sh.append(abs(unit(Xn[tr][lp == 1].mean(0) - Xn[tr][lp == 0].mean(0)) @ v))
            rc = np.abs(R @ v)
            pv, pu = Xn @ v, Xn @ md
            cos_rows.append({
                "site": site, "split": k, "fold": f, "cos_v_umd": abs(v @ md),
                "cos_v_leace_edit": abs(v @ le["edit"]), "cos_v_leace_read": abs(v @ le["read"]),
                "cos_umd_leace_edit": abs(md @ le["edit"]), "cos_umd_leace_read": abs(md @ le["read"]),
                "cos_v_leace_edit_crossfit": float(np.abs(lcf["edit"] @ v).mean()),
                "cos_v_leace_read_crossfit": float(np.abs(lcf["read"] @ v).mean()),
                "cos_leace_edit_read": abs(le["edit"] @ le["read"]),
                "rand_mean": rc.mean(), "rand_p95": np.percentile(rc, 95),
                "shuf_mean": float(np.mean(sh)), "shuf_p95": float(np.percentile(sh, 95)),
                "var_ratio_v": pv.var() / tot_var, "var_ratio_umd": pu.var() / tot_var,
                "md_norm": float(np.linalg.norm(Xn[tr][lab == 1].mean(0) - Xn[tr][lab == 0].mean(0))),
                "n_train": len(tr), "n_test": len(te)})
            V_np.append(v)
            U_np.append(md)
            shuf_np.append(sh)
            rand_np.append(rc.astype(np.float32))

            # ---- 2. active component patching (held-out swaps of this basis)
            with torch.no_grad():
                pick = te[:64]
                m, _ = runner.patched([prompts_a[j] for j in pick], X[pick], site, U[(k, f)])
                sc["u_md_self_patch_max_abs_dM"] = max(sc["u_md_self_patch_max_abs_dM"],
                                                       float(np.abs(m.cpu().numpy() - nat["M"][pick]).max()))
            for name, basis in (("v", B[(k, f)]), ("u_md", U[(k, f)])):
                df = r2.evaluate(runner, items, reps, nat, te, site, basis, detail=True)
                df["rank"], df["split"], df["fold"], df["direction"], df["site"] = 1, k, f, name, site
                act_swaps.append(df)
                act_summ.append({"site": site, "direction": name, "split": k, "fold": f,
                                 **r2.summarize(df, items)})
        np.savez_compressed(out / f"directions_site{site}.npz", keys=np.array(keys), v=np.stack(V_np),
                            u_md=np.stack(U_np), shuf_cos=np.array(shuf_np), rand_cos=np.stack(rand_np))
        log(f"site {site}: directions and active patching done")

        # ---- 3a. active mean ablation (cross-fitted: each item under the 3 bases holding its pair out)
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        RND = {(k, f): [torch.nn.functional.normalize(torch.randn(d, generator=gen), dim=0).to(dev)[:, None]
                        for _ in range(args.n_rand_abl)] for (k, f) in keys}
        dirs = {"v": B, "u_md": U, **{f"rand{j}": {kk: RND[kk][j] for kk in keys} for j in range(args.n_rand_abl)}}
        mu_a = {}
        with torch.no_grad():
            for name, D in dirs.items():
                for (k, f) in keys:
                    u = D[(k, f)]
                    mu = float((X @ u[:, 0]).mean())
                    mu_a[(name, k, f)] = mu
                    _, te = r2.fold_indices(items, assigns[k], f)
                    src = (mu * u[:, 0])[None].expand(len(te), d)
                    res = patch([prompts_a[j] for j in te], src, site, u)
                    act_abl.append(pd.DataFrame({"site": site, "direction": name, "split": k, "fold": f, "item": te,
                                                 "mu": mu, **res}))
            # identity ablation: mu replaced by each item's own projection
            u = B[keys[0]]
            _, te = r2.fold_indices(items, assigns[keys[0][0]], keys[0][1])
            src = (X[te] @ u) * u[:, 0][None]
            res = patch([prompts_a[j] for j in te], src, site, u)
            sc["identity_ablation_active_max_abs_dM"] = float(np.abs(res["M"] - nat["M"][te]).max())
        log(f"site {site}: active ablation done")

        # ---- 2b / 3b. passives
        if do_passive:
            S = P_states[:, si]
            pp = []
            with torch.no_grad():
                for name, D in (("v", B), ("u_md", U)):
                    res = {k: np.empty(len(pat_plan), np.float32) for k in ("M",) + READ}
                    for key, g in pat_plan.groupby(["split", "fold"]):
                        ix = g.index.to_numpy()
                        base, donor = g.base.to_numpy(), g.donor.to_numpy()
                        for i in range(0, len(ix), args.batch):
                            sl = slice(i, i + args.batch)
                            r = patch([prompts_p[j] for j in base[sl]], S[torch.as_tensor(donor[sl], device=dev)],
                                      site, D[key])
                            for kk in r:
                                res[kk][ix[sl]] = r[kk]
                    pp.append(pd.DataFrame({"row": pat_plan.row, "direction": name, **res}))
            pd.concat(pp, ignore_index=True).to_parquet(out / f"passive_patches_site{site}.parquet", index=False)
            log(f"site {site}: passive patching done ({len(pat_plan)} rows x 2)")
            pa = []
            Sp = S[torch.as_tensor(abl_pids, device=dev)]
            with torch.no_grad():
                for name, D in dirs.items():
                    res = {k: np.empty(len(abl_plan), np.float32) for k in ("M",) + READ}
                    mus = np.empty(len(abl_plan))
                    for key, g in abl_plan.groupby(["split", "fold"]):
                        u = D[key]
                        mu = float((Sp @ u[:, 0]).mean())
                        ix = g.index.to_numpy()
                        base = g.base.to_numpy()
                        mus[ix] = mu
                        for i in range(0, len(ix), args.batch):
                            sl = slice(i, i + args.batch)
                            src = (mu * u[:, 0])[None].expand(len(base[sl]), d)
                            r = patch([prompts_p[j] for j in base[sl]], src, site, u)
                            for kk in r:
                                res[kk][ix[sl]] = r[kk]
                    pa.append(pd.DataFrame({"abl_row": abl_plan.index, "item_id": abl_plan.item_id,
                                            "split": abl_plan.split, "side": abl_plan.side, "base": abl_plan.base,
                                            "direction": name, "mu": mus, **res}))
                # identity ablation on the first 512 rows
                g = abl_plan.iloc[:512]
                u = B[(int(g.split.iloc[0]), int(g.fold.iloc[0]))]
                g = g[(g.split == g.split.iloc[0]) & (g.fold == g.fold.iloc[0])]
                hb = S[torch.as_tensor(g.base.to_numpy(), device=dev)]
                r = patch([prompts_p[j] for j in g.base], (hb @ u) * u[:, 0][None], site, u)
                sc["identity_ablation_passive_max_abs_dM"] = float(np.abs(r["M"] - pnat["M"][g.base]).max())
            pd.concat(pa, ignore_index=True).to_parquet(out / f"ablation_passive_site{site}.parquet", index=False)
            log(f"site {site}: passive ablation done ({len(abl_plan)} rows x {len(dirs)})")
        checks["sites"][str(site)] = sc
        log(f"site {site} checks {sc} ({time.time() - ts:.0f}s)")
        pd.DataFrame(cos_rows).to_csv(out / "cosines.csv", index=False)
        pd.DataFrame(act_summ).to_csv(out / "active_summary.csv", index=False)

    det = pd.concat(act_swaps, ignore_index=True)
    for col in ("verb", "pair_id", "subject", "source", "tok"):
        det[f"base_{col}"] = items[col].values[det.base]
        det[f"src_{col}"] = items[col].values[det.src]
    for k in ("O", "I", "det", "pron", "refl"):
        det[f"lp_{k}_base"] = nat[k][det.base]
        det[f"lp_{k}_src"] = nat[k][det.src]
    det.to_csv(out / "active_swaps.csv.gz", index=False)
    ab = pd.concat(act_abl, ignore_index=True)
    for col in ("verb", "cls", "pair_id", "subject"):
        ab[col] = items[col].values[ab["item"]]
    ab.to_csv(out / "ablation_active.csv.gz", index=False)
    an = items[["verb", "cls", "pair_id", "subject", "prompt"]].copy()
    an["M"] = nat["M"]
    for k in ("O", "I", "det", "pron", "refl"):
        an[k] = nat[k]
    an.to_csv(out / "active_natural.csv", index=False)
    (out / "checks.json").write_text(json.dumps(checks, indent=2, default=str) + "\n")
    meta["minutes"] = (time.time() - t0) / 60
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    log(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", default="4,6,8,12,17")
    ap.add_argument("--root", default="../results/das_round2")
    ap.add_argument("--pairs", default="../data/das_round2/train_pairs.csv")
    ap.add_argument("--items", default="../data/das_round2/passive_test/items.csv")
    ap.add_argument("--prompts", default="../data/das_round2/passive_test/prompts.csv")
    ap.add_argument("--plan", default="../data/das_round2/passive_test/plan.csv.gz")
    ap.add_argument("--leace-insample", default="../results/pythia/leace_insample/erasers.npz")
    ap.add_argument("--leace-crossfit", default="../results/pythia/leace_necessity/erasers.npz")
    ap.add_argument("--out-dir", default="../results/pythia/das_direction_check")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=100)
    ap.add_argument("--n-rand-cos", type=int, default=10000)
    ap.add_argument("--n-rand-abl", type=int, default=10)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--max-items", type=int, default=0, help="smoke test: about this many primary passive items")
    ap.add_argument("--skip-passive", action="store_true")
    ap.add_argument("--bos", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--fp32-unembed", action=argparse.BooleanOptionalAction, default=None)
    run(ap.parse_args())
