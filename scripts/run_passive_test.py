#!/usr/bin/env python3
"""Run the passive test of the round-2 direction (spec: reports/passive_das_prep/passive_test_plan.md).

Inputs: `build_passive_test.py` outputs (`prompts.csv`, `plan.csv.gz`) and the
rank-1 fold bases of each site (`bases.pt`, keys r1_s{split}_f{fold}).

1. Unpatched pass over every prompt: readouts and last-token states at the
   requested sites -> `natural.parquet`.
2. Step 1 (projection): each prompt's state projected onto each of the 15
   fold bases, raw (no sign alignment or scaling; done in the analysis)
   -> `projections_site{s}.parquet`.
3. Step 2 (transfer): every plan row patched at site s with the same
   interchange as training (`Runner.patched`): h <- h + ((h_src - h) d) d,
   h_src = the donor's state at that site -> `patches_site{s}.parquet`.

Readouts at the last token (log-probs): O / I / det / pron / refl (the
training sets), " by", ".", " the", " him", and M = log P(O) - log P(I).
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

SINGLE = {"by": " by", "dot": ".", "the": " the", "him": " him"}
READ = ("O", "I", "det", "pron", "refl", "by", "dot", "the", "him")


class PassiveRunner(Runner):
    def __init__(self, args):
        super().__init__(args)
        for k, t in SINGLE.items():
            enc = self.tok.encode(t, add_special_tokens=False)
            if len(enc) != 1:
                raise SystemExit(f"token {t!r} is not a single token")
            self.ids[k] = self.torch.tensor(enc, device=self.device)

    def readout(self, logits, anchors, full=False):
        torch = self.torch
        rows = torch.arange(logits.shape[0], device=logits.device)
        z = logits[rows, anchors].float()
        m = torch.logsumexp(z[:, self.ids["O"]], -1) - torch.logsumexp(z[:, self.ids["I"]], -1)
        if not full:
            return m, None
        lp = torch.log_softmax(z, -1)
        return m, {k: torch.logsumexp(lp[:, self.ids[k]], -1) for k in READ}


def site_bases(args, site):
    import torch

    d = Path(args.site_dirs[str(site)])
    folds = pd.read_csv(d / "pairs_folds.csv")
    ref = pd.read_csv(args.folds)
    cols = ["pair_id", "fold_split0", "fold_split1", "fold_split2"]
    if not folds[cols].sort_values("pair_id").reset_index(drop=True).equals(
            ref[cols].sort_values("pair_id").reset_index(drop=True)):
        raise SystemExit(f"site {site}: fold assignment differs from the plan's")
    b = torch.load(d / "bases.pt")
    return {(int(k.split("_s")[1].split("_")[0]), int(k.split("_f")[1])): v.float()
            for k, v in b.items() if k.startswith("r1_")}, hashlib.sha256((d / "bases.pt").read_bytes()).hexdigest()


def checks(runner, args, P, states, si, site, bases, nat):
    """Fidelity checks: (a) self-patch = unpatched; (b) recompute held-out active swaps of the site's run."""
    import torch

    prompts = P.prompt.tolist()
    out = {}
    rng = np.random.default_rng(0)
    pick = rng.choice(len(P), 512, replace=False)
    with torch.no_grad():
        m, _ = runner.patched([prompts[j] for j in pick], states[torch.as_tensor(pick, device=runner.device), si],
                              site, bases[(0, 0)])
    out["self_patch_max_abs_dM"] = float(np.abs(m.cpu().numpy() - nat["M"][pick]).max())
    d = Path(args.site_dirs[str(site)])
    hs = d / "heldout_swaps.csv.gz"
    if hs.exists():
        det = pd.read_csv(hs, usecols=["base", "src", "M_patched", "rank", "split", "fold"])
        det = det[det["rank"] == 1].sample(512, random_state=0)
        items = pd.read_csv(d / "items.csv")
        pid = dict(zip(P.prompt, P.pid))
        diffs = []
        with torch.no_grad():
            for (k, f), g in det.groupby(["split", "fold"]):
                src = [pid[p] for p in items.prompt.values[g.src]]
                m, _ = runner.patched(list(items.prompt.values[g.base]),
                                      states[torch.as_tensor(src, device=runner.device), si], site, bases[(k, f)])
                diffs.append(np.abs(m.cpu().numpy() - g.M_patched.to_numpy()))
        diffs = np.concatenate(diffs)
        out["heldout_recompute_max_abs_dM"] = float(diffs.max())
        out["heldout_recompute_median_abs_dM"] = float(np.median(diffs))
    print(f"site {site} checks: {out}", flush=True)
    return out


def run(args):
    import torch

    args.site_dirs = dict(x.split("=") for x in args.site_dirs.split(","))
    sites = [int(s) for s in args.sites.split(",")]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runner = PassiveRunner(args)
    P = pd.read_csv(args.prompts)
    plan = pd.read_csv(args.plan)
    if not (P.pid.to_numpy() == np.arange(len(P))).all():
        raise SystemExit("prompt ids must be 0..n-1")
    prompts = P.prompt.tolist()
    meta = {"args": vars(args), "prompts": len(P), "plan_rows": len(plan),
            "plan_sha256": hashlib.sha256(pd.util.hash_pandas_object(plan, index=False).values.tobytes()).hexdigest(),
            "bases_sha256": {}, "checks": {}}

    # 1. unpatched pass (hidden_states[s] is site s: 0 = embeddings, i = output of layer i-1)
    t0 = time.time()
    states = torch.empty(len(P), len(sites), runner.model.config.hidden_size, device=runner.device)
    nat = {k: np.empty(len(P), np.float32) for k in ("M",) + READ}
    with torch.no_grad():
        for i in range(0, len(P), args.batch):
            enc, anchors = runner.encode(prompts[i:i + args.batch])
            o = runner.model(**enc, output_hidden_states=True, use_cache=False)
            m, parts = runner.readout(o.logits, anchors, full=True)
            rows = torch.arange(len(anchors), device=runner.device)
            states[i:i + len(m)] = torch.stack([o.hidden_states[s][rows, anchors].float() for s in sites], 1)
            nat["M"][i:i + len(m)] = m.cpu().numpy()
            for k in READ:
                nat[k][i:i + len(m)] = parts[k].cpu().numpy()
    pd.DataFrame({"pid": P.pid, **nat}).to_parquet(out / "natural.parquet", index=False)
    print(f"natural pass: {len(P)} prompts in {time.time() - t0:.0f}s", flush=True)

    groups = {k: g for k, g in plan.groupby(["split", "fold"])}
    for si, site in enumerate(sites):
        bases, sha = site_bases(args, site)
        bases = {k: v.to(runner.device) for k, v in bases.items()}
        meta["bases_sha256"][str(site)] = sha
        meta["checks"][str(site)] = checks(runner, args, P, states, si, site, bases, nat)
        # 2. projections
        cols = {f"s{k[0]}_f{k[1]}": (states[:, si] @ b[:, 0]).cpu().numpy()
                for k, b in sorted(bases.items())}
        pd.DataFrame({"pid": P.pid, **cols}).to_parquet(out / f"projections_site{site}.parquet", index=False)
        # 3. patches
        t0 = time.time()
        res = {k: np.full(len(plan), np.nan, np.float32) for k in ("M",) + READ}
        with torch.no_grad():
            for key, g in groups.items():
                basis = bases[key]
                rows, base, donor = g.row.to_numpy(), g.base.to_numpy(), g.donor.to_numpy()
                for i in range(0, len(g), args.batch):
                    sl = slice(i, i + args.batch)
                    m, parts = runner.patched([prompts[j] for j in base[sl]],
                                              states[torch.as_tensor(donor[sl], device=runner.device), si],
                                              site, basis, full=True)
                    res["M"][rows[sl]] = m.cpu().numpy()
                    for k in READ:
                        res[k][rows[sl]] = parts[k].cpu().numpy()
        if np.isnan(res["M"]).any():
            raise SystemExit(f"site {site}: unfilled plan rows")
        pd.DataFrame({"row": plan.row, **res}).to_parquet(out / f"patches_site{site}.parquet", index=False)
        print(f"site {site}: {len(plan)} patches in {time.time() - t0:.0f}s", flush=True)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", default="data/das_round2/passive_test/prompts.csv")
    ap.add_argument("--plan", default="data/das_round2/passive_test/plan.csv.gz")
    ap.add_argument("--folds", default="results/das_round2/final_strict/pairs_folds.csv")
    ap.add_argument("--sites", default="8,12,17")
    ap.add_argument("--site-dirs", default="8=results/das_round2/final_strict_site8,"
                                           "12=results/das_round2/final_strict_site12,"
                                           "17=results/das_round2/final_strict")
    ap.add_argument("--out-dir", default="results/das_round2/passive_test")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=1024)
    run(ap.parse_args())
