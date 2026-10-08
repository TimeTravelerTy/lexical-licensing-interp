#!/usr/bin/env python3
"""Object relatives and tough constructions: natural pass, projections, patches (round3_plan.md, C9).

Inputs: `build_constructions.py` outputs (`prompts.csv`, `plan.csv.gz`); the
active-trained rank-1 bases of every DAS site (`bases_rank1.npz`).
- natural pass over every prompt: readouts and per-token log-probs of the
  declared token union -> `natural.parquet`, `natural_tokens.npz`;
- projections of every prompt onto each site's 15 fold bases -> `projections.npz`;
- every plan row patched at each site (interchange along the item's fold basis at
  the verb's last token) -> `patches_site{s}.parquet`; per-token log-probs of the
  token union for bad-base T / I rows at sites 8 and 17 -> `patch_tokens_site{s}.npz`.

Readouts (log-probs at the last token): O, I, det, pron, refl, " by", ".",
" the", " him" (as the passive test) and PREP, MAIN, END (punctuation), END_BROAD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_passive_test import READ as PREAD, PassiveRunner

SITES = (4, 6, 8, 10, 12, 14, 16, 17)
TOKEN_SITES = (8, 17)
KEYS = [(k, f) for k in range(3) for f in range(5)]
SETS = {
    "PREP": (" to", " in", " with", " on", " at", " for", " from", " into", " over", " about", " of", " upon",
             " against", " through", " after", " under", " around", " across", " toward", " without", " during",
             " behind", " near", " onto", " by", " as"),
    "MAIN": (" was", " is", " has", " had", " will", " would", " could", " can", " must", " should", " might", " did",
             " does"),
    "END": (".", ",", "!", "?", ";"),
    "END_BROAD": (".", ",", "!", "?", ";", " and", " but", " because", " if", " when", " so"),
}
READ = ("M",) + tuple(PREAD) + tuple(SETS)


class ConsRunner(PassiveRunner):
    def __init__(self, args):
        super().__init__(args)
        torch = self.torch
        for name, toks in SETS.items():
            ids = []
            for t in toks:
                enc = self.tok.encode(t, add_special_tokens=False)
                if len(enc) != 1:
                    raise SystemExit(f"token {t!r} is not a single token")
                ids.append(enc[0])
            self.ids[name] = torch.tensor(ids, device=self.device)
        union = sorted(set(int(i) for k in ("O", "PREP", "MAIN", "END_BROAD") for i in self.ids[k].tolist()))
        self.union = torch.tensor(union, device=self.device)

    def readout(self, logits, anchors, full=False):
        torch = self.torch
        rows = torch.arange(logits.shape[0], device=logits.device)
        z = logits[rows, anchors].float()
        m = torch.logsumexp(z[:, self.ids["O"]], -1) - torch.logsumexp(z[:, self.ids["I"]], -1)
        if not full:
            return m, None
        lp = torch.log_softmax(z, -1)
        parts = {k: torch.logsumexp(lp[:, self.ids[k]], -1) for k in READ[1:]}
        parts["_tokens"] = lp[:, self.union]
        return m, parts


def run(args):
    import torch

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runner = ConsRunner(args)
    P = pd.read_csv(Path(args.data_dir) / "prompts.csv")
    plan = pd.read_csv(Path(args.data_dir) / "plan.csv.gz")
    assert (P.pid.to_numpy() == np.arange(len(P))).all()
    prompts = P.prompt.tolist()
    npz = np.load(args.bases)
    bases = {s: {k: torch.tensor(npz[f"das_site{s}_s{k[0]}_f{k[1]}"].reshape(-1, 1), dtype=torch.float32,
                                 device=runner.device) for k in KEYS} for s in SITES}
    meta = {"args": vars(args), "prompts": len(P), "plan_rows": len(plan), "union_tokens":
            [runner.tok.decode([int(i)]) for i in runner.union.tolist()], "checks": {},
            "bases_sha256": hashlib.sha256(Path(args.bases).read_bytes()).hexdigest(),
            "plan_sha256": hashlib.sha256(pd.util.hash_pandas_object(plan, index=False).values.tobytes()).hexdigest()}
    t0 = time.time()
    states = torch.empty(len(P), len(SITES), runner.model.config.hidden_size, device=runner.device)
    nat = {k: np.empty(len(P), np.float32) for k in READ}
    ntok = np.empty((len(P), len(runner.union)), np.float32)
    proj = np.empty((len(P), len(SITES), 15), np.float32)
    Bst = torch.stack([torch.cat([bases[s][k] for k in KEYS], 1) for s in SITES])  # [S, d, 15]
    with torch.no_grad():
        for i in range(0, len(P), args.batch):
            enc, anchors = runner.encode(prompts[i:i + args.batch])
            o = runner.model(**enc, output_hidden_states=True, use_cache=False)
            m, parts = runner.readout(o.logits, anchors, full=True)
            rows = torch.arange(len(anchors), device=runner.device)
            x = torch.stack([o.hidden_states[s][rows, anchors].float() for s in SITES], 1)
            states[i:i + len(m)] = x
            proj[i:i + len(m)] = torch.einsum("bsd,sdk->bsk", x, Bst).cpu().numpy()
            nat["M"][i:i + len(m)] = m.cpu().numpy()
            for k in READ[1:]:
                nat[k][i:i + len(m)] = parts[k].cpu().numpy()
            ntok[i:i + len(m)] = parts["_tokens"].cpu().numpy()
    pd.DataFrame({"pid": P.pid, **nat}).to_parquet(out / "natural.parquet", index=False)
    np.savez_compressed(out / "natural_tokens.npz", pid=P.pid.to_numpy(), tokens=np.array(meta["union_tokens"]), lp=ntok)
    np.savez_compressed(out / "projections.npz", pid=P.pid.to_numpy(), sites=np.array(SITES),
                        keys=np.array([f"s{k}_f{f}" for k, f in KEYS]), proj=proj)
    print(f"natural pass: {len(P)} prompts in {time.time() - t0:.0f}s", flush=True)
    groups = {k: g for k, g in plan.groupby(["split", "fold"])}
    tok_rows = plan[(plan.side == "bad") & plan.cond.isin(["T", "I"])].row.to_numpy()
    tok_pos = {int(r): j for j, r in enumerate(tok_rows)}
    for si, site in enumerate(SITES):
        t0 = time.time()
        with torch.no_grad():  # self-patch check
            pick = np.random.default_rng(0).choice(len(P), 256, replace=False)
            m, _ = runner.patched([prompts[j] for j in pick], states[torch.as_tensor(pick, device=runner.device), si],
                                  site, bases[site][(0, 0)])
        meta["checks"][str(site)] = {"self_patch_max_abs_dM": float(np.abs(m.cpu().numpy() - nat["M"][pick]).max())}
        res = {k: np.full(len(plan), np.nan, np.float32) for k in READ}
        ptok = np.full((len(tok_rows), len(runner.union)), np.nan, np.float32) if site in TOKEN_SITES else None
        with torch.no_grad():
            for key, g in groups.items():
                rows, base, donor = g.row.to_numpy(), g.base.to_numpy(), g.donor.to_numpy()
                for i in range(0, len(g), args.batch):
                    sl = slice(i, i + args.batch)
                    m, parts = runner.patched([prompts[j] for j in base[sl]],
                                              states[torch.as_tensor(donor[sl], device=runner.device), si],
                                              site, bases[site][key], full=True)
                    res["M"][rows[sl]] = m.cpu().numpy()
                    for k in READ[1:]:
                        res[k][rows[sl]] = parts[k].cpu().numpy()
                    if ptok is not None:
                        lpt = parts["_tokens"].cpu().numpy()
                        for t, r in enumerate(rows[sl]):
                            if int(r) in tok_pos:
                                ptok[tok_pos[int(r)]] = lpt[t]
        if np.isnan(res["M"]).any():
            raise SystemExit(f"site {site}: unfilled plan rows")
        pd.DataFrame({"row": plan.row, **res}).to_parquet(out / f"patches_site{site}.parquet", index=False)
        if ptok is not None:
            np.savez_compressed(out / f"patch_tokens_site{site}.npz", rows=tok_rows, lp=ptok)
        print(f"site {site}: {len(plan)} patches in {time.time() - t0:.0f}s; {meta['checks'][str(site)]}", flush=True)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/constructions")
    ap.add_argument("--bases", default="results/das_round2/bases_rank1.npz")
    ap.add_argument("--out-dir", default="results/das_round2/constructions")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=1024)
    run(ap.parse_args())
