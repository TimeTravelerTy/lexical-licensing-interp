#!/usr/bin/env python3
"""Round 4, A4: the B5 "by lever" (reports/round4/plan.md, A4).

On the B5 active test plan (bad active bases; arms 1 = G / B passive donors and 3 = T / I active
donors; all splits), four interventions per site, split and fold (all unit norm):
- dp:     interchange along d_p (recomputed here for paired comparisons on the same rows);
- shuf:   interchange along the shuffled-label passive basis s (`shuf_s{k}_f{f}`);
- perp:   interchange along e = (d_p - (d_p.s) s) / |.|;
- dpminus: d_p's own displacement with its s component removed,
          delta = ((h_src - h).d_p) (d_p - (d_p.s) s)   (dose-preserving control).
Per-row displacement norms are saved. Nulls on the split-0 G / B rows of bad bases: 100 random
rank-1 directions norm-matched row by row to the shuf displacement, and 100 to the perp displacement
(point-estimate D per draw, as `run_reverse_test.py`).

Outputs (`--out-dir`): `patches_{name}_site{s}.parquet` (row, readouts, disp_norm),
`null_{name}_site{s}.npz`, `run_meta.json`.
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

READ = ("M", "O", "I", "det", "pron", "refl", "by", "dot", "the", "him")
KEYS = [(k, f) for k in range(3) for f in range(5)]
NAMES = ("dp", "shuf", "perp", "dpminus")


class LeverRunner(PassiveRunner):
    def patched_rw(self, prompts, src_h, site, read, write, match_norm=None):
        """delta = ((src - h).read) write at the last token; optional norm-matching to the displacement of
        an interchange along `match_norm` (unit vector). Returns readouts and per-row |delta|."""
        torch = self.torch
        enc, anchors = self.encode(prompts)
        rows = torch.arange(len(anchors), device=self.device)
        norms = {}

        def patch(hidden):
            h = hidden.float()
            base = h[rows, anchors]
            delta = ((src_h - base) @ read)[:, None] * write[None]
            if match_norm is not None:
                ref = ((src_h - base) @ match_norm)[:, None] * match_norm[None]
                delta = delta * (ref.norm(dim=-1, keepdim=True) / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6))
            norms["d"] = delta.norm(dim=-1)
            h = h.clone()
            h[rows, anchors] = base + delta
            return h.to(hidden.dtype)

        handle = self.model.gpt_neox.layers[site - 1].register_forward_hook(
            lambda m, i, o: replace_output(o, patch(normalize_output(o))))
        try:
            out = self.model(**enc, use_cache=False)
        finally:
            handle.remove()
        m, parts = self.readout(out.logits, anchors, full=True)
        return m, parts, norms["d"]


def unit(v):
    return v / np.linalg.norm(v)


def run(args):
    import torch

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runner = LeverRunner(args)
    ddir = Path(args.data_dir)
    P = pd.read_csv(ddir / "prompts.csv")
    items = pd.read_csv(ddir / "items.csv")
    plan = pd.read_csv(ddir / "plan.csv.gz")
    want = json.loads((ddir / "plan_meta.json").read_text())["plan_sha256"]
    got = hashlib.sha256(pd.util.hash_pandas_object(plan, index=False).values.tobytes()).hexdigest()
    assert got == want, ("plan hash differs from plan_meta.json", got, want)
    prompts = P.prompt.tolist()
    sites = [int(s) for s in args.sites.split(",")]
    rows_all = plan[(plan.side == "bad") & plan.cond.isin(["G", "B", "T", "I"])].reset_index(drop=True)
    sel0 = np.flatnonzero(((rows_all.split == 0) & rows_all.cond.isin(["G", "B"])).to_numpy())
    r0 = rows_all.iloc[sel0].reset_index(drop=True)
    pair_of = dict(zip(items.item_id, items.pair_id))
    meta = {"args": vars(args), "plan_sha256": got, "rows": len(rows_all), "null_rows": len(r0), "cosines": {},
            "bases_sha256": {}}
    need = np.unique(np.r_[rows_all.base.to_numpy(), rows_all.donor.to_numpy()])
    states = torch.zeros(len(P), len(sites), runner.model.config.hidden_size, device=runner.device)
    nat = {k: np.full(len(P), np.nan, np.float32) for k in READ}
    with torch.no_grad():
        for i in range(0, len(need), args.batch):
            ids = need[i:i + args.batch]
            enc, anchors = runner.encode([prompts[j] for j in ids])
            o = runner.model(**enc, output_hidden_states=True, use_cache=False)
            m, parts = runner.readout(o.logits, anchors, full=True)
            rr = torch.arange(len(anchors), device=runner.device)
            idx = torch.as_tensor(ids, device=runner.device)
            states[idx] = torch.stack([o.hidden_states[s][rr, anchors].float() for s in sites], 1)
            nat["M"][ids] = m.cpu().numpy()
            for k in READ[1:]:
                nat[k][ids] = parts[k].cpu().numpy()
    pd.DataFrame({"pid": P.pid, **nat}).to_parquet(out / "natural.parquet", index=False)
    natb = pd.DataFrame(nat).loc[r0.base.to_numpy()].to_numpy()

    def point_d(vals):
        d = pd.DataFrame(vals - natb, columns=READ)
        d["item"], d["cond"] = r0.item_id.to_numpy(), r0.cond.to_numpy()
        m = d.groupby(["item", "cond"])[list(READ)].mean()
        dd = m.xs("G", level="cond") - m.xs("B", level="cond")
        dd["pair"] = dd.index.map(pair_of)
        return dd.groupby("pair")[list(READ)].mean().mean().to_numpy()

    for si, site in enumerate(sites):
        t0 = time.time()
        z = np.load(Path(args.reverse_dir) / f"final_site{site}" / "bases_np.npz")
        meta["bases_sha256"][str(site)] = hashlib.sha256(
            (Path(args.reverse_dir) / f"final_site{site}" / "bases_np.npz").read_bytes()).hexdigest()
        V = {}
        cos = {}
        for k, f in KEYS:
            dp, s = unit(z[f"r1_s{k}_f{f}"].reshape(-1)), unit(z[f"shuf_s{k}_f{f}"].reshape(-1))
            c = float(dp @ s)
            e = unit(dp - c * s)
            tt = lambda v: torch.tensor(v, dtype=torch.float32, device=runner.device)
            V[(k, f)] = {"dp": (tt(dp), tt(dp)), "shuf": (tt(s), tt(s)), "perp": (tt(e), tt(e)),
                         "dpminus": (tt(dp), tt(dp - c * s))}
            cos[f"s{k}_f{f}"] = c
        meta["cosines"][str(site)] = cos
        res = {n: {k: np.full(len(rows_all), np.nan, np.float32) for k in READ + ("disp_norm",)} for n in NAMES}
        with torch.no_grad():
            for (k, f), g in rows_all.assign(pos=np.arange(len(rows_all))).groupby(["split", "fold"]):
                for i in range(0, len(g), args.batch):
                    gg = g.iloc[i:i + args.batch]
                    src = states[torch.as_tensor(gg.donor.to_numpy(), device=runner.device), si]
                    for n in NAMES:
                        rd, wr = V[(k, f)][n]
                        m, parts, dn = runner.patched_rw([prompts[j] for j in gg.base], src, site, rd, wr)
                        pos = gg.pos.to_numpy()
                        res[n]["M"][pos] = m.cpu().numpy()
                        for key in READ[1:]:
                            res[n][key][pos] = parts[key].cpu().numpy()
                        res[n]["disp_norm"][pos] = dn.cpu().numpy()
        for n in NAMES:
            assert not np.isnan(res[n]["M"]).any()
            pd.DataFrame({"row": rows_all.row.to_numpy(), **res[n]}).to_parquet(out / f"patches_{n}_site{site}.parquet",
                                                                                index=False)
        # nulls: random rank-1 directions norm-matched to the shuf / perp displacement, split-0 G / B rows
        gen = torch.Generator(device="cpu").manual_seed(7)
        for n in ("shuf", "perp"):
            das_vals = np.stack([res[n][key][sel0] for key in READ], 1)
            draws = []
            with torch.no_grad():
                for _ in range(args.n_random):
                    vals = np.empty((len(r0), len(READ)), np.float32)
                    for f, g in r0.assign(pos=np.arange(len(r0))).groupby("fold"):
                        rnd = torch.linalg.qr(torch.randn(runner.model.config.hidden_size, 1, generator=gen))[0][:, 0]
                        rnd = rnd.to(runner.device)
                        ref = V[(0, f)][n][0]
                        for i in range(0, len(g), args.batch):
                            gg = g.iloc[i:i + args.batch]
                            src = states[torch.as_tensor(gg.donor.to_numpy(), device=runner.device), si]
                            m, parts, _ = runner.patched_rw([prompts[j] for j in gg.base], src, site, rnd, rnd,
                                                            match_norm=ref)
                            vals[gg.pos.to_numpy()] = np.stack([m.cpu().numpy()] + [parts[key].cpu().numpy()
                                                                                    for key in READ[1:]], 1)
                    draws.append(point_d(vals))
            np.savez_compressed(out / f"null_{n}_site{site}.npz", readouts=np.array(READ), das=point_d(das_vals),
                                draws=np.array(draws))
        print(f"site {site}: {len(rows_all)} rows x {len(NAMES)} + 2 x {args.n_random} null draws in "
              f"{time.time() - t0:.0f}s", flush=True)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/das_round2/reverse_test")
    ap.add_argument("--reverse-dir", default="results/das_round2/reverse")
    ap.add_argument("--out-dir", default="results/round4/bylever")
    ap.add_argument("--sites", default="4,6,8,10,12,14,16,17")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--n-random", type=int, default=100)
    a = ap.parse_args()
    run(a)
