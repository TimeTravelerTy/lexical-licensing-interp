#!/usr/bin/env python3
"""Nonce-verb passives, natural pass (round3_plan.md, C8). No patching.

For every prompt of `build_nonce_passive.py`: readouts at the last token
(M, O, I, det, pron, refl, " by", ".", " the", " him"; log-probs) and the
last-token residual at each DAS site projected onto that site's 15 fold bases
(`bases_rank1.npz`). Tokenization check: the probe's tokens are identical
with and without the context (the probe is a token-level suffix of the full
prompt), so the measured token is the probe verb's last subtoken.

Causal check (secondary): at sites 6 and 8, the matched-T passive probe's
state is interchanged along d_s (split-0 fold bases, one patch per fold) into
the matched-I probe of the same lemma and slot, and the reverse.

Outputs (`--out-dir`): `natural.parquet`, `projections.npz` ([prompt, site,
basis] and the site list), `causal_swap.parquet`, `run_meta.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_passive_test import READ, PassiveRunner

SITES = (4, 6, 8, 10, 12, 14, 16, 17)
CAUSAL = (6, 8)
KEYS = [(k, f) for k in range(3) for f in range(5)]


def run(args):
    import torch

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runner = PassiveRunner(args)
    P = pd.read_csv(Path(args.data_dir) / "prompts.csv")
    tok = runner.tok
    bad = []
    for r in P[P.kind == "nonce"].itertuples():
        full, probe = tok.encode(r.prompt, add_special_tokens=False), tok.encode(
            (" " if r.prompt != r.probe else "") + r.probe, add_special_tokens=False)
        if full[-len(probe):] != probe:
            bad.append(r.prompt)
    if bad:
        raise SystemExit(f"{len(bad)} prompts where the probe is not a token suffix, e.g. {bad[:3]}")
    npz = np.load(args.bases)
    B = torch.tensor(np.stack([np.stack([npz[f"das_site{s}_s{k}_f{f}"].reshape(-1) for k, f in KEYS], 1)
                               for s in SITES]), dtype=torch.float32, device=runner.device)  # [S, d, 15]
    prompts = P.prompt.tolist()
    nat = {k: np.empty(len(P), np.float32) for k in ("M",) + READ}
    proj = np.empty((len(P), len(SITES), 15), np.float32)
    swap = P[(P.kind == "nonce") & (P.probe_type == "passive") & P.cond.isin(["matched_T", "matched_I"])]
    keep = {int(i): j for j, i in enumerate(swap.pid)}
    held = torch.empty(len(swap), len(CAUSAL), runner.model.config.hidden_size, device=runner.device)
    with torch.no_grad():
        for i in range(0, len(P), args.batch):
            enc, anchors = runner.encode(prompts[i:i + args.batch])
            o = runner.model(**enc, output_hidden_states=True, use_cache=False)
            m, parts = runner.readout(o.logits, anchors, full=True)
            rows = torch.arange(len(anchors), device=runner.device)
            x = torch.stack([o.hidden_states[s][rows, anchors].float() for s in SITES], 1)  # [b, S, d]
            for b_, pid in enumerate(P.pid.to_numpy()[i:i + len(m)]):
                if int(pid) in keep:
                    held[keep[int(pid)]] = x[b_, [SITES.index(c) for c in CAUSAL]]
            proj[i:i + len(m)] = torch.einsum("bsd,sdk->bsk", x, B).cpu().numpy()
            nat["M"][i:i + len(m)] = m.cpu().numpy()
            for k in READ:
                nat[k][i:i + len(m)] = parts[k].cpu().numpy()
    pd.DataFrame({"pid": P.pid, **nat}).to_parquet(out / "natural.parquet", index=False)
    # causal check: matched-T probe's coordinate along d_s into the matched-I probe and the reverse
    sw = swap.set_index(["lemma", "slot", "cond"]).pid
    res = []
    with torch.no_grad():
        for ci, site in enumerate(CAUSAL):
            for f in range(5):
                basis = torch.tensor(npz[f"das_site{site}_s0_f{f}"].reshape(-1, 1), dtype=torch.float32,
                                     device=runner.device)
                for base_c, don_c in (("matched_I", "matched_T"), ("matched_T", "matched_I")):
                    keys = [(l, sl) for l, sl in sorted({(a, b) for a, b, _ in sw.index})]
                    bp = [int(sw[(l, sl, base_c)]) for l, sl in keys]
                    dp = [int(sw[(l, sl, don_c)]) for l, sl in keys]
                    for i in range(0, len(bp), args.batch):
                        src = held[torch.as_tensor([keep[d] for d in dp[i:i + args.batch]], device=runner.device), ci]
                        m, parts = runner.patched([prompts[j] for j in bp[i:i + args.batch]], src, site, basis, full=True)
                        for t, (l, sl) in enumerate(keys[i:i + args.batch]):
                            res.append({"site": site, "fold": f, "base_cond": base_c, "lemma": l, "slot": sl,
                                        "base_pid": bp[i + t], "M": float(m[t]),
                                        **{k: float(parts[k][t]) for k in READ}})
    pd.DataFrame(res).to_parquet(out / "causal_swap.parquet", index=False)
    np.savez_compressed(out / "projections.npz", pid=P.pid.to_numpy(), sites=np.array(SITES),
                        keys=np.array([f"s{k}_f{f}" for k, f in KEYS]), proj=proj)
    meta = {"args": vars(args), "prompts": len(P), "token_suffix_check": "passed",
            "bases_sha256": hashlib.sha256(Path(args.bases).read_bytes()).hexdigest()}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/nonce_passive")
    ap.add_argument("--bases", default="results/das_round2/bases_rank1.npz")
    ap.add_argument("--out-dir", default="results/das_round2/nonce_passive")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=256)
    run(ap.parse_args())
