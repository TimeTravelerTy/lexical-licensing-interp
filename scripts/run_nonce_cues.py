#!/usr/bin/env python3
"""Round 4, Part C: nonce cue tests, natural pass (reports/round4/plan.md, C8 / C9). No patching.

For every prompt of `build_nonce_cues.py`: readouts at the last token (as round-3 C8: M, O, I, det,
pron, refl, " by", ".", " the", " him", PREP without " by") and the last-token residual at each
DAS site projected onto its 15 fold bases. Token checks:
- the probe is a token-level suffix of the full prompt (so the readout is at the probe verb);
- for the ing / s forms, whether the probe's final token id occurs anywhere in the context
  (recorded per lemma and form; the analysis drops lemmas where it does).

Outputs (`--out-dir`): `natural.parquet`, `projections.npz`, `token_check.csv`, `run_meta.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_nonce_passive import KEYS, READ, SITES, NonceRunner


def run(args):
    import torch

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runner = NonceRunner(args)
    tok = runner.tok
    P = pd.read_csv(Path(args.data_dir) / "prompts.csv", keep_default_na=False)
    bad, checks = [], []
    for r in P[P.kind == "nonce"].itertuples():
        full = tok.encode(r.prompt, add_special_tokens=False)
        probe = tok.encode(" " + r.probe, add_special_tokens=False)
        if full[-len(probe):] != probe:
            bad.append(r.prompt)
        if r.slot == 0:
            ctx = tok.encode(r.context, add_special_tokens=False)
            checks.append({"lemma": r.lemma, "cond": r.cond, "probe_final_id": probe[-1],
                           "probe_final_token": tok.decode([probe[-1]]), "in_context": probe[-1] in set(ctx)})
    if bad:
        raise SystemExit(f"{len(bad)} prompts where the probe is not a token suffix, e.g. {bad[:3]}")
    tc = pd.DataFrame(checks)
    tc.to_csv(out / "token_check.csv", index=False)
    npz = np.load(args.bases)
    B = torch.tensor(np.stack([np.stack([npz[f"das_site{s}_s{k}_f{f}"].reshape(-1) for k, f in KEYS], 1)
                               for s in SITES]), dtype=torch.float32, device=runner.device)  # [S, d, 15]
    prompts = P.prompt.tolist()
    nat = {k: np.empty(len(P), np.float32) for k in ("M",) + READ}
    proj = np.empty((len(P), len(SITES), 15), np.float32)
    with torch.no_grad():
        for i in range(0, len(P), args.batch):
            enc, anchors = runner.encode(prompts[i:i + args.batch])
            o = runner.model(**enc, output_hidden_states=True, use_cache=False)
            m, parts = runner.readout(o.logits, anchors, full=True)
            rows = torch.arange(len(anchors), device=runner.device)
            x = torch.stack([o.hidden_states[s][rows, anchors].float() for s in SITES], 1)
            proj[i:i + len(m)] = torch.einsum("bsd,sdk->bsk", x, B).cpu().numpy()
            nat["M"][i:i + len(m)] = m.cpu().numpy()
            for k in READ:
                nat[k][i:i + len(m)] = parts[k].cpu().numpy()
    pd.DataFrame({"pid": P.pid, **nat}).to_parquet(out / "natural.parquet", index=False)
    np.savez_compressed(out / "projections.npz", pid=P.pid.to_numpy(), sites=np.array(SITES),
                        keys=np.array([f"s{k}_f{f}" for k, f in KEYS]), proj=proj)
    meta = {"args": vars(args), "prompts": len(P), "token_suffix_check": "passed",
            "final_token_in_context": tc.groupby("cond").in_context.sum().to_dict(),
            "bases_sha256": hashlib.sha256(Path(args.bases).read_bytes()).hexdigest()}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps(meta, default=str, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/round4/nonce_cues")
    ap.add_argument("--bases", default="results/das_round2/bases_rank1.npz")
    ap.add_argument("--out-dir", default="results/round4/nonce_cues")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=256)
    run(ap.parse_args())
