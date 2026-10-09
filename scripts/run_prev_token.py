#!/usr/bin/env python3
"""Round 4, A5(a): generic previous-token scores for every attention head.

Score of head (l, h) = mean attention from position t to t - 1 over positions t >= 1, on
(i) 256 random-token sequences of 64 tokens (ids uniform in [1000, 50000), seed 17) and
(ii) the natural-text passages of `build_aux_heads_text.py` (first 64 tokens of each; passages
shorter than 48 tokens are an error). A previous-token head scores >= 0.5 on both (plan.md, A5).

Output (`--out`): `prev_token_scores.csv` (layer, head, random, natural).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, local_files_only=True)
    try:
        model.set_attn_implementation("eager")
    except Exception:
        model.config._attn_implementation = "eager"
    model = model.to(args.device).eval()
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    rng = np.random.default_rng(args.seed)
    rand = torch.as_tensor(rng.integers(1000, 50000, size=(args.n, args.length)), device=args.device)
    texts = Path(args.text).read_text(encoding="utf-8").strip().split("\n")
    ids = [tok.encode(t, add_special_tokens=False) for t in texts]
    assert min(len(i) for i in ids) >= 48, "a natural passage is shorter than 48 tokens"
    T = min(args.length, min(len(i) for i in ids))
    nat = torch.as_tensor([i[:T] for i in ids], device=args.device)
    scores = {}
    with torch.no_grad():
        for name, x in (("random", rand), ("natural", nat)):
            acc = torch.zeros(L, H, device=args.device, dtype=torch.float64)
            n = 0
            for i in range(0, len(x), args.batch):
                o = model(input_ids=x[i:i + args.batch], output_attentions=True, use_cache=False)
                for l, a in enumerate(o.attentions):  # [B, H, T, T]
                    t = torch.arange(1, a.shape[-1], device=args.device)
                    acc[l] += a[:, :, t, t - 1].double().mean(-1).sum(0)
                n += len(x[i:i + args.batch])
            scores[name] = (acc / n).cpu().numpy()
    rows = [{"layer": l, "head": h, "name": f"L{l}H{h}", "random": float(scores["random"][l, h]),
             "natural": float(scores["natural"][l, h])} for l in range(L) for h in range(H)]
    df = pd.DataFrame(rows)
    df["prev_token_head"] = (df.random >= 0.5) & (df.natural >= 0.5)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "prev_token_scores.csv", index=False)
    print(df.sort_values("natural", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text", default="data/round4/aux_heads/natural_text.txt")
    ap.add_argument("--out", default="results/round4/aux_heads")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--length", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
