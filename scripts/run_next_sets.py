#!/usr/bin/env python3
"""Round 6, B3: next-token log-mass of token sets after each prefix (reports/round6/plan.md, B3). fp32, TF32 off.

Input: prefixes.csv (id, text) and sets.json (build_alternation.py). For every prefix, the log-softmax at its last token
and, per set, log sum_{t in set} P(t). No BOS (as A1). Check: the batched result reproduces an unbatched forward pass for
a sample of prefixes to 1e-4.
Output: next_sets.parquet (id + one column per set), next_sets_meta.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(args.device)
    data = Path(args.data_dir)
    P = pd.read_csv(data / "prefixes.csv", keep_default_na=False)
    S = json.loads((data / "sets.json").read_text())["ids"]
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, local_files_only=True).to(dev).eval()
    sid = {k: torch.tensor(v, device=dev) for k, v in S.items()}
    enc = [tok.encode(t, add_special_tokens=False) for t in P.text]
    order = np.argsort([len(e) for e in enc], kind="stable")
    out = {k: np.zeros(len(P)) for k in S}
    t0 = time.time()

    def forward(idx):
        seqs = [enc[i] for i in idx]
        T = max(map(len, seqs))
        x = torch.full((len(idx), T), tok.eos_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), T), dtype=torch.long)
        for r, s in enumerate(seqs):
            x[r, :len(s)] = torch.tensor(s)
            att[r, :len(s)] = 1
        with torch.inference_mode():
            lg = model(input_ids=x.to(dev), attention_mask=att.to(dev), use_cache=False).logits.float()
        last = torch.tensor([len(s) - 1 for s in seqs], device=dev)
        lp = torch.log_softmax(lg[torch.arange(len(idx), device=dev), last], -1)
        return {k: torch.logsumexp(lp[:, v], -1).double().cpu().numpy() for k, v in sid.items()}

    for s in range(0, len(order), args.batch):
        b = order[s:s + args.batch]
        r = forward(b)
        for k in S:
            out[k][b] = r[k]
    rng = np.random.default_rng(0)
    chk = 0.0
    for i in rng.choice(len(P), min(64, len(P)), replace=False):
        r = forward([i])
        chk = max(chk, max(abs(r[k][0] - out[k][i]) for k in S))
    print(f"{len(P)} prefixes in {time.time() - t0:.0f}s; batched vs single max diff {chk:.1e}", flush=True)
    assert chk <= 1e-4, chk
    res = pd.DataFrame({"id": P.id, **out})
    o = Path(args.out_dir)
    o.mkdir(parents=True, exist_ok=True)
    res.to_parquet(o / "next_sets.parquet", index=False)
    meta = {"model": args.model, "dtype": "float32", "tf32": False, "rows": len(res), "check_batched_vs_single": chk,
            "inputs": {f: sha(data / f) for f in ("prefixes.csv", "sets.json")},
            "outputs": {"next_sets.parquet": sha(o / "next_sets.parquet")}}
    (o / "next_sets_meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/round6/alternation")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out-dir", default="results/round6/alternation/pythia-1.4b")
    main(ap.parse_args())
