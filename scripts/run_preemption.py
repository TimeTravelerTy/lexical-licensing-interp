#!/usr/bin/env python3
"""Round 6, C5: probe log-probs after in-context exposure (reports/round6/plan.md, C5). fp32, TF32 off.

Input: data/round6/preempt/contexts.csv (build_preemption.py) and A1's freq nouns (data/round6/frame_d11/prefixes.csv).
For every context and noun N, the probes "The N was" / "The N has" / "The N has been" are appended (after a space; k = 0:
the probe alone, as A1) and the participle's subtoken log-probs are scored (teacher forcing, no BOS). Sequences are
sorted by length and batched by a token budget. Check: batched log-probs reproduce single-sequence passes for a sample
to --tol nats (1e-3; run clarification after job 9069908: 1.7e-4 > the first 1e-4 tolerance), checked after saving.
Output: preempt_scores.parquet (cid, noun, frame [was / has / hasbeen], lp, lp1, n_tok), preempt_meta.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

FRAMES = {"was": "The {} was", "has": "The {} has", "hasbeen": "The {} has been"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(args.device)
    C = pd.read_csv(Path(args.data_dir) / "contexts.csv", keep_default_na=False)
    pre = pd.read_csv(args.prefixes, keep_default_na=False)
    nouns = list(pre[pre.set == "freq"].noun)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, local_files_only=True).to(dev).eval()
    vid = {v: tok.encode(" " + v, add_special_tokens=False) for v in C.verb.unique()}
    cid_ids = {c: tok.encode(t, add_special_tokens=False) if t else [] for c, t in zip(C.cid, C.context)}
    pid = {(n, f): tok.encode(t.format(n), add_special_tokens=False) for n in nouns for f, t in FRAMES.items()}
    pid_sp = {(n, f): tok.encode(" " + t.format(n), add_special_tokens=False) for n in nouns for f, t in FRAMES.items()}
    keys, seqs, starts = [], [], []
    for c, v in zip(C.cid, C.verb):
        ctx = cid_ids[c]
        for n in nouns:
            for f in FRAMES:
                p = pid_sp[(n, f)] if ctx else pid[(n, f)]
                keys.append((c, n, f))
                seqs.append(ctx + p + vid[v])
                starts.append(len(ctx) + len(p))
    L = np.array([len(s) for s in seqs])
    order = np.argsort(L, kind="stable")
    lp_sum = np.zeros(len(seqs))
    lp_first = np.zeros(len(seqs))
    ntok = np.array([len(s) - st for s, st in zip(seqs, starts)])
    t0 = time.time()

    def score(idx):
        T = int(L[idx].max())
        x = torch.full((len(idx), T), tok.eos_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), T), dtype=torch.long)
        for r, i in enumerate(idx):
            x[r, :L[i]] = torch.tensor(seqs[i])
            att[r, :L[i]] = 1
        x, att = x.to(dev), att.to(dev)
        with torch.inference_mode():
            lp = torch.log_softmax(model(input_ids=x, attention_mask=att, use_cache=False).logits.float(), -1)
        out = []
        for r, i in enumerate(idx):
            st = starts[i]
            t = x[r, st:L[i]]
            sub = lp[r, st - 1:L[i] - 1].gather(-1, t[:, None]).squeeze(-1).double().cpu().numpy()
            out.append((float(sub.sum()), float(sub[0])))
        return out

    i = 0
    nb = 0
    while i < len(order):
        T = L[order[i]]
        n = max(1, min(args.max_batch, args.token_budget // int(L[order[min(i + args.max_batch, len(order)) - 1]])))
        idx = order[i:i + n]
        for j, (a, b) in zip(idx, score(idx)):
            lp_sum[j], lp_first[j] = a, b
        i += n
        nb += 1
        if nb % 500 == 0:
            print(f"{i}/{len(order)} sequences, {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(0)
    chk = 0.0
    for j in rng.choice(len(seqs), 64, replace=False):
        a, _ = score(np.array([j]))[0]
        chk = max(chk, abs(a - lp_sum[j]))
    print(f"done in {time.time() - t0:.0f}s; batched vs single max diff {chk:.1e}", flush=True)
    S = pd.DataFrame(keys, columns=["cid", "noun", "frame"])
    S["lp"], S["lp1"], S["n_tok"] = lp_sum, lp_first, ntok
    o = Path(args.out_dir)
    o.mkdir(parents=True, exist_ok=True)
    S.to_parquet(o / "preempt_scores.parquet", index=False)
    meta = {"model": args.model, "dtype": "float32", "tf32": False, "rows": len(S), "check_batched_vs_single": chk,
            "seconds": round(time.time() - t0), "inputs": {"contexts.csv": sha(Path(args.data_dir) / "contexts.csv"),
                                                          "prefixes.csv": sha(args.prefixes)},
            "outputs": {"preempt_scores.parquet": sha(o / "preempt_scores.parquet")}}
    (o / "preempt_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    assert chk <= args.tol, chk  # round 6 C5 run clarification: 1e-3 (fp32 padding noise), checked after saving


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/round6/preempt")
    ap.add_argument("--prefixes", default="data/round6/frame_d11/prefixes.csv")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--token-budget", type=int, default=32768)
    ap.add_argument("--max-batch", type=int, default=512)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--out-dir", default="results/round6/preempt")
    main(ap.parse_args())
