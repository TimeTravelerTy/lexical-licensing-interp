#!/usr/bin/env python3
"""Next-token readout after each prompt: tracked-token log-probs and top-k.

Tokenization matches `score_matched_passives.py` (no special tokens), so
log P(" by" | prefix) here equals the `by_lp` of the full-sentence scores up
to kernel noise.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

TRACKED = (" the", " a", " an", " by", ".", ",", " his", " her", " their", " its", " this", " that",
           " some", " him", " them", " it", " and", " to", " in", " with", " on", " at", " for",
           " from", " as", " into", " up", " out", " off", " down", " over", "\n",
           "Yes", " Yes", "No", " No")


def run(args):
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(line) for line in Path(args.data).open(encoding="utf-8") if line.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tracked = {}
    for tok in TRACKED:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if len(ids) == 1:
            tracked[tok] = ids[0]
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype,
                                                 local_files_only=not args.allow_download)
    device = torch.device(args.device)
    model.to(device).eval()
    ids_t = torch.tensor(list(tracked.values()), device=device)

    lengths = [len(tokenizer.encode(r["prompt"], add_special_tokens=False)) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: lengths[i])
    names = [f"lp[{json.dumps(t)[1:-1]}]" for t in tracked]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["prompt_id", "n_tokens", "last_token"] + names + ["topk"])
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            enc = tokenizer([rows[i]["prompt"] for i in idx], padding=True, return_tensors="pt",
                            add_special_tokens=False).to(device)
            last = enc.attention_mask.sum(1) - 1
            with torch.inference_mode():
                logits = model(**enc, use_cache=False).logits
                lp = F.log_softmax(logits[torch.arange(len(idx), device=device), last].float(), dim=-1)
                top = lp.topk(args.top_k, dim=-1)
                sel = lp[:, ids_t].cpu().tolist()
            top_ids, top_lp = top.indices.cpu().tolist(), top.values.cpu().tolist()
            for j, i in enumerate(idx):
                topk = [[tokenizer.decode([t]), round(v, 4)] for t, v in zip(top_ids[j], top_lp[j])]
                last_tok = tokenizer.decode([int(enc.input_ids[j, last[j]])])
                writer.writerow([rows[i]["prompt_id"], int(last[j]) + 1, last_tok]
                                + [f"{v:.5f}" for v in sel[j]] + [json.dumps(topk, ensure_ascii=False)])
            if (start // args.batch_size) % 50 == 0:
                print(f"Read {min(start + len(idx), len(order))}/{len(order)}", flush=True)
    print(f"Wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/passive_das_prep/prompts.jsonl")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16", "float32"))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--out", default="results/passive_das_prep/pythia14b_readout.csv")
    run(ap.parse_args())
