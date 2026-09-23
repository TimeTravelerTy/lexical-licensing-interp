#!/usr/bin/env python3
"""Score matched FreqBLiMP passive sentences with an autoregressive LM."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def score_batch(model, tokenizer, device, examples):
    import torch
    import torch.nn.functional as F

    texts = [x["sentence"] for x in examples]
    encoded = tokenizer(texts, padding=True, return_tensors="pt", return_offsets_mapping=True,
                        add_special_tokens=False)
    offsets = encoded.pop("offset_mapping").tolist()
    encoded = encoded.to(device)
    with torch.inference_mode():
        logits = model(**encoded, use_cache=False).logits[:, :-1, :].float()
        token_lp = F.log_softmax(logits, dim=-1).gather(
            -1, encoded.input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    scores = []
    for i, example in enumerate(examples):
        verb_start = len(example["prefix"])
        verb_end = verb_start + len(example["verb"])
        parts = {"whole_lp": 0.0, "verb_lp": 0.0, "suffix_lp": 0.0,
                 "by_lp": 0.0, "n_scored_tokens": 0, "verb_tokens": 0}
        by_end = verb_end + len(" by")
        for j in range(1, int(encoded.attention_mask[i].sum().item())):
            start, end = offsets[i][j]
            lp = float(token_lp[i, j - 1].item())
            parts["whole_lp"] += lp
            parts["n_scored_tokens"] += 1
            if end > verb_start and start < verb_end:
                parts["verb_lp"] += lp
                parts["verb_tokens"] += 1
            elif start >= verb_end:
                parts["suffix_lp"] += lp
                if example["paradigm"] == "passive_1" and start < by_end:
                    parts["by_lp"] += lp
        scores.append(parts)
    return scores


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(line) for line in Path(args.data).open(encoding="utf-8") if line.strip()]
    if not rows:
        raise SystemExit("No input rows")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download,
                                               use_fast=True)
    if not tokenizer.is_fast:
        raise SystemExit("A fast tokenizer is required for verb/suffix offsets")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype,
                                                 local_files_only=not args.allow_download)
    device = torch.device(args.device)
    model.to(device).eval()
    model.config.use_cache = False
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = None
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            def examples(side):
                return [{"sentence": row[f"sentence_{side}"], "prefix": row["prefix"],
                         "verb": row[f"{side}_verb"], "paradigm": row["paradigm"]} for row in batch]
            good = score_batch(model, tokenizer, device, examples("good"))
            bad = score_batch(model, tokenizer, device, examples("bad"))
            for row, g, b in zip(batch, good, bad):
                result = dict(row)
                for key, value in g.items():
                    result[f"good_{key}"] = value
                for key, value in b.items():
                    result[f"bad_{key}"] = value
                for segment in ("whole", "verb", "suffix", "by"):
                    result[f"{segment}_margin"] = g[f"{segment}_lp"] - b[f"{segment}_lp"]
                result["correct"] = int(result["whole_margin"] > 0)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(result))
                    writer.writeheader()
                writer.writerow(result)
            print(f"Scored {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    print(f"Wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/matched_passives/pairs.jsonl")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=("auto", "float16", "bfloat16", "float32"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--out", default="results/matched_passives/pythia14b_scores.csv")
    run(ap.parse_args())
