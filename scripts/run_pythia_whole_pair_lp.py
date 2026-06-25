#!/usr/bin/env python3
"""Whole-pair sentence log-probability checks for v2 lexical licensing rows.

This scores the full experimental-frame sentence for matched good/bad rows:

    LP(full good sentence) - LP(full bad sentence)

Unlike the next-token sanity check, this does not ask whether the model expects
an object immediately after the verb. The verb-final `prompt` is retained only
as metadata for later interventions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from run_pythia_attribution_patching import (
    DEFAULT_MODEL,
    DEFAULT_REGIMES,
    DEFAULT_SUBTASKS,
    iter_jsonl,
    parse_csv_arg,
    shell_safe_slug,
    stable_unit,
)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def load_rows(path: Path, subtasks: set[str], regimes: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        if row.get("alignment_status") != "aligned":
            continue
        if row.get("subtask") not in subtasks:
            continue
        if row.get("regime") not in regimes:
            continue
        if not row.get("full_sentence"):
            raise SystemExit(
                "Rows are missing full_sentence. Rebuild with scripts/build_aligned_templates_v2.py "
                "after the whole-pair fields were added."
            )
        rows.append(row)
    return rows


def sort_key(row: dict[str, Any], seed: int) -> tuple[float, str, int]:
    stable_key = "|".join(
        [
            str(row.get("regime", "")),
            str(row.get("subtask", "")),
            str(row.get("template_id", "")),
            str(row.get("context_id", "")),
            str(row.get("side", "")),
            str(row.get("verb", "")),
            str(row.get("source_idx", "")),
            str(row.get("row_id", "")),
        ]
    )
    return (stable_unit(stable_key, seed), str(row.get("verb", "")), int(row.get("source_idx", 0)))


def build_pairs(rows: list[dict[str, Any]], max_pairs_per_group: int | None, seed: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[tuple[str, str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (
            str(row["regime"]),
            str(row["subtask"]),
            str(row["template_id"]),
            str(row.get("context_id", row["template_id"])),
        )
        grouped[key][str(row["side"])].append(row)

    pairs: list[dict[str, Any]] = []
    skip_counts: dict[str, int] = defaultdict(int)
    for (regime, subtask, template_id, context_id), sides in sorted(grouped.items()):
        good_rows = sorted(sides.get("good", []), key=lambda row: sort_key(row, seed))
        bad_rows = sorted(sides.get("bad", []), key=lambda row: sort_key(row, seed))
        if not good_rows or not bad_rows:
            skip_counts["missing_good_or_bad_side"] += max(len(good_rows), len(bad_rows), 1)
            continue
        n_pairs = min(len(good_rows), len(bad_rows))
        if max_pairs_per_group is not None:
            if n_pairs > max_pairs_per_group:
                skip_counts["max_pairs_per_group"] += n_pairs - max_pairs_per_group
            n_pairs = min(n_pairs, max_pairs_per_group)
        for index, (good, bad) in enumerate(zip(good_rows[:n_pairs], bad_rows[:n_pairs])):
            if str(good.get("verb")) == str(bad.get("verb")):
                skip_counts["same_good_bad_verb"] += 1
                continue
            pairs.append(
                {
                    "pair_key": f"{regime}|{subtask}|{template_id}|{context_id}|{index}|{good.get('verb')}|{bad.get('verb')}",
                    "regime": regime,
                    "subtask": subtask,
                    "template_id": template_id,
                    "context_id": context_id,
                    "subject": good.get("subject", ""),
                    "subject_class": good.get("subject_class", ""),
                    "good_row_id": good.get("row_id", ""),
                    "bad_row_id": bad.get("row_id", ""),
                    "good_verb": good.get("verb", ""),
                    "bad_verb": bad.get("verb", ""),
                    "good_source_zipf": good.get("source_zipf", ""),
                    "bad_source_zipf": bad.get("source_zipf", ""),
                    "good_source_zipf_regime": good.get("source_zipf_regime", ""),
                    "bad_source_zipf_regime": bad.get("source_zipf_regime", ""),
                    "good_inventory_source": good.get("inventory_source", ""),
                    "bad_inventory_source": bad.get("inventory_source", ""),
                    "frame_continuation": good.get("frame_continuation", ""),
                    "whole_pair_object": good.get("whole_pair_object", ""),
                    "good_sentence": str(good["full_sentence"]),
                    "bad_sentence": str(bad["full_sentence"]),
                    "good_prompt": str(good["prompt"]),
                    "bad_prompt": str(bad["prompt"]),
                    "good_anchor_token_index": good.get("anchor_token_index", ""),
                    "bad_anchor_token_index": bad.get("anchor_token_index", ""),
                }
            )
    return pairs, dict(skip_counts)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(detail_rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        grouped[tuple(str(row.get(key, "")) for key in keys)].append(row)

    out: list[dict[str, Any]] = []
    for key_values, rows in sorted(grouped.items()):
        margins = [float(row["good_minus_bad_lp"]) for row in rows]
        out.append(
            {
                **dict(zip(keys, key_values)),
                "n": len(rows),
                "mean_good_minus_bad_lp": mean(margins),
                "success_rate": mean([1.0 if value > 0 else 0.0 for value in margins]),
                "min_good_minus_bad_lp": min(margins),
                "max_good_minus_bad_lp": max(margins),
            }
        )
    return out


def sequence_logprobs(model: Any, tokenizer: Any, sentences: list[str], device: Any) -> list[float]:
    import torch
    import torch.nn.functional as F

    inputs = tokenizer(sentences, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    outputs = model(**inputs, output_hidden_states=False, use_cache=False)
    logprobs = F.log_softmax(outputs.logits[:, :-1, :].float(), dim=-1)
    labels = inputs.input_ids[:, 1:]
    token_logprobs = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    mask = inputs.attention_mask[:, 1:].float()
    return (token_logprobs * mask).sum(dim=-1).detach().float().cpu().tolist()


def run(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    subtasks = set(parse_csv_arg(args.subtasks))
    regimes = set(parse_csv_arg(args.regimes))
    rows = load_rows(Path(args.data), subtasks, regimes)
    pairs, skip_counts = build_pairs(rows, args.max_pairs_per_group, args.seed)
    if not pairs:
        raise SystemExit("No valid whole-pair examples after filtering.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, local_files_only=not args.allow_download)
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    model.config.use_cache = False

    detail_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start : start + args.batch_size]
            good_lps = sequence_logprobs(model, tokenizer, [str(pair["good_sentence"]) for pair in batch], device)
            bad_lps = sequence_logprobs(model, tokenizer, [str(pair["bad_sentence"]) for pair in batch], device)
            for pair, good_lp, bad_lp in zip(batch, good_lps, bad_lps):
                margin = good_lp - bad_lp
                detail_rows.append(
                    {
                        **pair,
                        "good_sentence_logprob": good_lp,
                        "bad_sentence_logprob": bad_lp,
                        "good_minus_bad_lp": margin,
                        "good_preferred": int(margin > 0),
                    }
                )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_slug = args.run_name or shell_safe_slug(
        f"{args.model}_{'-'.join(sorted(subtasks))}_{'-'.join(sorted(regimes))}_whole_pair_lp"
    )
    detail_csv = out_dir / f"{run_slug}.detail.csv"
    summary_csv = out_dir / f"{run_slug}.summary.csv"
    source_csv = out_dir / f"{run_slug}.source_summary.csv"
    manifest_json = out_dir / f"{run_slug}.manifest.json"

    write_csv(detail_csv, detail_rows)
    summary_rows = summarize(detail_rows, ("regime", "subtask"))
    source_rows = summarize(
        detail_rows,
        ("regime", "subtask", "good_source_zipf_regime", "bad_source_zipf_regime", "good_inventory_source", "bad_inventory_source"),
    )
    write_csv(summary_csv, summary_rows)
    write_csv(source_csv, source_rows)

    manifest = {
        "model": args.model,
        "data": str(Path(args.data).resolve()),
        "subtasks": sorted(subtasks),
        "regimes": sorted(regimes),
        "rows_loaded": len(rows),
        "pairs_scored": len(detail_rows),
        "skip_counts": skip_counts,
        "detail_csv": str(detail_csv),
        "summary_csv": str(summary_csv),
        "source_summary_csv": str(source_csv),
        "overall": summarize(detail_rows, ("regime",)),
        "note": "Whole-pair behavioral gate: success means LP(full grammatical-frame sentence) > LP(full ungrammatical-frame sentence).",
    }
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/aligned_templates_v2/lexical_licensing_v2_aligned.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--subtasks", default=",".join(DEFAULT_SUBTASKS))
    ap.add_argument("--regimes", default=",".join(DEFAULT_REGIMES))
    ap.add_argument("--max-pairs-per-group", type=int, default=None)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--out-dir", default="results/whole_pair_lp")
    ap.add_argument("--run-name", default="")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
