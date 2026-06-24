#!/usr/bin/env python3
"""Baseline log-odds sanity checks for v2 lexical-licensing rows.

This is a diagnostic, not a hard filter. It measures whether the base model
already prefers each row's expected continuation over its contrast continuation:

    log p(expected_target | prompt) - log p(contrast_target | prompt)

It can also score a fixed unrelated target pair over the same prompts, for
example:

    log p(" red" | prompt) - log p(" blue" | prompt)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from run_pythia_attribution_patching import DEFAULT_MODEL, DEFAULT_REGIMES, DEFAULT_SUBTASKS, iter_jsonl, parse_csv_arg, shell_safe_slug, token_id


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
        rows.append(row)
    return rows


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
        logodds = [float(row["expected_logodds"]) for row in rows]
        out.append(
            {
                **dict(zip(keys, key_values)),
                "n": len(rows),
                "mean_expected_logodds": mean(logodds),
                "success_rate": mean([1.0 if value > 0 else 0.0 for value in logodds]),
                "min_expected_logodds": min(logodds),
                "max_expected_logodds": max(logodds),
            }
        )
    return out


def target_pair_for_row(row: dict[str, Any], args: argparse.Namespace) -> tuple[str, str]:
    if args.target_mode == "original":
        return str(row["expected_target"]), str(row["contrast_target"])
    if args.target_mode == "fixed_pair":
        return args.fixed_expected_target, args.fixed_contrast_target
    raise ValueError(f"unknown target mode: {args.target_mode}")


def run(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    subtasks = set(parse_csv_arg(args.subtasks))
    regimes = set(parse_csv_arg(args.regimes))
    rows = load_rows(Path(args.data), subtasks, regimes)
    if not rows:
        raise SystemExit("No aligned rows after filtering.")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, local_files_only=not args.allow_download)
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    model.config.use_cache = False

    row_targets = [target_pair_for_row(row, args) for row in rows]
    targets = sorted({target for pair in row_targets for target in pair})
    target_ids = {target: token_id(tokenizer, target) for target in targets}

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["prompt_token_count"])].append(row)

    detail_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for prompt_len, group_rows in sorted(grouped.items()):
            for start in range(0, len(group_rows), args.batch_size):
                batch = group_rows[start : start + args.batch_size]
                inputs = tokenizer(
                    [str(row["prompt"]) for row in batch],
                    return_tensors="pt",
                    padding=False,
                    add_special_tokens=False,
                ).to(device)
                if inputs.input_ids.shape[1] != prompt_len:
                    raise RuntimeError(f"Tokenizer length mismatch for prompt_len={prompt_len}")
                outputs = model(**inputs, output_hidden_states=False, use_cache=False)
                logprobs = F.log_softmax(outputs.logits[:, -1, :].float(), dim=-1)
                row_indices = torch.arange(len(batch), device=device)
                expected = torch.tensor([target_ids[target_pair_for_row(row, args)[0]] for row in batch], device=device)
                contrast = torch.tensor([target_ids[target_pair_for_row(row, args)[1]] for row in batch], device=device)
                logodds = logprobs[row_indices, expected] - logprobs[row_indices, contrast]
                expected_lp = logprobs[row_indices, expected]
                contrast_lp = logprobs[row_indices, contrast]

                for row, odds, exp_lp, con_lp in zip(
                    batch,
                    logodds.float().cpu().tolist(),
                    expected_lp.float().cpu().tolist(),
                    contrast_lp.float().cpu().tolist(),
                ):
                    detail_rows.append(
                        {
                            "row_id": row.get("row_id", ""),
                            "regime": row.get("regime", ""),
                            "subtask": row.get("subtask", ""),
                            "side": row.get("side", ""),
                            "verb": row.get("verb", ""),
                            "subject": row.get("subject", ""),
                            "subject_class": row.get("subject_class", ""),
                            "context_schema_id": row.get("context_schema_id", ""),
                            "source_zipf": row.get("source_zipf", ""),
                            "source_zipf_regime": row.get("source_zipf_regime", ""),
                            "inventory_source": row.get("inventory_source", ""),
                            "target_mode": args.target_mode,
                            "expected_target": target_pair_for_row(row, args)[0],
                            "contrast_target": target_pair_for_row(row, args)[1],
                            "original_expected_target": row.get("expected_target", ""),
                            "original_contrast_target": row.get("contrast_target", ""),
                            "expected_logprob": exp_lp,
                            "contrast_logprob": con_lp,
                            "expected_logodds": odds,
                            "expected_preferred": int(odds > 0),
                        }
                    )
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_slug = args.run_name or shell_safe_slug(f"{args.model}_{'-'.join(sorted(subtasks))}_{'-'.join(sorted(regimes))}_sanity")
    detail_csv = out_dir / f"{run_slug}.detail.csv"
    summary_csv = out_dir / f"{run_slug}.summary.csv"
    subject_csv = out_dir / f"{run_slug}.subject_summary.csv"
    source_csv = out_dir / f"{run_slug}.source_summary.csv"
    manifest_json = out_dir / f"{run_slug}.manifest.json"

    write_csv(detail_csv, detail_rows)
    summary_rows = summarize(detail_rows, ("regime", "subtask", "side"))
    subject_rows = summarize(detail_rows, ("regime", "subtask", "side", "subject_class"))
    source_rows = summarize(detail_rows, ("regime", "subtask", "side", "source_zipf_regime", "inventory_source"))
    write_csv(summary_csv, summary_rows)
    write_csv(subject_csv, subject_rows)
    write_csv(source_csv, source_rows)

    manifest = {
        "model": args.model,
        "data": str(Path(args.data).resolve()),
        "subtasks": sorted(subtasks),
        "regimes": sorted(regimes),
        "target_mode": args.target_mode,
        "target_ids": target_ids,
        "fixed_expected_target": args.fixed_expected_target if args.target_mode == "fixed_pair" else "",
        "fixed_contrast_target": args.fixed_contrast_target if args.target_mode == "fixed_pair" else "",
        "rows_loaded": len(rows),
        "detail_csv": str(detail_csv),
        "summary_csv": str(summary_csv),
        "subject_summary_csv": str(subject_csv),
        "source_summary_csv": str(source_csv),
        "overall": summarize(detail_rows, ("regime",)),
        "note": (
            "Baseline diagnostic only. Low scores identify cells/items to inspect; they are not automatically excluded. "
            "For target_mode=fixed_pair, prompts are unchanged and original expected/contrast targets are ignored."
        ),
    }
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/aligned_templates_v2/lexical_licensing_v2_aligned.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--subtasks", default=",".join(DEFAULT_SUBTASKS))
    ap.add_argument("--regimes", default=",".join(DEFAULT_REGIMES))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--out-dir", default="results/v2_sanity")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--target-mode", choices=("original", "fixed_pair"), default="original")
    ap.add_argument("--fixed-expected-target", default=" red")
    ap.add_argument("--fixed-contrast-target", default=" blue")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
