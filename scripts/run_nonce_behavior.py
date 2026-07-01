#!/usr/bin/env python3
"""Score object-vs-period continuation probability after nonce probes."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from run_pythia_attribution_patching import DEFAULT_MODEL, shell_safe_slug, token_id


DEFAULT_DATA = "data/nonce_frames/nonce_frames.jsonl"
DEFAULT_OUT_DIR = "results/nonce_frames"
DEFAULT_OBJECT_STARTS = (
    " the",
    " a",
    " an",
    " this",
    " that",
    " my",
    " his",
    " her",
    " their",
    " its",
    " our",
    " your",
    " it",
    " them",
)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def sem(values: list[float]) -> float:
    return stdev(values) / math.sqrt(len(values)) if len(values) >= 2 else math.nan


def parse_object_starts(value: str) -> tuple[str, ...]:
    if not value:
        return DEFAULT_OBJECT_STARTS
    targets: list[str] = []
    for part in value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        if stripped[0].isalnum():
            stripped = f" {stripped}"
        targets.append(stripped)
    return tuple(targets)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [row for row in iter_jsonl(path) if row.get("alignment_status", "aligned") == "aligned"]
    if not rows:
        raise SystemExit(f"No aligned rows loaded from {path}")
    frames = {str(row.get("frame", "")) for row in rows}
    if not {"T+", "T-"} <= frames:
        raise SystemExit(f"Expected both T+ and T- frames, got {sorted(frames)}")
    return rows


def continuation_ids(tokenizer: Any, object_starts: tuple[str, ...], period: str) -> tuple[list[int], int, list[str]]:
    object_ids: list[int] = []
    kept_object_starts: list[str] = []
    for target in object_starts:
        try:
            target_id = token_id(tokenizer, target)
        except ValueError:
            continue
        if target_id not in object_ids:
            object_ids.append(target_id)
            kept_object_starts.append(target)
    if not object_ids:
        raise SystemExit("No object-start targets are single tokens for this tokenizer.")
    return object_ids, token_id(tokenizer, period), kept_object_starts


def score_batch(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    object_ids: list[int],
    period_id: int,
    device: Any,
) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    prompts = [str(row["full_context"]) for row in rows]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    outputs = model(**inputs, output_hidden_states=False, use_cache=False)
    last_indices = inputs.attention_mask.sum(dim=-1) - 1
    batch_indices = torch.arange(inputs.input_ids.shape[0], device=device)
    logits = outputs.logits[batch_indices, last_indices, :].float()
    probs = F.softmax(logits, dim=-1)
    object_mass = probs[:, object_ids].sum(dim=-1)
    period_mass = probs[:, period_id]
    denom = object_mass + period_mass
    object_vs_period = object_mass / denom.clamp_min(1e-30)

    out: list[dict[str, Any]] = []
    for row, obj, per, obj_v_per, last_idx in zip(
        rows,
        object_mass.detach().float().cpu().tolist(),
        period_mass.detach().float().cpu().tolist(),
        object_vs_period.detach().float().cpu().tolist(),
        last_indices.detach().cpu().tolist(),
    ):
        item = dict(row)
        item["scored_token_index"] = int(last_idx)
        item["p_object_raw"] = obj
        item["p_period"] = per
        item["p_object_vs_period"] = obj_v_per
        out.append(item)
    return out


def summarize_by_lemma(detail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in detail_rows:
        grouped[str(row["lemma"])][str(row["frame"])].append(row)

    out: list[dict[str, Any]] = []
    for lemma, by_frame in sorted(grouped.items()):
        plus = by_frame.get("T+", [])
        minus = by_frame.get("T-", [])
        if not plus or not minus:
            continue
        plus_values = [float(row["p_object_vs_period"]) for row in plus]
        minus_values = [float(row["p_object_vs_period"]) for row in minus]
        out.append(
            {
                "lemma": lemma,
                "n_t_plus": len(plus_values),
                "n_t_minus": len(minus_values),
                "mean_p_object_t_plus": mean(plus_values),
                "mean_p_object_t_minus": mean(minus_values),
                "delta_p_object": mean(plus_values) - mean(minus_values),
                "mean_raw_object_t_plus": mean([float(row["p_object_raw"]) for row in plus]),
                "mean_raw_object_t_minus": mean([float(row["p_object_raw"]) for row in minus]),
                "mean_period_t_plus": mean([float(row["p_period"]) for row in plus]),
                "mean_period_t_minus": mean([float(row["p_period"]) for row in minus]),
                "probe_form_token_count": plus[0].get("probe_form_token_count", ""),
                "probe_form_token_pieces": plus[0].get("probe_form_token_pieces", ""),
                "probe_subject": plus[0].get("probe_subject", ""),
                "probe_subject_animacy": plus[0].get("probe_subject_animacy", ""),
            }
        )
    return out


def summarize_pooled(lemma_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deltas = [float(row["delta_p_object"]) for row in lemma_rows]
    return [
        {
            "n_lemmas": len(lemma_rows),
            "mean_delta_p_object": mean(deltas),
            "sem_delta_p_object": sem(deltas),
            "median_delta_p_object": sorted(deltas)[len(deltas) // 2] if deltas else math.nan,
            "frac_delta_positive": mean([1.0 if value > 0 else 0.0 for value in deltas]),
            "min_delta_p_object": min(deltas) if deltas else math.nan,
            "max_delta_p_object": max(deltas) if deltas else math.nan,
        }
    ]


def run(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_rows(Path(args.data))
    object_starts = parse_object_starts(args.object_starts)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    object_ids, period_id, kept_object_starts = continuation_ids(tokenizer, object_starts, args.period_target)

    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, local_files_only=not args.allow_download)
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    model.config.use_cache = False

    detail_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            detail_rows.extend(score_batch(model, tokenizer, batch, object_ids, period_id, device))
            if device.type == "cuda":
                torch.cuda.empty_cache()

    lemma_rows = summarize_by_lemma(detail_rows)
    pooled_rows = summarize_pooled(lemma_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_slug = args.run_name or shell_safe_slug(f"{args.model}_nonce_behavior")
    detail_csv = out_dir / f"{run_slug}.behavior_detail.csv"
    by_lemma_csv = out_dir / f"{run_slug}.behavior_by_lemma.csv"
    summary_csv = out_dir / f"{run_slug}.behavior_summary.csv"
    manifest_json = out_dir / f"{run_slug}.behavior_manifest.json"
    write_csv(detail_csv, detail_rows)
    write_csv(by_lemma_csv, lemma_rows)
    write_csv(summary_csv, pooled_rows)

    manifest = {
        "model": args.model,
        "data": str(Path(args.data)),
        "rows_scored": len(detail_rows),
        "lemmas_scored": len(lemma_rows),
        "object_starts": kept_object_starts,
        "object_token_ids": object_ids,
        "period_target": args.period_target,
        "period_token_id": period_id,
        "detail_csv": str(detail_csv),
        "by_lemma_csv": str(by_lemma_csv),
        "summary_csv": str(summary_csv),
        "pooled": pooled_rows[0] if pooled_rows else {},
        "note": "Behavioral gate compares object-start continuation mass against period only; EOS is not included.",
    }
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--object-starts", default="")
    ap.add_argument("--period-target", default=".")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--run-name", default="")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
