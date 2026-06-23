#!/usr/bin/env python3
"""Confirm candidate residual-stream sites with exact activation patching."""

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
    DirectedPair,
    build_directed_pairs,
    load_aligned_rows,
    parse_csv_arg,
    shell_safe_slug,
    token_id,
)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def parse_site(site: str) -> int:
    if site == "resid_embed":
        return 0
    prefix = "resid_post_layer_"
    if not site.startswith(prefix):
        raise ValueError(f"Unsupported site name: {site}")
    return int(site[len(prefix) :]) + 1


def site_name(site_index: int) -> str:
    if site_index == 0:
        return "resid_embed"
    return f"resid_post_layer_{site_index - 1:02d}"


def sites_from_summary(summary_csv: Path, top_k: int) -> list[str]:
    rows: list[dict[str, str]] = []
    with summary_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows.sort(key=lambda r: -float(r["mean_abs_attribution"]))
    sites: list[str] = []
    seen: set[str] = set()
    for row in rows:
        site = row["site"]
        if site not in seen:
            seen.add(site)
            sites.append(site)
        if len(sites) >= top_k:
            break
    return sites


def normalize_model_output(output: Any) -> Any:
    if isinstance(output, tuple):
        return output[0]
    return output


def replace_model_output(output: Any, hidden: Any) -> Any:
    if isinstance(output, tuple):
        return (hidden,) + tuple(output[1:])
    return hidden


def write_summary(detail_rows: list[dict[str, Any]], summary_csv: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        key = (
            str(row["regime"]),
            str(row["subtask"]),
            str(row["direction"]),
            str(row["site"]),
            int(row["site_index"]),
        )
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (regime, subtask, direction, site, site_index), rows in sorted(grouped.items()):
        effects = [float(r["exact_effect"]) for r in rows]
        normalized = [float(r["normalized_effect"]) for r in rows if r["normalized_effect"] != ""]
        summary_rows.append(
            {
                "regime": regime,
                "subtask": subtask,
                "direction": direction,
                "site": site,
                "site_index": site_index,
                "n": len(rows),
                "mean_exact_effect": mean(effects),
                "mean_abs_exact_effect": mean([abs(v) for v in effects]),
                "mean_normalized_effect": mean(normalized),
                "mean_corrupt_metric": mean([float(r["corrupt_metric"]) for r in rows]),
                "mean_patched_metric": mean([float(r["patched_metric"]) for r in rows]),
                "mean_clean_metric": mean([float(r["clean_metric"]) for r in rows]),
            }
        )
    summary_rows.sort(key=lambda r: -float(r["mean_abs_exact_effect"]))

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)
    return summary_rows


def metric(logits: Any, clean_target: Any, corrupt_target: Any) -> Any:
    import torch
    import torch.nn.functional as F

    row_indices = torch.arange(logits.shape[0], device=logits.device)
    logprobs = F.log_softmax(logits[:, -1, :].float(), dim=-1)
    return logprobs[row_indices, clean_target] - logprobs[row_indices, corrupt_target]


def patch_forward(
    model: Any,
    corrupt_inputs: Any,
    site_index: int,
    anchor: int,
    clean_site: Any,
) -> Any:
    import torch

    row_indices = torch.arange(clean_site.shape[0], device=clean_site.device)

    def patch_tensor(hidden: Any) -> Any:
        patched = hidden.clone()
        patched[row_indices, anchor, :] = clean_site.to(dtype=patched.dtype)
        return patched

    if site_index == 0:
        handle = model.gpt_neox.embed_in.register_forward_hook(
            lambda _module, _inputs, output: patch_tensor(output)
        )
    else:
        layer_idx = site_index - 1

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = normalize_model_output(output)
            return replace_model_output(output, patch_tensor(hidden))

        handle = model.gpt_neox.layers[layer_idx].register_forward_hook(hook)
    try:
        return model(**corrupt_inputs, output_hidden_states=False, use_cache=False)
    finally:
        handle.remove()


def clean_forward_with_site(
    model: Any,
    clean_inputs: Any,
    site_index: int,
    anchor: int,
) -> tuple[Any, Any]:
    captured: dict[str, Any] = {}

    def capture_tensor(hidden: Any) -> None:
        captured["site"] = hidden[:, anchor, :].detach()

    if site_index == 0:
        handle = model.gpt_neox.embed_in.register_forward_hook(
            lambda _module, _inputs, output: capture_tensor(output)
        )
    else:
        layer_idx = site_index - 1

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            capture_tensor(normalize_model_output(output))

        handle = model.gpt_neox.layers[layer_idx].register_forward_hook(hook)
    try:
        outputs = model(**clean_inputs, output_hidden_states=False, use_cache=False)
    finally:
        handle.remove()
    return outputs, captured["site"]


def run(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    subtask_filter = set(parse_csv_arg(args.subtasks))
    regime_filter = set(parse_csv_arg(args.regimes))
    directions = set(parse_csv_arg(args.directions))
    rows = load_aligned_rows(Path(args.data), subtask_filter, regime_filter)
    pairs, skip_counts = build_directed_pairs(rows, directions, args.max_pairs_per_group, args.pairing, args.seed)
    if not pairs:
        raise SystemExit("No valid directed pairs after filtering.")

    if args.summary_csv:
        sites = sites_from_summary(Path(args.summary_csv), args.top_k)
    else:
        sites = list(parse_csv_arg(args.sites))
    if not sites:
        raise SystemExit("Provide --sites or --summary-csv.")
    site_indices = [parse_site(site) for site in sites]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_slug = args.run_name or shell_safe_slug(
        f"{args.model}_{'-'.join(sorted(subtask_filter))}_{'-'.join(sorted(regime_filter))}_exact"
    )
    detail_csv = out_dir / f"{run_slug}.detail.csv"
    summary_csv = out_dir / f"{run_slug}.summary.csv"
    manifest_json = out_dir / f"{run_slug}.manifest.json"

    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=not args.allow_download,
    )
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    model.config.use_cache = False

    target_ids = {target: token_id(tokenizer, target) for p in pairs for target in (p.clean_target, p.corrupt_target)}

    batches: dict[tuple[str, str, str, int, int], list[DirectedPair]] = defaultdict(list)
    for pair in pairs:
        batches[(pair.regime, pair.subtask, pair.direction, pair.prompt_token_count, pair.anchor_token_index)].append(pair)

    detail_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_key, batch_pairs in sorted(batches.items()):
            for start in range(0, len(batch_pairs), args.batch_size):
                batch = batch_pairs[start : start + args.batch_size]
                clean_inputs = tokenizer(
                    [p.clean_prompt for p in batch],
                    return_tensors="pt",
                    padding=False,
                    add_special_tokens=False,
                ).to(device)
                corrupt_inputs = tokenizer(
                    [p.corrupt_prompt for p in batch],
                    return_tensors="pt",
                    padding=False,
                    add_special_tokens=False,
                ).to(device)
                expected_len = batch[0].prompt_token_count
                if clean_inputs.input_ids.shape[1] != expected_len or corrupt_inputs.input_ids.shape[1] != expected_len:
                    raise RuntimeError(f"Tokenizer length mismatch in batch {batch_key}")

                clean_target = torch.tensor([target_ids[p.clean_target] for p in batch], device=device)
                corrupt_target = torch.tensor([target_ids[p.corrupt_target] for p in batch], device=device)
                anchor = batch[0].anchor_token_index

                clean_outputs = model(**clean_inputs, output_hidden_states=False, use_cache=False)
                corrupt_outputs = model(**corrupt_inputs, output_hidden_states=False, use_cache=False)
                clean_metric = metric(clean_outputs.logits, clean_target, corrupt_target)
                corrupt_metric = metric(corrupt_outputs.logits, clean_target, corrupt_target)

                for site_index in site_indices:
                    clean_outputs, clean_site = clean_forward_with_site(model, clean_inputs, site_index, anchor)
                    clean_metric = metric(clean_outputs.logits, clean_target, corrupt_target)
                    patched_outputs = patch_forward(model, corrupt_inputs, site_index, anchor, clean_site)
                    patched_metric = metric(patched_outputs.logits, clean_target, corrupt_target)
                    exact_effect = patched_metric - corrupt_metric
                    denom = clean_metric - corrupt_metric

                    for pair, patch_m, corrupt_m, clean_m, effect, gap in zip(
                        batch,
                        patched_metric.float().cpu().tolist(),
                        corrupt_metric.float().cpu().tolist(),
                        clean_metric.float().cpu().tolist(),
                        exact_effect.float().cpu().tolist(),
                        denom.float().cpu().tolist(),
                    ):
                        normalized = "" if abs(gap) < 1e-8 else effect / gap
                        detail_rows.append(
                            {
                                "model": args.model,
                                "pair_key": pair.pair_key,
                                "regime": pair.regime,
                                "subtask": pair.subtask,
                                "template_id": pair.template_id,
                                "context_id": pair.context_id,
                                "subject": pair.subject,
                                "subject_class": pair.subject_class,
                                "source_idx": pair.source_idx,
                                "direction": pair.direction,
                                "clean_side": pair.clean_side,
                                "corrupt_side": pair.corrupt_side,
                                "clean_verb": pair.clean_verb,
                                "corrupt_verb": pair.corrupt_verb,
                                "clean_source_zipf_regime": pair.clean_source_zipf_regime,
                                "corrupt_source_zipf_regime": pair.corrupt_source_zipf_regime,
                                "clean_inventory_source": pair.clean_inventory_source,
                                "corrupt_inventory_source": pair.corrupt_inventory_source,
                                "clean_target": pair.clean_target,
                                "corrupt_target": pair.corrupt_target,
                                "anchor_token_index": pair.anchor_token_index,
                                "site": site_name(site_index),
                                "site_index": site_index,
                                "corrupt_metric": corrupt_m,
                                "patched_metric": patch_m,
                                "clean_metric": clean_m,
                                "exact_effect": effect,
                                "normalized_effect": normalized,
                            }
                        )

                if device.type == "cuda":
                    torch.cuda.empty_cache()

    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(detail_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)
    summary_rows = write_summary(detail_rows, summary_csv)
    manifest = {
        "model": args.model,
        "data": str(Path(args.data).resolve()),
        "run_name": run_slug,
        "subtasks": sorted(subtask_filter),
        "regimes": sorted(regime_filter),
        "directions": sorted(directions),
        "pairing": args.pairing,
        "sites": [site_name(i) for i in site_indices],
        "rows_loaded": len(rows),
        "directed_pairs": len(pairs),
        "skip_counts": skip_counts,
        "detail_csv": str(detail_csv),
        "summary_csv": str(summary_csv),
        "top_sites_by_abs_exact_effect": summary_rows[:20],
    }
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/aligned_templates/lexical_licensing_aligned.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--subtasks", default=",".join(DEFAULT_SUBTASKS))
    ap.add_argument("--regimes", default=",".join(DEFAULT_REGIMES))
    ap.add_argument("--directions", default="good_to_bad,bad_to_good")
    ap.add_argument("--pairing", choices=("source_idx", "balanced_pool"), default="source_idx")
    ap.add_argument("--max-pairs-per-group", type=int, default=None)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--sites", default="")
    ap.add_argument("--summary-csv", default="")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--out-dir", default="results/exact_patching")
    ap.add_argument("--run-name", default="")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
