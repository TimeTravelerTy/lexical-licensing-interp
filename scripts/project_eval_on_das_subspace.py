#!/usr/bin/env python3
"""Project eval verb-context activations onto a saved DAS subspace.

The DAS intervention only depends on the projection matrix, so the sign of a
rank-1 coordinate is arbitrary. This diagnostic therefore reports both signed
coordinates and orientation-free separability metrics.

This is a d-validation diagnostic, not evidence for storage-vs-computation.
Real-verb projection separation is compatible with both a stored lexical lookup
and an online licensing computation. The load-bearing validation cell is whether
the learned direction treats inchoative-good examples like licensed examples,
instead of flipping with the next-token object/no-object target.
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
    build_directed_pairs,
    load_aligned_rows,
    parse_csv_arg,
    shell_safe_slug,
)
from run_pythia_das_v1 import (
    apply_control,
    make_transfer_splits,
    retokenize_pairs,
)
from run_pythia_exact_patching import normalize_model_output, parse_site


DEFAULT_DATA = "data/aligned_templates_v2/lexical_licensing_v2_aligned.jsonl"
DEFAULT_SUBSPACE = (
    "results/das_v2_transfer/"
    "20260623-pythia14b-v2-das-head-to-low-l23-none-s17-verb_to_subject_anchor.subspace.pt"
)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def sem(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    return stdev(values) / math.sqrt(len(values))


def binary_auc(values: list[float], labels: list[str], positive_label: str) -> float:
    pos = [v for v, label in zip(values, labels) if label == positive_label]
    neg = [v for v, label in zip(values, labels) if label != positive_label]
    if not pos or not neg:
        return math.nan
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def best_threshold_accuracy(values: list[float], labels: list[str], positive_label: str) -> tuple[float, float, str]:
    if not values:
        return math.nan, math.nan, ""
    unique_values = sorted(set(values))
    if len(unique_values) == 1:
        thresholds = [unique_values[0]]
    else:
        thresholds = [unique_values[0] - 1.0]
        thresholds.extend((a + b) / 2.0 for a, b in zip(unique_values, unique_values[1:]))
        thresholds.append(unique_values[-1] + 1.0)

    best_acc = -1.0
    best_threshold = thresholds[0]
    best_direction = "positive_high"
    total = len(values)
    for threshold in thresholds:
        for direction in ("positive_high", "positive_low"):
            correct = 0
            for value, label in zip(values, labels):
                if direction == "positive_high":
                    pred = positive_label if value >= threshold else "__negative__"
                else:
                    pred = positive_label if value <= threshold else "__negative__"
                actual = positive_label if label == positive_label else "__negative__"
                correct += int(pred == actual)
            acc = correct / total
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
                best_direction = direction
    return best_acc, best_threshold, best_direction


def orthonormal_basis(raw: Any) -> Any:
    import torch

    q, _r = torch.linalg.qr(raw.float(), mode="reduced")
    return q


def load_basis(path: Path, expected_site: str, expected_rank: int | None, device: Any) -> Any:
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    raw = checkpoint["raw"] if isinstance(checkpoint, dict) and "raw" in checkpoint else checkpoint
    if raw.ndim == 1:
        raw = raw[:, None]
    if isinstance(checkpoint, dict) and checkpoint.get("site") not in {None, expected_site}:
        raise SystemExit(f"Subspace site mismatch: checkpoint={checkpoint.get('site')} --site={expected_site}")
    if expected_rank is not None and raw.shape[1] != expected_rank:
        raise SystemExit(f"Subspace rank mismatch: checkpoint={raw.shape[1]} --rank={expected_rank}")
    return orthonormal_basis(raw).to(device)


def capture_site(model: Any, inputs: Any, site_index: int, anchor: int) -> Any:
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
        model(**inputs, output_hidden_states=False, use_cache=False)
    finally:
        handle.remove()
    return captured["site"]


def make_projection_records(pairs: list[Any], split_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    pair_rows: list[dict[str, Any]] = []

    def add_record(pair: Any, role: str) -> tuple[Any, ...]:
        side = getattr(pair, f"{role}_side")
        verb = getattr(pair, f"{role}_verb")
        prompt = getattr(pair, f"{role}_prompt")
        expected_target = getattr(pair, f"{role}_target")
        source_zipf_regime = getattr(pair, f"{role}_source_zipf_regime", "")
        inventory_source = getattr(pair, f"{role}_inventory_source", "")
        key = (
            split_name,
            pair.regime,
            pair.subtask,
            pair.template_id,
            pair.context_id,
            side,
            verb,
            prompt,
        )
        records_by_key.setdefault(
            key,
            {
                "split": split_name,
                "regime": pair.regime,
                "subtask": pair.subtask,
                "template_id": pair.template_id,
                "context_id": pair.context_id,
                "subject": pair.subject,
                "subject_class": pair.subject_class,
                "side": side,
                "expected_target": expected_target,
                "licensing_status": "licensed" if side == "good" else "unlicensed",
                "object_target_status": "object_expected" if expected_target == " the" else "no_object_expected",
                "verb": verb,
                "prompt": prompt,
                "source_zipf_regime": source_zipf_regime,
                "inventory_source": inventory_source,
                "prompt_token_count": pair.prompt_token_count,
                "anchor_token_index": pair.anchor_token_index,
            },
        )
        return key

    for pair in pairs:
        clean_key = add_record(pair, "clean")
        corrupt_key = add_record(pair, "corrupt")
        pair_rows.append(
            {
                "split": split_name,
                "pair_key": pair.pair_key,
                "regime": pair.regime,
                "subtask": pair.subtask,
                "template_id": pair.template_id,
                "context_id": pair.context_id,
                "subject": pair.subject,
                "subject_class": pair.subject_class,
                "direction": pair.direction,
                "clean_side": pair.clean_side,
                "corrupt_side": pair.corrupt_side,
                "clean_verb": pair.clean_verb,
                "corrupt_verb": pair.corrupt_verb,
                "clean_record_key": clean_key,
                "corrupt_record_key": corrupt_key,
            }
        )
    return list(records_by_key.values()), pair_rows


def project_records(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    basis: Any,
    site_index: int,
    batch_size: int,
    device: Any,
) -> list[dict[str, Any]]:
    import torch

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(int(record["prompt_token_count"]), int(record["anchor_token_index"]))].append(record)

    projected: list[dict[str, Any]] = []
    with torch.no_grad():
        for (_prompt_len, anchor), group in sorted(grouped.items()):
            for start in range(0, len(group), batch_size):
                batch = group[start : start + batch_size]
                inputs = tokenizer(
                    [r["prompt"] for r in batch],
                    return_tensors="pt",
                    padding=False,
                    add_special_tokens=False,
                ).to(device)
                hidden = capture_site(model, inputs, site_index, anchor)
                coords = hidden.float() @ basis.float()
                projected_hidden = coords @ basis.float().T
                projection_norm = projected_hidden.norm(dim=-1)
                residual_norm = hidden.float().norm(dim=-1)
                for row, coord, proj_n, resid_n in zip(
                    batch,
                    coords.detach().float().cpu().tolist(),
                    projection_norm.detach().float().cpu().tolist(),
                    residual_norm.detach().float().cpu().tolist(),
                ):
                    out = dict(row)
                    if not isinstance(coord, list):
                        coord = [float(coord)]
                    for idx, value in enumerate(coord):
                        out[f"coord_{idx}"] = value
                    out["projection_norm"] = proj_n
                    out["residual_norm"] = resid_n
                    out["projection_norm_frac"] = "" if resid_n == 0 else proj_n / resid_n
                    projected.append(out)
    return projected


def summarize_labels(
    detail_rows: list[dict[str, Any]],
    label_columns: list[str],
    split_filter: str,
) -> list[dict[str, Any]]:
    rows = [r for r in detail_rows if split_filter == "all" or r["split"] == split_filter]
    summaries: list[dict[str, Any]] = []
    group_cols_by_label = {
        "side": ("split", "regime", "subtask"),
        "licensing_status": ("split", "regime", "subtask"),
        "expected_target": ("split", "regime", "subtask"),
        "object_target_status": ("split", "regime", "subtask"),
        "subtask": ("split", "regime"),
        "source_zipf_regime": ("split", "regime", "subtask"),
        "inventory_source": ("split", "regime", "subtask"),
        "subject_class": ("split", "regime", "subtask"),
    }
    for label_col in label_columns:
        group_cols = group_cols_by_label.get(label_col, ("split", "regime"))
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get(label_col, "") != "":
                grouped[tuple(row[col] for col in group_cols)].append(row)
        for group_key, group_rows in sorted(grouped.items()):
            labels = sorted({str(r[label_col]) for r in group_rows})
            if len(labels) != 2:
                continue
            values = [float(r["coord_0"]) for r in group_rows]
            row_labels = [str(r[label_col]) for r in group_rows]
            label_a, label_b = labels
            values_a = [v for v, label in zip(values, row_labels) if label == label_a]
            values_b = [v for v, label in zip(values, row_labels) if label == label_b]
            auc = binary_auc(values, row_labels, label_b)
            best_acc, threshold, direction = best_threshold_accuracy(values, row_labels, label_b)
            summaries.append(
                {
                    "label_column": label_col,
                    "group_columns": "|".join(group_cols),
                    "group_key": "|".join(str(v) for v in group_key),
                    "label_a": label_a,
                    "label_b": label_b,
                    "n_a": len(values_a),
                    "n_b": len(values_b),
                    "mean_coord_a": mean(values_a),
                    "mean_coord_b": mean(values_b),
                    "mean_diff_b_minus_a": mean(values_b) - mean(values_a),
                    "sem_a": sem(values_a),
                    "sem_b": sem(values_b),
                    "auc_b_gt_a": auc,
                    "orientation_free_auc": max(auc, 1.0 - auc),
                    "best_threshold_acc": best_acc,
                    "best_threshold": threshold,
                    "best_threshold_direction_for_label_b": direction,
                }
            )
    return summaries


def coord_values(rows: list[dict[str, Any]], **filters: str) -> list[float]:
    return [
        float(row["coord_0"])
        for row in rows
        if all(str(row.get(key, "")) == value for key, value in filters.items())
    ]


def summarize_validation_cells(detail_rows: list[dict[str, Any]], split_filter: str) -> list[dict[str, Any]]:
    """Summarize the causative-vs-inchoative alignment test.

    The rank-1 sign is arbitrary, so each row is internally calibrated by the
    causative cell. If inchoative-good is licensed in the same direction as
    causative-good, the calibrated inchoative margin is positive. If the axis is
    mostly the object/no-object next-token proxy, this margin tends to flip.
    """

    rows = [r for r in detail_rows if split_filter == "all" or r["split"] == split_filter]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["regime"]))].append(row)

    summaries: list[dict[str, Any]] = []
    for (split, regime), group_rows in sorted(grouped.items()):
        caus_good = coord_values(group_rows, subtask="causative", side="good")
        caus_bad = coord_values(group_rows, subtask="causative", side="bad")
        inch_good = coord_values(group_rows, subtask="inchoative", side="good")
        inch_bad = coord_values(group_rows, subtask="inchoative", side="bad")
        if not all([caus_good, caus_bad, inch_good, inch_bad]):
            continue

        caus_margin = mean(caus_good) - mean(caus_bad)
        orientation = 1.0 if caus_margin >= 0 else -1.0
        inch_margin = mean(inch_good) - mean(inch_bad)
        calibrated_inch_margin = orientation * inch_margin
        target_proxy_margin = mean(
            coord_values(group_rows, object_target_status="object_expected")
        ) - mean(coord_values(group_rows, object_target_status="no_object_expected"))
        calibrated_target_proxy_margin = orientation * target_proxy_margin

        inchoative_values = inch_good + inch_bad
        inchoative_labels = ["good"] * len(inch_good) + ["bad"] * len(inch_bad)
        inch_auc_good_high = binary_auc(inchoative_values, inchoative_labels, "good")
        summaries.append(
            {
                "split": split,
                "regime": regime,
                "cell": "causative_calibrated_inchoative_good_vs_bad",
                "n_causative_good": len(caus_good),
                "n_causative_bad": len(caus_bad),
                "n_inchoative_good": len(inch_good),
                "n_inchoative_bad": len(inch_bad),
                "causative_good_minus_bad": caus_margin,
                "inchoative_good_minus_bad": inch_margin,
                "calibrated_inchoative_good_minus_bad": calibrated_inch_margin,
                "licensing_axis_pass": int(calibrated_inch_margin > 0),
                "object_expected_minus_no_object_expected": target_proxy_margin,
                "calibrated_object_expected_minus_no_object_expected": calibrated_target_proxy_margin,
                "inchoative_auc_good_gt_bad": inch_auc_good_high,
                "inchoative_orientation_free_auc": max(inch_auc_good_high, 1.0 - inch_auc_good_high),
            }
        )

        by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group_rows:
            by_context[str(row["context_id"])].append(row)
        context_margins: list[float] = []
        for context_rows in by_context.values():
            cg = coord_values(context_rows, subtask="causative", side="good")
            cb = coord_values(context_rows, subtask="causative", side="bad")
            ig = coord_values(context_rows, subtask="inchoative", side="good")
            ib = coord_values(context_rows, subtask="inchoative", side="bad")
            if not all([cg, cb, ig, ib]):
                continue
            context_orientation = 1.0 if mean(cg) - mean(cb) >= 0 else -1.0
            context_margins.append(context_orientation * (mean(ig) - mean(ib)))
        if context_margins:
            summaries.append(
                {
                    "split": split,
                    "regime": regime,
                    "cell": "contextwise_calibrated_inchoative_good_vs_bad",
                    "n_causative_good": "",
                    "n_causative_bad": "",
                    "n_inchoative_good": "",
                    "n_inchoative_bad": "",
                    "causative_good_minus_bad": "",
                    "inchoative_good_minus_bad": "",
                    "calibrated_inchoative_good_minus_bad": mean(context_margins),
                    "licensing_axis_pass": int(mean(context_margins) > 0),
                    "object_expected_minus_no_object_expected": "",
                    "calibrated_object_expected_minus_no_object_expected": "",
                    "inchoative_auc_good_gt_bad": "",
                    "inchoative_orientation_free_auc": "",
                    "n_contexts": len(context_margins),
                    "frac_contexts_licensing_aligned": mean([1.0 if v > 0 else 0.0 for v in context_margins]),
                }
            )
    return summaries


def summarize_pairs(pair_rows: list[dict[str, Any]], coord_by_key: dict[tuple[Any, ...], float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail: list[dict[str, Any]] = []
    for row in pair_rows:
        clean_coord = coord_by_key[row["clean_record_key"]]
        corrupt_coord = coord_by_key[row["corrupt_record_key"]]
        out = {k: v for k, v in row.items() if not k.endswith("_record_key")}
        out["clean_coord_0"] = clean_coord
        out["corrupt_coord_0"] = corrupt_coord
        out["clean_minus_corrupt_coord_0"] = clean_coord - corrupt_coord
        out["abs_clean_minus_corrupt_coord_0"] = abs(clean_coord - corrupt_coord)
        detail.append(out)

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detail:
        grouped[(row["split"], row["regime"], row["subtask"], row["direction"])].append(row)
    summary: list[dict[str, Any]] = []
    for (split, regime, subtask, direction), rows in sorted(grouped.items()):
        deltas = [float(r["clean_minus_corrupt_coord_0"]) for r in rows]
        summary.append(
            {
                "split": split,
                "regime": regime,
                "subtask": subtask,
                "direction": direction,
                "n": len(rows),
                "mean_clean_minus_corrupt_coord_0": mean(deltas),
                "mean_abs_clean_minus_corrupt_coord_0": mean([abs(v) for v in deltas]),
                "sem_clean_minus_corrupt_coord_0": sem(deltas),
                "frac_delta_positive": mean([1.0 if v > 0 else 0.0 for v in deltas]),
            }
        )
    return detail, summary


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
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tuple_key_from_record(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["split"],
        row["regime"],
        row["subtask"],
        row["template_id"],
        row["context_id"],
        row["side"],
        row["verb"],
        row["prompt"],
    )


def run(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    subtask_filter = set(parse_csv_arg(args.subtasks))
    train_regimes = set(parse_csv_arg(args.train_regimes))
    eval_regimes = set(parse_csv_arg(args.eval_regimes))
    regime_filter = train_regimes | eval_regimes
    directions = set(parse_csv_arg(args.directions))
    site_index = parse_site(args.site)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_aligned_rows(Path(args.data), subtask_filter, regime_filter)
    pairs, skip_counts = build_directed_pairs(rows, directions, args.max_pairs_per_group, args.pairing, args.seed)
    pairs = [apply_control(pair, args.control, args.dummy_verb, args.seed) for pair in pairs]
    pairs = retokenize_pairs(tokenizer, pairs)
    train_pairs, eval_pairs = make_transfer_splits(
        pairs,
        train_regimes,
        eval_regimes,
        args.eval_frac,
        args.split_by,
        args.seed,
    )
    split_pairs = {"train": train_pairs, "eval": eval_pairs}
    selected_pairs = []
    for split_name, split_list in split_pairs.items():
        if args.project_split in {split_name, "all"}:
            selected_pairs.extend((split_name, pair) for pair in split_list)
    if not selected_pairs:
        raise SystemExit(f"No pairs selected for --project-split {args.project_split}")

    all_records: list[dict[str, Any]] = []
    all_pair_rows: list[dict[str, Any]] = []
    for split_name in ("train", "eval"):
        split_list = [pair for s, pair in selected_pairs if s == split_name]
        records, pair_rows = make_projection_records(split_list, split_name)
        all_records.extend(records)
        all_pair_rows.extend(pair_rows)

    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=not args.allow_download,
    )
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad_(False)

    basis = load_basis(Path(args.subspace), args.site, args.rank, device)
    if basis.shape[0] != model.config.hidden_size:
        raise SystemExit(f"Subspace d_model mismatch: basis={basis.shape[0]} model={model.config.hidden_size}")

    detail_rows = project_records(model, tokenizer, all_records, basis, site_index, args.batch_size, device)
    coord_by_key = {tuple_key_from_record(row): float(row["coord_0"]) for row in detail_rows}
    pair_detail, pair_summary = summarize_pairs(all_pair_rows, coord_by_key)
    label_summary = summarize_labels(detail_rows, list(parse_csv_arg(args.label_columns)), args.project_split)
    validation_summary = summarize_validation_cells(detail_rows, args.project_split)

    out_dir = Path(args.out_dir)
    run_slug = args.run_name or shell_safe_slug(f"{args.model}_{args.site}_das_projection")
    detail_csv = out_dir / f"{run_slug}.projection_detail.csv"
    validation_summary_csv = out_dir / f"{run_slug}.validation_cell_summary.csv"
    label_summary_csv = out_dir / f"{run_slug}.label_separation_summary.csv"
    pair_detail_csv = out_dir / f"{run_slug}.pair_delta_detail.csv"
    pair_summary_csv = out_dir / f"{run_slug}.pair_delta_summary.csv"
    manifest_json = out_dir / f"{run_slug}.manifest.json"

    write_csv(detail_csv, detail_rows)
    write_csv(validation_summary_csv, validation_summary)
    write_csv(label_summary_csv, label_summary)
    write_csv(pair_detail_csv, pair_detail)
    write_csv(pair_summary_csv, pair_summary)
    manifest = {
        "model": args.model,
        "data": str(Path(args.data)),
        "subspace": str(Path(args.subspace)),
        "site": args.site,
        "rank": int(basis.shape[1]),
        "subtasks": sorted(subtask_filter),
        "train_regimes": sorted(train_regimes),
        "eval_regimes": sorted(eval_regimes),
        "project_split": args.project_split,
        "directions": sorted(directions),
        "pairing": args.pairing,
        "rows_loaded": len(rows),
        "directed_pairs_after_retokenize": len(pairs),
        "train_pairs": len(train_pairs),
        "eval_pairs": len(eval_pairs),
        "projected_unique_prompt_contexts": len(detail_rows),
        "skip_counts": skip_counts,
        "detail_csv": str(detail_csv),
        "validation_summary_csv": str(validation_summary_csv),
        "label_summary_csv": str(label_summary_csv),
        "pair_detail_csv": str(pair_detail_csv),
        "pair_summary_csv": str(pair_summary_csv),
        "note": (
            "D-validation only: real-verb separation does not distinguish lookup from computation. "
            "The primary instrument check is validation_summary_csv, where the sign is calibrated "
            "by causative good-minus-bad and then applied to inchoative good-minus-bad. "
            "Rank-1 coordinate sign is arbitrary for DAS; use orientation_free_auc for uncalibrated summaries."
        ),
    }
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--subspace", default=DEFAULT_SUBSPACE)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--site", default="resid_post_layer_23")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--subtasks", default="causative,inchoative")
    ap.add_argument("--train-regimes", default="head")
    ap.add_argument("--eval-regimes", default="head,low")
    ap.add_argument("--project-split", choices=("train", "eval", "all"), default="eval")
    ap.add_argument("--directions", default="good_to_bad,bad_to_good")
    ap.add_argument("--pairing", choices=("source_idx", "balanced_pool"), default="balanced_pool")
    ap.add_argument("--max-pairs-per-group", type=int, default=None)
    ap.add_argument("--eval-frac", type=float, default=0.25)
    ap.add_argument("--split-by", choices=("pair", "lemma_pair"), default="lemma_pair")
    ap.add_argument(
        "--control",
        choices=("none", "shuffled_label", "dummy_verb", "random_direction", "red_blue"),
        default="none",
    )
    ap.add_argument("--dummy-verb", default="do")
    ap.add_argument(
        "--label-columns",
        default="side,licensing_status,expected_target,object_target_status,subtask,source_zipf_regime,inventory_source,subject_class",
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--out-dir", default="results/das_projection")
    ap.add_argument("--run-name", default="")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
