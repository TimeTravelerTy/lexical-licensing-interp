#!/usr/bin/env python3
"""Project nonce-frame verb-final activations onto a saved DAS/axis direction."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from run_pythia_attribution_patching import DEFAULT_MODEL, shell_safe_slug
from run_pythia_exact_patching import normalize_model_output, parse_site


DEFAULT_DATA = "data/nonce_frames/nonce_frames.jsonl"
DEFAULT_OUT_DIR = "results/nonce_frames"
DEFAULT_SUBSPACE = "results/whole_pair_das/20260626-pythia14b-v2-wholepair-das-l20-none-s17.subspace.pt"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def sem(values: list[float]) -> float:
    return stdev(values) / math.sqrt(len(values)) if len(values) >= 2 else math.nan


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
    return rows


def orthonormal_basis(raw: Any) -> Any:
    import torch

    q, _r = torch.linalg.qr(raw.float(), mode="reduced")
    return q


def load_basis(path: Path, expected_site: str, expected_rank: int | None, device: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    raw = checkpoint["raw"] if isinstance(checkpoint, dict) and "raw" in checkpoint else checkpoint
    if raw.ndim == 1:
        raw = raw[:, None]
    if isinstance(checkpoint, dict) and checkpoint.get("site") not in {None, expected_site}:
        raise SystemExit(f"Subspace site mismatch: checkpoint={checkpoint.get('site')} --site={expected_site}")
    if expected_rank is not None and raw.shape[1] != expected_rank:
        raise SystemExit(f"Subspace rank mismatch: checkpoint={raw.shape[1]} --rank={expected_rank}")
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    return orthonormal_basis(raw).to(device), metadata


def capture_site(model: Any, inputs: Any, site_index: int, anchor: int) -> Any:
    captured: dict[str, Any] = {}

    def capture_tensor(hidden: Any) -> None:
        captured["site"] = hidden[:, anchor, :].detach()

    if site_index == 0:
        handle = model.gpt_neox.embed_in.register_forward_hook(lambda _module, _inputs, output: capture_tensor(output))
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


def inferred_calibration_csv(subspace: Path) -> Path | None:
    candidate = subspace.with_name(subspace.name.replace(".subspace.pt", ".projection_detail.csv"))
    return candidate if candidate.exists() else None


def check_behavior_gate(path: Path | None, min_abs_delta: float) -> dict[str, Any]:
    if path is None:
        return {
            "behavior_summary_csv": "",
            "behavior_gate_status": "not_checked",
            "min_abs_delta_p": min_abs_delta,
        }
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"Behavior summary is empty: {path}")
    row = rows[0]
    if "mean_delta_p_object" not in row:
        raise SystemExit(f"Behavior summary lacks mean_delta_p_object: {path}")
    delta = float(row["mean_delta_p_object"])
    if abs(delta) < min_abs_delta:
        raise SystemExit(
            f"Behavior gate failed: abs(mean_delta_p_object)={abs(delta):.6g} < {min_abs_delta:.6g}. "
            "Strengthen frames before projection."
        )
    return {
        "behavior_summary_csv": str(path),
        "behavior_gate_status": "passed",
        "mean_delta_p_object": delta,
        "min_abs_delta_p": min_abs_delta,
    }


def calibration_orientation(path: Path | None) -> tuple[float, dict[str, Any]]:
    if path is None or not path.exists():
        return 1.0, {
            "calibration_csv": "",
            "orientation": 1.0,
            "calibration_status": "missing_calibration_csv_raw_orientation",
        }
    causative: list[float] = []
    inchoative: list[float] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "coord_0" not in row:
                continue
            subtask = str(row.get("subtask", ""))
            if subtask == "causative":
                causative.append(float(row["coord_0"]))
            elif subtask == "inchoative":
                inchoative.append(float(row["coord_0"]))
    if not causative or not inchoative:
        return 1.0, {
            "calibration_csv": str(path),
            "orientation": 1.0,
            "calibration_status": "insufficient_calibration_rows_raw_orientation",
            "n_causative": len(causative),
            "n_inchoative": len(inchoative),
        }
    raw_margin = mean(inchoative) - mean(causative)
    orientation = 1.0 if raw_margin >= 0 else -1.0
    return orientation, {
        "calibration_csv": str(path),
        "orientation": orientation,
        "calibration_status": "calibrated_inchoative_positive_causative_negative",
        "n_causative": len(causative),
        "n_inchoative": len(inchoative),
        "mean_causative_raw_coord": mean(causative),
        "mean_inchoative_raw_coord": mean(inchoative),
        "raw_inchoative_minus_causative": raw_margin,
        "calibrated_inchoative_minus_causative": orientation * raw_margin,
    }


def project_rows(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    basis: Any,
    orientation: float,
    site_index: int,
    batch_size: int,
    device: Any,
) -> list[dict[str, Any]]:
    import torch

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        prompt_len = int(row.get("full_context_token_count", 0) or len(tokenizer.encode(row["full_context"], add_special_tokens=False)))
        anchor = int(row.get("verb_final_tok_idx", prompt_len - 1))
        grouped[(prompt_len, anchor)].append(row)

    projected: list[dict[str, Any]] = []
    with torch.no_grad():
        for (_prompt_len, anchor), group in sorted(grouped.items()):
            for start in range(0, len(group), batch_size):
                batch = group[start : start + batch_size]
                inputs = tokenizer(
                    [str(row["full_context"]) for row in batch],
                    return_tensors="pt",
                    padding=True,
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
                    item = dict(row)
                    if not isinstance(coord, list):
                        coord = [float(coord)]
                    for idx, value in enumerate(coord):
                        item[f"coord_{idx}_raw"] = value
                        item[f"coord_{idx}_calibrated"] = orientation * value
                    item["projection_norm"] = proj_n
                    item["residual_norm"] = resid_n
                    item["projection_norm_frac"] = "" if resid_n == 0 else proj_n / resid_n
                    projected.append(item)
    return projected


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
        plus_raw = [float(row["coord_0_raw"]) for row in plus]
        minus_raw = [float(row["coord_0_raw"]) for row in minus]
        plus_cal = [float(row["coord_0_calibrated"]) for row in plus]
        minus_cal = [float(row["coord_0_calibrated"]) for row in minus]
        out.append(
            {
                "lemma": lemma,
                "n_t_plus": len(plus),
                "n_t_minus": len(minus),
                "mean_coord_raw_t_plus": mean(plus_raw),
                "mean_coord_raw_t_minus": mean(minus_raw),
                "delta_coord_raw": mean(plus_raw) - mean(minus_raw),
                "mean_coord_calibrated_t_plus": mean(plus_cal),
                "mean_coord_calibrated_t_minus": mean(minus_cal),
                "delta_coord_calibrated": mean(plus_cal) - mean(minus_cal),
                "probe_form_token_count": plus[0].get("probe_form_token_count", ""),
                "probe_form_token_pieces": plus[0].get("probe_form_token_pieces", ""),
                "probe_subject": plus[0].get("probe_subject", ""),
                "probe_subject_animacy": plus[0].get("probe_subject_animacy", ""),
            }
        )
    return out


def summarize_pooled(lemma_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw = [float(row["delta_coord_raw"]) for row in lemma_rows]
    cal = [float(row["delta_coord_calibrated"]) for row in lemma_rows]
    return [
        {
            "n_lemmas": len(lemma_rows),
            "mean_delta_coord_raw": mean(raw),
            "sem_delta_coord_raw": sem(raw),
            "frac_raw_delta_positive": mean([1.0 if value > 0 else 0.0 for value in raw]),
            "mean_delta_coord_calibrated": mean(cal),
            "sem_delta_coord_calibrated": sem(cal),
            "frac_calibrated_delta_positive": mean([1.0 if value > 0 else 0.0 for value in cal]),
        }
    ]


def run(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_rows(Path(args.data))
    site_index = parse_site(args.site)
    behavior_gate = check_behavior_gate(Path(args.behavior_summary) if args.behavior_summary else None, args.min_abs_delta_p)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, local_files_only=not args.allow_download)
    device = torch.device(args.device)
    model.to(device)
    model.eval()
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad_(False)

    basis, subspace_metadata = load_basis(Path(args.subspace), args.site, args.rank, device)
    if basis.shape[0] != model.config.hidden_size:
        raise SystemExit(f"Subspace d_model mismatch: basis={basis.shape[0]} model={model.config.hidden_size}")

    calibration_csv = Path(args.calibration_csv) if args.calibration_csv else inferred_calibration_csv(Path(args.subspace))
    orientation, calibration = calibration_orientation(calibration_csv)
    detail_rows = project_rows(model, tokenizer, rows, basis, orientation, site_index, args.batch_size, device)
    lemma_rows = summarize_by_lemma(detail_rows)
    summary_rows = summarize_pooled(lemma_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_slug = args.run_name or shell_safe_slug(f"{args.model}_{args.site}_nonce_projection")
    detail_csv = out_dir / f"{run_slug}.projection_detail.csv"
    by_lemma_csv = out_dir / f"{run_slug}.projection_by_lemma.csv"
    summary_csv = out_dir / f"{run_slug}.projection_summary.csv"
    manifest_json = out_dir / f"{run_slug}.projection_manifest.json"
    write_csv(detail_csv, detail_rows)
    write_csv(by_lemma_csv, lemma_rows)
    write_csv(summary_csv, summary_rows)

    manifest = {
        "model": args.model,
        "data": str(Path(args.data)),
        "site": args.site,
        "rank": args.rank,
        "subspace": str(Path(args.subspace)),
        "subspace_metadata_site": subspace_metadata.get("site", "") if isinstance(subspace_metadata, dict) else "",
        "rows_projected": len(detail_rows),
        "lemmas_projected": len(lemma_rows),
        "detail_csv": str(detail_csv),
        "by_lemma_csv": str(by_lemma_csv),
        "summary_csv": str(summary_csv),
        "calibration": calibration,
        "behavior_gate": behavior_gate,
        "pooled": summary_rows[0] if summary_rows else {},
        "note": "Projection is descriptive only. No axis refit is performed.",
    }
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--site", default="resid_post_layer_20")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--subspace", default=DEFAULT_SUBSPACE)
    ap.add_argument("--calibration-csv", default="")
    ap.add_argument("--behavior-summary", default="", help="Optional behavior_summary.csv gate from run_nonce_behavior.py")
    ap.add_argument("--min-abs-delta-p", type=float, default=0.02)
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
