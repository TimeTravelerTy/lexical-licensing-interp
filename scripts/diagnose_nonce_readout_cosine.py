#!/usr/bin/env python3
"""Compare a saved object-expectation axis with next-token readout directions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from project_nonce_frames import calibration_orientation, inferred_calibration_csv, load_basis
from run_nonce_behavior import DEFAULT_OBJECT_STARTS, continuation_ids, parse_object_starts, target_label
from run_pythia_attribution_patching import DEFAULT_MODEL, shell_safe_slug


DEFAULT_OUT_DIR = "results/nonce_frames"
DEFAULT_SUBSPACE = "results/whole_pair_das/20260626-pythia14b-v2-wholepair-das-l20-none-s17.subspace.pt"


def cosine(left: Any, right: Any) -> float:
    denom = left.norm() * right.norm()
    if float(denom) == 0.0:
        return math.nan
    return float((left @ right) / denom)


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


def run(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
    object_starts = parse_object_starts(args.object_starts)
    object_ids, period_id, kept_object_starts = continuation_ids(tokenizer, object_starts, args.period_target)

    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, local_files_only=not args.allow_download)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    device = torch.device(args.device)
    basis, subspace_metadata = load_basis(Path(args.subspace), args.site, args.rank, device)
    if basis.shape[1] != 1:
        raise SystemExit(f"This diagnostic expects rank 1, got rank={basis.shape[1]}")
    calibration_csv = Path(args.calibration_csv) if args.calibration_csv else inferred_calibration_csv(Path(args.subspace))
    orientation, calibration = calibration_orientation(calibration_csv)
    axis_raw = basis[:, 0].float().cpu()
    axis_calibrated = orientation * axis_raw

    unembed = model.get_output_embeddings().weight.detach().float().cpu()
    period_vec = unembed[period_id]
    object_vecs = unembed[object_ids]
    object_centroid = object_vecs.mean(dim=0)
    centroid_minus_period = object_centroid - period_vec

    rows: list[dict[str, Any]] = [
        {
            "target": "<object_centroid>",
            "label": "object_centroid",
            "token_id": "",
            "cos_raw_axis_readout_minus_period": cosine(axis_raw, centroid_minus_period),
            "cos_calibrated_axis_readout_minus_period": cosine(axis_calibrated, centroid_minus_period),
            "readout_direction_norm": float(centroid_minus_period.norm()),
        }
    ]
    for target, token in zip(kept_object_starts, object_ids):
        direction = unembed[token] - period_vec
        rows.append(
            {
                "target": target,
                "label": target_label(target),
                "token_id": token,
                "cos_raw_axis_readout_minus_period": cosine(axis_raw, direction),
                "cos_calibrated_axis_readout_minus_period": cosine(axis_calibrated, direction),
                "readout_direction_norm": float(direction.norm()),
            }
        )

    summary = rows[0]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_slug = args.run_name or shell_safe_slug(f"{args.model}_{args.site}_readout_cosine")
    detail_csv = out_dir / f"{run_slug}.readout_cosine.csv"
    manifest_json = out_dir / f"{run_slug}.readout_cosine_manifest.json"
    write_csv(detail_csv, rows)
    manifest = {
        "model": args.model,
        "site": args.site,
        "rank": args.rank,
        "subspace": str(Path(args.subspace)),
        "subspace_metadata_site": subspace_metadata.get("site", "") if isinstance(subspace_metadata, dict) else "",
        "calibration": calibration,
        "object_starts": kept_object_starts,
        "object_token_ids": object_ids,
        "period_target": args.period_target,
        "period_token_id": period_id,
        "detail_csv": str(detail_csv),
        "summary": summary,
        "note": "Cosines compare the saved rank-1 residual axis against lm_head token-vector differences.",
    }
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--site", default="resid_post_layer_20")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--subspace", default=DEFAULT_SUBSPACE)
    ap.add_argument("--calibration-csv", default="")
    ap.add_argument("--object-starts", default=",".join(DEFAULT_OBJECT_STARTS))
    ap.add_argument("--period-target", default=".")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--run-name", default="")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
