#!/usr/bin/env python3
"""Merge nonce behavior and projection outputs into the final grid."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_OUT_DIR = "results/nonce_frames"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    mu = mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def sem(values: list[float]) -> float:
    return stdev(values) / math.sqrt(len(values)) if len(values) >= 2 else math.nan


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def sign(value: float, threshold: float) -> int:
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def merge_rows(
    behavior_rows: list[dict[str, str]],
    projection_rows: list[dict[str, str]],
    behavior_threshold: float,
    projection_threshold: float,
) -> list[dict[str, Any]]:
    projection_by_lemma = {row["lemma"]: row for row in projection_rows}
    out: list[dict[str, Any]] = []
    for b_row in behavior_rows:
        lemma = b_row["lemma"]
        p_row = projection_by_lemma.get(lemma)
        if p_row is None:
            continue
        delta_p = float(b_row["delta_p_object"])
        delta_proj = float(p_row["delta_coord_calibrated"])
        behavior_s = sign(delta_p, behavior_threshold)
        projection_s = sign(delta_proj, projection_threshold)
        behavior_status = "moves" if behavior_s != 0 else "flat"
        if behavior_s == 0:
            projection_status = "aligns" if projection_s == 0 else "off_axis"
        else:
            projection_status = "aligns" if projection_s == behavior_s else "off_axis"
        out.append(
            {
                "lemma": lemma,
                "delta_p_object": delta_p,
                "behavior_status": behavior_status,
                "mean_p_object_t_plus": float(b_row["mean_p_object_t_plus"]),
                "mean_p_object_t_minus": float(b_row["mean_p_object_t_minus"]),
                "delta_coord_calibrated": delta_proj,
                "projection_status": projection_status,
                "mean_coord_calibrated_t_plus": float(p_row["mean_coord_calibrated_t_plus"]),
                "mean_coord_calibrated_t_minus": float(p_row["mean_coord_calibrated_t_minus"]),
                "delta_coord_raw": float(p_row["delta_coord_raw"]),
                "probe_form_token_count": b_row.get("probe_form_token_count", ""),
                "probe_form_token_pieces": b_row.get("probe_form_token_pieces", ""),
                "probe_subject": b_row.get("probe_subject", ""),
                "probe_subject_animacy": b_row.get("probe_subject_animacy", ""),
            }
        )
    return out


def grid_rows(merged: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in merged:
        grouped[(str(row["behavior_status"]), str(row["projection_status"]))].append(row)
    out: list[dict[str, Any]] = []
    for behavior_status in ("moves", "flat"):
        for projection_status in ("aligns", "off_axis"):
            rows = grouped.get((behavior_status, projection_status), [])
            out.append(
                {
                    "behavior_status": behavior_status,
                    "projection_status": projection_status,
                    "n": len(rows),
                    "mean_delta_p_object": mean([float(row["delta_p_object"]) for row in rows]),
                    "mean_delta_coord_calibrated": mean([float(row["delta_coord_calibrated"]) for row in rows]),
                    "lemmas": " ".join(str(row["lemma"]) for row in rows[:20]),
                }
            )
    return out


def pooled_rows(merged: list[dict[str, Any]]) -> list[dict[str, Any]]:
    delta_p = [float(row["delta_p_object"]) for row in merged]
    delta_proj = [float(row["delta_coord_calibrated"]) for row in merged]
    aligns = [1.0 if row["projection_status"] == "aligns" else 0.0 for row in merged]
    return [
        {
            "n_lemmas": len(merged),
            "mean_delta_p_object": mean(delta_p),
            "sem_delta_p_object": sem(delta_p),
            "frac_delta_p_positive": mean([1.0 if value > 0 else 0.0 for value in delta_p]),
            "mean_delta_coord_calibrated": mean(delta_proj),
            "sem_delta_coord_calibrated": sem(delta_proj),
            "frac_delta_coord_positive": mean([1.0 if value > 0 else 0.0 for value in delta_proj]),
            "frac_projection_status_aligns": mean(aligns),
        }
    ]


def write_report(
    path: Path,
    merged: list[dict[str, Any]],
    grid: list[dict[str, Any]],
    pooled: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    pooled_row = pooled[0] if pooled else {}
    lines = [
        "# Nonce Frame Experiment Report",
        "",
        "Behavior is the gate: `DeltaP = P(object | T+) - P(object | T-)`, where object mass is compared only against period.",
        "Projection is descriptive and uses the pre-existing axis/subspace; no refit is performed here.",
        "",
        "## Inputs",
        "",
        f"- behavior: `{args.behavior}`",
        f"- projection: `{args.projection}`",
        f"- behavior threshold: `{args.behavior_threshold}`",
        f"- projection threshold: `{args.projection_threshold}`",
        "",
        "## Pooled",
        "",
        f"- n lemmas: {pooled_row.get('n_lemmas', 0)}",
        f"- mean DeltaP: {pooled_row.get('mean_delta_p_object', math.nan)}",
        f"- SEM DeltaP: {pooled_row.get('sem_delta_p_object', math.nan)}",
        f"- mean calibrated DeltaProjection: {pooled_row.get('mean_delta_coord_calibrated', math.nan)}",
        f"- SEM calibrated DeltaProjection: {pooled_row.get('sem_delta_coord_calibrated', math.nan)}",
        "",
        "## Grid",
        "",
        "| Behavior | Projection | n | mean DeltaP | mean DeltaProjection |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in grid:
        lines.append(
            f"| {row['behavior_status']} | {row['projection_status']} | {row['n']} | "
            f"{row['mean_delta_p_object']} | {row['mean_delta_coord_calibrated']} |"
        )
    lines.extend(
        [
            "",
            "## Per-Lemma Preview",
            "",
            "| lemma | DeltaP | behavior | DeltaProjection | projection |",
            "| --- | ---: | --- | ---: | --- |",
        ]
    )
    for row in sorted(merged, key=lambda item: -abs(float(item["delta_p_object"])))[:25]:
        lines.append(
            f"| {row['lemma']} | {row['delta_p_object']} | {row['behavior_status']} | "
            f"{row['delta_coord_calibrated']} | {row['projection_status']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    behavior_rows = read_csv(Path(args.behavior))
    projection_rows = read_csv(Path(args.projection))
    merged = merge_rows(behavior_rows, projection_rows, args.behavior_threshold, args.projection_threshold)
    if not merged:
        raise SystemExit("No overlapping lemma rows between behavior and projection files.")
    grid = grid_rows(merged)
    pooled = pooled_rows(merged)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or "nonce_summary"
    merged_csv = out_dir / f"{run_name}.merged_by_lemma.csv"
    grid_csv = out_dir / f"{run_name}.grid.csv"
    pooled_csv = out_dir / f"{run_name}.pooled.csv"
    report_md = Path(args.report) if args.report else out_dir / f"{run_name}.report.md"
    manifest_json = out_dir / f"{run_name}.manifest.json"
    write_csv(merged_csv, merged)
    write_csv(grid_csv, grid)
    write_csv(pooled_csv, pooled)
    write_report(report_md, merged, grid, pooled, args)
    manifest = {
        "behavior": args.behavior,
        "projection": args.projection,
        "n_lemmas": len(merged),
        "behavior_threshold": args.behavior_threshold,
        "projection_threshold": args.projection_threshold,
        "merged_csv": str(merged_csv),
        "grid_csv": str(grid_csv),
        "pooled_csv": str(pooled_csv),
        "report_md": str(report_md),
        "pooled": pooled[0],
        "grid": grid,
    }
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior", required=True, help="*.behavior_by_lemma.csv from scripts/run_nonce_behavior.py")
    ap.add_argument("--projection", required=True, help="*.projection_by_lemma.csv from scripts/project_nonce_frames.py")
    ap.add_argument("--behavior-threshold", type=float, default=0.02)
    ap.add_argument("--projection-threshold", type=float, default=0.0)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--run-name", default="")
    ap.add_argument("--report", default="")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
