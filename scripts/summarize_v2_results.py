#!/usr/bin/env python3
"""Summarize v2 head-to-low sanity, localization, and DAS outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


CONTROL_RE = re.compile(
    r"-l23-(?P<control>none|shuffled_label|dummy_verb|dummy_pair|random_direction|red_blue)-s(?P<seed>\d+)"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def as_float(value: Any) -> float:
    if value in {"", None}:
        return math.nan
    return float(value)


def finite(values: Iterable[float]) -> list[float]:
    return [value for value in values if not math.isnan(value)]


def bootstrap_ci_by_key(
    rows: list[dict[str, str]],
    value_col: str,
    key_col: str = "pair_key",
    n_boot: int = 1000,
    seed: int = 17,
) -> tuple[float, float]:
    by_key: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = as_float(row.get(value_col))
        if not math.isnan(value):
            by_key[str(row.get(key_col, len(by_key)))].append(value)
    keys = sorted(by_key)
    if len(keys) < 2:
        return math.nan, math.nan
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_boot):
        values: list[float] = []
        for key in (rng.choice(keys) for _ in keys):
            values.extend(by_key[key])
        samples.append(mean(values))
    samples.sort()
    return samples[int(0.025 * n_boot)], samples[int(0.975 * n_boot)]


def summarize_sanity(rows: list[dict[str, str]], group_cols: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(col, "") for col in group_cols)].append(row)
    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        logodds = finite(as_float(row.get("expected_logodds")) for row in group)
        out.append(
            {
                **dict(zip(group_cols, key)),
                "n": len(group),
                "mean_expected_logodds": mean(logodds),
                "expected_preferred_rate": mean([1.0 if value > 0 else 0.0 for value in logodds]),
                "min_expected_logodds": min(logodds) if logodds else math.nan,
                "max_expected_logodds": max(logodds) if logodds else math.nan,
            }
        )
    return out


def infer_control(path: Path) -> tuple[str, str]:
    match = CONTROL_RE.search(path.name)
    if not match:
        return "unknown", ""
    return match.group("control"), match.group("seed")


def summarize_das(rows: list[dict[str, str]], group_cols: tuple[str, ...], control: str, seed: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(col, "") for col in group_cols)].append(row)
    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        effect = finite(as_float(row.get("effect")) for row in group)
        norm = [] if control == "red_blue" else finite(as_float(row.get("normalized_effect")) for row in group)
        corrupt = finite(as_float(row.get("corrupt_metric")) for row in group)
        patched = finite(as_float(row.get("patched_metric")) for row in group)
        clean = finite(as_float(row.get("clean_metric")) for row in group)
        ci_lo, ci_hi = bootstrap_ci_by_key(group, "effect")
        out.append(
            {
                "control": control,
                "seed": seed,
                **dict(zip(group_cols, key)),
                "n": len(group),
                "mean_corrupt_metric": mean(corrupt),
                "mean_patched_metric": mean(patched),
                "mean_clean_metric": mean(clean),
                "mean_effect": mean(effect),
                "effect_ci95_lo": ci_lo,
                "effect_ci95_hi": ci_hi,
                "mean_normalized_effect": mean(norm),
                "patched_success_rate": mean([float(row.get("patched_success", 0)) for row in group]),
            }
        )
    return out


def summarize_das_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("control", "")), str(row.get("seed", "")), str(row.get("regime", "")))].append(row)

    out: list[dict[str, Any]] = []
    for (control, seed, regime), group in sorted(grouped.items()):
        weighted_n = sum(int(row.get("n", 0)) for row in group)
        if weighted_n <= 0:
            continue

        def weighted_mean(col: str) -> float:
            values = [
                (as_float(row.get(col)), int(row.get("n", 0)))
                for row in group
                if not math.isnan(as_float(row.get(col)))
            ]
            denom = sum(n for _value, n in values)
            return sum(value * n for value, n in values) / denom if denom else math.nan

        out.append(
            {
                "control": control,
                "seed": seed,
                "regime": regime,
                "n": weighted_n,
                "mean_effect": weighted_mean("mean_effect"),
                "mean_normalized_effect": weighted_mean("mean_normalized_effect"),
                "patched_success_rate": weighted_mean("patched_success_rate"),
            }
        )
    return out


def summarize_ap(rows: list[dict[str, str]], top_n: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "regime": row.get("regime", ""),
                "subtask": row.get("subtask", ""),
                "direction": row.get("direction", ""),
                "site": row.get("site", ""),
                "site_index": int(row.get("site_index", 0)),
                "n": int(row.get("n", 0)),
                "mean_attribution": as_float(row.get("mean_attribution")),
                "mean_abs_attribution": as_float(row.get("mean_abs_attribution")),
                "mean_corrupt_metric": as_float(row.get("mean_corrupt_metric")),
                "mean_clean_metric": as_float(row.get("mean_clean_metric")),
            }
        )
    return top_sites_by_group(out, top_n, "mean_abs_attribution")


def summarize_exact(rows: list[dict[str, str]], top_n: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "regime": row.get("regime", ""),
                "subtask": row.get("subtask", ""),
                "direction": row.get("direction", ""),
                "site": row.get("site", ""),
                "site_index": int(row.get("site_index", 0)),
                "n": int(row.get("n", 0)),
                "mean_exact_effect": as_float(row.get("mean_exact_effect")),
                "mean_abs_exact_effect": as_float(row.get("mean_abs_exact_effect")),
                "mean_normalized_effect": as_float(row.get("mean_normalized_effect")),
                "mean_corrupt_metric": as_float(row.get("mean_corrupt_metric")),
                "mean_patched_metric": as_float(row.get("mean_patched_metric")),
                "mean_clean_metric": as_float(row.get("mean_clean_metric")),
            }
        )
    return top_sites_by_group(out, top_n, "mean_abs_exact_effect")


def top_sites_by_group(rows: list[dict[str, Any]], top_n: int, metric_col: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["regime"], row["subtask"], row["direction"])].append(row)
    if not groups:
        return []
    per_group = max(1, top_n // len(groups))
    selected: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda group: (group[1], group[0], group[2])):
        group_rows = sorted(groups[key], key=lambda row: -row[metric_col])
        selected.extend(group_rows[:per_group])
    return selected


def markdown_table(rows: list[dict[str, Any]], cols: tuple[str, ...], max_rows: int = 20) -> list[str]:
    if not rows:
        return ["_No rows found._"]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in rows[:max_rows]:
        values = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}" if not math.isnan(value) else "nan"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--run-prefix", default="20260623-pythia14b-v2")
    ap.add_argument("--top-ap", type=int, default=24)
    args = ap.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sanity_path = results_root / "v2_sanity" / f"{args.run_prefix}-sanity-head-low.detail.csv"
    if not sanity_path.exists():
        matches = sorted((results_root / "v2_sanity").glob(f"{args.run_prefix}-sanity-head-low*.detail.csv"))
        sanity_path = matches[0] if matches else sanity_path
    sanity_detail = read_csv(sanity_path)
    red_blue_sanity_path = results_root / "v2_sanity" / f"{args.run_prefix}-sanity-red-blue-head-low.detail.csv"
    if not red_blue_sanity_path.exists():
        matches = sorted((results_root / "v2_sanity").glob(f"{args.run_prefix}*red*blue*.detail.csv"))
        red_blue_sanity_path = matches[0] if matches else red_blue_sanity_path
    red_blue_sanity_detail = read_csv(red_blue_sanity_path)
    ap_summary = read_csv(results_root / "attribution_patching_v2" / f"{args.run_prefix}-ap-head-low.summary.csv")
    exact_summary = read_csv(results_root / "exact_patching_v2" / f"{args.run_prefix}-exact-head-low.summary.csv")

    das_eval_summaries: list[dict[str, Any]] = []
    das_subject_summaries: list[dict[str, Any]] = []
    das_source_summaries: list[dict[str, Any]] = []
    das_detail_paths = sorted((results_root / "das_v2_transfer").glob(f"{args.run_prefix}-das-head-to-low-l23-*-s*.eval_detail.csv"))
    has_corrected_dummy = any("-dummy_pair-" in path.name for path in das_detail_paths)
    skipped_das_files: list[str] = []
    for detail_path in das_detail_paths:
        control, seed = infer_control(detail_path)
        if has_corrected_dummy and control == "dummy_verb":
            skipped_das_files.append(str(detail_path))
            continue
        rows = read_csv(detail_path)
        das_eval_summaries.extend(summarize_das(rows, ("regime", "subtask", "direction"), control, seed))
        das_subject_summaries.extend(summarize_das(rows, ("regime", "subtask", "subject_class"), control, seed))
        das_source_summaries.extend(
            summarize_das(rows, ("regime", "subtask", "clean_source_zipf_regime", "clean_inventory_source"), control, seed)
        )

    sanity_main = summarize_sanity(sanity_detail, ("regime", "subtask", "side"))
    sanity_source = summarize_sanity(sanity_detail, ("regime", "subtask", "side", "source_zipf_regime", "inventory_source"))
    red_blue_sanity_main = summarize_sanity(red_blue_sanity_detail, ("regime", "subtask", "side"))
    red_blue_sanity_source = summarize_sanity(
        red_blue_sanity_detail,
        ("regime", "subtask", "side", "source_zipf_regime", "inventory_source"),
    )
    ap_top = summarize_ap(ap_summary, args.top_ap)
    exact_top = summarize_exact(exact_summary, args.top_ap)
    das_comparison = summarize_das_comparison(das_eval_summaries)

    write_csv(out_dir / "v2_sanity_summary.csv", sanity_main)
    write_csv(out_dir / "v2_sanity_source_summary.csv", sanity_source)
    write_csv(out_dir / "v2_red_blue_sanity_summary.csv", red_blue_sanity_main)
    write_csv(out_dir / "v2_red_blue_sanity_source_summary.csv", red_blue_sanity_source)
    write_csv(out_dir / "v2_ap_top_sites.csv", ap_top)
    write_csv(out_dir / "v2_exact_top_sites.csv", exact_top)
    write_csv(out_dir / "v2_das_eval_summary.csv", das_eval_summaries)
    write_csv(out_dir / "v2_das_subject_summary.csv", das_subject_summaries)
    write_csv(out_dir / "v2_das_source_summary.csv", das_source_summaries)
    write_csv(out_dir / "v2_das_control_comparison.csv", das_comparison)

    lines = [
        "# V2 Head-to-Low Results",
        "",
        "## Baseline Sanity",
        "",
        *markdown_table(
            sanity_main,
            ("regime", "subtask", "side", "n", "mean_expected_logodds", "expected_preferred_rate"),
            max_rows=16,
        ),
        "",
        "## Red/Blue Baseline",
        "",
        "Prompt text is unchanged; this scores `log p(\" red\") - log p(\" blue\")` and ignores the original targets.",
        "",
        *markdown_table(
            red_blue_sanity_main,
            ("regime", "subtask", "side", "n", "mean_expected_logodds", "expected_preferred_rate"),
            max_rows=16,
        ),
        "",
        "## Attribution Patching Top Sites",
        "",
        *markdown_table(
            ap_top,
            ("regime", "subtask", "direction", "site", "n", "mean_attribution", "mean_abs_attribution"),
            max_rows=args.top_ap,
        ),
        "",
        "## Exact Patching Top Sites",
        "",
        *markdown_table(
            exact_top,
            (
                "regime",
                "subtask",
                "direction",
                "site",
                "n",
                "mean_exact_effect",
                "mean_normalized_effect",
            ),
            max_rows=args.top_ap,
        ),
        "",
        "## DAS Summary",
        "",
        "Aggregate control comparison across subtasks and directions:",
        "",
        *markdown_table(
            das_comparison,
            (
                "control",
                "seed",
                "regime",
                "n",
                "mean_effect",
                "mean_normalized_effect",
                "patched_success_rate",
            ),
            max_rows=20,
        ),
        "",
        "Per-subtask and per-direction detail:",
        "",
        *markdown_table(
            das_eval_summaries,
            (
                "control",
                "seed",
                "regime",
                "subtask",
                "direction",
                "n",
                "mean_effect",
                "effect_ci95_lo",
                "effect_ci95_hi",
                "mean_normalized_effect",
                "patched_success_rate",
            ),
            max_rows=40,
        ),
        "",
        "## Output Files",
        "",
        "- `reports/v2_sanity_summary.csv`",
        "- `reports/v2_sanity_source_summary.csv`",
        "- `reports/v2_red_blue_sanity_summary.csv`",
        "- `reports/v2_red_blue_sanity_source_summary.csv`",
        "- `reports/v2_ap_top_sites.csv`",
        "- `reports/v2_exact_top_sites.csv`",
        "- `reports/v2_das_eval_summary.csv`",
        "- `reports/v2_das_control_comparison.csv`",
        "- `reports/v2_das_subject_summary.csv`",
        "- `reports/v2_das_source_summary.csv`",
    ]
    report_path = out_dir / "v2_head_low_results_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "run_prefix": args.run_prefix,
        "sanity_rows": len(sanity_detail),
        "red_blue_sanity_rows": len(red_blue_sanity_detail),
        "ap_summary_rows": len(ap_summary),
        "exact_summary_rows": len(exact_summary),
        "das_eval_summary_rows": len(das_eval_summaries),
        "skipped_das_files": skipped_das_files,
        "report": str(report_path),
    }
    (out_dir / "v2_results_summary_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
