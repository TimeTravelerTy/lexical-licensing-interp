#!/usr/bin/env python3
"""Run residual-stream attribution patching on aligned lexical templates.

This is a localization screen, not the final causal intervention. It estimates
the effect of replacing a corrupt prompt's residual stream with the matching
clean prompt's residual stream at the prompt-final verb token:

    grad_corrupt_metric · (clean_activation - corrupt_activation)

The metric is:

    log p(clean_expected_target | corrupt prompt) -
    log p(corrupt_expected_target | corrupt prompt)

Rows must come from the tokenizer-verified aligned template JSONL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SUBTASKS = ("causative", "inchoative")
DEFAULT_REGIMES = ("head", "low")
DEFAULT_MODEL = "EleutherAI/pythia-1.4b"


@dataclass(frozen=True)
class DirectedPair:
    pair_key: str
    regime: str
    subtask: str
    template_id: str
    source_idx: int
    direction: str
    clean_side: str
    corrupt_side: str
    clean_prompt: str
    corrupt_prompt: str
    clean_target: str
    corrupt_target: str
    anchor_token_index: int
    prompt_token_count: int
    clean_verb: str
    corrupt_verb: str
    context_id: str = ""
    subject: str = ""
    subject_class: str = ""
    clean_source_zipf_regime: str = ""
    corrupt_source_zipf_regime: str = ""
    clean_inventory_source: str = ""
    corrupt_inventory_source: str = ""


def stable_unit(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def parse_csv_arg(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def shell_safe_slug(value: str) -> str:
    return value.replace("/", "__").replace(".", "_").replace("-", "_")


def load_aligned_rows(
    path: Path,
    subtask_filter: set[str],
    regime_filter: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        if row.get("alignment_status") != "aligned":
            continue
        if row.get("subtask") not in subtask_filter:
            continue
        if row.get("regime") not in regime_filter:
            continue
        if row.get("intervention_anchor") != "verb_final_subtoken":
            continue
        rows.append(row)
    return rows


def build_directed_pairs(
    rows: list[dict[str, Any]],
    directions: set[str],
    max_pairs_per_group: int | None,
    pairing: str = "source_idx",
    seed: int = 17,
) -> tuple[list[DirectedPair], dict[str, int]]:
    if pairing == "balanced_pool":
        return build_directed_pairs_from_balanced_pools(rows, directions, max_pairs_per_group, seed)
    if pairing != "source_idx":
        raise ValueError(f"unknown pairing mode: {pairing}")

    grouped: dict[tuple[str, str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (
            str(row["regime"]),
            str(row["subtask"]),
            str(row["template_id"]),
            int(row["source_idx"]),
        )
        grouped[key][str(row["side"])] = row

    counts_by_group: dict[tuple[str, str, str], int] = defaultdict(int)
    skip_counts: dict[str, int] = defaultdict(int)
    pairs: list[DirectedPair] = []
    for (regime, subtask, template_id, source_idx), sides in sorted(grouped.items()):
        if "good" not in sides or "bad" not in sides:
            skip_counts["missing_good_or_bad_side"] += 1
            continue
        good = sides["good"]
        bad = sides["bad"]
        if int(good["prompt_token_count"]) != int(bad["prompt_token_count"]):
            skip_counts["prompt_token_count_mismatch"] += 1
            continue
        if int(good["anchor_token_index"]) != int(bad["anchor_token_index"]):
            skip_counts["anchor_token_index_mismatch"] += 1
            continue

        group_key = (regime, subtask, template_id)
        if max_pairs_per_group is not None and counts_by_group[group_key] >= max_pairs_per_group:
            skip_counts["max_pairs_per_group"] += 1
            continue
        counts_by_group[group_key] += 1

        pair_key = f"{regime}|{subtask}|{template_id}|{source_idx}"
        if "good_to_bad" in directions:
            pairs.append(
                DirectedPair(
                    pair_key=pair_key,
                    regime=regime,
                    subtask=subtask,
                    template_id=template_id,
                    source_idx=source_idx,
                    direction="good_to_bad",
                    clean_side="good",
                    corrupt_side="bad",
                    clean_prompt=str(good["prompt"]),
                    corrupt_prompt=str(bad["prompt"]),
                    clean_target=str(good["expected_target"]),
                    corrupt_target=str(bad["expected_target"]),
                    anchor_token_index=int(good["anchor_token_index"]),
                    prompt_token_count=int(good["prompt_token_count"]),
                    clean_verb=str(good["verb"]),
                    corrupt_verb=str(bad["verb"]),
                    context_id=str(good.get("context_id", "")),
                    subject=str(good.get("subject", "")),
                    subject_class=str(good.get("subject_class", "")),
                    clean_source_zipf_regime=str(good.get("source_zipf_regime", "")),
                    corrupt_source_zipf_regime=str(bad.get("source_zipf_regime", "")),
                    clean_inventory_source=str(good.get("inventory_source", "")),
                    corrupt_inventory_source=str(bad.get("inventory_source", "")),
                )
            )
        if "bad_to_good" in directions:
            pairs.append(
                DirectedPair(
                    pair_key=pair_key,
                    regime=regime,
                    subtask=subtask,
                    template_id=template_id,
                    source_idx=source_idx,
                    direction="bad_to_good",
                    clean_side="bad",
                    corrupt_side="good",
                    clean_prompt=str(bad["prompt"]),
                    corrupt_prompt=str(good["prompt"]),
                    clean_target=str(bad["expected_target"]),
                    corrupt_target=str(good["expected_target"]),
                    anchor_token_index=int(bad["anchor_token_index"]),
                    prompt_token_count=int(bad["prompt_token_count"]),
                    clean_verb=str(bad["verb"]),
                    corrupt_verb=str(good["verb"]),
                    context_id=str(bad.get("context_id", "")),
                    subject=str(bad.get("subject", "")),
                    subject_class=str(bad.get("subject_class", "")),
                    clean_source_zipf_regime=str(bad.get("source_zipf_regime", "")),
                    corrupt_source_zipf_regime=str(good.get("source_zipf_regime", "")),
                    clean_inventory_source=str(bad.get("inventory_source", "")),
                    corrupt_inventory_source=str(good.get("inventory_source", "")),
                )
            )
    return pairs, dict(skip_counts)


def build_directed_pairs_from_balanced_pools(
    rows: list[dict[str, Any]],
    directions: set[str],
    max_pairs_per_group: int | None,
    seed: int,
) -> tuple[list[DirectedPair], dict[str, int]]:
    grouped: dict[tuple[str, str, str, str, int, int], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        key = (
            str(row["regime"]),
            str(row["subtask"]),
            str(row["template_id"]),
            str(row.get("context_id", row["template_id"])),
            int(row["prompt_token_count"]),
            int(row["anchor_token_index"]),
        )
        grouped[key][str(row["side"])].append(row)

    skip_counts: dict[str, int] = defaultdict(int)
    pairs: list[DirectedPair] = []
    for (regime, subtask, template_id, context_id, prompt_len, anchor), sides in sorted(grouped.items()):
        good_rows = sides.get("good", [])
        bad_rows = sides.get("bad", [])
        if not good_rows or not bad_rows:
            skip_counts["missing_good_or_bad_side"] += max(len(good_rows), len(bad_rows), 1)
            continue

        def sort_key(row: dict[str, Any]) -> tuple[float, str, int]:
            stable_key = "|".join(
                [
                    regime,
                    subtask,
                    template_id,
                    context_id,
                    str(row.get("side", "")),
                    str(row.get("verb", "")),
                    str(row.get("source_idx", "")),
                    str(row.get("row_id", "")),
                ]
            )
            return (stable_unit(stable_key, seed), str(row.get("verb", "")), int(row.get("source_idx", 0)))

        good_rows = sorted(good_rows, key=sort_key)
        bad_rows = sorted(bad_rows, key=sort_key)
        n_pairs = min(len(good_rows), len(bad_rows))
        if max_pairs_per_group is not None:
            if n_pairs > max_pairs_per_group:
                skip_counts["max_pairs_per_group"] += n_pairs - max_pairs_per_group
            n_pairs = min(n_pairs, max_pairs_per_group)

        for index, (good, bad) in enumerate(zip(good_rows[:n_pairs], bad_rows[:n_pairs])):
            if str(good.get("verb")) == str(bad.get("verb")):
                skip_counts["same_clean_corrupt_verb"] += 1
                continue
            pair_key = (
                f"{regime}|{subtask}|{template_id}|{context_id}|"
                f"{index}|{good.get('verb')}|{bad.get('verb')}"
            )
            if "good_to_bad" in directions:
                pairs.append(
                    DirectedPair(
                        pair_key=pair_key,
                        regime=regime,
                        subtask=subtask,
                        template_id=template_id,
                        source_idx=index,
                        direction="good_to_bad",
                        clean_side="good",
                        corrupt_side="bad",
                        clean_prompt=str(good["prompt"]),
                        corrupt_prompt=str(bad["prompt"]),
                        clean_target=str(good["expected_target"]),
                        corrupt_target=str(bad["expected_target"]),
                        anchor_token_index=anchor,
                        prompt_token_count=prompt_len,
                        clean_verb=str(good["verb"]),
                        corrupt_verb=str(bad["verb"]),
                        context_id=context_id,
                        subject=str(good.get("subject", "")),
                        subject_class=str(good.get("subject_class", "")),
                        clean_source_zipf_regime=str(good.get("source_zipf_regime", "")),
                        corrupt_source_zipf_regime=str(bad.get("source_zipf_regime", "")),
                        clean_inventory_source=str(good.get("inventory_source", "")),
                        corrupt_inventory_source=str(bad.get("inventory_source", "")),
                    )
                )
            if "bad_to_good" in directions:
                pairs.append(
                    DirectedPair(
                        pair_key=pair_key,
                        regime=regime,
                        subtask=subtask,
                        template_id=template_id,
                        source_idx=index,
                        direction="bad_to_good",
                        clean_side="bad",
                        corrupt_side="good",
                        clean_prompt=str(bad["prompt"]),
                        corrupt_prompt=str(good["prompt"]),
                        clean_target=str(bad["expected_target"]),
                        corrupt_target=str(good["expected_target"]),
                        anchor_token_index=anchor,
                        prompt_token_count=prompt_len,
                        clean_verb=str(bad["verb"]),
                        corrupt_verb=str(good["verb"]),
                        context_id=context_id,
                        subject=str(bad.get("subject", "")),
                        subject_class=str(bad.get("subject_class", "")),
                        clean_source_zipf_regime=str(bad.get("source_zipf_regime", "")),
                        corrupt_source_zipf_regime=str(good.get("source_zipf_regime", "")),
                        clean_inventory_source=str(bad.get("inventory_source", "")),
                        corrupt_inventory_source=str(good.get("inventory_source", "")),
                    )
                )
    return pairs, dict(skip_counts)


def token_id(tokenizer: Any, target: str) -> int:
    ids = tokenizer.encode(target, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Target {target!r} is not a single token: {ids}")
    return int(ids[0])


def site_name(index: int) -> str:
    if index == 0:
        return "resid_embed"
    return f"resid_post_layer_{index - 1:02d}"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def write_summary(detail_rows: list[dict[str, Any]], output_csv: Path) -> list[dict[str, Any]]:
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
        attrs = [float(r["attribution"]) for r in rows]
        summary_rows.append(
            {
                "regime": regime,
                "subtask": subtask,
                "direction": direction,
                "site": site,
                "site_index": site_index,
                "n": len(rows),
                "mean_attribution": mean(attrs),
                "mean_abs_attribution": mean([abs(v) for v in attrs]),
                "mean_corrupt_metric": mean([float(r["corrupt_metric"]) for r in rows]),
                "mean_clean_metric": mean([float(r["clean_metric"]) for r in rows]),
            }
        )
    summary_rows.sort(
        key=lambda r: (
            str(r["subtask"]),
            str(r["regime"]),
            str(r["direction"]),
            -float(r["mean_abs_attribution"]),
        )
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)
    return summary_rows


def run(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = args.model
    subtask_filter = set(parse_csv_arg(args.subtasks))
    regime_filter = set(parse_csv_arg(args.regimes))
    directions = set(parse_csv_arg(args.directions))
    if not directions <= {"good_to_bad", "bad_to_good"}:
        raise SystemExit("--directions must contain only good_to_bad,bad_to_good")

    rows = load_aligned_rows(Path(args.data), subtask_filter, regime_filter)
    pairs, skip_counts = build_directed_pairs(rows, directions, args.max_pairs_per_group, args.pairing, args.seed)
    if not pairs:
        raise SystemExit("No valid directed pairs after filtering.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_slug = args.run_name or shell_safe_slug(
        f"{model_name}_{'-'.join(sorted(subtask_filter))}_{'-'.join(sorted(regime_filter))}"
    )
    detail_csv = out_dir / f"{run_slug}.detail.csv"
    summary_csv = out_dir / f"{run_slug}.summary.csv"
    manifest_json = out_dir / f"{run_slug}.manifest.json"

    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=not args.allow_download,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
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
        batches[
            (
                pair.regime,
                pair.subtask,
                pair.direction,
                pair.prompt_token_count,
                pair.anchor_token_index,
            )
        ].append(pair)

    detail_rows: list[dict[str, Any]] = []
    total_batches = 0
    for batch_key, batch_pairs in sorted(batches.items()):
        for start in range(0, len(batch_pairs), args.batch_size):
            batch = batch_pairs[start : start + args.batch_size]
            total_batches += 1
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
            row_indices = torch.arange(len(batch), device=device)

            with torch.no_grad():
                clean_outputs = model(**clean_inputs, output_hidden_states=True, use_cache=False)
                clean_hidden_sites = [h[:, anchor, :].detach() for h in clean_outputs.hidden_states]
                clean_logprobs = F.log_softmax(clean_outputs.logits[:, -1, :].float(), dim=-1)
                clean_metric = clean_logprobs[row_indices, clean_target] - clean_logprobs[row_indices, corrupt_target]

            model.zero_grad(set_to_none=True)
            corrupt_outputs = model(**corrupt_inputs, output_hidden_states=True, use_cache=False)
            corrupt_hidden_states = list(corrupt_outputs.hidden_states)
            for hidden in corrupt_hidden_states:
                hidden.retain_grad()
            corrupt_logprobs = F.log_softmax(corrupt_outputs.logits[:, -1, :].float(), dim=-1)
            corrupt_metric = corrupt_logprobs[row_indices, clean_target] - corrupt_logprobs[row_indices, corrupt_target]
            corrupt_metric.sum().backward()

            for site_index, hidden in enumerate(corrupt_hidden_states):
                grad_site = hidden.grad[:, anchor, :]
                corrupt_site = hidden[:, anchor, :].detach()
                attribution = (grad_site * (clean_hidden_sites[site_index] - corrupt_site)).sum(dim=-1)
                for pair, attr, corrupt_m, clean_m in zip(
                    batch,
                    attribution.detach().float().cpu().tolist(),
                    corrupt_metric.detach().float().cpu().tolist(),
                    clean_metric.detach().float().cpu().tolist(),
                ):
                    detail_rows.append(
                        {
                            "model": model_name,
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
                            "attribution": attr,
                            "corrupt_metric": corrupt_m,
                            "clean_metric": clean_m,
                        }
                    )

            del clean_outputs, corrupt_outputs, clean_hidden_sites, corrupt_hidden_states
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(detail_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)
    summary_rows = write_summary(detail_rows, summary_csv)

    top_sites = sorted(summary_rows, key=lambda r: -float(r["mean_abs_attribution"]))[:20]
    manifest = {
        "model": model_name,
        "data": str(Path(args.data).resolve()),
        "run_name": run_slug,
        "subtasks": sorted(subtask_filter),
        "regimes": sorted(regime_filter),
        "directions": sorted(directions),
        "pairing": args.pairing,
        "rows_loaded": len(rows),
        "directed_pairs": len(pairs),
        "skip_counts": skip_counts,
        "batch_size": args.batch_size,
        "total_batches": total_batches,
        "detail_csv": str(detail_csv),
        "summary_csv": str(summary_csv),
        "top_sites_by_abs_mean": top_sites,
        "note": "Attribution patching screen only; confirm top sites with exact activation patching before DAS.",
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
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--out-dir", default="results/attribution_patching")
    ap.add_argument("--run-name", default="")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
