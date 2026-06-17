#!/usr/bin/env python3
"""Build and optionally tokenizer-verify aligned lexical-licensing templates."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REGIME_FILES = {
    "head": "/Users/tyronewhite/masters_research_code/blimp-rare/data/processed/freq_blimp_head_3_5-5_5_cond1.jsonl",
    "tail": "/Users/tyronewhite/masters_research_code/blimp-rare/data/processed/freq_blimp_tail_2_4-3_2_cond1.jsonl",
    "xtail": "/Users/tyronewhite/masters_research_code/blimp-rare/data/processed/freq_blimp_xtail_1_2-2_2_cond1.jsonl",
}

SUBTASKS = {
    "causative",
    "inchoative",
    "transitive",
    "intransitive",
    "drop_argument",
}

TEMPLATES = {
    "object_frame": {
        "prompt": "In the scene, the technician will {verb}",
        "prefix_family": "agent_object_probe",
        "good_target": " the",
        "bad_target": ".",
        "description": "Tests licensing of an object continuation after an animate-subject verb.",
    },
    "object_frame_worker": {
        "prompt": "After lunch, the worker will {verb}",
        "prefix_family": "agent_object_probe",
        "good_target": " the",
        "bad_target": ".",
        "description": "Second object-frame scaffold with a different fixed prefix.",
    },
    "inchoative_frame": {
        "prompt": "In the scene, the glass will {verb}",
        "prefix_family": "patient_no_object_probe",
        "good_target": ".",
        "bad_target": " the",
        "description": "Tests licensing of no-object/inchoative continuation for patient-subject verbs.",
    },
    "inchoative_frame_tomorrow": {
        "prompt": "Tomorrow, the door will {verb}",
        "prefix_family": "patient_no_object_probe",
        "good_target": ".",
        "bad_target": " the",
        "description": "Second no-object scaffold with a different fixed prefix.",
    },
    "drop_object_frame": {
        "prompt": "In the scene, the artist will {verb}",
        "prefix_family": "agent_drop_object_probe",
        "good_target": ".",
        "bad_target": " the",
        "description": "Tests verbs that permit object drop against strict transitives.",
    },
}

SUBTASK_TO_TEMPLATE_AND_LABELS = {
    "causative": ("object_frame", "object_licensed", "object_unlicensed"),
    "transitive": ("object_frame_worker", "object_licensed", "object_unlicensed"),
    "inchoative": ("inchoative_frame", "no_object_licensed", "no_object_unlicensed"),
    "intransitive": ("inchoative_frame_tomorrow", "no_object_licensed", "no_object_unlicensed"),
    "drop_argument": ("drop_object_frame", "drop_object_licensed", "drop_object_unlicensed"),
}


@dataclass(frozen=True)
class VerbCandidate:
    regime: str
    subtask: str
    side: str
    lemma: str
    surface: str
    tag: str | None
    base_frame: str | None
    src_lemma: str | None
    source_idx: int
    source_text: str


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def verb_swaps(meta: dict[str, Any], key: str) -> list[dict[str, Any]]:
    swaps = meta.get(key)
    if not isinstance(swaps, list):
        return []
    return [s for s in swaps if s.get("pos") == "verb" and s.get("lemma") and s.get("new")]


def collect_candidates(limit_per_bucket: int) -> list[VerbCandidate]:
    seen: set[tuple[str, str, str, str]] = set()
    counts: Counter[tuple[str, str, str]] = Counter()
    candidates: list[VerbCandidate] = []
    for regime, raw_path in REGIME_FILES.items():
        path = Path(raw_path)
        for rec in iter_jsonl(path):
            subtask = rec.get("subtask")
            if subtask not in SUBTASKS:
                continue
            idx = int(rec.get("idx", len(candidates)))
            meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
            for side, swap_key, text_key in (
                ("good", "g_swaps", "good_freq"),
                ("bad", "b_swaps", "bad_freq"),
            ):
                bucket = (regime, subtask, side)
                if counts[bucket] >= limit_per_bucket:
                    continue
                swaps = verb_swaps(meta, swap_key)
                if not swaps:
                    continue
                swap = swaps[0]
                lemma = str(swap["lemma"]).strip().lower()
                surface = str(swap["new"]).strip().lower()
                if not lemma or " " in lemma or "-" in lemma:
                    continue
                key = (regime, subtask, side, lemma)
                if key in seen:
                    continue
                seen.add(key)
                counts[bucket] += 1
                candidates.append(
                    VerbCandidate(
                        regime=regime,
                        subtask=subtask,
                        side=side,
                        lemma=lemma,
                        surface=surface,
                        tag=swap.get("tag"),
                        base_frame=swap.get("base_frame"),
                        src_lemma=swap.get("src_lemma"),
                        source_idx=idx,
                        source_text=str(rec.get(text_key, "")),
                    )
                )
    return candidates


def make_row(candidate: VerbCandidate) -> dict[str, Any]:
    template_id, good_label, bad_label = SUBTASK_TO_TEMPLATE_AND_LABELS[candidate.subtask]
    template = TEMPLATES[template_id]
    label = good_label if candidate.side == "good" else bad_label
    prompt = template["prompt"].format(verb=candidate.lemma)
    expected_target = template["good_target"] if candidate.side == "good" else template["bad_target"]
    contrast_target = template["bad_target"] if candidate.side == "good" else template["good_target"]
    return {
        "regime": candidate.regime,
        "subtask": candidate.subtask,
        "template_id": template_id,
        "template_prompt": template["prompt"],
        "prefix_family": template["prefix_family"],
        "prompt": prompt,
        "verb": candidate.lemma,
        "source_surface": candidate.surface,
        "source_tag": candidate.tag,
        "base_frame": candidate.base_frame,
        "src_lemma": candidate.src_lemma,
        "side": candidate.side,
        "licensing_label": label,
        "expected_target": expected_target,
        "contrast_target": contrast_target,
        "source_idx": candidate.source_idx,
        "source_text": candidate.source_text,
        "intervention_anchor": "verb_final_subtoken",
        "anchor_index_policy": "per_example_last_prompt_token",
        "decision_target_is_proxy": True,
        "alignment_status": "unverified",
    }


def load_tokenizer(model_name: str, local_files_only: bool):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install transformers to run tokenizer verification.") from exc
    return AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)


def verify_rows(rows: list[dict[str, Any]], model_name: str, local_files_only: bool) -> dict[str, Any]:
    tokenizer = load_tokenizer(model_name, local_files_only=local_files_only)
    summary: dict[str, Any] = {"model": model_name, "groups": {}}
    target_token_lengths = {}
    for target in sorted({r["expected_target"] for r in rows} | {r["contrast_target"] for r in rows}):
        target_token_lengths[target] = len(tokenizer.encode(target, add_special_tokens=False))

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["regime"], row["subtask"], row["template_id"])].append(row)

    for group_key, group_rows in grouped.items():
        regime, subtask, template_id = group_key
        prompt_counts = Counter()
        verb_region_counts = Counter()
        prefix_counts = Counter()
        for row in group_rows:
            prefix = row["template_prompt"].split("{verb}")[0]
            prompt = row["prompt"]
            prefix_len = len(tokenizer.encode(prefix, add_special_tokens=False))
            prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
            verb_region_len = len(tokenizer.encode(" " + row["verb"], add_special_tokens=False))
            row["tokenizer_model"] = model_name
            row["prefix_token_count"] = prefix_len
            row["prompt_token_count"] = prompt_len
            row["verb_region_token_count"] = verb_region_len
            row["anchor_token_index"] = prompt_len - 1
            row["decision_token_index"] = prompt_len - 1
            row["expected_target_token_count"] = target_token_lengths[row["expected_target"]]
            row["contrast_target_token_count"] = target_token_lengths[row["contrast_target"]]
            prompt_counts[prompt_len] += 1
            verb_region_counts[verb_region_len] += 1
            prefix_counts[prefix_len] += 1

        retained_prompt_len, retained_n = prompt_counts.most_common(1)[0]
        retained_prefix_len, _ = prefix_counts.most_common(1)[0]
        for row in group_rows:
            aligned = (
                row["prompt_token_count"] == retained_prompt_len
                and row["prefix_token_count"] == retained_prefix_len
                and row["expected_target_token_count"] == 1
                and row["contrast_target_token_count"] == 1
            )
            row["alignment_status"] = "aligned" if aligned else "rejected_tokenization"
        summary["groups"]["|".join(group_key)] = {
            "n": len(group_rows),
            "retained_n": retained_n,
            "retained_prompt_token_count": retained_prompt_len,
            "decision_token_index": retained_prompt_len - 1,
            "prefix_token_counts": dict(prefix_counts),
            "prompt_token_counts": dict(prompt_counts),
            "verb_region_token_counts": dict(verb_region_counts),
        }
    summary["target_token_lengths"] = target_token_lengths
    return summary


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_report(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any] | None) -> None:
    counts = Counter((r["regime"], r["subtask"], r["side"], r["alignment_status"]) for r in rows)
    prefix_target_counts = Counter(
        (
            r["regime"],
            r["template_id"],
            r["template_prompt"],
            r["expected_target"],
            r["alignment_status"],
        )
        for r in rows
    )
    lines = [
        "# Aligned Template Build Report",
        "",
        "## Design Notes",
        "",
        "- Intervention anchor: `verb_final_subtoken`.",
        "- Anchor index policy: per-example last prompt token, not one global absolute index.",
        "- Decision target is a proxy: object introducer ` the` versus sentence end `.`.",
        "- The current fixed-subject templates are a preview; final DAS should add prefix-only, shuffled-label, and semantic-fit controls.",
        "",
        "## Template Preview",
        "",
    ]
    preview = rows[:12]
    for row in preview:
        lines.append(
            f"- `{row['template_id']}` `{row['regime']}` `{row['subtask']}` "
            f"`{row['side']}`: {row['prompt']!r} -> {row['expected_target']!r}"
        )
    lines.extend(["", "## Counts", ""])
    for key, value in sorted(counts.items()):
        regime, subtask, side, status = key
        lines.append(f"- `{regime}` `{subtask}` `{side}` `{status}`: {value}")
    lines.extend(["", "## Prefix/Target Balance", ""])
    for key, value in sorted(prefix_target_counts.items()):
        regime, template_id, template_prompt, target, status = key
        if status != "aligned":
            continue
        lines.append(
            f"- `{regime}` `{template_id}` target {target!r}: {value} aligned rows "
            f"for prefix `{template_prompt}`"
        )
    if summary is not None:
        lines.extend(["", "## Tokenization Summary", ""])
        lines.append(f"- model: `{summary['model']}`")
        lines.append(f"- target token lengths: `{summary['target_token_lengths']}`")
        for group, payload in sorted(summary["groups"].items()):
            lines.append(
                f"- `{group}`: retained {payload['retained_n']}/{payload['n']} "
                f"at prompt_token_count={payload['retained_prompt_token_count']} "
                f"(decision_index={payload['decision_token_index']}); "
                f"all prompt counts={payload['prompt_token_counts']}; "
                f"verb-region counts={payload['verb_region_token_counts']}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-bucket", type=int, default=80)
    ap.add_argument("--model", default=None, help="Optional tokenizer model for alignment verification.")
    ap.add_argument(
        "--allow-tokenizer-download",
        action="store_true",
        help="Allow transformers to download tokenizer files. Defaults to offline/cache-only loading.",
    )
    ap.add_argument("--out", default="data/aligned_templates/lexical_licensing_candidates.jsonl")
    ap.add_argument("--aligned-out", default="data/aligned_templates/lexical_licensing_aligned.jsonl")
    ap.add_argument("--report", default="reports/aligned_template_report.md")
    args = ap.parse_args()

    candidates = collect_candidates(limit_per_bucket=args.limit_per_bucket)
    rows = [make_row(candidate) for candidate in candidates]
    summary = (
        verify_rows(rows, args.model, local_files_only=not args.allow_tokenizer_download)
        if args.model
        else None
    )
    out = Path(args.out)
    write_jsonl(out, rows)
    if args.model:
        write_jsonl(Path(args.aligned_out), [r for r in rows if r["alignment_status"] == "aligned"])
    write_report(Path(args.report), rows, summary)
    print(out)
    print(args.report)


if __name__ == "__main__":
    main()
