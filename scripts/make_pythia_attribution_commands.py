#!/usr/bin/env python3
"""Generate the next Pythia 1.4B attribution-patching commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_pythia_attribution_patching import build_directed_pairs, load_aligned_rows


RUNS = (
    {
        "name": "primary",
        "subtasks": "causative,inchoative",
        "why": "strongest and most stable behavioral gate signal",
    },
    {
        "name": "secondary",
        "subtasks": "transitive,intransitive",
        "why": "above chance but weaker behavioral gate signal",
    },
    {
        "name": "drop_argument",
        "subtasks": "drop_argument",
        "why": "separate contrast; tail behavior is weak/near chance",
    },
)


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/pythia_attribution_next_steps_20260617.md")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--run-prefix", default="20260617-pythia14b-ap")
    args = ap.parse_args()

    repo = Path.cwd()
    runner = repo / "scripts" / "run_pythia_attribution_patching.py"
    exact_runner = repo / "scripts" / "run_pythia_exact_patching.py"
    job_runner = repo / "scripts" / "make_tsubame_v1_job.py"
    data = repo / "data" / "aligned_templates" / "lexical_licensing_aligned.jsonl"
    out_dir = repo / "results" / "attribution_patching"

    lines = [
        "# Pythia 1.4B Attribution Patching Next Steps",
        "",
        "Run order:",
        "",
    ]
    for idx, run in enumerate(RUNS, start=1):
        lines.append(f"{idx}. `{run['name']}`: `{run['subtasks']}` ({run['why']}).")
    lines.extend(["", "Expected matched-pair scope:", ""])
    for run in RUNS:
        rows = load_aligned_rows(data, set(run["subtasks"].split(",")), {"head", "tail", "xtail"})
        pairs, skips = build_directed_pairs(rows, {"good_to_bad", "bad_to_good"}, None)
        undirected_pairs = len(pairs) // 2
        missing = skips.get("missing_good_or_bad_side", 0)
        lines.append(
            f"- `{run['name']}`: {len(rows)} aligned rows, "
            f"{undirected_pairs} matched prompt pairs, {len(pairs)} directed pairs "
            f"({missing} unpaired aligned side groups)."
        )
    lines.extend(
        [
            "",
            "All runs use aligned rows only, both patch directions, all regimes, and the per-example `verb_final_subtoken` anchor.",
            "The output files are detail CSV, summary CSV, and manifest JSON under `results/attribution_patching/`.",
            "",
            "## Commands",
            "",
            "### Attribution Screen",
            "",
            "```bash",
            "cd " + shell_quote(str(repo)),
        ]
    )
    for run in RUNS:
        run_name = f"{args.run_prefix}-{run['name']}"
        cmd = (
            "python3 "
            f"{shell_quote(str(runner))} "
            f"--data {shell_quote(str(data))} "
            f"--model {shell_quote(args.model)} "
            f"--subtasks {shell_quote(run['subtasks'])} "
            "--regimes head,tail,xtail "
            "--directions good_to_bad,bad_to_good "
            f"--batch-size {args.batch_size} "
            f"--device {shell_quote(args.device)} "
            f"--dtype {shell_quote(args.dtype)} "
            f"--out-dir {shell_quote(str(out_dir))} "
            f"--run-name {shell_quote(run_name)}"
        )
        lines.append(cmd)
    lines.extend(
        [
            "```",
            "",
            "### Exact Patching Confirmation",
            "",
            "Run this after the primary attribution summary exists. It confirms the top residual-stream sites with real activation replacement.",
            "",
            "```bash",
            "cd " + shell_quote(str(repo)),
            "python3 "
            f"{shell_quote(str(exact_runner))} "
            f"--data {shell_quote(str(data))} "
            f"--model {shell_quote(args.model)} "
            "--subtasks causative,inchoative "
            "--regimes head,tail,xtail "
            "--directions good_to_bad,bad_to_good "
            f"--summary-csv {shell_quote(str(out_dir / f'{args.run_prefix}-primary.summary.csv'))} "
            "--top-k 6 "
            f"--batch-size {args.batch_size} "
            f"--device {shell_quote(args.device)} "
            f"--dtype {shell_quote(args.dtype)} "
            f"--out-dir {shell_quote(str(repo / 'results' / 'exact_patching'))} "
            f"--run-name {shell_quote(f'{args.run_prefix}-primary-exact')}",
            "```",
            "",
            "### TSUBAME Job Scripts",
            "",
            "Generate temporary SGE scripts after the repo is present on TSUBAME. Copy them to the remote `jobs/` directory and submit with `qsub -g tga-sip_arase`.",
            "",
            "```bash",
            "cd " + shell_quote(str(repo)),
            f"python3 {shell_quote(str(job_runner))} primary_attribution",
            f"python3 {shell_quote(str(job_runner))} primary_exact",
            f"python3 {shell_quote(str(job_runner))} secondary_attribution",
            f"python3 {shell_quote(str(job_runner))} drop_argument_attribution",
            "```",
            "",
            "## Interpretation Gate",
            "",
            "- Treat these as localization screens only.",
            "- Promote a site to exact patching only if it is consistent across direction and at least the head/tail regimes.",
            "- Do not pool `drop_argument` with causative/inchoative unless its top sites independently match.",
            "- Do not train DAS on this fixed-subject scaffold for final claims; use it to debug localization and the patching infrastructure.",
        ]
    )

    out = repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
