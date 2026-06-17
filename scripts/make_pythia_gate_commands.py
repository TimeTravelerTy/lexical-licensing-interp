#!/usr/bin/env python3
"""Generate behavioral-gate commands for Pythia/FreqBLiMP scoring.

The actual scorer lives in the sibling `blimp-rare` repo. This script creates a
reviewable shell script that writes outputs back into this repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_MODELS = (
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
)
DEFAULT_REGIMES = ("head", "tail", "xtail")
DEFAULT_SUBTASKS = (
    "causative",
    "inchoative",
    "transitive",
    "intransitive",
    "drop_argument",
    "passive_1",
    "passive_2",
    "principle_A_domain_3",
)


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--blimp-rare-root",
        default="/Users/tyronewhite/masters_research_code/blimp-rare",
    )
    ap.add_argument(
        "--freq-blimp-root",
        default="/Users/tyronewhite/masters_research_code/freq-blimp",
    )
    ap.add_argument("--out", default="scripts/run_pythia_behavior_gate.sh")
    ap.add_argument("--run-timestamp", default="20260617-pythia-gate")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    repo = Path.cwd()
    blimp_rare = Path(args.blimp_rare_root)
    freq_blimp = Path(args.freq_blimp_root)
    scorer = blimp_rare / "scripts" / "score_acceptability_methods.py"
    out_dir = repo / "results" / "acceptability_pair_scores_pythia"
    manifest_dir = repo / "results" / "manifests" / "pythia_behavior_gate"

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"cd {shell_quote(str(blimp_rare))}",
        f"mkdir -p {shell_quote(str(out_dir))} {shell_quote(str(manifest_dir))}",
        "",
    ]
    for model in DEFAULT_MODELS:
        model_slug = model.split("/")[-1].replace(".", "_")
        for regime in DEFAULT_REGIMES:
            data_paths = [
                freq_blimp / "data" / "freqblimp" / regime / f"{subtask}.jsonl"
                for subtask in DEFAULT_SUBTASKS
            ]
            manifest = manifest_dir / f"{args.run_timestamp}-{regime}-{model_slug}.json"
            data_args = " ".join(shell_quote(str(path)) for path in data_paths)
            cmd = (
                "python3 "
                f"{shell_quote(str(scorer))} "
                f"--data {data_args} "
                f"--models {shell_quote(model)} "
                "--methods lp_readout in_template_lp "
                "--variant freq "
                f"--batch-size {args.batch_size} "
                f"--max-length {args.max_length} "
                f"--device {shell_quote(args.device)} "
                f"--dtype {shell_quote(args.dtype)} "
                f"--out-dir {shell_quote(str(out_dir))} "
                f"--run-timestamp {shell_quote(f'{args.run_timestamp}-{regime}')} "
                f"--manifest-out {shell_quote(str(manifest))}"
            )
            lines.extend(
                [
                    f"echo '[Run] {model} {regime}'",
                    cmd,
                    "",
                ]
            )

    out_path = repo / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_path.chmod(0o755)
    print(out_path)


if __name__ == "__main__":
    main()

