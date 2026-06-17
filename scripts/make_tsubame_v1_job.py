#!/usr/bin/env python3
"""Create TSUBAME SGE job scripts for the v1 Pythia patching workflow."""

from __future__ import annotations

import argparse
from pathlib import Path


RUNS = {
    "primary_attribution": {
        "job_name": "llv1_ap_primary",
        "command": (
            "python3 scripts/run_pythia_attribution_patching.py "
            "--data data/aligned_templates/lexical_licensing_aligned.jsonl "
            "--model EleutherAI/pythia-1.4b "
            "--subtasks causative,inchoative "
            "--regimes head,tail,xtail "
            "--directions good_to_bad,bad_to_good "
            "--batch-size 8 "
            "--device cuda "
            "--dtype bfloat16 "
            "--allow-download "
            "--out-dir results/attribution_patching "
            "--run-name 20260617-pythia14b-ap-primary"
        ),
        "hours": "03:00:00",
    },
    "secondary_attribution": {
        "job_name": "llv1_ap_secondary",
        "command": (
            "python3 scripts/run_pythia_attribution_patching.py "
            "--data data/aligned_templates/lexical_licensing_aligned.jsonl "
            "--model EleutherAI/pythia-1.4b "
            "--subtasks transitive,intransitive "
            "--regimes head,tail,xtail "
            "--directions good_to_bad,bad_to_good "
            "--batch-size 8 "
            "--device cuda "
            "--dtype bfloat16 "
            "--allow-download "
            "--out-dir results/attribution_patching "
            "--run-name 20260617-pythia14b-ap-secondary"
        ),
        "hours": "03:00:00",
    },
    "drop_argument_attribution": {
        "job_name": "llv1_ap_drop",
        "command": (
            "python3 scripts/run_pythia_attribution_patching.py "
            "--data data/aligned_templates/lexical_licensing_aligned.jsonl "
            "--model EleutherAI/pythia-1.4b "
            "--subtasks drop_argument "
            "--regimes head,tail,xtail "
            "--directions good_to_bad,bad_to_good "
            "--batch-size 8 "
            "--device cuda "
            "--dtype bfloat16 "
            "--allow-download "
            "--out-dir results/attribution_patching "
            "--run-name 20260617-pythia14b-ap-drop_argument"
        ),
        "hours": "02:00:00",
    },
    "primary_exact": {
        "job_name": "llv1_exact_primary",
        "command": (
            "python3 scripts/run_pythia_exact_patching.py "
            "--data data/aligned_templates/lexical_licensing_aligned.jsonl "
            "--model EleutherAI/pythia-1.4b "
            "--subtasks causative,inchoative "
            "--regimes head,tail,xtail "
            "--directions good_to_bad,bad_to_good "
            "--summary-csv results/attribution_patching/20260617-pythia14b-ap-primary.summary.csv "
            "--top-k 6 "
            "--batch-size 8 "
            "--device cuda "
            "--dtype bfloat16 "
            "--allow-download "
            "--out-dir results/exact_patching "
            "--run-name 20260617-pythia14b-ap-primary-exact"
        ),
        "hours": "03:00:00",
    },
    "secondary_exact": {
        "job_name": "llv1_exact_secondary",
        "command": (
            "python3 scripts/run_pythia_exact_patching.py "
            "--data data/aligned_templates/lexical_licensing_aligned.jsonl "
            "--model EleutherAI/pythia-1.4b "
            "--subtasks transitive,intransitive "
            "--regimes head,tail,xtail "
            "--directions good_to_bad,bad_to_good "
            "--summary-csv results/attribution_patching/20260617-pythia14b-ap-secondary.summary.csv "
            "--top-k 6 "
            "--batch-size 8 "
            "--device cuda "
            "--dtype bfloat16 "
            "--allow-download "
            "--out-dir results/exact_patching "
            "--run-name 20260617-pythia14b-ap-secondary-exact"
        ),
        "hours": "03:00:00",
    },
    "drop_argument_exact": {
        "job_name": "llv1_exact_drop",
        "command": (
            "python3 scripts/run_pythia_exact_patching.py "
            "--data data/aligned_templates/lexical_licensing_aligned.jsonl "
            "--model EleutherAI/pythia-1.4b "
            "--subtasks drop_argument "
            "--regimes head,tail,xtail "
            "--directions good_to_bad,bad_to_good "
            "--summary-csv results/attribution_patching/20260617-pythia14b-ap-drop_argument.summary.csv "
            "--top-k 6 "
            "--batch-size 8 "
            "--device cuda "
            "--dtype bfloat16 "
            "--allow-download "
            "--out-dir results/exact_patching "
            "--run-name 20260617-pythia14b-ap-drop_argument-exact"
        ),
        "hours": "02:00:00",
    },
    "das_primary_l23": {
        "job_name": "llv1_das_l23",
        "command": (
            "python3 scripts/run_pythia_das_v1.py "
            "--data data/aligned_templates/lexical_licensing_aligned.jsonl "
            "--model EleutherAI/pythia-1.4b "
            "--site resid_post_layer_23 "
            "--subtasks causative,inchoative "
            "--regimes head,tail,xtail "
            "--directions good_to_bad,bad_to_good "
            "--rank 1 "
            "--epochs 30 "
            "--batch-size 8 "
            "--lr 0.05 "
            "--control none "
            "--device cuda "
            "--dtype bfloat16 "
            "--allow-download "
            "--out-dir results/das_v1 "
            "--run-name 20260617-pythia14b-das-primary-l23"
        ),
        "hours": "03:00:00",
    },
    "das_primary_l23_shuffled": {
        "job_name": "llv1_das_shuf",
        "command": (
            "python3 scripts/run_pythia_das_v1.py "
            "--data data/aligned_templates/lexical_licensing_aligned.jsonl "
            "--model EleutherAI/pythia-1.4b "
            "--site resid_post_layer_23 "
            "--subtasks causative,inchoative "
            "--regimes head,tail,xtail "
            "--directions good_to_bad,bad_to_good "
            "--rank 1 "
            "--epochs 30 "
            "--batch-size 8 "
            "--lr 0.05 "
            "--control shuffled_label "
            "--device cuda "
            "--dtype bfloat16 "
            "--allow-download "
            "--out-dir results/das_v1 "
            "--run-name 20260617-pythia14b-das-primary-l23-shuffled"
        ),
        "hours": "03:00:00",
    },
    "das_primary_l23_dummy": {
        "job_name": "llv1_das_dummy",
        "command": (
            "python3 scripts/run_pythia_das_v1.py "
            "--data data/aligned_templates/lexical_licensing_aligned.jsonl "
            "--model EleutherAI/pythia-1.4b "
            "--site resid_post_layer_23 "
            "--subtasks causative,inchoative "
            "--regimes head,tail,xtail "
            "--directions good_to_bad,bad_to_good "
            "--rank 1 "
            "--epochs 30 "
            "--batch-size 8 "
            "--lr 0.05 "
            "--control dummy_verb "
            "--dummy-verb do "
            "--device cuda "
            "--dtype bfloat16 "
            "--allow-download "
            "--out-dir results/das_v1 "
            "--run-name 20260617-pythia14b-das-primary-l23-dummy"
        ),
        "hours": "03:00:00",
    },
    "das_primary_l23_random": {
        "job_name": "llv1_das_rand",
        "command": (
            "python3 scripts/run_pythia_das_v1.py "
            "--data data/aligned_templates/lexical_licensing_aligned.jsonl "
            "--model EleutherAI/pythia-1.4b "
            "--site resid_post_layer_23 "
            "--subtasks causative,inchoative "
            "--regimes head,tail,xtail "
            "--directions good_to_bad,bad_to_good "
            "--rank 1 "
            "--epochs 30 "
            "--batch-size 8 "
            "--lr 0.05 "
            "--control random_direction "
            "--device cuda "
            "--dtype bfloat16 "
            "--allow-download "
            "--out-dir results/das_v1 "
            "--run-name 20260617-pythia14b-das-primary-l23-random"
        ),
        "hours": "02:00:00",
    },
}


def script_text(run_key: str, remote_root: str, conda_env: str) -> str:
    spec = RUNS[run_key]
    return f"""#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt={spec["hours"]}
#$ -N {spec["job_name"]}
#$ -o ./logs/$JOB_NAME.$JOB_ID.out
#$ -e ./logs/$JOB_NAME.$JOB_ID.err
#$ -V

set -e

module purge
module load cuda
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate {conda_env}

cd {remote_root}
mkdir -p logs results/attribution_patching results/exact_patching results/das_v1
export HF_HOME=/gs/fs/tga-sip_arase/tyrone/huggingface_cache
export HF_HUB_CACHE=$HF_HOME/hub
mkdir -p "$HF_HOME"

{spec["command"]}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", choices=sorted(RUNS))
    ap.add_argument(
        "--remote-root",
        default="/gs/fs/tga-sip_arase/tyrone/lexical_licensing_interp",
    )
    ap.add_argument("--conda-env", default="py312")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    text = script_text(args.run, args.remote_root, args.conda_env)
    if args.out:
        out = Path(args.out)
    else:
        out = Path("/tmp") / f"{RUNS[args.run]['job_name']}.sh"
    out.write_text(text, encoding="utf-8")
    out.chmod(0o755)
    print(out)


if __name__ == "__main__":
    main()
