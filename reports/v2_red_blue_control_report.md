# V2 Red/Blue DAS Control

## Purpose

This control tests whether the v2 head-to-low DAS transfer is specific to the
original lexical-licensing readout axis, or whether the same late-residual DAS
setup can transfer an unrelated binary next-token axis.

The prompts, clean/corrupt activation sources, v2 balanced-pool pairing, train
regime, eval regime, site, rank, and seed are unchanged. Only the readout
targets are replaced:

- clean target: ` red`
- corrupt target: ` blue`

## Tokenizer Check

Verified against the cached `EleutherAI/pythia-1.4b` tokenizer JSON at:

`/Users/tyronewhite/.cache/huggingface/hub/models--EleutherAI--pythia-1.4b/snapshots/fedc38a16eea3bd36a96b906d78d11d2ce18ed79/tokenizer.json`

The tokenizer model is BPE with a ByteLevel pre-tokenizer. The leading-space
targets are single vocabulary entries:

| target | tokenizer token | token id |
| --- | --- | --- |
| ` red` | `Ġred` | 2502 |
| ` blue` | `Ġblue` | 4797 |

## Existing V2 Control Baseline

Current aggregate comparison from `reports/v2_das_eval_summary.csv`, weighted by
the row counts in each subtask/direction cell:

| control | seed | regime | n | mean effect | mean normalized effect | patched success |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| dummy_pair | 17 | head | 406 | 0.0000 | nan | 0.5000 |
| dummy_pair | 17 | low | 1120 | 0.0000 | nan | 0.5000 |
| none | 17 | head | 312 | 12.4769 | 3.4605 | 0.9936 |
| none | 17 | low | 1120 | 11.2010 | 2.7103 | 0.9563 |
| random_direction | 17 | head | 312 | 0.0027 | -0.0046 | 0.2628 |
| random_direction | 17 | low | 1120 | 0.0027 | 0.0014 | 0.2795 |
| shuffled_label | 17 | head | 312 | -0.0126 | 0.5203 | 0.5192 |
| shuffled_label | 17 | low | 1120 | -0.0092 | 0.5912 | 0.5241 |
| red_blue | 17 | head | pending | pending | pending | pending |
| red_blue | 17 | low | pending | pending | pending | pending |

## Required Runs

Baseline red/blue preference:

```bash
python3 scripts/run_pythia_v2_sanity.py \
  --data data/aligned_templates_v2/lexical_licensing_v2_aligned.jsonl \
  --model EleutherAI/pythia-1.4b \
  --subtasks causative,inchoative \
  --regimes head,low \
  --target-mode fixed_pair \
  --fixed-expected-target " red" \
  --fixed-contrast-target " blue" \
  --batch-size 16 \
  --device cuda \
  --dtype bfloat16 \
  --allow-download \
  --out-dir results/v2_sanity \
  --run-name 20260623-pythia14b-v2-sanity-red-blue-head-low
```

DAS red/blue control:

```bash
python3 scripts/run_pythia_das_v1.py \
  --data data/aligned_templates_v2/lexical_licensing_v2_aligned.jsonl \
  --model EleutherAI/pythia-1.4b \
  --subtasks causative,inchoative \
  --train-regimes head \
  --eval-regimes head,low \
  --site resid_post_layer_23 \
  --rank 1 \
  --pairing balanced_pool \
  --seed 17 \
  --control red_blue \
  --batch-size 8 \
  --device cuda \
  --dtype bfloat16 \
  --allow-download \
  --out-dir results/das_v2_transfer \
  --run-name 20260623-pythia14b-v2-das-head-to-low-l23-red_blue-s17
```

After both runs complete, regenerate the report summaries:

```bash
python3 scripts/summarize_v2_results.py --run-prefix 20260623-pythia14b-v2
```

## Interpretation Gate

If `red_blue` has near-zero DAS effect on low, the existing head-to-low transfer
is more plausibly tied to the original lexical-licensing continuation axis.

If `red_blue` transfers strongly on low, the licensing-specific interpretation is
weaker because the same machinery can also transfer an arbitrary binary target
axis.
