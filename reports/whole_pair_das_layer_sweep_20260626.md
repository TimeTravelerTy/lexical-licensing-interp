# Whole-Pair DAS Layer Sweep, 2026-06-26

## Setup

- model: `EleutherAI/pythia-1.4b`
- data: `data/aligned_templates_v2/lexical_licensing_v2_aligned.jsonl`
- layers: `resid_post_layer_16`, `18`, `20`, compared with prior `23`
- rank: 1
- train: `head`
- eval: held-out `head` plus all `low`
- controls per layer: `none`, `shuffled_label`, `random_direction`
- TSUBAME job: `8017460` (`llwp_das_l`)
- code commit: `8375efc`

This follows the same whole-pair DAS protocol as the layer-23 run. The
injection objective is post-anchor suffix log probability because the
intervention is at the prompt-final verb token.

## Headline

The lower layers do not fix the read-side licensing-axis problem. Layers 16,
18, 20, and 23 all show strong causal injection effects on the suffix
preference objective, but all four layers fail the causative-calibrated
inchoative validation cell.

So the layer-23 result was not just a final-layer next-token-prediction
artifact. The learned rank-1 whole-pair direction is causal for the trained
preference, but its read-side sign remains subtask-polarized rather than a
stable licensed-versus-unlicensed axis.

## Injection Strength

These are eval split averages over `good_to_bad` and `bad_to_good`. Mean signed
effect is positive when the intervention moves toward the objective clean side.

| layer | regime | subtask | mean signed suffix effect | success |
| ---: | --- | --- | ---: | ---: |
| 16 | head | causative | 3.456 | 1.000 |
| 16 | head | inchoative | 2.560 | 1.000 |
| 16 | low | causative | 2.786 | 0.966 |
| 16 | low | inchoative | 1.371 | 0.936 |
| 18 | head | causative | 3.719 | 1.000 |
| 18 | head | inchoative | 2.798 | 1.000 |
| 18 | low | causative | 2.806 | 0.961 |
| 18 | low | inchoative | 1.649 | 0.963 |
| 20 | head | causative | 4.084 | 1.000 |
| 20 | head | inchoative | 2.899 | 0.984 |
| 20 | low | causative | 2.939 | 0.964 |
| 20 | low | inchoative | 1.681 | 0.954 |
| 23 | head | causative | 5.091 | 1.000 |
| 23 | head | inchoative | 3.844 | 0.968 |
| 23 | low | causative | 4.155 | 0.968 |
| 23 | low | inchoative | 2.584 | 0.939 |

Injection gets stronger later, but it is already reliable at layer 16.

## Projection/Read-Side Separation

Projection reads the learned subspace from verb-anchor activations without
intervention.

| layer | regime | subtask | good-bad coord | orientation-free AUC | best threshold acc |
| ---: | --- | --- | ---: | ---: | ---: |
| 16 | head | causative | -12.159 | 1.000 | 0.991 |
| 16 | head | inchoative | 8.323 | 0.995 | 0.984 |
| 16 | low | causative | -10.554 | 0.971 | 0.948 |
| 16 | low | inchoative | 6.666 | 0.891 | 0.839 |
| 18 | head | causative | -14.655 | 1.000 | 0.991 |
| 18 | head | inchoative | 10.460 | 0.996 | 0.984 |
| 18 | low | causative | -12.462 | 0.965 | 0.939 |
| 18 | low | inchoative | 9.343 | 0.912 | 0.834 |
| 20 | head | causative | -15.771 | 0.999 | 0.991 |
| 20 | head | inchoative | 11.868 | 0.995 | 0.968 |
| 20 | low | causative | -13.889 | 0.964 | 0.930 |
| 20 | low | inchoative | 10.598 | 0.917 | 0.838 |
| 23 | head | causative | -17.471 | 1.000 | 0.991 |
| 23 | head | inchoative | 11.634 | 0.986 | 0.968 |
| 23 | low | causative | -15.175 | 0.969 | 0.938 |
| 23 | low | inchoative | 9.553 | 0.903 | 0.825 |

The direction separates examples strongly at every layer. The sign flip is
also present at every layer: causative good-bad is negative and inchoative
good-bad is positive.

## Validation Cell

The sign is calibrated by the causative good-minus-bad margin. A licensing axis
should then give a positive calibrated inchoative good-minus-bad margin.

| layer | regime | causative good-bad | inchoative good-bad | calibrated inchoative margin | pass |
| ---: | --- | ---: | ---: | ---: | ---: |
| 16 | head | -12.159 | 8.323 | -8.323 | 0 |
| 16 | low | -10.554 | 6.666 | -6.666 | 0 |
| 18 | head | -14.655 | 10.460 | -10.460 | 0 |
| 18 | low | -12.462 | 9.343 | -9.343 | 0 |
| 20 | head | -15.771 | 11.868 | -11.868 | 0 |
| 20 | low | -13.889 | 10.598 | -10.598 | 0 |
| 23 | head | -17.471 | 11.634 | -11.634 | 0 |
| 23 | low | -15.175 | 9.553 | -9.553 | 0 |

## Controls

The `random_direction` and `shuffled_label` controls remain near chance/zero
on the injection effect and do not produce a stable, meaningful validation
pattern. Some control rows can pass the sign-calibrated validation cell by
accident because their margins are small and unstable; that is not evidence for
a licensing axis.

## Interpretation

The whole-pair objective produces a robust causal intervention across the late
residual stream. However, the projection/read-side test says the learned
rank-1 directions are not stable licensing directions. The same subspace that
can causally move suffix likelihood separates causatives and inchoatives in
opposite signed orientations.

This weakens the hypothesis that the layer-23 failure was only caused by the
last layer focusing on next-token prediction. The problem appears already by
layer 16 under this rank-1 whole-pair DAS setup.

## Outputs

Local result files are under `results/whole_pair_das/` with prefixes:

```text
20260626-pythia14b-v2-wholepair-das-l{16,18,20}-{none,shuffled_label,random_direction}-s17
```

Layer 23 comparison files use:

```text
20260626-pythia14b-v2-wholepair-das-l23-{none,shuffled_label,random_direction}-s17
```
