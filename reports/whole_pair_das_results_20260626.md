# Whole-Pair DAS Injection And Projection Results, 2026-06-26

## Setup

- model: `EleutherAI/pythia-1.4b`
- data: `data/aligned_templates_v2/lexical_licensing_v2_aligned.jsonl`
- site: `resid_post_layer_23`
- rank: 1
- train: `head`
- eval: held-out `head` plus all `low`
- subtasks: `causative,inchoative`
- directed interventions: `good_to_bad,bad_to_good`
- split: lemma-pair, `eval_frac=0.25`, seed 17
- TSUBAME job: `8016969` (`llwp_das_h`)
- code commit: `6d704d5`

The intervention is at the prompt-final verb token. Since a hook at that token
cannot change the probability already assigned to the verb token itself, the
training/injection metric is post-anchor suffix log probability. The output
also records full-sentence log probability, but the causal effect is only
interpretable on the suffix after the verb anchor.

## Headline

Whole-pair DAS injection strongly moves the model's post-anchor sentence score
in the intended direction. The learned direction is causal for the suffix
preference objective.

The read-side projection test is still not a clean licensing axis. When the
rank-1 coordinate is calibrated so that causative-good is positive, the
inchoative good-minus-bad margin is negative in both eval regimes. The learned
axis separates examples strongly, but it flips between causatives and
inchoatives.

## Injection: Eval Split

Mean signed suffix effect is positive when the intervention moves toward the
objective clean side. Success is the fraction of examples with positive signed
effect.

| control | regime | subtask | direction | n | mean signed suffix effect | success |
| --- | --- | --- | --- | ---: | ---: | ---: |
| none | head | causative | bad_to_good | 115 | 5.894 | 1.000 |
| none | head | causative | good_to_bad | 115 | 4.288 | 1.000 |
| none | head | inchoative | bad_to_good | 62 | 4.309 | 0.968 |
| none | head | inchoative | good_to_bad | 62 | 3.379 | 0.968 |
| none | low | causative | bad_to_good | 280 | 4.737 | 0.968 |
| none | low | causative | good_to_bad | 280 | 3.574 | 0.968 |
| none | low | inchoative | bad_to_good | 280 | 2.809 | 0.939 |
| none | low | inchoative | good_to_bad | 280 | 2.360 | 0.939 |

Controls behave as expected:

- `random_direction`: eval effects are near zero, with success around chance.
- `shuffled_label`: eval effects are near zero, with success around chance.
- shuffled-label training loss remains high, while the real-label training loss
  quickly drops near zero.

## Projection/Read-Side: Eval Split

Projection reads the learned subspace from full-sentence verb-anchor
activations without intervention.

| control | regime | subtask | good-bad coord | orientation-free AUC | best threshold acc |
| --- | --- | --- | ---: | ---: | ---: |
| none | head | causative | -17.471 | 1.000 | 0.991 |
| none | head | inchoative | 11.634 | 0.986 | 0.968 |
| none | low | causative | -15.175 | 0.969 | 0.938 |
| none | low | inchoative | 9.553 | 0.903 | 0.825 |

The direction separates good from bad within each subtask, but the sign flips:
causative good-minus-bad is negative, while inchoative good-minus-bad is
positive.

## Validation Cell

The sign is calibrated by the causative good-minus-bad margin. A licensing
axis should then produce a positive calibrated inchoative good-minus-bad
margin.

| regime | causative good-bad | inchoative good-bad | calibrated inchoative margin | pass |
| --- | ---: | ---: | ---: | ---: |
| head | -17.471 | 11.634 | -11.634 | 0 |
| low | -15.175 | 9.553 | -9.553 | 0 |

## Interpretation

The whole-pair objective fixes the old next-token readout problem for the
injection test: the intervention can causally raise or lower the continuation
likelihood in the expected direction for both causative and inchoative cells.

But the read-side learned coordinate does not validate as a single
licensed-versus-unlicensed axis. It remains subtask-polarized: causative and
inchoative examples separate in opposite signed directions. This means the
current rank-1 layer-23 whole-pair DAS direction is a strong causal instrument
for the trained suffix preference, but not yet evidence for a stable lexical
licensing representation.

## Outputs

Local copies are under `results/whole_pair_das/` with prefix:

```text
20260626-pythia14b-v2-wholepair-das-l23-{none,shuffled_label,random_direction}-s17
```

Primary files:

- `*.injection_summary.csv`
- `*.projection_summary.csv`
- `*.validation_cell_summary.csv`
- `*.manifest.json`
- `*.subspace.pt`
