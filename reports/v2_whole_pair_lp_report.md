# V2 Whole-Pair LP Results, 2026-06-25

## Setup

- model: `EleutherAI/pythia-1.4b`
- data: `data/aligned_templates_v2/lexical_licensing_v2_aligned.jsonl`
- rows loaded: 3,560
- balanced whole-pair comparisons: 1,320
- TSUBAME job: `8011769` (`llv2_whlp`)
- code commit: `31f953e`

The score is full-sentence log probability:

```text
LP(full grammatical-frame sentence) - LP(full ungrammatical-frame sentence)
```

This differs from the old verb-final `the`/`.` proxy and does not directly ask
whether an object follows the verb.

## Summary

| regime | subtask | n | mean good-bad LP | success rate | min | max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| head | causative | 440 | 4.654 | 0.893 | -5.997 | 14.384 |
| head | inchoative | 320 | 3.026 | 0.809 | -4.172 | 11.095 |
| low | causative | 280 | 5.506 | 0.989 | -1.162 | 14.130 |
| low | inchoative | 280 | 5.568 | 0.932 | -6.315 | 13.583 |

## Interpretation

Whole-pair LP recovers the grammatical contrast in all primary cells. This
supports the readout-target diagnosis: the previous DAS direction was trained
on an object/no-object next-token proxy, while full-sentence likelihood contains
enough information to prefer the intended grammatical frame even for inchoatives.

The next discriminator is whole-pair-trained DAS. If that still recovers the
object-prior direction, the stronger claim is that the model behavior is
available in full-sentence scoring but not cleanly localized in the tested
subspace/site.

## Outputs

- `results/whole_pair_lp/20260625-pythia14b-v2-whole-pair-lp.detail.csv`
- `results/whole_pair_lp/20260625-pythia14b-v2-whole-pair-lp.summary.csv`
- `results/whole_pair_lp/20260625-pythia14b-v2-whole-pair-lp.source_summary.csv`
- `results/whole_pair_lp/20260625-pythia14b-v2-whole-pair-lp.manifest.json`
