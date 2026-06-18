# V1 Head-To-Tail DAS Transfer Report, 2026-06-18

## Question

Does a rank-1 DAS direction trained only on `head` causative/inchoative pairs
remain usable as an intervention direction inside `tail` and `xtail` pairs?

Here, transfer means:

```text
train subspace on head source/base pairs
freeze subspace
evaluate by intervening within tail or xtail source/base pairs
```

This is different from the earlier pooled v1 DAS run, which trained and
evaluated over `head,tail,xtail` together.

## Setup

- model: `EleutherAI/pythia-1.4b`
- site: `resid_post_layer_23`
- token: prompt-final `verb_final_subtoken`
- rank: 1
- subtasks: `causative,inchoative`
- train regimes: `head`
- eval regimes: `head,tail,xtail`
- head split: lemma-pair held-out, `eval_frac=0.25`

The head transfer run used 56 directed training pairs. Evaluation used:

- 20 held-out head directed pairs;
- 50 tail directed pairs;
- 34 xtail directed pairs.

Completed TSUBAME jobs:

- `7972933` trained head-only DAS.
- `7972934` retrain-on-shuffled-label control.
- `7972935` dummy-verb control.
- `7972936` random-direction control.

Only stderr signature: Transformers `torch_dtype` deprecation warning.

## Outputs

Local artifacts are under `results/das_v1_transfer/`:

- `20260618-pythia14b-das-transfer-head-l23.*`
- `20260618-pythia14b-das-transfer-head-l23-shuffled.*`
- `20260618-pythia14b-das-transfer-head-l23-dummy.*`
- `20260618-pythia14b-das-transfer-head-l23-random.*`

## Main Result

| Eval regime | n | Mean patched metric | Mean effect | Success |
| --- | ---: | ---: | ---: | ---: |
| head held-out | 20 | 12.846 | 15.756 | 1.000 |
| tail transfer | 50 | 7.141 | 8.795 | 0.860 |
| xtail transfer | 34 | 6.581 | 7.792 | 0.794 |

The head-trained direction transfers clearly to lower-frequency regimes. The
effect weakens from head to tail to xtail, but remains strong and usually flips
the patched base prompt toward the source target.

Overall eval metrics across all eval regimes:

- corrupt/base metric: `-1.751`
- patched metric: `8.055`
- source/clean metric: `1.751`
- mean effect: `9.806`
- success: `0.865`
- normalized effect: `2.013`

As before, the intervention overshoots the source-prompt metric. This should be
reported as a learned direction that strongly controls the proxy target
preference, not as faithful reconstruction of the natural source state.

## Controls

| Control | Eval regime | n | Mean patched metric | Mean effect | Success |
| --- | --- | ---: | ---: | ---: | ---: |
| shuffled-label retrain | head | 20 | -0.233 | -0.650 | 0.650 |
| shuffled-label retrain | tail | 50 | -0.329 | 0.077 | 0.480 |
| shuffled-label retrain | xtail | 34 | -0.571 | 0.236 | 0.412 |
| dummy verb | head | 24 | 0.000 | 0.000 | 0.500 |
| dummy verb | tail | 50 | 0.000 | 0.000 | 0.500 |
| dummy verb | xtail | 34 | 0.000 | 0.000 | 0.500 |
| random direction | head | 20 | -2.967 | -0.056 | 0.200 |
| random direction | tail | 50 | -1.650 | 0.003 | 0.300 |
| random direction | xtail | 34 | -1.208 | 0.004 | 0.382 |

The controls support the transfer readout:

- `random_direction` has near-zero effect, so the learned direction matters.
- `dummy_verb` is exactly chance/no effect, so the no-verb scaffold does not
  produce a useful intervention.
- `shuffled_label` is near chance on transfer regimes, especially tail/xtail.
  It is not a competitive explanation for the main transfer result.

## Interpretation

This is the strongest v1 result so far: a direction trained on high-frequency
head examples is usable for interventions within tail and xtail examples.

Still, v1 remains a fixed-subject scaffold. The result supports infrastructure
and motivates v2, but does not yet rule out subject/frame confounds in the final
causal claim. For final claims, v2 should decorrelate subject type from target
frame and then repeat this exact transfer protocol.
