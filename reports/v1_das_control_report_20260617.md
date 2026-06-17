# V1 DAS Control Report, 2026-06-17

## Scope

This report covers infrastructure DAS on the fixed-subject v1 scaffold:

- model: `EleutherAI/pythia-1.4b`
- site: `resid_post_layer_23`
- token: prompt-final `verb_final_subtoken`
- rank: 1
- primary contrast: `causative,inchoative`

Important implementation correction: an earlier DAS run used
`output_hidden_states` to read clean activations but patched module outputs.
For GPT-NeoX, the final `output_hidden_states` entry is after
`final_layer_norm`, so that mixed representation spaces. The corrected
`hookfix` runs capture the clean activation from the same module hook used for
patching. Treat non-`hookfix` DAS outputs as superseded.

## Completed Jobs

- `7970733` `llv1_das_l23`: trained DAS, hook-aligned.
- `7970734` `llv1_das_shuf`: retrain-on-shuffled-label control.
- `7970735` `llv1_das_dummy`: dummy-verb/no-informative-verb control.
- `7970736` `llv1_das_rand`: random-direction same-site/same-rank control.

Only stderr signature: Transformers `torch_dtype` deprecation warning.

## Local Outputs

Under `results/das_v1/`:

- `20260617-pythia14b-das-primary-l23-hookfix.*`
- `20260617-pythia14b-das-primary-l23-shuffled-hookfix.*`
- `20260617-pythia14b-das-primary-l23-dummy-hookfix.*`
- `20260617-pythia14b-das-primary-l23-random-hookfix.*`

## Headline Metrics

| Run | Control | Eval n | Eval patched metric | Eval effect | Eval success | Eval normalized |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| primary | none | 34 | 7.341 | 9.834 | 0.912 | 2.413 |
| shuffled | retrain on permuted labels | 34 | 0.296 | -0.364 | 0.588 | 0.379 |
| dummy | replace verbs with `do` | 44 | 0.000 | 0.000 | 0.500 | n/a |
| random | untrained random rank-1 direction | 34 | -2.467 | 0.026 | 0.265 | 0.009 |

## Interpretation

The trained v1 DAS subspace at `resid_post_layer_23` strongly moves held-out
primary examples in the intended direction. It overshoots the clean-prompt
metric, so the right readout is intervention success/effect, not faithful
reconstruction of the clean state.

The controls behave as expected:

- `random_direction` has effectively zero effect. This says the trained
  direction is not just any rank-1 direction at the site.
- `shuffled_label` can fit some training structure but drops sharply on held-out
  eval. This is the main Geiger-style interpretability-illusion check because
  the rotation/subspace is retrained on permuted labels.
- `dummy_verb` has zero mean effect and chance success. This says the current
  DAS machinery is not producing a useful intervention when the verb signal is
  removed.

The dummy-verb result does not answer the subject-animacy confound. It only
answers: "Can this fixed template be solved with no informative verb?" The
animacy question requires v2 data where subject type and target frame are
decorrelated, for example animate-subject non-alternating intransitives and
patient-subject obligatory transitives.

## Next Step

Do not broaden claims from v1. The next useful work is either:

1. run a hook-aligned layer sweep over `resid_post_layer_16` through
   `resid_post_layer_23` to see whether layer 23 is special after training, or
2. build v2 data with subject/frame decorrelation and repeat this DAS protocol.

For final claims, prioritize option 2.
