# V1 Pythia 1.4B Patching Report, 2026-06-17

## Run Status

Remote root: `/gs/fs/tga-sip_arase/tyrone/lexical_licensing_interp`

Completed TSUBAME jobs:

- `7969295` `llv1_ap_primary`: causative/inchoative attribution patching.
- `7969310` `llv1_exact_primary`: causative/inchoative exact patching.
- `7969311` `llv1_ap_secondary`: transitive/intransitive attribution patching.
- `7969312` `llv1_ap_drop`: drop-argument attribution patching.
- `7969358` `llv1_exact_secondary`: transitive/intransitive exact patching.
- `7969359` `llv1_exact_drop`: drop-argument exact patching.

Job `7969278` failed before computation because the model was not cached and the
first job used offline loading. This was fixed by using a project HF cache at
`/gs/fs/tga-sip_arase/tyrone/huggingface_cache` and passing
`--allow-download`.

Only stderr signature in completed jobs:

- `torch_dtype` deprecation warning from Transformers.

## Local Outputs

Attribution outputs:

- `results/attribution_patching/20260617-pythia14b-ap-primary.{detail.csv,summary.csv,manifest.json}`
- `results/attribution_patching/20260617-pythia14b-ap-secondary.{detail.csv,summary.csv,manifest.json}`
- `results/attribution_patching/20260617-pythia14b-ap-drop_argument.{detail.csv,summary.csv,manifest.json}`

Exact patching outputs:

- `results/exact_patching/20260617-pythia14b-ap-primary-exact.{detail.csv,summary.csv,manifest.json}`
- `results/exact_patching/20260617-pythia14b-ap-secondary-exact.{detail.csv,summary.csv,manifest.json}`
- `results/exact_patching/20260617-pythia14b-ap-drop_argument-exact.{detail.csv,summary.csv,manifest.json}`

## Attribution Screen

Top attribution site per regime/subtask/direction:

| Contrast | Regime | Direction | Top site | n | Mean abs attr | Mean attr |
| --- | --- | --- | --- | ---: | ---: | ---: |
| causative | head | bad->good | `resid_post_layer_23` | 20 | 6.037 | 5.829 |
| causative | head | good->bad | `resid_post_layer_21` | 20 | 6.851 | 6.748 |
| inchoative | head | bad->good | `resid_post_layer_21` | 18 | 7.352 | 7.352 |
| inchoative | head | good->bad | `resid_post_layer_16` | 18 | 5.793 | 5.793 |
| causative | tail | bad->good | `resid_post_layer_21` | 12 | 5.428 | 0.494 |
| causative | tail | good->bad | `resid_post_layer_18` | 12 | 5.185 | 3.894 |
| inchoative | tail | bad->good | `resid_post_layer_19` | 13 | 4.284 | 3.984 |
| inchoative | tail | good->bad | `resid_post_layer_23` | 13 | 3.322 | 3.017 |
| causative | xtail | bad->good | `resid_post_layer_23` | 10 | 2.808 | 2.498 |
| causative | xtail | good->bad | `resid_post_layer_21` | 10 | 4.665 | 3.981 |
| inchoative | xtail | bad->good | `resid_post_layer_19` | 7 | 3.634 | 2.728 |
| inchoative | xtail | good->bad | `resid_post_layer_13` | 7 | 3.025 | 3.025 |
| transitive | head | bad->good | `resid_post_layer_18` | 19 | 6.378 | 6.378 |
| transitive | head | good->bad | `resid_post_layer_21` | 19 | 6.004 | 5.996 |
| intransitive | head | bad->good | `resid_post_layer_19` | 17 | 5.821 | 5.821 |
| intransitive | head | good->bad | `resid_post_layer_23` | 17 | 5.274 | 5.274 |
| drop_argument | head | bad->good | `resid_post_layer_21` | 15 | 8.108 | 6.699 |
| drop_argument | head | good->bad | `resid_post_layer_23` | 15 | 6.757 | 5.774 |

Readout: attribution consistently points to a late residual-stream band,
roughly layers 16-23, with strongest and cleanest head-regime effects. Tail and
xtail remain positive but are smaller and less directionally clean.

## Exact Patching

Top exact site per regime/subtask/direction:

| Contrast | Regime | Direction | Top site | n | Mean abs effect | Mean effect | Mean normalized |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| causative | head | bad->good | `resid_post_layer_23` | 20 | 6.219 | 6.009 | 1.024 |
| causative | head | good->bad | `resid_post_layer_23` | 20 | 6.159 | 5.975 | 1.001 |
| inchoative | head | bad->good | `resid_post_layer_23` | 18 | 5.439 | 5.439 | 1.039 |
| inchoative | head | good->bad | `resid_post_layer_23` | 18 | 5.682 | 5.682 | 1.085 |
| causative | tail | bad->good | `resid_post_layer_23` | 12 | 4.477 | 3.706 | 1.112 |
| causative | tail | good->bad | `resid_post_layer_23` | 12 | 4.406 | 3.734 | 0.963 |
| inchoative | tail | bad->good | `resid_post_layer_23` | 13 | 3.543 | 3.120 | 1.084 |
| inchoative | tail | good->bad | `resid_post_layer_23` | 13 | 3.692 | 3.452 | 1.097 |
| causative | xtail | bad->good | `resid_post_layer_23` | 10 | 2.872 | 2.509 | 0.920 |
| causative | xtail | good->bad | `resid_post_layer_23` | 10 | 3.000 | 2.669 | 1.394 |
| inchoative | xtail | bad->good | `resid_post_layer_16` | 7 | 2.545 | 2.348 | 1.052 |
| inchoative | xtail | good->bad | `resid_post_layer_23` | 7 | 3.000 | 2.804 | 1.416 |
| transitive | head | bad->good | `resid_post_layer_16` | 19 | 5.326 | 5.326 | 1.000 |
| transitive | head | good->bad | `resid_post_layer_16` | 19 | 5.326 | 5.326 | 1.000 |
| intransitive | head | bad->good | `resid_post_layer_16` | 17 | 5.272 | 5.272 | 1.000 |
| intransitive | head | good->bad | `resid_post_layer_16` | 17 | 5.272 | 5.272 | 1.000 |
| drop_argument | head | bad->good | `resid_post_layer_23` | 15 | 6.779 | 5.763 | 1.013 |
| drop_argument | head | good->bad | `resid_post_layer_23` | 15 | 6.965 | 6.006 | 1.069 |
| drop_argument | tail | bad->good | `resid_post_layer_23` | 12 | 5.362 | 2.810 | 1.009 |
| drop_argument | tail | good->bad | `resid_post_layer_23` | 12 | 5.456 | 2.997 | 1.060 |
| drop_argument | xtail | bad->good | `resid_post_layer_23` | 9 | 4.000 | 2.389 | 0.839 |
| drop_argument | xtail | good->bad | `resid_post_layer_23` | 9 | 4.031 | 2.372 | 0.984 |

Exact patching confirms that late residual-stream replacement is sufficient to
recover the clean-prompt logit-difference metric, usually with normalized effect
near 1.0. This is a strong v1 infrastructure/localization result.

Important caveat: exact patching of late residual sites is not a unique-layer
claim. Several late sites can fully restore the metric because replacing a late
residual vector at the decision token largely overwrites downstream state. The
v1 conclusion should therefore be a late-residual band candidate, not a single
privileged layer.

## Interpretation

- Primary causative/inchoative: use late residual stream as the v1 DAS/search
  target, with `resid_post_layer_23` as the strongest exact-patching anchor and
  layers 16-23 as the broader candidate band.
- Secondary transitive/intransitive: same late-band pattern appears, so these
  are viable secondary validation contrasts.
- Drop-argument: exact patching also works, but keep it separate. Its behavioral
  gate was weaker in tail, and its attribution signs were less clean in some
  frequency bands.
- Do not make final semantic causal claims from this fixed-subject scaffold.
  The next dataset version still needs semantic-fit controls, prefix-only and
  shuffled-label baselines, and held-out lemma splits.

## Next V1 Step

For infrastructure-only DAS, train/evaluate first on the primary contrast using:

- model: `EleutherAI/pythia-1.4b`;
- site: prompt-final `verb_final_subtoken`;
- representation candidates: `resid_post_layer_23` first, then a small sweep
  over layers 16-23;
- train/eval split: lemma-held-out if available even in v1, otherwise report
  that v1 is scaffold-only;
- controls before treating any DAS result as interpretable:
  - retrain-on-`shuffled_label`, which tests whether DAS can fit arbitrary
    label structure;
  - `random_direction`, which tests whether the learned subspace is special
    relative to a same-rank random subspace at the same site;
  - `dummy_verb`, which tests whether the fixed template is solvable with no
    informative verb.

The dummy-verb baseline is not a substitute for decorrelating subject type from
target frame. The animacy/frame confound needs a v2 data fix: include animate
subjects with non-alternating intransitives and patient subjects with obligatory
transitives, so verb identity is the only reliable frame predictor.
