# V2 DAS Plan

## Context From V1

V1 has strong infrastructure evidence but is not a final causal dataset:

- fixed-subject scaffold leaves subject/frame and animacy/frame confounds;
- the proxy decision is only ` the` versus `.`;
- localization and exact patching point to a late residual-stream band, with
  `resid_post_layer_23` as the strongest practical anchor;
- head-trained rank-1 DAS transfers to tail/xtail, but the head held-out set in
  the transfer run was only 20 directed pairs.

The `n=20` head evaluation is too small for a final claim. V2 should increase
sample count and remove the fixed-subject confound before repeating DAS.

Current data scale:

- upstream processed FreqBLiMP has 1,000 rows per regime/subtask;
- current aligned scaffold has 726 rows total;
- aligned bucket sizes are only about 15-30 verbs per
  `(regime, subtask, side)`;
- the v1 head-transfer DAS used 56 directed head training pairs and 20 directed
  held-out head pairs.
- strict tail/xtail splits leave almost no bad-side support, especially for
  inchoative bad verbs.

The limiting issue is not just the current builder cap. The v1 DAS pair builder
requires same-`source_idx` good/bad pairs, which sharply limits usable held-out
pairs. V2 should keep source provenance but build DAS pairs from balanced pools
of verified good and bad verbs within each contrast/context family.

## V2 Dataset Requirements

Build `data/aligned_templates_v2/lexical_licensing_v2_aligned.jsonl` with one
row per verified prompt/verb/target example.

Required row fields:

- existing v1 fields: `regime`, `subtask`, `template_id`, `prompt`, `verb`,
  `side`, `licensing_label`, `expected_target`, `contrast_target`,
  `intervention_anchor`, `anchor_token_index`, `prompt_token_count`,
  `alignment_status`;
- new fields: `context_id`, `context_schema_id`, `subject`, `subject_class`,
  `subject_role_probe`, `subject_source`, `object`, `object_class`,
  `object_role_probe`, `object_source`, `frame_label`, `semantic_fit_source`,
  `split_lemma_key`, `control_type`, `source_zipf`, `source_zipf_regime`,
  `inventory_source`.

Core constraints:

- every `context_id` must contain both target labels, so prefix/subject cannot
  predict the answer;
- subjects and object metadata should be sampled from the FreqBLiMP overlay
  rather than held fixed in the template;
- sampled subject contexts must be reused across both target labels, so a
  sampled noun cannot by itself identify the answer;
- held-out splits must be by lemma or lemma family, not by row;
- prompt-final intervention location remains `verb_final_subtoken`;
- target continuations must be single tokenizer tokens for the chosen model;
- drop-argument remains a separate contrast unless v2 attribution says
  otherwise.
- subject compatibility is not a hard lexical filter for v2. Instead, subject
  and source-band metadata are retained for post-hoc breakdowns.

Controls to include as explicit rows or generated pair modes:

- `prefix_only`;
- `dummy_verb`;
- `shuffled_label`;
- `random_verb_same_frequency_band`;
- passive/principle-A negative controls.

## Sample Size Target

Minimum viable V2:

- primary contrasts: `causative,inchoative`;
- regimes: `head,low`, where `low=1.2-3.2` combines the former tail and xtail
  bands;
- preserve `source_zipf_regime` so low can be analyzed as `xtail`, `tail`, and
  `low_gap` after the primary head-vs-low test;
- at least 25 verified lemma pairs per side in the low regime across the primary
  contrasts;
- 75/25 lemma-held-out split, giving roughly 150 directed train pairs and 50
  directed held-out pairs per regime when using both directions.

Preferred V2:

- at least 100 verified lemma pairs per regime per primary subtask;
- 200 directed held-out head examples across primary contrasts;
- 200 directed low-frequency transfer examples;
- 3 random seeds for DAS training and shuffled-label controls.

If the raw lexical inventory cannot support 100 clean pairs for a sparse side,
use all available verified lemmas and report the ceiling. Do not silently
downsample the richer side to the first N rows; sample deterministically by seed.

## Build Plan

1. Create a v2 builder rather than mutating `build_aligned_templates.py`.
   Suggested script: `scripts/build_aligned_templates_v2.py`.
2. For `causative` and `inchoative`, read candidate verbs from the curated
   lists in `freq-blimp/generation_projects/blimp/overlay_guards.py`; do not
   infer those verbs from generated JSONL text.
3. Assign primary regimes with `head=3.5-5.5` and `low=1.2-3.2`, while
   retaining historical source-band metadata:
   `xtail=1.2-2.2`, `tail=2.4-3.2`, and `low_gap=2.2-2.4`.
4. Add repo-local supplemental bad-side verbs from
   `data/aligned_templates_v2/supplemental_bad_verbs.json`, after the same
   Zipf-band check, to avoid the low-frequency bad side collapsing to one or
   zero lemmas.
5. Use canonical FreqBLiMP JSONL only for subtasks without curated verb lists.
   The archived `blimp-rare` generation/swap pipeline is not a v2 source.
6. Sample subject/object fillers from `freq-blimp/vocabulary_overlay.csv`.
   Prefer `frequent=1` and `sg=1` nouns under the role constraint, then fall
   back to the full role-matching pool when support is sparse.
7. Build balanced per-contrast pools by
   `(regime, subtask, side, context_id)`.
8. Generate multiple context schemas, then instantiate each schema with
   deterministic FreqBLiMP overlay subject/object variants. Treat schema roles
   such as animate-agent or patient/theme as probe labels, not as guaranteed
   semantic classes for every sampled noun.
9. Tokenizer-verify each generated row and retain only aligned rows.
10. Write a build report with counts by
   `(regime, subtask, side, context_id, subject_class, subject_source)` and
   explicit rejected tokenization counts.
11. Run a baseline sanity check measuring
    `log p(expected_target | prompt) - log p(contrast_target | prompt)`.
12. Add a pair-building mode for DAS that pairs verified good/bad rows within a
   balanced pool, rather than requiring same `source_idx`.

## DAS Protocol

Run localization first on v2:

- model: `EleutherAI/pythia-1.4b`;
- sites: late residual band `resid_post_layer_16` through
  `resid_post_layer_23`;
- primary anchor: prompt-final `verb_final_subtoken`;
- objective: same log-odds target shift as v1.

Then run DAS:

- train on head primary contrasts only;
- validate on held-out head lemmas;
- freeze the learned subspace and evaluate transfer on low;
- rank sweep: `1,2,4`;
- seed sweep: at least `17,23,41`;
- controls: shuffled-label retrain, dummy/prefix-only, random direction,
  random same-frequency verb.

Report:

- mean corrupt, clean, patched log-odds;
- mean intervention effect;
- success rate;
- normalized effect, with overshoot called out separately;
- bootstrap confidence intervals over held-out lemma pairs;
- per-context-family and sampled-subject breakdown to check that no subject drives the
  result.
- source-band and inventory-source breakdowns, especially for supplemental bad
  verbs.

## Success Criteria

Treat v2 DAS as interpretable only if all are true:

- head held-out effect is positive and stable across seeds;
- low transfer remains positive under the frozen head-trained subspace;
- shuffled-label and random-direction controls are near chance/zero effect;
- prefix-only/dummy-verb controls do not produce a useful intervention;
- results hold when grouped by sampled subject, subject source, and context family;
- the effect is not dependent on one sparse subtask or a tiny held-out split.

## Implementation Status

Implemented locally:

- `scripts/build_aligned_templates_v2.py`;
- `data/aligned_templates_v2/supplemental_bad_verbs.json`;
- `data/aligned_templates_v2/lexical_licensing_v2_candidates.jsonl`;
- `reports/aligned_template_v2_report.md`;
- `scripts/run_pythia_v2_sanity.py`;
- `--pairing balanced_pool` in attribution patching, exact patching, and DAS.

Build candidates:

```bash
python3 scripts/build_aligned_templates_v2.py --target-per-side-context 100
```

The default builder samples 4 subject/object variants per context schema from
the FreqBLiMP overlay. Override with `--subject-variants-per-context`.

For `causative`/`inchoative`, the verb inventory is now curated-list backed:
`_CAUSATIVE_ALTERNATING_VERBS`, `_CAUSATIVE_BAD_*`, and
`_DROP_ARGUMENT_BAD_VERBS` in FreqBLiMP's `overlay_guards.py`. This avoids the
old `blimp-rare` swapping pipeline and avoids treating overlay replacement
surface forms as the verb inventory. Low-frequency bad-side support is expanded
by the supplemental repo-local inventory and remains explicitly labeled by
inventory source.

Build tokenizer-verified aligned rows in an environment with `transformers` and
the Pythia tokenizer available:

```bash
python3 scripts/build_aligned_templates_v2.py \
  --subtasks causative,inchoative \
  --model EleutherAI/pythia-1.4b \
  --target-per-side-context 100 \
  --subject-variants-per-context 4
```

On TSUBAME, set `FREQ_BLIMP_ROOT=/gs/fs/tga-sip_arase/tyrone/freq-blimp`
before running the builder.

Run v2 localization/DAS with balanced-pool pairing:

```bash
python3 scripts/run_pythia_attribution_patching.py \
  --data data/aligned_templates_v2/lexical_licensing_v2_aligned.jsonl \
  --subtasks causative,inchoative \
  --regimes head,low \
  --pairing balanced_pool

python3 scripts/run_pythia_das_v1.py \
  --data data/aligned_templates_v2/lexical_licensing_v2_aligned.jsonl \
  --subtasks causative,inchoative \
  --train-regimes head \
  --eval-regimes head,low \
  --pairing balanced_pool
```

Run the whole-pair LP behavioral gate before retraining DAS on a whole-sentence
objective:

```bash
python3 scripts/run_pythia_whole_pair_lp.py \
  --data data/aligned_templates_v2/lexical_licensing_v2_aligned.jsonl \
  --subtasks causative,inchoative \
  --regimes head,low \
  --run-name 20260625-pythia14b-v2-whole-pair-lp
```

Run the unrelated target-token capacity control with the same v2 prompts and
activation sources, but a fixed readout axis:

Tokenizer sanity for `EleutherAI/pythia-1.4b`:

- ` red` is a single token: `Ġred`, id `2502`;
- ` blue` is a single token: `Ġblue`, id `4797`.

```bash
python3 scripts/run_pythia_v2_sanity.py \
  --data data/aligned_templates_v2/lexical_licensing_v2_aligned.jsonl \
  --subtasks causative,inchoative \
  --regimes head,low \
  --target-mode fixed_pair \
  --fixed-expected-target " red" \
  --fixed-contrast-target " blue" \
  --run-name 20260623-pythia14b-v2-sanity-red-blue-head-low

python3 scripts/run_pythia_das_v1.py \
  --data data/aligned_templates_v2/lexical_licensing_v2_aligned.jsonl \
  --subtasks causative,inchoative \
  --train-regimes head \
  --eval-regimes head,low \
  --site resid_post_layer_23 \
  --rank 1 \
  --pairing balanced_pool \
  --seed 17 \
  --control red_blue \
  --run-name 20260623-pythia14b-v2-das-head-to-low-l23-red_blue-s17
```

Run the injection-site specificity control by reusing the trained lexical
subspace, reading the projected clean-corrupt delta from the verb-final slot,
and moving only the evaluation-time patch site away from the verb:

```bash
for layer in 18 20; do
  python3 scripts/run_pythia_das_v1.py \
    --data data/aligned_templates_v2/lexical_licensing_v2_aligned.jsonl \
    --subtasks causative,inchoative \
    --train-regimes head \
    --eval-regimes head,low \
    --site "resid_post_layer_${layer}" \
    --rank 1 \
    --pairing balanced_pool \
    --seed 17 \
    --control none \
    --run-name "20260623-pythia14b-v2-das-head-to-low-l${layer}-none-s17"

  python3 scripts/run_pythia_das_v1.py \
    --data data/aligned_templates_v2/lexical_licensing_v2_aligned.jsonl \
    --subtasks causative,inchoative \
    --train-regimes head \
    --eval-regimes head,low \
    --site "resid_post_layer_${layer}" \
    --rank 1 \
    --pairing balanced_pool \
    --seed 17 \
    --control none \
    --load-subspace "results/das_v2_transfer/20260623-pythia14b-v2-das-head-to-low-l${layer}-none-s17.subspace.pt" \
    --eval-source-anchor verb_final_subtoken \
    --eval-anchor subject_final_subtoken \
    --run-name "20260623-pythia14b-v2-das-head-to-low-l${layer}-none-s17-verb_to_subject_anchor"
done
```

The score is still the next-token continuation after the full verb-final
prompt. Running this before the final layer leaves later attention layers
available, so a subject-token intervention has a possible path to the final
verb-position logits.

Tokenizer IDs were verified from the cached `EleutherAI/pythia-1.4b`
tokenizer JSON. Completed v2 model outputs are summarized under `reports/` and
stored under `results/`.
