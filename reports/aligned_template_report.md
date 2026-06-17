# Aligned Template Build Report

## Design Notes

- Intervention anchor: `verb_final_subtoken`.
- Anchor index policy: per-example last prompt token, not one global absolute index.
- Decision target is a proxy: object introducer ` the` versus sentence end `.`.
- The current fixed-subject templates are a preview; final DAS should add prefix-only, shuffled-label, and semantic-fit controls.

## Template Preview

- `object_frame` `head` `causative` `good`: 'In the scene, the technician will equip' -> ' the'
- `object_frame` `head` `causative` `bad`: 'In the scene, the technician will arrive' -> '.'
- `object_frame` `head` `causative` `good`: 'In the scene, the technician will conduct' -> ' the'
- `object_frame` `head` `causative` `bad`: 'In the scene, the technician will arise' -> '.'
- `object_frame` `head` `causative` `good`: 'In the scene, the technician will implement' -> ' the'
- `object_frame` `head` `causative` `bad`: 'In the scene, the technician will expire' -> '.'
- `object_frame` `head` `causative` `good`: 'In the scene, the technician will refuse' -> ' the'
- `object_frame` `head` `causative` `bad`: 'In the scene, the technician will descend' -> '.'
- `object_frame` `head` `causative` `good`: 'In the scene, the technician will bake' -> ' the'
- `object_frame` `head` `causative` `bad`: 'In the scene, the technician will react' -> '.'
- `object_frame` `head` `causative` `good`: 'In the scene, the technician will engage' -> ' the'
- `object_frame` `head` `causative` `bad`: 'In the scene, the technician will zoom' -> '.'

## Counts

- `head` `causative` `bad` `aligned`: 30
- `head` `causative` `good` `aligned`: 29
- `head` `causative` `good` `rejected_tokenization`: 1
- `head` `drop_argument` `bad` `aligned`: 30
- `head` `drop_argument` `good` `aligned`: 30
- `head` `inchoative` `bad` `aligned`: 29
- `head` `inchoative` `bad` `rejected_tokenization`: 1
- `head` `inchoative` `good` `aligned`: 30
- `head` `intransitive` `bad` `aligned`: 29
- `head` `intransitive` `bad` `rejected_tokenization`: 1
- `head` `intransitive` `good` `aligned`: 30
- `head` `transitive` `bad` `aligned`: 30
- `head` `transitive` `good` `aligned`: 29
- `head` `transitive` `good` `rejected_tokenization`: 1
- `tail` `causative` `bad` `aligned`: 22
- `tail` `causative` `bad` `rejected_tokenization`: 8
- `tail` `causative` `good` `aligned`: 22
- `tail` `causative` `good` `rejected_tokenization`: 8
- `tail` `drop_argument` `bad` `aligned`: 22
- `tail` `drop_argument` `bad` `rejected_tokenization`: 8
- `tail` `drop_argument` `good` `aligned`: 23
- `tail` `drop_argument` `good` `rejected_tokenization`: 7
- `tail` `inchoative` `bad` `aligned`: 23
- `tail` `inchoative` `bad` `rejected_tokenization`: 7
- `tail` `inchoative` `good` `aligned`: 22
- `tail` `inchoative` `good` `rejected_tokenization`: 8
- `tail` `intransitive` `bad` `aligned`: 23
- `tail` `intransitive` `bad` `rejected_tokenization`: 7
- `tail` `intransitive` `good` `aligned`: 24
- `tail` `intransitive` `good` `rejected_tokenization`: 6
- `tail` `transitive` `bad` `aligned`: 23
- `tail` `transitive` `bad` `rejected_tokenization`: 7
- `tail` `transitive` `good` `aligned`: 25
- `tail` `transitive` `good` `rejected_tokenization`: 5
- `xtail` `causative` `bad` `aligned`: 22
- `xtail` `causative` `bad` `rejected_tokenization`: 8
- `xtail` `causative` `good` `aligned`: 19
- `xtail` `causative` `good` `rejected_tokenization`: 11
- `xtail` `drop_argument` `bad` `aligned`: 17
- `xtail` `drop_argument` `bad` `rejected_tokenization`: 13
- `xtail` `drop_argument` `good` `aligned`: 20
- `xtail` `drop_argument` `good` `rejected_tokenization`: 10
- `xtail` `inchoative` `bad` `aligned`: 15
- `xtail` `inchoative` `bad` `rejected_tokenization`: 15
- `xtail` `inchoative` `good` `aligned`: 24
- `xtail` `inchoative` `good` `rejected_tokenization`: 6
- `xtail` `intransitive` `bad` `aligned`: 20
- `xtail` `intransitive` `bad` `rejected_tokenization`: 10
- `xtail` `intransitive` `good` `aligned`: 21
- `xtail` `intransitive` `good` `rejected_tokenization`: 9
- `xtail` `transitive` `bad` `aligned`: 23
- `xtail` `transitive` `bad` `rejected_tokenization`: 7
- `xtail` `transitive` `good` `aligned`: 20
- `xtail` `transitive` `good` `rejected_tokenization`: 10

## Prefix/Target Balance

- `head` `drop_object_frame` target ' the': 30 aligned rows for prefix `In the scene, the artist will {verb}`
- `head` `drop_object_frame` target '.': 30 aligned rows for prefix `In the scene, the artist will {verb}`
- `head` `inchoative_frame` target ' the': 29 aligned rows for prefix `In the scene, the glass will {verb}`
- `head` `inchoative_frame` target '.': 30 aligned rows for prefix `In the scene, the glass will {verb}`
- `head` `inchoative_frame_tomorrow` target ' the': 29 aligned rows for prefix `Tomorrow, the door will {verb}`
- `head` `inchoative_frame_tomorrow` target '.': 30 aligned rows for prefix `Tomorrow, the door will {verb}`
- `head` `object_frame` target ' the': 29 aligned rows for prefix `In the scene, the technician will {verb}`
- `head` `object_frame` target '.': 30 aligned rows for prefix `In the scene, the technician will {verb}`
- `head` `object_frame_worker` target ' the': 29 aligned rows for prefix `After lunch, the worker will {verb}`
- `head` `object_frame_worker` target '.': 30 aligned rows for prefix `After lunch, the worker will {verb}`
- `tail` `drop_object_frame` target ' the': 22 aligned rows for prefix `In the scene, the artist will {verb}`
- `tail` `drop_object_frame` target '.': 23 aligned rows for prefix `In the scene, the artist will {verb}`
- `tail` `inchoative_frame` target ' the': 23 aligned rows for prefix `In the scene, the glass will {verb}`
- `tail` `inchoative_frame` target '.': 22 aligned rows for prefix `In the scene, the glass will {verb}`
- `tail` `inchoative_frame_tomorrow` target ' the': 23 aligned rows for prefix `Tomorrow, the door will {verb}`
- `tail` `inchoative_frame_tomorrow` target '.': 24 aligned rows for prefix `Tomorrow, the door will {verb}`
- `tail` `object_frame` target ' the': 22 aligned rows for prefix `In the scene, the technician will {verb}`
- `tail` `object_frame` target '.': 22 aligned rows for prefix `In the scene, the technician will {verb}`
- `tail` `object_frame_worker` target ' the': 25 aligned rows for prefix `After lunch, the worker will {verb}`
- `tail` `object_frame_worker` target '.': 23 aligned rows for prefix `After lunch, the worker will {verb}`
- `xtail` `drop_object_frame` target ' the': 17 aligned rows for prefix `In the scene, the artist will {verb}`
- `xtail` `drop_object_frame` target '.': 20 aligned rows for prefix `In the scene, the artist will {verb}`
- `xtail` `inchoative_frame` target ' the': 15 aligned rows for prefix `In the scene, the glass will {verb}`
- `xtail` `inchoative_frame` target '.': 24 aligned rows for prefix `In the scene, the glass will {verb}`
- `xtail` `inchoative_frame_tomorrow` target ' the': 20 aligned rows for prefix `Tomorrow, the door will {verb}`
- `xtail` `inchoative_frame_tomorrow` target '.': 21 aligned rows for prefix `Tomorrow, the door will {verb}`
- `xtail` `object_frame` target ' the': 19 aligned rows for prefix `In the scene, the technician will {verb}`
- `xtail` `object_frame` target '.': 22 aligned rows for prefix `In the scene, the technician will {verb}`
- `xtail` `object_frame_worker` target ' the': 20 aligned rows for prefix `After lunch, the worker will {verb}`
- `xtail` `object_frame_worker` target '.': 23 aligned rows for prefix `After lunch, the worker will {verb}`

## Tokenization Summary

- model: `EleutherAI/pythia-1.4b`
- target token lengths: `{' the': 1, '.': 1}`
- `head|causative|object_frame`: retained 59/60 at prompt_token_count=8 (decision_index=7); all prompt counts={8: 59, 9: 1}; verb-region counts={1: 59, 2: 1}
- `head|drop_argument|drop_object_frame`: retained 60/60 at prompt_token_count=8 (decision_index=7); all prompt counts={8: 60}; verb-region counts={1: 60}
- `head|inchoative|inchoative_frame`: retained 59/60 at prompt_token_count=8 (decision_index=7); all prompt counts={8: 59, 9: 1}; verb-region counts={1: 59, 2: 1}
- `head|intransitive|inchoative_frame_tomorrow`: retained 59/60 at prompt_token_count=7 (decision_index=6); all prompt counts={7: 59, 8: 1}; verb-region counts={1: 59, 2: 1}
- `head|transitive|object_frame_worker`: retained 59/60 at prompt_token_count=7 (decision_index=6); all prompt counts={7: 59, 8: 1}; verb-region counts={1: 59, 2: 1}
- `tail|causative|object_frame`: retained 44/60 at prompt_token_count=9 (decision_index=8); all prompt counts={9: 44, 10: 8, 8: 8}; verb-region counts={2: 44, 3: 8, 1: 8}
- `tail|drop_argument|drop_object_frame`: retained 45/60 at prompt_token_count=9 (decision_index=8); all prompt counts={10: 6, 9: 45, 8: 9}; verb-region counts={3: 6, 2: 45, 1: 9}
- `tail|inchoative|inchoative_frame`: retained 45/60 at prompt_token_count=9 (decision_index=8); all prompt counts={9: 45, 10: 8, 8: 7}; verb-region counts={2: 45, 3: 8, 1: 7}
- `tail|intransitive|inchoative_frame_tomorrow`: retained 47/60 at prompt_token_count=8 (decision_index=7); all prompt counts={7: 6, 8: 47, 9: 7}; verb-region counts={1: 6, 2: 47, 3: 7}
- `tail|transitive|object_frame_worker`: retained 48/60 at prompt_token_count=8 (decision_index=7); all prompt counts={8: 48, 9: 4, 7: 8}; verb-region counts={2: 48, 3: 4, 1: 8}
- `xtail|causative|object_frame`: retained 41/60 at prompt_token_count=9 (decision_index=8); all prompt counts={9: 41, 10: 19}; verb-region counts={2: 41, 3: 19}
- `xtail|drop_argument|drop_object_frame`: retained 37/60 at prompt_token_count=9 (decision_index=8); all prompt counts={10: 22, 9: 37, 11: 1}; verb-region counts={3: 22, 2: 37, 4: 1}
- `xtail|inchoative|inchoative_frame`: retained 39/60 at prompt_token_count=9 (decision_index=8); all prompt counts={9: 39, 10: 19, 11: 2}; verb-region counts={2: 39, 3: 19, 4: 2}
- `xtail|intransitive|inchoative_frame_tomorrow`: retained 41/60 at prompt_token_count=8 (decision_index=7); all prompt counts={8: 41, 9: 16, 10: 1, 11: 1, 7: 1}; verb-region counts={2: 41, 3: 16, 4: 1, 5: 1, 1: 1}
- `xtail|transitive|object_frame_worker`: retained 43/60 at prompt_token_count=8 (decision_index=7); all prompt counts={8: 43, 9: 16, 10: 1}; verb-region counts={2: 43, 3: 16, 4: 1}
