# Reports index

Each stage was declared (plan, predictions, decision rules) and committed
before its runs. Read the plan, then its results file.

| Stage | Question | Plan | Results |
|---|---|---|---|
| v1 (June) | Behavioural gate, attribution / exact patching, first DAS on aligned templates | `pythia_attribution_next_steps_20260617.md` | `v1_*`, `pythia_behavior_gate_20260617.md` |
| v2 (June) | Balanced-pool DAS, head → low transfer, whole-pair LP | `../docs/v2_das_plan.md` | `v2_*`, `whole_pair_*` |
| Passive pilots (Sept) | Matched passives, released vs curated contexts, fit ratings | `../docs/matched_passives.md`, `../docs/passive_band_cross.md`, `../docs/fit_ratings.md` | `matched_passives*/`, `passive_band_cross/`, `fit_ratings/` |
| Passive DAS prep | Consistency, baselines, *by* split, DAS datasets | — | `passive_das_prep/README.md` |
| DAS round 2 | Rank-1 "object next" direction on actives; passive test (sites 8, 12, 17); controls; site-8 mechanism | `passive_das_prep/round2.md`, `passive_das_prep/passive_test_plan.md`, `passive_das_prep/followup_plan.md` | `passive_das_prep/das_round2_results*.md`, `passive_test_results*.md`, `site8_mechanism_results.md`, `followup_results.md`, `frequency_results.md` |
| Round 3 | Reliability, late projection, dose decomposition; reverse DAS, get-passives, path patching; nonce verbs, gap constructions | `passive_das_prep/round3_plan.md` | `passive_das_prep/round3_results.md` (generated: `round3_*.md`) |
| Round 4 | Confirm current claims; filler-gap test; nonce cues; translation stage; restoration | `round4/plan.md` | `round4/results.md` (generated: `round4/*.md`) |

Outputs live under `../results/` (round 2–3: `das_round2/`; round 4:
`round4/`); regenerable inputs under `../data/`. Large raw dumps are
gitignored; their hashes are in the run metadata.
