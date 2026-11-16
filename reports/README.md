# Reports index

Each stage was declared (plan, predictions, decision rules) and committed
before its runs. Read the plan, then its results file.

Paper-level plan (claims, status, priorities, deadlines):
`../docs/paper_plan.md`.

| Stage | Question | Plan | Results |
|---|---|---|---|
| v1 (June) | Behavioural gate, attribution / exact patching, first DAS on aligned templates | `pythia_attribution_next_steps_20260617.md` | `v1_*`, `pythia_behavior_gate_20260617.md` |
| v2 (June) | Balanced-pool DAS, head → low transfer, whole-pair LP | `../docs/v2_das_plan.md` | `v2_*`, `whole_pair_*` |
| Passive pilots (Sept) | Matched passives, released vs curated contexts, fit ratings | `../docs/matched_passives.md`, `../docs/passive_band_cross.md`, `../docs/fit_ratings.md` | `matched_passives*/`, `passive_band_cross/`, `fit_ratings/` |
| Passive DAS prep | Consistency, baselines, *by* split, DAS datasets | — | `passive_das_prep/README.md` |
| DAS round 2 | Rank-1 "object next" direction on actives; passive test (sites 8, 12, 17); controls; site-8 mechanism | `passive_das_prep/round2.md`, `passive_das_prep/passive_test_plan.md`, `passive_das_prep/followup_plan.md` | `passive_das_prep/das_round2_results*.md`, `passive_test_results*.md`, `site8_mechanism_results.md`, `followup_results.md`, `frequency_results.md` |
| Round 3 | Reliability, late projection, dose decomposition; reverse DAS, get-passives, path patching; nonce verbs, gap constructions | `passive_das_prep/round3_plan.md` | `passive_das_prep/round3_results.md` (generated: `round3_*.md`) |
| Round 4 | Confirm current claims; filler-gap test; nonce cues; translation stage; restoration | `round4/plan.md` | `round4/results.md` (generated: `round4/*.md`) |
| Round 5 | Locate the rare-verb deficit (prediction side); finish mechanism tests; generalization; scale-up prep | `round5/plan.md` | `round5/results.md` (generated: `round5/*.md`) |
| Round 6 | Paper 1: where the rare-verb deficit is, what exposure explains it, passive specificity, restoration / preemption | `round6/plan.md` | `round6/results.md` (generated: `round6/*.md`) |
| Round 7 | Paper 1 follow-ups: per-verb training trajectories (A8) | `round7/plan.md` | `round7/a8_trajectories.md` |
| Gemma replication | Port the pipeline to `google/gemma-3-1b-pt`: refilter, behaviour, DAS port / sweeps, transfer, depth curve, nonce | `gemma/step1_refilter.md` (each step states its rules) | `gemma/*.md`, `gemma/transfer/`, `gemma/depth_curve/`, `gemma/nonce/summary.md` |
| OLMo replication | Third model, `allenai/OLMo-2-0425-1B`: behaviour, DAS sweep, finals at sites 0–15, transfer, passive depth curve | `olmo/rules.md`, `olmo/finals_rules.md` | `olmo/behavior.md`, `olmo/das_sweep.md`, `olmo/transfer/`, `olmo/depth_curve/` |
| Replication criteria | T1–T3 and S1 applied to Pythia, Gemma and OLMo; split edit + u_md readings (R-A1, R-A2, S1/T1 per instrument, FI) | `olmo/finals_rules.md` §3; `split_edit_replication_rules.md` | `replication_criteria.md`; `gemma/split_edit.md`, `olmo/split_edit.md`; `selfscaled_passive_classes.md` (exploratory) |
| Pythia necessity and direction check | LEACE necessity (in-sample, cross-fitted, permutation control, learning curve); DAS direction v vs u_md, mean ablation | `pythia/leace_learning_curve_rules.md`, `pythia/das_direction_check_plan.md` | `pythia/leace_*.md`, `pythia/das_direction_check.md` |

`results_index.md` lists every experiment of rounds 1–7 with its key number, declared
outcome, report and figure, tagged main / appendix / drop (suggestions for the paper).

Outputs live under `../results/` (round 2–3: `das_round2/`; rounds 4–7:
`round4/` … `round7/`; replications: `gemma/`, `olmo/`); regenerable inputs under `../data/`. Large raw dumps are
gitignored; their hashes are in the run metadata.
