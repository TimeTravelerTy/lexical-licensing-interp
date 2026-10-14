# Round 5, A4: output-side restoration (Tail single-token participles)

Spec: `plan.md`, A4. Run: `run_restore.py`; analysis: `analyze_restore.py`. Edits move each rare single-token participle's unembedding row along Δh̄ so its affinity becomes a*. 95% CIs: pairs within band × contexts (curated), items (released), documents / verbs (LM loss).

Checks: {"affinity_reproduces_max": 8.076507294063617e-07, "edit_target_err_oracle": 5.963399661440549e-07, "unedited_rows_max_oracle": 0.0, "edit_target_err_oracle_T": 5.963399661440549e-07, "unedited_rows_max_oracle_T": 0.0, "edit_target_err_oracle_I": 3.9269173335299e-07, "unedited_rows_max_oracle_I": 0.0, "edit_target_err_labelfree": 6.293471144225293e-07, "unedited_rows_max_labelfree": 0.0, "unedited_reproduces_fp32_max": 0.0004418492317199152}.

Label-free calibration (Head single-token verbs): {"alpha": -3.825583036269375, "beta": 4.083541959970332, "cv_r2_head": 0.4355287612249107, "cv_r2_shuffled_95": 0.005777039009562997, "resid_mean_T": 0.052310622640696934, "resid_mean_I": -0.49171985282254477, "resid_zipf_slope": -0.02812812753690269, "z_range_head": [-0.2222161536179467, 1.3407278986196909], "z_range_rare": [0.017422165126330703, 1.2303634334327043]}

## Curated `passive_1`

Affected pairs: 12 (['tail/approximate/muse', 'tail/bribe/correspond', 'tail/dice/cooperate', 'tail/dope/bargain', 'tail/evade/inquire', 'tail/exert/slump', 'tail/heap/grin', 'tail/orphan/adhere', 'tail/preclude/fluctuate', 'tail/slander/glare', 'tail/tame/chuckle', 'tail/utter/creep']).

| Condition | affected: Δ margin | affected: accuracy after (%) | affected: Δ accuracy (pp) | Head pairs: Δ margin | share of deficit closed |
|---|---|---|---|---|---|
| oracle | +0.528 [+0.238, +0.830] | +77.8 [+67.0, +88.4] | +6.5 [+2.1, +11.5] | +0.000 [+0.000, +0.000] | +0.49 [-1.67, +3.20] |
| oracle_T | +0.198 [+0.004, +0.488] | +73.0 [+61.0, +85.1] | +1.7 [-0.7, +5.7] | +0.000 [+0.000, +0.000] | +0.18 [-0.86, +1.55] |
| oracle_I | +0.330 [+0.094, +0.600] | +76.1 [+65.7, +85.9] | +4.8 [+1.2, +9.1] | -0.000 [-0.000, +0.000] | +0.31 [-0.76, +1.69] |
| labelfree | +0.247 [-0.083, +0.584] | +74.9 [+63.7, +85.8] | +3.6 [-1.0, +8.5] | +0.000 [-0.000, +0.000] | +0.23 [-0.82, +1.36] |

Unedited: affected-pair accuracy +71.3 [+59.7, +82.4]%; margins: Head pairs +2.956 [+2.223, +3.619], affected pairs +1.878 [+0.907, +2.888]; deficit +1.078 [-0.165, +2.268].

Accuracy (%) and whole-sentence margin by band, all pairs (95% CI):

| Condition | Head | Tail | XTail |
|---|---|---|---|
| unedited | +81.4 [+73.0, +88.5] / +2.96 [+2.22, +3.62] | +77.7 [+71.6, +83.2] / +2.76 [+2.12, +3.39] | +72.9 [+67.5, +77.8] / +2.13 [+1.59, +2.67] |
| oracle | +81.4 [+73.0, +88.5] / +2.96 [+2.22, +3.62] | +79.3 [+73.5, +84.5] / +2.88 [+2.25, +3.50] | +72.9 [+67.5, +77.8] / +2.13 [+1.59, +2.67] |
| oracle_T | +81.4 [+73.0, +88.5] / +2.96 [+2.22, +3.62] | +78.2 [+72.1, +83.7] / +2.81 [+2.16, +3.43] | +72.9 [+67.5, +77.8] / +2.13 [+1.59, +2.67] |
| oracle_I | +81.4 [+73.0, +88.5] / +2.96 [+2.22, +3.62] | +78.9 [+73.0, +84.2] / +2.84 [+2.21, +3.47] | +72.9 [+67.5, +77.8] / +2.13 [+1.59, +2.67] |
| labelfree | +81.4 [+73.0, +88.5] / +2.96 [+2.22, +3.62] | +78.6 [+72.6, +83.9] / +2.82 [+2.18, +3.44] | +72.9 [+67.5, +77.8] / +2.13 [+1.59, +2.67] |

## Curated `passive_2`

Affected pairs: 12 (['tail/approximate/muse', 'tail/bribe/correspond', 'tail/dice/cooperate', 'tail/dope/bargain', 'tail/evade/inquire', 'tail/exert/slump', 'tail/heap/grin', 'tail/orphan/adhere', 'tail/preclude/fluctuate', 'tail/slander/glare', 'tail/tame/chuckle', 'tail/utter/creep']).

| Condition | affected: Δ margin | affected: accuracy after (%) | affected: Δ accuracy (pp) | Head pairs: Δ margin | share of deficit closed |
|---|---|---|---|---|---|
| oracle | +0.528 [+0.259, +0.821] | +82.9 [+70.8, +92.7] | +5.1 [+1.6, +9.4] | -0.000 [-0.000, +0.000] | +1.36 [-10.12, +8.80] |
| oracle_T | +0.198 [-0.000, +0.482] | +80.2 [+67.5, +90.5] | +2.3 [-0.3, +6.4] | +0.000 [-0.000, +0.000] | +0.51 [-3.17, +3.26] |
| oracle_I | +0.330 [+0.087, +0.594] | +80.6 [+66.3, +92.4] | +2.8 [+0.5, +5.6] | -0.000 [-0.000, -0.000] | +0.85 [-6.82, +6.40] |
| labelfree | +0.247 [-0.063, +0.566] | +80.4 [+67.9, +91.1] | +2.6 [-0.6, +6.1] | -0.000 [-0.000, +0.000] | +0.64 [-4.20, +4.31] |

Unedited: affected-pair accuracy +77.8 [+62.8, +90.2]%; margins: Head pairs +2.565 [+1.891, +3.305], affected pairs +2.177 [+0.993, +3.233]; deficit +0.388 [-0.873, +1.761].

Accuracy (%) and whole-sentence margin by band, all pairs (95% CI):

| Condition | Head | Tail | XTail |
|---|---|---|---|
| unedited | +82.7 [+74.5, +89.8] / +2.56 [+1.89, +3.31] | +81.4 [+75.6, +86.7] / +2.60 [+2.05, +3.14] | +70.0 [+63.6, +76.2] / +1.51 [+1.01, +2.01] |
| oracle | +82.7 [+74.5, +89.8] / +2.56 [+1.89, +3.31] | +82.7 [+77.0, +87.5] / +2.73 [+2.22, +3.24] | +70.0 [+63.6, +76.2] / +1.51 [+1.01, +2.01] |
| oracle_T | +82.7 [+74.5, +89.8] / +2.56 [+1.89, +3.31] | +82.0 [+76.3, +87.0] / +2.65 [+2.12, +3.17] | +70.0 [+63.6, +76.2] / +1.51 [+1.01, +2.01] |
| oracle_I | +82.7 [+74.5, +89.8] / +2.56 [+1.89, +3.31] | +82.1 [+76.3, +87.3] / +2.68 [+2.14, +3.20] | +70.0 [+63.6, +76.2] / +1.51 [+1.01, +2.01] |
| labelfree | +82.7 [+74.5, +89.8] / +2.56 [+1.89, +3.31] | +82.0 [+76.4, +87.1] / +2.66 [+2.15, +3.18] | +70.0 [+63.6, +76.2] / +1.51 [+1.01, +2.01] |

Label-free − 0.5 × oracle (affected, passive_1): -0.017 [-0.218, +0.185] → **not achieved**.

## Released FreqBLiMP

Items containing an edited token: 1442 (passive_2: 164, passive_1: 155, transitive: 76, intransitive: 51, wh_vs_that_with_gap_long_distance: 48, wh_questions_subject_gap_long_distance: 43, wh_questions_subject_gap: 43, wh_island: 38, animate_subject_passive: 37, ellipsis_n_bar_2: 34, wh_vs_that_with_gap: 32, ellipsis_n_bar_1: 29, principle_A_case_2: 29, coordinate_structure_constraint_complex_left_branch: 29, principle_A_c_command: 28, wh_questions_object_gap: 26, sentential_negation_npi_scope: 26, complex_NP_island: 26, only_npi_scope: 25, wh_vs_that_no_gap_long_distance: 24, anaphor_number_agreement: 23, determiner_noun_agreement_1: 23, determiner_noun_agreement_2: 23, distractor_agreement_relative_clause: 22, matrix_question_npi_licensor_present: 21, determiner_noun_agreement_with_adj_irregular_2: 19, determiner_noun_agreement_with_adj_irregular_1: 19, principle_A_case_1: 18, superlative_quantifiers_1: 18, determiner_noun_agreement_with_adjective_1: 18, determiner_noun_agreement_with_adj_2: 18, principle_A_domain_1: 16, determiner_noun_agreement_irregular_2: 15, only_npi_licensor_present: 15, principle_A_reconstruction: 14, sentential_subject_island: 14, coordinate_structure_constraint_object_extraction: 13, irregular_plural_subject_verb_agreement_1: 13, expletive_it_object_raising: 13, determiner_noun_agreement_irregular_1: 13, adjunct_island: 13, superlative_quantifiers_2: 13, left_branch_island_simple_question: 12, sentential_negation_npi_licensor_present: 12, principle_A_domain_2: 11, regular_plural_subject_verb_agreement_1: 10, distractor_agreement_relational_noun: 10, wh_vs_that_no_gap: 10, irregular_plural_subject_verb_agreement_2: 9, npi_present_2: 8, npi_present_1: 8, left_branch_island_echo_question: 7, regular_plural_subject_verb_agreement_2: 7, irregular_past_participle_adjectives: 1).

**Passive paradigms** (accuracy % / whole-sentence margin, 95% CI over items; Δ = condition − unedited):

| paradigm | band | items (affected) | unedited | Δ oracle | Δ oracle_T | Δ oracle_I | Δ labelfree | affected: unedited acc → oracle / oracle_T / oracle_I / labelfree |
|---|---|---|---|---|---|---|---|---|
| passive_1 | head | 1000 (0) | +79.1 [+76.7, +81.6] / +2.69 [+2.50, +2.89] | +0.0 [+0.0, +0.0] / +0.000 [-0.000, +0.000] | +0.0 [+0.0, +0.0] / -0.000 [-0.000, +0.000] | +0.0 [+0.0, +0.0] / +0.000 [+0.000, +0.000] | +0.0 [+0.0, +0.0] / +0.000 [-0.000, +0.000] | — |
| passive_1 | tail | 1000 (155) | +79.2 [+76.7, +81.7] / +2.84 [+2.63, +3.07] | +0.4 [+0.1, +0.8] / +0.040 [+0.029, +0.051] | +0.0 [+0.0, +0.0] / +0.011 [+0.005, +0.018] | +0.4 [+0.1, +0.8] / +0.029 [+0.021, +0.038] | +0.1 [-0.3, +0.5] / +0.015 [+0.006, +0.024] | 73% → 75% / 73% / 75% / 74% |
| passive_1 | xtail | 1000 (0) | +69.4 [+66.5, +72.2] / +1.83 [+1.62, +2.05] | +0.0 [+0.0, +0.0] / +0.000 [+0.000, +0.000] | +0.0 [+0.0, +0.0] / -0.000 [-0.000, +0.000] | +0.0 [+0.0, +0.0] / +0.000 [+0.000, +0.000] | +0.0 [+0.0, +0.0] / +0.000 [-0.000, +0.000] | — |
| passive_2 | head | 1000 (0) | +81.5 [+79.1, +83.8] / +2.46 [+2.29, +2.64] | +0.0 [+0.0, +0.0] / -0.000 [-0.000, -0.000] | +0.0 [+0.0, +0.0] / -0.000 [-0.000, +0.000] | +0.0 [+0.0, +0.0] / -0.000 [-0.000, +0.000] | +0.0 [+0.0, +0.0] / -0.000 [-0.000, +0.000] | — |
| passive_2 | tail | 1000 (164) | +79.3 [+76.8, +81.7] / +2.44 [+2.25, +2.63] | +0.7 [+0.2, +1.2] / +0.047 [+0.036, +0.060] | +0.1 [+0.0, +0.3] / +0.013 [+0.006, +0.021] | +0.6 [+0.2, +1.1] / +0.034 [+0.026, +0.044] | +0.4 [+0.1, +0.8] / +0.017 [+0.007, +0.027] | 77% → 82% / 78% / 81% / 80% |
| passive_2 | xtail | 1000 (0) | +73.2 [+70.5, +75.8] / +1.76 [+1.57, +1.94] | +0.0 [+0.0, +0.0] / +0.000 [-0.000, +0.000] | +0.0 [+0.0, +0.0] / -0.000 [-0.000, +0.000] | +0.0 [+0.0, +0.0] / +0.000 [-0.000, +0.000] | +0.0 [+0.0, +0.0] / -0.000 [-0.000, +0.000] | — |

**Specificity, all 67 paradigms × 3 bands** (signed accuracy change per cell in `a4_cells.csv`; 201 unaffected cells, 60 cells with affected items):

| Condition | unaffected pooled: Δ acc (pp) | unaffected cells: mean / max |Δ| (pp) | cells with |Δ| > 0.5 pp | affected non-passive items: Δ acc (pp), n |
|---|---|---|---|---|
| oracle | +0.00 [+0.00, +0.00] | 0.00 / 0.00 | 0 | -0.1 [-0.4, +0.2], n 1123 |
| oracle_T | +0.00 [+0.00, +0.00] | 0.00 / 0.00 | 0 | -0.1 [-0.3, +0.0], n 1123 |
| oracle_I | +0.00 [+0.00, +0.00] | 0.00 / 0.00 | 0 | +0.0 [-0.3, +0.3], n 1123 |
| labelfree | +0.00 [+0.00, +0.00] | 0.00 / 0.00 | 0 | -0.3 [-0.7, +0.1], n 1123 |

## LM loss

Pile-validation windows per edited verb (occurrences found, deduplicated): {'adhered': 20, 'branched': 16, 'chuckled': 18, 'corresponded': 20, 'crept': 18, 'glared': 13, 'grinned': 11, 'inquired': 14, 'mutated': 20, 'abbreviated': 19, 'accorded': 18, 'approximated': 20, 'calibrated': 20, 'clenched': 11, 'deprecated': 16, 'diced': 17, 'disseminated': 19, 'doped': 18, 'elicited': 19, 'excised': 20, 'exerted': 20, 'formatted': 18, 'gripped': 13, 'inhibited': 20, 'perturbed': 18, 'precluded': 20, 'purified': 19, 'signaled': 14, 'subtracted': 20, 'superimposed': 20, 'transcribed': 18, 'uttered': 12}; verbs with none: none.

| Condition | Pile-10k: Δ NLL / token | Pile-val: Δ NLL / token (token-weighted) | Pile-val: per-verb macro Δ / token | edited tokens: Δ NLL pooled | per-verb macro | after be / perfect / other |
|---|---|---|---|---|---|---|
| oracle | +0.0000 [-0.0000, +0.0000] | +0.0001 [+0.0000, +0.0003] | +0.0001 [+0.0000, +0.0003] | +0.045 [-0.000, +0.099] | +0.058 [+0.012, +0.110] | +0.024 (n 196) / +0.028 (n 17) / +0.052 (n 619) |
| oracle_T | +0.0000 [-0.0000, +0.0000] | +0.0001 [+0.0000, +0.0002] | +0.0001 [+0.0000, +0.0002] | +0.042 [+0.005, +0.087] | +0.042 [+0.008, +0.083] | +0.026 (n 196) / +0.115 (n 17) / +0.044 (n 619) |
| oracle_I | +0.0000 [-0.0000, +0.0000] | +0.0000 [-0.0001, +0.0001] | +0.0000 [-0.0001, +0.0001] | +0.004 [-0.022, +0.034] | +0.017 [-0.015, +0.055] | -0.002 (n 196) / -0.087 (n 17) / +0.008 (n 619) |
| labelfree | +0.0000 [-0.0000, +0.0000] | +0.0002 [+0.0001, +0.0003] | +0.0002 [+0.0001, +0.0003] | +0.075 [+0.028, +0.133] | +0.087 [+0.040, +0.140] | +0.033 (n 196) / +0.134 (n 17) / +0.087 (n 619) |

## Declared decisions

- Oracle improves rare passives (affected curated passive_1, Δ margin CI above 0 and ≥ 0.25): **True**.
- Specific (Head ±0.05; unaffected ±0.5 pp; Pile-10k < +0.002): oracle **True**, label-free True.
- Label-free achieves half the oracle: **False**.
- Side arms (reported): T only improves False; I only improves True.

