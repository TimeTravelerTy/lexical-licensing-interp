# OLMo-2-1B: nonce verbs and DAS controls (results, Nov 23)

Rules: `reports/olmo/nonce_controls_rules.md` (declared before running). TSUBAME jobs 9531737 (nonce), 9531738 (controls), commit e84686f. Generated reports: `nonce/round3_nonce.md`, `nonce/round4_nonce_cues.md`, `nonce/splits/`, `das_controls.md`. Onset table: `results/olmo/das/nonce/onset_by_depth.csv`.

**Checks:** 195 bases exported, all bit-identical to `transfer/bases_rank1.npz`; the runners' token-suffix checks passed; no traceback. OLMo's tokenizer: 17 of 80 lemmas have a bare "ed" last piece, 68 have the same -ed token count as Pythia.

## Nonce: onset by depth (round-3 C8, passive probe, z along each site's d)

Decision (a) context-sensitive (Δ_matched CI above 0 and ≥ 0.05 z); (b) verb-specific (Δ_balanced, same bar). All 80 lemmas:

| site | depth | Δ_matched z | (a) | Δ_balanced z | (b) |
|---:|---:|---|---|---|---|
| 0 | 0.00 | +0.000 | no | +0.000 | no |
| 1 | 0.06 | −0.003 [−0.004, −0.002] | no | +0.000 | no |
| 2 | 0.12 | −0.003 [−0.004, −0.002] | no | +0.001 | no |
| 3 | 0.19 | +0.005 [+0.003, +0.006] | no | −0.000 | no |
| 4 | 0.25 | −0.004 [−0.005, −0.002] | no | +0.000 | no |
| **5** | **0.31** | +0.072 [+0.068, +0.077] | **yes** | +0.052 [+0.046, +0.057] | **yes** |
| 6 | 0.38 | +0.063 [+0.057, +0.069] | yes | +0.038 [+0.033, +0.043] | no |
| 7 | 0.44 | +0.052 [+0.046, +0.057] | yes | +0.042 [+0.037, +0.047] | no |
| 8 | 0.50 | +0.004 [−0.003, +0.010] | **no** | +0.035 [+0.030, +0.040] | no |
| 9 | 0.56 | +0.139 [+0.132, +0.146] | yes | +0.122 [+0.114, +0.130] | yes |
| 11 | 0.69 | +0.142 | yes | +0.111 | yes |
| 12 | 0.75 | +0.151 | yes | +0.098 | yes |
| 13 | 0.81 | +0.185 | yes | +0.098 | yes |

First "yes", by relative depth: OLMo (a) 0.31, (b) 0.31; Pythia (a) 0.12, (b) 0.25; Gemma (a) 0.23, (b) 0.23. The same onset (site 5) holds in all three OLMo tokenizer splits (bare-ed, stem-final, same count as Pythia), except (b) in the 17 bare-ed lemmas, which never reaches yes in 0–8.

**Declared reading: the replication criterion is not met as declared.** (a) does hold by depth ≤ 0.375 (at 0.31). But it does not stay "yes" at every later site: site 8 falls to +0.004 (CI includes 0), between +0.05 at 7 and +0.14 at 9. Sites 6–8 also miss the 0.05 bar for (b) while their CIs stay above 0. Description: OLMo's nonce verbs pick up the class from context from 0.31 of depth, as in the other two models (0.12–0.25), with a dip at site 8 on that site's own basis.

**Declared decisions (sites 3 / 5 / 9):** (a) no / yes / yes; (b) no / yes / yes; *by* raised verb-specifically, yes at all three (Δ_balanced *by* +0.21 vs PREP\by +0.01). Causal swap (matched-T coordinate into matched-I): none at 3; at 5 *by* +0.03 and O +0.02; at 9 *by* +0.06 and O +0.18, all with CIs above 0.

**Round 4 (C9, which cue drives the shift):** NP-driven at sites 5 and 9 (adjacent NP +0.067 / +0.143 z against transitivity +0.020 / +0.044), neither at 3. This is the same reading as Pythia at 6 / 8 ("an NP follows", context-derived). *by*: both (primary), NP-driven (secondary). C8 (inflection mismatch): the d shift survives with -ing at 5 and 9 but not with -s, and *by* survives both.

## DAS controls (active task)

At every OLMo site (0–16), trained held-out IIA is above the 95th percentile of 20 random rank-1 directions (random ≈ unpatched ≈ 0.015). A linear probe reads T vs I at 0.88 already at site 0 and at ≥ 0.99 from site 3, where trained DAS IIA is still 0.43. So, as in Pythia and Gemma, the low early IIA is not missing class information: the class is linearly present before a single direction controls the readout. Passive-task controls are not run (OLMo has no passive sweep).

## What this changes

- **Claim "shallow, context-derived class (nonce onset)":** OLMo now has a result instead of "not run". Onset at 0.31 of depth, NP-driven, as in Pythia and Gemma. **Flag:** the declared criterion fails on one site (8). The paper should show the curve and say "onset at 0.31, with a dip at 0.50", not "stays separated from onset".
- **Controls:** none; they match the other two models.
