# Passive test of the round-2 direction: results

Spec, declared and committed before any passive evaluation: `passive_test_plan.md`. Run: `run_passive_test.py` (OLMo-2-1B v2, jobs/v2_chain.sh, job 9551323); analysis: `analyze_passive_test.py` (2,000 draws, seed 17; three-way cluster bootstrap over base verb pairs within band, contexts and donor verb pairs). Tables: `results/das_round2/passive_test/*_summary.csv`.

Fidelity checks (per site): a self-patch reproduces the unpatched readout exactly (max |dM| = 0); recomputing 512 held-out active swaps of each site's own run gives median |dM| 0.000, 0.000, 0.000, 0.000 and max 0.03, 0.02, 0.07, 0.04 (sites 13, 12, 11, 9; bf16 batch effects, M is on a scale of about +-5).

## Verdict (primary population: 64 pairs with a plain bad verb, all bands pooled)

D = mean change after a transitive active donor minus after an intransitive active donor, on bad passive bases ("The house was emerged"), in nats.

| Site | Role | δ_s | D log P(O) | class | D log P(" by") | class | D log P(".") | class | Outcome |
|---:|---|---:|---|---|---|---|---|---|---|
| 13 | primary | 0.67 | 4.05 [3.85, 4.25] | RISE | -1.80 [-2.01, -1.60] | FALL | -2.56 [-2.79, -2.34] | FALL | **surface (object next)** |
| 12 | secondary | 0.63 | 3.57 [3.38, 3.76] | RISE | -0.73 [-0.86, -0.61] | FALL | -1.59 [-1.75, -1.44] | FALL | **surface (object next)** |
| 11 | secondary | 0.59 | 2.34 [2.18, 2.51] | RISE | 0.26 [0.20, 0.32] | NO RISE | -0.78 [-0.88, -0.69] | FALL | **surface (object next)** |
| 9 | secondary | 0.54 | 1.53 [1.40, 1.66] | RISE | 0.48 [0.42, 0.54] | NO RISE | -0.37 [-0.43, -0.31] | NO RISE | **surface (object next)** |

Rules (declared): RISE = 95% CI above 0 and estimate >= δ_s; NO RISE = 90% CI within ±δ_s; outcome from the O and " by" classes.

### In probabilities

Mean probability over primary bad passive bases (and, for reference, good passives unpatched).

| Condition | P(O) | P(pronouns) | P(" the") | P(" by") | P(".") | P(I) |
|---|---:|---:|---:|---:|---:|---:|
| bad passive, unpatched | 0.015 | 0.0004 | 0.005 | 0.093 | 0.013 | 0.666 |
| site 9, T donor | 0.060 | 0.0038 | 0.025 | 0.135 | 0.010 | 0.583 |
| site 9, I donor | 0.014 | 0.0004 | 0.004 | 0.089 | 0.013 | 0.673 |
| site 11, T donor | 0.132 | 0.0128 | 0.055 | 0.111 | 0.008 | 0.525 |
| site 11, I donor | 0.013 | 0.0003 | 0.004 | 0.088 | 0.014 | 0.675 |
| site 12, T donor | 0.412 | 0.1309 | 0.158 | 0.053 | 0.004 | 0.301 |
| site 12, I donor | 0.015 | 0.0005 | 0.004 | 0.092 | 0.013 | 0.669 |
| site 13, T donor | 0.709 | 0.5737 | 0.079 | 0.021 | 0.002 | 0.137 |
| site 13, I donor | 0.018 | 0.0018 | 0.005 | 0.094 | 0.013 | 0.663 |
| good passive, unpatched | 0.016 | 0.0004 | 0.006 | 0.178 | 0.015 | 0.637 |

### Where the change goes (D on bad bases, primary)

| Readout | site 13 | site 12 | site 11 | site 9 |
|---|---|---|---|---|
| log P(O) | 4.05 [3.85, 4.25] | 3.57 [3.38, 3.76] | 2.34 [2.18, 2.51] | 1.53 [1.40, 1.66] |
| log P(det) | 2.26 [2.07, 2.46] | 3.23 [3.06, 3.41] | 2.23 [2.08, 2.40] | 1.48 [1.35, 1.61] |
| log P(pron) | 7.27 [6.93, 7.58] | 5.88 [5.59, 6.14] | 3.52 [3.29, 3.74] | 2.16 [2.00, 2.32] |
| log P(refl) | 1.97 [1.75, 2.20] | 3.61 [3.38, 3.82] | 3.32 [3.05, 3.59] | 2.17 [1.99, 2.36] |
| log P(" the") | 3.01 [2.78, 3.24] | 4.00 [3.79, 4.21] | 2.85 [2.65, 3.06] | 2.00 [1.85, 2.14] |
| log P(" him") | 4.30 [3.97, 4.62] | 6.55 [6.21, 6.86] | 4.60 [4.31, 4.88] | 2.98 [2.77, 3.17] |
| log P(" by") | -1.80 [-2.01, -1.60] | -0.73 [-0.86, -0.61] | 0.26 [0.20, 0.32] | 0.48 [0.42, 0.54] |
| log P(".") | -2.56 [-2.79, -2.34] | -1.59 [-1.75, -1.44] | -0.78 [-0.88, -0.69] | -0.37 [-0.43, -0.31] |
| log P(I) | -1.89 [-2.10, -1.69] | -0.91 [-1.02, -0.80] | -0.27 [-0.31, -0.23] | -0.15 [-0.18, -0.13] |
| M | 5.94 [5.59, 6.29] | 4.48 [4.21, 4.75] | 2.61 [2.42, 2.81] | 1.68 [1.54, 1.83] |

D(" by") as a fraction of the natural good − bad gap in log P(" by") on the same items (0.48 nats): site 13 -3.77, site 12 -1.53, site 11 0.54, site 9 1.01. D(O) as a fraction of the site's active effect: site 13 1.21, site 12 1.14, site 11 0.80, site 9 0.57.

## Controls (primary, Δ from unpatched)

| Condition | site 13: O / by / . | site 12: O / by / . | site 11: O / by / . | site 9: O / by / . |
|---|---|---|---|
| T donor → bad passive | +4.14 / -1.80 / -2.58 | +3.54 / -0.74 / -1.56 | +2.24 / +0.20 / -0.72 | +1.45 / +0.43 / -0.35 |
| I donor → bad passive (voice-change baseline) | +0.09 / -0.00 / -0.01 | -0.04 / -0.01 / +0.03 | -0.10 / -0.06 / +0.06 | -0.08 / -0.06 / +0.02 |
| own active → bad passive (same verb) | +0.09 / -0.01 / -0.01 | -0.03 / -0.03 / +0.03 | -0.11 / -0.07 / +0.06 | -0.07 / -0.06 / +0.02 |
| good passive → bad passive | +0.40 / -0.00 / -0.13 | +0.46 / +0.04 / -0.15 | +0.38 / +0.11 / -0.12 | +0.29 / +0.15 / -0.05 |
| T donor → good passive (good → good) | +4.02 / -1.05 / -1.76 | +3.37 / -0.49 / -1.02 | +1.85 / +0.05 / -0.48 | +1.20 / +0.13 / -0.27 |
| I donor → good passive | -0.22 / +0.03 / +0.13 | -0.27 / -0.02 / +0.15 | -0.25 / -0.09 / +0.19 | -0.24 / -0.12 / +0.10 |
| own active → good passive (same verb) | +3.73 / -0.84 / -1.52 | +3.08 / -0.35 / -0.86 | +1.66 / +0.05 / -0.43 | +1.07 / +0.11 / -0.23 |
| bad passive → good passive | -0.28 / +0.03 / +0.14 | -0.26 / -0.01 / +0.12 | -0.20 / -0.05 / +0.13 | -0.18 / -0.07 / +0.08 |
| D on good bases | +4.24 / -1.08 / -1.89 | +3.64 / -0.47 / -1.17 | +2.10 / +0.14 / -0.67 | +1.43 / +0.25 / -0.36 |
| D, participle-matched donors | +3.95 / -1.76 / -2.50 | +3.44 / -0.70 / -1.53 | +2.22 / +0.25 / -0.73 | +1.45 / +0.46 / -0.35 |
| natural good − bad (unpatched) | -0.29 / +0.48 / +0.01 | -0.29 / +0.48 / +0.01 | -0.29 / +0.48 / +0.01 | -0.29 / +0.48 / +0.01 |

## Step 1: projection of natural participles onto d

z: 0 = mean of held-out intransitive actives, 1 = held-out transitive actives (per basis, cross-fitted). Gap = good − bad (two-way bootstrap over pairs and contexts / subjects).

| Site | Frame | Band | Pairs | Gap in z | good z | bad z | AUC items | AUC verbs | Win rate |
|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| 13 | passive | all | 64 | 0.14 [0.11, 0.17] | 0.11 | -0.03 | 0.836 | 0.882 | 0.861 |
| 13 | passive | head | 9 | 0.17 [0.09, 0.26] | 0.24 | 0.07 | 0.895 | 0.975 | 0.881 |
| 13 | passive | tail | 28 | 0.14 [0.10, 0.20] | 0.10 | -0.04 | 0.858 | 0.915 | 0.856 |
| 13 | passive | xtail | 27 | 0.12 [0.09, 0.16] | 0.08 | -0.04 | 0.853 | 0.911 | 0.861 |
| 13 | active | all | 64 | 0.95 [0.88, 1.01] | 0.94 | -0.01 | 0.998 | 1.000 | 1.000 |
| 13 | active | head | 9 | 0.95 [0.84, 1.05] | 0.93 | -0.02 | 1.000 | 1.000 | 1.000 |
| 13 | active | tail | 28 | 1.03 [0.93, 1.12] | 0.97 | -0.06 | 1.000 | 1.000 | 1.000 |
| 13 | active | xtail | 27 | 0.86 [0.75, 0.97] | 0.91 | 0.05 | 0.996 | 0.997 | 1.000 |
| 12 | passive | all | 64 | 0.18 [0.15, 0.20] | 0.22 | 0.05 | 0.939 | 0.965 | 0.941 |
| 12 | passive | head | 9 | 0.26 [0.20, 0.32] | 0.31 | 0.05 | 0.993 | 1.000 | 0.989 |
| 12 | passive | tail | 28 | 0.18 [0.14, 0.22] | 0.22 | 0.04 | 0.943 | 0.969 | 0.934 |
| 12 | passive | xtail | 27 | 0.15 [0.11, 0.19] | 0.20 | 0.05 | 0.920 | 0.955 | 0.932 |
| 12 | active | all | 64 | 0.93 [0.86, 1.00] | 0.93 | 0.00 | 0.999 | 1.000 | 1.000 |
| 12 | active | head | 9 | 0.96 [0.84, 1.06] | 0.93 | -0.03 | 1.000 | 1.000 | 1.000 |
| 12 | active | tail | 28 | 1.01 [0.91, 1.09] | 0.97 | -0.04 | 1.000 | 1.000 | 1.000 |
| 12 | active | xtail | 27 | 0.85 [0.74, 0.95] | 0.90 | 0.05 | 0.997 | 0.999 | 1.000 |
| 11 | passive | all | 64 | 0.21 [0.18, 0.23] | 0.32 | 0.11 | 0.963 | 0.981 | 0.964 |
| 11 | passive | head | 9 | 0.28 [0.23, 0.33] | 0.40 | 0.12 | 0.997 | 1.000 | 1.000 |
| 11 | passive | tail | 28 | 0.21 [0.17, 0.25] | 0.31 | 0.10 | 0.969 | 0.989 | 0.960 |
| 11 | passive | xtail | 27 | 0.18 [0.14, 0.21] | 0.30 | 0.13 | 0.948 | 0.978 | 0.956 |
| 11 | active | all | 64 | 0.93 [0.86, 0.99] | 0.94 | 0.01 | 0.999 | 1.000 | 1.000 |
| 11 | active | head | 9 | 0.94 [0.82, 1.05] | 0.94 | -0.01 | 1.000 | 1.000 | 1.000 |
| 11 | active | tail | 28 | 0.99 [0.89, 1.08] | 0.96 | -0.03 | 1.000 | 1.000 | 1.000 |
| 11 | active | xtail | 27 | 0.86 [0.76, 0.96] | 0.91 | 0.05 | 0.997 | 0.999 | 1.000 |
| 9 | passive | all | 64 | 0.24 [0.21, 0.27] | 0.34 | 0.10 | 0.955 | 0.973 | 0.958 |
| 9 | passive | head | 9 | 0.27 [0.21, 0.33] | 0.40 | 0.13 | 0.985 | 1.000 | 0.998 |
| 9 | passive | tail | 28 | 0.25 [0.19, 0.30] | 0.32 | 0.08 | 0.958 | 0.972 | 0.944 |
| 9 | passive | xtail | 27 | 0.23 [0.19, 0.28] | 0.34 | 0.10 | 0.949 | 0.979 | 0.959 |
| 9 | active | all | 64 | 0.93 [0.86, 1.00] | 0.94 | 0.01 | 0.999 | 1.000 | 1.000 |
| 9 | active | head | 9 | 0.91 [0.78, 1.01] | 0.91 | -0.00 | 1.000 | 1.000 | 1.000 |
| 9 | active | tail | 28 | 1.00 [0.89, 1.09] | 0.96 | -0.04 | 1.000 | 1.000 | 1.000 |
| 9 | active | xtail | 27 | 0.87 [0.76, 0.98] | 0.92 | 0.05 | 0.997 | 1.000 | 1.000 |

## By band (primary) and separate groups (descriptive)

D on bad bases, O / " by" / "." (point estimates; CIs in `transfer_summary.csv`).

| Population | Band | Pairs | site 13 | site 12 | site 11 | site 9 |
|---|---|---:|---|---|---|
| primary (plain) | head | 9 | +3.32 / -1.20 / -2.43 | +2.64 / -0.19 / -1.54 | +2.00 / +0.60 / -1.10 | +1.35 / +0.90 / -0.55 |
| primary (plain) | tail | 28 | +4.00 / -1.72 / -2.48 | +3.55 / -0.64 / -1.47 | +2.41 / +0.24 / -0.76 | +1.55 / +0.46 / -0.35 |
| primary (plain) | xtail | 27 | +4.35 / -2.08 / -2.69 | +3.91 / -1.01 / -1.73 | +2.38 / +0.16 / -0.70 | +1.57 / +0.37 / -0.34 |
| primary, original pairs only | all | 59 | +4.07 / -1.79 / -2.56 | +3.60 / -0.73 / -1.60 | +2.38 / +0.26 / -0.80 | +1.54 / +0.48 / -0.39 |
| primary, without bet/appear | all | 63 | +4.06 / -1.81 / -2.57 | +3.59 / -0.74 / -1.59 | +2.34 / +0.25 / -0.77 | +1.53 / +0.47 / -0.37 |
| primary, cross-fitted orig_head pairs | all | 8 | +3.28 / -1.23 / -2.42 | +2.61 / -0.21 / -1.54 | +1.99 / +0.58 / -1.08 | +1.34 / +0.90 / -0.57 |
| primary, non-training pairs | all | 56 | +4.16 / -1.88 / -2.59 | +3.71 / -0.81 / -1.60 | +2.39 / +0.21 / -0.74 | +1.56 / +0.42 / -0.34 |
| primary + contaminated_bad | all | 71 | +4.09 / -1.80 / -2.54 | +3.60 / -0.74 / -1.58 | +2.33 / +0.26 / -0.76 | +1.54 / +0.48 / -0.35 |
| primary, by reliable | all | 41 | +4.04 / -1.84 / -2.60 | +3.58 / -0.76 / -1.63 | +2.38 / +0.27 / -0.78 | +1.55 / +0.52 / -0.38 |
| primary, by negative | all | 15 | +4.06 / -1.60 / -2.47 | +3.54 / -0.57 / -1.50 | +2.36 / +0.24 / -0.85 | +1.53 / +0.41 / -0.40 |
| new Head eval pairs (prep_object) | all | 6 | +5.12 / -1.41 / -2.23 | +3.99 / -0.69 / -1.19 | +2.35 / +0.95 / -0.14 | +2.19 / +1.45 / +0.40 |
| prep_object, pseudo_passive_ok | all | 56 | +4.86 / -1.32 / -2.02 | +4.01 / -0.58 / -1.01 | +2.57 / +0.71 / -0.17 | +2.32 / +1.18 / +0.33 |
| prep_object, pseudo_passive_bad | all | 18 | +4.49 / -1.31 / -2.29 | +3.92 / -0.45 / -1.37 | +2.59 / +0.45 / -0.70 | +1.81 / +0.68 / -0.39 |
| contaminated_bad | all | 7 | +4.43 / -1.80 / -2.36 | +3.82 / -0.78 / -1.44 | +2.27 / +0.28 / -0.57 | +1.59 / +0.46 / -0.20 |
| bad-side-high by | all | 11 | +4.27 / -1.40 / -2.19 | +3.73 / -0.43 / -1.25 | +2.47 / +0.32 / -0.67 | +1.74 / +0.47 / -0.28 |
| original, by reliable | all | 96 | +4.46 / -1.58 / -2.33 | +3.81 / -0.66 / -1.35 | +2.47 / +0.48 / -0.50 | +1.90 / +0.82 / -0.07 |
| original, by negative | all | 27 | +4.23 / -1.49 / -2.30 | +3.66 / -0.52 / -1.32 | +2.46 / +0.34 / -0.72 | +1.73 / +0.54 / -0.29 |
| original, by n.s. positive | all | 3 | +4.61 / -2.01 / -2.49 | +4.18 / -1.13 / -1.78 | +2.40 / +0.16 / -0.77 | +1.60 / +0.40 / -0.42 |
| all pairs | all | 145 | +4.44 / -1.55 / -2.31 | +3.80 / -0.64 / -1.33 | +2.45 / +0.46 / -0.52 | +1.87 / +0.77 / -0.09 |

