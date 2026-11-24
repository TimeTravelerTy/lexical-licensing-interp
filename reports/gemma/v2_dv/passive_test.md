# Passive test of the round-2 direction: results

Spec, declared and committed before any passive evaluation: `passive_test_plan.md`. Run: `run_passive_test.py` (Gemma-3-1B v2, jobs/v2_chain.sh, job 9558133); analysis: `analyze_passive_test.py` (2,000 draws, seed 17; three-way cluster bootstrap over base verb pairs within band, contexts and donor verb pairs). Tables: `results/das_round2/passive_test/*_summary.csv`.

Fidelity checks (per site): a self-patch reproduces the unpatched readout exactly (max |dM| = 0); recomputing 512 held-out active swaps of each site's own run gives median |dM| 0.000, 0.000, 0.000, 0.000 and max 0.05, 0.06, 0.04, 0.09 (sites 15, 12, 8, 5; bf16 batch effects, M is on a scale of about +-5).

## Verdict (primary population: 64 pairs with a plain bad verb, all bands pooled)

D = mean change after a transitive active donor minus after an intransitive active donor, on bad passive bases ("The house was emerged"), in nats.

| Site | Role | δ_s | D log P(O) | class | D log P(" by") | class | D log P(".") | class | Outcome |
|---:|---|---:|---|---|---|---|---|---|---|
| 15 | primary | 0.58 | 2.97 [2.76, 3.19] | RISE | 0.92 [0.82, 1.02] | RISE | -0.68 [-0.80, -0.58] | FALL | **mixed** |
| 12 | secondary | 0.56 | 1.73 [1.57, 1.89] | RISE | 0.99 [0.87, 1.12] | RISE | -0.58 [-0.65, -0.51] | FALL | **mixed** |
| 8 | secondary | 0.53 | 1.16 [1.04, 1.29] | RISE | 0.72 [0.61, 0.83] | RISE | -0.14 [-0.19, -0.10] | NO RISE | **mixed** |
| 5 | secondary | 0.54 | 0.86 [0.70, 1.03] | RISE | 0.52 [0.40, 0.65] | unresolved | 0.00 [-0.09, 0.09] | NO RISE | **surface (object next)** |

Rules (declared): RISE = 95% CI above 0 and estimate >= δ_s; NO RISE = 90% CI within ±δ_s; outcome from the O and " by" classes.

### In probabilities

Mean probability over primary bad passive bases (and, for reference, good passives unpatched).

| Condition | P(O) | P(pronouns) | P(" the") | P(" by") | P(".") | P(I) |
|---|---:|---:|---:|---:|---:|---:|
| bad passive, unpatched | 0.017 | 0.0006 | 0.006 | 0.113 | 0.022 | 0.682 |
| site 5, T donor | 0.056 | 0.0097 | 0.019 | 0.163 | 0.021 | 0.645 |
| site 5, I donor | 0.017 | 0.0006 | 0.006 | 0.113 | 0.022 | 0.687 |
| site 8, T donor | 0.054 | 0.0049 | 0.020 | 0.182 | 0.020 | 0.596 |
| site 8, I donor | 0.016 | 0.0005 | 0.005 | 0.105 | 0.023 | 0.691 |
| site 12, T donor | 0.089 | 0.0076 | 0.030 | 0.211 | 0.015 | 0.538 |
| site 12, I donor | 0.015 | 0.0004 | 0.005 | 0.097 | 0.024 | 0.703 |
| site 15, T donor | 0.253 | 0.0369 | 0.081 | 0.229 | 0.013 | 0.384 |
| site 15, I donor | 0.014 | 0.0004 | 0.004 | 0.100 | 0.022 | 0.702 |
| good passive, unpatched | 0.020 | 0.0008 | 0.007 | 0.179 | 0.023 | 0.653 |

### Where the change goes (D on bad bases, primary)

| Readout | site 15 | site 12 | site 8 | site 5 |
|---|---|---|---|---|
| log P(O) | 2.97 [2.76, 3.19] | 1.73 [1.57, 1.89] | 1.16 [1.04, 1.29] | 0.86 [0.70, 1.03] |
| log P(det) | 2.71 [2.52, 2.92] | 1.62 [1.47, 1.78] | 1.09 [0.97, 1.21] | 0.72 [0.58, 0.86] |
| log P(pron) | 4.56 [4.27, 4.84] | 2.78 [2.55, 2.99] | 2.10 [1.90, 2.29] | 1.91 [1.58, 2.24] |
| log P(refl) | 5.15 [4.72, 5.60] | 3.11 [2.75, 3.49] | 1.86 [1.62, 2.11] | 1.86 [1.52, 2.23] |
| log P(" the") | 3.19 [2.97, 3.41] | 1.93 [1.77, 2.10] | 1.36 [1.23, 1.50] | 0.94 [0.76, 1.13] |
| log P(" him") | 5.38 [5.02, 5.72] | 3.45 [3.17, 3.71] | 2.60 [2.37, 2.83] | 2.55 [2.16, 2.95] |
| log P(" by") | 0.92 [0.82, 1.02] | 0.99 [0.87, 1.12] | 0.72 [0.61, 0.83] | 0.52 [0.40, 0.65] |
| log P(".") | -0.68 [-0.80, -0.58] | -0.58 [-0.65, -0.51] | -0.14 [-0.19, -0.10] | 0.00 [-0.09, 0.09] |
| log P(I) | -0.69 [-0.81, -0.59] | -0.28 [-0.33, -0.24] | -0.15 [-0.18, -0.13] | -0.07 [-0.11, -0.03] |
| M | 3.66 [3.37, 3.97] | 2.01 [1.82, 2.21] | 1.31 [1.17, 1.45] | 0.93 [0.74, 1.12] |

D(" by") as a fraction of the natural good − bad gap in log P(" by") on the same items (0.41 nats): site 15 2.24, site 12 2.42, site 8 1.75, site 5 1.26. D(O) as a fraction of the site's active effect: site 15 1.03, site 12 0.62, site 8 0.44, site 5 0.32.

## Controls (primary, Δ from unpatched)

| Condition | site 15: O / by / . | site 12: O / by / . | site 8: O / by / . | site 5: O / by / . |
|---|---|---|---|
| T donor → bad passive | +2.74 / +0.78 / -0.67 | +1.59 / +0.82 / -0.50 | +1.10 / +0.66 / -0.12 | +0.86 / +0.53 / +0.01 |
| I donor → bad passive (voice-change baseline) | -0.23 / -0.13 / +0.02 | -0.14 / -0.17 / +0.08 | -0.06 / -0.06 / +0.02 | +0.00 / +0.01 / +0.01 |
| own active → bad passive (same verb) | -0.12 / -0.12 / +0.01 | -0.09 / -0.17 / +0.06 | -0.04 / -0.07 / +0.01 | +0.00 / +0.00 / -0.01 |
| good passive → bad passive | +0.47 / +0.22 / -0.07 | +0.41 / +0.35 / -0.13 | +0.46 / +0.41 / -0.06 | +0.28 / +0.35 / +0.05 |
| T donor → good passive (good → good) | +2.67 / +0.41 / -0.49 | +1.40 / +0.31 / -0.33 | +0.72 / +0.16 / -0.12 | +0.21 / +0.05 / +0.04 |
| I donor → good passive | -0.51 / -0.17 / +0.08 | -0.40 / -0.34 / +0.21 | -0.34 / -0.30 / +0.15 | -0.41 / -0.85 / +0.27 |
| own active → good passive (same verb) | +2.45 / +0.35 / -0.46 | +1.26 / +0.25 / -0.30 | +0.49 / +0.11 / -0.10 | +0.05 / +0.01 / +0.01 |
| bad passive → good passive | -0.33 / -0.09 / +0.06 | -0.29 / -0.18 / +0.15 | -0.30 / -0.22 / +0.14 | -0.38 / -0.84 / +0.27 |
| D on good bases | +3.17 / +0.58 / -0.57 | +1.80 / +0.65 / -0.53 | +1.06 / +0.45 / -0.27 | +0.62 / +0.90 / -0.23 |
| D, participle-matched donors | +2.62 / +0.81 / -0.61 | +1.56 / +0.87 / -0.52 | +0.99 / +0.60 / -0.12 | +0.61 / +0.41 / +0.02 |
| natural good − bad (unpatched) | -0.19 / +0.41 / -0.09 | -0.19 / +0.41 / -0.09 | -0.19 / +0.41 / -0.09 | -0.19 / +0.41 / -0.09 |

## Step 1: projection of natural participles onto d

z: 0 = mean of held-out intransitive actives, 1 = held-out transitive actives (per basis, cross-fitted). Gap = good − bad (two-way bootstrap over pairs and contexts / subjects).

| Site | Frame | Band | Pairs | Gap in z | good z | bad z | AUC items | AUC verbs | Win rate |
|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| 15 | passive | all | 64 | 0.15 [0.13, 0.18] | 0.28 | 0.12 | 0.859 | 0.918 | 0.857 |
| 15 | passive | head | 9 | 0.26 [0.18, 0.34] | 0.35 | 0.09 | 0.964 | 1.000 | 0.957 |
| 15 | passive | tail | 28 | 0.15 [0.11, 0.19] | 0.27 | 0.12 | 0.859 | 0.931 | 0.863 |
| 15 | passive | xtail | 27 | 0.13 [0.09, 0.17] | 0.26 | 0.13 | 0.809 | 0.861 | 0.818 |
| 15 | active | all | 64 | 0.88 [0.80, 0.96] | 0.92 | 0.04 | 0.991 | 0.992 | 0.987 |
| 15 | active | head | 9 | 0.96 [0.81, 1.09] | 0.94 | -0.02 | 1.000 | 1.000 | 1.000 |
| 15 | active | tail | 28 | 0.93 [0.80, 1.04] | 0.93 | 0.01 | 0.995 | 0.997 | 0.969 |
| 15 | active | xtail | 27 | 0.81 [0.68, 0.92] | 0.90 | 0.09 | 0.981 | 0.979 | 1.000 |
| 12 | passive | all | 64 | 0.26 [0.23, 0.29] | 0.43 | 0.17 | 0.947 | 0.969 | 0.947 |
| 12 | passive | head | 9 | 0.39 [0.31, 0.47] | 0.48 | 0.09 | 0.999 | 1.000 | 1.000 |
| 12 | passive | tail | 28 | 0.26 [0.21, 0.31] | 0.42 | 0.16 | 0.953 | 0.978 | 0.946 |
| 12 | passive | xtail | 27 | 0.22 [0.17, 0.26] | 0.41 | 0.19 | 0.913 | 0.945 | 0.931 |
| 12 | active | all | 64 | 0.89 [0.81, 0.96] | 0.92 | 0.03 | 0.992 | 0.994 | 0.984 |
| 12 | active | head | 9 | 0.94 [0.81, 1.06] | 0.94 | -0.01 | 1.000 | 1.000 | 1.000 |
| 12 | active | tail | 28 | 0.93 [0.80, 1.04] | 0.93 | -0.01 | 0.996 | 0.997 | 0.969 |
| 12 | active | xtail | 27 | 0.83 [0.71, 0.94] | 0.91 | 0.08 | 0.982 | 0.986 | 0.995 |
| 8 | passive | all | 64 | 0.41 [0.37, 0.46] | 0.52 | 0.11 | 0.975 | 0.979 | 0.970 |
| 8 | passive | head | 9 | 0.63 [0.55, 0.70] | 0.61 | -0.02 | 1.000 | 1.000 | 1.000 |
| 8 | passive | tail | 28 | 0.44 [0.36, 0.50] | 0.53 | 0.10 | 0.986 | 0.991 | 0.954 |
| 8 | passive | xtail | 27 | 0.32 [0.26, 0.38] | 0.48 | 0.16 | 0.945 | 0.951 | 0.976 |
| 8 | active | all | 64 | 0.82 [0.75, 0.89] | 0.85 | 0.02 | 0.992 | 0.993 | 0.982 |
| 8 | active | head | 9 | 0.95 [0.83, 1.06] | 0.91 | -0.04 | 1.000 | 1.000 | 1.000 |
| 8 | active | tail | 28 | 0.85 [0.72, 0.97] | 0.87 | 0.01 | 0.993 | 0.996 | 0.959 |
| 8 | active | xtail | 27 | 0.75 [0.65, 0.85] | 0.81 | 0.06 | 0.985 | 0.985 | 1.000 |
| 5 | passive | all | 64 | 0.56 [0.50, 0.61] | 0.57 | 0.02 | 0.983 | 0.985 | 0.993 |
| 5 | passive | head | 9 | 0.87 [0.74, 1.01] | 0.71 | -0.17 | 1.000 | 1.000 | 1.000 |
| 5 | passive | tail | 28 | 0.57 [0.47, 0.66] | 0.58 | 0.01 | 0.989 | 0.990 | 1.000 |
| 5 | passive | xtail | 27 | 0.44 [0.36, 0.52] | 0.52 | 0.08 | 0.964 | 0.967 | 0.985 |
| 5 | active | all | 64 | 0.77 [0.70, 0.84] | 0.79 | 0.01 | 0.992 | 0.993 | 0.987 |
| 5 | active | head | 9 | 1.06 [0.92, 1.19] | 0.95 | -0.10 | 1.000 | 1.000 | 1.000 |
| 5 | active | tail | 28 | 0.81 [0.67, 0.93] | 0.81 | 0.00 | 0.993 | 0.994 | 0.974 |
| 5 | active | xtail | 27 | 0.64 [0.55, 0.73] | 0.70 | 0.06 | 0.985 | 0.986 | 0.995 |

## By band (primary) and separate groups (descriptive)

D on bad bases, O / " by" / "." (point estimates; CIs in `transfer_summary.csv`).

| Population | Band | Pairs | site 15 | site 12 | site 8 | site 5 |
|---|---|---:|---|---|---|
| primary (plain) | head | 9 | +2.57 / +1.43 / -0.84 | +1.53 / +1.80 / -0.52 | +1.13 / +1.40 / -0.12 | +1.01 / +1.07 / +0.13 |
| primary (plain) | tail | 28 | +2.87 / +0.94 / -0.63 | +1.68 / +0.93 / -0.57 | +1.14 / +0.65 / -0.17 | +0.75 / +0.37 / -0.01 |
| primary (plain) | xtail | 27 | +3.20 / +0.72 / -0.69 | +1.85 / +0.79 / -0.60 | +1.19 / +0.56 / -0.12 | +0.92 / +0.49 / -0.02 |
| primary, original pairs only | all | 59 | +2.98 / +0.94 / -0.67 | +1.72 / +1.00 / -0.58 | +1.16 / +0.71 / -0.14 | +0.83 / +0.52 / +0.01 |
| primary, without bet/appear | all | 63 | +2.96 / +0.91 / -0.68 | +1.73 / +0.98 / -0.58 | +1.15 / +0.71 / -0.14 | +0.85 / +0.51 / -0.00 |
| primary, cross-fitted orig_head pairs | all | 8 | +2.51 / +1.45 / -0.79 | +1.48 / +1.81 / -0.52 | +1.10 / +1.41 / -0.12 | +0.98 / +1.08 / +0.09 |
| primary, non-training pairs | all | 56 | +3.03 / +0.84 / -0.67 | +1.76 / +0.88 / -0.58 | +1.17 / +0.62 / -0.15 | +0.84 / +0.44 / -0.01 |
| primary + contaminated_bad | all | 71 | +3.00 / +0.95 / -0.66 | +1.75 / +1.02 / -0.55 | +1.18 / +0.74 / -0.12 | +0.89 / +0.55 / +0.03 |
| primary, by reliable | all | 41 | +3.04 / +0.97 / -0.65 | +1.79 / +1.06 / -0.57 | +1.21 / +0.76 / -0.12 | +0.84 / +0.57 / +0.02 |
| primary, by negative | all | 15 | +2.78 / +0.88 / -0.80 | +1.59 / +0.89 / -0.64 | +1.06 / +0.61 / -0.20 | +0.82 / +0.40 / -0.01 |
| new Head eval pairs (prep_object) | all | 6 | +3.40 / +2.19 / +0.62 | +2.37 / +1.72 / +0.09 | +1.64 / +1.19 / +0.35 | +1.11 / +0.85 / +0.44 |
| prep_object, pseudo_passive_ok | all | 56 | +3.27 / +1.52 / +0.34 | +2.37 / +1.29 / -0.03 | +1.70 / +0.95 / +0.33 | +1.09 / +0.79 / +0.53 |
| prep_object, pseudo_passive_bad | all | 18 | +3.31 / +1.28 / -0.45 | +2.11 / +1.11 / -0.40 | +1.47 / +0.76 / -0.05 | +1.12 / +0.51 / +0.18 |
| contaminated_bad | all | 7 | +3.29 / +1.30 / -0.41 | +1.94 / +1.27 / -0.29 | +1.41 / +0.97 / +0.09 | +1.16 / +0.89 / +0.25 |
| bad-side-high by | all | 11 | +2.93 / +0.82 / -0.75 | +1.81 / +0.71 / -0.69 | +1.25 / +0.50 / -0.23 | +0.86 / +0.20 / +0.02 |
| original, by reliable | all | 96 | +3.19 / +1.28 / -0.16 | +2.07 / +1.20 / -0.27 | +1.46 / +0.88 / +0.13 | +0.99 / +0.72 / +0.32 |
| original, by negative | all | 27 | +2.94 / +1.00 / -0.66 | +1.83 / +0.94 / -0.61 | +1.28 / +0.67 / -0.17 | +0.88 / +0.39 / +0.05 |
| original, by n.s. positive | all | 3 | +3.08 / +0.80 / -0.37 | +1.47 / +0.70 / -0.35 | +0.93 / +0.54 / -0.10 | +0.76 / +0.43 / -0.01 |
| all pairs | all | 145 | +3.14 / +1.21 / -0.25 | +2.03 / +1.13 / -0.33 | +1.42 / +0.83 / +0.06 | +1.00 / +0.64 / +0.24 |

