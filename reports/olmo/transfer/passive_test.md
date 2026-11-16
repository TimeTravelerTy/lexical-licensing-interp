# Passive test of the round-2 direction: results

Spec, declared and committed before any passive evaluation: `passive_test_plan.md`. Run: `run_passive_test.py` (OLMo-2-1B, jobs/o_tr_main.sh); analysis: `analyze_passive_test.py` (2,000 draws, seed 17; three-way cluster bootstrap over base verb pairs within band, contexts and donor verb pairs). Tables: `results/das_round2/passive_test/*_summary.csv`.

Fidelity checks (per site): a self-patch reproduces the unpatched readout exactly (max |dM| = 0); recomputing 512 held-out active swaps of each site's own run gives median |dM| 0.000, 0.000, 0.000, 0.000 and max 0.03, 0.04, 0.04, 0.04 (sites 13, 12, 11, 9; bf16 batch effects, M is on a scale of about +-5).

## Verdict (primary population: 64 pairs with a plain bad verb, all bands pooled)

D = mean change after a transitive active donor minus after an intransitive active donor, on bad passive bases ("The house was emerged"), in nats.

| Site | Role | δ_s | D log P(O) | class | D log P(" by") | class | D log P(".") | class | Outcome |
|---:|---|---:|---|---|---|---|---|---|---|
| 13 | primary | 0.62 | 3.67 [3.48, 3.87] | RISE | -1.09 [-1.26, -0.94] | FALL | -2.09 [-2.31, -1.90] | FALL | **surface (object next)** |
| 12 | secondary | 0.62 | 3.44 [3.25, 3.64] | RISE | -0.74 [-0.88, -0.62] | FALL | -1.62 [-1.79, -1.46] | FALL | **surface (object next)** |
| 11 | secondary | 0.58 | 2.40 [2.23, 2.58] | RISE | 0.14 [0.08, 0.21] | NO RISE | -1.03 [-1.16, -0.92] | FALL | **surface (object next)** |
| 9 | secondary | 0.53 | 1.71 [1.58, 1.85] | RISE | 0.43 [0.37, 0.49] | NO RISE | -0.54 [-0.61, -0.47] | FALL | **surface (object next)** |

Rules (declared): RISE = 95% CI above 0 and estimate >= δ_s; NO RISE = 90% CI within ±δ_s; outcome from the O and " by" classes.

### In probabilities

Mean probability over primary bad passive bases (and, for reference, good passives unpatched).

| Condition | P(O) | P(pronouns) | P(" the") | P(" by") | P(".") | P(I) |
|---|---:|---:|---:|---:|---:|---:|
| bad passive, unpatched | 0.015 | 0.0004 | 0.005 | 0.093 | 0.013 | 0.666 |
| site 9, T donor | 0.077 | 0.0050 | 0.031 | 0.134 | 0.009 | 0.560 |
| site 9, I donor | 0.015 | 0.0004 | 0.004 | 0.092 | 0.013 | 0.668 |
| site 11, T donor | 0.150 | 0.0217 | 0.059 | 0.103 | 0.006 | 0.504 |
| site 11, I donor | 0.014 | 0.0004 | 0.004 | 0.091 | 0.013 | 0.671 |
| site 12, T donor | 0.380 | 0.1867 | 0.101 | 0.053 | 0.004 | 0.332 |
| site 12, I donor | 0.015 | 0.0005 | 0.005 | 0.093 | 0.013 | 0.667 |
| site 13, T donor | 0.490 | 0.1771 | 0.185 | 0.038 | 0.003 | 0.219 |
| site 13, I donor | 0.017 | 0.0008 | 0.005 | 0.094 | 0.013 | 0.663 |
| good passive, unpatched | 0.016 | 0.0004 | 0.006 | 0.178 | 0.015 | 0.637 |

### Where the change goes (D on bad bases, primary)

| Readout | site 13 | site 12 | site 11 | site 9 |
|---|---|---|---|---|
| log P(O) | 3.67 [3.48, 3.87] | 3.44 [3.25, 3.64] | 2.40 [2.23, 2.58] | 1.71 [1.58, 1.85] |
| log P(det) | 3.25 [3.07, 3.43] | 2.81 [2.65, 2.97] | 2.24 [2.08, 2.41] | 1.65 [1.53, 1.79] |
| log P(pron) | 6.07 [5.76, 6.36] | 6.17 [5.89, 6.45] | 3.88 [3.62, 4.14] | 2.36 [2.20, 2.53] |
| log P(refl) | 4.19 [3.96, 4.41] | 3.42 [3.20, 3.65] | 3.15 [2.88, 3.44] | 2.28 [2.08, 2.47] |
| log P(" the") | 4.04 [3.81, 4.26] | 3.49 [3.30, 3.69] | 2.82 [2.61, 3.02] | 2.15 [1.99, 2.30] |
| log P(" him") | 6.09 [5.73, 6.42] | 6.23 [5.90, 6.55] | 4.73 [4.42, 5.03] | 3.14 [2.92, 3.36] |
| log P(" by") | -1.09 [-1.26, -0.94] | -0.74 [-0.88, -0.62] | 0.14 [0.08, 0.21] | 0.43 [0.37, 0.49] |
| log P(".") | -2.09 [-2.31, -1.90] | -1.62 [-1.79, -1.46] | -1.03 [-1.16, -0.92] | -0.54 [-0.61, -0.47] |
| log P(I) | -1.28 [-1.44, -1.12] | -0.79 [-0.90, -0.68] | -0.31 [-0.36, -0.26] | -0.19 [-0.22, -0.16] |
| M | 4.95 [4.65, 5.26] | 4.23 [3.96, 4.50] | 2.71 [2.51, 2.93] | 1.89 [1.75, 2.06] |

D(" by") as a fraction of the natural good − bad gap in log P(" by") on the same items (0.48 nats): site 13 -2.29, site 12 -1.55, site 11 0.30, site 9 0.90. D(O) as a fraction of the site's active effect: site 13 1.18, site 12 1.11, site 11 0.83, site 9 0.64.

## Controls (primary, Δ from unpatched)

| Condition | site 13: O / by / . | site 12: O / by / . | site 11: O / by / . | site 9: O / by / . |
|---|---|---|---|
| T donor → bad passive | +3.74 / -1.09 / -2.10 | +3.43 / -0.74 / -1.59 | +2.34 / +0.11 / -0.98 | +1.69 / +0.41 / -0.53 |
| I donor → bad passive (voice-change baseline) | +0.07 / +0.00 / -0.00 | -0.01 / -0.00 / +0.02 | -0.06 / -0.03 / +0.05 | -0.02 / -0.02 / +0.01 |
| own active → bad passive (same verb) | +0.09 / -0.02 / -0.02 | -0.00 / -0.02 / +0.02 | -0.06 / -0.04 / +0.04 | +0.01 / -0.02 / +0.01 |
| good passive → bad passive | +0.42 / +0.02 / -0.16 | +0.46 / +0.03 / -0.18 | +0.38 / +0.10 / -0.16 | +0.30 / +0.14 / -0.08 |
| T donor → good passive (good → good) | +3.55 / -0.62 / -1.37 | +3.19 / -0.46 / -1.03 | +1.96 / +0.02 / -0.66 | +1.50 / +0.12 / -0.41 |
| I donor → good passive | -0.24 / +0.01 / +0.16 | -0.31 / -0.00 / +0.16 | -0.25 / -0.08 / +0.22 | -0.21 / -0.09 / +0.12 |
| own active → good passive (same verb) | +3.19 / -0.40 / -1.10 | +2.70 / -0.25 / -0.76 | +1.61 / +0.04 / -0.54 | +1.24 / +0.10 / -0.33 |
| bad passive → good passive | -0.30 / +0.02 / +0.17 | -0.31 / +0.01 / +0.15 | -0.22 / -0.05 / +0.18 | -0.20 / -0.07 / +0.11 |
| D on good bases | +3.79 / -0.63 / -1.53 | +3.50 / -0.46 / -1.19 | +2.21 / +0.09 / -0.87 | +1.71 / +0.21 / -0.52 |
| D, participle-matched donors | +3.55 / -1.06 / -2.02 | +3.28 / -0.69 / -1.53 | +2.27 / +0.15 / -0.97 | +1.62 / +0.41 / -0.51 |
| natural good − bad (unpatched) | -0.29 / +0.48 / +0.01 | -0.29 / +0.48 / +0.01 | -0.29 / +0.48 / +0.01 | -0.29 / +0.48 / +0.01 |

## Step 1: projection of natural participles onto d

z: 0 = mean of held-out intransitive actives, 1 = held-out transitive actives (per basis, cross-fitted). Gap = good − bad (two-way bootstrap over pairs and contexts / subjects).

| Site | Frame | Band | Pairs | Gap in z | good z | bad z | AUC items | AUC verbs | Win rate |
|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| 13 | passive | all | 64 | 0.16 [0.13, 0.19] | 0.14 | -0.01 | 0.886 | 0.927 | 0.905 |
| 13 | passive | head | 9 | 0.27 [0.21, 0.33] | 0.30 | 0.03 | 0.981 | 1.000 | 0.996 |
| 13 | passive | tail | 28 | 0.15 [0.11, 0.20] | 0.13 | -0.03 | 0.894 | 0.943 | 0.888 |
| 13 | passive | xtail | 27 | 0.13 [0.10, 0.17] | 0.11 | -0.02 | 0.873 | 0.912 | 0.891 |
| 13 | active | all | 64 | 0.91 [0.84, 0.96] | 0.91 | 0.01 | 0.999 | 1.000 | 1.000 |
| 13 | active | head | 9 | 1.01 [0.91, 1.10] | 0.95 | -0.06 | 1.000 | 1.000 | 1.000 |
| 13 | active | tail | 28 | 0.97 [0.88, 1.06] | 0.93 | -0.04 | 1.000 | 1.000 | 1.000 |
| 13 | active | xtail | 27 | 0.80 [0.71, 0.89] | 0.88 | 0.08 | 0.998 | 1.000 | 1.000 |
| 12 | passive | all | 64 | 0.20 [0.17, 0.22] | 0.23 | 0.03 | 0.944 | 0.969 | 0.943 |
| 12 | passive | head | 9 | 0.35 [0.30, 0.40] | 0.36 | 0.01 | 0.999 | 1.000 | 1.000 |
| 12 | passive | tail | 28 | 0.19 [0.15, 0.24] | 0.21 | 0.02 | 0.949 | 0.980 | 0.927 |
| 12 | passive | xtail | 27 | 0.15 [0.12, 0.19] | 0.20 | 0.05 | 0.920 | 0.955 | 0.940 |
| 12 | active | all | 64 | 0.89 [0.83, 0.95] | 0.89 | 0.00 | 0.999 | 1.000 | 1.000 |
| 12 | active | head | 9 | 1.02 [0.91, 1.11] | 0.97 | -0.05 | 1.000 | 1.000 | 1.000 |
| 12 | active | tail | 28 | 0.96 [0.87, 1.05] | 0.91 | -0.05 | 1.000 | 1.000 | 1.000 |
| 12 | active | xtail | 27 | 0.78 [0.70, 0.87] | 0.85 | 0.07 | 0.999 | 1.000 | 1.000 |
| 11 | passive | all | 64 | 0.21 [0.19, 0.24] | 0.28 | 0.07 | 0.961 | 0.977 | 0.963 |
| 11 | passive | head | 9 | 0.35 [0.31, 0.39] | 0.40 | 0.05 | 0.999 | 1.000 | 1.000 |
| 11 | passive | tail | 28 | 0.21 [0.17, 0.26] | 0.27 | 0.05 | 0.972 | 0.989 | 0.957 |
| 11 | passive | xtail | 27 | 0.17 [0.13, 0.20] | 0.26 | 0.09 | 0.934 | 0.959 | 0.957 |
| 11 | active | all | 64 | 0.89 [0.83, 0.94] | 0.90 | 0.01 | 0.999 | 1.000 | 1.000 |
| 11 | active | head | 9 | 0.99 [0.88, 1.08] | 0.96 | -0.03 | 1.000 | 1.000 | 1.000 |
| 11 | active | tail | 28 | 0.95 [0.85, 1.04] | 0.91 | -0.04 | 1.000 | 1.000 | 1.000 |
| 11 | active | xtail | 27 | 0.79 [0.71, 0.87] | 0.85 | 0.06 | 0.999 | 1.000 | 1.000 |
| 9 | passive | all | 64 | 0.23 [0.21, 0.26] | 0.27 | 0.03 | 0.948 | 0.966 | 0.953 |
| 9 | passive | head | 9 | 0.32 [0.27, 0.38] | 0.37 | 0.05 | 0.992 | 1.000 | 1.000 |
| 9 | passive | tail | 28 | 0.23 [0.18, 0.28] | 0.24 | 0.01 | 0.954 | 0.973 | 0.931 |
| 9 | passive | xtail | 27 | 0.21 [0.17, 0.25] | 0.26 | 0.05 | 0.937 | 0.962 | 0.960 |
| 9 | active | all | 64 | 0.87 [0.81, 0.93] | 0.89 | 0.02 | 0.999 | 0.999 | 1.000 |
| 9 | active | head | 9 | 0.97 [0.86, 1.07] | 0.94 | -0.03 | 1.000 | 1.000 | 1.000 |
| 9 | active | tail | 28 | 0.94 [0.84, 1.03] | 0.91 | -0.04 | 1.000 | 1.000 | 1.000 |
| 9 | active | xtail | 27 | 0.77 [0.68, 0.85] | 0.85 | 0.09 | 1.000 | 1.000 | 1.000 |

## By band (primary) and separate groups (descriptive)

D on bad bases, O / " by" / "." (point estimates; CIs in `transfer_summary.csv`).

| Population | Band | Pairs | site 13 | site 12 | site 11 | site 9 |
|---|---|---:|---|---|---|
| primary (plain) | head | 9 | +2.97 / -0.82 / -2.19 | +2.48 / -0.20 / -1.62 | +2.03 / +0.52 / -1.38 | +1.48 / +0.87 / -0.79 |
| primary (plain) | tail | 28 | +3.67 / -1.05 / -2.03 | +3.41 / -0.67 / -1.51 | +2.47 / +0.12 / -1.01 | +1.73 / +0.40 / -0.51 |
| primary (plain) | xtail | 27 | +3.91 / -1.23 / -2.13 | +3.80 / -0.98 / -1.73 | +2.45 / +0.05 / -0.94 | +1.77 / +0.32 / -0.48 |
| primary, original pairs only | all | 59 | +3.70 / -1.10 / -2.10 | +3.46 / -0.74 / -1.63 | +2.43 / +0.15 / -1.06 | +1.72 / +0.44 / -0.56 |
| primary, without bet/appear | all | 63 | +3.68 / -1.10 / -2.09 | +3.45 / -0.75 / -1.62 | +2.40 / +0.13 / -1.03 | +1.71 / +0.42 / -0.54 |
| primary, cross-fitted orig_head pairs | all | 8 | +2.93 / -0.85 / -2.18 | +2.44 / -0.22 / -1.63 | +2.01 / +0.49 / -1.37 | +1.46 / +0.85 / -0.82 |
| primary, non-training pairs | all | 56 | +3.78 / -1.13 / -2.08 | +3.58 / -0.81 / -1.62 | +2.45 / +0.09 / -0.99 | +1.74 / +0.37 / -0.50 |
| primary + contaminated_bad | all | 71 | +3.70 / -1.09 / -2.08 | +3.47 / -0.75 / -1.61 | +2.39 / +0.14 / -1.01 | +1.71 / +0.43 / -0.52 |
| primary, by reliable | all | 41 | +3.66 / -1.13 / -2.14 | +3.43 / -0.77 / -1.65 | +2.42 / +0.15 / -1.05 | +1.71 / +0.47 / -0.55 |
| primary, by negative | all | 15 | +3.72 / -0.99 / -2.05 | +3.42 / -0.58 / -1.53 | +2.44 / +0.16 / -1.11 | +1.72 / +0.37 / -0.57 |
| new Head eval pairs (prep_object) | all | 6 | +4.39 / -1.06 / -2.16 | +3.62 / -0.83 / -1.33 | +2.26 / +0.81 / -0.40 | +2.36 / +1.33 / +0.23 |
| prep_object, pseudo_passive_ok | all | 56 | +4.31 / -0.84 / -1.69 | +3.72 / -0.69 / -1.16 | +2.51 / +0.58 / -0.40 | +2.49 / +1.06 / +0.15 |
| prep_object, pseudo_passive_bad | all | 18 | +4.12 / -0.74 / -1.86 | +3.68 / -0.40 / -1.33 | +2.56 / +0.38 / -0.92 | +1.97 / +0.63 / -0.54 |
| contaminated_bad | all | 7 | +3.91 / -1.09 / -1.92 | +3.72 / -0.82 / -1.51 | +2.35 / +0.12 / -0.82 | +1.76 / +0.38 / -0.36 |
| bad-side-high by | all | 11 | +3.99 / -0.89 / -1.83 | +3.59 / -0.50 / -1.31 | +2.54 / +0.23 / -0.88 | +1.94 / +0.40 / -0.43 |
| original, by reliable | all | 96 | +3.99 / -0.96 / -1.92 | +3.60 / -0.71 / -1.43 | +2.47 / +0.35 / -0.75 | +2.06 / +0.74 / -0.25 |
| original, by negative | all | 27 | +3.88 / -0.93 / -1.90 | +3.50 / -0.56 / -1.37 | +2.52 / +0.24 / -0.95 | +1.92 / +0.48 / -0.46 |
| original, by n.s. positive | all | 3 | +4.16 / -1.20 / -1.93 | +4.09 / -1.02 / -1.73 | +2.47 / +0.07 / -0.96 | +1.84 / +0.37 / -0.57 |
| all pairs | all | 145 | +3.98 / -0.95 / -1.90 | +3.59 / -0.68 / -1.40 | +2.46 / +0.34 / -0.76 | +2.04 / +0.70 / -0.26 |

