# Passive test of the round-2 direction: results

Spec, declared and committed before any passive evaluation: `passive_test_plan.md`. Run: `run_passive_test.py` (TSUBAME job 8916904, commit `f769ec9`); analysis: `analyze_passive_test.py` (2,000 draws, seed 17; three-way cluster bootstrap over base verb pairs within band, contexts and donor verb pairs). Tables: `results/das_round2/passive_test/*_summary.csv`.

Fidelity checks (per site): a self-patch reproduces the unpatched readout exactly (max |dM| = 0); recomputing 512 held-out active swaps of each site's own run gives median |dM| 0.004, 0.001, 0.000 and max 0.43, 0.36, 0.29 (sites 17, 12, 8; bf16 batch effects, M is on a scale of about +-5).

## Verdict (primary population: 64 pairs with a plain bad verb, all bands pooled)

D = mean change after a transitive active donor minus after an intransitive active donor, on bad passive bases ("The house was emerged"), in nats.

| Site | Role | δ_s | D log P(O) | class | D log P(" by") | class | D log P(".") | class | Outcome |
|---:|---|---:|---|---|---|---|---|---|---|
| 17 | primary | 0.57 | 3.58 [3.36, 3.79] | RISE | -0.93 [-1.10, -0.77] | FALL | -1.99 [-2.23, -1.77] | FALL | **surface (object next)** |
| 12 | secondary | 0.53 | 1.17 [1.01, 1.33] | RISE | 0.81 [0.72, 0.89] | RISE | -0.54 [-0.60, -0.48] | FALL | **mixed** |
| 8 | secondary | 0.49 | 0.36 [0.30, 0.44] | NO RISE | 0.72 [0.63, 0.82] | RISE | -0.01 [-0.05, 0.05] | NO RISE | **abstract (takes an object)** |

Rules (declared): RISE = 95% CI above 0 and estimate >= δ_s; NO RISE = 90% CI within ±δ_s; outcome from the O and " by" classes.

### In probabilities

Mean probability over primary bad passive bases (and, for reference, good passives unpatched).

| Condition | P(O) | P(pronouns) | P(" the") | P(" by") | P(".") | P(I) |
|---|---:|---:|---:|---:|---:|---:|
| bad passive, unpatched | 0.016 | 0.0004 | 0.004 | 0.069 | 0.042 | 0.697 |
| site 8, T donor | 0.023 | 0.0012 | 0.006 | 0.117 | 0.042 | 0.656 |
| site 8, I donor | 0.016 | 0.0004 | 0.003 | 0.065 | 0.043 | 0.702 |
| site 12, T donor | 0.051 | 0.0049 | 0.016 | 0.123 | 0.030 | 0.603 |
| site 12, I donor | 0.015 | 0.0004 | 0.003 | 0.061 | 0.045 | 0.710 |
| site 17, T donor | 0.470 | 0.0510 | 0.194 | 0.034 | 0.010 | 0.231 |
| site 17, I donor | 0.018 | 0.0005 | 0.004 | 0.068 | 0.043 | 0.698 |
| good passive, unpatched | 0.018 | 0.0006 | 0.006 | 0.158 | 0.043 | 0.663 |

### Where the change goes (D on bad bases, primary)

| Readout | site 17 | site 12 | site 8 |
|---|---|---|---|
| log P(O) | 3.58 [3.36, 3.79] | 1.17 [1.01, 1.33] | 0.36 [0.30, 0.44] |
| log P(det) | 3.49 [3.26, 3.69] | 1.09 [0.94, 1.24] | 0.33 [0.27, 0.39] |
| log P(pron) | 5.11 [4.78, 5.42] | 2.32 [2.08, 2.57] | 0.88 [0.77, 0.99] |
| log P(refl) | 4.20 [3.90, 4.47] | 2.32 [2.10, 2.54] | 1.27 [1.14, 1.41] |
| log P(" the") | 4.37 [4.09, 4.64] | 1.59 [1.42, 1.78] | 0.55 [0.48, 0.63] |
| log P(" him") | 5.92 [5.50, 6.30] | 2.88 [2.59, 3.19] | 1.40 [1.25, 1.56] |
| log P(" by") | -0.93 [-1.10, -0.77] | 0.81 [0.72, 0.89] | 0.72 [0.63, 0.82] |
| log P(".") | -1.99 [-2.23, -1.77] | -0.54 [-0.60, -0.48] | -0.01 [-0.05, 0.05] |
| log P(I) | -1.28 [-1.45, -1.12] | -0.18 [-0.20, -0.15] | -0.07 [-0.09, -0.05] |
| M | 4.87 [4.52, 5.18] | 1.34 [1.17, 1.52] | 0.43 [0.36, 0.51] |

D(" by") as a fraction of the natural good − bad gap in log P(" by") on the same items (0.71 nats): site 17 -1.30, site 12 1.13, site 8 1.01. D(O) as a fraction of the site's active effect: site 17 1.25, site 12 0.44, site 8 0.15.

## Controls (primary, Δ from unpatched)

| Condition | site 17: O / by / . | site 12: O / by / . | site 8: O / by / . |
|---|---|---|---|
| T donor → bad passive | +3.63 / -0.92 / -1.98 | +1.11 / +0.69 / -0.46 | +0.36 / +0.65 / -0.01 |
| I donor → bad passive (voice-change baseline) | +0.05 / +0.01 / +0.01 | -0.06 / -0.12 / +0.08 | -0.01 / -0.07 / -0.01 |
| own active → bad passive (same verb) | +0.04 / -0.01 / -0.01 | -0.07 / -0.15 / +0.07 | -0.00 / -0.10 / -0.02 |
| good passive → bad passive | +0.48 / +0.05 / -0.13 | +0.29 / +0.34 / -0.17 | +0.09 / +0.36 / +0.01 |
| T donor → good passive (good → good) | +3.54 / -0.65 / -1.49 | +0.78 / +0.21 / -0.29 | +0.40 / +0.13 / -0.07 |
| I donor → good passive | -0.19 / -0.02 / +0.16 | -0.32 / -0.24 / +0.27 | -0.19 / -0.15 / +0.06 |
| own active → good passive (same verb) | +3.21 / -0.48 / -1.23 | +0.57 / +0.16 / -0.23 | +0.29 / +0.09 / -0.06 |
| bad passive → good passive | -0.24 / -0.01 / +0.16 | -0.24 / -0.16 / +0.20 | -0.17 / -0.11 / +0.07 |
| D on good bases | +3.73 / -0.63 / -1.66 | +1.10 / +0.45 / -0.56 | +0.60 / +0.28 / -0.13 |
| D, participle-matched donors | +3.39 / -0.85 / -1.86 | +1.03 / +0.75 / -0.49 | +0.32 / +0.67 / -0.00 |
| natural good − bad (unpatched) | -0.22 / +0.71 / -0.02 | -0.22 / +0.71 / -0.02 | -0.22 / +0.71 / -0.02 |

## Step 1: projection of natural participles onto d

z: 0 = mean of held-out intransitive actives, 1 = held-out transitive actives (per basis, cross-fitted). Gap = good − bad (two-way bootstrap over pairs and contexts / subjects).

| Site | Frame | Band | Pairs | Gap in z | good z | bad z | AUC items | AUC verbs | Win rate |
|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| 17 | passive | all | 64 | 0.18 [0.16, 0.21] | 0.20 | 0.01 | 0.915 | 0.963 | 0.941 |
| 17 | passive | head | 9 | 0.28 [0.22, 0.35] | 0.28 | -0.00 | 0.975 | 1.000 | 0.996 |
| 17 | passive | tail | 28 | 0.20 [0.16, 0.25] | 0.18 | -0.02 | 0.942 | 0.986 | 0.946 |
| 17 | passive | xtail | 27 | 0.13 [0.10, 0.16] | 0.18 | 0.05 | 0.867 | 0.930 | 0.918 |
| 17 | active | all | 64 | 0.92 [0.85, 0.98] | 0.92 | 0.00 | 0.998 | 0.999 | 1.000 |
| 17 | active | head | 9 | 0.95 [0.81, 1.06] | 0.92 | -0.03 | 1.000 | 1.000 | 1.000 |
| 17 | active | tail | 28 | 1.00 [0.89, 1.10] | 0.95 | -0.05 | 1.000 | 1.000 | 1.000 |
| 17 | active | xtail | 27 | 0.82 [0.72, 0.92] | 0.89 | 0.07 | 0.995 | 0.996 | 1.000 |
| 12 | passive | all | 64 | 0.34 [0.31, 0.37] | 0.50 | 0.16 | 0.989 | 0.995 | 0.999 |
| 12 | passive | head | 9 | 0.50 [0.45, 0.57] | 0.57 | 0.07 | 1.000 | 1.000 | 1.000 |
| 12 | passive | tail | 28 | 0.36 [0.31, 0.41] | 0.50 | 0.14 | 0.999 | 1.000 | 0.999 |
| 12 | passive | xtail | 27 | 0.28 [0.24, 0.31] | 0.48 | 0.21 | 0.971 | 0.984 | 0.999 |
| 12 | active | all | 64 | 0.90 [0.84, 0.96] | 0.89 | -0.01 | 0.999 | 0.999 | 1.000 |
| 12 | active | head | 9 | 0.97 [0.86, 1.07] | 0.95 | -0.02 | 1.000 | 1.000 | 1.000 |
| 12 | active | tail | 28 | 0.96 [0.86, 1.04] | 0.91 | -0.05 | 1.000 | 1.000 | 1.000 |
| 12 | active | xtail | 27 | 0.81 [0.73, 0.90] | 0.85 | 0.04 | 0.994 | 0.996 | 1.000 |
| 8 | passive | all | 64 | 0.36 [0.33, 0.40] | 0.45 | 0.09 | 0.985 | 0.992 | 0.996 |
| 8 | passive | head | 9 | 0.50 [0.45, 0.56] | 0.50 | 0.01 | 1.000 | 1.000 | 1.000 |
| 8 | passive | tail | 28 | 0.38 [0.32, 0.45] | 0.45 | 0.07 | 0.995 | 1.000 | 0.999 |
| 8 | passive | xtail | 27 | 0.30 [0.25, 0.35] | 0.43 | 0.13 | 0.967 | 0.982 | 0.991 |
| 8 | active | all | 64 | 0.87 [0.80, 0.93] | 0.87 | 0.00 | 0.998 | 0.999 | 1.000 |
| 8 | active | head | 9 | 0.96 [0.83, 1.08] | 0.91 | -0.05 | 1.000 | 1.000 | 1.000 |
| 8 | active | tail | 28 | 0.92 [0.82, 1.02] | 0.90 | -0.03 | 1.000 | 1.000 | 1.000 |
| 8 | active | xtail | 27 | 0.78 [0.69, 0.86] | 0.83 | 0.05 | 0.995 | 0.996 | 1.000 |

## By band (primary) and separate groups (descriptive)

D on bad bases, O / " by" / "." (point estimates; CIs in `transfer_summary.csv`).

| Population | Band | Pairs | site 17 | site 12 | site 8 |
|---|---|---:|---|---|---|
| primary (plain) | head | 9 | +2.97 / -0.90 / -2.07 | +1.20 / +1.05 / -0.69 | +0.40 / +1.08 / -0.19 |
| primary (plain) | tail | 28 | +3.56 / -0.94 / -1.98 | +1.12 / +0.79 / -0.52 | +0.33 / +0.68 / -0.03 |
| primary (plain) | xtail | 27 | +3.81 / -0.92 / -1.98 | +1.21 / +0.74 / -0.51 | +0.38 / +0.64 / +0.08 |
| primary, original pairs only | all | 59 | +3.61 / -0.93 / -1.99 | +1.17 / +0.79 / -0.54 | +0.37 / +0.71 / -0.01 |
| primary, without bet/appear | all | 63 | +3.59 / -0.93 / -1.99 | +1.17 / +0.80 / -0.54 | +0.36 / +0.71 / -0.00 |
| primary, cross-fitted orig_head pairs | all | 8 | +2.94 / -0.93 / -2.08 | +1.20 / +1.04 / -0.69 | +0.39 / +1.08 / -0.19 |
| primary, non-training pairs | all | 56 | +3.67 / -0.93 / -1.98 | +1.16 / +0.77 / -0.52 | +0.36 / +0.67 / +0.02 |
| primary + contaminated_bad | all | 71 | +3.62 / -0.94 / -1.99 | +1.15 / +0.80 / -0.53 | +0.37 / +0.71 / +0.01 |
| primary, by reliable | all | 41 | +3.61 / -0.97 / -2.08 | +1.20 / +0.85 / -0.52 | +0.37 / +0.78 / +0.02 |
| primary, by negative | all | 15 | +3.55 / -0.86 / -1.81 | +1.12 / +0.68 / -0.58 | +0.34 / +0.58 / -0.10 |
| new Head eval pairs (prep_object) | all | 6 | +4.52 / -0.73 / -2.39 | +1.40 / +1.17 / -0.22 | +0.91 / +1.34 / +0.58 |
| prep_object, pseudo_passive_ok | all | 56 | +4.31 / -0.70 / -1.84 | +1.47 / +1.03 / -0.17 | +1.01 / +1.41 / +0.72 |
| prep_object, pseudo_passive_bad | all | 18 | +3.91 / -0.59 / -1.81 | +1.29 / +0.92 / -0.47 | +0.59 / +0.95 / +0.09 |
| contaminated_bad | all | 7 | +3.91 / -0.99 / -1.95 | +1.01 / +0.71 / -0.46 | +0.44 / +0.61 / +0.12 |
| bad-side-high by | all | 11 | +3.78 / -0.80 / -1.85 | +1.07 / +0.63 / -0.52 | +0.40 / +0.57 / -0.13 |
| original, by reliable | all | 96 | +3.95 / -0.82 / -1.91 | +1.31 / +0.93 / -0.35 | +0.68 / +1.08 / +0.38 |
| original, by negative | all | 27 | +3.76 / -0.82 / -1.86 | +1.16 / +0.75 / -0.55 | +0.43 / +0.74 / -0.04 |
| original, by n.s. positive | all | 3 | +3.86 / -0.66 / -1.68 | +1.10 / +0.58 / -0.47 | +0.41 / +0.45 / -0.02 |
| all pairs | all | 145 | +3.92 / -0.80 / -1.91 | +1.29 / +0.90 / -0.38 | +0.65 / +1.01 / +0.29 |

