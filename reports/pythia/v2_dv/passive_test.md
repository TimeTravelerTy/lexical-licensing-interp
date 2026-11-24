# Passive test of the round-2 direction: results

Spec, declared and committed before any passive evaluation: `passive_test_plan.md`. Run: `run_passive_test.py` (Pythia-1.4B v2, jobs/v2_chain.sh, job 9554765); analysis: `analyze_passive_test.py` (2,000 draws, seed 17; three-way cluster bootstrap over base verb pairs within band, contexts and donor verb pairs). Tables: `results/das_round2/passive_test/*_summary.csv`.

Fidelity checks (per site): a self-patch reproduces the unpatched readout exactly (max |dM| = 0); recomputing 512 held-out active swaps of each site's own run gives median |dM| 0.004, 0.001, 0.000 and max 0.44, 0.38, 0.25 (sites 17, 12, 8; bf16 batch effects, M is on a scale of about +-5).

## Verdict (primary population: 64 pairs with a plain bad verb, all bands pooled)

D = mean change after a transitive active donor minus after an intransitive active donor, on bad passive bases ("The house was emerged"), in nats.

| Site | Role | δ_s | D log P(O) | class | D log P(" by") | class | D log P(".") | class | Outcome |
|---:|---|---:|---|---|---|---|---|---|---|
| 17 | primary | 0.57 | 3.75 [3.51, 3.97] | RISE | -1.07 [-1.27, -0.88] | FALL | -2.11 [-2.36, -1.85] | FALL | **surface (object next)** |
| 12 | secondary | 0.53 | 0.94 [0.80, 1.08] | RISE | 0.83 [0.74, 0.92] | RISE | -0.37 [-0.42, -0.32] | NO RISE | **mixed** |
| 8 | secondary | 0.49 | 0.25 [0.19, 0.32] | NO RISE | 0.78 [0.68, 0.88] | RISE | 0.06 [0.01, 0.12] | NO RISE | **abstract (takes an object)** |

Rules (declared): RISE = 95% CI above 0 and estimate >= δ_s; NO RISE = 90% CI within ±δ_s; outcome from the O and " by" classes.

### In probabilities

Mean probability over primary bad passive bases (and, for reference, good passives unpatched).

| Condition | P(O) | P(pronouns) | P(" the") | P(" by") | P(".") | P(I) |
|---|---:|---:|---:|---:|---:|---:|
| bad passive, unpatched | 0.016 | 0.0004 | 0.004 | 0.069 | 0.042 | 0.697 |
| site 8, T donor | 0.020 | 0.0011 | 0.005 | 0.119 | 0.044 | 0.660 |
| site 8, I donor | 0.016 | 0.0004 | 0.003 | 0.062 | 0.042 | 0.705 |
| site 12, T donor | 0.040 | 0.0042 | 0.012 | 0.120 | 0.034 | 0.621 |
| site 12, I donor | 0.015 | 0.0003 | 0.003 | 0.059 | 0.045 | 0.713 |
| site 17, T donor | 0.535 | 0.0768 | 0.222 | 0.031 | 0.009 | 0.189 |
| site 17, I donor | 0.017 | 0.0005 | 0.004 | 0.068 | 0.043 | 0.701 |
| good passive, unpatched | 0.018 | 0.0006 | 0.006 | 0.158 | 0.043 | 0.663 |

### Where the change goes (D on bad bases, primary)

| Readout | site 17 | site 12 | site 8 |
|---|---|---|---|
| log P(O) | 3.75 [3.51, 3.97] | 0.94 [0.80, 1.08] | 0.25 [0.19, 0.32] |
| log P(det) | 3.61 [3.38, 3.83] | 0.85 [0.73, 0.99] | 0.21 [0.15, 0.27] |
| log P(pron) | 5.59 [5.22, 5.92] | 2.17 [1.94, 2.41] | 0.86 [0.75, 0.98] |
| log P(refl) | 4.25 [3.95, 4.52] | 2.06 [1.86, 2.27] | 1.10 [0.98, 1.24] |
| log P(" the") | 4.55 [4.24, 4.83] | 1.36 [1.20, 1.53] | 0.47 [0.41, 0.54] |
| log P(" him") | 6.43 [5.96, 6.84] | 2.85 [2.56, 3.15] | 1.41 [1.25, 1.58] |
| log P(" by") | -1.07 [-1.27, -0.88] | 0.83 [0.74, 0.92] | 0.78 [0.68, 0.88] |
| log P(".") | -2.11 [-2.36, -1.85] | -0.37 [-0.42, -0.32] | 0.06 [0.01, 0.12] |
| log P(I) | -1.56 [-1.77, -1.37] | -0.15 [-0.17, -0.12] | -0.06 [-0.08, -0.05] |
| M | 5.31 [4.92, 5.68] | 1.08 [0.93, 1.24] | 0.31 [0.25, 0.39] |

D(" by") as a fraction of the natural good − bad gap in log P(" by") on the same items (0.71 nats): site 17 -1.50, site 12 1.17, site 8 1.09. D(O) as a fraction of the site's active effect: site 17 1.31, site 12 0.36, site 8 0.10.

## Controls (primary, Δ from unpatched)

| Condition | site 17: O / by / . | site 12: O / by / . | site 8: O / by / . |
|---|---|---|---|
| T donor → bad passive | +3.77 / -1.07 / -2.09 | +0.87 / +0.66 / -0.30 | +0.24 / +0.66 / +0.05 |
| I donor → bad passive (voice-change baseline) | +0.02 / +0.00 / +0.02 | -0.07 / -0.17 / +0.07 | -0.01 / -0.12 / -0.02 |
| own active → bad passive (same verb) | +0.02 / -0.02 / +0.00 | -0.08 / -0.21 / +0.07 | +0.00 / -0.18 / -0.03 |
| good passive → bad passive | +0.51 / +0.07 / -0.10 | +0.26 / +0.35 / -0.13 | +0.06 / +0.39 / +0.04 |
| T donor → good passive (good → good) | +3.65 / -0.73 / -1.52 | +0.58 / +0.18 / -0.19 | +0.29 / +0.12 / -0.03 |
| I donor → good passive | -0.22 / -0.04 / +0.15 | -0.31 / -0.26 / +0.23 | -0.18 / -0.18 / +0.03 |
| own active → good passive (same verb) | +3.39 / -0.59 / -1.32 | +0.47 / +0.14 / -0.17 | +0.22 / +0.08 / -0.03 |
| bad passive → good passive | -0.24 / -0.03 / +0.13 | -0.23 / -0.15 / +0.15 | -0.14 / -0.12 / +0.03 |
| D on good bases | +3.87 / -0.69 / -1.67 | +0.89 / +0.44 / -0.42 | +0.47 / +0.30 / -0.06 |
| D, participle-matched donors | +3.58 / -0.96 / -1.95 | +0.82 / +0.78 / -0.34 | +0.21 / +0.73 / +0.06 |
| natural good − bad (unpatched) | -0.22 / +0.71 / -0.02 | -0.22 / +0.71 / -0.02 | -0.22 / +0.71 / -0.02 |

## Step 1: projection of natural participles onto d

z: 0 = mean of held-out intransitive actives, 1 = held-out transitive actives (per basis, cross-fitted). Gap = good − bad (two-way bootstrap over pairs and contexts / subjects).

| Site | Frame | Band | Pairs | Gap in z | good z | bad z | AUC items | AUC verbs | Win rate |
|---:|---|---|---:|---|---:|---:|---:|---:|---:|
| 17 | passive | all | 64 | 0.18 [0.16, 0.21] | 0.21 | 0.03 | 0.915 | 0.959 | 0.949 |
| 17 | passive | head | 9 | 0.24 [0.18, 0.31] | 0.30 | 0.05 | 0.958 | 0.963 | 0.998 |
| 17 | passive | tail | 28 | 0.21 [0.17, 0.25] | 0.21 | -0.00 | 0.944 | 0.986 | 0.947 |
| 17 | passive | xtail | 27 | 0.14 [0.11, 0.17] | 0.19 | 0.05 | 0.873 | 0.933 | 0.933 |
| 17 | active | all | 64 | 0.95 [0.87, 1.02] | 0.94 | -0.01 | 0.997 | 0.999 | 0.993 |
| 17 | active | head | 9 | 0.90 [0.75, 1.01] | 0.90 | 0.00 | 1.000 | 1.000 | 1.000 |
| 17 | active | tail | 28 | 1.04 [0.92, 1.14] | 0.98 | -0.06 | 0.999 | 1.000 | 0.985 |
| 17 | active | xtail | 27 | 0.88 [0.76, 0.99] | 0.92 | 0.04 | 0.991 | 0.995 | 1.000 |
| 12 | passive | all | 64 | 0.35 [0.32, 0.38] | 0.55 | 0.20 | 0.991 | 0.996 | 0.998 |
| 12 | passive | head | 9 | 0.43 [0.39, 0.48] | 0.58 | 0.15 | 1.000 | 1.000 | 1.000 |
| 12 | passive | tail | 28 | 0.37 [0.32, 0.42] | 0.56 | 0.19 | 0.996 | 1.000 | 0.996 |
| 12 | passive | xtail | 27 | 0.31 [0.27, 0.35] | 0.54 | 0.23 | 0.982 | 0.989 | 0.998 |
| 12 | active | all | 64 | 0.95 [0.88, 1.02] | 0.93 | -0.02 | 0.998 | 0.998 | 0.996 |
| 12 | active | head | 9 | 0.91 [0.78, 1.01] | 0.91 | 0.01 | 1.000 | 1.000 | 1.000 |
| 12 | active | tail | 28 | 1.02 [0.91, 1.11] | 0.96 | -0.06 | 1.000 | 1.000 | 0.990 |
| 12 | active | xtail | 27 | 0.90 [0.79, 1.01] | 0.91 | 0.01 | 0.993 | 0.993 | 1.000 |
| 8 | passive | all | 64 | 0.38 [0.35, 0.42] | 0.50 | 0.12 | 0.992 | 0.998 | 1.000 |
| 8 | passive | head | 9 | 0.43 [0.38, 0.48] | 0.51 | 0.09 | 1.000 | 1.000 | 1.000 |
| 8 | passive | tail | 28 | 0.41 [0.35, 0.47] | 0.51 | 0.11 | 0.997 | 1.000 | 0.999 |
| 8 | passive | xtail | 27 | 0.35 [0.30, 0.40] | 0.49 | 0.14 | 0.983 | 0.990 | 1.000 |
| 8 | active | all | 64 | 0.93 [0.85, 0.99] | 0.91 | -0.02 | 0.999 | 0.999 | 1.000 |
| 8 | active | head | 9 | 0.87 [0.74, 0.98] | 0.86 | -0.01 | 1.000 | 1.000 | 1.000 |
| 8 | active | tail | 28 | 0.99 [0.89, 1.09] | 0.94 | -0.05 | 1.000 | 1.000 | 1.000 |
| 8 | active | xtail | 27 | 0.87 [0.77, 0.98] | 0.88 | 0.01 | 0.996 | 0.996 | 1.000 |

## By band (primary) and separate groups (descriptive)

D on bad bases, O / " by" / "." (point estimates; CIs in `transfer_summary.csv`).

| Population | Band | Pairs | site 17 | site 12 | site 8 |
|---|---|---:|---|---|---|
| primary (plain) | head | 9 | +3.11 / -0.91 / -2.05 | +0.92 / +1.16 / -0.46 | +0.24 / +1.19 / -0.06 |
| primary (plain) | tail | 28 | +3.70 / -1.09 / -2.10 | +0.88 / +0.84 / -0.37 | +0.21 / +0.76 / +0.04 |
| primary (plain) | xtail | 27 | +4.01 / -1.11 / -2.13 | +1.01 / +0.71 / -0.34 | +0.29 / +0.66 / +0.13 |
| primary, original pairs only | all | 59 | +3.78 / -1.07 / -2.10 | +0.94 / +0.82 / -0.37 | +0.25 / +0.77 / +0.06 |
| primary, without bet/appear | all | 63 | +3.76 / -1.08 / -2.11 | +0.94 / +0.83 / -0.37 | +0.25 / +0.77 / +0.06 |
| primary, cross-fitted orig_head pairs | all | 8 | +3.09 / -0.94 / -2.07 | +0.93 / +1.15 / -0.47 | +0.24 / +1.19 / -0.06 |
| primary, non-training pairs | all | 56 | +3.84 / -1.09 / -2.11 | +0.94 / +0.79 / -0.36 | +0.25 / +0.72 / +0.08 |
| primary + contaminated_bad | all | 71 | +3.79 / -1.07 / -2.10 | +0.92 / +0.82 / -0.36 | +0.26 / +0.77 / +0.07 |
| primary, by reliable | all | 41 | +3.78 / -1.10 / -2.18 | +0.96 / +0.88 / -0.34 | +0.26 / +0.83 / +0.10 |
| primary, by negative | all | 15 | +3.71 / -1.05 / -1.94 | +0.89 / +0.70 / -0.43 | +0.22 / +0.66 / -0.03 |
| new Head eval pairs (prep_object) | all | 6 | +4.76 / -0.74 / -2.42 | +1.23 / +1.24 / +0.11 | +0.84 / +1.38 / +0.71 |
| prep_object, pseudo_passive_ok | all | 56 | +4.56 / -0.82 / -1.94 | +1.30 / +1.07 / +0.07 | +0.91 / +1.44 / +0.78 |
| prep_object, pseudo_passive_bad | all | 18 | +4.10 / -0.73 / -1.92 | +1.12 / +0.95 / -0.29 | +0.50 / +0.98 / +0.12 |
| contaminated_bad | all | 7 | +4.12 / -1.10 / -2.02 | +0.80 / +0.72 / -0.31 | +0.35 / +0.66 / +0.16 |
| bad-side-high by | all | 11 | +3.93 / -0.98 / -1.98 | +0.85 / +0.64 / -0.39 | +0.29 / +0.64 / -0.09 |
| original, by reliable | all | 96 | +4.16 / -0.94 / -2.01 | +1.12 / +0.96 / -0.15 | +0.58 / +1.12 / +0.44 |
| original, by negative | all | 27 | +3.92 / -0.99 / -1.99 | +0.93 / +0.78 / -0.39 | +0.32 / +0.81 / +0.02 |
| original, by n.s. positive | all | 3 | +4.12 / -0.82 / -1.80 | +0.96 / +0.49 / -0.36 | +0.32 / +0.42 / +0.04 |
| all pairs | all | 145 | +4.12 / -0.93 / -2.02 | +1.09 / +0.93 / -0.19 | +0.54 / +1.06 / +0.35 |

