# Round 5, C8: filler-gap natural gates (fp32, non-DAS primary)

Spec: `plan.md`, C8. Run: `run_fillergap.py natural --dtype float32` (job 9007314); analysis: `analyze_fillergap_r5.py gates`. Interaction on L = [T − I]_what − [T − I]_that of L; on O = [T − I]_that − [T − I]_what of O. T and I verbs resampled independently, contexts jointly (2,000 draws, seed 17). Kept I verbs: round 4's half-A lists.

## B6 embedded wh ("I know that / what NAME V-ed")

Kept I verbs (round 4): 18 (10 non-DAS, 8 DAS). An fp32 half-A I gate would drop none and add none.

| Population | contexts | n T | n I | interaction on L | interaction on O | passes |
|---|---|---:|---:|---|---|---|
| primary: non-DAS | half B | 97 | 10 | +1.84 [+1.29, +2.32] | +0.47 [+0.01, +0.91] | yes |
| primary: non-DAS | all contexts | 97 | 10 | +1.80 [+1.24, +2.27] | +0.49 [+0.02, +0.93] | yes |
| DAS bases, pooled | half B | 106 | 8 | +2.29 [+1.53, +2.94] | +0.66 [+0.23, +1.14] | yes |
| DAS bases, pooled | all contexts | 106 | 8 | +2.27 [+1.49, +2.89] | +0.65 [+0.19, +1.10] | yes |
| DAS bases, non-DAS T | half B | 97 | 8 | +2.30 [+1.50, +2.93] | +0.64 [+0.20, +1.09] | yes |
| DAS bases, non-DAS T | all contexts | 97 | 8 | +2.27 [+1.47, +2.88] | +0.62 [+0.16, +1.07] | yes |
| round-4 population (all T, kept I) | half B | 106 | 18 | +2.04 [+1.54, +2.49] | +0.57 [+0.23, +0.93] | yes |
| round-4 population (all T, kept I) | all contexts | 106 | 18 | +2.01 [+1.51, +2.45] | +0.57 [+0.24, +0.90] | yes |
| non-DAS T, all non-DAS I (sensitivity) | half B | 97 | 38 | +1.13 [+0.77, +1.49] | +0.99 [+0.70, +1.28] | yes |
| non-DAS T, all non-DAS I (sensitivity) | all contexts | 97 | 38 | +1.10 [+0.76, +1.43] | +0.99 [+0.72, +1.27] | yes |

**Natural gate (primary: non-DAS, half B): passes.**

Strata (primary population):

| Stratum | contexts | interaction on L | interaction on O | passes |
|---|---|---|---|---|
| forgot | half B (3) | +1.62 [+1.05, +2.12] | +0.37 [-0.03, +0.80] | no |
| forgot | all 6 (6) | +1.58 [+1.01, +2.07] | +0.42 [+0.01, +0.89] | yes |
| heard | half B (3) | +1.42 [+0.75, +2.04] | +0.68 [+0.03, +1.25] | yes |
| heard | all 6 (6) | +1.36 [+0.74, +2.01] | +0.72 [+0.00, +1.28] | yes |
| know | half B (3) | +2.31 [+1.83, +2.79] | +0.49 [+0.09, +0.87] | yes |
| know | all 6 (6) | +2.20 [+1.69, +2.67] | +0.47 [+0.07, +0.83] | yes |
| remember | half B (3) | +2.02 [+1.40, +2.60] | +0.35 [-0.10, +0.82] | no |
| remember | all 6 (6) | +2.07 [+1.47, +2.63] | +0.33 [-0.12, +0.81] | no |

## B7 matrix wh ("AUX NAME V" / "What AUX NAME V")

Kept I verbs (round 4): 32 (23 non-DAS, 9 DAS). An fp32 half-A I gate would drop none and add none.

| Population | contexts | n T | n I | interaction on L | interaction on O | passes |
|---|---|---:|---:|---|---|---|
| primary: non-DAS | half B | 97 | 23 | +1.84 [+1.44, +2.28] | +1.07 [+0.76, +1.38] | yes |
| primary: non-DAS | all contexts | 97 | 23 | +1.85 [+1.44, +2.25] | +1.07 [+0.80, +1.33] | yes |
| DAS bases, pooled | half B | 106 | 9 | +2.15 [+1.46, +2.84] | +0.85 [+0.49, +1.21] | yes |
| DAS bases, pooled | all contexts | 106 | 9 | +2.16 [+1.51, +2.87] | +0.85 [+0.53, +1.18] | yes |
| DAS bases, non-DAS T | half B | 97 | 9 | +2.16 [+1.50, +2.90] | +0.82 [+0.46, +1.21] | yes |
| DAS bases, non-DAS T | all contexts | 97 | 9 | +2.17 [+1.49, +2.87] | +0.82 [+0.48, +1.18] | yes |
| round-4 population (all T, kept I) | half B | 106 | 32 | +1.92 [+1.54, +2.32] | +1.03 [+0.74, +1.32] | yes |
| round-4 population (all T, kept I) | all contexts | 106 | 32 | +1.93 [+1.57, +2.30] | +1.03 [+0.77, +1.29] | yes |
| non-DAS T, all non-DAS I (sensitivity) | half B | 97 | 38 | +1.23 [+0.84, +1.63] | +1.04 [+0.73, +1.32] | yes |
| non-DAS T, all non-DAS I (sensitivity) | all contexts | 97 | 38 | +1.24 [+0.87, +1.65] | +1.03 [+0.76, +1.30] | yes |

**Natural gate (primary: non-DAS, half B): passes.**

Strata (primary population):

| Stratum | contexts | interaction on L | interaction on O | passes |
|---|---|---|---|---|
| Can | half B (3) | +1.87 [+1.41, +2.35] | +1.63 [+1.33, +1.94] | yes |
| Can | all 6 (6) | +1.88 [+1.41, +2.35] | +1.53 [+1.22, +1.84] | yes |
| Did | half B (3) | +2.13 [+1.67, +2.60] | +0.95 [+0.68, +1.23] | yes |
| Did | all 6 (6) | +2.20 [+1.75, +2.67] | +1.00 [+0.74, +1.27] | yes |
| Will | half B (3) | +1.96 [+1.48, +2.45] | +1.02 [+0.76, +1.26] | yes |
| Will | all 6 (6) | +1.94 [+1.52, +2.38] | +1.04 [+0.81, +1.28] | yes |
| Would | half B (3) | +1.43 [+1.08, +1.83] | +0.69 [+0.38, +1.01] | yes |
| Would | all 6 (6) | +1.36 [+0.99, +1.75] | +0.70 [+0.40, +1.03] | yes |

B7 base-form gate ("NAME can V", AUC over verb means along d_s): site 9 1.00, site 11 1.00, site 12 1.00, site 13 1.00

