# Round 5, C8: filler-gap natural gates (fp32, non-DAS primary)

Spec: `plan.md`, C8. Run: `run_fillergap.py natural --dtype float32` (job 9007314); analysis: `analyze_fillergap_r5.py gates`. Interaction on L = [T − I]_what − [T − I]_that of L; on O = [T − I]_that − [T − I]_what of O. T and I verbs resampled independently, contexts jointly (2,000 draws, seed 17). Kept I verbs: round 4's half-A lists.

## B6 embedded wh ("I know that / what NAME V-ed")

Kept I verbs (round 4): 39 (28 non-DAS, 11 DAS). An fp32 half-A I gate would drop none and add none.

| Population | contexts | n T | n I | interaction on L | interaction on O | passes |
|---|---|---:|---:|---|---|---|
| primary: non-DAS | half B | 97 | 28 | +1.42 [+1.01, +1.84] | +0.78 [+0.46, +1.12] | yes |
| primary: non-DAS | all contexts | 97 | 28 | +1.43 [+1.04, +1.83] | +0.79 [+0.47, +1.12] | yes |
| DAS bases, pooled | half B | 106 | 11 | +2.20 [+1.35, +2.99] | +0.33 [-0.17, +0.76] | no |
| DAS bases, pooled | all contexts | 106 | 11 | +2.22 [+1.43, +3.01] | +0.31 [-0.17, +0.74] | no |
| DAS bases, non-DAS T | half B | 97 | 11 | +2.23 [+1.43, +3.04] | +0.33 [-0.17, +0.76] | no |
| DAS bases, non-DAS T | all contexts | 97 | 11 | +2.26 [+1.49, +2.99] | +0.31 [-0.17, +0.75] | no |
| round-4 population (all T, kept I) | half B | 106 | 39 | +1.61 [+1.18, +2.06] | +0.65 [+0.34, +0.95] | yes |
| round-4 population (all T, kept I) | all contexts | 106 | 39 | +1.63 [+1.21, +2.02] | +0.65 [+0.38, +0.96] | yes |
| non-DAS T, all non-DAS I (sensitivity) | half B | 97 | 38 | +1.12 [+0.72, +1.52] | +0.92 [+0.62, +1.24] | yes |
| non-DAS T, all non-DAS I (sensitivity) | all contexts | 97 | 38 | +1.13 [+0.76, +1.49] | +0.94 [+0.65, +1.24] | yes |

**Natural gate (primary: non-DAS, half B): passes.**

Strata (primary population):

| Stratum | contexts | interaction on L | interaction on O | passes |
|---|---|---|---|---|
| forgot | half B (3) | +1.41 [+0.95, +1.86] | +0.59 [+0.26, +0.91] | yes |
| forgot | all 6 (6) | +1.48 [+1.05, +1.89] | +0.59 [+0.26, +0.94] | yes |
| heard | half B (3) | +1.13 [+0.73, +1.52] | +0.92 [+0.61, +1.23] | yes |
| heard | all 6 (6) | +1.18 [+0.79, +1.58] | +0.90 [+0.52, +1.23] | yes |
| know | half B (3) | +1.98 [+1.57, +2.42] | +0.69 [+0.35, +1.04] | yes |
| know | all 6 (6) | +1.91 [+1.49, +2.33] | +0.75 [+0.41, +1.09] | yes |
| remember | half B (3) | +1.15 [+0.72, +1.61] | +0.90 [+0.54, +1.27] | yes |
| remember | all 6 (6) | +1.14 [+0.70, +1.56] | +0.92 [+0.57, +1.24] | yes |

## B7 matrix wh ("AUX NAME V" / "What AUX NAME V")

Kept I verbs (round 4): 45 (34 non-DAS, 11 DAS). An fp32 half-A I gate would drop none and add none.

| Population | contexts | n T | n I | interaction on L | interaction on O | passes |
|---|---|---:|---:|---|---|---|
| primary: non-DAS | half B | 97 | 34 | +1.21 [+0.81, +1.68] | +0.97 [+0.65, +1.27] | yes |
| primary: non-DAS | all contexts | 97 | 34 | +1.19 [+0.81, +1.57] | +0.96 [+0.68, +1.25] | yes |
| DAS bases, pooled | half B | 106 | 11 | +1.56 [+0.91, +2.23] | +1.18 [+0.83, +1.52] | yes |
| DAS bases, pooled | all contexts | 106 | 11 | +1.51 [+0.89, +2.16] | +1.17 [+0.85, +1.50] | yes |
| DAS bases, non-DAS T | half B | 97 | 11 | +1.54 [+0.88, +2.19] | +1.16 [+0.82, +1.50] | yes |
| DAS bases, non-DAS T | all contexts | 97 | 11 | +1.49 [+0.89, +2.14] | +1.15 [+0.82, +1.48] | yes |
| round-4 population (all T, kept I) | half B | 106 | 45 | +1.31 [+0.95, +1.71] | +1.03 [+0.77, +1.30] | yes |
| round-4 population (all T, kept I) | all contexts | 106 | 45 | +1.28 [+0.94, +1.64] | +1.02 [+0.78, +1.26] | yes |
| non-DAS T, all non-DAS I (sensitivity) | half B | 97 | 38 | +0.98 [+0.49, +1.44] | +1.00 [+0.72, +1.31] | yes |
| non-DAS T, all non-DAS I (sensitivity) | all contexts | 97 | 38 | +0.97 [+0.58, +1.38] | +0.98 [+0.72, +1.27] | yes |

**Natural gate (primary: non-DAS, half B): passes.**

Strata (primary population):

| Stratum | contexts | interaction on L | interaction on O | passes |
|---|---|---|---|---|
| Can | half B (3) | +1.10 [+0.70, +1.52] | +1.32 [+0.86, +1.80] | yes |
| Can | all 6 (6) | +1.06 [+0.67, +1.45] | +1.33 [+0.95, +1.70] | yes |
| Did | half B (3) | +1.69 [+1.24, +2.18] | +0.98 [+0.69, +1.28] | yes |
| Did | all 6 (6) | +1.58 [+1.14, +2.05] | +0.96 [+0.65, +1.24] | yes |
| Will | half B (3) | +0.73 [+0.33, +1.15] | +0.88 [+0.61, +1.14] | yes |
| Will | all 6 (6) | +0.77 [+0.40, +1.15] | +0.83 [+0.56, +1.10] | yes |
| Would | half B (3) | +1.33 [+0.87, +1.83] | +0.69 [+0.42, +0.98] | yes |
| Would | all 6 (6) | +1.34 [+0.91, +1.75] | +0.70 [+0.43, +0.97] | yes |

B7 base-form gate ("NAME can V", AUC over verb means along d_s): site 4 1.00, site 6 1.00, site 8 1.00, site 10 1.00, site 12 1.00, site 14 1.00, site 16 1.00, site 17 1.00

