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

**Natural gate (primary: non-DAS, half B): passes.**

Strata (primary population):

| Stratum | contexts | interaction on L | interaction on O | passes |
|---|---|---|---|---|
| forgot | half B (3) | +1.41 [+0.97, +1.85] | +0.59 [+0.26, +0.92] | yes |
| forgot | all 6 (6) | +1.48 [+1.04, +1.88] | +0.59 [+0.26, +0.95] | yes |
| heard | half B (3) | +1.13 [+0.74, +1.51] | +0.92 [+0.60, +1.23] | yes |
| heard | all 6 (6) | +1.18 [+0.77, +1.58] | +0.90 [+0.53, +1.24] | yes |
| know | half B (3) | +1.98 [+1.56, +2.39] | +0.69 [+0.32, +1.02] | yes |
| know | all 6 (6) | +1.91 [+1.50, +2.32] | +0.75 [+0.42, +1.07] | yes |
| remember | half B (3) | +1.15 [+0.71, +1.57] | +0.90 [+0.53, +1.26] | yes |
| remember | all 6 (6) | +1.14 [+0.72, +1.57] | +0.92 [+0.58, +1.25] | yes |

## B7 matrix wh ("AUX NAME V" / "What AUX NAME V")

Kept I verbs (round 4): 45 (34 non-DAS, 11 DAS). An fp32 half-A I gate would drop none and add none.

| Population | contexts | n T | n I | interaction on L | interaction on O | passes |
|---|---|---:|---:|---|---|---|
| primary: non-DAS | half B | 97 | 34 | +1.21 [+0.80, +1.64] | +0.97 [+0.67, +1.28] | yes |
| primary: non-DAS | all contexts | 97 | 34 | +1.19 [+0.80, +1.59] | +0.96 [+0.67, +1.23] | yes |
| DAS bases, pooled | half B | 106 | 11 | +1.56 [+0.94, +2.26] | +1.18 [+0.84, +1.52] | yes |
| DAS bases, pooled | all contexts | 106 | 11 | +1.51 [+0.90, +2.14] | +1.17 [+0.84, +1.49] | yes |
| DAS bases, non-DAS T | half B | 97 | 11 | +1.54 [+0.88, +2.25] | +1.16 [+0.81, +1.51] | yes |
| DAS bases, non-DAS T | all contexts | 97 | 11 | +1.49 [+0.91, +2.12] | +1.15 [+0.83, +1.48] | yes |
| round-4 population (all T, kept I) | half B | 106 | 45 | +1.31 [+0.93, +1.72] | +1.03 [+0.78, +1.30] | yes |
| round-4 population (all T, kept I) | all contexts | 106 | 45 | +1.28 [+0.92, +1.65] | +1.02 [+0.76, +1.26] | yes |

**Natural gate (primary: non-DAS, half B): passes.**

Strata (primary population):

| Stratum | contexts | interaction on L | interaction on O | passes |
|---|---|---|---|---|
| Can | half B (3) | +1.10 [+0.73, +1.54] | +1.32 [+0.84, +1.76] | yes |
| Can | all 6 (6) | +1.06 [+0.66, +1.46] | +1.33 [+0.94, +1.71] | yes |
| Did | half B (3) | +1.69 [+1.22, +2.16] | +0.98 [+0.69, +1.26] | yes |
| Did | all 6 (6) | +1.58 [+1.12, +2.03] | +0.96 [+0.65, +1.26] | yes |
| Will | half B (3) | +0.73 [+0.32, +1.13] | +0.88 [+0.60, +1.15] | yes |
| Will | all 6 (6) | +0.77 [+0.37, +1.19] | +0.83 [+0.56, +1.11] | yes |
| Would | half B (3) | +1.33 [+0.89, +1.80] | +0.69 [+0.40, +0.98] | yes |
| Would | all 6 (6) | +1.34 [+0.92, +1.77] | +0.70 [+0.45, +0.96] | yes |

B7 base-form gate ("NAME can V", AUC over verb means along d_s): site 4 1.00, site 6 1.00, site 8 1.00, site 10 1.00, site 12 1.00, site 14 1.00, site 16 1.00, site 17 1.00

