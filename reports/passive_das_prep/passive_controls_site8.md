# Site-8 passive-side controls (post hoc)

Criteria written before running: `passive_test_plan.md` (addendum) and `followup_plan.md`. Script `run_passive_controls.py`; analysis `analyze_passive_test.py controls`. Primary population (64 pairs), bad passive bases, T vs I donors; δ = 0.4914.

## Shuffled-label DAS: a weaker copy of d

Shuffled-label DAS is not a null here. Permuting class labels across training verbs leaves about half of them correctly labelled, so training still recovers a partial version of d.

- Retrained shuffled-label bases: held-out active IIA 0.199 (the original run's control: 0.199, per-fold max |difference| 0.000); positive-control fraction 0.37 vs 0.78 for DAS.
- |cosine| with the DAS basis of the same fold: mean 0.69 (per fold 0.61–0.77; `bases_rank1.npz`). Random rank-1 directions in 2048 dimensions give about 0.02.
- Its passive effect has the same profile as DAS at about 0.6× on " by".

| Readout | DAS D | class | Shuffled-label D | class | DAS − shuffled |
|---|---|---|---|---|---|
| M | 0.43 [0.35, 0.52] |  | 0.24 [0.19, 0.29] |  | 0.19 [0.15, 0.24] |
| log P(O) | 0.36 [0.29, 0.44] | NO RISE | 0.19 [0.15, 0.23] | NO RISE | 0.17 [0.14, 0.22] |
| log P(I) | -0.07 [-0.09, -0.05] |  | -0.05 [-0.06, -0.04] |  | -0.02 [-0.03, -0.01] |
| log P(det) | 0.33 [0.26, 0.40] |  | 0.17 [0.14, 0.21] |  | 0.15 [0.12, 0.20] |
| log P(pron) | 0.88 [0.76, 0.99] |  | 0.46 [0.39, 0.53] |  | 0.42 [0.34, 0.50] |
| log P(refl) | 1.27 [1.13, 1.42] |  | 0.62 [0.54, 0.69] |  | 0.65 [0.57, 0.75] |
| log P(" by") | 0.72 [0.63, 0.82] | RISE | 0.44 [0.39, 0.51] | unresolved | 0.27 [0.22, 0.33] |
| log P(".") | -0.01 [-0.06, 0.05] | NO RISE | 0.02 [-0.01, 0.05] | NO RISE | -0.02 [-0.05, 0.01] |
| log P(" the") | 0.55 [0.48, 0.63] |  | 0.28 [0.24, 0.33] |  | 0.27 [0.22, 0.32] |
| log P(" him") | 1.40 [1.25, 1.57] |  | 0.77 [0.68, 0.87] |  | 0.63 [0.52, 0.75] |

Shuffled-label outcome by the declared rule: unresolved.

## Random rank-1 directions (split 0, 100 draws)

D per draw = mean over primary pairs (point estimate). DAS recomputed on the same split-0 rows.

| Control | Readout | DAS (split 0) | Null mean | Null 5th–95th pct | Max abs | Draws ≥ DAS | Draws ≤ DAS | Draws with abs ≥ δ |
|---|---|---:|---:|---|---:|---:|---:|---:|
| random | log P(O) | +0.366 | -0.000 | [-0.002, +0.002] | 0.003 | 0.00 | 1.00 | 0 |
| random | log P(" by") | +0.739 | +0.001 | [-0.002, +0.004] | 0.006 | 0.00 | 1.00 | 0 |
| random | log P(".") | +0.000 | +0.000 | [-0.002, +0.003] | 0.005 | 0.51 | 0.49 | 0 |
| random | log P(pron) | +0.885 | +0.000 | [-0.003, +0.003] | 0.007 | 0.00 | 1.00 | 0 |
| random | log P(" the") | +0.559 | -0.000 | [-0.003, +0.002] | 0.004 | 0.00 | 1.00 | 0 |
| random | M | +0.434 | -0.000 | [-0.002, +0.002] | 0.004 | 0.00 | 1.00 | 0 |
| random_normmatched | log P(O) | +0.366 | +0.009 | [-0.007, +0.029] | 0.038 | 0.00 | 1.00 | 0 |
| random_normmatched | log P(" by") | +0.739 | +0.007 | [-0.022, +0.032] | 0.047 | 0.00 | 1.00 | 0 |
| random_normmatched | log P(".") | +0.000 | +0.017 | [-0.004, +0.040] | 0.065 | 0.88 | 0.12 | 0 |
| random_normmatched | log P(pron) | +0.885 | +0.025 | [-0.008, +0.051] | 0.077 | 0.00 | 1.00 | 0 |
| random_normmatched | log P(" the") | +0.559 | +0.022 | [-0.001, +0.041] | 0.072 | 0.00 | 1.00 | 0 |
| random_normmatched | M | +0.434 | +0.010 | [-0.007, +0.031] | 0.038 | 0.00 | 1.00 | 0 |

