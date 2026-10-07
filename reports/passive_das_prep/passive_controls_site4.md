# Site-4 passive-side controls (post hoc)

Criteria written before running: `passive_test_plan.md` (addendum) and `followup_plan.md`. Script `run_passive_controls.py`; analysis `analyze_passive_test.py controls`. Primary population (64 pairs), bad passive bases, T vs I donors; δ = 0.4177.

## Random rank-1 directions (split 0, 100 draws)

D per draw = mean over primary pairs (point estimate). DAS recomputed on the same split-0 rows.

| Control | Readout | DAS (split 0) | Null mean | Null 5th–95th pct | Max abs | Draws ≥ DAS | Draws ≤ DAS | Draws with abs ≥ δ |
|---|---|---:|---:|---|---:|---:|---:|---:|
| random_normmatched | log P(O) | +0.106 | +0.010 | [-0.005, +0.023] | 0.029 | 0.00 | 1.00 | 0 |
| random_normmatched | log P(" by") | +0.589 | +0.005 | [-0.025, +0.027] | 0.041 | 0.00 | 1.00 | 0 |
| random_normmatched | log P(".") | +0.129 | +0.020 | [+0.008, +0.031] | 0.041 | 0.00 | 1.00 | 0 |
| random_normmatched | log P(pron) | +0.092 | +0.026 | [+0.001, +0.049] | 0.082 | 0.00 | 1.00 | 0 |
| random_normmatched | log P(" the") | +0.157 | +0.023 | [+0.006, +0.042] | 0.052 | 0.00 | 1.00 | 0 |
| random_normmatched | M | +0.119 | +0.009 | [-0.007, +0.023] | 0.028 | 0.00 | 1.00 | 0 |

