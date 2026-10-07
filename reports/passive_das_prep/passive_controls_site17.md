# Site-17 passive-side controls (post hoc)

Criteria written before running: `passive_test_plan.md` (addendum) and `followup_plan.md`. Script `run_passive_controls.py`; analysis `analyze_passive_test.py controls`. Primary population (64 pairs), bad passive bases, T vs I donors; δ = 0.5716.

## Random rank-1 directions (split 0, 100 draws)

D per draw = mean over primary pairs (point estimate). DAS recomputed on the same split-0 rows.

| Control | Readout | DAS (split 0) | Null mean | Null 5th–95th pct | Max abs | Draws ≥ DAS | Draws ≤ DAS | Draws with abs ≥ δ |
|---|---|---:|---:|---|---:|---:|---:|---:|
| random_normmatched | log P(O) | +3.576 | -0.017 | [-0.056, +0.028] | 0.097 | 0.00 | 1.00 | 0 |
| random_normmatched | log P(" by") | -0.946 | +0.001 | [-0.061, +0.061] | 0.100 | 1.00 | 0.00 | 0 |
| random_normmatched | log P(".") | -1.987 | -0.044 | [-0.090, +0.008] | 0.128 | 1.00 | 0.00 | 0 |
| random_normmatched | log P(pron) | +5.150 | -0.025 | [-0.093, +0.054] | 0.137 | 0.00 | 1.00 | 0 |
| random_normmatched | log P(" the") | +4.364 | +0.008 | [-0.045, +0.067] | 0.121 | 0.00 | 1.00 | 0 |
| random_normmatched | M | +4.857 | -0.019 | [-0.062, +0.031] | 0.098 | 0.00 | 1.00 | 0 |

