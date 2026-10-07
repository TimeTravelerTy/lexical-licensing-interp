# Site-6 passive-side controls (post hoc)

Criteria written before running: `passive_test_plan.md` (addendum) and `followup_plan.md`. Script `run_passive_controls.py`; analysis `analyze_passive_test.py controls`. Primary population (64 pairs), bad passive bases, T vs I donors; δ = 0.4654.

## Random rank-1 directions (split 0, 100 draws)

D per draw = mean over primary pairs (point estimate). DAS recomputed on the same split-0 rows.

| Control | Readout | DAS (split 0) | Null mean | Null 5th–95th pct | Max abs | Draws ≥ DAS | Draws ≤ DAS | Draws with abs ≥ δ |
|---|---|---:|---:|---|---:|---:|---:|---:|
| random_normmatched | log P(O) | +0.221 | +0.005 | [-0.013, +0.023] | 0.044 | 0.00 | 1.00 | 0 |
| random_normmatched | log P(" by") | +0.696 | +0.006 | [-0.024, +0.034] | 0.064 | 0.00 | 1.00 | 0 |
| random_normmatched | log P(".") | +0.097 | +0.020 | [-0.003, +0.037] | 0.039 | 0.00 | 1.00 | 0 |
| random_normmatched | log P(pron) | +0.474 | +0.021 | [-0.009, +0.048] | 0.100 | 0.00 | 1.00 | 0 |
| random_normmatched | log P(" the") | +0.323 | +0.020 | [-0.005, +0.040] | 0.072 | 0.00 | 1.00 | 0 |
| random_normmatched | M | +0.261 | +0.005 | [-0.013, +0.022] | 0.044 | 0.00 | 1.00 | 0 |

