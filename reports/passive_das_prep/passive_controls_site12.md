# Site-12 passive-side controls (post hoc)

Criteria written before running: `passive_test_plan.md` (addendum) and `followup_plan.md`. Script `run_passive_controls.py`; analysis `analyze_passive_test.py controls`. Primary population (64 pairs), bad passive bases, T vs I donors; δ = 0.5281.

## Random rank-1 directions (split 0, 100 draws)

D per draw = mean over primary pairs (point estimate). DAS recomputed on the same split-0 rows.

| Control | Readout | DAS (split 0) | Null mean | Null 5th–95th pct | Max abs | Draws ≥ DAS | Draws ≤ DAS | Draws with abs ≥ δ |
|---|---|---:|---:|---|---:|---:|---:|---:|
| random_normmatched | log P(O) | +1.170 | +0.009 | [-0.012, +0.035] | 0.050 | 0.00 | 1.00 | 0 |
| random_normmatched | log P(" by") | +0.815 | -0.002 | [-0.031, +0.034] | 0.077 | 0.00 | 1.00 | 0 |
| random_normmatched | log P(".") | -0.532 | +0.012 | [-0.016, +0.048] | 0.084 | 1.00 | 0.00 | 0 |
| random_normmatched | log P(pron) | +2.306 | +0.044 | [+0.005, +0.081] | 0.168 | 0.00 | 1.00 | 0 |
| random_normmatched | log P(" the") | +1.602 | +0.027 | [+0.000, +0.057] | 0.100 | 0.00 | 1.00 | 0 |
| random_normmatched | M | +1.348 | +0.009 | [-0.015, +0.039] | 0.051 | 0.00 | 1.00 | 0 |

