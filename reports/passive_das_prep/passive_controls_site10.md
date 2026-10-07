# Site-10 passive-side controls (post hoc)

Criteria written before running: `passive_test_plan.md` (addendum) and `followup_plan.md`. Script `run_passive_controls.py`; analysis `analyze_passive_test.py controls`. Primary population (64 pairs), bad passive bases, T vs I donors; δ = 0.5009.

## Random rank-1 directions (split 0, 100 draws)

D per draw = mean over primary pairs (point estimate). DAS recomputed on the same split-0 rows.

| Control | Readout | DAS (split 0) | Null mean | Null 5th–95th pct | Max abs | Draws ≥ DAS | Draws ≤ DAS | Draws with abs ≥ δ |
|---|---|---:|---:|---|---:|---:|---:|---:|
| random_normmatched | log P(O) | +0.687 | +0.018 | [-0.002, +0.041] | 0.059 | 0.00 | 1.00 | 0 |
| random_normmatched | log P(" by") | +0.781 | +0.007 | [-0.028, +0.038] | 0.049 | 0.00 | 1.00 | 0 |
| random_normmatched | log P(".") | -0.282 | +0.023 | [-0.006, +0.047] | 0.068 | 1.00 | 0.00 | 0 |
| random_normmatched | log P(pron) | +1.427 | +0.060 | [+0.019, +0.101] | 0.161 | 0.00 | 1.00 | 0 |
| random_normmatched | log P(" the") | +0.991 | +0.042 | [+0.017, +0.066] | 0.091 | 0.00 | 1.00 | 0 |
| random_normmatched | M | +0.803 | +0.018 | [-0.004, +0.041] | 0.064 | 0.00 | 1.00 | 0 |

