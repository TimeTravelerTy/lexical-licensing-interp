# Site-16 passive-side controls (post hoc)

Criteria written before running: `passive_test_plan.md` (addendum) and `followup_plan.md`. Script `run_passive_controls.py`; analysis `analyze_passive_test.py controls`. Primary population (64 pairs), bad passive bases, T vs I donors; δ = 0.5656.

## Random rank-1 directions (split 0, 100 draws)

D per draw = mean over primary pairs (point estimate). DAS recomputed on the same split-0 rows.

| Control | Readout | DAS (split 0) | Null mean | Null 5th–95th pct | Max abs | Draws ≥ DAS | Draws ≤ DAS | Draws with abs ≥ δ |
|---|---|---:|---:|---|---:|---:|---:|---:|
| random_normmatched | log P(O) | +3.459 | -0.002 | [-0.036, +0.035] | 0.066 | 0.00 | 1.00 | 0 |
| random_normmatched | log P(" by") | -0.496 | +0.007 | [-0.038, +0.052] | 0.074 | 1.00 | 0.00 | 0 |
| random_normmatched | log P(".") | -1.572 | -0.016 | [-0.061, +0.035] | 0.078 | 1.00 | 0.00 | 0 |
| random_normmatched | log P(pron) | +5.265 | +0.016 | [-0.049, +0.087] | 0.196 | 0.00 | 1.00 | 0 |
| random_normmatched | log P(" the") | +4.298 | +0.007 | [-0.046, +0.047] | 0.130 | 0.00 | 1.00 | 0 |
| random_normmatched | M | +4.516 | -0.004 | [-0.042, +0.039] | 0.064 | 0.00 | 1.00 | 0 |

