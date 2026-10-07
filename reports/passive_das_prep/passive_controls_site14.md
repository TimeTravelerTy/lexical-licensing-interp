# Site-14 passive-side controls (post hoc)

Criteria written before running: `passive_test_plan.md` (addendum) and `followup_plan.md`. Script `run_passive_controls.py`; analysis `analyze_passive_test.py controls`. Primary population (64 pairs), bad passive bases, T vs I donors; δ = 0.5510.

## Random rank-1 directions (split 0, 100 draws)

D per draw = mean over primary pairs (point estimate). DAS recomputed on the same split-0 rows.

| Control | Readout | DAS (split 0) | Null mean | Null 5th–95th pct | Max abs | Draws ≥ DAS | Draws ≤ DAS | Draws with abs ≥ δ |
|---|---|---:|---:|---|---:|---:|---:|---:|
| random_normmatched | log P(O) | +2.407 | +0.010 | [-0.013, +0.039] | 0.058 | 0.00 | 1.00 | 0 |
| random_normmatched | log P(" by") | +0.457 | -0.001 | [-0.038, +0.041] | 0.074 | 0.00 | 1.00 | 0 |
| random_normmatched | log P(".") | -0.901 | +0.003 | [-0.034, +0.039] | 0.076 | 1.00 | 0.00 | 0 |
| random_normmatched | log P(pron) | +4.070 | +0.053 | [+0.014, +0.095] | 0.165 | 0.00 | 1.00 | 0 |
| random_normmatched | log P(" the") | +3.076 | +0.035 | [+0.000, +0.066] | 0.109 | 0.00 | 1.00 | 0 |
| random_normmatched | M | +2.807 | +0.010 | [-0.015, +0.042] | 0.058 | 0.00 | 1.00 | 0 |

