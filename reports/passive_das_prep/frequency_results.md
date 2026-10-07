# Frequency and behaviour vs the passive d gap (step 4)

Spec: `followup_plan.md`, step 4. Script: `analyze_frequency.py`. 64 primary pairs (one observation per pair; verbs do not repeat). 95% CIs: pair bootstrap, 2,000 draws, seed 17. z: 0 = held-out active intransitive level, 1 = held-out active transitive level, at each site.

## Slope on participle Zipf (per Zipf unit)

| Site | Passive gap: mean | slope | Active gap: mean | slope | Passive/active ratio: mean | slope |
|---:|---:|---|---:|---|---:|---|
| 4 | 0.58 | +0.089 [+0.048, +0.143] | 0.75 | +0.178 [+0.122, +0.244] | 0.86 | -0.074 [-0.116, -0.026] |
| 6 | 0.46 | +0.057 [+0.027, +0.094] | 0.85 | +0.081 [+0.030, +0.138] | 0.54 | +0.013 [-0.026, +0.051] |
| 8 | 0.36 | +0.065 [+0.037, +0.099] | 0.87 | +0.058 [-0.000, +0.117] | 0.42 | +0.051 [+0.015, +0.090] |
| 10 | 0.39 | +0.080 [+0.052, +0.114] | 0.90 | +0.054 [+0.000, +0.112] | 0.43 | +0.069 [+0.034, +0.110] |
| 12 | 0.34 | +0.079 [+0.054, +0.106] | 0.90 | +0.055 [+0.001, +0.114] | 0.39 | +0.069 [+0.037, +0.104] |
| 14 | 0.29 | +0.068 [+0.045, +0.093] | 0.91 | +0.053 [-0.004, +0.112] | 0.32 | +0.060 [+0.028, +0.094] |
| 16 | 0.23 | +0.065 [+0.041, +0.095] | 0.91 | +0.056 [-0.003, +0.118] | 0.25 | +0.060 [+0.027, +0.097] |
| 17 | 0.18 | +0.054 [+0.028, +0.085] | 0.92 | +0.053 [-0.014, +0.122] | 0.21 | +0.055 [+0.016, +0.101] |

Lemma Zipf as the predictor (secondary):

| Site | Passive gap slope | Active gap slope | Ratio slope |
|---:|---|---|---|
| 4 | +0.050 [+0.005, +0.098] | +0.123 [+0.058, +0.190] | -0.103 [-0.218, -0.032] |
| 6 | +0.038 [-0.002, +0.078] | +0.029 [-0.030, +0.090] | +0.034 [+0.004, +0.068] |
| 8 | +0.047 [+0.011, +0.083] | +0.011 [-0.055, +0.072] | +0.062 [+0.023, +0.115] |
| 10 | +0.063 [+0.027, +0.099] | +0.014 [-0.053, +0.071] | +0.079 [+0.036, +0.139] |
| 12 | +0.057 [+0.023, +0.088] | +0.011 [-0.055, +0.068] | +0.074 [+0.028, +0.137] |
| 14 | +0.044 [+0.013, +0.074] | +0.002 [-0.062, +0.059] | +0.058 [+0.017, +0.108] |
| 16 | +0.038 [+0.007, +0.068] | -0.005 [-0.072, +0.054] | +0.048 [+0.010, +0.089] |
| 17 | +0.027 [-0.002, +0.057] | -0.009 [-0.081, +0.054] | +0.038 [-0.004, +0.082] |

Participle Zipf range over the 64 pairs: 1.35–5.18 (mean 2.64).

## Passive gap vs behavioural passive margins

| Site | Behaviour | Pairs | Spearman | Pearson |
|---:|---|---:|---|---|
| 4 | released by margin (task 0) | 59 | +0.24 [+0.00, +0.45] | +0.17 [-0.04, +0.40] |
| 4 | single-prompt " by" preference | 64 | +0.20 [-0.04, +0.42] | +0.20 [-0.01, +0.41] |
| 4 | curated passive_2 whole-sentence LP margin | 59 | +0.24 [+0.01, +0.45] | +0.22 [+0.00, +0.42] |
| 6 | released by margin (task 0) | 59 | +0.05 [-0.21, +0.28] | +0.01 [-0.23, +0.26] |
| 6 | single-prompt " by" preference | 64 | +0.05 [-0.19, +0.28] | +0.06 [-0.17, +0.28] |
| 6 | curated passive_2 whole-sentence LP margin | 59 | +0.10 [-0.16, +0.35] | +0.13 [-0.13, +0.37] |
| 8 | released by margin (task 0) | 59 | +0.08 [-0.19, +0.34] | +0.02 [-0.25, +0.30] |
| 8 | single-prompt " by" preference | 64 | +0.11 [-0.14, +0.35] | +0.10 [-0.15, +0.34] |
| 8 | curated passive_2 whole-sentence LP margin | 59 | +0.05 [-0.21, +0.31] | +0.04 [-0.19, +0.28] |
| 10 | released by margin (task 0) | 59 | +0.08 [-0.18, +0.34] | +0.04 [-0.22, +0.31] |
| 10 | single-prompt " by" preference | 64 | +0.11 [-0.12, +0.35] | +0.11 [-0.13, +0.33] |
| 10 | curated passive_2 whole-sentence LP margin | 59 | +0.06 [-0.19, +0.31] | +0.05 [-0.16, +0.27] |
| 12 | released by margin (task 0) | 59 | +0.11 [-0.13, +0.36] | +0.05 [-0.20, +0.32] |
| 12 | single-prompt " by" preference | 64 | +0.13 [-0.10, +0.37] | +0.11 [-0.12, +0.34] |
| 12 | curated passive_2 whole-sentence LP margin | 59 | +0.02 [-0.23, +0.26] | +0.01 [-0.21, +0.24] |
| 14 | released by margin (task 0) | 59 | +0.14 [-0.11, +0.39] | +0.07 [-0.18, +0.33] |
| 14 | single-prompt " by" preference | 64 | +0.18 [-0.06, +0.41] | +0.13 [-0.10, +0.36] |
| 14 | curated passive_2 whole-sentence LP margin | 59 | +0.03 [-0.23, +0.28] | +0.03 [-0.21, +0.25] |
| 16 | released by margin (task 0) | 59 | +0.18 [-0.07, +0.41] | +0.09 [-0.17, +0.35] |
| 16 | single-prompt " by" preference | 64 | +0.20 [-0.03, +0.40] | +0.13 [-0.12, +0.36] |
| 16 | curated passive_2 whole-sentence LP margin | 59 | +0.06 [-0.19, +0.30] | +0.00 [-0.23, +0.22] |
| 17 | released by margin (task 0) | 59 | +0.14 [-0.13, +0.38] | +0.03 [-0.24, +0.31] |
| 17 | single-prompt " by" preference | 64 | +0.18 [-0.07, +0.41] | +0.10 [-0.18, +0.36] |
| 17 | curated passive_2 whole-sentence LP margin | 59 | -0.03 [-0.28, +0.22] | -0.06 [-0.29, +0.17] |
