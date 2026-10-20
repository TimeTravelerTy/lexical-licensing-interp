# Round 6, A6: the D11 frame contrast over Pythia checkpoints

Spec: `plan.md`, A6. Runs: `run_frame_contrast.py --revision step<N>` (freq prefixes); analysis: `analyze_checkpoints.py`. Exposure E = c_total × step / 143000 (A1's c_total). 95% CIs: pairs within band (token contrast: band × cell) × nouns within animacy, 2,000 draws, identical across checkpoints. Readings as A2 (bound 0.25 nats).

## Stationarity diagnostic (limited)

12 slices of 25,000,000 stored tokens starting at steps 128, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000, 130000, 142000; lowercase single-token D11 participles. Rate ratio last-to-first slice from a quasi-Poisson trend on log step (95% CI):

| group | verbs | total count | rate ratio (95% CI) | dispersion |
|---|---:|---:|---|---|
| Head single-token | 89 | 307,556 | 1.010 [0.996, 1.025] | 1.7 |
| Tail single-token | 44 | 32,703 | 1.015 [0.957, 1.077] | 3.0 |
| defined | 1 | 26,822 | 0.980 [0.912, 1.052] | 3.6 |
| received | 1 | 26,666 | 1.026 [0.977, 1.078] | 1.7 |
| sent | 1 | 19,819 | 1.022 [0.947, 1.104] | 3.1 |
| happened | 1 | 14,966 | 1.016 [0.951, 1.086] | 1.8 |
| placed | 1 | 13,609 | 1.038 [0.952, 1.131] | 2.7 |
| died | 1 | 13,270 | 1.049 [0.970, 1.134] | 2.1 |
| prepared | 1 | 11,160 | 0.967 [0.913, 1.025] | 1.0 |
| occurred | 1 | 10,650 | 1.014 [0.938, 1.098] | 1.8 |
| lived | 1 | 8,233 | 1.010 [0.926, 1.102] | 1.7 |
| arrived | 1 | 7,719 | 0.960 [0.896, 1.030] | 1.0 |

**Stationary within ±20% (Head and Tail pooled): True.** A diagnostic only: it does not cover capitalized or unmarked forms, multi-token verbs, or the genre concentration of individual rare verbs.

## pythia-1.4b: trajectories (freq)

| step | checks (recon / identity) | L Head / Tail / XTail | ΔL Head − XTail | reading | side | ΔL_tok | reading | side (token-std.) | Δ_pass(I) Head / XTail | Head AUC | D_L |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 1e-05 / 2e-05 | +0.07 / -0.05 / +0.02 | +0.06 [-0.12, +0.23] | negligible | no deficit | -0.02 [-0.18, +0.14] | negligible | no deficit | +0.06 / +0.05 | 0.54 | +0.07 [-0.15, +0.32] |
| 128 | 9e-06 / 1e-05 | +0.10 / -0.06 / +0.02 | +0.08 [-0.12, +0.26] | negligible | no deficit | -0.10 [-0.28, +0.07] | negligible | no deficit | +0.06 / -0.05 | 0.63 | -0.03 [-0.26, +0.21] |
| 256 | 1e-05 / 1e-05 | +0.28 / -0.07 / -0.09 | +0.37 [+0.14, +0.62] | deficit | transitive-side | -0.02 [-0.26, +0.23] | negligible | no deficit | +0.25 / +0.28 | 0.69 | +0.19 [-0.13, +0.52] |
| 512 | 1e-05 / 2e-05 | +1.78 / +0.06 / +0.02 | +1.76 [+1.30, +2.23] | deficit | intransitive-side | -0.11 [-0.46, +0.24] | unresolved | no deficit | -1.19 / +1.09 | 0.90 | +1.09 [+0.31, +1.87] |
| 1000 | 2e-05 / 2e-05 | +3.19 / +1.06 / +0.17 | +3.03 [+2.44, +3.57] | deficit | intransitive-side | +0.66 [+0.21, +1.12] | deficit | intransitive-side | -2.95 / +0.40 | 0.95 | +1.32 [+0.30, +2.30] |
| 2000 | 2e-05 / 3e-05 | +3.23 / +1.57 / +0.33 | +2.90 [+2.30, +3.50] | deficit | intransitive-side | +1.26 [+0.78, +1.78] | deficit | intransitive-side | -2.59 / +0.61 | 0.94 | +1.44 [+0.45, +2.45] |
| 3000 | 3e-05 / 4e-05 | +3.49 / +2.01 / +0.59 | +2.90 [+2.26, +3.53] | deficit | intransitive-side | +1.46 [+0.96, +1.99] | deficit | intransitive-side | -2.80 / +0.49 | 0.94 | +1.38 [+0.41, +2.39] |
| 4000 | 3e-05 / 5e-05 | +3.32 / +2.21 / +0.94 | +2.38 [+1.71, +3.03] | deficit | intransitive-side | +1.52 [+0.89, +2.14] | deficit | intransitive-side | -2.95 / -0.12 | 0.93 | +1.20 [+0.26, +2.18] |
| 6000 | 2e-05 / 4e-05 | +3.33 / +2.32 / +1.52 | +1.81 [+1.15, +2.44] | deficit | intransitive-side | +1.03 [+0.39, +1.66] | deficit | intransitive-side | -3.24 / -0.97 | 0.93 | +1.23 [+0.39, +2.09] |
| 10000 | 2e-05 / 3e-05 | +3.30 / +2.64 / +1.42 | +1.88 [+1.18, +2.56] | deficit | intransitive-side | +1.63 [+1.01, +2.25] | deficit | intransitive-side | -2.90 / -0.68 | 0.95 | +1.23 [+0.40, +2.07] |
| 16000 | 2e-05 / 3e-05 | +3.07 / +2.44 / +1.70 | +1.37 [+0.70, +2.06] | deficit | intransitive-side | +1.07 [+0.52, +1.62] | deficit | intransitive-side | -2.79 / -1.03 | 0.92 | +1.21 [+0.34, +2.11] |
| 25000 | 2e-05 / 3e-05 | +3.12 / +2.47 / +1.90 | +1.21 [+0.56, +1.89] | deficit | intransitive-side | +0.93 [+0.34, +1.50] | deficit | intransitive-side | -2.85 / -1.30 | 0.94 | +1.36 [+0.55, +2.19] |
| 40000 | 3e-05 / 3e-05 | +3.20 / +2.48 / +1.87 | +1.33 [+0.64, +2.01] | deficit | intransitive-side | +0.94 [+0.32, +1.50] | deficit | intransitive-side | -2.78 / -1.08 | 0.93 | +1.41 [+0.55, +2.26] |
| 63000 | 2e-05 / 3e-05 | +3.24 / +2.34 / +1.61 | +1.63 [+0.90, +2.37] | deficit | intransitive-side | +1.14 [+0.50, +1.75] | deficit | intransitive-side | -2.90 / -0.82 | 0.93 | +1.43 [+0.55, +2.30] |
| 100000 | 2e-05 / 3e-05 | +3.16 / +2.49 / +1.88 | +1.28 [+0.57, +2.01] | deficit | intransitive-side | +1.04 [+0.49, +1.58] | deficit | intransitive-side | -2.82 / -1.15 | 0.92 | +1.44 [+0.51, +2.33] |
| 143000 | 3e-05 / 4e-05 | +3.37 / +2.56 / +1.96 | +1.41 [+0.70, +2.15] | deficit | intransitive-side | +0.99 [+0.44, +1.55] | deficit | intransitive-side | -2.89 / -1.12 | 0.93 | +1.56 [+0.65, +2.48] |

Onset (first step after which every later checkpoint reads deficit; descriptive): Head − XTail 256, token-standardized 1000.

- Curve collapse [Δ_pass(I)] (c_total, all steps; 16 steps × 281 verbs): CV MSE M_E 2.123, M_Cg 2.077, M_2D 2.029, M_Eg 1.985, M_g 2.237. **CC1** M_E vs M_2D relative gain of M_2D +0.044 [+0.007, +0.070] → **unresolved**; **CC2** vs M_Cg +0.021 [-0.014, +0.049] → **unresolved**; M_E beats step-only by > 2%: **False** (+0.051 [-0.005, +0.107]); diagnostic step beyond E +0.065 [+0.041, +0.087].
  Step offsets in M_Eg (relative to the first step): 128: +0.06, 256: +0.46, 512: +0.87, 1000: -0.14, 2000: +0.24, 3000: +0.14, 4000: -0.20, 6000: -0.57, 10000: -0.17, 16000: +0.02, 25000: +0.07, 40000: +0.47, 63000: +0.78, 100000: +0.76, 143000: +0.88
  Matched cells (|Δ log10 E| < 0.1, steps ≥ 4× apart): 226550 pairs; mean Δ_pass(I) later − earlier +0.080 (median +0.102).
- Curve collapse [Δ_pass(I)] (c_total, steps ≥ 1000; 12 steps × 281 verbs): CV MSE M_E 2.546, M_Cg 2.508, M_2D 2.491, M_Eg 2.458, M_g 2.766. **CC1** M_E vs M_2D relative gain of M_2D +0.022 [-0.020, +0.053] → **unresolved**; **CC2** vs M_Cg +0.015 [-0.026, +0.050] → **unresolved**; M_E beats step-only by > 2%: **True** (+0.079 [+0.029, +0.129]); diagnostic step beyond E +0.035 [+0.011, +0.066].
- Curve collapse [Δ_pass(I)] (c_total, steps ≥ 4000; 9 steps × 281 verbs): CV MSE M_E 2.730, M_Cg 2.645, M_2D 2.637, M_Eg 2.615, M_g 2.887. **CC1** M_E vs M_2D relative gain of M_2D +0.034 [-0.010, +0.072] → **unresolved**; **CC2** vs M_Cg +0.031 [-0.013, +0.072] → **unresolved**; M_E beats step-only by > 2%: **False** (+0.054 [+0.006, +0.102]); diagnostic step beyond E +0.042 [+0.011, +0.083].
- Curve collapse [Δ_pass(I)] (c_marked, all steps; 16 steps × 281 verbs): CV MSE M_E 2.123, M_Cg 2.077, M_2D 2.029, M_Eg 1.985, M_g 2.237. **CC1** M_E vs M_2D relative gain of M_2D +0.044 [+0.007, +0.070] → **unresolved**; **CC2** vs M_Cg +0.021 [-0.014, +0.048] → **unresolved**; M_E beats step-only by > 2%: **False** (+0.051 [-0.005, +0.107]); diagnostic step beyond E +0.065 [+0.041, +0.087].

## pythia-410m: trajectories (freq)

| step | checks (recon / identity) | L Head / Tail / XTail | ΔL Head − XTail | reading | side | ΔL_tok | reading | side (token-std.) | Δ_pass(I) Head / XTail | Head AUC | D_L |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 6e-06 / 8e-06 | +0.03 / -0.07 / -0.01 | +0.04 [-0.13, +0.21] | negligible | no deficit | -0.06 [-0.20, +0.08] | negligible | no deficit | +0.03 / +0.02 | 0.50 | +0.13 [-0.11, +0.36] |
| 128 | 7e-06 / 9e-06 | +0.07 / -0.07 / -0.02 | +0.09 [-0.10, +0.28] | negligible | no deficit | -0.06 [-0.22, +0.11] | negligible | no deficit | +0.02 / +0.10 | 0.54 | +0.24 [+0.02, +0.45] |
| 256 | 9e-06 / 1e-05 | +0.18 / -0.04 / +0.07 | +0.12 [-0.11, +0.35] | unresolved | no deficit | -0.12 [-0.39, +0.15] | unresolved | no deficit | +0.24 / +0.22 | 0.63 | +0.37 [+0.11, +0.62] |
| 512 | 1e-05 / 2e-05 | +1.35 / +0.07 / +0.30 | +1.05 [+0.62, +1.51] | deficit | intransitive-side | -0.48 [-0.86, -0.07] | surplus | no deficit | -0.54 / +0.62 | 0.88 | +0.75 [+0.10, +1.39] |
| 1000 | 1e-05 / 2e-05 | +2.58 / +0.67 / +0.17 | +2.41 [+1.84, +3.00] | deficit | intransitive-side | +0.13 [-0.27, +0.50] | unresolved | no deficit | -2.70 / +0.71 | 0.94 | +0.93 [+0.12, +1.76] |
| 2000 | 1e-05 / 2e-05 | +3.32 / +1.94 / +0.36 | +2.96 [+2.35, +3.59] | deficit | intransitive-side | +1.72 [+1.17, +2.30] | deficit | intransitive-side | -3.15 / +0.13 | 0.95 | +1.17 [+0.29, +2.14] |
| 3000 | 1e-05 / 2e-05 | +3.47 / +2.05 / +0.67 | +2.79 [+2.17, +3.42] | deficit | intransitive-side | +1.58 [+1.10, +2.07] | deficit | intransitive-side | -2.82 / +0.26 | 0.94 | +1.23 [+0.28, +2.23] |
| 4000 | 1e-05 / 2e-05 | +3.01 / +1.94 / +0.77 | +2.24 [+1.64, +2.84] | deficit | intransitive-side | +1.32 [+0.78, +1.83] | deficit | intransitive-side | -2.92 / -0.33 | 0.93 | +0.78 [-0.10, +1.72] |
| 6000 | 1e-05 / 2e-05 | +3.22 / +2.30 / +0.98 | +2.25 [+1.62, +2.88] | deficit | intransitive-side | +1.72 [+1.25, +2.16] | deficit | intransitive-side | -2.93 / -0.43 | 0.93 | +0.91 [+0.04, +1.81] |
| 10000 | 1e-05 / 2e-05 | +3.09 / +2.44 / +1.30 | +1.80 [+1.14, +2.45] | deficit | intransitive-side | +1.60 [+1.08, +2.13] | deficit | intransitive-side | -2.70 / -0.54 | 0.94 | +0.88 [-0.02, +1.80] |
| 16000 | 1e-05 / 2e-05 | +3.23 / +2.33 / +1.25 | +1.99 [+1.30, +2.68] | deficit | intransitive-side | +1.56 [+1.00, +2.12] | deficit | intransitive-side | -2.97 / -0.57 | 0.93 | +1.22 [+0.30, +2.18] |
| 25000 | 2e-05 / 3e-05 | +3.15 / +2.34 / +1.34 | +1.81 [+1.09, +2.49] | deficit | intransitive-side | +1.39 [+0.86, +1.91] | deficit | intransitive-side | -3.00 / -0.71 | 0.94 | +1.35 [+0.41, +2.34] |
| 40000 | 1e-05 / 2e-05 | +3.18 / +2.23 / +1.49 | +1.70 [+1.03, +2.34] | deficit | intransitive-side | +1.12 [+0.56, +1.68] | deficit | intransitive-side | -2.92 / -0.67 | 0.94 | +1.32 [+0.42, +2.21] |
| 63000 | 1e-05 / 2e-05 | +3.18 / +2.40 / +1.52 | +1.66 [+0.97, +2.31] | deficit | intransitive-side | +1.22 [+0.62, +1.82] | deficit | intransitive-side | -3.05 / -0.62 | 0.94 | +1.34 [+0.47, +2.23] |
| 100000 | 2e-05 / 2e-05 | +3.31 / +2.71 / +1.79 | +1.53 [+0.86, +2.19] | deficit | intransitive-side | +1.32 [+0.67, +1.96] | deficit | intransitive-side | -3.36 / -1.34 | 0.94 | +1.35 [+0.42, +2.30] |
| 143000 | 2e-05 / 3e-05 | +3.37 / +2.67 / +1.72 | +1.65 [+0.98, +2.30] | deficit | intransitive-side | +1.26 [+0.64, +1.85] | deficit | intransitive-side | -3.22 / -1.07 | 0.95 | +1.39 [+0.48, +2.35] |

Onset (first step after which every later checkpoint reads deficit; descriptive): Head − XTail 512, token-standardized 2000.

- Curve collapse [Δ_pass(I)] (c_total, all steps; 16 steps × 281 verbs): CV MSE M_E 1.933, M_Cg 1.930, M_2D 1.885, M_Eg 1.829, M_g 2.099. **CC1** M_E vs M_2D relative gain of M_2D +0.025 [-0.009, +0.047] → **unresolved**; **CC2** vs M_Cg +0.002 [-0.032, +0.026] → **unresolved**; M_E beats step-only by > 2%: **True** (+0.079 [+0.025, +0.134]); diagnostic step beyond E +0.054 [+0.030, +0.079].
  Step offsets in M_Eg (relative to the first step): 128: +0.10, 256: +0.37, 512: +0.64, 1000: +0.15, 2000: -0.37, 3000: -0.07, 4000: -0.38, 6000: -0.33, 10000: -0.10, 16000: +0.08, 25000: +0.24, 40000: +0.59, 63000: +0.76, 100000: +0.38, 143000: +0.76
  Matched cells (|Δ log10 E| < 0.1, steps ≥ 4× apart): 226550 pairs; mean Δ_pass(I) later − earlier +0.133 (median +0.123).
- Curve collapse [Δ_pass(I)] (c_total, steps ≥ 1000; 12 steps × 281 verbs): CV MSE M_E 2.349, M_Cg 2.293, M_2D 2.284, M_Eg 2.279, M_g 2.575. **CC1** M_E vs M_2D relative gain of M_2D +0.028 [-0.012, +0.057] → **unresolved**; **CC2** vs M_Cg +0.024 [-0.013, +0.056] → **unresolved**; M_E beats step-only by > 2%: **True** (+0.088 [+0.041, +0.138]); diagnostic step beyond E +0.030 [+0.007, +0.061].
- Curve collapse [Δ_pass(I)] (c_total, steps ≥ 4000; 9 steps × 281 verbs): CV MSE M_E 2.471, M_Cg 2.394, M_2D 2.408, M_Eg 2.388, M_g 2.677. **CC1** M_E vs M_2D relative gain of M_2D +0.026 [-0.017, +0.059] → **unresolved**; **CC2** vs M_Cg +0.031 [-0.010, +0.067] → **unresolved**; M_E beats step-only by > 2%: **True** (+0.077 [+0.030, +0.128]); diagnostic step beyond E +0.034 [+0.005, +0.071].
- Curve collapse [Δ_pass(I)] (c_marked, all steps; 16 steps × 281 verbs): CV MSE M_E 1.933, M_Cg 1.930, M_2D 1.885, M_Eg 1.829, M_g 2.099. **CC1** M_E vs M_2D relative gain of M_2D +0.025 [-0.010, +0.047] → **unresolved**; **CC2** vs M_Cg +0.002 [-0.032, +0.026] → **unresolved**; M_E beats step-only by > 2%: **True** (+0.079 [+0.025, +0.135]); diagnostic step beyond E +0.054 [+0.030, +0.079].

## pythia-6.9b: trajectories (freq)

| step | checks (recon / identity) | L Head / Tail / XTail | ΔL Head − XTail | reading | side | ΔL_tok | reading | side (token-std.) | Δ_pass(I) Head / XTail | Head AUC | D_L |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 3e-05 / 4e-05 | +1.56 / +0.29 / +0.01 | +1.55 [+1.08, +2.03] | deficit | intransitive-side | +0.06 [-0.33, +0.49] | unresolved | no deficit | -0.60 / +0.46 | 0.90 | +0.72 [+0.10, +1.37] |
| 2000 | 4e-05 / 6e-05 | +3.31 / +1.64 / +0.65 | +2.66 [+2.02, +3.31] | deficit | intransitive-side | +1.05 [+0.49, +1.61] | deficit | intransitive-side | -3.56 / -0.31 | 0.94 | +1.45 [+0.36, +2.56] |
| 8000 | 4e-05 / 8e-05 | +3.20 / +2.54 / +1.80 | +1.40 [+0.63, +2.14] | deficit | intransitive-side | +1.04 [+0.41, +1.65] | deficit | intransitive-side | -3.01 / -1.15 | 0.92 | +1.17 [+0.24, +2.13] |
| 23000 | 3e-05 / 4e-05 | +3.16 / +2.51 / +1.63 | +1.53 [+0.81, +2.24] | deficit | intransitive-side | +1.09 [+0.48, +1.71] | deficit | intransitive-side | -2.73 / -0.94 | 0.93 | +1.38 [+0.45, +2.31] |
| 63000 | 3e-05 / 4e-05 | +3.24 / +2.50 / +1.71 | +1.53 [+0.84, +2.23] | deficit | intransitive-side | +1.18 [+0.62, +1.74] | deficit | intransitive-side | -3.11 / -1.23 | 0.94 | +1.30 [+0.43, +2.17] |
| 143000 | 3e-05 / 5e-05 | +3.50 / +2.64 / +1.86 | +1.64 [+0.93, +2.35] | deficit | intransitive-side | +1.04 [+0.47, +1.59] | deficit | intransitive-side | -3.27 / -1.34 | 0.94 | +1.40 [+0.51, +2.29] |

Onset (first step after which every later checkpoint reads deficit; descriptive): Head − XTail 512, token-standardized 2000.

- Curve collapse [Δ_pass(I)] (c_total, all steps; 6 steps × 281 verbs): CV MSE M_E 2.534, M_Cg 2.357, M_2D 2.317, M_Eg 2.332, M_g 2.550. **CC1** M_E vs M_2D relative gain of M_2D +0.086 [+0.044, +0.113] → **collapse fails**; **CC2** vs M_Cg +0.070 [+0.031, +0.099] → **frequency + step more predictive**; M_E beats step-only by > 2%: **False** (+0.006 [-0.042, +0.061]); diagnostic step beyond E +0.080 [+0.052, +0.107].
  Step offsets in M_Eg (relative to the first step): 2000: -1.29, 8000: -1.17, 23000: -0.44, 63000: -0.29, 143000: -0.10
  Matched cells (|Δ log10 E| < 0.1, steps ≥ 4× apart): 30485 pairs; mean Δ_pass(I) later − earlier +0.179 (median +0.216).
- Curve collapse [Δ_pass(I)] (c_total, steps ≥ 1000; 5 steps × 281 verbs): CV MSE M_E 2.666, M_Cg 2.600, M_2D 2.558, M_Eg 2.574, M_g 2.841. **CC1** M_E vs M_2D relative gain of M_2D +0.040 [-0.004, +0.080] → **unresolved**; **CC2** vs M_Cg +0.025 [-0.019, +0.068] → **unresolved**; M_E beats step-only by > 2%: **True** (+0.062 [+0.025, +0.100]); diagnostic step beyond E +0.034 [+0.004, +0.077].
- Curve collapse [Δ_pass(I)] (c_total, steps ≥ 4000; 4 steps × 281 verbs): CV MSE M_E 2.710, M_Cg 2.636, M_2D 2.637, M_Eg 2.616, M_g 2.864. **CC1** M_E vs M_2D relative gain of M_2D +0.027 [-0.012, +0.061] → **unresolved**; **CC2** vs M_Cg +0.028 [-0.011, +0.064] → **unresolved**; M_E beats step-only by > 2%: **False** (+0.054 [+0.008, +0.104]); diagnostic step beyond E +0.035 [+0.008, +0.070].
- Curve collapse [Δ_pass(I)] (c_marked, all steps; 6 steps × 281 verbs): CV MSE M_E 2.534, M_Cg 2.357, M_2D 2.317, M_Eg 2.332, M_g 2.550. **CC1** M_E vs M_2D relative gain of M_2D +0.086 [+0.044, +0.113] → **collapse fails**; **CC2** vs M_Cg +0.070 [+0.031, +0.099] → **frequency + step more predictive**; M_E beats step-only by > 2%: **False** (+0.006 [-0.042, +0.061]); diagnostic step beyond E +0.080 [+0.052, +0.107].

