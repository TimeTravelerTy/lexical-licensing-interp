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

Band means and side parts (positive part = contributes to the deficit):

| step | Δ_pass(T) Head / Tail / XTail | Δ_pass(I) Head / Tail / XTail | Head − XTail parts T / −I | token-std. parts T / −I |
|---|---|---|---|---|
| 64 | +0.13 / -0.02 / +0.07 | +0.06 / +0.03 / +0.05 | +0.06 [-0.06, +0.18] / -0.00 [-0.12, +0.12] | -0.07 [-0.17, +0.04] / +0.05 [-0.06, +0.15] |
| 128 | +0.16 / -0.01 / -0.03 | +0.06 / +0.05 / -0.05 | +0.19 [+0.04, +0.32] / -0.11 [-0.24, +0.02] | -0.05 [-0.17, +0.06] / -0.05 [-0.18, +0.08] |
| 256 | +0.53 / +0.21 / +0.19 | +0.25 / +0.29 / +0.28 | +0.34 [+0.15, +0.52] / +0.03 [-0.15, +0.23] | -0.07 [-0.25, +0.11] / +0.05 [-0.11, +0.21] |
| 512 | +0.58 / +0.74 / +1.11 | -1.19 / +0.67 / +1.09 | -0.52 [-0.86, -0.20] / +2.28 [+1.89, +2.70] | -0.31 [-0.57, -0.04] / +0.20 [-0.04, +0.43] |
| 1000 | +0.24 / +0.27 / +0.56 | -2.95 / -0.79 / +0.40 | -0.32 [-0.79, +0.11] / +3.35 [+2.91, +3.82] | -0.12 [-0.41, +0.18] / +0.78 [+0.44, +1.13] |
| 2000 | +0.64 / +0.69 / +0.94 | -2.59 / -0.88 / +0.61 | -0.30 [-0.75, +0.11] / +3.20 [+2.72, +3.69] | -0.06 [-0.38, +0.26] / +1.32 [+0.90, +1.75] |
| 3000 | +0.69 / +0.71 / +1.07 | -2.80 / -1.30 / +0.49 | -0.39 [-0.83, +0.05] / +3.28 [+2.72, +3.84] | -0.20 [-0.58, +0.17] / +1.66 [+1.21, +2.10] |
| 4000 | +0.37 / +0.42 / +0.81 | -2.95 / -1.78 / -0.12 | -0.44 [-0.89, -0.01] / +2.82 [+2.23, +3.41] | -0.18 [-0.54, +0.19] / +1.70 [+1.21, +2.17] |
| 6000 | +0.08 / +0.09 / +0.55 | -3.24 / -2.23 / -0.97 | -0.46 [-0.91, -0.04] / +2.27 [+1.69, +2.82] | -0.15 [-0.61, +0.29] / +1.18 [+0.72, +1.62] |
| 10000 | +0.40 / +0.36 / +0.74 | -2.90 / -2.28 / -0.68 | -0.34 [-0.78, +0.06] / +2.22 [+1.60, +2.81] | -0.12 [-0.51, +0.27] / +1.75 [+1.25, +2.24] |
| 16000 | +0.28 / +0.31 / +0.67 | -2.79 / -2.13 / -1.03 | -0.39 [-0.85, +0.06] / +1.76 [+1.18, +2.35] | -0.16 [-0.56, +0.24] / +1.22 [+0.76, +1.70] |
| 25000 | +0.27 / +0.22 / +0.60 | -2.85 / -2.24 / -1.30 | -0.33 [-0.76, +0.07] / +1.54 [+0.99, +2.10] | -0.28 [-0.66, +0.11] / +1.20 [+0.78, +1.64] |
| 40000 | +0.42 / +0.41 / +0.78 | -2.78 / -2.07 / -1.08 | -0.37 [-0.81, +0.06] / +1.70 [+1.13, +2.29] | -0.31 [-0.73, +0.09] / +1.25 [+0.80, +1.69] |
| 63000 | +0.34 / +0.36 / +0.80 | -2.90 / -1.98 / -0.82 | -0.46 [-0.92, -0.01] / +2.09 [+1.52, +2.67] | -0.27 [-0.68, +0.14] / +1.41 [+0.95, +1.88] |
| 100000 | +0.34 / +0.28 / +0.73 | -2.82 / -2.21 / -1.15 | -0.39 [-0.85, +0.06] / +1.67 [+1.11, +2.27] | -0.32 [-0.70, +0.04] / +1.37 [+0.93, +1.79] |
| 143000 | +0.48 / +0.33 / +0.84 | -2.89 / -2.23 / -1.12 | -0.36 [-0.85, +0.10] / +1.77 [+1.20, +2.38] | -0.41 [-0.78, -0.05] / +1.40 [+0.94, +1.85] |

Onset (first step after which every later checkpoint reads deficit; descriptive): Head − XTail 256, token-standardized 1000.

- Curve collapse [Δ_pass(T)] (c_total, all steps; 16 steps × 281 verbs): CV MSE M_E 1.323, M_Cg 1.289, M_2D 1.305, M_Eg 1.274, M_g 1.285. **CC1** M_E vs M_2D relative gain of M_2D +0.014 [-0.033, +0.041] → **unresolved**; **CC2** vs M_Cg +0.026 [-0.016, +0.053] → **unresolved**; M_E beats step-only by > 2%: **False** (-0.030 [-0.054, -0.012]); diagnostic step beyond E +0.037 [+0.016, +0.062].
  Step offsets in M_Eg (relative to the first step): 128: -0.02, 256: +0.24, 512: +0.86, 1000: +0.44, 2000: +0.90, 3000: +1.02, 4000: +0.77, 6000: +0.52, 10000: +0.84, 16000: +0.84, 25000: +0.83, 40000: +1.08, 63000: +1.11, 100000: +1.10, 143000: +1.23
  Matched cells (|Δ log10 E| < 0.1, steps ≥ 4× apart): 221751 pairs; mean Δ_pass(T) later − earlier +0.335 (median +0.369).
- Curve collapse [Δ_pass(T)] (c_total, steps ≥ 1000; 12 steps × 281 verbs): CV MSE M_E 1.602, M_Cg 1.583, M_2D 1.607, M_Eg 1.574, M_g 1.578. **CC1** M_E vs M_2D relative gain of M_2D -0.003 [-0.045, +0.020] → **unresolved**; **CC2** vs M_Cg +0.012 [-0.028, +0.037] → **unresolved**; M_E beats step-only by > 2%: **False** (-0.015 [-0.043, +0.002]); diagnostic step beyond E +0.017 [+0.001, +0.039].
- Curve collapse [Δ_pass(T)] (c_total, steps ≥ 4000; 9 steps × 281 verbs): CV MSE M_E 1.666, M_Cg 1.646, M_2D 1.656, M_Eg 1.640, M_g 1.652. **CC1** M_E vs M_2D relative gain of M_2D +0.006 [-0.035, +0.034] → **unresolved**; **CC2** vs M_Cg +0.012 [-0.027, +0.042] → **unresolved**; M_E beats step-only by > 2%: **False** (-0.009 [-0.043, +0.013]); diagnostic step beyond E +0.016 [-0.005, +0.040].
- Curve collapse [Δ_pass(T)] (c_marked, all steps; 16 steps × 281 verbs): CV MSE M_E 1.323, M_Cg 1.289, M_2D 1.305, M_Eg 1.274, M_g 1.285. **CC1** M_E vs M_2D relative gain of M_2D +0.014 [-0.033, +0.041] → **unresolved**; **CC2** vs M_Cg +0.026 [-0.016, +0.054] → **unresolved**; M_E beats step-only by > 2%: **False** (-0.030 [-0.054, -0.012]); diagnostic step beyond E +0.037 [+0.017, +0.063].

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

Band means and side parts (positive part = contributes to the deficit):

| step | Δ_pass(T) Head / Tail / XTail | Δ_pass(I) Head / Tail / XTail | Head − XTail parts T / −I | token-std. parts T / −I |
|---|---|---|---|---|
| 64 | +0.06 / -0.02 / +0.02 | +0.03 / +0.04 / +0.02 | +0.04 [-0.07, +0.15] / -0.00 [-0.12, +0.11] | -0.05 [-0.15, +0.06] / -0.01 [-0.13, +0.10] |
| 128 | +0.09 / -0.00 / +0.08 | +0.02 / +0.07 / +0.10 | +0.01 [-0.11, +0.12] / +0.08 [-0.05, +0.21] | -0.08 [-0.18, +0.01] / +0.03 [-0.10, +0.14] |
| 256 | +0.42 / +0.22 / +0.29 | +0.24 / +0.26 / +0.22 | +0.14 [-0.02, +0.28] / -0.02 [-0.18, +0.16] | -0.20 [-0.37, -0.04] / +0.08 [-0.11, +0.27] |
| 512 | +0.81 / +0.56 / +0.92 | -0.54 / +0.49 / +0.62 | -0.11 [-0.41, +0.20] / +1.16 [+0.81, +1.52] | -0.41 [-0.64, -0.18] / -0.07 [-0.34, +0.21] |
| 1000 | -0.12 / +0.39 / +0.88 | -2.70 / -0.28 / +0.71 | -1.01 [-1.42, -0.65] / +3.42 [+2.99, +3.87] | -0.38 [-0.67, -0.08] / +0.51 [+0.18, +0.84] |
| 2000 | +0.17 / +0.61 / +0.50 | -3.15 / -1.34 / +0.13 | -0.32 [-0.77, +0.13] / +3.28 [+2.77, +3.78] | +0.39 [-0.03, +0.82] / +1.33 [+0.92, +1.72] |
| 3000 | +0.65 / +0.75 / +0.93 | -2.82 / -1.30 / +0.26 | -0.29 [-0.73, +0.14] / +3.08 [+2.51, +3.62] | -0.06 [-0.40, +0.30] / +1.63 [+1.21, +2.06] |
| 4000 | +0.09 / +0.21 / +0.44 | -2.92 / -1.72 / -0.33 | -0.35 [-0.77, +0.07] / +2.59 [+2.04, +3.13] | +0.04 [-0.34, +0.42] / +1.28 [+0.85, +1.69] |
| 6000 | +0.30 / +0.34 / +0.55 | -2.93 / -1.96 / -0.43 | -0.25 [-0.69, +0.15] / +2.49 [+1.95, +3.04] | +0.14 [-0.17, +0.45] / +1.58 [+1.16, +1.98] |
| 10000 | +0.39 / +0.44 / +0.76 | -2.70 / -2.01 / -0.54 | -0.37 [-0.82, +0.05] / +2.16 [+1.64, +2.67] | -0.06 [-0.46, +0.34] / +1.66 [+1.21, +2.08] |
| 16000 | +0.27 / +0.31 / +0.67 | -2.97 / -2.02 / -0.57 | -0.41 [-0.89, +0.04] / +2.40 [+1.81, +2.96] | -0.00 [-0.38, +0.39] / +1.56 [+1.09, +2.04] |
| 25000 | +0.15 / +0.30 / +0.62 | -3.00 / -2.04 / -0.71 | -0.48 [-0.94, -0.05] / +2.28 [+1.70, +2.87] | -0.08 [-0.47, +0.29] / +1.47 [+1.03, +1.89] |
| 40000 | +0.26 / +0.38 / +0.82 | -2.92 / -1.86 / -0.67 | -0.56 [-1.01, -0.14] / +2.25 [+1.66, +2.83] | -0.18 [-0.57, +0.23] / +1.30 [+0.84, +1.75] |
| 63000 | +0.14 / +0.44 / +0.90 | -3.05 / -1.96 / -0.62 | -0.76 [-1.21, -0.34] / +2.43 [+1.82, +3.02] | -0.19 [-0.58, +0.20] / +1.40 [+0.88, +1.93] |
| 100000 | -0.05 / +0.15 / +0.45 | -3.36 / -2.56 / -1.34 | -0.49 [-0.95, -0.06] / +2.02 [+1.44, +2.61] | -0.11 [-0.50, +0.29] / +1.43 [+0.92, +1.94] |
| 143000 | +0.15 / +0.31 / +0.65 | -3.22 / -2.36 / -1.07 | -0.51 [-0.97, -0.07] / +2.16 [+1.56, +2.75] | -0.22 [-0.64, +0.22] / +1.47 [+0.98, +1.95] |

Onset (first step after which every later checkpoint reads deficit; descriptive): Head − XTail 512, token-standardized 2000.

- Curve collapse [Δ_pass(T)] (c_total, all steps; 16 steps × 281 verbs): CV MSE M_E 1.263, M_Cg 1.248, M_2D 1.260, M_Eg 1.232, M_g 1.242. **CC1** M_E vs M_2D relative gain of M_2D +0.002 [-0.038, +0.025] → **unresolved**; **CC2** vs M_Cg +0.012 [-0.023, +0.034] → **unresolved**; M_E beats step-only by > 2%: **False** (-0.017 [-0.039, +0.003]); diagnostic step beyond E +0.025 [+0.009, +0.043].
  Step offsets in M_Eg (relative to the first step): 128: +0.03, 256: +0.27, 512: +0.74, 1000: +0.53, 2000: +0.58, 3000: +0.92, 4000: +0.42, 6000: +0.59, 10000: +0.78, 16000: +0.72, 25000: +0.74, 40000: +0.92, 63000: +1.01, 100000: +0.74, 143000: +0.98
  Matched cells (|Δ log10 E| < 0.1, steps ≥ 4× apart): 221751 pairs; mean Δ_pass(T) later − earlier +0.315 (median +0.332).
- Curve collapse [Δ_pass(T)] (c_total, steps ≥ 1000; 12 steps × 281 verbs): CV MSE M_E 1.554, M_Cg 1.548, M_2D 1.565, M_Eg 1.535, M_g 1.541. **CC1** M_E vs M_2D relative gain of M_2D -0.007 [-0.044, +0.012] → **collapse holds**; **CC2** vs M_Cg +0.004 [-0.030, +0.025] → **unresolved**; M_E beats step-only by > 2%: **False** (-0.008 [-0.030, +0.006]); diagnostic step beyond E +0.012 [-0.001, +0.028].
- Curve collapse [Δ_pass(T)] (c_total, steps ≥ 4000; 9 steps × 281 verbs): CV MSE M_E 1.576, M_Cg 1.569, M_2D 1.576, M_Eg 1.559, M_g 1.564. **CC1** M_E vs M_2D relative gain of M_2D +0.000 [-0.036, +0.020] → **unresolved**; **CC2** vs M_Cg +0.005 [-0.029, +0.026] → **unresolved**; M_E beats step-only by > 2%: **False** (-0.008 [-0.035, +0.011]); diagnostic step beyond E +0.011 [-0.002, +0.029].
- Curve collapse [Δ_pass(T)] (c_marked, all steps; 16 steps × 281 verbs): CV MSE M_E 1.263, M_Cg 1.248, M_2D 1.260, M_Eg 1.231, M_g 1.242. **CC1** M_E vs M_2D relative gain of M_2D +0.002 [-0.038, +0.025] → **unresolved**; **CC2** vs M_Cg +0.012 [-0.023, +0.034] → **unresolved**; M_E beats step-only by > 2%: **False** (-0.017 [-0.039, +0.003]); diagnostic step beyond E +0.025 [+0.009, +0.043].

## pythia-6.9b: trajectories (freq)

| step | checks (recon / identity) | L Head / Tail / XTail | ΔL Head − XTail | reading | side | ΔL_tok | reading | side (token-std.) | Δ_pass(I) Head / XTail | Head AUC | D_L |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 3e-05 / 4e-05 | +1.56 / +0.29 / +0.01 | +1.55 [+1.08, +2.03] | deficit | intransitive-side | +0.06 [-0.33, +0.49] | unresolved | no deficit | -0.60 / +0.46 | 0.90 | +0.72 [+0.10, +1.37] |
| 2000 | 4e-05 / 6e-05 | +3.31 / +1.64 / +0.65 | +2.66 [+2.02, +3.31] | deficit | intransitive-side | +1.05 [+0.49, +1.61] | deficit | intransitive-side | -3.56 / -0.31 | 0.94 | +1.45 [+0.36, +2.56] |
| 8000 | 4e-05 / 8e-05 | +3.20 / +2.54 / +1.80 | +1.40 [+0.63, +2.14] | deficit | intransitive-side | +1.04 [+0.41, +1.65] | deficit | intransitive-side | -3.01 / -1.15 | 0.92 | +1.17 [+0.24, +2.13] |
| 23000 | 3e-05 / 4e-05 | +3.16 / +2.51 / +1.63 | +1.53 [+0.81, +2.24] | deficit | intransitive-side | +1.09 [+0.48, +1.71] | deficit | intransitive-side | -2.73 / -0.94 | 0.93 | +1.38 [+0.45, +2.31] |
| 63000 | 3e-05 / 4e-05 | +3.24 / +2.50 / +1.71 | +1.53 [+0.84, +2.23] | deficit | intransitive-side | +1.18 [+0.62, +1.74] | deficit | intransitive-side | -3.11 / -1.23 | 0.94 | +1.30 [+0.43, +2.17] |
| 143000 | 3e-05 / 5e-05 | +3.50 / +2.64 / +1.86 | +1.64 [+0.93, +2.35] | deficit | intransitive-side | +1.04 [+0.47, +1.59] | deficit | intransitive-side | -3.27 / -1.34 | 0.94 | +1.40 [+0.51, +2.29] |

Band means and side parts (positive part = contributes to the deficit):

| step | Δ_pass(T) Head / Tail / XTail | Δ_pass(I) Head / Tail / XTail | Head − XTail parts T / −I | token-std. parts T / −I |
|---|---|---|---|---|
| 512 | +0.96 / +0.69 / +0.48 | -0.60 / +0.39 / +0.46 | +0.48 [+0.15, +0.81] / +1.06 [+0.71, +1.45] | -0.04 [-0.33, +0.27] / +0.10 [-0.18, +0.39] |
| 2000 | -0.25 / -0.11 / +0.34 | -3.56 / -1.75 / -0.31 | -0.59 [-1.07, -0.13] / +3.24 [+2.75, +3.72] | -0.17 [-0.54, +0.18] / +1.22 [+0.77, +1.65] |
| 8000 | +0.19 / +0.23 / +0.65 | -3.01 / -2.31 / -1.15 | -0.47 [-0.99, +0.02] / +1.86 [+1.27, +2.43] | -0.32 [-0.73, +0.08] / +1.36 [+0.86, +1.85] |
| 23000 | +0.43 / +0.44 / +0.69 | -2.73 / -2.07 / -0.94 | -0.26 [-0.73, +0.18] / +1.80 [+1.26, +2.37] | -0.21 [-0.63, +0.19] / +1.30 [+0.84, +1.75] |
| 63000 | +0.12 / +0.16 / +0.48 | -3.11 / -2.34 / -1.23 | -0.36 [-0.83, +0.09] / +1.88 [+1.32, +2.46] | -0.27 [-0.67, +0.12] / +1.45 [+0.99, +1.93] |
| 143000 | +0.23 / +0.13 / +0.52 | -3.27 / -2.51 / -1.34 | -0.29 [-0.77, +0.17] / +1.93 [+1.35, +2.53] | -0.48 [-0.86, -0.11] / +1.51 [+1.07, +1.95] |

Onset (first step after which every later checkpoint reads deficit; descriptive): Head − XTail 512, token-standardized 2000.

- Curve collapse [Δ_pass(T)] (c_total, all steps; 6 steps × 281 verbs): CV MSE M_E 1.666, M_Cg 1.646, M_2D 1.636, M_Eg 1.613, M_g 1.639. **CC1** M_E vs M_2D relative gain of M_2D +0.018 [-0.027, +0.045] → **unresolved**; **CC2** vs M_Cg +0.012 [-0.032, +0.041] → **unresolved**; M_E beats step-only by > 2%: **False** (-0.017 [-0.044, +0.003]); diagnostic step beyond E +0.032 [+0.009, +0.055].
  Step offsets in M_Eg (relative to the first step): 2000: -0.56, 8000: -0.03, 23000: +0.30, 63000: +0.22, 143000: +0.37
  Matched cells (|Δ log10 E| < 0.1, steps ≥ 4× apart): 29870 pairs; mean Δ_pass(T) later − earlier +0.285 (median +0.257).
- Curve collapse [Δ_pass(T)] (c_total, steps ≥ 1000; 5 steps × 281 verbs): CV MSE M_E 1.786, M_Cg 1.749, M_2D 1.755, M_Eg 1.729, M_g 1.749. **CC1** M_E vs M_2D relative gain of M_2D +0.017 [-0.027, +0.046] → **unresolved**; **CC2** vs M_Cg +0.021 [-0.022, +0.052] → **unresolved**; M_E beats step-only by > 2%: **False** (-0.021 [-0.053, +0.001]); diagnostic step beyond E +0.032 [+0.006, +0.060].
- Curve collapse [Δ_pass(T)] (c_total, steps ≥ 4000; 4 steps × 281 verbs): CV MSE M_E 1.733, M_Cg 1.730, M_2D 1.732, M_Eg 1.716, M_g 1.745. **CC1** M_E vs M_2D relative gain of M_2D +0.000 [-0.035, +0.019] → **collapse holds**; **CC2** vs M_Cg +0.002 [-0.032, +0.022] → **unresolved**; M_E beats step-only by > 2%: **False** (+0.007 [-0.030, +0.030]); diagnostic step beyond E +0.009 [-0.005, +0.027].
- Curve collapse [Δ_pass(T)] (c_marked, all steps; 6 steps × 281 verbs): CV MSE M_E 1.666, M_Cg 1.646, M_2D 1.636, M_Eg 1.613, M_g 1.639. **CC1** M_E vs M_2D relative gain of M_2D +0.018 [-0.027, +0.045] → **unresolved**; **CC2** vs M_Cg +0.012 [-0.032, +0.041] → **unresolved**; M_E beats step-only by > 2%: **False** (-0.017 [-0.044, +0.003]); diagnostic step beyond E +0.032 [+0.009, +0.055].

