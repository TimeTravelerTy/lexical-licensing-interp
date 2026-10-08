# A4. Robustness of low early IIA

Spec: `round3_plan.md`, A4. Script: `analyze_token_iia.py`.

## Rank sweep (sites 4 and 6)

- Site 4: mean held-out median pc fraction by rank 1: 0.621, 2: 0.640, 4: 0.678; rank rule chooses **rank 1**.
- Site 6: mean held-out median pc fraction by rank 1: 0.720, 2: 0.742, 4: 0.772; rank rule chooses **rank 1**.

| Site | Rank | Metric | Rank 1 | This rank | Paired gain [95% CI] |
|---:|---:|---|---:|---:|---|
| 4 | 2 | iia_cross | 0.446 | 0.470 | +0.024 [+0.011, +0.036] |
| 4 | 2 | pc_frac_median | 0.621 | 0.640 | +0.019 [+0.004, +0.033] |
| 4 | 4 | iia_cross | 0.446 | 0.513 | +0.067 [+0.048, +0.087] |
| 4 | 4 | pc_frac_median | 0.621 | 0.678 | +0.057 [+0.031, +0.085] |
| 6 | 2 | iia_cross | 0.656 | 0.673 | +0.017 [+0.004, +0.029] |
| 6 | 2 | pc_frac_median | 0.720 | 0.742 | +0.022 [+0.007, +0.037] |
| 6 | 4 | iia_cross | 0.656 | 0.700 | +0.044 [+0.022, +0.065] |
| 6 | 4 | pc_frac_median | 0.720 | 0.772 | +0.052 [+0.030, +0.076] |

## Token count, expansion pairs only (primary)

| Site | Direction | Group | Pairs | IIA | Gap fraction | Natural \|M\| |
|---:|---|---|---:|---|---|---|
| 4 | I←T (intransitive base) | single | 6 | 0.52 [0.20, 0.83] | 0.68 [0.50, 0.84] | 3.53 [2.22, 4.50] |
| 4 | I←T (intransitive base) | multi | 15 | 0.54 [0.35, 0.72] | 0.63 [0.51, 0.74] | 2.69 [2.31, 3.09] |
| 4 | I←T (intransitive base) | multi − single | 21 | 0.02 [-0.35, 0.39] | -0.05 [-0.25, 0.16] | -0.84 [-1.92, 0.48] |
| 4 | T←I (transitive base) | single | 6 | 0.40 [0.10, 0.71] | 0.33 [0.24, 0.40] | 1.38 [0.85, 1.86] |
| 4 | T←I (transitive base) | multi | 15 | 0.37 [0.22, 0.53] | 0.40 [0.29, 0.51] | 2.13 [1.84, 2.43] |
| 4 | T←I (transitive base) | multi − single | 21 | -0.03 [-0.40, 0.31] | 0.07 [-0.06, 0.20] | 0.75 [0.17, 1.34] |
| 6 | I←T (intransitive base) | single | 6 | 0.65 [0.34, 0.91] | 0.77 [0.63, 0.90] | 3.53 [2.22, 4.50] |
| 6 | I←T (intransitive base) | multi | 15 | 0.67 [0.50, 0.83] | 0.71 [0.59, 0.80] | 2.69 [2.30, 3.06] |
| 6 | I←T (intransitive base) | multi − single | 21 | 0.02 [-0.29, 0.36] | -0.06 [-0.24, 0.11] | -0.84 [-1.91, 0.49] |
| 6 | T←I (transitive base) | single | 6 | 0.62 [0.41, 0.84] | 0.46 [0.37, 0.55] | 1.38 [0.84, 1.86] |
| 6 | T←I (transitive base) | multi | 15 | 0.73 [0.57, 0.86] | 0.64 [0.53, 0.74] | 2.13 [1.83, 2.42] |
| 6 | T←I (transitive base) | multi − single | 21 | 0.10 [-0.17, 0.36] | 0.17 [0.04, 0.32] | 0.75 [0.19, 1.35] |
| 8 | I←T (intransitive base) | single | 6 | 0.75 [0.43, 0.97] | 0.82 [0.69, 0.93] | 3.53 [2.24, 4.53] |
| 8 | I←T (intransitive base) | multi | 15 | 0.77 [0.63, 0.89] | 0.76 [0.67, 0.86] | 2.69 [2.29, 3.07] |
| 8 | I←T (intransitive base) | multi − single | 21 | 0.02 [-0.24, 0.36] | -0.06 [-0.21, 0.11] | -0.84 [-1.91, 0.46] |
| 8 | T←I (transitive base) | single | 6 | 0.80 [0.68, 0.92] | 0.56 [0.45, 0.65] | 1.38 [0.90, 1.85] |
| 8 | T←I (transitive base) | multi | 15 | 0.84 [0.72, 0.94] | 0.75 [0.65, 0.84] | 2.13 [1.84, 2.43] |
| 8 | T←I (transitive base) | multi − single | 21 | 0.04 [-0.14, 0.20] | 0.19 [0.06, 0.33] | 0.75 [0.20, 1.35] |
| 10 | I←T (intransitive base) | single | 6 | 0.74 [0.43, 0.97] | 0.82 [0.70, 0.92] | 3.53 [2.22, 4.51] |
| 10 | I←T (intransitive base) | multi | 15 | 0.81 [0.67, 0.92] | 0.80 [0.70, 0.89] | 2.69 [2.31, 3.07] |
| 10 | I←T (intransitive base) | multi − single | 21 | 0.07 [-0.21, 0.41] | -0.02 [-0.16, 0.13] | -0.84 [-1.98, 0.51] |
| 10 | T←I (transitive base) | single | 6 | 0.85 [0.75, 0.95] | 0.60 [0.48, 0.69] | 1.38 [0.89, 1.86] |
| 10 | T←I (transitive base) | multi | 15 | 0.89 [0.80, 0.95] | 0.79 [0.69, 0.90] | 2.13 [1.85, 2.42] |
| 10 | T←I (transitive base) | multi − single | 21 | 0.03 [-0.09, 0.16] | 0.19 [0.06, 0.35] | 0.75 [0.18, 1.32] |
| 12 | I←T (intransitive base) | single | 6 | 0.83 [0.65, 0.98] | 0.89 [0.80, 0.97] | 3.53 [2.17, 4.50] |
| 12 | I←T (intransitive base) | multi | 15 | 0.91 [0.80, 0.98] | 0.90 [0.81, 0.99] | 2.69 [2.32, 3.05] |
| 12 | I←T (intransitive base) | multi − single | 21 | 0.07 [-0.11, 0.27] | 0.01 [-0.11, 0.14] | -0.84 [-1.89, 0.52] |
| 12 | T←I (transitive base) | single | 6 | 0.94 [0.91, 0.98] | 0.76 [0.63, 0.89] | 1.38 [0.88, 1.86] |
| 12 | T←I (transitive base) | multi | 15 | 0.96 [0.92, 0.99] | 0.94 [0.82, 1.07] | 2.13 [1.85, 2.42] |
| 12 | T←I (transitive base) | multi − single | 21 | 0.01 [-0.04, 0.06] | 0.18 [-0.00, 0.37] | 0.75 [0.17, 1.33] |
| 14 | I←T (intransitive base) | single | 6 | 0.89 [0.72, 0.99] | 0.98 [0.86, 1.09] | 3.53 [2.17, 4.50] |
| 14 | I←T (intransitive base) | multi | 15 | 0.97 [0.92, 1.00] | 1.00 [0.92, 1.07] | 2.69 [2.30, 3.07] |
| 14 | I←T (intransitive base) | multi − single | 21 | 0.07 [-0.05, 0.25] | 0.02 [-0.12, 0.16] | -0.84 [-1.92, 0.58] |
| 14 | T←I (transitive base) | single | 6 | 0.99 [0.98, 1.00] | 0.86 [0.74, 1.00] | 1.38 [0.88, 1.86] |
| 14 | T←I (transitive base) | multi | 15 | 0.98 [0.97, 1.00] | 0.99 [0.88, 1.12] | 2.13 [1.85, 2.44] |
| 14 | T←I (transitive base) | multi − single | 21 | -0.00 [-0.02, 0.01] | 0.13 [-0.05, 0.31] | 0.75 [0.21, 1.38] |
| 16 | I←T (intransitive base) | single | 6 | 0.94 [0.84, 1.00] | 1.04 [0.92, 1.13] | 3.53 [2.24, 4.50] |
| 16 | I←T (intransitive base) | multi | 15 | 0.98 [0.96, 1.00] | 1.09 [1.03, 1.15] | 2.69 [2.31, 3.08] |
| 16 | I←T (intransitive base) | multi − single | 21 | 0.05 [-0.02, 0.15] | 0.05 [-0.06, 0.18] | -0.84 [-1.92, 0.52] |
| 16 | T←I (transitive base) | single | 6 | 0.98 [0.97, 1.00] | 0.93 [0.80, 1.05] | 1.38 [0.88, 1.86] |
| 16 | T←I (transitive base) | multi | 15 | 0.98 [0.97, 0.99] | 1.02 [0.91, 1.15] | 2.13 [1.84, 2.44] |
| 16 | T←I (transitive base) | multi − single | 21 | -0.00 [-0.02, 0.02] | 0.09 [-0.07, 0.26] | 0.75 [0.20, 1.34] |
| 17 | I←T (intransitive base) | single | 6 | 0.94 [0.83, 1.00] | 1.05 [0.93, 1.12] | 3.53 [2.24, 4.53] |
| 17 | I←T (intransitive base) | multi | 15 | 0.99 [0.98, 1.00] | 1.12 [1.05, 1.18] | 2.69 [2.32, 3.08] |
| 17 | I←T (intransitive base) | multi − single | 21 | 0.05 [-0.01, 0.16] | 0.07 [-0.04, 0.19] | -0.84 [-1.87, 0.47] |
| 17 | T←I (transitive base) | single | 6 | 0.98 [0.97, 1.00] | 0.94 [0.84, 1.05] | 1.38 [0.89, 1.86] |
| 17 | T←I (transitive base) | multi | 15 | 0.98 [0.96, 0.99] | 0.99 [0.89, 1.10] | 2.13 [1.84, 2.44] |
| 17 | T←I (transitive base) | multi − single | 21 | -0.00 [-0.03, 0.02] | 0.05 [-0.10, 0.20] | 0.75 [0.16, 1.36] |

## Token count, all pairs (secondary)

| Site | Direction | Group | Pairs | IIA | Gap fraction | Natural \|M\| |
|---:|---|---|---:|---|---|---|
| 4 | I←T (intransitive base) | single | 14 | 0.54 [0.34, 0.72] | 0.64 [0.53, 0.76] | 3.22 [2.43, 3.83] |
| 4 | I←T (intransitive base) | multi | 15 | 0.54 [0.36, 0.71] | 0.63 [0.51, 0.74] | 2.69 [2.32, 3.05] |
| 4 | I←T (intransitive base) | multi − single | 29 | -0.00 [-0.27, 0.27] | -0.02 [-0.19, 0.13] | -0.53 [-1.25, 0.28] |
| 4 | T←I (transitive base) | single | 14 | 0.34 [0.17, 0.53] | 0.33 [0.27, 0.38] | 1.73 [1.29, 2.17] |
| 4 | T←I (transitive base) | multi | 15 | 0.37 [0.22, 0.52] | 0.40 [0.30, 0.51] | 2.13 [1.85, 2.42] |
| 4 | T←I (transitive base) | multi − single | 29 | 0.02 [-0.21, 0.26] | 0.07 [-0.04, 0.19] | 0.41 [-0.11, 0.93] |
| 6 | I←T (intransitive base) | single | 14 | 0.65 [0.48, 0.82] | 0.75 [0.64, 0.86] | 3.22 [2.53, 3.87] |
| 6 | I←T (intransitive base) | multi | 15 | 0.67 [0.50, 0.82] | 0.71 [0.60, 0.81] | 2.69 [2.31, 3.07] |
| 6 | I←T (intransitive base) | multi − single | 29 | 0.01 [-0.21, 0.24] | -0.05 [-0.20, 0.11] | -0.53 [-1.28, 0.28] |
| 6 | T←I (transitive base) | single | 14 | 0.57 [0.42, 0.73] | 0.46 [0.39, 0.53] | 1.73 [1.33, 2.18] |
| 6 | T←I (transitive base) | multi | 15 | 0.73 [0.57, 0.85] | 0.64 [0.53, 0.74] | 2.13 [1.84, 2.43] |
| 6 | T←I (transitive base) | multi − single | 29 | 0.15 [-0.06, 0.36] | 0.17 [0.04, 0.30] | 0.41 [-0.13, 0.91] |
| 8 | I←T (intransitive base) | single | 14 | 0.74 [0.59, 0.88] | 0.81 [0.71, 0.90] | 3.22 [2.54, 3.83] |
| 8 | I←T (intransitive base) | multi | 15 | 0.77 [0.63, 0.89] | 0.76 [0.65, 0.85] | 2.69 [2.32, 3.08] |
| 8 | I←T (intransitive base) | multi − single | 29 | 0.03 [-0.18, 0.22] | -0.05 [-0.19, 0.09] | -0.53 [-1.28, 0.22] |
| 8 | T←I (transitive base) | single | 14 | 0.77 [0.64, 0.88] | 0.57 [0.48, 0.64] | 1.73 [1.30, 2.15] |
| 8 | T←I (transitive base) | multi | 15 | 0.84 [0.72, 0.94] | 0.75 [0.65, 0.84] | 2.13 [1.85, 2.42] |
| 8 | T←I (transitive base) | multi − single | 29 | 0.07 [-0.10, 0.24] | 0.18 [0.06, 0.31] | 0.41 [-0.11, 0.90] |
| 10 | I←T (intransitive base) | single | 14 | 0.76 [0.60, 0.90] | 0.81 [0.73, 0.90] | 3.22 [2.52, 3.83] |
| 10 | I←T (intransitive base) | multi | 15 | 0.81 [0.67, 0.92] | 0.80 [0.70, 0.89] | 2.69 [2.33, 3.08] |
| 10 | I←T (intransitive base) | multi − single | 29 | 0.05 [-0.16, 0.24] | -0.02 [-0.15, 0.11] | -0.53 [-1.29, 0.27] |
| 10 | T←I (transitive base) | single | 14 | 0.83 [0.67, 0.93] | 0.61 [0.53, 0.68] | 1.73 [1.29, 2.18] |
| 10 | T←I (transitive base) | multi | 15 | 0.89 [0.81, 0.95] | 0.79 [0.68, 0.89] | 2.13 [1.84, 2.44] |
| 10 | T←I (transitive base) | multi − single | 29 | 0.06 [-0.07, 0.23] | 0.18 [0.05, 0.31] | 0.41 [-0.11, 0.95] |
| 12 | I←T (intransitive base) | single | 14 | 0.84 [0.73, 0.93] | 0.88 [0.80, 0.96] | 3.22 [2.54, 3.86] |
| 12 | I←T (intransitive base) | multi | 15 | 0.91 [0.80, 0.98] | 0.90 [0.81, 0.98] | 2.69 [2.32, 3.06] |
| 12 | I←T (intransitive base) | multi − single | 29 | 0.07 [-0.07, 0.21] | 0.02 [-0.09, 0.14] | -0.53 [-1.25, 0.26] |
| 12 | T←I (transitive base) | single | 14 | 0.90 [0.76, 0.98] | 0.76 [0.66, 0.85] | 1.73 [1.30, 2.18] |
| 12 | T←I (transitive base) | multi | 15 | 0.96 [0.92, 0.99] | 0.94 [0.82, 1.07] | 2.13 [1.85, 2.43] |
| 12 | T←I (transitive base) | multi − single | 29 | 0.06 [-0.04, 0.21] | 0.18 [0.03, 0.33] | 0.41 [-0.12, 0.94] |
| 14 | I←T (intransitive base) | single | 14 | 0.91 [0.80, 0.98] | 0.97 [0.89, 1.04] | 3.22 [2.54, 3.83] |
| 14 | I←T (intransitive base) | multi | 15 | 0.97 [0.92, 1.00] | 1.00 [0.92, 1.07] | 2.69 [2.32, 3.09] |
| 14 | I←T (intransitive base) | multi − single | 29 | 0.06 [-0.03, 0.17] | 0.03 [-0.07, 0.14] | -0.53 [-1.25, 0.29] |
| 14 | T←I (transitive base) | single | 14 | 0.93 [0.81, 0.99] | 0.84 [0.76, 0.93] | 1.73 [1.32, 2.20] |
| 14 | T←I (transitive base) | multi | 15 | 0.98 [0.97, 1.00] | 0.99 [0.88, 1.12] | 2.13 [1.84, 2.44] |
| 14 | T←I (transitive base) | multi − single | 29 | 0.06 [-0.01, 0.18] | 0.15 [0.01, 0.30] | 0.41 [-0.15, 0.88] |
| 16 | I←T (intransitive base) | single | 14 | 0.94 [0.87, 0.99] | 1.04 [0.95, 1.12] | 3.22 [2.51, 3.85] |
| 16 | I←T (intransitive base) | multi | 15 | 0.98 [0.96, 1.00] | 1.09 [1.02, 1.16] | 2.69 [2.32, 3.09] |
| 16 | I←T (intransitive base) | multi − single | 29 | 0.04 [-0.01, 0.11] | 0.05 [-0.06, 0.16] | -0.53 [-1.27, 0.28] |
| 16 | T←I (transitive base) | single | 14 | 0.95 [0.89, 0.99] | 0.92 [0.83, 0.99] | 1.73 [1.31, 2.16] |
| 16 | T←I (transitive base) | multi | 15 | 0.98 [0.97, 0.99] | 1.02 [0.91, 1.14] | 2.13 [1.84, 2.44] |
| 16 | T←I (transitive base) | multi − single | 29 | 0.03 [-0.01, 0.10] | 0.11 [-0.03, 0.25] | 0.41 [-0.13, 0.92] |
| 17 | I←T (intransitive base) | single | 14 | 0.95 [0.88, 0.99] | 1.06 [0.98, 1.13] | 3.22 [2.52, 3.83] |
| 17 | I←T (intransitive base) | multi | 15 | 0.99 [0.98, 1.00] | 1.12 [1.05, 1.18] | 2.69 [2.31, 3.06] |
| 17 | I←T (intransitive base) | multi − single | 29 | 0.04 [-0.01, 0.11] | 0.06 [-0.04, 0.16] | -0.53 [-1.32, 0.25] |
| 17 | T←I (transitive base) | single | 14 | 0.97 [0.94, 0.99] | 0.93 [0.85, 1.01] | 1.73 [1.33, 2.16] |
| 17 | T←I (transitive base) | multi | 15 | 0.98 [0.96, 0.99] | 0.99 [0.89, 1.10] | 2.13 [1.85, 2.42] |
| 17 | T←I (transitive base) | multi − single | 29 | 0.01 [-0.02, 0.04] | 0.06 [-0.07, 0.20] | 0.41 [-0.10, 0.90] |
