# Round 5, C8: filler-gap patches with non-DAS verbs as the primary set (fp32)

Spec: `plan.md`, C8. Runs: `run_fillergap.py patch --dtype float32 --target-pool {nondas, all}`; analysis: `analyze_fillergap_r5.py patch` (round-4 statistics and reading rule). D = after T − after I active donors; D_in = coordinate at t_T − at t_I. nofill = "that" / no wh; fill = "what". Primary: kept non-DAS I bases, non-DAS targets, natural gaps and T reference from non-DAS verbs, half-B contexts.

## B6 embedded wh ("I know that / what NAME V-ed")

### Primary, T − I donors (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 4 | -0.50 [-0.71, -0.31] FALL | +1.74 [+1.43, +2.05] RISE | +1.61 [+1.34, +1.91] RISE | +0.89 [+0.65, +1.16] RISE | -0.72 [-0.92, -0.56] FALL | +2.52 [+2.14, +2.92] RISE | +0.51 [+0.09, +0.92] above | +0.78 [+0.53, +1.07] | **surface (object next)** |
| 6 | -0.70 [-0.93, -0.49] FALL | +2.04 [+1.73, +2.34] RISE | +1.76 [+1.48, +2.06] RISE | +0.96 [+0.68, +1.24] RISE | -0.80 [-0.99, -0.65] FALL | +2.75 [+2.38, +3.12] RISE | +0.72 [+0.34, +1.08] above | +0.71 [+0.46, +0.99] | **surface (object next)** |
| 8 | -0.78 [-1.01, -0.58] FALL | +2.12 [+1.82, +2.42] RISE | +1.75 [+1.47, +2.04] RISE | +1.01 [+0.75, +1.27] RISE | -0.74 [-0.90, -0.61] FALL | +2.71 [+2.36, +3.06] RISE | +0.68 [+0.34, +1.02] above | +0.58 [+0.33, +0.86] | **surface (object next)** |
| 10 | -1.22 [-1.49, -0.99] FALL | +2.20 [+1.91, +2.50] RISE | +1.39 [+1.14, +1.67] RISE | +0.65 [+0.42, +0.90] RISE | -0.74 [-0.90, -0.61] FALL | +2.75 [+2.41, +3.09] RISE | +0.89 [+0.57, +1.21] above | +0.55 [+0.31, +0.80] | **surface (object next)** |
| 12 | -1.81 [-2.15, -1.51] FALL | +2.42 [+2.14, +2.70] RISE | +1.00 [+0.81, +1.22] RISE | -0.01 [-0.23, +0.21] NO RISE | -1.01 [-1.21, -0.84] FALL | +3.14 [+2.79, +3.49] RISE | +1.37 [+1.06, +1.67] above | +0.72 [+0.49, +0.95] | **surface (object next)** |
| 14 | -2.55 [-2.94, -2.18] FALL | +2.61 [+2.35, +2.87] RISE | +0.12 [-0.06, +0.30] NO RISE | -1.48 [-1.76, -1.21] FALL | -1.60 [-1.93, -1.31] FALL | +3.48 [+3.14, +3.85] RISE | +1.81 [+1.55, +2.08] above | +0.87 [+0.61, +1.12] | **surface (object next)** |
| 16 | -3.12 [-3.56, -2.71] FALL | +2.72 [+2.45, +2.98] RISE | -0.14 [-0.33, +0.04] NO RISE | -2.42 [-2.79, -2.08] FALL | -2.28 [-2.69, -1.92] FALL | +3.56 [+3.21, +3.91] RISE | +2.01 [+1.77, +2.26] above | +0.83 [+0.59, +1.08] | **surface (object next)** |
| 17 | -3.44 [-3.81, -3.04] FALL | +2.76 [+2.48, +3.02] RISE | -0.57 [-0.76, -0.38] FALL | -3.06 [-3.40, -2.72] FALL | -2.49 [-2.86, -2.14] FALL | +3.70 [+3.33, +4.06] RISE | +2.06 [+1.82, +2.31] above | +0.94 [+0.70, +1.18] | **surface (object next)** |

### Primary, in range (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 4 | -0.46 [-0.66, -0.28] FALL | +1.71 [+1.41, +2.00] RISE | +1.44 [+1.21, +1.68] RISE | +0.85 [+0.64, +1.10] RISE | -0.58 [-0.74, -0.46] FALL | +2.34 [+1.98, +2.71] RISE | +0.33 [-0.11, +0.75] unresolved | +0.63 [+0.37, +0.91] | **mixed / unresolved** |
| 6 | -0.60 [-0.79, -0.41] FALL | +1.96 [+1.68, +2.24] RISE | +1.56 [+1.31, +1.83] RISE | +0.93 [+0.69, +1.18] RISE | -0.62 [-0.75, -0.51] FALL | +2.46 [+2.13, +2.81] RISE | +0.47 [+0.08, +0.85] above | +0.50 [+0.26, +0.77] | **surface (object next)** |
| 8 | -0.71 [-0.91, -0.53] FALL | +2.07 [+1.80, +2.34] RISE | +1.51 [+1.29, +1.74] RISE | +0.95 [+0.73, +1.19] RISE | -0.55 [-0.66, -0.46] FALL | +2.36 [+2.04, +2.67] RISE | +0.37 [+0.01, +0.73] unresolved | +0.29 [+0.06, +0.54] | **mixed / unresolved** |
| 10 | -1.16 [-1.38, -0.96] FALL | +2.18 [+1.93, +2.44] RISE | +1.15 [+0.96, +1.36] RISE | +0.68 [+0.49, +0.88] RISE | -0.47 [-0.56, -0.40] FALL | +2.35 [+2.08, +2.63] RISE | +0.37 [+0.03, +0.71] unresolved | +0.17 [-0.04, +0.40] | **mixed / unresolved** |
| 12 | -1.79 [-2.08, -1.54] FALL | +2.41 [+2.15, +2.65] RISE | +0.82 [+0.67, +0.99] RISE | +0.30 [+0.15, +0.49] unresolved | -0.51 [-0.60, -0.43] FALL | +2.69 [+2.42, +2.96] RISE | +0.71 [+0.36, +1.04] above | +0.28 [+0.07, +0.49] | **surface (object next)** |
| 14 | -2.51 [-2.86, -2.19] FALL | +2.63 [+2.40, +2.86] RISE | -0.09 [-0.18, +0.00] NO RISE | -0.50 [-0.63, -0.38] FALL | -0.41 [-0.52, -0.31] FALL | +2.74 [+2.51, +2.98] RISE | +0.76 [+0.40, +1.10] above | +0.11 [-0.11, +0.32] | **surface (object next)** |
| 16 | -3.13 [-3.47, -2.80] FALL | +2.81 [+2.58, +3.03] RISE | -0.26 [-0.31, -0.20] NO RISE | -0.62 [-0.71, -0.53] FALL | -0.36 [-0.46, -0.27] FALL | +2.67 [+2.49, +2.86] RISE | +0.71 [+0.39, +1.03] above | -0.13 [-0.34, +0.08] | **surface (object next)** |
| 17 | -3.58 [-3.86, -3.31] FALL | +2.86 [+2.64, +3.09] RISE | -0.32 [-0.38, -0.26] NO RISE | -0.64 [-0.72, -0.56] FALL | -0.32 [-0.40, -0.25] FALL | +2.65 [+2.47, +2.83] RISE | +0.68 [+0.37, +1.00] above | -0.22 [-0.43, +0.00] | **surface (object next)** |

### Readings by population (sites 6 and 8; strata at every site below)

| Population | verbs | contexts | site | T − I donors | in range |
|---|---:|---:|---:|---|---|
| primary: non-DAS, half B | 28 | 12 | 6 | surface (object next) | surface (object next) |
| primary: non-DAS, half B | 28 | 12 | 8 | surface (object next) | mixed / unresolved |
| primary: non-DAS, all contexts | 28 | 24 | 6 | surface (object next) | surface (object next) |
| primary: non-DAS, all contexts | 28 | 24 | 8 | surface (object next) | mixed / unresolved |
| DAS bases, pooled targets | 11 | 12 | 6 | mixed / unresolved | mixed / unresolved |
| DAS bases, pooled targets | 11 | 12 | 8 | mixed / unresolved | licensing-like crossover |
| DAS bases, non-DAS targets | 11 | 12 | 6 | mixed / unresolved | mixed / unresolved |
| DAS bases, non-DAS targets | 11 | 12 | 8 | mixed / unresolved | licensing-like crossover |
| round-4 population (fp32) | 39 | 12 | 6 | surface (object next) | mixed / unresolved |
| round-4 population (fp32) | 39 | 12 | 8 | surface (object next) | licensing-like crossover |
| multi-token, non-DAS | 25 | 12 | 6 | surface (object next) | surface (object next) |
| multi-token, non-DAS | 25 | 12 | 8 | surface (object next) | surface (object next) |

### Frame strata (primary population; in range / T − I donors per site)

| Stratum | contexts | site 4 | site 6 | site 8 | site 10 | site 12 | site 14 | site 16 | site 17 |
|---|---|---|---|---|---|---|---|---|---|
| forgot | all 6 | mix / surf | mix / surf | mix / surf | LIC / surf | mix / surf | surf / surf | surf / surf | surf / surf |
| forgot | half B | mix / surf | mix / surf | LIC / surf | LIC / surf | mix / surf | surf / surf | surf / surf | surf / surf |
| heard | all 6 | mix / surf | mix / surf | mix / surf | mix / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| heard | half B | mix / surf | mix / surf | mix / surf | mix / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| know | all 6 | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| know | half B | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| remember | all 6 | mix / surf | surf / surf | mix / surf | mix / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| remember | half B | mix / surf | surf / surf | mix / surf | mix / surf | surf / surf | surf / surf | surf / surf | surf / surf |

LIC = licensing-like crossover, surf = surface (object next), mix = mixed / unresolved.

## B7 matrix wh ("AUX NAME V" / "What AUX NAME V")

### Primary, T − I donors (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 4 | -0.63 [-0.86, -0.43] FALL | +2.28 [+2.00, +2.56] RISE | +1.75 [+1.50, +2.01] RISE | +1.06 [+0.82, +1.30] RISE | -0.69 [-0.83, -0.56] FALL | +2.31 [+1.98, +2.69] RISE | +0.27 [-0.12, +0.72] unresolved | +0.02 [-0.20, +0.28] | **mixed / unresolved** |
| 6 | -0.84 [-1.05, -0.64] FALL | +2.55 [+2.29, +2.81] RISE | +1.88 [+1.64, +2.13] RISE | +1.12 [+0.87, +1.35] RISE | -0.76 [-0.88, -0.65] FALL | +2.48 [+2.16, +2.82] RISE | +0.48 [+0.12, +0.87] above | -0.07 [-0.28, +0.16] | **surface (object next)** |
| 8 | -0.95 [-1.16, -0.76] FALL | +2.64 [+2.40, +2.89] RISE | +1.90 [+1.65, +2.15] RISE | +1.12 [+0.86, +1.37] RISE | -0.77 [-0.90, -0.66] FALL | +2.55 [+2.24, +2.86] RISE | +0.53 [+0.21, +0.89] above | -0.09 [-0.31, +0.13] | **surface (object next)** |
| 10 | -1.25 [-1.48, -1.02] FALL | +2.77 [+2.53, +3.02] RISE | +1.58 [+1.36, +1.80] RISE | +0.84 [+0.60, +1.06] RISE | -0.74 [-0.85, -0.63] FALL | +2.68 [+2.38, +2.98] RISE | +0.74 [+0.43, +1.09] above | -0.09 [-0.30, +0.13] | **surface (object next)** |
| 12 | -1.56 [-1.82, -1.31] FALL | +2.95 [+2.72, +3.19] RISE | +1.03 [+0.86, +1.22] RISE | +0.04 [-0.21, +0.27] NO RISE | -0.99 [-1.17, -0.82] FALL | +3.24 [+2.95, +3.53] RISE | +1.32 [+1.01, +1.67] above | +0.29 [+0.11, +0.48] | **surface (object next)** |
| 14 | -2.23 [-2.51, -1.94] FALL | +3.05 [+2.82, +3.28] RISE | +0.04 [-0.16, +0.23] NO RISE | -1.68 [-2.00, -1.39] FALL | -1.72 [-2.02, -1.43] FALL | +3.61 [+3.31, +3.92] RISE | +1.89 [+1.57, +2.23] above | +0.57 [+0.39, +0.76] | **surface (object next)** |
| 16 | -2.67 [-2.97, -2.37] FALL | +3.13 [+2.89, +3.37] RISE | -0.22 [-0.42, -0.02] NO RISE | -2.53 [-2.87, -2.21] FALL | -2.31 [-2.66, -1.98] FALL | +3.77 [+3.45, +4.09] RISE | +2.08 [+1.75, +2.42] above | +0.64 [+0.45, +0.85] | **surface (object next)** |
| 17 | -2.79 [-3.10, -2.48] FALL | +3.15 [+2.89, +3.42] RISE | -0.60 [-0.81, -0.39] FALL | -3.12 [-3.45, -2.79] FALL | -2.53 [-2.88, -2.19] FALL | +3.93 [+3.58, +4.26] RISE | +2.13 [+1.81, +2.47] above | +0.77 [+0.58, +0.98] | **surface (object next)** |

### Primary, in range (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 4 | -0.41 [-0.58, -0.25] FALL | +1.95 [+1.69, +2.22] RISE | +1.49 [+1.29, +1.71] RISE | +0.97 [+0.76, +1.18] RISE | -0.51 [-0.62, -0.42] FALL | +2.02 [+1.73, +2.37] RISE | -0.06 [-0.46, +0.42] not above | +0.07 [-0.13, +0.31] | **licensing-like crossover** |
| 6 | -0.60 [-0.75, -0.45] FALL | +2.28 [+2.05, +2.51] RISE | +1.67 [+1.47, +1.87] RISE | +1.08 [+0.85, +1.30] RISE | -0.59 [-0.69, -0.50] FALL | +2.22 [+1.95, +2.52] RISE | +0.15 [-0.20, +0.57] unresolved | -0.06 [-0.24, +0.16] | **mixed / unresolved** |
| 8 | -0.71 [-0.85, -0.58] FALL | +2.41 [+2.21, +2.62] RISE | +1.61 [+1.43, +1.81] RISE | +1.04 [+0.81, +1.26] RISE | -0.57 [-0.67, -0.49] FALL | +2.20 [+1.94, +2.46] RISE | +0.14 [-0.19, +0.52] not above | -0.21 [-0.41, -0.02] | **licensing-like crossover** |
| 10 | -0.97 [-1.14, -0.80] FALL | +2.58 [+2.39, +2.78] RISE | +1.29 [+1.14, +1.46] RISE | +0.82 [+0.63, +1.00] RISE | -0.48 [-0.56, -0.41] FALL | +2.23 [+2.01, +2.46] RISE | +0.17 [-0.15, +0.54] not above | -0.35 [-0.54, -0.16] | **licensing-like crossover** |
| 12 | -1.30 [-1.49, -1.11] FALL | +2.77 [+2.60, +2.97] RISE | +0.86 [+0.73, +1.00] RISE | +0.34 [+0.17, +0.50] unresolved | -0.52 [-0.62, -0.43] FALL | +2.69 [+2.48, +2.91] RISE | +0.63 [+0.32, +0.98] above | -0.09 [-0.25, +0.09] | **surface (object next)** |
| 14 | -1.97 [-2.18, -1.76] FALL | +3.01 [+2.83, +3.20] RISE | -0.06 [-0.19, +0.05] NO RISE | -0.54 [-0.70, -0.39] FALL | -0.48 [-0.60, -0.37] FALL | +2.86 [+2.65, +3.09] RISE | +0.82 [+0.50, +1.19] above | -0.14 [-0.33, +0.04] | **surface (object next)** |
| 16 | -2.42 [-2.63, -2.22] FALL | +3.17 [+2.99, +3.35] RISE | -0.26 [-0.37, -0.17] NO RISE | -0.65 [-0.79, -0.53] FALL | -0.39 [-0.48, -0.31] FALL | +2.76 [+2.57, +2.97] RISE | +0.73 [+0.44, +1.07] above | -0.41 [-0.59, -0.22] | **surface (object next)** |
| 17 | -2.60 [-2.81, -2.39] FALL | +3.22 [+3.04, +3.41] RISE | -0.41 [-0.53, -0.30] NO RISE | -0.77 [-0.93, -0.63] FALL | -0.36 [-0.45, -0.28] FALL | +2.73 [+2.56, +2.92] RISE | +0.70 [+0.43, +0.99] above | -0.49 [-0.67, -0.31] | **surface (object next)** |

### Readings by population (sites 6 and 8; strata at every site below)

| Population | verbs | contexts | site | T − I donors | in range |
|---|---:|---:|---:|---|---|
| primary: non-DAS, half B | 34 | 12 | 6 | surface (object next) | mixed / unresolved |
| primary: non-DAS, half B | 34 | 12 | 8 | surface (object next) | licensing-like crossover |
| primary: non-DAS, all contexts | 34 | 24 | 6 | mixed / unresolved | licensing-like crossover |
| primary: non-DAS, all contexts | 34 | 24 | 8 | surface (object next) | licensing-like crossover |
| DAS bases, pooled targets | 11 | 12 | 6 | mixed / unresolved | licensing-like crossover |
| DAS bases, pooled targets | 11 | 12 | 8 | mixed / unresolved | licensing-like crossover |
| DAS bases, non-DAS targets | 11 | 12 | 6 | mixed / unresolved | licensing-like crossover |
| DAS bases, non-DAS targets | 11 | 12 | 8 | mixed / unresolved | licensing-like crossover |
| round-4 population (fp32) | 45 | 12 | 6 | mixed / unresolved | licensing-like crossover |
| round-4 population (fp32) | 45 | 12 | 8 | mixed / unresolved | licensing-like crossover |
| multi-token, non-DAS | 25 | 12 | 6 | surface (object next) | mixed / unresolved |
| multi-token, non-DAS | 25 | 12 | 8 | surface (object next) | mixed / unresolved |

### Frame strata (primary population; in range / T − I donors per site)

| Stratum | contexts | site 4 | site 6 | site 8 | site 10 | site 12 | site 14 | site 16 | site 17 |
|---|---|---|---|---|---|---|---|---|---|
| Can | all 6 | mix / surf | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| Can | half B | mix / surf | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| Did | all 6 | mix / mix | LIC / surf | LIC / surf | LIC / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| Did | half B | mix / mix | mix / surf | LIC / surf | LIC / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| Will | all 6 | LIC / LIC | LIC / mix | LIC / mix | LIC / surf | mix / surf | surf / surf | surf / surf | surf / surf |
| Will | half B | LIC / LIC | LIC / mix | LIC / mix | LIC / surf | mix / surf | surf / surf | surf / surf | surf / surf |
| Would | all 6 | LIC / LIC | LIC / LIC | LIC / mix | LIC / surf | mix / surf | mix / surf | mix / surf | mix / surf |
| Would | half B | LIC / LIC | LIC / mix | LIC / mix | LIC / surf | mix / surf | mix / surf | mix / surf | mix / surf |

LIC = licensing-like crossover, surf = surface (object next), mix = mixed / unresolved.

