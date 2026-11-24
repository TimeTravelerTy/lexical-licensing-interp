# Round 5, C8: filler-gap patches with non-DAS verbs as the primary set (fp32)

Spec: `plan.md`, C8. Runs: `run_fillergap.py patch --dtype float32 --target-pool {nondas, all}`; analysis: `analyze_fillergap_r5.py patch` (round-4 statistics and reading rule). D = after T − after I active donors; D_in = coordinate at t_T − at t_I. nofill = "that" / no wh; fill = "what". Primary: kept non-DAS I bases, non-DAS targets, natural gaps and T reference from non-DAS verbs, half-B contexts.

## B6 embedded wh ("I know that / what NAME V-ed")

### Primary, T − I donors (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 0 | -0.13 [-0.28, +0.01] NO RISE | +0.95 [+0.62, +1.33] RISE | +0.57 [+0.37, +0.77] RISE | +0.41 [+0.24, +0.62] RISE | -0.15 [-0.23, -0.09] FALL | +1.12 [+0.80, +1.47] RISE | -0.92 [-1.44, -0.43] not above | +0.17 [+0.03, +0.32] | **mixed / unresolved** |
| 1 | -0.60 [-0.93, -0.31] FALL | +1.06 [+0.68, +1.47] RISE | +0.79 [+0.47, +1.17] RISE | +0.40 [+0.18, +0.66] RISE | -0.40 [-0.57, -0.23] FALL | +1.29 [+0.86, +1.73] RISE | -0.12 [-0.74, +0.43] not above | +0.23 [+0.08, +0.40] | **licensing-like crossover** |
| 2 | -1.05 [-1.53, -0.64] FALL | +1.37 [+0.96, +1.88] RISE | +0.78 [+0.45, +1.20] RISE | +0.29 [+0.02, +0.62] RISE | -0.49 [-0.71, -0.28] FALL | +1.76 [+1.31, +2.28] RISE | -0.02 [-0.73, +0.60] unresolved | +0.39 [+0.17, +0.62] | **mixed / unresolved** |
| 3 | -1.06 [-1.50, -0.66] FALL | +1.50 [+1.14, +1.90] RISE | +1.15 [+0.83, +1.55] RISE | +0.50 [+0.22, +0.85] RISE | -0.65 [-0.87, -0.44] FALL | +1.82 [+1.46, +2.20] RISE | +0.23 [-0.35, +0.76] unresolved | +0.32 [+0.12, +0.53] | **mixed / unresolved** |
| 4 | -1.56 [-2.11, -1.00] FALL | +2.34 [+1.93, +2.82] RISE | +1.45 [+1.03, +1.93] RISE | +0.46 [+0.07, +0.93] RISE | -0.99 [-1.26, -0.74] FALL | +2.89 [+2.47, +3.31] RISE | +1.01 [+0.61, +1.39] above | +0.55 [+0.25, +0.84] | **surface (object next)** |
| 5 | -2.12 [-2.56, -1.67] FALL | +2.58 [+2.10, +3.15] RISE | +1.32 [+0.88, +1.85] RISE | -0.10 [-0.49, +0.36] unresolved | -1.43 [-1.71, -1.14] FALL | +3.27 [+2.76, +3.79] RISE | +1.50 [+1.19, +1.80] above | +0.69 [+0.38, +0.99] | **surface (object next)** |
| 8 | -1.56 [-1.92, -1.15] FALL | +2.57 [+2.14, +3.06] RISE | +1.47 [+1.06, +1.94] RISE | +0.61 [+0.23, +1.06] RISE | -0.86 [-1.01, -0.72] FALL | +2.91 [+2.45, +3.41] RISE | +1.06 [+0.79, +1.33] above | +0.34 [-0.00, +0.66] | **surface (object next)** |
| 12 | -2.37 [-2.74, -1.93] FALL | +2.77 [+2.34, +3.24] RISE | +1.50 [+1.12, +1.95] RISE | +0.43 [+0.04, +0.85] RISE | -1.07 [-1.30, -0.87] FALL | +3.23 [+2.76, +3.74] RISE | +1.23 [+0.96, +1.51] above | +0.47 [+0.14, +0.79] | **surface (object next)** |
| 15 | -2.94 [-3.40, -2.48] FALL | +2.91 [+2.48, +3.37] RISE | +2.35 [+1.92, +2.83] RISE | +0.85 [+0.33, +1.38] RISE | -1.51 [-1.82, -1.22] FALL | +3.67 [+3.21, +4.16] RISE | +1.43 [+1.14, +1.71] above | +0.76 [+0.48, +1.03] | **surface (object next)** |

### Primary, in range (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 0 | -0.06 [-0.16, +0.04] NO RISE | +0.59 [+0.38, +0.82] RISE | +0.35 [+0.24, +0.45] unresolved | +0.26 [+0.16, +0.38] RISE | -0.08 [-0.12, -0.05] NO RISE | +0.71 [+0.54, +0.89] RISE | -1.25 [-1.76, -0.76] not above | +0.12 [-0.00, +0.24] | **mixed / unresolved** |
| 1 | -0.53 [-0.82, -0.24] FALL | +0.87 [+0.60, +1.15] RISE | +0.64 [+0.42, +0.86] RISE | +0.30 [+0.12, +0.49] RISE | -0.34 [-0.47, -0.22] FALL | +1.03 [+0.84, +1.25] RISE | -0.39 [-1.11, +0.28] not above | +0.17 [+0.00, +0.34] | **licensing-like crossover** |
| 2 | -0.78 [-1.22, -0.38] FALL | +1.07 [+0.65, +1.51] RISE | +0.62 [+0.34, +0.96] RISE | +0.21 [-0.04, +0.52] unresolved | -0.41 [-0.59, -0.24] FALL | +1.28 [+0.91, +1.68] RISE | -0.34 [-1.15, +0.41] not above | +0.21 [-0.00, +0.43] | **mixed / unresolved** |
| 3 | -0.95 [-1.35, -0.58] FALL | +1.35 [+1.05, +1.69] RISE | +1.09 [+0.83, +1.39] RISE | +0.46 [+0.21, +0.73] RISE | -0.63 [-0.84, -0.44] FALL | +1.66 [+1.42, +1.91] RISE | +0.18 [-0.46, +0.76] unresolved | +0.30 [+0.08, +0.53] | **mixed / unresolved** |
| 4 | -1.17 [-1.68, -0.65] FALL | +2.21 [+1.79, +2.69] RISE | +1.36 [+0.96, +1.83] RISE | +0.57 [+0.18, +1.02] RISE | -0.79 [-1.02, -0.58] FALL | +2.68 [+2.28, +3.10] RISE | +0.80 [+0.36, +1.23] above | +0.47 [+0.15, +0.77] | **surface (object next)** |
| 5 | -1.68 [-2.08, -1.29] FALL | +2.58 [+2.07, +3.16] RISE | +1.28 [+0.83, +1.81] RISE | +0.28 [-0.08, +0.71] unresolved | -1.00 [-1.24, -0.76] FALL | +3.10 [+2.61, +3.66] RISE | +1.22 [+0.89, +1.57] above | +0.52 [+0.24, +0.79] | **surface (object next)** |
| 8 | -1.53 [-1.85, -1.14] FALL | +2.54 [+2.12, +2.98] RISE | +1.43 [+1.05, +1.88] RISE | +0.72 [+0.37, +1.16] RISE | -0.71 [-0.84, -0.58] FALL | +2.71 [+2.30, +3.17] RISE | +0.83 [+0.55, +1.12] above | +0.17 [-0.11, +0.44] | **surface (object next)** |
| 12 | -2.50 [-2.82, -2.16] FALL | +2.73 [+2.30, +3.21] RISE | +1.25 [+0.95, +1.59] RISE | +0.60 [+0.29, +0.94] RISE | -0.64 [-0.79, -0.51] FALL | +2.60 [+2.27, +2.98] RISE | +0.69 [+0.32, +1.04] above | -0.13 [-0.46, +0.18] | **surface (object next)** |
| 15 | -3.31 [-3.70, -2.93] FALL | +2.81 [+2.39, +3.27] RISE | +1.52 [+1.21, +1.86] RISE | +0.85 [+0.50, +1.22] RISE | -0.67 [-0.83, -0.51] FALL | +2.55 [+2.24, +2.92] RISE | +0.65 [+0.29, +1.01] above | -0.26 [-0.56, +0.04] | **surface (object next)** |

### Readings by population (sites 6 and 8; strata at every site below)

| Population | verbs | contexts | site | T − I donors | in range |
|---|---:|---:|---:|---|---|
| primary: non-DAS, half B | 16 | 12 | 5 | surface (object next) | surface (object next) |
| primary: non-DAS, half B | 16 | 12 | 8 | surface (object next) | surface (object next) |
| primary: non-DAS, half B | 16 | 12 | 12 | surface (object next) | surface (object next) |
| primary: non-DAS, half B | 16 | 12 | 15 | surface (object next) | surface (object next) |
| primary: non-DAS, all contexts | 16 | 24 | 5 | surface (object next) | surface (object next) |
| primary: non-DAS, all contexts | 16 | 24 | 8 | surface (object next) | surface (object next) |
| primary: non-DAS, all contexts | 16 | 24 | 12 | surface (object next) | surface (object next) |
| primary: non-DAS, all contexts | 16 | 24 | 15 | surface (object next) | surface (object next) |
| DAS bases, pooled targets | 7 | 12 | 5 | surface (object next) | surface (object next) |
| DAS bases, pooled targets | 7 | 12 | 8 | surface (object next) | surface (object next) |
| DAS bases, pooled targets | 7 | 12 | 12 | surface (object next) | surface (object next) |
| DAS bases, pooled targets | 7 | 12 | 15 | surface (object next) | surface (object next) |
| DAS bases, non-DAS targets | 7 | 12 | 5 | surface (object next) | surface (object next) |
| DAS bases, non-DAS targets | 7 | 12 | 8 | surface (object next) | surface (object next) |
| DAS bases, non-DAS targets | 7 | 12 | 12 | surface (object next) | surface (object next) |
| DAS bases, non-DAS targets | 7 | 12 | 15 | surface (object next) | surface (object next) |
| round-4 population (fp32) | 23 | 12 | 5 | surface (object next) | surface (object next) |
| round-4 population (fp32) | 23 | 12 | 8 | surface (object next) | surface (object next) |
| round-4 population (fp32) | 23 | 12 | 12 | surface (object next) | surface (object next) |
| round-4 population (fp32) | 23 | 12 | 15 | surface (object next) | surface (object next) |
| multi-token, non-DAS | 7 | 12 | 5 | surface (object next) | surface (object next) |
| multi-token, non-DAS | 7 | 12 | 8 | surface (object next) | surface (object next) |
| multi-token, non-DAS | 7 | 12 | 12 | surface (object next) | surface (object next) |
| multi-token, non-DAS | 7 | 12 | 15 | surface (object next) | surface (object next) |

### Frame strata (primary population; in range / T − I donors per site)

| Stratum | contexts | site 0 | site 1 | site 2 | site 3 | site 4 | site 5 | site 8 | site 12 | site 15 |
|---|---|---|---|---|---|---|---|---|---|---|
| forgot | all 6 | mix / mix | LIC / LIC | mix / mix | LIC / mix | surf / surf | surf / surf | surf / surf | mix / surf | mix / surf |
| forgot | half B | mix / mix | LIC / LIC | mix / mix | mix / mix | surf / surf | surf / surf | surf / surf | mix / surf | mix / surf |
| heard | all 6 | mix / mix | mix / LIC | mix / mix | mix / mix | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| heard | half B | mix / mix | mix / LIC | mix / mix | mix / mix | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| know | all 6 | mix / mix | LIC / LIC | mix / mix | mix / mix | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| know | half B | mix / mix | LIC / LIC | mix / mix | mix / mix | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| remember | all 6 | mix / mix | LIC / mix | LIC / mix | mix / mix | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| remember | half B | mix / mix | LIC / mix | LIC / mix | mix / mix | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf |

LIC = licensing-like crossover, surf = surface (object next), mix = mixed / unresolved.

## B7 matrix wh ("AUX NAME V" / "What AUX NAME V")

### Primary, T − I donors (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 0 | -0.08 [-0.20, +0.03] NO RISE | +0.98 [+0.70, +1.29] RISE | +0.44 [+0.26, +0.61] unresolved | +0.36 [+0.19, +0.52] RISE | -0.07 [-0.20, +0.05] unresolved | +0.83 [+0.57, +1.11] RISE | -0.84 [-1.26, -0.42] not above | -0.15 [-0.29, -0.02] | **mixed / unresolved** |
| 1 | -0.72 [-1.06, -0.44] FALL | +1.32 [+0.88, +1.78] RISE | +0.87 [+0.56, +1.21] RISE | +0.49 [+0.26, +0.75] RISE | -0.39 [-0.57, -0.23] FALL | +1.06 [+0.69, +1.44] RISE | +0.03 [-0.43, +0.49] unresolved | -0.26 [-0.43, -0.13] | **inconclusive (base-form gate)** |
| 2 | -1.37 [-1.89, -0.94] FALL | +1.87 [+1.46, +2.34] RISE | +1.00 [+0.67, +1.39] RISE | +0.59 [+0.34, +0.88] RISE | -0.41 [-0.58, -0.26] FALL | +1.54 [+1.19, +1.94] RISE | +0.41 [-0.04, +0.85] unresolved | -0.33 [-0.51, -0.17] | **inconclusive (base-form gate)** |
| 3 | -1.66 [-2.24, -1.14] FALL | +2.09 [+1.70, +2.49] RISE | +1.52 [+1.18, +1.88] RISE | +0.91 [+0.67, +1.16] RISE | -0.61 [-0.81, -0.44] FALL | +1.66 [+1.33, +1.99] RISE | +0.46 [+0.07, +0.84] unresolved | -0.43 [-0.63, -0.25] | **inconclusive (base-form gate)** |
| 4 | -2.36 [-2.95, -1.77] FALL | +2.75 [+2.30, +3.17] RISE | +1.76 [+1.46, +2.09] RISE | +0.87 [+0.61, +1.14] RISE | -0.89 [-1.08, -0.73] FALL | +2.57 [+2.16, +2.98] RISE | +1.41 [+1.08, +1.74] above | -0.17 [-0.43, +0.08] | **surface (object next)** |
| 5 | -3.50 [-4.15, -2.88] FALL | +2.88 [+2.46, +3.27] RISE | +1.60 [+1.23, +1.99] RISE | +0.28 [-0.05, +0.61] unresolved | -1.32 [-1.56, -1.11] FALL | +3.04 [+2.62, +3.46] RISE | +2.00 [+1.73, +2.28] above | +0.16 [-0.07, +0.38] | **surface (object next)** |
| 8 | -2.16 [-2.56, -1.77] FALL | +3.14 [+2.78, +3.48] RISE | +1.63 [+1.37, +1.89] RISE | +0.86 [+0.62, +1.12] RISE | -0.76 [-0.89, -0.64] FALL | +2.59 [+2.26, +2.92] RISE | +1.34 [+1.08, +1.61] above | -0.55 [-0.81, -0.31] | **surface (object next)** |
| 12 | -2.68 [-3.09, -2.29] FALL | +3.36 [+3.00, +3.69] RISE | +1.68 [+1.42, +1.98] RISE | +0.60 [+0.34, +0.89] RISE | -1.08 [-1.28, -0.90] FALL | +3.22 [+2.86, +3.55] RISE | +1.78 [+1.53, +2.04] above | -0.14 [-0.34, +0.07] | **surface (object next)** |
| 15 | -3.31 [-3.76, -2.86] FALL | +3.81 [+3.49, +4.12] RISE | +2.70 [+2.37, +3.03] RISE | +1.14 [+0.77, +1.51] RISE | -1.56 [-1.80, -1.33] FALL | +4.01 [+3.67, +4.31] RISE | +2.03 [+1.77, +2.29] above | +0.20 [+0.03, +0.37] | **surface (object next)** |

### Primary, in range (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 0 | -0.03 [-0.08, +0.01] NO RISE | +0.39 [+0.29, +0.49] RISE | +0.18 [+0.12, +0.25] NO RISE | +0.14 [+0.08, +0.20] NO RISE | -0.04 [-0.07, -0.02] NO RISE | +0.33 [+0.24, +0.41] RISE | -1.20 [-1.59, -0.79] not above | -0.06 [-0.12, +0.00] | **mixed / unresolved** |
| 1 | -0.42 [-0.65, -0.21] unresolved | +1.00 [+0.72, +1.31] RISE | +0.63 [+0.39, +0.91] RISE | +0.31 [+0.01, +0.61] unresolved | -0.32 [-0.53, -0.14] FALL | +0.80 [+0.56, +1.04] RISE | -0.41 [-0.89, +0.08] not above | -0.21 [-0.41, -0.04] | **inconclusive (base-form gate)** |
| 2 | -0.71 [-1.08, -0.42] FALL | +1.16 [+0.81, +1.55] RISE | +0.56 [+0.26, +0.88] RISE | +0.33 [+0.06, +0.61] RISE | -0.23 [-0.39, -0.06] FALL | +0.94 [+0.67, +1.21] RISE | -0.18 [-0.69, +0.32] not above | -0.23 [-0.47, -0.02] | **inconclusive (base-form gate)** |
| 3 | -1.40 [-1.99, -0.92] FALL | +1.99 [+1.68, +2.32] RISE | +1.41 [+1.13, +1.72] RISE | +0.84 [+0.64, +1.08] RISE | -0.57 [-0.74, -0.42] FALL | +1.57 [+1.31, +1.83] RISE | +0.38 [-0.01, +0.76] unresolved | -0.42 [-0.65, -0.21] | **inconclusive (base-form gate)** |
| 4 | -1.96 [-2.58, -1.40] FALL | +2.83 [+2.40, +3.25] RISE | +1.71 [+1.38, +2.06] RISE | +0.90 [+0.63, +1.20] RISE | -0.80 [-0.99, -0.63] FALL | +2.57 [+2.17, +2.98] RISE | +1.21 [+0.85, +1.58] above | -0.26 [-0.55, +0.01] | **surface (object next)** |
| 5 | -2.94 [-3.63, -2.31] FALL | +3.13 [+2.75, +3.51] RISE | +1.65 [+1.30, +2.03] RISE | +0.57 [+0.25, +0.90] RISE | -1.08 [-1.29, -0.88] FALL | +3.10 [+2.69, +3.49] RISE | +1.70 [+1.37, +2.02] above | -0.03 [-0.28, +0.22] | **surface (object next)** |
| 8 | -1.89 [-2.22, -1.59] FALL | +3.15 [+2.82, +3.49] RISE | +1.62 [+1.39, +1.88] RISE | +0.96 [+0.73, +1.22] RISE | -0.66 [-0.75, -0.56] FALL | +2.55 [+2.24, +2.85] RISE | +1.11 [+0.84, +1.39] above | -0.61 [-0.85, -0.37] | **surface (object next)** |
| 12 | -2.75 [-3.11, -2.42] FALL | +3.34 [+2.99, +3.68] RISE | +1.30 [+1.10, +1.51] RISE | +0.70 [+0.50, +0.92] RISE | -0.59 [-0.71, -0.49] FALL | +2.60 [+2.35, +2.85] RISE | +1.15 [+0.87, +1.44] above | -0.75 [-0.98, -0.50] | **surface (object next)** |
| 15 | -3.65 [-4.04, -3.28] FALL | +3.49 [+3.18, +3.77] RISE | +1.48 [+1.30, +1.69] RISE | +0.95 [+0.75, +1.18] RISE | -0.53 [-0.62, -0.44] FALL | +2.41 [+2.20, +2.63] RISE | +0.98 [+0.69, +1.26] above | -1.07 [-1.30, -0.84] | **surface (object next)** |

### Readings by population (sites 6 and 8; strata at every site below)

| Population | verbs | contexts | site | T − I donors | in range |
|---|---:|---:|---:|---|---|
| primary: non-DAS, half B | 26 | 12 | 5 | surface (object next) | surface (object next) |
| primary: non-DAS, half B | 26 | 12 | 8 | surface (object next) | surface (object next) |
| primary: non-DAS, half B | 26 | 12 | 12 | surface (object next) | surface (object next) |
| primary: non-DAS, half B | 26 | 12 | 15 | surface (object next) | surface (object next) |
| primary: non-DAS, all contexts | 26 | 24 | 5 | surface (object next) | surface (object next) |
| primary: non-DAS, all contexts | 26 | 24 | 8 | surface (object next) | surface (object next) |
| primary: non-DAS, all contexts | 26 | 24 | 12 | surface (object next) | surface (object next) |
| primary: non-DAS, all contexts | 26 | 24 | 15 | surface (object next) | surface (object next) |
| DAS bases, pooled targets | 11 | 12 | 5 | surface (object next) | surface (object next) |
| DAS bases, pooled targets | 11 | 12 | 8 | surface (object next) | surface (object next) |
| DAS bases, pooled targets | 11 | 12 | 12 | surface (object next) | surface (object next) |
| DAS bases, pooled targets | 11 | 12 | 15 | surface (object next) | surface (object next) |
| DAS bases, non-DAS targets | 11 | 12 | 5 | surface (object next) | surface (object next) |
| DAS bases, non-DAS targets | 11 | 12 | 8 | surface (object next) | surface (object next) |
| DAS bases, non-DAS targets | 11 | 12 | 12 | surface (object next) | surface (object next) |
| DAS bases, non-DAS targets | 11 | 12 | 15 | surface (object next) | surface (object next) |
| round-4 population (fp32) | 37 | 12 | 5 | surface (object next) | surface (object next) |
| round-4 population (fp32) | 37 | 12 | 8 | surface (object next) | surface (object next) |
| round-4 population (fp32) | 37 | 12 | 12 | surface (object next) | surface (object next) |
| round-4 population (fp32) | 37 | 12 | 15 | surface (object next) | surface (object next) |
| multi-token, non-DAS | 12 | 12 | 5 | surface (object next) | surface (object next) |
| multi-token, non-DAS | 12 | 12 | 8 | surface (object next) | surface (object next) |
| multi-token, non-DAS | 12 | 12 | 12 | surface (object next) | surface (object next) |
| multi-token, non-DAS | 12 | 12 | 15 | surface (object next) | surface (object next) |

### Frame strata (primary population; in range / T − I donors per site)

| Stratum | contexts | site 0 | site 1 | site 2 | site 3 | site 4 | site 5 | site 8 | site 12 | site 15 |
|---|---|---|---|---|---|---|---|---|---|---|
| Can | all 6 | mix / mix | n/a / n/a | n/a / n/a | n/a / n/a | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| Can | half B | mix / mix | n/a / n/a | n/a / n/a | n/a / n/a | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| Did | all 6 | mix / mix | n/a / n/a | n/a / n/a | n/a / n/a | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| Did | half B | mix / mix | n/a / n/a | n/a / n/a | n/a / n/a | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| Will | all 6 | mix / mix | n/a / n/a | n/a / n/a | n/a / n/a | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| Will | half B | mix / mix | n/a / n/a | n/a / n/a | n/a / n/a | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| Would | all 6 | mix / mix | n/a / n/a | n/a / n/a | n/a / n/a | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| Would | half B | mix / mix | n/a / n/a | n/a / n/a | n/a / n/a | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf |

LIC = licensing-like crossover, surf = surface (object next), mix = mixed / unresolved.

