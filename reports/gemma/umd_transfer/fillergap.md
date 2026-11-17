# Round 5, C8: filler-gap patches with non-DAS verbs as the primary set (fp32)

Spec: `plan.md`, C8. Runs: `run_fillergap.py patch --dtype float32 --target-pool {nondas, all}`; analysis: `analyze_fillergap_r5.py patch` (round-4 statistics and reading rule). D = after T − after I active donors; D_in = coordinate at t_T − at t_I. nofill = "that" / no wh; fill = "what". Primary: kept non-DAS I bases, non-DAS targets, natural gaps and T reference from non-DAS verbs, half-B contexts.

## B6 embedded wh ("I know that / what NAME V-ed")

### Primary, T − I donors (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 0 | -0.04 [-0.14, +0.05] NO RISE | +0.31 [+0.16, +0.48] RISE | +0.20 [+0.10, +0.30] NO RISE | +0.15 [+0.06, +0.24] NO RISE | -0.05 [-0.09, -0.02] NO RISE | +0.38 [+0.23, +0.53] RISE | -1.62 [-2.16, -1.09] not above | +0.07 [-0.02, +0.16] | **mixed / unresolved** |
| 1 | -0.06 [-0.16, +0.05] NO RISE | +0.53 [+0.30, +0.80] RISE | +0.41 [+0.22, +0.62] RISE | +0.31 [+0.16, +0.49] RISE | -0.09 [-0.15, -0.05] unresolved | +0.62 [+0.36, +0.90] RISE | -1.35 [-1.92, -0.79] not above | +0.09 [-0.02, +0.22] | **mixed / unresolved** |
| 2 | -0.07 [-0.17, +0.03] NO RISE | +0.65 [+0.42, +0.92] RISE | +0.53 [+0.33, +0.75] RISE | +0.39 [+0.23, +0.57] RISE | -0.14 [-0.21, -0.08] FALL | +0.77 [+0.51, +1.06] RISE | -1.15 [-1.78, -0.57] not above | +0.12 [-0.01, +0.26] | **mixed / unresolved** |
| 3 | -0.20 [-0.38, -0.05] NO RISE | +0.83 [+0.55, +1.16] RISE | +0.67 [+0.44, +0.92] RISE | +0.42 [+0.25, +0.61] RISE | -0.24 [-0.38, -0.13] FALL | +1.02 [+0.70, +1.37] RISE | -0.80 [-1.53, -0.12] not above | +0.19 [+0.04, +0.36] | **mixed / unresolved** |
| 4 | -0.17 [-0.32, -0.04] NO RISE | +0.92 [+0.63, +1.26] RISE | +0.89 [+0.61, +1.20] RISE | +0.61 [+0.38, +0.89] RISE | -0.27 [-0.39, -0.18] FALL | +1.22 [+0.88, +1.60] RISE | -0.58 [-1.21, -0.03] not above | +0.30 [+0.08, +0.54] | **mixed / unresolved** |
| 5 | -0.32 [-0.52, -0.13] unresolved | +1.11 [+0.73, +1.58] RISE | +0.90 [+0.57, +1.26] RISE | +0.54 [+0.29, +0.84] RISE | -0.35 [-0.49, -0.23] FALL | +1.27 [+0.79, +1.80] RISE | -0.07 [-0.67, +0.46] not above | +0.15 [-0.05, +0.38] | **mixed / unresolved** |
| 8 | -0.47 [-0.67, -0.30] FALL | +1.07 [+0.76, +1.41] RISE | +0.73 [+0.47, +1.02] RISE | +0.35 [+0.18, +0.56] RISE | -0.38 [-0.50, -0.26] FALL | +1.21 [+0.87, +1.61] RISE | +0.31 [-0.22, +0.77] unresolved | +0.14 [-0.10, +0.37] | **mixed / unresolved** |
| 12 | -0.55 [-0.73, -0.38] FALL | +1.11 [+0.78, +1.47] RISE | +0.66 [+0.35, +0.97] RISE | +0.33 [+0.09, +0.57] RISE | -0.33 [-0.46, -0.21] FALL | +1.45 [+1.08, +1.86] RISE | -0.30 [-0.94, +0.28] not above | +0.33 [+0.14, +0.54] | **licensing-like crossover** |
| 15 | -0.90 [-1.16, -0.66] FALL | +1.67 [+1.33, +2.03] RISE | +1.38 [+1.05, +1.73] RISE | +0.90 [+0.57, +1.24] RISE | -0.49 [-0.64, -0.35] FALL | +1.97 [+1.63, +2.32] RISE | -0.11 [-0.64, +0.37] not above | +0.30 [+0.11, +0.48] | **licensing-like crossover** |

### Primary, in range (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 0 | -0.00 [-0.05, +0.04] NO RISE | +0.16 [+0.09, +0.24] NO RISE | +0.12 [+0.07, +0.17] NO RISE | +0.10 [+0.05, +0.15] NO RISE | -0.02 [-0.04, -0.01] NO RISE | +0.21 [+0.14, +0.28] NO RISE | -1.75 [-2.31, -1.21] not above | +0.04 [-0.01, +0.10] | **mixed / unresolved** |
| 1 | -0.04 [-0.10, +0.04] NO RISE | +0.30 [+0.20, +0.41] NO RISE | +0.22 [+0.13, +0.31] NO RISE | +0.18 [+0.10, +0.26] NO RISE | -0.04 [-0.06, -0.02] NO RISE | +0.37 [+0.28, +0.46] unresolved | -1.61 [-2.19, -1.05] not above | +0.07 [-0.01, +0.15] | **mixed / unresolved** |
| 2 | -0.03 [-0.10, +0.04] NO RISE | +0.47 [+0.34, +0.60] unresolved | +0.39 [+0.26, +0.51] unresolved | +0.31 [+0.19, +0.42] RISE | -0.08 [-0.12, -0.05] NO RISE | +0.57 [+0.46, +0.67] RISE | -1.35 [-1.98, -0.76] not above | +0.10 [-0.00, +0.20] | **mixed / unresolved** |
| 3 | -0.13 [-0.25, -0.03] NO RISE | +0.60 [+0.42, +0.79] RISE | +0.45 [+0.32, +0.58] RISE | +0.31 [+0.20, +0.42] RISE | -0.14 [-0.22, -0.07] FALL | +0.77 [+0.60, +0.94] RISE | -1.10 [-1.84, -0.38] not above | +0.17 [+0.05, +0.29] | **mixed / unresolved** |
| 4 | -0.10 [-0.19, -0.01] NO RISE | +0.73 [+0.50, +0.95] RISE | +0.60 [+0.42, +0.77] RISE | +0.45 [+0.29, +0.62] RISE | -0.15 [-0.22, -0.09] FALL | +0.88 [+0.68, +1.06] RISE | -0.97 [-1.60, -0.37] not above | +0.15 [-0.02, +0.34] | **mixed / unresolved** |
| 5 | -0.20 [-0.33, -0.06] NO RISE | +0.96 [+0.69, +1.23] RISE | +0.52 [+0.35, +0.69] RISE | +0.36 [+0.20, +0.53] RISE | -0.16 [-0.24, -0.10] FALL | +0.81 [+0.64, +0.99] RISE | -1.02 [-1.79, -0.31] not above | -0.15 [-0.34, +0.03] | **mixed / unresolved** |
| 8 | -0.37 [-0.56, -0.19] unresolved | +1.31 [+0.91, +1.71] RISE | +1.05 [+0.70, +1.42] RISE | +0.63 [+0.37, +0.94] RISE | -0.42 [-0.57, -0.27] FALL | +1.59 [+1.18, +2.02] RISE | +0.06 [-0.52, +0.57] not above | +0.28 [-0.01, +0.58] | **mixed / unresolved** |
| 12 | -0.49 [-0.64, -0.36] FALL | +1.28 [+0.92, +1.64] RISE | +0.74 [+0.43, +1.03] RISE | +0.39 [+0.14, +0.66] RISE | -0.35 [-0.46, -0.25] FALL | +1.47 [+1.19, +1.77] RISE | -0.25 [-0.90, +0.31] not above | +0.19 [-0.07, +0.41] | **licensing-like crossover** |
| 15 | -0.91 [-1.13, -0.69] FALL | +1.85 [+1.53, +2.20] RISE | +0.73 [+0.58, +0.89] RISE | +0.52 [+0.35, +0.70] RISE | -0.21 [-0.28, -0.14] FALL | +1.10 [+0.97, +1.24] RISE | -0.79 [-1.38, -0.22] not above | -0.75 [-1.04, -0.50] | **licensing-like crossover** |

### Readings by population (sites 6 and 8; strata at every site below)

| Population | verbs | contexts | site | T − I donors | in range |
|---|---:|---:|---:|---|---|
| primary: non-DAS, half B | 16 | 12 | 5 | mixed / unresolved | mixed / unresolved |
| primary: non-DAS, half B | 16 | 12 | 8 | mixed / unresolved | mixed / unresolved |
| primary: non-DAS, half B | 16 | 12 | 12 | licensing-like crossover | licensing-like crossover |
| primary: non-DAS, half B | 16 | 12 | 15 | licensing-like crossover | licensing-like crossover |
| primary: non-DAS, all contexts | 16 | 24 | 5 | mixed / unresolved | mixed / unresolved |
| primary: non-DAS, all contexts | 16 | 24 | 8 | mixed / unresolved | mixed / unresolved |
| primary: non-DAS, all contexts | 16 | 24 | 12 | licensing-like crossover | licensing-like crossover |
| primary: non-DAS, all contexts | 16 | 24 | 15 | licensing-like crossover | licensing-like crossover |
| DAS bases, pooled targets | 7 | 12 | 5 | mixed / unresolved | mixed / unresolved |
| DAS bases, pooled targets | 7 | 12 | 8 | mixed / unresolved | licensing-like crossover |
| DAS bases, pooled targets | 7 | 12 | 12 | mixed / unresolved | mixed / unresolved |
| DAS bases, pooled targets | 7 | 12 | 15 | licensing-like crossover | licensing-like crossover |
| DAS bases, non-DAS targets | 7 | 12 | 5 | mixed / unresolved | mixed / unresolved |
| DAS bases, non-DAS targets | 7 | 12 | 8 | mixed / unresolved | mixed / unresolved |
| DAS bases, non-DAS targets | 7 | 12 | 12 | mixed / unresolved | mixed / unresolved |
| DAS bases, non-DAS targets | 7 | 12 | 15 | licensing-like crossover | licensing-like crossover |
| round-4 population (fp32) | 23 | 12 | 5 | mixed / unresolved | mixed / unresolved |
| round-4 population (fp32) | 23 | 12 | 8 | mixed / unresolved | mixed / unresolved |
| round-4 population (fp32) | 23 | 12 | 12 | licensing-like crossover | licensing-like crossover |
| round-4 population (fp32) | 23 | 12 | 15 | licensing-like crossover | licensing-like crossover |
| multi-token, non-DAS | 7 | 12 | 5 | surface (object next) | mixed / unresolved |
| multi-token, non-DAS | 7 | 12 | 8 | mixed / unresolved | mixed / unresolved |
| multi-token, non-DAS | 7 | 12 | 12 | mixed / unresolved | mixed / unresolved |
| multi-token, non-DAS | 7 | 12 | 15 | mixed / unresolved | licensing-like crossover |

### Frame strata (primary population; in range / T − I donors per site)

| Stratum | contexts | site 0 | site 1 | site 2 | site 3 | site 4 | site 5 | site 8 | site 12 | site 15 |
|---|---|---|---|---|---|---|---|---|---|---|
| forgot | all 6 | mix / mix | mix / mix | mix / mix | mix / mix | mix / mix | mix / mix | LIC / mix | mix / LIC | LIC / LIC |
| forgot | half B | mix / mix | mix / mix | mix / mix | mix / mix | mix / mix | mix / mix | mix / mix | mix / LIC | LIC / LIC |
| heard | all 6 | mix / mix | mix / mix | mix / mix | mix / mix | mix / mix | mix / mix | LIC / mix | mix / mix | LIC / LIC |
| heard | half B | mix / mix | mix / mix | mix / mix | mix / mix | mix / mix | mix / mix | LIC / mix | mix / mix | LIC / LIC |
| know | all 6 | mix / mix | mix / mix | mix / mix | mix / mix | mix / mix | mix / mix | mix / LIC | LIC / mix | LIC / LIC |
| know | half B | mix / mix | mix / mix | mix / mix | mix / mix | mix / mix | mix / mix | mix / mix | LIC / mix | LIC / LIC |
| remember | all 6 | mix / mix | mix / mix | mix / mix | mix / mix | mix / mix | mix / mix | mix / surf | LIC / LIC | LIC / LIC |
| remember | half B | mix / mix | mix / mix | mix / mix | mix / mix | mix / mix | mix / mix | mix / surf | LIC / LIC | LIC / LIC |

LIC = licensing-like crossover, surf = surface (object next), mix = mixed / unresolved.

## B7 matrix wh ("AUX NAME V" / "What AUX NAME V")

### Primary, T − I donors (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 0 | -0.02 [-0.09, +0.06] NO RISE | +0.34 [+0.20, +0.50] RISE | +0.13 [+0.04, +0.22] NO RISE | +0.13 [+0.02, +0.23] NO RISE | -0.00 [-0.11, +0.08] NO RISE | +0.22 [+0.10, +0.34] unresolved | -1.33 [-1.72, -0.93] not above | -0.13 [-0.25, -0.02] | **inconclusive (base-form gate)** |
| 1 | -0.02 [-0.10, +0.04] NO RISE | +0.61 [+0.39, +0.86] RISE | +0.37 [+0.22, +0.55] unresolved | +0.21 [+0.03, +0.38] unresolved | -0.16 [-0.35, -0.03] FALL | +0.42 [+0.23, +0.64] RISE | -1.12 [-1.53, -0.71] not above | -0.19 [-0.33, -0.07] | **mixed / unresolved** |
| 2 | -0.02 [-0.17, +0.09] NO RISE | +0.78 [+0.53, +1.07] RISE | +0.55 [+0.37, +0.77] RISE | +0.33 [+0.11, +0.55] RISE | -0.22 [-0.42, -0.08] FALL | +0.60 [+0.37, +0.85] RISE | -0.90 [-1.32, -0.46] not above | -0.19 [-0.33, -0.07] | **mixed / unresolved** |
| 3 | -0.16 [-0.37, -0.00] NO RISE | +1.16 [+0.80, +1.57] RISE | +0.79 [+0.57, +1.06] RISE | +0.54 [+0.34, +0.76] RISE | -0.25 [-0.41, -0.14] FALL | +0.93 [+0.63, +1.27] RISE | -0.42 [-0.87, +0.06] not above | -0.23 [-0.41, -0.08] | **inconclusive (base-form gate)** |
| 4 | -0.10 [-0.32, +0.04] NO RISE | +1.30 [+0.96, +1.70] RISE | +0.95 [+0.66, +1.25] RISE | +0.77 [+0.53, +1.04] RISE | -0.17 [-0.29, -0.04] FALL | +1.14 [+0.83, +1.48] RISE | -0.37 [-0.77, +0.04] not above | -0.17 [-0.39, +0.05] | **mixed / unresolved** |
| 5 | -0.32 [-0.55, -0.15] unresolved | +1.36 [+0.86, +1.83] RISE | +1.04 [+0.69, +1.43] RISE | +0.73 [+0.48, +1.03] RISE | -0.31 [-0.43, -0.20] FALL | +1.12 [+0.70, +1.54] RISE | +0.22 [-0.24, +0.67] unresolved | -0.24 [-0.40, -0.09] | **mixed / unresolved** |
| 8 | -0.40 [-0.63, -0.21] unresolved | +1.39 [+1.05, +1.74] RISE | +0.94 [+0.69, +1.20] RISE | +0.61 [+0.44, +0.81] RISE | -0.33 [-0.42, -0.24] FALL | +1.23 [+0.92, +1.54] RISE | +0.52 [+0.09, +0.91] unresolved | -0.17 [-0.36, +0.00] | **mixed / unresolved** |
| 12 | -0.30 [-0.43, -0.17] NO RISE | +1.46 [+1.10, +1.82] RISE | +0.89 [+0.61, +1.22] RISE | +0.57 [+0.37, +0.80] RISE | -0.32 [-0.45, -0.21] FALL | +1.41 [+1.08, +1.75] RISE | -0.08 [-0.56, +0.39] not above | -0.04 [-0.17, +0.08] | **mixed / unresolved** |
| 15 | -0.43 [-0.59, -0.29] unresolved | +2.29 [+1.93, +2.64] RISE | +1.85 [+1.52, +2.21] RISE | +1.39 [+1.11, +1.68] RISE | -0.46 [-0.59, -0.36] FALL | +1.97 [+1.67, +2.28] RISE | +0.09 [-0.29, +0.46] not above | -0.32 [-0.48, -0.17] | **mixed / unresolved** |

### Primary, in range (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 0 | -0.00 [-0.02, +0.01] NO RISE | +0.07 [+0.05, +0.10] NO RISE | +0.04 [+0.02, +0.06] NO RISE | +0.03 [+0.01, +0.05] NO RISE | -0.00 [-0.01, +0.00] NO RISE | +0.05 [+0.03, +0.07] NO RISE | -1.44 [-1.83, -1.02] not above | -0.03 [-0.05, -0.01] | **inconclusive (base-form gate)** |
| 1 | +0.00 [-0.04, +0.05] NO RISE | +0.39 [+0.31, +0.48] unresolved | +0.24 [+0.17, +0.32] NO RISE | +0.14 [-0.01, +0.25] NO RISE | -0.10 [-0.25, -0.01] unresolved | +0.28 [+0.17, +0.37] NO RISE | -1.24 [-1.63, -0.84] not above | -0.11 [-0.22, -0.02] | **mixed / unresolved** |
| 2 | -0.08 [-0.27, +0.05] NO RISE | +0.53 [+0.42, +0.64] RISE | +0.35 [+0.26, +0.46] NO RISE | +0.18 [-0.03, +0.33] NO RISE | -0.17 [-0.36, -0.04] FALL | +0.42 [+0.26, +0.56] unresolved | -1.08 [-1.49, -0.63] not above | -0.11 [-0.22, -0.00] | **mixed / unresolved** |
| 3 | -0.11 [-0.25, +0.01] NO RISE | +0.93 [+0.73, +1.17] RISE | +0.58 [+0.46, +0.71] RISE | +0.40 [+0.27, +0.52] RISE | -0.18 [-0.34, -0.08] FALL | +0.79 [+0.62, +0.97] RISE | -0.63 [-1.06, -0.17] not above | -0.14 [-0.30, -0.01] | **inconclusive (base-form gate)** |
| 4 | -0.09 [-0.30, +0.05] NO RISE | +1.13 [+0.91, +1.36] RISE | +0.71 [+0.54, +0.89] RISE | +0.60 [+0.44, +0.75] RISE | -0.11 [-0.19, -0.01] unresolved | +0.88 [+0.69, +1.07] RISE | -0.48 [-0.85, -0.10] not above | -0.25 [-0.45, -0.06] | **mixed / unresolved** |
| 5 | -0.21 [-0.46, -0.02] NO RISE | +1.55 [+1.30, +1.82] RISE | +0.85 [+0.66, +1.05] RISE | +0.66 [+0.49, +0.83] RISE | -0.20 [-0.28, -0.11] FALL | +1.07 [+0.84, +1.30] RISE | -0.13 [-0.58, +0.33] not above | -0.48 [-0.62, -0.32] | **mixed / unresolved** |
| 8 | -0.20 [-0.34, -0.08] NO RISE | +1.68 [+1.39, +1.96] RISE | +1.03 [+0.78, +1.29] RISE | +0.76 [+0.55, +0.98] RISE | -0.28 [-0.35, -0.20] FALL | +1.34 [+1.07, +1.62] RISE | +0.06 [-0.40, +0.50] not above | -0.33 [-0.60, -0.05] | **mixed / unresolved** |
| 12 | -0.27 [-0.38, -0.16] NO RISE | +1.69 [+1.51, +1.87] RISE | +0.99 [+0.79, +1.20] RISE | +0.70 [+0.52, +0.90] RISE | -0.29 [-0.37, -0.23] FALL | +1.44 [+1.24, +1.64] RISE | +0.03 [-0.38, +0.43] not above | -0.25 [-0.39, -0.11] | **mixed / unresolved** |
| 15 | -0.49 [-0.63, -0.36] FALL | +2.39 [+2.15, +2.65] RISE | +0.73 [+0.62, +0.84] RISE | +0.58 [+0.48, +0.69] RISE | -0.15 [-0.19, -0.11] FALL | +0.83 [+0.72, +0.96] RISE | -0.58 [-0.95, -0.23] not above | -1.57 [-1.77, -1.35] | **licensing-like crossover** |

### Readings by population (sites 6 and 8; strata at every site below)

| Population | verbs | contexts | site | T − I donors | in range |
|---|---:|---:|---:|---|---|
| primary: non-DAS, half B | 26 | 12 | 5 | mixed / unresolved | mixed / unresolved |
| primary: non-DAS, half B | 26 | 12 | 8 | mixed / unresolved | mixed / unresolved |
| primary: non-DAS, half B | 26 | 12 | 12 | mixed / unresolved | mixed / unresolved |
| primary: non-DAS, half B | 26 | 12 | 15 | mixed / unresolved | licensing-like crossover |
| primary: non-DAS, all contexts | 26 | 24 | 5 | mixed / unresolved | mixed / unresolved |
| primary: non-DAS, all contexts | 26 | 24 | 8 | mixed / unresolved | mixed / unresolved |
| primary: non-DAS, all contexts | 26 | 24 | 12 | mixed / unresolved | mixed / unresolved |
| primary: non-DAS, all contexts | 26 | 24 | 15 | mixed / unresolved | licensing-like crossover |
| DAS bases, pooled targets | 11 | 12 | 5 | licensing-like crossover | mixed / unresolved |
| DAS bases, pooled targets | 11 | 12 | 8 | mixed / unresolved | mixed / unresolved |
| DAS bases, pooled targets | 11 | 12 | 12 | mixed / unresolved | mixed / unresolved |
| DAS bases, pooled targets | 11 | 12 | 15 | mixed / unresolved | mixed / unresolved |
| DAS bases, non-DAS targets | 11 | 12 | 5 | licensing-like crossover | mixed / unresolved |
| DAS bases, non-DAS targets | 11 | 12 | 8 | mixed / unresolved | mixed / unresolved |
| DAS bases, non-DAS targets | 11 | 12 | 12 | mixed / unresolved | mixed / unresolved |
| DAS bases, non-DAS targets | 11 | 12 | 15 | mixed / unresolved | mixed / unresolved |
| round-4 population (fp32) | 37 | 12 | 5 | mixed / unresolved | mixed / unresolved |
| round-4 population (fp32) | 37 | 12 | 8 | mixed / unresolved | mixed / unresolved |
| round-4 population (fp32) | 37 | 12 | 12 | mixed / unresolved | mixed / unresolved |
| round-4 population (fp32) | 37 | 12 | 15 | mixed / unresolved | licensing-like crossover |
| multi-token, non-DAS | 12 | 12 | 5 | surface (object next) | surface (object next) |
| multi-token, non-DAS | 12 | 12 | 8 | mixed / unresolved | mixed / unresolved |
| multi-token, non-DAS | 12 | 12 | 12 | mixed / unresolved | mixed / unresolved |
| multi-token, non-DAS | 12 | 12 | 15 | mixed / unresolved | licensing-like crossover |

### Frame strata (primary population; in range / T − I donors per site)

| Stratum | contexts | site 0 | site 1 | site 2 | site 3 | site 4 | site 5 | site 8 | site 12 | site 15 |
|---|---|---|---|---|---|---|---|---|---|---|
| Can | all 6 | n/a / n/a | mix / mix | mix / mix | n/a / n/a | mix / mix | mix / surf | surf / surf | mix / mix | mix / mix |
| Can | half B | n/a / n/a | mix / mix | mix / mix | n/a / n/a | mix / mix | mix / mix | mix / surf | mix / mix | mix / mix |
| Did | all 6 | n/a / n/a | mix / mix | mix / mix | n/a / n/a | mix / mix | mix / mix | mix / mix | mix / mix | LIC / LIC |
| Did | half B | n/a / n/a | mix / mix | mix / mix | n/a / n/a | mix / mix | mix / mix | mix / mix | mix / mix | LIC / LIC |
| Will | all 6 | n/a / n/a | mix / mix | mix / mix | n/a / n/a | mix / mix | mix / mix | mix / mix | mix / mix | mix / LIC |
| Will | half B | n/a / n/a | mix / mix | mix / mix | n/a / n/a | mix / mix | mix / mix | mix / mix | mix / mix | mix / mix |
| Would | all 6 | n/a / n/a | mix / mix | mix / mix | n/a / n/a | mix / mix | mix / mix | mix / mix | mix / mix | LIC / LIC |
| Would | half B | n/a / n/a | mix / mix | mix / mix | n/a / n/a | mix / mix | mix / mix | mix / mix | mix / mix | LIC / LIC |

LIC = licensing-like crossover, surf = surface (object next), mix = mixed / unresolved.

