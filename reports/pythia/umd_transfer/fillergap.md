# Round 5, C8: filler-gap patches with non-DAS verbs as the primary set (fp32)

Spec: `plan.md`, C8. Runs: `run_fillergap.py patch --dtype float32 --target-pool {nondas, all}`; analysis: `analyze_fillergap_r5.py patch` (round-4 statistics and reading rule). D = after T − after I active donors; D_in = coordinate at t_T − at t_I. nofill = "that" / no wh; fill = "what". Primary: kept non-DAS I bases, non-DAS targets, natural gaps and T reference from non-DAS verbs, half-B contexts.

## B6 embedded wh ("I know that / what NAME V-ed")

### Primary, T − I donors (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 4 | -0.15 [-0.24, -0.06] NO RISE | +0.86 [+0.65, +1.07] RISE | +0.98 [+0.80, +1.18] RISE | +0.66 [+0.52, +0.82] RISE | -0.32 [-0.44, -0.22] FALL | +1.53 [+1.22, +1.84] RISE | -0.42 [-0.93, +0.08] not above | +0.67 [+0.46, +0.88] | **mixed / unresolved** |
| 6 | -0.21 [-0.32, -0.10] NO RISE | +1.08 [+0.85, +1.30] RISE | +1.20 [+1.00, +1.43] RISE | +0.84 [+0.66, +1.03] RISE | -0.37 [-0.48, -0.27] FALL | +1.82 [+1.52, +2.14] RISE | -0.22 [-0.69, +0.23] not above | +0.74 [+0.51, +0.99] | **mixed / unresolved** |
| 8 | -0.21 [-0.32, -0.12] NO RISE | +1.19 [+0.96, +1.41] RISE | +1.34 [+1.14, +1.55] RISE | +0.95 [+0.77, +1.13] RISE | -0.39 [-0.50, -0.30] FALL | +1.86 [+1.58, +2.15] RISE | -0.21 [-0.63, +0.20] not above | +0.68 [+0.44, +0.92] | **mixed / unresolved** |
| 12 | -0.43 [-0.57, -0.30] FALL | +1.45 [+1.22, +1.69] RISE | +1.01 [+0.85, +1.19] RISE | +0.68 [+0.53, +0.85] RISE | -0.33 [-0.41, -0.26] FALL | +1.89 [+1.64, +2.14] RISE | -0.18 [-0.60, +0.23] not above | +0.43 [+0.23, +0.65] | **licensing-like crossover** |
| 17 | -1.41 [-1.66, -1.16] FALL | +2.07 [+1.82, +2.31] RISE | -0.09 [-0.28, +0.07] NO RISE | -0.94 [-1.15, -0.73] FALL | -0.85 [-1.03, -0.68] FALL | +2.93 [+2.61, +3.24] RISE | +1.27 [+0.97, +1.57] above | +0.87 [+0.63, +1.11] | **surface (object next)** |

### Primary, in range (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 4 | -0.13 [-0.21, -0.05] NO RISE | +0.80 [+0.61, +0.99] RISE | +0.80 [+0.66, +0.94] RISE | +0.56 [+0.45, +0.68] RISE | -0.24 [-0.33, -0.16] FALL | +1.30 [+1.06, +1.55] RISE | -0.66 [-1.18, -0.15] not above | +0.50 [+0.32, +0.68] | **mixed / unresolved** |
| 6 | -0.17 [-0.27, -0.08] NO RISE | +0.96 [+0.76, +1.13] RISE | +0.90 [+0.77, +1.06] RISE | +0.66 [+0.53, +0.79] RISE | -0.25 [-0.32, -0.18] FALL | +1.40 [+1.20, +1.61] RISE | -0.57 [-1.05, -0.10] not above | +0.45 [+0.24, +0.66] | **mixed / unresolved** |
| 8 | -0.17 [-0.25, -0.09] NO RISE | +1.09 [+0.90, +1.27] RISE | +0.99 [+0.87, +1.11] RISE | +0.72 [+0.61, +0.84] RISE | -0.26 [-0.33, -0.20] FALL | +1.40 [+1.21, +1.57] RISE | -0.57 [-1.01, -0.15] not above | +0.31 [+0.13, +0.52] | **mixed / unresolved** |
| 12 | -0.39 [-0.51, -0.28] unresolved | +1.40 [+1.20, +1.60] RISE | +0.75 [+0.64, +0.87] RISE | +0.53 [+0.43, +0.65] RISE | -0.22 [-0.27, -0.17] FALL | +1.42 [+1.26, +1.57] RISE | -0.55 [-0.96, -0.14] not above | +0.02 [-0.15, +0.21] | **mixed / unresolved** |
| 17 | -1.38 [-1.59, -1.17] FALL | +2.17 [+1.96, +2.38] RISE | -0.00 [-0.11, +0.10] NO RISE | -0.26 [-0.37, -0.15] NO RISE | -0.26 [-0.32, -0.20] FALL | +2.30 [+2.12, +2.49] RISE | +0.34 [-0.00, +0.70] unresolved | +0.13 [-0.07, +0.35] | **mixed / unresolved** |

### Readings by population (sites 6 and 8; strata at every site below)

| Population | verbs | contexts | site | T − I donors | in range |
|---|---:|---:|---:|---|---|
| primary: non-DAS, half B | 28 | 12 | 8 | mixed / unresolved | mixed / unresolved |
| primary: non-DAS, half B | 28 | 12 | 12 | licensing-like crossover | mixed / unresolved |
| primary: non-DAS, half B | 28 | 12 | 17 | surface (object next) | mixed / unresolved |
| primary: non-DAS, all contexts | 28 | 24 | 8 | mixed / unresolved | mixed / unresolved |
| primary: non-DAS, all contexts | 28 | 24 | 12 | licensing-like crossover | mixed / unresolved |
| primary: non-DAS, all contexts | 28 | 24 | 17 | surface (object next) | mixed / unresolved |
| DAS bases, pooled targets | 11 | 12 | 8 | mixed / unresolved | mixed / unresolved |
| DAS bases, pooled targets | 11 | 12 | 12 | mixed / unresolved | mixed / unresolved |
| DAS bases, pooled targets | 11 | 12 | 17 | surface (object next) | mixed / unresolved |
| DAS bases, non-DAS targets | 11 | 12 | 8 | mixed / unresolved | mixed / unresolved |
| DAS bases, non-DAS targets | 11 | 12 | 12 | mixed / unresolved | mixed / unresolved |
| DAS bases, non-DAS targets | 11 | 12 | 17 | surface (object next) | mixed / unresolved |
| round-4 population (fp32) | 39 | 12 | 8 | mixed / unresolved | mixed / unresolved |
| round-4 population (fp32) | 39 | 12 | 12 | mixed / unresolved | mixed / unresolved |
| round-4 population (fp32) | 39 | 12 | 17 | surface (object next) | mixed / unresolved |
| multi-token, non-DAS | 25 | 12 | 8 | mixed / unresolved | mixed / unresolved |
| multi-token, non-DAS | 25 | 12 | 12 | licensing-like crossover | licensing-like crossover |
| multi-token, non-DAS | 25 | 12 | 17 | surface (object next) | mixed / unresolved |

### Frame strata (primary population; in range / T − I donors per site)

| Stratum | contexts | site 4 | site 6 | site 8 | site 12 | site 17 |
|---|---|---|---|---|---|---|
| forgot | all 6 | mix / mix | mix / mix | mix / mix | mix / LIC | mix / surf |
| forgot | half B | mix / mix | mix / mix | mix / mix | mix / LIC | mix / surf |
| heard | all 6 | mix / mix | mix / mix | mix / mix | mix / mix | mix / surf |
| heard | half B | mix / mix | mix / mix | mix / mix | mix / mix | mix / surf |
| know | all 6 | mix / mix | mix / mix | mix / mix | mix / LIC | mix / surf |
| know | half B | mix / mix | mix / mix | mix / mix | LIC / LIC | mix / surf |
| remember | all 6 | mix / mix | mix / mix | mix / mix | LIC / LIC | mix / surf |
| remember | half B | mix / mix | mix / mix | mix / mix | LIC / LIC | mix / surf |

LIC = licensing-like crossover, surf = surface (object next), mix = mixed / unresolved.

## B7 matrix wh ("AUX NAME V" / "What AUX NAME V")

### Primary, T − I donors (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 4 | -0.12 [-0.25, +0.01] NO RISE | +1.27 [+1.02, +1.54] RISE | +1.09 [+0.88, +1.32] RISE | +0.75 [+0.57, +0.93] RISE | -0.34 [-0.44, -0.25] FALL | +1.42 [+1.14, +1.75] RISE | -0.54 [-1.00, -0.01] not above | +0.16 [-0.01, +0.36] | **mixed / unresolved** |
| 6 | -0.28 [-0.40, -0.16] FALL | +1.56 [+1.31, +1.81] RISE | +1.32 [+1.14, +1.52] RISE | +0.95 [+0.77, +1.13] RISE | -0.38 [-0.46, -0.30] FALL | +1.58 [+1.32, +1.85] RISE | -0.45 [-0.84, -0.02] not above | +0.03 [-0.14, +0.21] | **licensing-like crossover** |
| 8 | -0.30 [-0.42, -0.20] FALL | +1.66 [+1.42, +1.91] RISE | +1.44 [+1.24, +1.66] RISE | +1.02 [+0.82, +1.22] RISE | -0.42 [-0.51, -0.34] FALL | +1.68 [+1.43, +1.94] RISE | -0.41 [-0.78, -0.00] not above | +0.02 [-0.16, +0.21] | **licensing-like crossover** |
| 12 | -0.28 [-0.41, -0.16] FALL | +1.90 [+1.68, +2.12] RISE | +1.08 [+0.92, +1.25] RISE | +0.72 [+0.57, +0.88] RISE | -0.35 [-0.43, -0.28] FALL | +1.80 [+1.57, +2.03] RISE | -0.37 [-0.74, +0.05] not above | -0.10 [-0.26, +0.06] | **licensing-like crossover** |
| 17 | -0.87 [-1.04, -0.70] FALL | +2.43 [+2.21, +2.67] RISE | -0.17 [-0.37, +0.04] NO RISE | -1.05 [-1.26, -0.85] FALL | -0.88 [-1.06, -0.72] FALL | +3.12 [+2.81, +3.42] RISE | +1.32 [+0.98, +1.68] above | +0.68 [+0.51, +0.87] | **surface (object next)** |

### Primary, in range (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 4 | -0.05 [-0.16, +0.04] NO RISE | +0.95 [+0.75, +1.17] RISE | +0.80 [+0.66, +0.97] RISE | +0.59 [+0.46, +0.73] RISE | -0.22 [-0.28, -0.15] FALL | +1.09 [+0.87, +1.37] RISE | -0.94 [-1.39, -0.44] not above | +0.14 [-0.01, +0.31] | **mixed / unresolved** |
| 6 | -0.17 [-0.25, -0.09] NO RISE | +1.14 [+0.96, +1.33] RISE | +0.97 [+0.84, +1.09] RISE | +0.73 [+0.59, +0.86] RISE | -0.24 [-0.30, -0.18] FALL | +1.16 [+0.99, +1.35] RISE | -0.89 [-1.28, -0.48] not above | +0.02 [-0.11, +0.16] | **mixed / unresolved** |
| 8 | -0.20 [-0.27, -0.13] NO RISE | +1.27 [+1.09, +1.43] RISE | +0.99 [+0.87, +1.11] RISE | +0.73 [+0.60, +0.85] RISE | -0.26 [-0.32, -0.20] FALL | +1.19 [+1.03, +1.37] RISE | -0.86 [-1.24, -0.46] not above | -0.07 [-0.21, +0.07] | **mixed / unresolved** |
| 12 | -0.20 [-0.30, -0.11] unresolved | +1.64 [+1.49, +1.80] RISE | +0.78 [+0.68, +0.89] RISE | +0.56 [+0.45, +0.66] RISE | -0.23 [-0.28, -0.18] FALL | +1.32 [+1.17, +1.47] RISE | -0.74 [-1.11, -0.35] not above | -0.32 [-0.45, -0.19] | **mixed / unresolved** |
| 17 | -0.74 [-0.87, -0.61] FALL | +2.41 [+2.26, +2.56] RISE | -0.07 [-0.19, +0.04] NO RISE | -0.33 [-0.45, -0.22] unresolved | -0.26 [-0.33, -0.19] FALL | +2.29 [+2.14, +2.45] RISE | +0.26 [-0.06, +0.61] not above | -0.12 [-0.25, +0.02] | **mixed / unresolved** |

### Readings by population (sites 6 and 8; strata at every site below)

| Population | verbs | contexts | site | T − I donors | in range |
|---|---:|---:|---:|---|---|
| primary: non-DAS, half B | 34 | 12 | 8 | licensing-like crossover | mixed / unresolved |
| primary: non-DAS, half B | 34 | 12 | 12 | licensing-like crossover | mixed / unresolved |
| primary: non-DAS, half B | 34 | 12 | 17 | surface (object next) | mixed / unresolved |
| primary: non-DAS, all contexts | 34 | 24 | 8 | licensing-like crossover | mixed / unresolved |
| primary: non-DAS, all contexts | 34 | 24 | 12 | licensing-like crossover | mixed / unresolved |
| primary: non-DAS, all contexts | 34 | 24 | 17 | surface (object next) | mixed / unresolved |
| DAS bases, pooled targets | 11 | 12 | 8 | mixed / unresolved | mixed / unresolved |
| DAS bases, pooled targets | 11 | 12 | 12 | mixed / unresolved | mixed / unresolved |
| DAS bases, pooled targets | 11 | 12 | 17 | surface (object next) | surface (object next) |
| DAS bases, non-DAS targets | 11 | 12 | 8 | mixed / unresolved | mixed / unresolved |
| DAS bases, non-DAS targets | 11 | 12 | 12 | mixed / unresolved | mixed / unresolved |
| DAS bases, non-DAS targets | 11 | 12 | 17 | surface (object next) | mixed / unresolved |
| round-4 population (fp32) | 45 | 12 | 8 | mixed / unresolved | mixed / unresolved |
| round-4 population (fp32) | 45 | 12 | 12 | mixed / unresolved | mixed / unresolved |
| round-4 population (fp32) | 45 | 12 | 17 | surface (object next) | mixed / unresolved |
| multi-token, non-DAS | 25 | 12 | 8 | licensing-like crossover | mixed / unresolved |
| multi-token, non-DAS | 25 | 12 | 12 | licensing-like crossover | mixed / unresolved |
| multi-token, non-DAS | 25 | 12 | 17 | surface (object next) | mixed / unresolved |

### Frame strata (primary population; in range / T − I donors per site)

| Stratum | contexts | site 4 | site 6 | site 8 | site 12 | site 17 |
|---|---|---|---|---|---|---|
| Can | all 6 | mix / mix | mix / LIC | mix / LIC | mix / LIC | surf / surf |
| Can | half B | mix / mix | mix / LIC | mix / LIC | mix / mix | surf / surf |
| Did | all 6 | mix / mix | mix / mix | mix / mix | mix / mix | mix / surf |
| Did | half B | mix / mix | mix / mix | mix / mix | mix / mix | mix / surf |
| Will | all 6 | mix / mix | LIC / LIC | LIC / LIC | LIC / LIC | mix / surf |
| Will | half B | mix / mix | LIC / LIC | LIC / LIC | LIC / LIC | mix / surf |
| Would | all 6 | mix / mix | mix / mix | mix / LIC | mix / LIC | mix / surf |
| Would | half B | mix / mix | mix / mix | mix / LIC | mix / LIC | mix / surf |

LIC = licensing-like crossover, surf = surface (object next), mix = mixed / unresolved.

