# Round 5, C8: filler-gap patches with non-DAS verbs as the primary set (fp32)

Spec: `plan.md`, C8. Runs: `run_fillergap.py patch --dtype float32 --target-pool {nondas, all}`; analysis: `analyze_fillergap_r5.py patch` (round-4 statistics and reading rule). D = after T − after I active donors; D_in = coordinate at t_T − at t_I. nofill = "that" / no wh; fill = "what". Primary: kept non-DAS I bases, non-DAS targets, natural gaps and T reference from non-DAS verbs, half-B contexts.

## B6 embedded wh ("I know that / what NAME V-ed")

### Primary, T − I donors (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | -1.42 [-1.82, -1.05] FALL | +2.60 [+2.12, +3.15] RISE | +1.83 [+1.58, +2.11] RISE | +0.96 [+0.66, +1.26] RISE | -0.87 [-1.08, -0.65] FALL | +2.92 [+2.54, +3.33] RISE | +0.52 [+0.05, +0.91] unresolved | +0.32 [+0.05, +0.57] | **mixed / unresolved** |
| 11 | -2.07 [-2.55, -1.60] FALL | +2.82 [+2.36, +3.30] RISE | +0.67 [+0.25, +1.04] RISE | -0.23 [-0.50, +0.06] unresolved | -0.90 [-1.12, -0.65] FALL | +3.41 [+3.07, +3.77] RISE | +0.96 [+0.34, +1.40] above | +0.59 [+0.31, +0.85] | **surface (object next)** |
| 12 | -2.40 [-2.95, -1.89] FALL | +2.97 [+2.49, +3.47] RISE | -0.03 [-0.30, +0.18] NO RISE | -1.29 [-1.66, -1.03] FALL | -1.26 [-1.59, -0.96] FALL | +3.84 [+3.45, +4.28] RISE | +1.52 [+1.19, +1.80] above | +0.87 [+0.56, +1.18] | **surface (object next)** |
| 13 | -3.40 [-3.81, -2.99] FALL | +3.19 [+2.74, +3.64] RISE | -0.04 [-0.46, +0.31] NO RISE | -2.73 [-3.06, -2.43] FALL | -2.68 [-3.21, -2.15] FALL | +4.18 [+3.69, +4.77] RISE | +2.00 [+1.77, +2.22] above | +0.98 [+0.69, +1.33] | **surface (object next)** |

### Primary, in range (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | -1.50 [-1.93, -1.11] FALL | +2.68 [+2.18, +3.23] RISE | +1.63 [+1.41, +1.88] RISE | +0.93 [+0.67, +1.17] RISE | -0.70 [-0.87, -0.51] FALL | +2.67 [+2.28, +3.06] RISE | +0.24 [-0.29, +0.66] unresolved | -0.01 [-0.26, +0.25] | **mixed / unresolved** |
| 11 | -2.26 [-2.72, -1.76] FALL | +2.88 [+2.42, +3.34] RISE | +0.32 [+0.02, +0.59] unresolved | -0.02 [-0.22, +0.18] NO RISE | -0.34 [-0.44, -0.23] FALL | +2.51 [+2.20, +2.83] RISE | +0.10 [-0.65, +0.62] not above | -0.37 [-0.69, -0.06] | **mixed / unresolved** |
| 12 | -2.62 [-3.15, -2.12] FALL | +3.04 [+2.56, +3.52] RISE | -0.21 [-0.36, -0.09] NO RISE | -0.44 [-0.60, -0.31] FALL | -0.23 [-0.33, -0.14] FALL | +2.68 [+2.40, +2.97] RISE | +0.26 [-0.26, +0.68] unresolved | -0.36 [-0.65, -0.04] | **mixed / unresolved** |
| 13 | -3.71 [-4.07, -3.35] FALL | +3.32 [+2.87, +3.75] RISE | -0.18 [-0.35, -0.02] NO RISE | -0.54 [-0.70, -0.38] FALL | -0.36 [-0.52, -0.22] FALL | +2.78 [+2.47, +3.09] RISE | +0.35 [-0.37, +0.91] unresolved | -0.54 [-0.97, -0.09] | **mixed / unresolved** |

### Readings by population (sites 6 and 8; strata at every site below)

| Population | verbs | contexts | site | T − I donors | in range |
|---|---:|---:|---:|---|---|
| primary: non-DAS, half B | 10 | 12 | 9 | mixed / unresolved | mixed / unresolved |
| primary: non-DAS, half B | 10 | 12 | 11 | surface (object next) | mixed / unresolved |
| primary: non-DAS, half B | 10 | 12 | 12 | surface (object next) | mixed / unresolved |
| primary: non-DAS, half B | 10 | 12 | 13 | surface (object next) | mixed / unresolved |
| primary: non-DAS, all contexts | 10 | 24 | 9 | mixed / unresolved | mixed / unresolved |
| primary: non-DAS, all contexts | 10 | 24 | 11 | surface (object next) | mixed / unresolved |
| primary: non-DAS, all contexts | 10 | 24 | 12 | surface (object next) | mixed / unresolved |
| primary: non-DAS, all contexts | 10 | 24 | 13 | surface (object next) | mixed / unresolved |
| DAS bases, pooled targets | 8 | 12 | 9 | surface (object next) | mixed / unresolved |
| DAS bases, pooled targets | 8 | 12 | 11 | surface (object next) | mixed / unresolved |
| DAS bases, pooled targets | 8 | 12 | 12 | surface (object next) | surface (object next) |
| DAS bases, pooled targets | 8 | 12 | 13 | surface (object next) | mixed / unresolved |
| DAS bases, non-DAS targets | 8 | 12 | 9 | surface (object next) | mixed / unresolved |
| DAS bases, non-DAS targets | 8 | 12 | 11 | surface (object next) | mixed / unresolved |
| DAS bases, non-DAS targets | 8 | 12 | 12 | surface (object next) | surface (object next) |
| DAS bases, non-DAS targets | 8 | 12 | 13 | surface (object next) | mixed / unresolved |
| round-4 population (fp32) | 18 | 12 | 9 | surface (object next) | mixed / unresolved |
| round-4 population (fp32) | 18 | 12 | 11 | surface (object next) | mixed / unresolved |
| round-4 population (fp32) | 18 | 12 | 12 | surface (object next) | mixed / unresolved |
| round-4 population (fp32) | 18 | 12 | 13 | surface (object next) | mixed / unresolved |
| multi-token, non-DAS | 8 | 12 | 9 | surface (object next) | mixed / unresolved |
| multi-token, non-DAS | 8 | 12 | 11 | surface (object next) | mixed / unresolved |
| multi-token, non-DAS | 8 | 12 | 12 | surface (object next) | mixed / unresolved |
| multi-token, non-DAS | 8 | 12 | 13 | surface (object next) | surface (object next) |

### Frame strata (primary population; in range / T − I donors per site)

| Stratum | contexts | site 9 | site 11 | site 12 | site 13 |
|---|---|---|---|---|---|
| forgot | all 6 | LIC / mix | mix / surf | mix / surf | mix / surf |
| forgot | half B | LIC / mix | mix / surf | mix / surf | mix / surf |
| heard | all 6 | mix / mix | mix / mix | mix / surf | mix / surf |
| heard | half B | mix / mix | mix / mix | mix / surf | mix / surf |
| know | all 6 | mix / surf | mix / surf | mix / surf | mix / surf |
| know | half B | mix / surf | mix / surf | mix / surf | mix / surf |
| remember | all 6 | mix / surf | mix / surf | mix / surf | mix / surf |
| remember | half B | mix / mix | mix / surf | mix / surf | mix / surf |

LIC = licensing-like crossover, surf = surface (object next), mix = mixed / unresolved.

## B7 matrix wh ("AUX NAME V" / "What AUX NAME V")

### Primary, T − I donors (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | -1.66 [-1.95, -1.39] FALL | +3.31 [+2.97, +3.70] RISE | +1.99 [+1.70, +2.29] RISE | +1.14 [+0.85, +1.45] RISE | -0.85 [-0.99, -0.70] FALL | +3.15 [+2.80, +3.52] RISE | +0.89 [+0.54, +1.24] above | -0.16 [-0.41, +0.09] | **surface (object next)** |
| 11 | -2.25 [-2.61, -1.94] FALL | +3.60 [+3.26, +3.97] RISE | +0.94 [+0.68, +1.19] RISE | -0.21 [-0.50, +0.05] unresolved | -1.15 [-1.40, -0.92] FALL | +4.00 [+3.63, +4.40] RISE | +1.61 [+1.25, +1.94] above | +0.40 [+0.19, +0.62] | **surface (object next)** |
| 12 | -2.52 [-2.89, -2.17] FALL | +3.72 [+3.38, +4.09] RISE | +0.45 [+0.26, +0.63] unresolved | -1.60 [-1.90, -1.31] FALL | -2.05 [-2.36, -1.74] FALL | +4.40 [+4.01, +4.82] RISE | +2.18 [+1.94, +2.43] above | +0.68 [+0.50, +0.88] | **surface (object next)** |
| 13 | -3.23 [-3.59, -2.88] FALL | +3.96 [+3.61, +4.30] RISE | +0.58 [+0.37, +0.76] RISE | -2.88 [-3.27, -2.52] FALL | -3.46 [-3.91, -3.01] FALL | +4.76 [+4.34, +5.18] RISE | +2.58 [+2.33, +2.84] above | +0.80 [+0.62, +0.99] | **surface (object next)** |

### Primary, in range (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | -1.41 [-1.63, -1.20] FALL | +3.41 [+3.08, +3.77] RISE | +1.66 [+1.42, +1.90] RISE | +1.08 [+0.84, +1.33] RISE | -0.58 [-0.68, -0.48] FALL | +2.90 [+2.61, +3.24] RISE | +0.41 [+0.04, +0.79] unresolved | -0.50 [-0.77, -0.24] | **mixed / unresolved** |
| 11 | -2.19 [-2.49, -1.90] FALL | +3.67 [+3.36, +4.02] RISE | +0.30 [+0.09, +0.50] NO RISE | -0.02 [-0.19, +0.14] NO RISE | -0.32 [-0.40, -0.24] FALL | +2.90 [+2.66, +3.17] RISE | +0.42 [+0.01, +0.81] unresolved | -0.77 [-1.05, -0.50] | **mixed / unresolved** |
| 12 | -2.55 [-2.87, -2.24] FALL | +3.80 [+3.50, +4.15] RISE | -0.18 [-0.28, -0.07] NO RISE | -0.45 [-0.56, -0.34] FALL | -0.27 [-0.34, -0.20] FALL | +3.07 [+2.84, +3.33] RISE | +0.59 [+0.25, +0.92] unresolved | -0.73 [-1.01, -0.45] | **mixed / unresolved** |
| 13 | -3.38 [-3.68, -3.09] FALL | +4.08 [+3.78, +4.40] RISE | +0.01 [-0.10, +0.11] NO RISE | -0.39 [-0.54, -0.26] FALL | -0.40 [-0.56, -0.28] FALL | +3.16 [+2.86, +3.49] RISE | +0.67 [+0.22, +1.11] above | -0.92 [-1.32, -0.49] | **surface (object next)** |

### Readings by population (sites 6 and 8; strata at every site below)

| Population | verbs | contexts | site | T − I donors | in range |
|---|---:|---:|---:|---|---|
| primary: non-DAS, half B | 23 | 12 | 9 | surface (object next) | mixed / unresolved |
| primary: non-DAS, half B | 23 | 12 | 11 | surface (object next) | mixed / unresolved |
| primary: non-DAS, half B | 23 | 12 | 12 | surface (object next) | mixed / unresolved |
| primary: non-DAS, half B | 23 | 12 | 13 | surface (object next) | surface (object next) |
| primary: non-DAS, all contexts | 23 | 24 | 9 | surface (object next) | mixed / unresolved |
| primary: non-DAS, all contexts | 23 | 24 | 11 | surface (object next) | mixed / unresolved |
| primary: non-DAS, all contexts | 23 | 24 | 12 | surface (object next) | mixed / unresolved |
| primary: non-DAS, all contexts | 23 | 24 | 13 | surface (object next) | surface (object next) |
| DAS bases, pooled targets | 9 | 12 | 9 | mixed / unresolved | mixed / unresolved |
| DAS bases, pooled targets | 9 | 12 | 11 | surface (object next) | mixed / unresolved |
| DAS bases, pooled targets | 9 | 12 | 12 | surface (object next) | mixed / unresolved |
| DAS bases, pooled targets | 9 | 12 | 13 | surface (object next) | mixed / unresolved |
| DAS bases, non-DAS targets | 9 | 12 | 9 | mixed / unresolved | licensing-like crossover |
| DAS bases, non-DAS targets | 9 | 12 | 11 | surface (object next) | mixed / unresolved |
| DAS bases, non-DAS targets | 9 | 12 | 12 | surface (object next) | mixed / unresolved |
| DAS bases, non-DAS targets | 9 | 12 | 13 | surface (object next) | mixed / unresolved |
| round-4 population (fp32) | 32 | 12 | 9 | surface (object next) | mixed / unresolved |
| round-4 population (fp32) | 32 | 12 | 11 | surface (object next) | mixed / unresolved |
| round-4 population (fp32) | 32 | 12 | 12 | surface (object next) | mixed / unresolved |
| round-4 population (fp32) | 32 | 12 | 13 | surface (object next) | mixed / unresolved |
| multi-token, non-DAS | 13 | 12 | 9 | surface (object next) | mixed / unresolved |
| multi-token, non-DAS | 13 | 12 | 11 | surface (object next) | surface (object next) |
| multi-token, non-DAS | 13 | 12 | 12 | surface (object next) | surface (object next) |
| multi-token, non-DAS | 13 | 12 | 13 | surface (object next) | surface (object next) |

### Frame strata (primary population; in range / T − I donors per site)

| Stratum | contexts | site 9 | site 11 | site 12 | site 13 |
|---|---|---|---|---|---|
| Can | all 6 | surf / surf | surf / surf | surf / surf | surf / surf |
| Can | half B | surf / surf | surf / surf | surf / surf | surf / surf |
| Did | all 6 | surf / surf | mix / surf | surf / surf | surf / surf |
| Did | half B | surf / surf | mix / surf | surf / surf | surf / surf |
| Will | all 6 | LIC / surf | mix / surf | mix / surf | mix / surf |
| Will | half B | LIC / surf | mix / surf | mix / surf | mix / surf |
| Would | all 6 | LIC / surf | mix / surf | mix / surf | surf / surf |
| Would | half B | LIC / surf | mix / surf | mix / surf | surf / surf |

LIC = licensing-like crossover, surf = surface (object next), mix = mixed / unresolved.

