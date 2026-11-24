# Round 5, C8: filler-gap patches with non-DAS verbs as the primary set (fp32)

Spec: `plan.md`, C8. Runs: `run_fillergap.py patch --dtype float32 --target-pool {nondas, all}`; analysis: `analyze_fillergap_r5.py patch` (round-4 statistics and reading rule). D = after T − after I active donors; D_in = coordinate at t_T − at t_I. nofill = "that" / no wh; fill = "what". Primary: kept non-DAS I bases, non-DAS targets, natural gaps and T reference from non-DAS verbs, half-B contexts.

## B6 embedded wh ("I know that / what NAME V-ed")

### Primary, T − I donors (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 4 | -0.62 [-0.91, -0.38] FALL | +1.80 [+1.45, +2.13] RISE | +1.54 [+1.27, +1.84] RISE | +0.74 [+0.46, +1.04] RISE | -0.81 [-1.05, -0.60] FALL | +2.58 [+2.18, +3.01] RISE | +0.61 [+0.14, +1.05] above | +0.78 [+0.50, +1.07] | **surface (object next)** |
| 6 | -0.85 [-1.14, -0.57] FALL | +2.16 [+1.83, +2.49] RISE | +1.78 [+1.49, +2.09] RISE | +0.84 [+0.55, +1.15] RISE | -0.94 [-1.16, -0.74] FALL | +2.87 [+2.48, +3.26] RISE | +0.88 [+0.48, +1.27] above | +0.71 [+0.45, +0.98] | **surface (object next)** |
| 8 | -0.90 [-1.17, -0.66] FALL | +2.24 [+1.92, +2.55] RISE | +1.77 [+1.50, +2.06] RISE | +0.94 [+0.67, +1.21] RISE | -0.83 [-1.02, -0.68] FALL | +2.84 [+2.47, +3.21] RISE | +0.82 [+0.45, +1.17] above | +0.61 [+0.36, +0.87] | **surface (object next)** |
| 12 | -1.83 [-2.16, -1.53] FALL | +2.52 [+2.24, +2.80] RISE | +1.17 [+0.97, +1.40] RISE | -0.02 [-0.23, +0.22] NO RISE | -1.18 [-1.41, -0.97] FALL | +3.28 [+2.92, +3.65] RISE | +1.49 [+1.17, +1.79] above | +0.76 [+0.53, +0.99] | **surface (object next)** |
| 17 | -3.89 [-4.30, -3.45] FALL | +2.84 [+2.55, +3.10] RISE | -0.23 [-0.41, -0.05] NO RISE | -3.59 [-3.98, -3.19] FALL | -3.36 [-3.78, -2.93] FALL | +3.78 [+3.40, +4.14] RISE | +2.16 [+1.92, +2.41] above | +0.94 [+0.69, +1.19] | **surface (object next)** |

### Primary, in range (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 4 | -0.55 [-0.81, -0.32] FALL | +1.77 [+1.44, +2.08] RISE | +1.38 [+1.15, +1.62] RISE | +0.74 [+0.49, +0.98] RISE | -0.64 [-0.84, -0.49] FALL | +2.44 [+2.06, +2.82] RISE | +0.43 [-0.03, +0.87] unresolved | +0.67 [+0.40, +0.95] | **mixed / unresolved** |
| 6 | -0.74 [-0.98, -0.50] FALL | +2.14 [+1.81, +2.45] RISE | +1.65 [+1.39, +1.93] RISE | +0.88 [+0.61, +1.16] RISE | -0.76 [-0.92, -0.62] FALL | +2.69 [+2.33, +3.08] RISE | +0.70 [+0.28, +1.10] above | +0.56 [+0.29, +0.84] | **surface (object next)** |
| 8 | -0.82 [-1.05, -0.62] FALL | +2.24 [+1.94, +2.52] RISE | +1.58 [+1.35, +1.83] RISE | +0.93 [+0.68, +1.17] RISE | -0.65 [-0.77, -0.55] FALL | +2.59 [+2.26, +2.93] RISE | +0.60 [+0.23, +0.97] above | +0.35 [+0.11, +0.61] | **surface (object next)** |
| 12 | -1.85 [-2.13, -1.58] FALL | +2.58 [+2.31, +2.83] RISE | +0.98 [+0.82, +1.19] RISE | +0.28 [+0.10, +0.48] unresolved | -0.70 [-0.82, -0.60] FALL | +3.06 [+2.76, +3.36] RISE | +1.07 [+0.73, +1.39] above | +0.48 [+0.26, +0.70] | **surface (object next)** |
| 17 | -4.18 [-4.48, -3.88] FALL | +2.98 [+2.76, +3.19] RISE | -0.19 [-0.25, -0.13] NO RISE | -0.75 [-0.85, -0.64] FALL | -0.56 [-0.67, -0.46] FALL | +3.10 [+2.92, +3.29] RISE | +1.13 [+0.83, +1.43] above | +0.12 [-0.09, +0.33] | **surface (object next)** |

### Readings by population (sites 6 and 8; strata at every site below)

| Population | verbs | contexts | site | T − I donors | in range |
|---|---:|---:|---:|---|---|
| primary: non-DAS, half B | 28 | 12 | 8 | surface (object next) | surface (object next) |
| primary: non-DAS, half B | 28 | 12 | 12 | surface (object next) | surface (object next) |
| primary: non-DAS, half B | 28 | 12 | 17 | surface (object next) | surface (object next) |
| primary: non-DAS, all contexts | 28 | 24 | 8 | surface (object next) | surface (object next) |
| primary: non-DAS, all contexts | 28 | 24 | 12 | surface (object next) | surface (object next) |
| primary: non-DAS, all contexts | 28 | 24 | 17 | surface (object next) | surface (object next) |
| DAS bases, pooled targets | 11 | 12 | 8 | mixed / unresolved | licensing-like crossover |
| DAS bases, pooled targets | 11 | 12 | 12 | surface (object next) | surface (object next) |
| DAS bases, pooled targets | 11 | 12 | 17 | surface (object next) | surface (object next) |
| DAS bases, non-DAS targets | 11 | 12 | 8 | mixed / unresolved | licensing-like crossover |
| DAS bases, non-DAS targets | 11 | 12 | 12 | surface (object next) | surface (object next) |
| DAS bases, non-DAS targets | 11 | 12 | 17 | surface (object next) | surface (object next) |
| round-4 population (fp32) | 39 | 12 | 8 | surface (object next) | mixed / unresolved |
| round-4 population (fp32) | 39 | 12 | 12 | surface (object next) | surface (object next) |
| round-4 population (fp32) | 39 | 12 | 17 | surface (object next) | surface (object next) |
| multi-token, non-DAS | 25 | 12 | 8 | surface (object next) | surface (object next) |
| multi-token, non-DAS | 25 | 12 | 12 | surface (object next) | surface (object next) |
| multi-token, non-DAS | 25 | 12 | 17 | surface (object next) | surface (object next) |

### Frame strata (primary population; in range / T − I donors per site)

| Stratum | contexts | site 4 | site 6 | site 8 | site 12 | site 17 |
|---|---|---|---|---|---|---|
| forgot | all 6 | mix / surf | surf / surf | mix / surf | surf / surf | surf / surf |
| forgot | half B | mix / surf | surf / surf | mix / surf | surf / surf | surf / surf |
| heard | all 6 | mix / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| heard | half B | mix / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| know | all 6 | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| know | half B | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| remember | all 6 | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| remember | half B | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf |

LIC = licensing-like crossover, surf = surface (object next), mix = mixed / unresolved.

## B7 matrix wh ("AUX NAME V" / "What AUX NAME V")

### Primary, T − I donors (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 4 | -0.77 [-1.03, -0.55] FALL | +2.34 [+2.05, +2.62] RISE | +1.77 [+1.51, +2.05] RISE | +1.01 [+0.77, +1.26] RISE | -0.75 [-0.93, -0.60] FALL | +2.38 [+2.04, +2.78] RISE | +0.36 [-0.05, +0.85] unresolved | +0.04 [-0.19, +0.31] | **mixed / unresolved** |
| 6 | -1.00 [-1.25, -0.78] FALL | +2.67 [+2.41, +2.92] RISE | +1.96 [+1.71, +2.22] RISE | +1.08 [+0.82, +1.31] RISE | -0.88 [-1.03, -0.73] FALL | +2.60 [+2.27, +2.94] RISE | +0.64 [+0.27, +1.07] above | -0.07 [-0.29, +0.16] | **surface (object next)** |
| 8 | -1.07 [-1.29, -0.86] FALL | +2.74 [+2.49, +2.99] RISE | +1.95 [+1.70, +2.20] RISE | +1.09 [+0.83, +1.34] RISE | -0.86 [-1.00, -0.73] FALL | +2.67 [+2.37, +2.99] RISE | +0.67 [+0.34, +1.06] above | -0.07 [-0.28, +0.16] | **surface (object next)** |
| 12 | -1.69 [-1.96, -1.43] FALL | +3.06 [+2.82, +3.30] RISE | +1.18 [+1.01, +1.37] RISE | +0.01 [-0.24, +0.24] NO RISE | -1.17 [-1.38, -0.98] FALL | +3.42 [+3.11, +3.72] RISE | +1.48 [+1.16, +1.85] above | +0.36 [+0.18, +0.55] | **surface (object next)** |
| 17 | -3.13 [-3.47, -2.80] FALL | +3.26 [+2.98, +3.53] RISE | -0.20 [-0.43, +0.03] NO RISE | -3.59 [-3.94, -3.24] FALL | -3.39 [-3.79, -3.02] FALL | +4.02 [+3.66, +4.37] RISE | +2.25 [+1.93, +2.59] above | +0.76 [+0.57, +0.96] | **surface (object next)** |

### Primary, in range (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 4 | -0.50 [-0.70, -0.32] FALL | +2.03 [+1.76, +2.31] RISE | +1.53 [+1.32, +1.77] RISE | +0.96 [+0.74, +1.18] RISE | -0.58 [-0.71, -0.46] FALL | +2.14 [+1.83, +2.52] RISE | +0.07 [-0.35, +0.58] unresolved | +0.12 [-0.11, +0.37] | **mixed / unresolved** |
| 6 | -0.76 [-0.93, -0.58] FALL | +2.50 [+2.26, +2.73] RISE | +1.80 [+1.60, +2.02] RISE | +1.10 [+0.86, +1.33] RISE | -0.70 [-0.82, -0.59] FALL | +2.44 [+2.14, +2.77] RISE | +0.37 [-0.01, +0.81] unresolved | -0.05 [-0.26, +0.17] | **mixed / unresolved** |
| 8 | -0.85 [-1.02, -0.69] FALL | +2.62 [+2.41, +2.83] RISE | +1.75 [+1.55, +1.96] RISE | +1.07 [+0.84, +1.29] RISE | -0.68 [-0.79, -0.57] FALL | +2.43 [+2.16, +2.71] RISE | +0.36 [+0.01, +0.74] unresolved | -0.19 [-0.40, +0.02] | **mixed / unresolved** |
| 12 | -1.51 [-1.72, -1.31] FALL | +2.98 [+2.81, +3.17] RISE | +1.03 [+0.88, +1.18] RISE | +0.30 [+0.11, +0.48] unresolved | -0.73 [-0.85, -0.62] FALL | +3.13 [+2.90, +3.37] RISE | +1.06 [+0.76, +1.42] above | +0.15 [-0.04, +0.34] | **surface (object next)** |
| 17 | -3.02 [-3.26, -2.79] FALL | +3.36 [+3.17, +3.55] RISE | -0.28 [-0.41, -0.16] NO RISE | -0.85 [-1.04, -0.68] FALL | -0.57 [-0.70, -0.46] FALL | +3.16 [+2.99, +3.34] RISE | +1.11 [+0.86, +1.39] above | -0.20 [-0.38, -0.01] | **surface (object next)** |

### Readings by population (sites 6 and 8; strata at every site below)

| Population | verbs | contexts | site | T − I donors | in range |
|---|---:|---:|---:|---|---|
| primary: non-DAS, half B | 34 | 12 | 8 | surface (object next) | mixed / unresolved |
| primary: non-DAS, half B | 34 | 12 | 12 | surface (object next) | surface (object next) |
| primary: non-DAS, half B | 34 | 12 | 17 | surface (object next) | surface (object next) |
| primary: non-DAS, all contexts | 34 | 24 | 8 | surface (object next) | mixed / unresolved |
| primary: non-DAS, all contexts | 34 | 24 | 12 | surface (object next) | surface (object next) |
| primary: non-DAS, all contexts | 34 | 24 | 17 | surface (object next) | surface (object next) |
| DAS bases, pooled targets | 11 | 12 | 8 | mixed / unresolved | licensing-like crossover |
| DAS bases, pooled targets | 11 | 12 | 12 | surface (object next) | surface (object next) |
| DAS bases, pooled targets | 11 | 12 | 17 | surface (object next) | surface (object next) |
| DAS bases, non-DAS targets | 11 | 12 | 8 | mixed / unresolved | licensing-like crossover |
| DAS bases, non-DAS targets | 11 | 12 | 12 | surface (object next) | surface (object next) |
| DAS bases, non-DAS targets | 11 | 12 | 17 | surface (object next) | surface (object next) |
| round-4 population (fp32) | 45 | 12 | 8 | surface (object next) | mixed / unresolved |
| round-4 population (fp32) | 45 | 12 | 12 | surface (object next) | surface (object next) |
| round-4 population (fp32) | 45 | 12 | 17 | surface (object next) | surface (object next) |
| multi-token, non-DAS | 25 | 12 | 8 | surface (object next) | surface (object next) |
| multi-token, non-DAS | 25 | 12 | 12 | surface (object next) | surface (object next) |
| multi-token, non-DAS | 25 | 12 | 17 | surface (object next) | surface (object next) |

### Frame strata (primary population; in range / T − I donors per site)

| Stratum | contexts | site 4 | site 6 | site 8 | site 12 | site 17 |
|---|---|---|---|---|---|---|
| Can | all 6 | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| Can | half B | surf / surf | surf / surf | surf / surf | surf / surf | surf / surf |
| Did | all 6 | mix / surf | mix / surf | mix / surf | surf / surf | surf / surf |
| Did | half B | mix / surf | mix / surf | mix / surf | surf / surf | surf / surf |
| Will | all 6 | LIC / mix | LIC / mix | LIC / mix | surf / surf | surf / surf |
| Will | half B | LIC / mix | LIC / mix | LIC / mix | surf / surf | surf / surf |
| Would | all 6 | LIC / LIC | LIC / mix | LIC / mix | surf / surf | surf / surf |
| Would | half B | LIC / mix | LIC / mix | LIC / mix | surf / surf | surf / surf |

LIC = licensing-like crossover, surf = surface (object next), mix = mixed / unresolved.

