# Round 5, C8: filler-gap patches with non-DAS verbs as the primary set (fp32)

Spec: `plan.md`, C8. Runs: `run_fillergap.py patch --dtype float32 --target-pool {nondas, all}`; analysis: `analyze_fillergap_r5.py patch` (round-4 statistics and reading rule). D = after T − after I active donors; D_in = coordinate at t_T − at t_I. nofill = "that" / no wh; fill = "what". Primary: kept non-DAS I bases, non-DAS targets, natural gaps and T reference from non-DAS verbs, half-B contexts.

## B6 embedded wh ("I know that / what NAME V-ed")

### Primary, T − I donors (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | -1.50 [-1.93, -1.09] FALL | +2.47 [+1.97, +3.04] RISE | +1.74 [+1.46, +2.02] RISE | +0.90 [+0.59, +1.23] RISE | -0.84 [-1.05, -0.63] FALL | +2.85 [+2.43, +3.30] RISE | +0.51 [+0.10, +0.86] unresolved | +0.38 [+0.11, +0.64] | **mixed / unresolved** |
| 11 | -2.28 [-2.77, -1.80] FALL | +2.74 [+2.25, +3.25] RISE | +0.39 [+0.04, +0.73] unresolved | -0.42 [-0.71, -0.11] FALL | -0.81 [-1.04, -0.59] FALL | +3.31 [+2.93, +3.70] RISE | +0.93 [+0.41, +1.34] above | +0.57 [+0.30, +0.83] | **surface (object next)** |
| 12 | -2.35 [-2.88, -1.91] FALL | +2.89 [+2.39, +3.42] RISE | -0.35 [-0.57, -0.15] unresolved | -1.27 [-1.59, -1.03] FALL | -0.92 [-1.15, -0.71] FALL | +3.65 [+3.20, +4.20] RISE | +1.37 [+1.10, +1.63] above | +0.75 [+0.42, +1.10] | **surface (object next)** |
| 13 | -2.57 [-3.07, -2.12] FALL | +2.95 [+2.49, +3.42] RISE | -0.21 [-0.66, +0.20] unresolved | -1.89 [-2.25, -1.59] FALL | -1.68 [-2.06, -1.32] FALL | +3.80 [+3.35, +4.34] RISE | +1.69 [+1.43, +1.94] above | +0.85 [+0.55, +1.16] | **surface (object next)** |

### Primary, in range (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | -1.44 [-1.86, -1.04] FALL | +2.45 [+1.94, +3.03] RISE | +1.38 [+1.16, +1.60] RISE | +0.84 [+0.60, +1.07] RISE | -0.54 [-0.69, -0.39] FALL | +2.41 [+2.01, +2.83] RISE | -0.01 [-0.50, +0.38] not above | -0.04 [-0.31, +0.22] | **licensing-like crossover** |
| 11 | -2.28 [-2.74, -1.79] FALL | +2.75 [+2.26, +3.25] RISE | +0.13 [-0.13, +0.38] NO RISE | -0.11 [-0.30, +0.07] unresolved | -0.24 [-0.33, -0.15] FALL | +2.24 [+1.94, +2.51] RISE | -0.18 [-0.83, +0.31] not above | -0.52 [-0.87, -0.18] | **mixed / unresolved** |
| 12 | -2.32 [-2.79, -1.87] FALL | +2.92 [+2.40, +3.43] RISE | -0.31 [-0.44, -0.20] NO RISE | -0.42 [-0.54, -0.31] FALL | -0.11 [-0.17, -0.05] NO RISE | +2.17 [+1.88, +2.45] RISE | -0.25 [-0.76, +0.17] not above | -0.75 [-1.12, -0.38] | **mixed / unresolved** |
| 13 | -2.70 [-3.13, -2.29] FALL | +3.08 [+2.61, +3.55] RISE | -0.15 [-0.32, +0.02] NO RISE | -0.37 [-0.52, -0.23] FALL | -0.22 [-0.29, -0.14] FALL | +2.40 [+2.14, +2.65] RISE | -0.03 [-0.64, +0.44] not above | -0.68 [-1.07, -0.30] | **mixed / unresolved** |

### Readings by population (sites 6 and 8; strata at every site below)

| Population | verbs | contexts | site | T − I donors | in range |
|---|---:|---:|---:|---|---|
| primary: non-DAS, half B | 10 | 12 | 9 | mixed / unresolved | licensing-like crossover |
| primary: non-DAS, half B | 10 | 12 | 11 | surface (object next) | mixed / unresolved |
| primary: non-DAS, half B | 10 | 12 | 12 | surface (object next) | mixed / unresolved |
| primary: non-DAS, half B | 10 | 12 | 13 | surface (object next) | mixed / unresolved |
| primary: non-DAS, all contexts | 10 | 24 | 9 | mixed / unresolved | licensing-like crossover |
| primary: non-DAS, all contexts | 10 | 24 | 11 | surface (object next) | mixed / unresolved |
| primary: non-DAS, all contexts | 10 | 24 | 12 | surface (object next) | mixed / unresolved |
| primary: non-DAS, all contexts | 10 | 24 | 13 | surface (object next) | mixed / unresolved |
| DAS bases, pooled targets | 8 | 12 | 9 | mixed / unresolved | licensing-like crossover |
| DAS bases, pooled targets | 8 | 12 | 11 | surface (object next) | mixed / unresolved |
| DAS bases, pooled targets | 8 | 12 | 12 | surface (object next) | mixed / unresolved |
| DAS bases, pooled targets | 8 | 12 | 13 | surface (object next) | mixed / unresolved |
| DAS bases, non-DAS targets | 8 | 12 | 9 | mixed / unresolved | licensing-like crossover |
| DAS bases, non-DAS targets | 8 | 12 | 11 | surface (object next) | mixed / unresolved |
| DAS bases, non-DAS targets | 8 | 12 | 12 | surface (object next) | mixed / unresolved |
| DAS bases, non-DAS targets | 8 | 12 | 13 | surface (object next) | mixed / unresolved |
| round-4 population (fp32) | 18 | 12 | 9 | mixed / unresolved | licensing-like crossover |
| round-4 population (fp32) | 18 | 12 | 11 | surface (object next) | mixed / unresolved |
| round-4 population (fp32) | 18 | 12 | 12 | surface (object next) | mixed / unresolved |
| round-4 population (fp32) | 18 | 12 | 13 | surface (object next) | mixed / unresolved |
| multi-token, non-DAS | 8 | 12 | 9 | surface (object next) | licensing-like crossover |
| multi-token, non-DAS | 8 | 12 | 11 | surface (object next) | mixed / unresolved |
| multi-token, non-DAS | 8 | 12 | 12 | surface (object next) | mixed / unresolved |
| multi-token, non-DAS | 8 | 12 | 13 | surface (object next) | mixed / unresolved |

### Frame strata (primary population; in range / T − I donors per site)

| Stratum | contexts | site 9 | site 11 | site 12 | site 13 |
|---|---|---|---|---|---|
| forgot | all 6 | LIC / mix | mix / surf | mix / surf | mix / surf |
| forgot | half B | LIC / mix | mix / surf | mix / surf | mix / surf |
| heard | all 6 | LIC / mix | mix / surf | mix / surf | mix / surf |
| heard | half B | LIC / mix | mix / surf | mix / surf | mix / surf |
| know | all 6 | LIC / surf | mix / surf | mix / surf | mix / surf |
| know | half B | LIC / surf | mix / surf | mix / surf | mix / surf |
| remember | all 6 | LIC / surf | mix / surf | mix / surf | mix / surf |
| remember | half B | LIC / mix | mix / surf | mix / surf | mix / surf |

LIC = licensing-like crossover, surf = surface (object next), mix = mixed / unresolved.

## B7 matrix wh ("AUX NAME V" / "What AUX NAME V")

### Primary, T − I donors (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | -1.57 [-1.91, -1.28] FALL | +3.16 [+2.81, +3.54] RISE | +1.87 [+1.57, +2.17] RISE | +1.11 [+0.84, +1.41] RISE | -0.76 [-0.89, -0.63] FALL | +2.96 [+2.62, +3.34] RISE | +0.69 [+0.37, +1.04] above | -0.19 [-0.44, +0.05] | **surface (object next)** |
| 11 | -2.42 [-2.82, -2.05] FALL | +3.50 [+3.17, +3.87] RISE | +0.61 [+0.38, +0.87] RISE | -0.36 [-0.65, -0.09] FALL | -0.97 [-1.21, -0.77] FALL | +3.80 [+3.41, +4.21] RISE | +1.42 [+1.09, +1.77] above | +0.30 [+0.09, +0.52] | **surface (object next)** |
| 12 | -2.37 [-2.71, -2.04] FALL | +3.63 [+3.30, +3.99] RISE | -0.04 [-0.24, +0.14] NO RISE | -1.55 [-1.83, -1.28] FALL | -1.50 [-1.86, -1.20] FALL | +4.23 [+3.86, +4.63] RISE | +2.00 [+1.74, +2.25] above | +0.60 [+0.41, +0.80] | **surface (object next)** |
| 13 | -2.29 [-2.59, -2.00] FALL | +3.66 [+3.29, +4.02] RISE | +0.51 [+0.30, +0.71] RISE | -1.74 [-2.04, -1.44] FALL | -2.25 [-2.62, -1.90] FALL | +4.35 [+3.93, +4.77] RISE | +2.27 [+2.02, +2.52] above | +0.69 [+0.51, +0.89] | **surface (object next)** |

### Primary, in range (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | -1.23 [-1.47, -1.01] FALL | +3.11 [+2.78, +3.47] RISE | +1.40 [+1.17, +1.61] RISE | +0.96 [+0.75, +1.17] RISE | -0.44 [-0.53, -0.35] FALL | +2.50 [+2.24, +2.78] RISE | -0.00 [-0.37, +0.36] not above | -0.61 [-0.86, -0.37] | **licensing-like crossover** |
| 11 | -2.15 [-2.47, -1.83] FALL | +3.50 [+3.18, +3.86] RISE | +0.09 [-0.11, +0.29] NO RISE | -0.12 [-0.28, +0.04] NO RISE | -0.21 [-0.28, -0.15] FALL | +2.47 [+2.25, +2.71] RISE | -0.03 [-0.42, +0.37] not above | -1.03 [-1.34, -0.76] | **mixed / unresolved** |
| 12 | -2.08 [-2.35, -1.82] FALL | +3.64 [+3.34, +3.98] RISE | -0.33 [-0.43, -0.23] NO RISE | -0.45 [-0.55, -0.37] FALL | -0.13 [-0.18, -0.08] NO RISE | +2.43 [+2.20, +2.69] RISE | -0.05 [-0.43, +0.33] not above | -1.20 [-1.51, -0.91] | **mixed / unresolved** |
| 13 | -2.17 [-2.41, -1.95] FALL | +3.80 [+3.50, +4.12] RISE | -0.00 [-0.11, +0.11] NO RISE | -0.25 [-0.33, -0.16] unresolved | -0.24 [-0.31, -0.18] FALL | +2.69 [+2.45, +2.93] RISE | +0.20 [-0.21, +0.61] not above | -1.11 [-1.48, -0.75] | **mixed / unresolved** |

### Readings by population (sites 6 and 8; strata at every site below)

| Population | verbs | contexts | site | T − I donors | in range |
|---|---:|---:|---:|---|---|
| primary: non-DAS, half B | 23 | 12 | 9 | surface (object next) | licensing-like crossover |
| primary: non-DAS, half B | 23 | 12 | 11 | surface (object next) | mixed / unresolved |
| primary: non-DAS, half B | 23 | 12 | 12 | surface (object next) | mixed / unresolved |
| primary: non-DAS, half B | 23 | 12 | 13 | surface (object next) | mixed / unresolved |
| primary: non-DAS, all contexts | 23 | 24 | 9 | surface (object next) | licensing-like crossover |
| primary: non-DAS, all contexts | 23 | 24 | 11 | surface (object next) | mixed / unresolved |
| primary: non-DAS, all contexts | 23 | 24 | 12 | surface (object next) | mixed / unresolved |
| primary: non-DAS, all contexts | 23 | 24 | 13 | surface (object next) | mixed / unresolved |
| DAS bases, pooled targets | 9 | 12 | 9 | mixed / unresolved | licensing-like crossover |
| DAS bases, pooled targets | 9 | 12 | 11 | surface (object next) | mixed / unresolved |
| DAS bases, pooled targets | 9 | 12 | 12 | surface (object next) | mixed / unresolved |
| DAS bases, pooled targets | 9 | 12 | 13 | surface (object next) | mixed / unresolved |
| DAS bases, non-DAS targets | 9 | 12 | 9 | mixed / unresolved | licensing-like crossover |
| DAS bases, non-DAS targets | 9 | 12 | 11 | surface (object next) | mixed / unresolved |
| DAS bases, non-DAS targets | 9 | 12 | 12 | surface (object next) | mixed / unresolved |
| DAS bases, non-DAS targets | 9 | 12 | 13 | surface (object next) | mixed / unresolved |
| round-4 population (fp32) | 32 | 12 | 9 | surface (object next) | licensing-like crossover |
| round-4 population (fp32) | 32 | 12 | 11 | surface (object next) | mixed / unresolved |
| round-4 population (fp32) | 32 | 12 | 12 | surface (object next) | mixed / unresolved |
| round-4 population (fp32) | 32 | 12 | 13 | surface (object next) | mixed / unresolved |
| multi-token, non-DAS | 13 | 12 | 9 | surface (object next) | licensing-like crossover |
| multi-token, non-DAS | 13 | 12 | 11 | surface (object next) | mixed / unresolved |
| multi-token, non-DAS | 13 | 12 | 12 | surface (object next) | mixed / unresolved |
| multi-token, non-DAS | 13 | 12 | 13 | surface (object next) | mixed / unresolved |

### Frame strata (primary population; in range / T − I donors per site)

| Stratum | contexts | site 9 | site 11 | site 12 | site 13 |
|---|---|---|---|---|---|
| Can | all 6 | mix / surf | mix / surf | mix / surf | mix / surf |
| Can | half B | mix / surf | mix / surf | mix / surf | mix / surf |
| Did | all 6 | LIC / surf | mix / surf | mix / surf | mix / surf |
| Did | half B | LIC / surf | mix / surf | mix / surf | mix / surf |
| Will | all 6 | LIC / mix | mix / surf | mix / surf | mix / surf |
| Will | half B | LIC / mix | mix / surf | mix / surf | mix / surf |
| Would | all 6 | LIC / mix | mix / surf | mix / surf | mix / surf |
| Would | half B | LIC / mix | mix / surf | mix / surf | mix / surf |

LIC = licensing-like crossover, surf = surface (object next), mix = mixed / unresolved.

