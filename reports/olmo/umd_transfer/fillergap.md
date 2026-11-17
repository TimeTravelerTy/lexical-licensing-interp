# Round 5, C8: filler-gap patches with non-DAS verbs as the primary set (fp32)

Spec: `plan.md`, C8. Runs: `run_fillergap.py patch --dtype float32 --target-pool {nondas, all}`; analysis: `analyze_fillergap_r5.py patch` (round-4 statistics and reading rule). D = after T − after I active donors; D_in = coordinate at t_T − at t_I. nofill = "that" / no wh; fill = "what". Primary: kept non-DAS I bases, non-DAS targets, natural gaps and T reference from non-DAS verbs, half-B contexts.

## B6 embedded wh ("I know that / what NAME V-ed")

### Primary, T − I donors (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | -0.71 [-0.97, -0.48] FALL | +1.68 [+1.28, +2.13] RISE | +1.15 [+0.95, +1.38] RISE | +0.71 [+0.50, +0.92] RISE | -0.44 [-0.61, -0.30] FALL | +2.11 [+1.78, +2.47] RISE | -0.28 [-0.84, +0.15] not above | +0.42 [+0.23, +0.63] | **licensing-like crossover** |
| 11 | -1.24 [-1.67, -0.85] FALL | +1.95 [+1.53, +2.41] RISE | +0.52 [+0.18, +0.87] RISE | +0.15 [-0.07, +0.38] unresolved | -0.37 [-0.52, -0.23] FALL | +2.18 [+1.86, +2.50] RISE | -0.24 [-0.84, +0.22] not above | +0.23 [-0.01, +0.45] | **mixed / unresolved** |
| 12 | -1.30 [-1.75, -0.87] FALL | +2.10 [+1.66, +2.55] RISE | -0.22 [-0.51, +0.01] NO RISE | -0.58 [-0.81, -0.38] FALL | -0.35 [-0.48, -0.23] FALL | +2.62 [+2.33, +2.94] RISE | +0.25 [-0.22, +0.67] not above | +0.52 [+0.22, +0.81] | **mixed / unresolved** |
| 13 | -1.71 [-2.19, -1.28] FALL | +2.27 [+1.87, +2.69] RISE | -0.41 [-0.83, -0.06] unresolved | -1.08 [-1.36, -0.84] FALL | -0.67 [-0.89, -0.45] FALL | +2.96 [+2.67, +3.26] RISE | +0.69 [+0.19, +1.10] above | +0.69 [+0.42, +0.94] | **surface (object next)** |

### Primary, in range (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | -0.59 [-0.83, -0.37] FALL | +1.56 [+1.15, +1.99] RISE | +0.77 [+0.63, +0.91] RISE | +0.53 [+0.40, +0.66] RISE | -0.23 [-0.34, -0.15] FALL | +1.55 [+1.30, +1.81] RISE | -0.86 [-1.48, -0.37] not above | -0.00 [-0.26, +0.23] | **licensing-like crossover** |
| 11 | -1.14 [-1.54, -0.78] FALL | +1.95 [+1.52, +2.41] RISE | +0.19 [-0.05, +0.43] NO RISE | +0.05 [-0.12, +0.22] NO RISE | -0.14 [-0.22, -0.06] unresolved | +1.35 [+1.11, +1.57] RISE | -1.05 [-1.75, -0.49] not above | -0.61 [-0.95, -0.31] | **mixed / unresolved** |
| 12 | -1.22 [-1.66, -0.81] FALL | +2.13 [+1.69, +2.58] RISE | -0.19 [-0.33, -0.06] NO RISE | -0.26 [-0.39, -0.16] unresolved | -0.08 [-0.12, -0.03] NO RISE | +1.60 [+1.40, +1.78] RISE | -0.82 [-1.44, -0.28] not above | -0.53 [-0.90, -0.19] | **mixed / unresolved** |
| 13 | -1.65 [-2.13, -1.20] FALL | +2.36 [+1.94, +2.79] RISE | -0.24 [-0.42, -0.08] NO RISE | -0.37 [-0.51, -0.25] FALL | -0.13 [-0.20, -0.07] NO RISE | +1.70 [+1.50, +1.89] RISE | -0.71 [-1.45, -0.11] not above | -0.66 [-1.09, -0.25] | **mixed / unresolved** |

### Readings by population (sites 6 and 8; strata at every site below)

| Population | verbs | contexts | site | T − I donors | in range |
|---|---:|---:|---:|---|---|
| primary: non-DAS, half B | 10 | 12 | 9 | licensing-like crossover | licensing-like crossover |
| primary: non-DAS, half B | 10 | 12 | 11 | mixed / unresolved | mixed / unresolved |
| primary: non-DAS, half B | 10 | 12 | 12 | mixed / unresolved | mixed / unresolved |
| primary: non-DAS, half B | 10 | 12 | 13 | surface (object next) | mixed / unresolved |
| primary: non-DAS, all contexts | 10 | 24 | 9 | licensing-like crossover | licensing-like crossover |
| primary: non-DAS, all contexts | 10 | 24 | 11 | mixed / unresolved | mixed / unresolved |
| primary: non-DAS, all contexts | 10 | 24 | 12 | mixed / unresolved | mixed / unresolved |
| primary: non-DAS, all contexts | 10 | 24 | 13 | surface (object next) | mixed / unresolved |
| DAS bases, pooled targets | 8 | 12 | 9 | mixed / unresolved | mixed / unresolved |
| DAS bases, pooled targets | 8 | 12 | 11 | mixed / unresolved | mixed / unresolved |
| DAS bases, pooled targets | 8 | 12 | 12 | mixed / unresolved | mixed / unresolved |
| DAS bases, pooled targets | 8 | 12 | 13 | surface (object next) | mixed / unresolved |
| DAS bases, non-DAS targets | 8 | 12 | 9 | mixed / unresolved | mixed / unresolved |
| DAS bases, non-DAS targets | 8 | 12 | 11 | mixed / unresolved | mixed / unresolved |
| DAS bases, non-DAS targets | 8 | 12 | 12 | mixed / unresolved | mixed / unresolved |
| DAS bases, non-DAS targets | 8 | 12 | 13 | surface (object next) | mixed / unresolved |
| round-4 population (fp32) | 18 | 12 | 9 | mixed / unresolved | mixed / unresolved |
| round-4 population (fp32) | 18 | 12 | 11 | mixed / unresolved | mixed / unresolved |
| round-4 population (fp32) | 18 | 12 | 12 | mixed / unresolved | mixed / unresolved |
| round-4 population (fp32) | 18 | 12 | 13 | surface (object next) | mixed / unresolved |
| multi-token, non-DAS | 8 | 12 | 9 | licensing-like crossover | licensing-like crossover |
| multi-token, non-DAS | 8 | 12 | 11 | licensing-like crossover | mixed / unresolved |
| multi-token, non-DAS | 8 | 12 | 12 | mixed / unresolved | mixed / unresolved |
| multi-token, non-DAS | 8 | 12 | 13 | surface (object next) | mixed / unresolved |

### Frame strata (primary population; in range / T − I donors per site)

| Stratum | contexts | site 9 | site 11 | site 12 | site 13 |
|---|---|---|---|---|---|
| forgot | all 6 | LIC / LIC | mix / LIC | mix / mix | mix / surf |
| forgot | half B | mix / LIC | mix / LIC | mix / mix | mix / surf |
| heard | all 6 | mix / LIC | mix / mix | mix / mix | mix / mix |
| heard | half B | LIC / LIC | mix / mix | mix / mix | mix / mix |
| know | all 6 | mix / LIC | mix / mix | mix / mix | mix / surf |
| know | half B | mix / LIC | mix / mix | mix / mix | mix / surf |
| remember | all 6 | LIC / LIC | mix / LIC | mix / mix | mix / surf |
| remember | half B | LIC / LIC | mix / LIC | mix / mix | mix / surf |

LIC = licensing-like crossover, surf = surface (object next), mix = mixed / unresolved.

## B7 matrix wh ("AUX NAME V" / "What AUX NAME V")

### Primary, T − I donors (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | -0.61 [-0.75, -0.49] FALL | +2.34 [+2.03, +2.66] RISE | +1.29 [+1.08, +1.53] RISE | +0.88 [+0.69, +1.09] RISE | -0.41 [-0.53, -0.30] FALL | +2.26 [+1.98, +2.55] RISE | -0.14 [-0.55, +0.27] not above | -0.09 [-0.32, +0.15] | **licensing-like crossover** |
| 11 | -0.96 [-1.16, -0.77] FALL | +2.68 [+2.40, +3.00] RISE | +0.77 [+0.49, +1.04] RISE | +0.32 [+0.14, +0.50] RISE | -0.45 [-0.59, -0.32] FALL | +2.51 [+2.26, +2.78] RISE | +0.03 [-0.37, +0.41] not above | -0.17 [-0.39, +0.04] | **licensing-like crossover** |
| 12 | -1.02 [-1.23, -0.83] FALL | +2.75 [+2.47, +3.04] RISE | -0.03 [-0.22, +0.17] NO RISE | -0.52 [-0.68, -0.36] FALL | -0.49 [-0.64, -0.35] FALL | +3.01 [+2.76, +3.28] RISE | +0.59 [+0.22, +0.95] unresolved | +0.26 [+0.06, +0.45] | **mixed / unresolved** |
| 13 | -1.08 [-1.27, -0.92] FALL | +2.89 [+2.61, +3.18] RISE | +0.28 [+0.13, +0.41] NO RISE | -0.55 [-0.70, -0.42] FALL | -0.83 [-1.01, -0.65] FALL | +3.47 [+3.18, +3.76] RISE | +1.11 [+0.79, +1.41] above | +0.58 [+0.42, +0.75] | **surface (object next)** |

### Primary, in range (half B)

| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | -0.41 [-0.49, -0.34] NO RISE | +2.04 [+1.79, +2.33] RISE | +0.79 [+0.64, +0.93] RISE | +0.59 [+0.46, +0.71] RISE | -0.20 [-0.27, -0.13] FALL | +1.64 [+1.48, +1.82] RISE | -0.84 [-1.29, -0.40] not above | -0.39 [-0.63, -0.17] | **mixed / unresolved** |
| 11 | -0.81 [-0.94, -0.67] FALL | +2.59 [+2.33, +2.89] RISE | +0.23 [+0.04, +0.42] NO RISE | +0.07 [-0.07, +0.21] NO RISE | -0.16 [-0.23, -0.09] unresolved | +1.56 [+1.42, +1.72] RISE | -0.91 [-1.37, -0.46] not above | -1.03 [-1.31, -0.78] | **mixed / unresolved** |
| 12 | -0.87 [-1.02, -0.72] FALL | +2.70 [+2.45, +2.96] RISE | -0.18 [-0.30, -0.06] NO RISE | -0.28 [-0.38, -0.19] unresolved | -0.10 [-0.15, -0.05] NO RISE | +1.79 [+1.63, +1.95] RISE | -0.70 [-1.14, -0.26] not above | -0.91 [-1.19, -0.65] | **mixed / unresolved** |
| 13 | -0.93 [-1.07, -0.81] FALL | +2.92 [+2.69, +3.18] RISE | +0.01 [-0.09, +0.09] NO RISE | -0.15 [-0.22, -0.10] NO RISE | -0.16 [-0.21, -0.11] unresolved | +1.98 [+1.81, +2.16] RISE | -0.50 [-0.97, -0.05] not above | -0.94 [-1.24, -0.68] | **mixed / unresolved** |

### Readings by population (sites 6 and 8; strata at every site below)

| Population | verbs | contexts | site | T − I donors | in range |
|---|---:|---:|---:|---|---|
| primary: non-DAS, half B | 23 | 12 | 9 | licensing-like crossover | mixed / unresolved |
| primary: non-DAS, half B | 23 | 12 | 11 | licensing-like crossover | mixed / unresolved |
| primary: non-DAS, half B | 23 | 12 | 12 | mixed / unresolved | mixed / unresolved |
| primary: non-DAS, half B | 23 | 12 | 13 | surface (object next) | mixed / unresolved |
| primary: non-DAS, all contexts | 23 | 24 | 9 | licensing-like crossover | mixed / unresolved |
| primary: non-DAS, all contexts | 23 | 24 | 11 | licensing-like crossover | mixed / unresolved |
| primary: non-DAS, all contexts | 23 | 24 | 12 | mixed / unresolved | mixed / unresolved |
| primary: non-DAS, all contexts | 23 | 24 | 13 | surface (object next) | mixed / unresolved |
| DAS bases, pooled targets | 9 | 12 | 9 | mixed / unresolved | mixed / unresolved |
| DAS bases, pooled targets | 9 | 12 | 11 | licensing-like crossover | mixed / unresolved |
| DAS bases, pooled targets | 9 | 12 | 12 | surface (object next) | mixed / unresolved |
| DAS bases, pooled targets | 9 | 12 | 13 | surface (object next) | mixed / unresolved |
| DAS bases, non-DAS targets | 9 | 12 | 9 | mixed / unresolved | mixed / unresolved |
| DAS bases, non-DAS targets | 9 | 12 | 11 | licensing-like crossover | mixed / unresolved |
| DAS bases, non-DAS targets | 9 | 12 | 12 | mixed / unresolved | mixed / unresolved |
| DAS bases, non-DAS targets | 9 | 12 | 13 | surface (object next) | mixed / unresolved |
| round-4 population (fp32) | 32 | 12 | 9 | mixed / unresolved | mixed / unresolved |
| round-4 population (fp32) | 32 | 12 | 11 | licensing-like crossover | mixed / unresolved |
| round-4 population (fp32) | 32 | 12 | 12 | surface (object next) | mixed / unresolved |
| round-4 population (fp32) | 32 | 12 | 13 | surface (object next) | mixed / unresolved |
| multi-token, non-DAS | 13 | 12 | 9 | mixed / unresolved | mixed / unresolved |
| multi-token, non-DAS | 13 | 12 | 11 | mixed / unresolved | mixed / unresolved |
| multi-token, non-DAS | 13 | 12 | 12 | surface (object next) | mixed / unresolved |
| multi-token, non-DAS | 13 | 12 | 13 | surface (object next) | mixed / unresolved |

### Frame strata (primary population; in range / T − I donors per site)

| Stratum | contexts | site 9 | site 11 | site 12 | site 13 |
|---|---|---|---|---|---|
| Can | all 6 | mix / mix | mix / mix | mix / surf | mix / surf |
| Can | half B | mix / mix | mix / mix | mix / surf | mix / surf |
| Did | all 6 | mix / mix | mix / mix | mix / mix | mix / surf |
| Did | half B | mix / mix | mix / mix | mix / mix | mix / surf |
| Will | all 6 | mix / mix | mix / LIC | mix / mix | mix / surf |
| Will | half B | mix / mix | mix / LIC | mix / mix | mix / surf |
| Would | all 6 | mix / LIC | mix / LIC | mix / mix | mix / surf |
| Would | half B | mix / LIC | mix / LIC | mix / mix | mix / surf |

LIC = licensing-like crossover, surf = surface (object next), mix = mixed / unresolved.

