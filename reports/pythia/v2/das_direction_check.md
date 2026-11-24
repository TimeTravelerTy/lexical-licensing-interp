# DAS direction check (Pythia-1.4B)

Rules, declared and committed before running: `das_direction_check_plan.md`. Run: `run_das_direction_check.py` (v2, job 9551316 (analysis re-run with --ref-ablation)); tables: `analyze_das_direction_check.py`; outputs `results/pythia/das_direction_check/`. Sites 4, 6, 8, 12, 17; 15 final-run rank-1 bases per site (3 splits x 5 folds).

## Checks

| Site | v held-out IIA vs stored (max abs diff per basis) | v active Δ log P(O) vs δ_s/0.2 | v passive D(O) vs stored | v passive D(by) vs stored | u_md self-patch max abs dM | identity ablation max abs dM (active / passive) | unit-norm max dev | batch effect max abs dM |
|---:|---|---|---|---|---:|---|---:|---:|
| 4 | 0.0000 (pass) | 2.045 vs 2.045 (pass) | 0.050 vs 0.050 (pass) | 0.639 vs 0.639 (pass) | 0 | 0 / 0 | 1.4e-07 | 0.42 |
| 6 | 0.0000 (pass) | 2.355 vs 2.355 (pass) | 0.147 vs 0.147 (pass) | 0.732 vs 0.732 (pass) | 0 | 0 / 0 | 1.4e-07 | 0.42 |
| 8 | 0.0000 (pass) | 2.499 vs 2.499 (pass) | 0.250 vs 0.250 (pass) | 0.780 vs 0.780 (pass) | 0 | 0 / 0 | 1.1e-07 | 0.42 |
| 12 | 0.0000 (pass) | 2.691 vs 2.691 (pass) | 0.938 vs 0.938 (pass) | 0.832 vs 0.832 (pass) | 0 | 0 / 0 | 1.5e-07 | 0.42 |
| 17 | 0.0000 (pass) | 2.940 vs 2.939 (pass) | 3.749 vs 3.749 (pass) | -1.072 vs -1.072 (pass) | 0 | 0 / 0 | 1.7e-07 | 0.42 |

Thresholds (declared): IIA 0.01, active effect 0.05, passive D 0.02 nats. Self-patch and identity ablation are compared with an unpatched forward of the same batch (zero basis); the batch-effect column is the max |dM| between that and the separate natural pass (bf16 batch composition).

## 1. Natural class axis

|cos| over the 15 bases: mean [min, max]. u_leace = in-sample single-site LEACE eraser (fit on passive `was` items), edit and read vectors; cross-fitted = mean over the 5 cross-fitted erasers. References pooled over bases: random isotropic unit vectors (10,000 per basis) and shuffled-label mean differences (100 verb-level permutations per basis), mean / 95th pct.

| Site | v·u_md | v·u_leace edit | v·u_leace read | v·edit (cross-fit) | v·read (cross-fit) | u_md·edit | u_md·read | random ref | shuffled-label ref |
|---:|---|---|---|---:|---:|---|---|---|---|
| 4 | 0.683 [0.671, 0.706] | 0.592 [0.585, 0.600] | 0.305 [0.299, 0.313] | 0.572 | 0.298 | 0.493 [0.474, 0.516] | 0.217 [0.205, 0.230] | 0.017 / 0.043 | 0.123 / 0.291 |
| 6 | 0.723 [0.703, 0.736] | 0.512 [0.496, 0.526] | 0.274 [0.268, 0.277] | 0.500 | 0.268 | 0.474 [0.462, 0.488] | 0.200 [0.190, 0.209] | 0.017 / 0.043 | 0.148 / 0.344 |
| 8 | 0.731 [0.721, 0.744] | 0.394 [0.383, 0.407] | 0.261 [0.256, 0.265] | 0.387 | 0.261 | 0.364 [0.351, 0.377] | 0.191 [0.184, 0.202] | 0.017 / 0.043 | 0.159 / 0.362 |
| 12 | 0.741 [0.734, 0.749] | 0.363 [0.350, 0.375] | 0.188 [0.183, 0.194] | 0.358 | 0.191 | 0.367 [0.352, 0.386] | 0.144 [0.138, 0.150] | 0.018 / 0.043 | 0.177 / 0.399 |
| 17 | 0.729 [0.715, 0.740] | 0.208 [0.195, 0.220] | 0.125 [0.122, 0.129] | 0.205 | 0.127 | 0.293 [0.281, 0.312] | 0.123 [0.116, 0.129] | 0.018 / 0.043 | 0.198 / 0.431 |

Variance of the projection of all 406 active items, as a multiple of the mean per-dimension variance (trace / d); mean [min, max] over bases.

| Site | along v | along u_md | |cos(edit, read)| |
|---:|---|---|---:|
| 4 | 40.978 [39.463, 43.037] | 82.263 [79.707, 84.967] | 0.545 |
| 6 | 59.396 [57.890, 60.886] | 107.751 [105.810, 111.261] | 0.436 |
| 8 | 66.801 [65.341, 69.490] | 120.323 [118.710, 122.828] | 0.368 |
| 12 | 84.799 [83.223, 86.347] | 147.330 [145.857, 149.643] | 0.263 |
| 17 | 107.174 [104.852, 108.683] | 197.328 [195.084, 199.669] | 0.237 |

## 2. Component patching: u_md vs v

### Held-out actives

IIA: cross-class held-out IIA, mean [min, max] over bases. PC frac: median positive-control fraction, mean over bases. Δ log P(O): intransitive base ← transitive source, mean over pairs (`per_pair`), 95% CI over pairs; the ratio row's CI is a paired pair bootstrap.

| Site | Direction | IIA | PC frac | same-class IIA | Δ log P(O) |
|---:|---|---|---:|---:|---|
| 4 | v | 0.442 [0.194, 0.591] | 0.605 | 0.987 | 2.045 [1.740, 2.339] |
| 4 | u_md | 0.134 [0.054, 0.212] | 0.296 | 0.985 | 1.237 [0.994, 1.498] |
| 4 | u_md / v | 0.304 | | | 0.605 [0.535, 0.670] |
| 6 | v | 0.659 [0.446, 0.774] | 0.726 | 0.993 | 2.355 [2.039, 2.673] |
| 6 | u_md | 0.217 [0.129, 0.312] | 0.402 | 0.987 | 1.559 [1.285, 1.839] |
| 6 | u_md / v | 0.329 | | | 0.662 [0.591, 0.731] |
| 8 | v | 0.793 [0.674, 0.863] | 0.792 | 0.991 | 2.499 [2.158, 2.837] |
| 8 | u_md | 0.279 [0.133, 0.406] | 0.458 | 0.988 | 1.724 [1.431, 2.026] |
| 8 | u_md / v | 0.351 | | | 0.690 [0.629, 0.746] |
| 12 | v | 0.911 [0.786, 0.988] | 0.903 | 0.986 | 2.691 [2.347, 3.033] |
| 12 | u_md | 0.365 [0.186, 0.565] | 0.486 | 0.992 | 1.827 [1.546, 2.109] |
| 12 | u_md / v | 0.401 | | | 0.679 [0.635, 0.719] |
| 17 | v | 0.984 [0.954, 1.000] | 1.159 | 0.982 | 2.940 [2.599, 3.273] |
| 17 | u_md | 0.689 [0.529, 0.855] | 0.659 | 0.989 | 2.264 [1.972, 2.569] |
| 17 | u_md / v | 0.700 | | | 0.770 [0.743, 0.798] |

### Passives (primary population, bad bases, D = T donor − I donor)

95% CI: three-way cluster bootstrap (2,000 draws, seed 17); ratio per draw on shared draws. Class: the declared passive-test rule with the site's δ_s.

| Site | Readout | D, v | class | D, u_md | class | u_md / v | stored D, v |
|---:|---|---|---|---|---|---|---:|
| 4 | M | 0.059 [0.010, 0.108] |  | 0.084 [0.040, 0.127] |  | 1.42 [0.84, 3.84] | 0.059 |
| 4 | log P(O) | 0.050 [0.001, 0.101] | NO RISE | 0.077 [0.033, 0.122] | NO RISE | 1.55 [0.77, 6.09] | 0.050 |
| 4 | log P(" by") | 0.639 [0.553, 0.728] | RISE | 0.447 [0.372, 0.525] | RISE | 0.70 [0.64, 0.76] | 0.639 |
| 4 | log P(".") | 0.187 [0.118, 0.258] | NO RISE | 0.069 [0.017, 0.124] | NO RISE | 0.37 [0.13, 0.53] | 0.187 |
| 6 | M | 0.183 [0.126, 0.243] |  | 0.097 [0.051, 0.143] |  | 0.53 [0.36, 0.67] | 0.183 |
| 6 | log P(O) | 0.147 [0.093, 0.202] | NO RISE | 0.078 [0.032, 0.124] | NO RISE | 0.53 [0.30, 0.70] | 0.147 |
| 6 | log P(" by") | 0.732 [0.643, 0.829] | RISE | 0.637 [0.548, 0.732] | RISE | 0.87 [0.81, 0.93] | 0.732 |
| 6 | log P(".") | 0.165 [0.106, 0.229] | NO RISE | 0.142 [0.083, 0.208] | NO RISE | 0.86 [0.68, 1.03] | 0.165 |
| 8 | M | 0.313 [0.238, 0.391] |  | 0.159 [0.107, 0.212] |  | 0.51 [0.41, 0.59] | 0.313 |
| 8 | log P(O) | 0.250 [0.185, 0.318] | NO RISE | 0.130 [0.082, 0.179] | NO RISE | 0.52 [0.40, 0.62] | 0.250 |
| 8 | log P(" by") | 0.780 [0.683, 0.882] | RISE | 0.631 [0.541, 0.725] | RISE | 0.81 [0.75, 0.87] | 0.780 |
| 8 | log P(".") | 0.062 [0.011, 0.116] | NO RISE | 0.129 [0.078, 0.182] | NO RISE | 2.07 [1.37, 6.33] | 0.062 |
| 12 | M | 1.084 [0.936, 1.250] |  | 0.241 [0.171, 0.317] |  | 0.22 [0.18, 0.26] | 1.084 |
| 12 | log P(O) | 0.938 [0.801, 1.089] | RISE | 0.172 [0.112, 0.237] | NO RISE | 0.18 [0.13, 0.23] | 0.938 |
| 12 | log P(" by") | 0.832 [0.736, 0.930] | RISE | 0.901 [0.797, 1.011] | RISE | 1.08 [1.02, 1.15] | 0.832 |
| 12 | log P(".") | -0.370 [-0.422, -0.318] | NO RISE | -0.016 [-0.072, 0.038] | NO RISE | 0.04 [-0.11, 0.18] | -0.370 |
| 17 | M | 5.314 [4.917, 5.686] |  | 2.554 [2.294, 2.802] |  | 0.48 [0.46, 0.50] | 5.314 |
| 17 | log P(O) | 3.749 [3.501, 3.969] | RISE | 2.237 [2.010, 2.448] | RISE | 0.60 [0.56, 0.63] | 3.749 |
| 17 | log P(" by") | -1.072 [-1.265, -0.874] | FALL | 0.436 [0.355, 0.514] | NO RISE | -0.41 [-0.54, -0.31] | -1.072 |
| 17 | log P(".") | -2.106 [-2.359, -1.837] | FALL | -0.617 [-0.724, -0.514] | FALL | 0.29 [0.26, 0.33] | -2.106 |

## 3. Mean ablation in natural runs

h ← h + (μ − h·u) u at the verb's last token, μ = grand mean of h·u over the frame's items (per basis). Natural = unpatched readouts computed in the same batches as the ablations (zero basis); the gap from the separate natural pass is in `summary_ablation.csv` (gap_nat_pass). Actives: T − I gap over the 29 pairs (each item averaged over the 3 bases holding its pair out); 95% CI over pairs. Passives: good − bad gap over the 64 primary pairs (each item averaged over its 3 plan bases); 95% CI two-way (pairs within band, contexts). Random: 10 random unit directions per basis, mean [min, max] of the share's point estimate.

| Site | Frame | Gap | natural | ablated v | share removed, v | ablated u_md | share removed, u_md | share removed, random |
|---:|---|---|---:|---:|---|---:|---|---|
| 4 | active | T − I log P(O) | 2.647 | 1.623 | 0.387 [0.310, 0.467] | 2.066 | 0.220 [0.160, 0.280] | -0.001 [-0.002, -0.000] |
| 4 | active | T − I M | 4.863 | 3.193 | 0.343 [0.279, 0.406] | 4.026 | 0.172 [0.132, 0.210] | 0.000 [-0.001, 0.001] |
| 4 | passive | good − bad log P(" by") | 0.713 | 0.386 | 0.459 [0.313, 0.863] | 0.523 | 0.266 [0.177, 0.503] | 0.001 [-0.001, 0.002] |
| 4 | passive | good − bad log P(O) | -0.221 | -0.284 | -0.289 [-2.535, 1.238] | -0.281 | -0.273 [-2.590, 1.205] | 0.002 [-0.002, 0.007] |
| 6 | active | T − I log P(O) | 2.647 | 1.268 | 0.521 [0.452, 0.590] | 1.889 | 0.287 [0.219, 0.354] | -0.001 [-0.002, -0.000] |
| 6 | active | T − I M | 4.863 | 2.612 | 0.463 [0.403, 0.522] | 3.731 | 0.233 [0.188, 0.276] | 0.000 [-0.001, 0.001] |
| 6 | passive | good − bad log P(" by") | 0.713 | 0.390 | 0.453 [0.305, 0.874] | 0.478 | 0.330 [0.223, 0.637] | 0.001 [-0.000, 0.003] |
| 6 | passive | good − bad log P(O) | -0.221 | -0.294 | -0.334 [-3.092, 1.375] | -0.269 | -0.218 [-1.920, 0.898] | 0.003 [0.000, 0.005] |
| 8 | active | T − I log P(O) | 2.647 | 0.959 | 0.638 [0.565, 0.710] | 1.743 | 0.342 [0.275, 0.408] | -0.001 [-0.003, -0.000] |
| 8 | active | T − I M | 4.863 | 2.091 | 0.570 [0.510, 0.629] | 3.460 | 0.288 [0.246, 0.332] | -0.000 [-0.001, 0.001] |
| 8 | passive | good − bad log P(" by") | 0.713 | 0.430 | 0.397 [0.268, 0.758] | 0.522 | 0.267 [0.181, 0.512] | 0.001 [0.000, 0.002] |
| 8 | passive | good − bad log P(O) | -0.221 | -0.313 | -0.421 [-3.806, 1.646] | -0.275 | -0.245 [-2.285, 1.037] | 0.003 [-0.001, 0.007] |
| 12 | active | T − I log P(O) | 2.647 | 0.618 | 0.766 [0.688, 0.849] | 1.695 | 0.360 [0.305, 0.413] | -0.001 [-0.003, -0.000] |
| 12 | active | T − I M | 4.863 | 1.339 | 0.725 [0.654, 0.797] | 3.234 | 0.335 [0.293, 0.377] | -0.000 [-0.001, 0.001] |
| 12 | passive | good − bad log P(" by") | 0.713 | 0.448 | 0.371 [0.249, 0.711] | 0.449 | 0.371 [0.251, 0.709] | 0.001 [0.001, 0.002] |
| 12 | passive | good − bad log P(O) | -0.221 | -0.459 | -1.083 [-9.814, 4.431] | -0.294 | -0.332 [-3.021, 1.317] | 0.002 [-0.003, 0.009] |
| 17 | active | T − I log P(O) | 2.647 | -0.126 | 1.047 [0.962, 1.141] | 1.133 | 0.572 [0.503, 0.642] | -0.000 [-0.001, 0.001] |
| 17 | active | T − I M | 4.863 | 0.019 | 0.996 [0.921, 1.078] | 2.282 | 0.531 [0.475, 0.591] | 0.000 [-0.000, 0.001] |
| 17 | passive | good − bad log P(" by") | 0.713 | 0.662 | 0.071 [0.032, 0.140] | 0.589 | 0.174 [0.119, 0.317] | 0.001 [-0.001, 0.003] |
| 17 | passive | good − bad log P(O) | -0.221 | -0.590 | -1.676 [-15.364, 7.464] | -0.420 | -0.905 [-8.740, 4.137] | 0.004 [-0.004, 0.008] |

### This v vs the reference v (`/gs/fs/tga-sip_arase/tyrone/lli_dirchk/results/pythia/das_direction_check`), paired

Share removed by this run's v minus the reference v's, on the same items and bootstrap draws, each share against its own run's natural readouts (zero-basis rows). Per-item max |Δ| of the none / u_md rows between the runs (bf16 run-to-run) and the natural-gap differences: {"active_none_umd_max_abs_dO": 0.1795, "active_site4_O_gap_nat_minus_ref": 0.0, "active_site4_M_gap_nat_minus_ref": 0.0, "active_site6_O_gap_nat_minus_ref": 0.0, "active_site6_M_gap_nat_minus_ref": 0.0, "active_site8_O_gap_nat_minus_ref": 0.0, "active_site8_M_gap_nat_minus_ref": 0.0, "active_site12_O_gap_nat_minus_ref": 0.0, "active_site12_M_gap_nat_minus_ref": 0.0, "active_site17_O_gap_nat_minus_ref": 0.0, "active_site17_M_gap_nat_minus_ref": 0.0, "passive_site4_none_umd_max_abs_by": 0.2696, "passive_site4_M_gap_nat_minus_ref": 0.0, "passive_site4_O_gap_nat_minus_ref": 0.0, "passive_site4_by_gap_nat_minus_ref": -0.0, "passive_site4_dot_gap_nat_minus_ref": 0.0, "passive_site6_none_umd_max_abs_by": 0.265, "passive_site6_M_gap_nat_minus_ref": 0.0, "passive_site6_O_gap_nat_minus_ref": 0.0, "passive_site6_by_gap_nat_minus_ref": -0.0, "passive_site6_dot_gap_nat_minus_ref": 0.0, "passive_site8_none_umd_max_abs_by": 0.1934, "passive_site8_M_gap_nat_minus_ref": 0.0, "passive_site8_O_gap_nat_minus_ref": 0.0, "passive_site8_by_gap_nat_minus_ref": -0.0, "passive_site8_dot_gap_nat_minus_ref": 0.0, "passive_site12_none_umd_max_abs_by": 0.2391, "passive_site12_M_gap_nat_minus_ref": 0.0, "passive_site12_O_gap_nat_minus_ref": 0.0, "passive_site12_by_gap_nat_minus_ref": -0.0, "passive_site12_dot_gap_nat_minus_ref": 0.0, "passive_site17_none_umd_max_abs_by": 0.2878, "passive_site17_M_gap_nat_minus_ref": 0.0, "passive_site17_O_gap_nat_minus_ref": 0.0, "passive_site17_by_gap_nat_minus_ref": -0.0, "passive_site17_dot_gap_nat_minus_ref": 0.0}.

| Site | Frame | Gap | share, v | share, reference v | difference [95% CI] |
|---:|---|---|---|---|---|
| 4 | active | T − I log P(O) | 0.387 [0.310, 0.467] | 0.430 [0.351, 0.511] | -0.044 [-0.079, -0.004] |
| 4 | active | T − I M | 0.343 [0.279, 0.406] | 0.359 [0.302, 0.417] | -0.016 [-0.053, 0.029] |
| 4 | passive | good − bad log P(" by") | 0.459 [0.313, 0.863] | 0.398 [0.269, 0.752] | 0.061 [0.033, 0.119] |
| 4 | passive | good − bad log P(O) | -0.289 [-2.535, 1.238] | -0.386 [-3.561, 1.625] | 0.097 [-0.477, 0.945] |
| 6 | active | T − I log P(O) | 0.521 [0.452, 0.590] | 0.537 [0.456, 0.615] | -0.016 [-0.050, 0.021] |
| 6 | active | T − I M | 0.463 [0.403, 0.522] | 0.465 [0.406, 0.520] | -0.003 [-0.035, 0.034] |
| 6 | passive | good − bad log P(" by") | 0.453 [0.305, 0.874] | 0.413 [0.277, 0.806] | 0.040 [0.018, 0.082] |
| 6 | passive | good − bad log P(O) | -0.334 [-3.092, 1.375] | -0.465 [-4.312, 1.882] | 0.131 [-0.561, 1.201] |
| 8 | active | T − I log P(O) | 0.638 [0.565, 0.710] | 0.630 [0.545, 0.712] | 0.008 [-0.032, 0.048] |
| 8 | active | T − I M | 0.570 [0.510, 0.629] | 0.558 [0.502, 0.616] | 0.012 [-0.023, 0.047] |
| 8 | passive | good − bad log P(" by") | 0.397 [0.268, 0.758] | 0.366 [0.246, 0.710] | 0.031 [0.009, 0.066] |
| 8 | passive | good − bad log P(O) | -0.421 [-3.806, 1.646] | -0.540 [-4.996, 2.151] | 0.120 [-0.495, 1.108] |
| 12 | active | T − I log P(O) | 0.766 [0.688, 0.849] | 0.739 [0.653, 0.824] | 0.027 [-0.014, 0.068] |
| 12 | active | T − I M | 0.725 [0.654, 0.797] | 0.706 [0.637, 0.774] | 0.019 [-0.019, 0.055] |
| 12 | passive | good − bad log P(" by") | 0.371 [0.249, 0.711] | 0.365 [0.244, 0.706] | 0.006 [-0.014, 0.026] |
| 12 | passive | good − bad log P(O) | -1.083 [-9.814, 4.431] | -1.189 [-10.888, 4.873] | 0.106 [-0.453, 1.047] |
| 17 | active | T − I log P(O) | 1.047 [0.962, 1.141] | 0.957 [0.867, 1.046] | 0.090 [0.056, 0.128] |
| 17 | active | T − I M | 0.996 [0.921, 1.078] | 0.914 [0.841, 0.990] | 0.082 [0.049, 0.117] |
| 17 | passive | good − bad log P(" by") | 0.071 [0.032, 0.140] | 0.049 [0.010, 0.103] | 0.022 [0.013, 0.043] |
| 17 | passive | good − bad log P(O) | -1.676 [-15.364, 7.464] | -1.572 [-14.623, 7.142] | -0.105 [-0.732, 0.300] |
