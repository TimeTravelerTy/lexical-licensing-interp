# DAS direction check (Pythia-1.4B)

Rules, declared and committed before running: `das_direction_check_plan.md`. Run: `run_das_direction_check.py` (TSUBAME job 9369381, commit 98d9e87); tables: `analyze_das_direction_check.py`; outputs `results/pythia/das_direction_check/`. Sites 4, 6, 8, 12, 17; 15 final-run rank-1 bases per site (3 splits x 5 folds).

## Checks

| Site | v held-out IIA vs stored (max abs diff per basis) | v active Δ log P(O) vs δ_s/0.2 | v passive D(O) vs stored | v passive D(by) vs stored | u_md self-patch max abs dM | identity ablation max abs dM (active / passive) | unit-norm max dev | batch effect max abs dM |
|---:|---|---|---|---|---:|---|---:|---:|
| 4 | 0.0000 (pass) | 2.088 vs 2.088 (pass) | 0.104 vs 0.104 (pass) | 0.577 vs 0.577 (pass) | 0 | 0 / 0 | 1.4e-07 | 0.42 |
| 6 | 0.0000 (pass) | 2.327 vs 2.327 (pass) | 0.226 vs 0.226 (pass) | 0.688 vs 0.688 (pass) | 0 | 0 / 0 | 1.9e-07 | 0.42 |
| 8 | 0.0000 (pass) | 2.457 vs 2.457 (pass) | 0.363 vs 0.363 (pass) | 0.719 vs 0.719 (pass) | 0 | 0 / 0 | 1.3e-07 | 0.42 |
| 12 | 0.0000 (pass) | 2.641 vs 2.640 (pass) | 1.167 vs 1.167 (pass) | 0.806 vs 0.806 (pass) | 0 | 0 / 0 | 1.0e-07 | 0.42 |
| 17 | 0.0000 (pass) | 2.858 vs 2.858 (pass) | 3.583 vs 3.583 (pass) | -0.929 vs -0.929 (pass) | 0 | 0 / 0 | 9.8e-08 | 0.42 |

Thresholds (declared): IIA 0.01, active effect 0.05, passive D 0.02 nats. Self-patch and identity ablation are compared with an unpatched forward of the same batch (zero basis); the batch-effect column is the max |dM| between that and the separate natural pass (bf16 batch composition).

## 1. Natural class axis

|cos| over the 15 bases: mean [min, max]. u_leace = in-sample single-site LEACE eraser (fit on passive `was` items), edit and read vectors; cross-fitted = mean over the 5 cross-fitted erasers. References pooled over bases: random isotropic unit vectors (10,000 per basis) and shuffled-label mean differences (100 verb-level permutations per basis), mean / 95th pct.

| Site | v·u_md | v·u_leace edit | v·u_leace read | v·edit (cross-fit) | v·read (cross-fit) | u_md·edit | u_md·read | random ref | shuffled-label ref |
|---:|---|---|---|---:|---:|---|---|---|---|
| 4 | 0.830 [0.820, 0.840] | 0.561 [0.552, 0.568] | 0.295 [0.287, 0.302] | 0.541 | 0.291 | 0.493 [0.474, 0.516] | 0.217 [0.205, 0.230] | 0.017 / 0.043 | 0.149 / 0.347 |
| 6 | 0.834 [0.824, 0.844] | 0.497 [0.483, 0.505] | 0.267 [0.261, 0.273] | 0.486 | 0.265 | 0.474 [0.462, 0.488] | 0.200 [0.190, 0.209] | 0.017 / 0.043 | 0.169 / 0.388 |
| 8 | 0.829 [0.823, 0.834] | 0.377 [0.366, 0.388] | 0.270 [0.265, 0.278] | 0.371 | 0.270 | 0.364 [0.351, 0.377] | 0.191 [0.184, 0.202] | 0.017 / 0.043 | 0.179 / 0.408 |
| 12 | 0.812 [0.808, 0.819] | 0.350 [0.343, 0.362] | 0.190 [0.184, 0.197] | 0.346 | 0.193 | 0.367 [0.352, 0.386] | 0.144 [0.138, 0.150] | 0.018 / 0.043 | 0.193 / 0.434 |
| 17 | 0.766 [0.753, 0.782] | 0.192 [0.175, 0.222] | 0.119 [0.109, 0.128] | 0.190 | 0.120 | 0.293 [0.281, 0.312] | 0.123 [0.116, 0.129] | 0.018 / 0.043 | 0.206 / 0.452 |

Variance of the projection of all 406 active items, as a multiple of the mean per-dimension variance (trace / d); mean [min, max] over bases.

| Site | along v | along u_md | |cos(edit, read)| |
|---:|---|---|---:|
| 4 | 56.334 [54.852, 58.163] | 82.263 [79.707, 84.967] | 0.545 |
| 6 | 74.229 [72.560, 76.896] | 107.751 [105.810, 111.261] | 0.436 |
| 8 | 81.083 [79.179, 83.411] | 120.323 [118.710, 122.828] | 0.368 |
| 12 | 96.687 [94.907, 99.096] | 147.330 [145.857, 149.643] | 0.263 |
| 17 | 113.916 [107.800, 118.415] | 197.328 [195.084, 199.669] | 0.237 |

## 2. Component patching: u_md vs v

### Held-out actives

IIA: cross-class held-out IIA, mean [min, max] over bases. PC frac: median positive-control fraction, mean over bases. Δ log P(O): intransitive base ← transitive source, mean over pairs (`per_pair`), 95% CI over pairs; the ratio row's CI is a paired pair bootstrap.

| Site | Direction | IIA | PC frac | same-class IIA | Δ log P(O) |
|---:|---|---|---:|---:|---|
| 4 | v | 0.446 [0.258, 0.580] | 0.621 | 0.986 | 2.088 [1.752, 2.421] |
| 4 | u_md | 0.134 [0.054, 0.212] | 0.296 | 0.985 | 1.237 [0.994, 1.498] |
| 4 | u_md / v | 0.301 | | | 0.592 [0.526, 0.652] |
| 6 | v | 0.656 [0.423, 0.803] | 0.720 | 0.991 | 2.327 [1.995, 2.655] |
| 6 | u_md | 0.217 [0.129, 0.312] | 0.402 | 0.987 | 1.559 [1.285, 1.839] |
| 6 | u_md / v | 0.330 | | | 0.670 [0.609, 0.730] |
| 8 | v | 0.782 [0.614, 0.887] | 0.777 | 0.990 | 2.457 [2.121, 2.793] |
| 8 | u_md | 0.279 [0.133, 0.406] | 0.458 | 0.988 | 1.724 [1.431, 2.026] |
| 8 | u_md / v | 0.356 | | | 0.702 [0.653, 0.745] |
| 12 | v | 0.898 [0.754, 0.964] | 0.870 | 0.990 | 2.641 [2.303, 2.972] |
| 12 | u_md | 0.365 [0.186, 0.565] | 0.486 | 0.992 | 1.827 [1.546, 2.109] |
| 12 | u_md / v | 0.407 | | | 0.692 [0.652, 0.729] |
| 17 | v | 0.973 [0.933, 1.000] | 1.067 | 0.987 | 2.858 [2.523, 3.186] |
| 17 | u_md | 0.689 [0.529, 0.855] | 0.659 | 0.989 | 2.264 [1.972, 2.569] |
| 17 | u_md / v | 0.708 | | | 0.792 [0.767, 0.817] |

### Passives (primary population, bad bases, D = T donor − I donor)

95% CI: three-way cluster bootstrap (2,000 draws, seed 17); ratio per draw on shared draws. Class: the declared passive-test rule with the site's δ_s.

| Site | Readout | D, v | class | D, u_md | class | u_md / v | stored D, v |
|---:|---|---|---|---|---|---|---:|
| 4 | M | 0.116 [0.069, 0.166] |  | 0.084 [0.040, 0.127] |  | 0.72 [0.50, 0.92] | 0.116 |
| 4 | log P(O) | 0.104 [0.057, 0.156] | NO RISE | 0.077 [0.033, 0.122] | NO RISE | 0.74 [0.50, 0.95] | 0.104 |
| 4 | log P(" by") | 0.577 [0.498, 0.660] | RISE | 0.446 [0.372, 0.525] | RISE | 0.77 [0.72, 0.83] | 0.577 |
| 4 | log P(".") | 0.124 [0.065, 0.186] | NO RISE | 0.069 [0.017, 0.124] | NO RISE | 0.56 [0.24, 0.73] | 0.124 |
| 6 | M | 0.267 [0.207, 0.330] |  | 0.097 [0.051, 0.143] |  | 0.36 [0.24, 0.46] | 0.267 |
| 6 | log P(O) | 0.226 [0.170, 0.287] | NO RISE | 0.078 [0.032, 0.124] | NO RISE | 0.34 [0.18, 0.46] | 0.226 |
| 6 | log P(" by") | 0.688 [0.598, 0.782] | RISE | 0.637 [0.548, 0.732] | RISE | 0.93 [0.88, 0.97] | 0.688 |
| 6 | log P(".") | 0.088 [0.031, 0.149] | NO RISE | 0.142 [0.083, 0.208] | NO RISE | 1.61 [1.30, 2.77] | 0.088 |
| 8 | M | 0.430 [0.354, 0.516] |  | 0.159 [0.107, 0.212] |  | 0.37 [0.29, 0.43] | 0.430 |
| 8 | log P(O) | 0.363 [0.295, 0.438] | NO RISE | 0.130 [0.082, 0.179] | NO RISE | 0.36 [0.27, 0.43] | 0.363 |
| 8 | log P(" by") | 0.719 [0.625, 0.815] | RISE | 0.631 [0.541, 0.725] | RISE | 0.88 [0.83, 0.92] | 0.719 |
| 8 | log P(".") | -0.006 [-0.057, 0.045] | NO RISE | 0.129 [0.078, 0.182] | NO RISE | -22.59 [-79.05, 72.74] | -0.006 |
| 12 | M | 1.343 [1.174, 1.527] |  | 0.241 [0.171, 0.317] |  | 0.18 [0.14, 0.21] | 1.343 |
| 12 | log P(O) | 1.167 [1.015, 1.332] | RISE | 0.172 [0.112, 0.237] | NO RISE | 0.15 [0.10, 0.18] | 1.167 |
| 12 | log P(" by") | 0.806 [0.720, 0.894] | RISE | 0.901 [0.797, 1.011] | RISE | 1.12 [1.07, 1.18] | 0.806 |
| 12 | log P(".") | -0.538 [-0.605, -0.475] | FALL | -0.016 [-0.072, 0.038] | NO RISE | 0.03 [-0.08, 0.13] | -0.538 |
| 17 | M | 4.866 [4.523, 5.196] |  | 2.554 [2.294, 2.802] |  | 0.52 [0.50, 0.55] | 4.866 |
| 17 | log P(O) | 3.583 [3.357, 3.795] | RISE | 2.237 [2.010, 2.448] | RISE | 0.62 [0.59, 0.66] | 3.583 |
| 17 | log P(" by") | -0.929 [-1.096, -0.763] | FALL | 0.436 [0.355, 0.514] | NO RISE | -0.47 [-0.61, -0.36] | -0.929 |
| 17 | log P(".") | -1.995 [-2.221, -1.761] | FALL | -0.617 [-0.724, -0.514] | FALL | 0.31 [0.28, 0.34] | -1.994 |

## 3. Mean ablation in natural runs

h ← h + (μ − h·u) u at the verb's last token, μ = grand mean of h·u over the frame's items (per basis). Natural = unpatched readouts computed in the same batches as the ablations (zero basis); the gap from the separate natural pass is in `summary_ablation.csv` (gap_nat_pass). Actives: T − I gap over the 29 pairs (each item averaged over the 3 bases holding its pair out); 95% CI over pairs. Passives: good − bad gap over the 64 primary pairs (each item averaged over its 3 plan bases); 95% CI two-way (pairs within band, contexts). Random: 10 random unit directions per basis, mean [min, max] of the share's point estimate.

| Site | Frame | Gap | natural | ablated v | share removed, v | ablated u_md | share removed, u_md | share removed, random |
|---:|---|---|---:|---:|---|---:|---|---|
| 4 | active | T − I log P(O) | 2.647 | 1.508 | 0.430 [0.351, 0.511] | 2.066 | 0.219 [0.160, 0.280] | -0.001 [-0.002, -0.000] |
| 4 | active | T − I M | 4.863 | 3.115 | 0.359 [0.302, 0.417] | 4.026 | 0.172 [0.132, 0.209] | 0.000 [-0.001, 0.001] |
| 4 | passive | good − bad log P(" by") | 0.713 | 0.429 | 0.398 [0.269, 0.752] | 0.523 | 0.266 [0.177, 0.503] | 0.001 [-0.001, 0.002] |
| 4 | passive | good − bad log P(O) | -0.221 | -0.306 | -0.386 [-3.561, 1.625] | -0.281 | -0.273 [-2.588, 1.205] | 0.002 [-0.002, 0.007] |
| 6 | active | T − I log P(O) | 2.647 | 1.226 | 0.537 [0.456, 0.615] | 1.889 | 0.287 [0.219, 0.354] | -0.001 [-0.002, -0.000] |
| 6 | active | T − I M | 4.863 | 2.600 | 0.465 [0.406, 0.520] | 3.730 | 0.233 [0.188, 0.277] | 0.000 [-0.000, 0.001] |
| 6 | passive | good − bad log P(" by") | 0.713 | 0.419 | 0.413 [0.277, 0.806] | 0.478 | 0.330 [0.223, 0.638] | 0.001 [-0.000, 0.003] |
| 6 | passive | good − bad log P(O) | -0.221 | -0.323 | -0.465 [-4.312, 1.882] | -0.269 | -0.218 [-1.915, 0.897] | 0.003 [0.000, 0.005] |
| 8 | active | T − I log P(O) | 2.647 | 0.980 | 0.630 [0.545, 0.712] | 1.743 | 0.342 [0.275, 0.408] | -0.001 [-0.002, -0.000] |
| 8 | active | T − I M | 4.863 | 2.149 | 0.558 [0.502, 0.616] | 3.460 | 0.288 [0.246, 0.332] | -0.000 [-0.001, 0.001] |
| 8 | passive | good − bad log P(" by") | 0.713 | 0.452 | 0.366 [0.246, 0.710] | 0.522 | 0.267 [0.181, 0.512] | 0.001 [0.000, 0.002] |
| 8 | passive | good − bad log P(O) | -0.221 | -0.340 | -0.540 [-4.996, 2.151] | -0.275 | -0.246 [-2.290, 1.040] | 0.003 [-0.001, 0.007] |
| 12 | active | T − I log P(O) | 2.647 | 0.690 | 0.739 [0.653, 0.824] | 1.695 | 0.360 [0.305, 0.413] | -0.001 [-0.003, -0.000] |
| 12 | active | T − I M | 4.863 | 1.430 | 0.706 [0.637, 0.774] | 3.234 | 0.335 [0.293, 0.377] | -0.000 [-0.001, 0.001] |
| 12 | passive | good − bad log P(" by") | 0.713 | 0.453 | 0.365 [0.244, 0.706] | 0.449 | 0.371 [0.251, 0.709] | 0.001 [0.001, 0.002] |
| 12 | passive | good − bad log P(O) | -0.221 | -0.483 | -1.189 [-10.888, 4.873] | -0.294 | -0.332 [-3.020, 1.317] | 0.002 [-0.003, 0.009] |
| 17 | active | T − I log P(O) | 2.647 | 0.114 | 0.957 [0.867, 1.046] | 1.133 | 0.572 [0.503, 0.642] | -0.000 [-0.001, 0.001] |
| 17 | active | T − I M | 4.863 | 0.419 | 0.914 [0.841, 0.990] | 2.282 | 0.531 [0.475, 0.590] | 0.000 [-0.000, 0.001] |
| 17 | passive | good − bad log P(" by") | 0.713 | 0.678 | 0.049 [0.010, 0.103] | 0.589 | 0.174 [0.119, 0.317] | 0.001 [-0.001, 0.003] |
| 17 | passive | good − bad log P(O) | -0.221 | -0.567 | -1.572 [-14.623, 7.142] | -0.420 | -0.905 [-8.736, 4.134] | 0.004 [-0.004, 0.008] |
