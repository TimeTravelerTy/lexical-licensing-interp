# Gemma replication: exploratory / secondary tables

**Exploratory / secondary; the declared results are in `transfer.md` and `controls_transfer.md`.** Script `scripts/analyze_gemma_exploratory.py` (definitions in its docstring); rules in `exploratory.md`.

## 2. In-range equivalence test (passive, secondary)

D_in = R(t_T) − R(t_I) on bad passive items, nats, 95% CI / 90% CI (two-way bootstrap); classes against the declared δ_s. The in-range outcome applies the declared surface / abstract / mixed rule to (O class, *by* class); the active-donor outcome is the declared result at the same site.

| model | site | depth | role | δ_s | D_in log P(O) | D_in log P(" by") | in-range outcome | active-donor outcome (declared) |
|---|---|---|---|---|---|---|---|---|
| Gemma-3-1B | 0 | 0.000 | exploratory (IIA-matched to Pythia 4) | 0.288 | +0.02 [+0.00, +0.03] / 90% [+0.00, +0.03] NO RISE | +0.07 [+0.05, +0.09] / 90% [+0.05, +0.08] NO RISE | nothing moves (active-specific) | unresolved |
| Gemma-3-1B | 1 | 0.038 | exploratory (IIA-matched to Pythia 6) | 0.410 | +0.01 [-0.01, +0.03] / 90% [-0.01, +0.03] NO RISE | +0.09 [+0.06, +0.12] / 90% [+0.07, +0.12] NO RISE | nothing moves (active-specific) | abstract (takes an object) |
| Gemma-3-1B | 4 | 0.154 | exploratory (IIA-matched to Pythia 8) | 0.519 | +0.10 [+0.07, +0.13] / 90% [+0.07, +0.13] NO RISE | +0.24 [+0.18, +0.30] / 90% [+0.19, +0.29] NO RISE | nothing moves (active-specific) | mixed |
| Pythia-1.4B | 4 | 0.167 | stored | 0.418 | +0.04 [+0.01, +0.07] / 90% [+0.01, +0.06] NO RISE | +0.37 [+0.33, +0.41] / 90% [+0.33, +0.41] NO RISE | nothing moves (active-specific) | abstract (takes an object) |
| Gemma-3-1B | 5 | 0.192 | declared (Pythia 0.8 rule) | 0.537 | +0.27 [+0.21, +0.33] / 90% [+0.22, +0.32] NO RISE | +0.31 [+0.24, +0.39] / 90% [+0.25, +0.37] NO RISE | nothing moves (active-specific) | mixed |
| Pythia-1.4B | 6 | 0.250 | stored | 0.465 | +0.06 [+0.04, +0.09] / 90% [+0.04, +0.08] NO RISE | +0.37 [+0.33, +0.41] / 90% [+0.34, +0.41] NO RISE | nothing moves (active-specific) | abstract (takes an object) |
| Gemma-3-1B | 8 | 0.308 | declared (Pythia 0.9 rule) | 0.531 | +0.50 [+0.45, +0.55] / 90% [+0.46, +0.54] unresolved | +0.36 [+0.32, +0.41] / 90% [+0.32, +0.40] NO RISE | unresolved | mixed |
| Pythia-1.4B | 8 | 0.333 | stored | 0.491 | +0.09 [+0.06, +0.11] / 90% [+0.07, +0.10] NO RISE | +0.31 [+0.28, +0.34] / 90% [+0.28, +0.34] NO RISE | nothing moves (active-specific) | abstract (takes an object) |
| Pythia-1.4B | 10 | 0.417 | stored | 0.501 | +0.17 [+0.15, +0.20] / 90% [+0.15, +0.19] NO RISE | +0.32 [+0.28, +0.35] / 90% [+0.29, +0.35] NO RISE | nothing moves (active-specific) | mixed |
| Gemma-3-1B | 12 | 0.462 | declared (within 0.05 of max) | 0.560 | +0.40 [+0.37, +0.44] / 90% [+0.37, +0.44] NO RISE | +0.31 [+0.28, +0.35] / 90% [+0.29, +0.34] NO RISE | nothing moves (active-specific) | mixed |
| Pythia-1.4B | 12 | 0.500 | stored | 0.528 | +0.26 [+0.22, +0.30] / 90% [+0.23, +0.29] NO RISE | +0.29 [+0.26, +0.32] / 90% [+0.27, +0.31] NO RISE | nothing moves (active-specific) | mixed |
| Gemma-3-1B | 15 | 0.577 | declared (frozen rule) | 0.576 | +0.43 [+0.40, +0.47] / 90% [+0.40, +0.46] NO RISE | +0.19 [+0.17, +0.21] / 90% [+0.18, +0.20] NO RISE | nothing moves (active-specific) | mixed |
| Pythia-1.4B | 14 | 0.583 | stored | 0.551 | +0.34 [+0.29, +0.39] / 90% [+0.30, +0.38] NO RISE | +0.21 [+0.19, +0.22] / 90% [+0.19, +0.22] NO RISE | nothing moves (active-specific) | surface (object next) |
| Pythia-1.4B | 16 | 0.667 | stored | 0.566 | +0.45 [+0.40, +0.50] / 90% [+0.41, +0.50] NO RISE | +0.10 [+0.09, +0.11] / 90% [+0.09, +0.11] NO RISE | nothing moves (active-specific) | surface (object next) |
| Pythia-1.4B | 17 | 0.708 | stored | 0.572 | +0.42 [+0.38, +0.47] / 90% [+0.39, +0.46] NO RISE | +0.04 [+0.03, +0.05] / 90% [+0.03, +0.05] NO RISE | nothing moves (active-specific) | surface (object next) |

## 3. Balanced shuffled-label control

Held-out cross-class IIA. Balanced: per permutation, the mean over the final run's 3 splits × 5 folds; mean / 95th percentile / max over the permutations, and the largest single-fold value. Unconstrained: the final run's shuffled-label control (one random permutation per fold), mean and fold max. Trained: the final run's DAS. max |phi| = largest correlation between true and permuted training-verb classes over all permutations and folds.

| model | site | depth | role | perms | folds / perm | balanced mean | balanced p95 | balanced max | balanced fold max | max abs phi | unconstrained mean | unconstrained fold max | trained DAS | random (norm-matched) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemma-3-1B | 0 | 0.000 | exploratory (IIA-matched to Pythia 4) | 20 | 15 | 0.106 | 0.121 | 0.133 | 0.389 | 0.043 | 0.121 | 0.369 | 0.342 | 0.006 |
| Gemma-3-1B | 1 | 0.038 | exploratory (IIA-matched to Pythia 6) | 20 | 15 | 0.206 | 0.248 | 0.257 | 0.634 | 0.043 | 0.240 | 0.569 | 0.631 | 0.020 |
| Gemma-3-1B | 4 | 0.154 | exploratory (IIA-matched to Pythia 8) | 20 | 15 | 0.240 | 0.294 | 0.312 | 0.657 | 0.043 | 0.263 | 0.480 | 0.810 | 0.009 |
| Gemma-3-1B | 5 | 0.192 | declared (Pythia 0.8 rule) | 20 | 15 | 0.259 | 0.296 | 0.301 | 0.712 | 0.043 | 0.290 | 0.550 | 0.851 | 0.005 |
| Gemma-3-1B | 8 | 0.308 | declared (Pythia 0.9 rule) | 20 | 15 | 0.309 | 0.336 | 0.352 | 0.751 | 0.043 | 0.351 | 0.768 | 0.875 | 0.006 |
| Pythia-1.4B | 8 | 0.333 | stored | 20 | 15 | 0.238 | 0.269 | 0.285 | 0.549 | 0.043 | 0.199 | 0.327 | 0.782 | 0.018 |
| Gemma-3-1B | 12 | 0.462 | declared (within 0.05 of max) | 20 | 15 | 0.345 | 0.398 | 0.406 | 0.708 | 0.043 | 0.410 | 0.984 | 0.944 | 0.007 |
| Pythia-1.4B | 12 | 0.500 | stored | 20 | 15 | 0.279 | 0.314 | 0.327 | 0.591 | 0.043 | 0.242 | 0.494 | 0.898 | 0.019 |
| Gemma-3-1B | 15 | 0.577 | declared (frozen rule) | 20 | 15 | 0.374 | 0.426 | 0.450 | 0.823 | 0.043 | 0.407 | 0.988 | 0.979 | 0.009 |
| Pythia-1.4B | 17 | 0.708 | stored | 20 | 15 | 0.392 | 0.442 | 0.451 | 0.813 | 0.043 | 0.341 | 0.621 | 0.973 | 0.019 |

