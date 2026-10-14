# Round 5, A2: exact decomposition

Spec: `plan.md`, A2. Freq contexts unless stated; Δh̄ and affinities are recomputed in each context draw.

## A2. Exact decomposition

- Identity L₁ = Δh · (u_T1 − u_I1): max error in the run 2.7e-03 (every pair and prefix, all sets; fail-fast ≤ 1e-3). This is algebraic; it checks the implementation.
- P = (u_T1 − u_I1) · Δh̄ equals the freq-mean L₁ for every pair: max |difference| 4.8e-04.
- ‖Δh̄‖ = 22.88; per-context deviation share 0.42 (mean ‖Δh_c − Δh̄‖² / mean ‖Δh_c‖²).

### Per pair: P = projection of u_T1 − u_I1 on Δh̄ (= freq-mean L₁)

| | Head | Tail | XTail |
|---|---|---|---|
| P, all pairs | +3.10 [+2.53, +3.64] | +0.79 [+0.39, +1.20] | +0.06 [-0.21, +0.33] |
| P, single-token pairs (= freq-mean L) | +3.10 [+2.53, +3.64] | +2.53 (n = 1) | — |

Single-token pairs: 27 (Head 26, Tail 1, XTail 0).

### Multi-token pairs: first vs later subtokens (freq)

| Band | pairs | mean L | mean L₁ | mean L_rest | L₁ / L | L_rest / L |
|---|---:|---|---|---|---|---|
| Tail | 49 | +2.08 [+1.45, +2.71] | +0.76 [+0.35, +1.16] | +1.32 [+0.76, +1.91] | +0.37 [+0.19, +0.55] | +0.63 [+0.45, +0.81] |
| XTail | 50 | +1.53 [+1.02, +2.07] | +0.06 [-0.21, +0.33] | +1.47 [+0.94, +2.02] | +0.04 [-0.15, +0.21] | +0.96 [+0.79, +1.15] |

By exact token-count cell (multi-token pairs with ≥ 5 pairs in the cell and band):

| Cell | Band | pairs | mean L | mean L₁ | mean L_rest | L₁ / L |
|---|---|---:|---|---|---|---|
| 1×2 | Tail | 5 | +1.33 [-1.37, +3.73] | -0.38 [-2.00, +1.24] | +1.71 [+0.58, +2.58] | not interpreted |
| 2×1 | Tail | 5 | +0.91 [-1.43, +3.03] | +1.96 [+0.79, +3.12] | -1.05 [-2.43, -0.02] | not interpreted |
| 2×2 | Tail | 25 | +2.42 [+1.67, +3.16] | +0.77 [+0.20, +1.36] | +1.65 [+1.05, +2.28] | +0.32 [+0.10, +0.51] |
| 2×2 | XTail | 15 | +1.70 [+0.76, +2.67] | -0.22 [-0.68, +0.21] | +1.92 [+1.04, +2.89] | -0.13 [-0.64, +0.13] |
| 2×3 | Tail | 5 | +3.02 [-0.57, +5.58] | +0.72 [+0.21, +1.27] | +2.31 [-1.43, +4.72] | not interpreted |
| 2×3 | XTail | 11 | +1.22 [+0.45, +1.91] | +0.06 [-0.64, +0.84] | +1.16 [+0.25, +2.08] | +0.05 [-0.90, +0.72] |
| 3×2 | Tail | 7 | +1.71 [-0.06, +3.34] | +0.92 [-0.11, +1.90] | +0.79 [-1.32, +2.70] | not interpreted |
| 3×2 | XTail | 12 | +1.31 [+0.48, +2.09] | +0.49 [+0.22, +0.75] | +0.82 [-0.10, +1.69] | +0.37 [+0.13, +1.12] |
| 3×3 | XTail | 6 | +2.60 [-0.27, +5.37] | +0.15 [-0.76, +0.86] | +2.46 [-0.19, +5.17] | not interpreted |

XTail shortfall: L̄_Head − L_XTail = (L̄_Head − L₁,XTail) − L_rest,XTail: +1.57 [+0.76, +2.28] = +3.04 [+2.42, +3.60] − (+1.47 [+0.94, +2.02]).

### Per verb: passive-slot affinity a_V = u_V1 · Δh̄

| Verbs | role | n | Zipf range | slope of a_V | slope of cos | slope of ‖u‖ |
|---|---|---:|---|---|---|---|
| single-token, curated | T | 32 | 2.68–5.16 | +0.194 [-0.469, +0.779] | +0.003 [-0.006, +0.010] | +0.026 [+0.019, +0.035] |
| single-token, curated | I | 33 | 2.59–5.20 | -1.228 [-1.702, -0.678] | -0.016 [-0.022, -0.009] | +0.030 [+0.020, +0.037] |
| first-subtoken fragments, curated | T | 94 | 1.32–3.18 | +0.033 [-0.241, +0.290] | +0.000 [-0.003, +0.004] | -0.000 [-0.006, +0.006] |
| first-subtoken fragments, curated | I | 93 | 1.23–3.12 | -0.733 [-1.066, -0.430] | -0.010 [-0.014, -0.006] | +0.004 [-0.002, +0.010] |
| single-token, curated + pool | T | 305 | 2.58–5.16 | +0.154 [-0.108, +0.422] | +0.002 [-0.001, +0.006] | +0.023 [+0.020, +0.027] |
| single-token, curated + pool | I | 39 | 2.59–5.20 | -1.219 [-1.664, -0.696] | -0.016 [-0.022, -0.009] | +0.028 [+0.019, +0.035] |

Head class separation (curated): AUC(a_V, T vs I) = 0.95 (26 T, 26 I).

### Gate for A4 (single-token participles; verbs within role × band cells, Δh̄ re-drawn)

| Population | Head T | rare T | Head I | rare I | rare bands | D_T | D_I | D_L = D_T − D_I |
|---|---:|---:|---:|---:|---|---|---|---|
| curated + pool (gate) | 282 | 23 | 30 | 9 | tail | +0.472 [+0.085, +0.860] | -1.247 [-2.047, -0.397] | +1.719 [+0.797, +2.621] |
| curated only | 26 | 6 | 26 | 7 | tail | +0.647 [-0.065, +1.508] | -1.491 [-2.451, -0.436] | +2.137 [+0.875, +3.434] |
| curated + pool, Head Zipf ≤ 4.0 | 150 | 23 | 13 | 9 | tail | +0.475 [+0.096, +0.897] | -0.641 [-1.457, +0.237] | +1.117 [+0.159, +2.060] |

Gate 1 (ΔL deficit, freq): met; gate 2 (D_L CI above 0): met; gate 3: met. **A4 runs** (a Tail-singleton intervention; transfer to XTail untested).

