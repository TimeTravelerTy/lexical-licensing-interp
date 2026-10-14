# Round 5, A2: exact decomposition

Spec: `plan.md`, A2. Freq contexts unless stated; Δh̄ and affinities are recomputed in each context draw.

## A2. Exact decomposition

- Identity L₁ = Δh · (u_T1 − u_I1): max error in the run 6.4e-05 (every pair and prefix, all sets; fail-fast ≤ 1e-3). This is algebraic; it checks the implementation.
- P = (u_T1 − u_I1) · Δh̄ equals the freq-mean L₁ for every pair: max |difference| 6.5e-06.
- ‖Δh̄‖ = 54.31; per-context deviation share 0.44 (mean ‖Δh_c − Δh̄‖² / mean ‖Δh_c‖²).

### Per pair: P = projection of u_T1 − u_I1 on Δh̄ (= freq-mean L₁)

| | Head | Tail | XTail |
|---|---|---|---|
| P, all pairs | +3.87 [+3.18, +4.49] | +1.00 [+0.50, +1.49] | +0.03 [-0.33, +0.37] |
| P, single-token pairs (= freq-mean L) | +3.87 [+3.18, +4.49] | +2.26 (n = 1) | — |

Single-token pairs: 27 (Head 26, Tail 1, XTail 0).

### Multi-token pairs: first vs later subtokens (freq)

| Band | pairs | mean L | mean L₁ | mean L_rest | L₁ / L | L_rest / L |
|---|---:|---|---|---|---|---|
| Tail | 49 | +2.86 [+2.16, +3.55] | +0.97 [+0.46, +1.46] | +1.89 [+1.28, +2.53] | +0.34 [+0.18, +0.50] | +0.66 [+0.50, +0.82] |
| XTail | 50 | +1.59 [+1.03, +2.18] | +0.03 [-0.33, +0.37] | +1.57 [+0.98, +2.15] | +0.02 [-0.24, +0.23] | +0.98 [+0.77, +1.24] |

By exact token-count cell (multi-token pairs with ≥ 5 pairs in the cell and band):

| Cell | Band | pairs | mean L | mean L₁ | mean L_rest | L₁ / L |
|---|---|---:|---|---|---|---|
| 1×2 | Tail | 5 | +1.60 [-2.30, +4.85] | -0.66 [-3.19, +1.53] | +2.26 [+0.58, +3.78] | not interpreted |
| 2×1 | Tail | 5 | +1.45 [-0.61, +3.52] | +2.57 [+1.51, +3.50] | -1.13 [-3.00, +0.15] | not interpreted |
| 2×2 | Tail | 25 | +3.45 [+2.51, +4.32] | +1.15 [+0.45, +1.80] | +2.30 [+1.46, +3.15] | +0.33 [+0.15, +0.52] |
| 2×2 | XTail | 15 | +1.71 [+1.21, +2.20] | -0.34 [-1.02, +0.34] | +2.05 [+1.30, +2.74] | -0.20 [-0.68, +0.18] |
| 2×3 | Tail | 5 | +3.51 [+2.22, +5.34] | +0.16 [-0.82, +1.27] | +3.35 [+2.08, +4.73] | +0.05 [-0.34, +0.29] |
| 2×3 | XTail | 11 | +1.67 [+0.46, +3.14] | +0.08 [-0.77, +0.97] | +1.59 [+0.43, +2.62] | +0.05 [-0.99, +0.52] |
| 3×2 | Tail | 7 | +2.38 [+0.88, +3.88] | +1.14 [+0.31, +2.23] | +1.24 [-0.33, +2.99] | +0.48 [+0.10, +1.24] |
| 3×2 | XTail | 12 | +1.29 [+0.10, +2.44] | +0.36 [-0.14, +0.85] | +0.92 [-0.25, +2.21] | +0.28 [-0.20, +1.32] |
| 3×3 | XTail | 6 | +2.04 [-0.34, +4.71] | +0.02 [-1.27, +1.21] | +2.03 [-0.49, +4.57] | not interpreted |

XTail shortfall: L̄_Head − L_XTail = (L̄_Head − L₁,XTail) − L_rest,XTail: +2.27 [+1.41, +3.09] = +3.84 [+3.07, +4.55] − (+1.57 [+0.98, +2.15]).

### Per verb: passive-slot affinity a_V = u_V1 · Δh̄

| Verbs | role | n | Zipf range | slope of a_V | slope of cos | slope of ‖u‖ |
|---|---|---:|---|---|---|---|
| single-token, curated | T | 32 | 2.68–5.16 | +0.106 [-0.924, +1.063] | +0.002 [-0.015, +0.019] | +0.040 [+0.006, +0.071] |
| single-token, curated | I | 33 | 2.59–5.20 | -1.461 [-1.963, -0.877] | -0.026 [-0.035, -0.016] | -0.000 [-0.036, +0.024] |
| first-subtoken fragments, curated | T | 94 | 1.32–3.18 | +0.002 [-0.319, +0.290] | -0.000 [-0.005, +0.004] | -0.002 [-0.025, +0.021] |
| first-subtoken fragments, curated | I | 93 | 1.23–3.12 | -1.040 [-1.529, -0.585] | -0.016 [-0.024, -0.009] | -0.003 [-0.026, +0.020] |
| single-token, curated + pool | T | 305 | 2.58–5.16 | +0.093 [-0.258, +0.482] | +0.002 [-0.004, +0.008] | -0.005 [-0.022, +0.010] |
| single-token, curated + pool | I | 39 | 2.59–5.20 | -1.500 [-1.988, -0.971] | -0.027 [-0.036, -0.018] | -0.008 [-0.043, +0.016] |

Head class separation (curated): AUC(a_V, T vs I) = 0.95 (26 T, 26 I).

### Gate for A4 (single-token participles; verbs within role × band cells, Δh̄ re-drawn)

| Population | Head T | rare T | Head I | rare I | rare bands | D_T | D_I | D_L = D_T − D_I |
|---|---:|---:|---:|---:|---|---|---|---|
| curated + pool (gate) | 282 | 23 | 30 | 9 | tail | +0.484 [-0.121, +1.131] | -1.630 [-2.471, -0.728] | +2.115 [+1.081, +3.187] |
| curated only | 26 | 6 | 26 | 7 | tail | +0.878 [-0.437, +2.540] | -1.873 [-2.878, -0.805] | +2.751 [+1.001, +4.757] |
| curated + pool, Head Zipf ≤ 4.0 | 150 | 23 | 13 | 9 | tail | +0.522 [-0.079, +1.199] | -0.987 [-1.893, -0.064] | +1.508 [+0.386, +2.578] |

Gate 1 (ΔL deficit, freq): met; gate 2 (D_L CI above 0): met; gate 3: met. **A4 runs** (a Tail-singleton intervention; transfer to XTail untested).

