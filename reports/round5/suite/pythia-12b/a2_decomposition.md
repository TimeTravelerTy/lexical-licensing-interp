# Round 5, A2: exact decomposition

Spec: `plan.md`, A2. Freq contexts unless stated; Δh̄ and affinities are recomputed in each context draw.

## A2. Exact decomposition

- Identity L₁ = Δh · (u_T1 − u_I1): max error in the run 5.5e-05 (every pair and prefix, all sets; fail-fast ≤ 1e-3). This is algebraic; it checks the implementation.
- P = (u_T1 − u_I1) · Δh̄ equals the freq-mean L₁ for every pair: max |difference| 6.2e-06.
- ‖Δh̄‖ = 51.87; per-context deviation share 0.46 (mean ‖Δh_c − Δh̄‖² / mean ‖Δh_c‖²).

### Per pair: P = projection of u_T1 − u_I1 on Δh̄ (= freq-mean L₁)

| | Head | Tail | XTail |
|---|---|---|---|
| P, all pairs | +3.92 [+3.15, +4.59] | +1.04 [+0.56, +1.53] | +0.00 [-0.33, +0.32] |
| P, single-token pairs (= freq-mean L) | +3.92 [+3.15, +4.59] | +2.64 (n = 1) | — |

Single-token pairs: 27 (Head 26, Tail 1, XTail 0).

### Multi-token pairs: first vs later subtokens (freq)

| Band | pairs | mean L | mean L₁ | mean L_rest | L₁ / L | L_rest / L |
|---|---:|---|---|---|---|---|
| Tail | 49 | +3.07 [+2.35, +3.79] | +1.01 [+0.50, +1.50] | +2.06 [+1.44, +2.69] | +0.33 [+0.19, +0.47] | +0.67 [+0.53, +0.81] |
| XTail | 50 | +2.04 [+1.38, +2.73] | +0.00 [-0.33, +0.32] | +2.04 [+1.38, +2.70] | +0.00 [-0.19, +0.15] | +1.00 [+0.85, +1.19] |

By exact token-count cell (multi-token pairs with ≥ 5 pairs in the cell and band):

| Cell | Band | pairs | mean L | mean L₁ | mean L_rest | L₁ / L |
|---|---|---:|---|---|---|---|
| 1×2 | Tail | 5 | +2.10 [-1.77, +5.39] | -0.39 [-2.91, +2.15] | +2.49 [+0.72, +4.28] | not interpreted |
| 2×1 | Tail | 5 | +1.14 [-1.42, +3.66] | +2.29 [+1.10, +3.57] | -1.16 [-3.24, +0.25] | not interpreted |
| 2×2 | Tail | 25 | +3.56 [+2.58, +4.48] | +1.17 [+0.41, +1.88] | +2.39 [+1.56, +3.20] | +0.33 [+0.14, +0.50] |
| 2×2 | XTail | 15 | +2.23 [+1.60, +2.93] | -0.36 [-0.98, +0.30] | +2.59 [+1.82, +3.26] | -0.16 [-0.51, +0.12] |
| 2×3 | Tail | 5 | +3.65 [+2.24, +5.94] | +0.36 [-0.50, +1.33] | +3.29 [+1.84, +4.93] | +0.10 [-0.18, +0.30] |
| 2×3 | XTail | 11 | +2.05 [+0.76, +3.56] | -0.09 [-0.75, +0.56] | +2.14 [+0.88, +3.34] | -0.05 [-0.59, +0.25] |
| 3×2 | Tail | 7 | +3.16 [+1.81, +4.38] | +1.15 [+0.36, +2.09] | +2.01 [+0.61, +3.45] | +0.37 [+0.12, +0.73] |
| 3×2 | XTail | 12 | +1.66 [+0.10, +3.20] | +0.47 [-0.10, +1.08] | +1.19 [-0.23, +2.64] | +0.28 [-0.12, +1.46] |
| 3×3 | XTail | 6 | +2.58 [-0.42, +5.82] | +0.06 [-1.42, +1.32] | +2.53 [-0.41, +5.55] | not interpreted |

XTail shortfall: L̄_Head − L_XTail = (L̄_Head − L₁,XTail) − L_rest,XTail: +1.88 [+0.91, +2.79] = +3.92 [+3.09, +4.68] − (+2.04 [+1.38, +2.70]).

### Per verb: passive-slot affinity a_V = u_V1 · Δh̄

| Verbs | role | n | Zipf range | slope of a_V | slope of cos | slope of ‖u‖ |
|---|---|---:|---|---|---|---|
| single-token, curated | T | 32 | 2.68–5.16 | +0.026 [-1.043, +0.960] | +0.001 [-0.017, +0.016] | +0.054 [+0.015, +0.089] |
| single-token, curated | I | 33 | 2.59–5.20 | -1.815 [-2.371, -1.184] | -0.031 [-0.041, -0.021] | +0.011 [-0.025, +0.036] |
| first-subtoken fragments, curated | T | 94 | 1.32–3.18 | +0.059 [-0.287, +0.375] | +0.001 [-0.004, +0.005] | -0.004 [-0.032, +0.025] |
| first-subtoken fragments, curated | I | 93 | 1.23–3.12 | -1.059 [-1.522, -0.619] | -0.016 [-0.023, -0.009] | -0.009 [-0.036, +0.018] |
| single-token, curated + pool | T | 305 | 2.58–5.16 | +0.052 [-0.297, +0.430] | +0.001 [-0.005, +0.007] | +0.007 [-0.011, +0.024] |
| single-token, curated + pool | I | 39 | 2.59–5.20 | -1.780 [-2.302, -1.194] | -0.031 [-0.040, -0.021] | +0.002 [-0.036, +0.027] |

Head class separation (curated): AUC(a_V, T vs I) = 0.95 (26 T, 26 I).

### Gate for A4 (single-token participles; verbs within role × band cells, Δh̄ re-drawn)

| Population | Head T | rare T | Head I | rare I | rare bands | D_T | D_I | D_L = D_T − D_I |
|---|---:|---:|---:|---:|---|---|---|---|
| curated + pool (gate) | 282 | 23 | 30 | 9 | tail | +0.490 [-0.103, +1.124] | -1.759 [-2.701, -0.777] | +2.249 [+1.143, +3.425] |
| curated only | 26 | 6 | 26 | 7 | tail | +0.786 [-0.556, +2.382] | -2.080 [-3.219, -0.894] | +2.866 [+1.049, +4.841] |
| curated + pool, Head Zipf ≤ 4.0 | 150 | 23 | 13 | 9 | tail | +0.552 [-0.019, +1.188] | -0.946 [-1.923, +0.022] | +1.498 [+0.336, +2.614] |

Gate 1 (ΔL deficit, freq): met; gate 2 (D_L CI above 0): met; gate 3: met. **A4 runs** (a Tail-singleton intervention; transfer to XTail untested).

