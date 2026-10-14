# Round 5, A2: exact decomposition

Spec: `plan.md`, A2. Freq contexts unless stated; Δh̄ and affinities are recomputed in each context draw.

## A2. Exact decomposition

- Identity L₁ = Δh · (u_T1 − u_I1): max error in the run 4.0e-05 (every pair and prefix, all sets; fail-fast ≤ 1e-3). This is algebraic; it checks the implementation.
- P = (u_T1 − u_I1) · Δh̄ equals the freq-mean L₁ for every pair: max |difference| 3.7e-06.
- ‖Δh̄‖ = 58.39; per-context deviation share 0.44 (mean ‖Δh_c − Δh̄‖² / mean ‖Δh_c‖²).

### Per pair: P = projection of u_T1 − u_I1 on Δh̄ (= freq-mean L₁)

| | Head | Tail | XTail |
|---|---|---|---|
| P, all pairs | +3.84 [+3.16, +4.46] | +0.98 [+0.45, +1.52] | +0.06 [-0.30, +0.39] |
| P, single-token pairs (= freq-mean L) | +3.84 [+3.16, +4.46] | +2.65 (n = 1) | — |

Single-token pairs: 27 (Head 26, Tail 1, XTail 0).

### Multi-token pairs: first vs later subtokens (freq)

| Band | pairs | mean L | mean L₁ | mean L_rest | L₁ / L | L_rest / L |
|---|---:|---|---|---|---|---|
| Tail | 49 | +2.83 [+2.12, +3.54] | +0.95 [+0.41, +1.49] | +1.88 [+1.26, +2.54] | +0.34 [+0.16, +0.50] | +0.66 [+0.50, +0.84] |
| XTail | 50 | +1.72 [+1.15, +2.28] | +0.06 [-0.30, +0.39] | +1.66 [+1.08, +2.26] | +0.03 [-0.19, +0.22] | +0.97 [+0.78, +1.19] |

By exact token-count cell (multi-token pairs with ≥ 5 pairs in the cell and band):

| Cell | Band | pairs | mean L | mean L₁ | mean L_rest | L₁ / L |
|---|---|---:|---|---|---|---|
| 1×2 | Tail | 5 | +2.09 [-2.07, +5.60] | -0.56 [-3.43, +1.67] | +2.65 [+0.87, +4.29] | not interpreted |
| 2×1 | Tail | 5 | +1.28 [-0.91, +3.47] | +2.40 [+1.14, +3.70] | -1.12 [-2.86, +0.06] | not interpreted |
| 2×2 | Tail | 25 | +3.28 [+2.36, +4.15] | +1.16 [+0.39, +1.91] | +2.12 [+1.38, +2.90] | +0.35 [+0.14, +0.54] |
| 2×2 | XTail | 15 | +1.50 [+0.97, +2.06] | -0.45 [-1.15, +0.30] | +1.95 [+1.24, +2.65] | -0.30 [-0.96, +0.17] |
| 2×3 | Tail | 5 | +4.01 [+2.39, +5.77] | +0.52 [-0.21, +1.29] | +3.49 [+1.47, +5.30] | +0.13 [-0.06, +0.42] |
| 2×3 | XTail | 11 | +2.22 [+0.89, +3.58] | -0.05 [-0.72, +0.63] | +2.27 [+1.05, +3.50] | -0.02 [-0.55, +0.24] |
| 3×2 | Tail | 7 | +2.24 [+0.48, +3.92] | +0.89 [-0.14, +1.96] | +1.35 [-0.43, +3.32] | +0.40 [-0.09, +1.46] |
| 3×2 | XTail | 12 | +1.40 [+0.01, +2.77] | +0.42 [-0.17, +0.93] | +0.98 [-0.31, +2.26] | +0.30 [-0.36, +1.35] |
| 3×3 | XTail | 6 | +1.90 [-0.28, +4.02] | +0.30 [-0.91, +1.14] | +1.60 [-0.98, +4.17] | not interpreted |

XTail shortfall: L̄_Head − L_XTail = (L̄_Head − L₁,XTail) − L_rest,XTail: +2.12 [+1.26, +2.94] = +3.78 [+3.00, +4.49] − (+1.66 [+1.08, +2.26]).

### Per verb: passive-slot affinity a_V = u_V1 · Δh̄

| Verbs | role | n | Zipf range | slope of a_V | slope of cos | slope of ‖u‖ |
|---|---|---:|---|---|---|---|
| single-token, curated | T | 32 | 2.68–5.16 | +0.247 [-0.745, +1.128] | +0.005 [-0.015, +0.023] | +0.017 [-0.007, +0.036] |
| single-token, curated | I | 33 | 2.59–5.20 | -1.578 [-2.089, -0.981] | -0.031 [-0.042, -0.019] | +0.011 [-0.016, +0.029] |
| first-subtoken fragments, curated | T | 94 | 1.32–3.18 | -0.015 [-0.377, +0.316] | -0.000 [-0.007, +0.006] | +0.004 [-0.010, +0.018] |
| first-subtoken fragments, curated | I | 93 | 1.23–3.12 | -1.062 [-1.537, -0.614] | -0.020 [-0.028, -0.011] | +0.004 [-0.013, +0.019] |
| single-token, curated + pool | T | 305 | 2.58–5.16 | +0.203 [-0.145, +0.583] | +0.004 [-0.003, +0.012] | -0.011 [-0.023, +0.000] |
| single-token, curated + pool | I | 39 | 2.59–5.20 | -1.547 [-2.025, -0.963] | -0.031 [-0.041, -0.020] | +0.007 [-0.019, +0.025] |

Head class separation (curated): AUC(a_V, T vs I) = 0.96 (26 T, 26 I).

### Gate for A4 (single-token participles; verbs within role × band cells, Δh̄ re-drawn)

| Population | Head T | rare T | Head I | rare I | rare bands | D_T | D_I | D_L = D_T − D_I |
|---|---:|---:|---:|---:|---|---|---|---|
| curated + pool (gate) | 282 | 23 | 30 | 9 | tail | +0.599 [+0.027, +1.201] | -1.493 [-2.432, -0.531] | +2.092 [+1.000, +3.233] |
| curated only | 26 | 6 | 26 | 7 | tail | +0.979 [-0.251, +2.466] | -1.820 [-2.923, -0.685] | +2.799 [+1.134, +4.778] |
| curated + pool, Head Zipf ≤ 4.0 | 150 | 23 | 13 | 9 | tail | +0.603 [+0.045, +1.223] | -0.759 [-1.724, +0.224] | +1.362 [+0.209, +2.499] |

Gate 1 (ΔL deficit, freq): met; gate 2 (D_L CI above 0): met; gate 3: met. **A4 runs** (a Tail-singleton intervention; transfer to XTail untested).

