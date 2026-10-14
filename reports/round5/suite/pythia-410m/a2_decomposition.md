# Round 5, A2: exact decomposition

Spec: `plan.md`, A2. Freq contexts unless stated; Δh̄ and affinities are recomputed in each context draw.

## A2. Exact decomposition

- Identity L₁ = Δh · (u_T1 − u_I1): max error in the run 3.5e-05 (every pair and prefix, all sets; fail-fast ≤ 1e-3). This is algebraic; it checks the implementation.
- P = (u_T1 − u_I1) · Δh̄ equals the freq-mean L₁ for every pair: max |difference| 3.1e-06.
- ‖Δh̄‖ = 56.52; per-context deviation share 0.42 (mean ‖Δh_c − Δh̄‖² / mean ‖Δh_c‖²).

### Per pair: P = projection of u_T1 − u_I1 on Δh̄ (= freq-mean L₁)

| | Head | Tail | XTail |
|---|---|---|---|
| P, all pairs | +3.86 [+3.27, +4.41] | +1.21 [+0.71, +1.74] | +0.07 [-0.30, +0.42] |
| P, single-token pairs (= freq-mean L) | +3.86 [+3.27, +4.41] | +2.95 (n = 1) | — |

Single-token pairs: 27 (Head 26, Tail 1, XTail 0).

### Multi-token pairs: first vs later subtokens (freq)

| Band | pairs | mean L | mean L₁ | mean L_rest | L₁ / L | L_rest / L |
|---|---:|---|---|---|---|---|
| Tail | 49 | +3.05 [+2.36, +3.79] | +1.18 [+0.67, +1.71] | +1.88 [+1.22, +2.53] | +0.38 [+0.24, +0.54] | +0.62 [+0.46, +0.76] |
| XTail | 50 | +1.42 [+0.87, +1.97] | +0.07 [-0.30, +0.42] | +1.35 [+0.77, +1.93] | +0.05 [-0.25, +0.30] | +0.95 [+0.70, +1.25] |

By exact token-count cell (multi-token pairs with ≥ 5 pairs in the cell and band):

| Cell | Band | pairs | mean L | mean L₁ | mean L_rest | L₁ / L |
|---|---|---:|---|---|---|---|
| 1×2 | Tail | 5 | +2.22 [-1.65, +5.68] | -0.34 [-2.62, +1.63] | +2.56 [+0.96, +4.39] | not interpreted |
| 2×1 | Tail | 5 | +1.79 [-0.39, +4.32] | +2.99 [+1.57, +4.66] | -1.21 [-2.59, +0.03] | not interpreted |
| 2×2 | Tail | 25 | +3.22 [+2.20, +4.22] | +1.18 [+0.43, +1.88] | +2.04 [+1.14, +2.86] | +0.37 [+0.16, +0.56] |
| 2×2 | XTail | 15 | +1.57 [+0.95, +2.15] | -0.46 [-1.14, +0.27] | +2.03 [+1.24, +2.83] | -0.30 [-0.95, +0.16] |
| 2×3 | Tail | 5 | +4.12 [+2.11, +6.27] | +0.82 [-0.08, +1.70] | +3.31 [+1.13, +5.24] | +0.20 [-0.02, +0.49] |
| 2×3 | XTail | 11 | +1.51 [+0.48, +2.82] | +0.07 [-0.80, +0.96] | +1.44 [+0.53, +2.33] | +0.04 [-1.14, +0.46] |
| 3×2 | Tail | 7 | +3.50 [+1.82, +5.11] | +1.54 [+0.61, +2.55] | +1.97 [+0.05, +3.82] | +0.44 [+0.15, +0.96] |
| 3×2 | XTail | 12 | +1.63 [+0.45, +2.88] | +0.55 [-0.02, +1.09] | +1.08 [-0.13, +2.37] | +0.34 [-0.02, +1.17] |
| 3×3 | XTail | 6 | +1.39 [-0.87, +3.77] | +0.19 [-1.13, +1.21] | +1.20 [-0.74, +3.24] | not interpreted |

XTail shortfall: L̄_Head − L_XTail = (L̄_Head − L₁,XTail) − L_rest,XTail: +2.44 [+1.66, +3.22] = +3.79 [+3.12, +4.45] − (+1.35 [+0.77, +1.93]).

### Per verb: passive-slot affinity a_V = u_V1 · Δh̄

| Verbs | role | n | Zipf range | slope of a_V | slope of cos | slope of ‖u‖ |
|---|---|---:|---|---|---|---|
| single-token, curated | T | 32 | 2.68–5.16 | +0.158 [-0.737, +0.929] | +0.004 [-0.018, +0.024] | +0.002 [-0.018, +0.017] |
| single-token, curated | I | 33 | 2.59–5.20 | -1.533 [-2.121, -0.830] | -0.038 [-0.054, -0.021] | +0.010 [-0.015, +0.025] |
| first-subtoken fragments, curated | T | 94 | 1.32–3.18 | +0.094 [-0.256, +0.421] | +0.002 [-0.006, +0.010] | +0.007 [-0.007, +0.021] |
| first-subtoken fragments, curated | I | 93 | 1.23–3.12 | -1.131 [-1.581, -0.684] | -0.027 [-0.037, -0.016] | +0.011 [-0.004, +0.026] |
| single-token, curated + pool | T | 305 | 2.58–5.16 | +0.055 [-0.269, +0.390] | +0.001 [-0.007, +0.010] | -0.016 [-0.026, -0.007] |
| single-token, curated + pool | I | 39 | 2.59–5.20 | -1.546 [-2.093, -0.911] | -0.038 [-0.053, -0.023] | +0.008 [-0.014, +0.024] |

Head class separation (curated): AUC(a_V, T vs I) = 0.96 (26 T, 26 I).

### Gate for A4 (single-token participles; verbs within role × band cells, Δh̄ re-drawn)

| Population | Head T | rare T | Head I | rare I | rare bands | D_T | D_I | D_L = D_T − D_I |
|---|---:|---:|---:|---:|---|---|---|---|
| curated + pool (gate) | 282 | 23 | 30 | 9 | tail | +0.342 [-0.225, +0.908] | -1.511 [-2.537, -0.414] | +1.853 [+0.642, +3.047] |
| curated only | 26 | 6 | 26 | 7 | tail | +0.844 [-0.206, +2.048] | -1.768 [-2.986, -0.417] | +2.612 [+0.908, +4.381] |
| curated + pool, Head Zipf ≤ 4.0 | 150 | 23 | 13 | 9 | tail | +0.390 [-0.165, +0.965] | -0.743 [-1.843, +0.398] | +1.133 [-0.194, +2.356] |

Gate 1 (ΔL deficit, freq): met; gate 2 (D_L CI above 0): met; gate 3: met. **A4 runs** (a Tail-singleton intervention; transfer to XTail untested).

