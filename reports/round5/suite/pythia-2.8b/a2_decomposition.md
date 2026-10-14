# Round 5, A2: exact decomposition

Spec: `plan.md`, A2. Freq contexts unless stated; Δh̄ and affinities are recomputed in each context draw.

## A2. Exact decomposition

- Identity L₁ = Δh · (u_T1 − u_I1): max error in the run 4.6e-05 (every pair and prefix, all sets; fail-fast ≤ 1e-3). This is algebraic; it checks the implementation.
- P = (u_T1 − u_I1) · Δh̄ equals the freq-mean L₁ for every pair: max |difference| 8.8e-06.
- ‖Δh̄‖ = 58.85; per-context deviation share 0.44 (mean ‖Δh_c − Δh̄‖² / mean ‖Δh_c‖²).

### Per pair: P = projection of u_T1 − u_I1 on Δh̄ (= freq-mean L₁)

| | Head | Tail | XTail |
|---|---|---|---|
| P, all pairs | +3.74 [+3.00, +4.40] | +1.06 [+0.56, +1.55] | +0.12 [-0.22, +0.47] |
| P, single-token pairs (= freq-mean L) | +3.74 [+3.00, +4.40] | +2.68 (n = 1) | — |

Single-token pairs: 27 (Head 26, Tail 1, XTail 0).

### Multi-token pairs: first vs later subtokens (freq)

| Band | pairs | mean L | mean L₁ | mean L_rest | L₁ / L | L_rest / L |
|---|---:|---|---|---|---|---|
| Tail | 49 | +3.00 [+2.28, +3.72] | +1.03 [+0.51, +1.53] | +1.97 [+1.35, +2.60] | +0.34 [+0.19, +0.50] | +0.66 [+0.50, +0.81] |
| XTail | 50 | +2.00 [+1.45, +2.59] | +0.12 [-0.22, +0.47] | +1.88 [+1.32, +2.45] | +0.06 [-0.13, +0.23] | +0.94 [+0.77, +1.13] |

By exact token-count cell (multi-token pairs with ≥ 5 pairs in the cell and band):

| Cell | Band | pairs | mean L | mean L₁ | mean L_rest | L₁ / L |
|---|---|---:|---|---|---|---|
| 1×2 | Tail | 5 | +1.90 [-2.01, +5.50] | -0.61 [-3.02, +1.75] | +2.51 [+0.80, +4.25] | not interpreted |
| 2×1 | Tail | 5 | +1.04 [-1.12, +3.24] | +2.16 [+1.04, +3.40] | -1.12 [-2.87, +0.10] | not interpreted |
| 2×2 | Tail | 25 | +3.43 [+2.44, +4.42] | +1.22 [+0.48, +1.96] | +2.20 [+1.36, +3.05] | +0.36 [+0.16, +0.54] |
| 2×2 | XTail | 15 | +2.36 [+1.61, +3.23] | -0.01 [-0.78, +0.81] | +2.38 [+1.68, +3.07] | -0.01 [-0.43, +0.29] |
| 2×3 | Tail | 5 | +3.94 [+2.28, +5.41] | +0.40 [-0.44, +1.12] | +3.54 [+1.47, +5.18] | +0.10 [-0.10, +0.39] |
| 2×3 | XTail | 11 | +2.02 [+0.76, +3.45] | +0.21 [-0.40, +0.87] | +1.81 [+0.46, +2.95] | +0.10 [-0.30, +0.49] |
| 3×2 | Tail | 7 | +2.97 [+1.95, +4.05] | +1.33 [+0.57, +2.18] | +1.64 [+0.47, +2.91] | +0.45 [+0.19, +0.79] |
| 3×2 | XTail | 12 | +1.68 [+0.44, +2.84] | +0.33 [-0.23, +0.89] | +1.35 [+0.13, +2.60] | +0.20 [-0.19, +0.78] |
| 3×3 | XTail | 6 | +1.99 [-0.26, +4.30] | -0.04 [-1.25, +0.99] | +2.04 [-0.05, +4.23] | not interpreted |

XTail shortfall: L̄_Head − L_XTail = (L̄_Head − L₁,XTail) − L_rest,XTail: +1.74 [+0.84, +2.61] = +3.62 [+2.81, +4.39] − (+1.88 [+1.32, +2.45]).

### Per verb: passive-slot affinity a_V = u_V1 · Δh̄

| Verbs | role | n | Zipf range | slope of a_V | slope of cos | slope of ‖u‖ |
|---|---|---:|---|---|---|---|
| single-token, curated | T | 32 | 2.68–5.16 | +0.268 [-0.769, +1.116] | +0.005 [-0.014, +0.021] | +0.024 [-0.003, +0.047] |
| single-token, curated | I | 33 | 2.59–5.20 | -1.764 [-2.321, -1.158] | -0.034 [-0.045, -0.023] | +0.007 [-0.022, +0.026] |
| first-subtoken fragments, curated | T | 94 | 1.32–3.18 | -0.020 [-0.392, +0.312] | -0.000 [-0.007, +0.005] | +0.003 [-0.012, +0.019] |
| first-subtoken fragments, curated | I | 93 | 1.23–3.12 | -1.050 [-1.567, -0.596] | -0.018 [-0.027, -0.010] | -0.000 [-0.017, +0.017] |
| single-token, curated + pool | T | 305 | 2.58–5.16 | +0.130 [-0.225, +0.514] | +0.002 [-0.004, +0.010] | -0.008 [-0.021, +0.005] |
| single-token, curated + pool | I | 39 | 2.59–5.20 | -1.721 [-2.239, -1.139] | -0.034 [-0.044, -0.023] | +0.002 [-0.027, +0.020] |

Head class separation (curated): AUC(a_V, T vs I) = 0.95 (26 T, 26 I).

### Gate for A4 (single-token participles; verbs within role × band cells, Δh̄ re-drawn)

| Population | Head T | rare T | Head I | rare I | rare bands | D_T | D_I | D_L = D_T − D_I |
|---|---:|---:|---:|---:|---|---|---|---|
| curated + pool (gate) | 282 | 23 | 30 | 9 | tail | +0.582 [-0.005, +1.182] | -1.531 [-2.515, -0.517] | +2.113 [+0.983, +3.261] |
| curated only | 26 | 6 | 26 | 7 | tail | +1.093 [-0.007, +2.515] | -1.906 [-3.080, -0.683] | +2.999 [+1.362, +4.803] |
| curated + pool, Head Zipf ≤ 4.0 | 150 | 23 | 13 | 9 | tail | +0.647 [+0.084, +1.269] | -0.684 [-1.664, +0.323] | +1.331 [+0.181, +2.461] |

Gate 1 (ΔL deficit, freq): met; gate 2 (D_L CI above 0): met; gate 3: met. **A4 runs** (a Tail-singleton intervention; transfer to XTail untested).

