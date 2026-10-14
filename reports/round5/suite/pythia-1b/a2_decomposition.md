# Round 5, A2: exact decomposition

Spec: `plan.md`, A2. Freq contexts unless stated; Δh̄ and affinities are recomputed in each context draw.

## A2. Exact decomposition

- Identity L₁ = Δh · (u_T1 − u_I1): max error in the run 4.2e-05 (every pair and prefix, all sets; fail-fast ≤ 1e-3). This is algebraic; it checks the implementation.
- P = (u_T1 − u_I1) · Δh̄ equals the freq-mean L₁ for every pair: max |difference| 2.9e-06.
- ‖Δh̄‖ = 52.72; per-context deviation share 0.45 (mean ‖Δh_c − Δh̄‖² / mean ‖Δh_c‖²).

### Per pair: P = projection of u_T1 − u_I1 on Δh̄ (= freq-mean L₁)

| | Head | Tail | XTail |
|---|---|---|---|
| P, all pairs | +4.02 [+3.34, +4.68] | +1.02 [+0.50, +1.57] | -0.04 [-0.37, +0.29] |
| P, single-token pairs (= freq-mean L) | +4.02 [+3.34, +4.68] | +3.28 (n = 1) | — |

Single-token pairs: 27 (Head 26, Tail 1, XTail 0).

### Multi-token pairs: first vs later subtokens (freq)

| Band | pairs | mean L | mean L₁ | mean L_rest | L₁ / L | L_rest / L |
|---|---:|---|---|---|---|---|
| Tail | 49 | +3.13 [+2.45, +3.85] | +0.98 [+0.44, +1.53] | +2.16 [+1.53, +2.79] | +0.31 [+0.16, +0.46] | +0.69 [+0.54, +0.84] |
| XTail | 50 | +1.73 [+1.22, +2.21] | -0.04 [-0.37, +0.29] | +1.77 [+1.22, +2.31] | -0.02 [-0.24, +0.16] | +1.02 [+0.84, +1.24] |

By exact token-count cell (multi-token pairs with ≥ 5 pairs in the cell and band):

| Cell | Band | pairs | mean L | mean L₁ | mean L_rest | L₁ / L |
|---|---|---:|---|---|---|---|
| 1×2 | Tail | 5 | +1.83 [-1.95, +5.69] | -0.25 [-3.01, +2.32] | +2.07 [+0.76, +3.57] | not interpreted |
| 2×1 | Tail | 5 | +1.65 [-0.37, +3.93] | +2.50 [+1.15, +4.02] | -0.85 [-2.35, +0.21] | not interpreted |
| 2×2 | Tail | 25 | +3.53 [+2.61, +4.45] | +1.09 [+0.33, +1.84] | +2.44 [+1.59, +3.27] | +0.31 [+0.10, +0.49] |
| 2×2 | XTail | 15 | +1.68 [+0.97, +2.38] | -0.45 [-1.09, +0.21] | +2.13 [+1.40, +2.84] | -0.26 [-0.89, +0.11] |
| 2×3 | Tail | 5 | +4.61 [+2.81, +6.54] | +0.60 [-0.13, +1.23] | +4.02 [+1.91, +6.01] | +0.13 [-0.03, +0.32] |
| 2×3 | XTail | 11 | +2.01 [+0.85, +3.17] | -0.10 [-0.80, +0.77] | +2.11 [+0.99, +3.29] | -0.05 [-0.59, +0.31] |
| 3×2 | Tail | 7 | +2.83 [+1.76, +3.77] | +0.86 [-0.30, +2.08] | +1.97 [+0.78, +3.25] | +0.30 [-0.12, +0.70] |
| 3×2 | XTail | 12 | +1.81 [+0.71, +2.83] | +0.46 [-0.10, +1.03] | +1.35 [+0.26, +2.46] | +0.25 [-0.07, +0.71] |
| 3×3 | XTail | 6 | +1.91 [-0.20, +4.04] | +0.01 [-0.96, +0.93] | +1.90 [+0.14, +3.68] | not interpreted |

XTail shortfall: L̄_Head − L_XTail = (L̄_Head − L₁,XTail) − L_rest,XTail: +2.29 [+1.45, +3.10] = +4.06 [+3.30, +4.78] − (+1.77 [+1.22, +2.31]).

### Per verb: passive-slot affinity a_V = u_V1 · Δh̄

| Verbs | role | n | Zipf range | slope of a_V | slope of cos | slope of ‖u‖ |
|---|---|---:|---|---|---|---|
| single-token, curated | T | 32 | 2.68–5.16 | +0.131 [-0.851, +0.983] | +0.003 [-0.017, +0.020] | +0.003 [-0.025, +0.024] |
| single-token, curated | I | 33 | 2.59–5.20 | -1.643 [-2.247, -0.961] | -0.035 [-0.047, -0.020] | -0.008 [-0.036, +0.011] |
| first-subtoken fragments, curated | T | 94 | 1.32–3.18 | +0.021 [-0.328, +0.355] | +0.000 [-0.006, +0.007] | +0.008 [-0.007, +0.022] |
| first-subtoken fragments, curated | I | 93 | 1.23–3.12 | -1.077 [-1.573, -0.630] | -0.021 [-0.030, -0.012] | +0.016 [-0.004, +0.033] |
| single-token, curated + pool | T | 305 | 2.58–5.16 | +0.146 [-0.191, +0.497] | +0.003 [-0.004, +0.010] | -0.030 [-0.043, -0.017] |
| single-token, curated + pool | I | 39 | 2.59–5.20 | -1.662 [-2.217, -1.050] | -0.035 [-0.047, -0.022] | -0.008 [-0.034, +0.011] |

Head class separation (curated): AUC(a_V, T vs I) = 0.96 (26 T, 26 I).

### Gate for A4 (single-token participles; verbs within role × band cells, Δh̄ re-drawn)

| Population | Head T | rare T | Head I | rare I | rare bands | D_T | D_I | D_L = D_T − D_I |
|---|---:|---:|---:|---:|---|---|---|---|
| curated + pool (gate) | 282 | 23 | 30 | 9 | tail | +0.590 [+0.062, +1.157] | -1.607 [-2.607, -0.491] | +2.198 [+0.998, +3.382] |
| curated only | 26 | 6 | 26 | 7 | tail | +0.938 [-0.216, +2.371] | -1.851 [-3.075, -0.506] | +2.789 [+0.987, +4.749] |
| curated + pool, Head Zipf ≤ 4.0 | 150 | 23 | 13 | 9 | tail | +0.640 [+0.109, +1.217] | -0.757 [-1.793, +0.310] | +1.397 [+0.157, +2.553] |

Gate 1 (ΔL deficit, freq): met; gate 2 (D_L CI above 0): met; gate 3: met. **A4 runs** (a Tail-singleton intervention; transfer to XTail untested).

