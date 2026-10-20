# Round 6, A2: the D11 frame contrast across the Pythia suite

Spec: `plan.md`, A2. Run: `run_frame_contrast.py --cur-all` on the 562 D11 participles; analysis: `analyze_frame_d11.py`. Δ_pass(V) = log P(V | was-prefix) − log P(V | has-prefix); L = Δ_pass(T) − Δ_pass(I). 281 pairs (Head 47, Tail 132, XTail 102; 62 held out). Primary: ΔL_tok = token-cell-weighted Tail − XTail over the exact cells 2×2, 2×3, 3×2, 3×3 (fixed weights 85 / 36 / 35 / 22). Rule (bound 0.25 nats): deficit = 95% CI above 0 and estimate ≥ 0.25; surplus = mirror; negligible = 90% CI within ±0.25; otherwise unresolved. Side (only given a deficit): intransitive-side = −ΔΔ_pass(I) CI above 0 and above ΔΔ_pass(T) (paired); dominance, not exclusivity. Held-out pairs: DAS-unexposed subgroup (most verbs scored in round 5); three-cell estimand (2×2, 2×3, 3×2). Draws are identical across models. 95% CIs: pairs within band × split (token contrast: band × cell × split) × contexts.

## Summary (freq prefixes)

| Model | ΔL_tok (Tail − XTail, token-standardized) | reading | ΔΔ_pass(T) / −ΔΔ_pass(I) | side | held-out three-cell ΔL | ΔL Head − XTail | Head AUC | singleton D_L |
|---|---|---|---|---|---|---|---|---|
| pythia-160m | +0.97 [+0.40, +1.54] | **deficit** | -0.22 / +1.19 | intransitive-side | +1.12 [-0.39, +2.64] (unresolved) | +1.52 [+0.92, +2.16] | 0.94 | +1.28 [+0.51, +2.01] |
| pythia-410m | +1.26 [+0.66, +1.86] | **deficit** | -0.22 / +1.47 | intransitive-side | +0.64 [-0.72, +2.02] (unresolved) | +1.65 [+0.97, +2.31] | 0.95 | +1.39 [+0.47, +2.31] |
| pythia-1b | +1.66 [+1.08, +2.22] | **deficit** | +0.04 / +1.62 | intransitive-side | +1.40 [-0.00, +2.90] (unresolved) | +1.64 [+0.94, +2.34] | 0.94 | +1.51 [+0.45, +2.45] |
| pythia-1.4b | +0.99 [+0.43, +1.53] | **deficit** | -0.41 / +1.40 | intransitive-side | +0.53 [-0.66, +1.63] (unresolved) | +1.41 [+0.71, +2.15] | 0.93 | +1.56 [+0.58, +2.47] |
| pythia-2.8b | +0.89 [+0.23, +1.52] | **deficit** | -0.41 / +1.30 | intransitive-side | +0.46 [-1.06, +1.98] (unresolved) | +1.20 [+0.50, +1.93] | 0.93 | +1.45 [+0.47, +2.39] |
| pythia-6.9b | +1.04 [+0.47, +1.57] | **deficit** | -0.48 / +1.51 | intransitive-side | +1.05 [-0.08, +2.19] (unresolved) | +1.64 [+0.93, +2.33] | 0.94 | +1.40 [+0.44, +2.29] |
| pythia-12b | +0.94 [+0.38, +1.48] | **deficit** | -0.48 / +1.42 | intransitive-side | +0.07 [-0.99, +1.10] (unresolved) | +1.39 [+0.68, +2.09] | 0.94 | +1.50 [+0.52, +2.43] |

## pythia-160m

Checks: first-subtoken reconstruction 1.6e-03, identity 2.9e-03 (tolerance 0.01); max |logit| at the slot 844.7; round-5 reproduction (94346 rows in common) max |Δ lp| 2.7e-03.

### freq (primary): band means

| Quantity | Head | Tail | XTail | Head − Tail | Head − XTail | Tail − XTail, token-standardized |
|---|---|---|---|---|---|---|
| L | +2.78 [+2.29, +3.28] | +2.10 [+1.78, +2.40] | +1.25 [+0.86, +1.67] | +0.68 [+0.13, +1.25] | +1.52 [+0.92, +2.16] | +0.97 [+0.40, +1.54] |
| Δ_pass(T) | +0.44 [+0.05, +0.81] | +0.33 [+0.04, +0.61] | +0.60 [+0.28, +0.93] | +0.12 [-0.31, +0.51] | -0.15 [-0.58, +0.28] | -0.22 [-0.65, +0.20] |
| Δ_pass(I) | -2.34 [-2.75, -1.93] | -1.77 [-2.06, -1.46] | -0.66 [-1.02, -0.31] | -0.57 [-1.03, -0.11] | -1.68 [-2.20, -1.16] | -1.19 [-1.65, -0.74] |
| M_has | -0.65 [-1.04, -0.26] | -0.63 [-0.93, -0.33] | -0.50 [-0.91, -0.12] | -0.02 [-0.47, +0.45] | -0.15 [-0.67, +0.39] | -0.13 [-0.66, +0.46] |
| L₁ | +2.72 [+2.20, +3.23] | +0.65 [+0.41, +0.90] | +0.14 [-0.06, +0.35] | +2.07 [+1.50, +2.63] | +2.58 [+2.02, +3.12] | +0.41 [+0.06, +0.78] |
| L_rest | +0.06 [-0.02, +0.19] | +1.45 [+1.09, +1.80] | +1.11 [+0.69, +1.57] | -1.39 [-1.75, -1.01] | -1.05 [-1.52, -0.61] | +0.55 [-0.06, +1.16] |

By token cell (Tail vs XTail; L, Δ_pass(T), Δ_pass(I)):

| Cell | Tail pairs | XTail pairs | L Tail | L XTail | Δ_pass(T) Tail | Δ_pass(T) XTail | Δ_pass(I) Tail | Δ_pass(I) XTail |
|---|---:|---:|---|---|---|---|---|---|
| 2×2 | 62 | 23 | +2.18 [+1.78, +2.59] | +1.81 [+1.09, +2.53] | +0.31 [-0.06, +0.67] | +0.98 [+0.47, +1.48] | -1.87 [-2.25, -1.49] | -0.83 [-1.43, -0.27] |
| 2×3 | 8 | 28 | +2.28 [+1.48, +3.02] | +1.15 [+0.43, +1.91] | +1.12 [+0.49, +1.72] | +0.38 [-0.19, +0.95] | -1.16 [-2.28, -0.02] | -0.78 [-1.32, -0.26] |
| 3×2 | 15 | 20 | +2.01 [+0.61, +3.23] | +1.09 [+0.25, +1.86] | +0.24 [-0.93, +1.28] | +0.81 [+0.24, +1.35] | -1.78 [-2.38, -1.12] | -0.28 [-0.83, +0.37] |
| 3×3 | 5 | 17 | +3.83 [+2.14, +5.11] | +0.77 [-0.32, +1.89] | +0.62 [-0.19, +1.34] | +0.14 [-0.63, +0.91] | -3.21 [-4.47, -2.07] | -0.62 [-1.69, +0.43] |

- **Primary ΔL_tok (Tail − XTail, token-standardized): +0.97 [+0.40, +1.54] → deficit**; ΔΔ_pass(T) -0.22 [-0.65, +0.20], −ΔΔ_pass(I) +1.19 [+0.74, +1.65] → **intransitive-side**. ΔM_has (token-standardized) -0.13 [-0.66, +0.46] → unresolved; reading unresolved.
- Sensitivities: OLS (L ~ band + T tokens + I tokens, multi-multi) +0.85 [+0.30, +1.42]; 2×2 only +0.37 [-0.47, +1.17].
- Head − XTail (token-confounded): ΔL +1.52 [+0.92, +2.16] → deficit; ΔM_has -0.15 [-0.67, +0.39]; reading unresolved. Head − Tail: ΔL +0.68 [+0.13, +1.25] → deficit.
- Three-cell contrast (2×2, 2×3, 3×2; fixed weights 85/36/35): all +0.67 [+0.06, +1.28] → deficit (intransitive-side; cells 2×2 (62/23), 2×3 (8/28), 3×2 (15/20)); trainable +0.54 [-0.15, +1.18] → unresolved (no deficit (contributions descriptive); cells 2×2 (46/18), 2×3 (6/23), 3×2 (11/12)); heldout +1.12 [-0.39, +2.64] → unresolved (no deficit (contributions descriptive); cells 2×2 (16/5), 2×3 (2/5), 3×2 (4/8)).
- DAS-unexposed (held-out) band L: Head +3.57 [+1.92, +5.32], Tail +1.80 [+1.08, +2.49], XTail +0.78 [-0.02, +1.57]. Trainable pairs, four-cell ΔL_tok: +0.78 [+0.14, +1.38] → deficit.

- Per verb (single-token): Head AUC(a_V, T vs I) 0.94; D_L = (ā_T,Head − ā_I,Head) − (ā_T,Tail − ā_I,Tail) +1.28 [+0.51, +2.01] (D_T +0.33 [-0.15, +0.78], D_I -0.95 [-1.54, -0.32]; n {'T_head': 44, 'T_tail': 24, 'I_head': 45, 'I_tail': 20}).

All prefix sets (ΔL_tok; reading; side; band means Head / Tail / XTail):

| set | ΔL_tok | reading | side | ΔL Head − XTail | band means |
|---|---|---|---|---|---|
| freq | +0.97 [+0.40, +1.54] | deficit | intransitive-side | +1.52 [+0.92, +2.16] | see freq table above |
| been | +1.06 [+0.47, +1.61] | deficit | intransitive-side | +1.51 [+0.85, +2.12] | L +2.94 / +2.25 / +1.43; Δ_pass(T) +0.95 / +0.87 / +0.98; Δ_pass(I) -2.00 / -1.37 / -0.45; M_has -0.65 / -0.63 / -0.50; L₁ +2.89 / +0.81 / +0.05; L_rest +0.06 / +1.44 / +1.38 |
| cur | +1.08 [+0.54, +1.66] | deficit | intransitive-side | +1.65 [+1.06, +2.27] | L +2.90 / +2.19 / +1.25; Δ_pass(T) +0.87 / +0.73 / +0.90; Δ_pass(I) -2.03 / -1.46 / -0.35; M_has -0.68 / -0.68 / -0.44; L₁ +2.82 / +0.83 / +0.15; L_rest +0.08 / +1.35 / +1.10 |
| they | +0.78 [+0.23, +1.30] | deficit | intransitive-side | +1.62 [+1.01, +2.17] | L +2.33 / +1.55 / +0.71; Δ_pass(T) -0.11 / +0.06 / +0.72; Δ_pass(I) -2.44 / -1.49 / +0.01; M_has -0.31 / -0.31 / -0.45; L₁ +2.30 / +0.31 / +0.05; L_rest +0.03 / +1.24 / +0.67 |

## pythia-410m

Checks: first-subtoken reconstruction 2.3e-05, identity 3.5e-05 (tolerance 0.01); max |logit| at the slot 19.6; round-5 reproduction (94346 rows in common) max |Δ lp| 0.0e+00.

### freq (primary): band means

| Quantity | Head | Tail | XTail | Head − Tail | Head − XTail | Tail − XTail, token-standardized |
|---|---|---|---|---|---|---|
| L | +3.37 [+2.81, +3.93] | +2.67 [+2.34, +3.05] | +1.72 [+1.32, +2.14] | +0.70 [+0.10, +1.31] | +1.65 [+0.97, +2.31] | +1.26 [+0.66, +1.86] |
| Δ_pass(T) | +0.15 [-0.33, +0.62] | +0.31 [-0.06, +0.71] | +0.65 [+0.26, +1.07] | -0.16 [-0.63, +0.28] | -0.51 [-0.97, -0.06] | -0.22 [-0.63, +0.20] |
| Δ_pass(I) | -3.22 [-3.72, -2.71] | -2.36 [-2.75, -1.94] | -1.07 [-1.53, -0.54] | -0.86 [-1.41, -0.34] | -2.16 [-2.75, -1.60] | -1.47 [-1.96, -0.99] |
| M_has | -0.55 [-0.96, -0.12] | -0.74 [-1.04, -0.44] | -0.50 [-0.89, -0.14] | +0.19 [-0.30, +0.68] | -0.05 [-0.58, +0.49] | -0.32 [-0.83, +0.21] |
| L₁ | +3.33 [+2.75, +3.91] | +0.92 [+0.59, +1.25] | +0.21 [-0.06, +0.49] | +2.41 [+1.73, +3.07] | +3.12 [+2.50, +3.75] | +0.61 [+0.17, +1.05] |
| L_rest | +0.04 [-0.05, +0.18] | +1.75 [+1.38, +2.11] | +1.51 [+1.06, +1.97] | -1.71 [-2.10, -1.33] | -1.48 [-1.92, -0.99] | +0.64 [+0.00, +1.23] |

By token cell (Tail vs XTail; L, Δ_pass(T), Δ_pass(I)):

| Cell | Tail pairs | XTail pairs | L Tail | L XTail | Δ_pass(T) Tail | Δ_pass(T) XTail | Δ_pass(I) Tail | Δ_pass(I) XTail |
|---|---:|---:|---|---|---|---|---|---|
| 2×2 | 62 | 23 | +2.58 [+2.11, +3.03] | +1.88 [+1.19, +2.55] | +0.22 [-0.27, +0.72] | +0.95 [+0.37, +1.57] | -2.35 [-2.84, -1.85] | -0.93 [-1.54, -0.24] |
| 2×3 | 8 | 28 | +3.10 [+1.93, +4.27] | +1.82 [+1.20, +2.44] | +1.11 [+0.47, +1.81] | +0.52 [+0.03, +1.10] | -1.99 [-3.17, -0.84] | -1.30 [-1.97, -0.64] |
| 3×2 | 15 | 20 | +3.16 [+1.96, +4.34] | +0.94 [+0.16, +1.70] | +0.64 [-0.08, +1.37] | +0.38 [-0.23, +1.04] | -2.52 [-3.46, -1.65] | -0.56 [-1.13, +0.03] |
| 3×3 | 5 | 17 | +4.03 [+2.75, +5.79] | +2.18 [+0.82, +3.48] | +0.39 [-1.17, +2.09] | +0.72 [+0.10, +1.38] | -3.64 [-4.52, -2.68] | -1.46 [-2.73, -0.21] |

- **Primary ΔL_tok (Tail − XTail, token-standardized): +1.26 [+0.66, +1.86] → deficit**; ΔΔ_pass(T) -0.22 [-0.63, +0.20], −ΔΔ_pass(I) +1.47 [+0.99, +1.96] → **intransitive-side**. ΔM_has (token-standardized) -0.32 [-0.83, +0.21] → unresolved; reading unresolved.
- Sensitivities: OLS (L ~ band + T tokens + I tokens, multi-multi) +1.24 [+0.69, +1.79]; 2×2 only +0.70 [-0.09, +1.51].
- Head − XTail (token-confounded): ΔL +1.65 [+0.97, +2.31] → deficit; ΔM_has -0.05 [-0.58, +0.49]; reading unresolved. Head − Tail: ΔL +0.70 [+0.10, +1.31] → deficit.
- Three-cell contrast (2×2, 2×3, 3×2; fixed weights 85/36/35): all +1.17 [+0.56, +1.77] → deficit (intransitive-side; cells 2×2 (62/23), 2×3 (8/28), 3×2 (15/20)); trainable +1.39 [+0.70, +2.06] → deficit (intransitive-side; cells 2×2 (46/18), 2×3 (6/23), 3×2 (11/12)); heldout +0.64 [-0.72, +2.02] → unresolved (no deficit (contributions descriptive); cells 2×2 (16/5), 2×3 (2/5), 3×2 (4/8)).
- DAS-unexposed (held-out) band L: Head +4.70 [+2.29, +6.72], Tail +2.45 [+1.82, +3.08], XTail +1.51 [+0.77, +2.28]. Trainable pairs, four-cell ΔL_tok: +1.35 [+0.69, +2.02] → deficit.

- Per verb (single-token): Head AUC(a_V, T vs I) 0.95; D_L = (ā_T,Head − ā_I,Head) − (ā_T,Tail − ā_I,Tail) +1.39 [+0.47, +2.31] (D_T +0.21 [-0.37, +0.74], D_I -1.18 [-1.91, -0.43]; n {'T_head': 44, 'T_tail': 24, 'I_head': 45, 'I_tail': 20}).

All prefix sets (ΔL_tok; reading; side; band means Head / Tail / XTail):

| set | ΔL_tok | reading | side | ΔL Head − XTail | band means |
|---|---|---|---|---|---|
| freq | +1.26 [+0.66, +1.86] | deficit | intransitive-side | +1.65 [+0.97, +2.31] | see freq table above |
| been | +1.29 [+0.69, +1.89] | deficit | intransitive-side | +1.65 [+0.97, +2.32] | L +3.51 / +2.85 / +1.85; Δ_pass(T) +1.14 / +1.15 / +1.30; Δ_pass(I) -2.37 / -1.69 / -0.55; M_has -0.55 / -0.74 / -0.50; L₁ +3.39 / +1.04 / +0.16; L_rest +0.11 / +1.80 / +1.70 |
| cur | +1.23 [+0.67, +1.82] | deficit | intransitive-side | +1.69 [+1.04, +2.34] | L +3.47 / +2.74 / +1.78; Δ_pass(T) +0.66 / +0.74 / +0.98; Δ_pass(I) -2.81 / -2.00 / -0.80; M_has -0.59 / -0.74 / -0.44; L₁ +3.40 / +1.12 / +0.26; L_rest +0.06 / +1.62 / +1.52 |
| they | +0.93 [+0.27, +1.59] | deficit | intransitive-side | +1.46 [+0.78, +2.08] | L +2.88 / +1.97 / +1.42; Δ_pass(T) -0.41 / -0.47 / +0.17; Δ_pass(I) -3.29 / -2.44 / -1.25; M_has -0.08 / -0.10 / +0.02; L₁ +2.87 / +0.31 / +0.01; L_rest +0.01 / +1.66 / +1.41 |

## pythia-1b

Checks: first-subtoken reconstruction 3.1e-05, identity 3.8e-05 (tolerance 0.01); max |logit| at the slot 17.0; round-5 reproduction (94346 rows in common) max |Δ lp| 0.0e+00.

### freq (primary): band means

| Quantity | Head | Tail | XTail | Head − Tail | Head − XTail | Tail − XTail, token-standardized |
|---|---|---|---|---|---|---|
| L | +3.49 [+2.89, +4.12] | +2.89 [+2.54, +3.25] | +1.85 [+1.47, +2.24] | +0.60 [-0.01, +1.27] | +1.64 [+0.94, +2.34] | +1.66 [+1.08, +2.22] |
| Δ_pass(T) | +0.36 [-0.13, +0.85] | +0.55 [+0.19, +0.96] | +0.83 [+0.45, +1.24] | -0.20 [-0.68, +0.26] | -0.47 [-0.94, -0.01] | +0.04 [-0.40, +0.49] |
| Δ_pass(I) | -3.13 [-3.61, -2.64] | -2.33 [-2.69, -1.96] | -1.02 [-1.47, -0.57] | -0.80 [-1.35, -0.27] | -2.11 [-2.69, -1.52] | -1.62 [-2.09, -1.15] |
| M_has | -0.64 [-1.07, -0.21] | -0.89 [-1.18, -0.59] | -0.47 [-0.80, -0.14] | +0.25 [-0.23, +0.74] | -0.17 [-0.68, +0.35] | -0.66 [-1.15, -0.20] |
| L₁ | +3.41 [+2.79, +4.05] | +0.86 [+0.55, +1.19] | +0.14 [-0.11, +0.41] | +2.55 [+1.87, +3.25] | +3.28 [+2.63, +3.94] | +0.71 [+0.28, +1.17] |
| L_rest | +0.08 [-0.02, +0.25] | +2.03 [+1.66, +2.40] | +1.72 [+1.24, +2.17] | -1.95 [-2.33, -1.55] | -1.64 [-2.11, -1.15] | +0.95 [+0.31, +1.60] |

By token cell (Tail vs XTail; L, Δ_pass(T), Δ_pass(I)):

| Cell | Tail pairs | XTail pairs | L Tail | L XTail | Δ_pass(T) Tail | Δ_pass(T) XTail | Δ_pass(I) Tail | Δ_pass(I) XTail |
|---|---:|---:|---|---|---|---|---|---|
| 2×2 | 62 | 23 | +2.83 [+2.44, +3.25] | +1.64 [+0.94, +2.29] | +0.40 [-0.06, +0.88] | +0.95 [+0.37, +1.53] | -2.43 [-2.89, -1.94] | -0.69 [-1.20, -0.16] |
| 2×3 | 8 | 28 | +3.82 [+2.68, +5.00] | +1.63 [+0.92, +2.34] | +1.65 [+0.62, +2.76] | +0.49 [+0.01, +1.01] | -2.17 [-3.36, -1.01] | -1.14 [-1.86, -0.41] |
| 3×2 | 15 | 20 | +3.29 [+2.09, +4.52] | +1.89 [+0.96, +2.74] | +0.93 [+0.18, +1.66] | +1.06 [+0.40, +1.68] | -2.36 [-3.44, -1.38] | -0.83 [-1.44, -0.14] |
| 3×3 | 5 | 17 | +5.42 [+3.86, +6.97] | +2.41 [+1.40, +3.48] | +1.39 [+0.19, +2.53] | +0.65 [-0.03, +1.36] | -4.03 [-5.11, -2.87] | -1.77 [-3.03, -0.61] |

- **Primary ΔL_tok (Tail − XTail, token-standardized): +1.66 [+1.08, +2.22] → deficit**; ΔΔ_pass(T) +0.04 [-0.40, +0.49], −ΔΔ_pass(I) +1.62 [+1.15, +2.09] → **intransitive-side**. ΔM_has (token-standardized) -0.66 [-1.15, -0.20] → surplus; reading unresolved.
- Sensitivities: OLS (L ~ band + T tokens + I tokens, multi-multi) +1.52 [+0.99, +2.05]; 2×2 only +1.20 [+0.45, +1.98].
- Head − XTail (token-confounded): ΔL +1.64 [+0.94, +2.34] → deficit; ΔM_has -0.17 [-0.68, +0.35]; reading unresolved. Head − Tail: ΔL +0.60 [-0.01, +1.27] → unresolved.
- Three-cell contrast (2×2, 2×3, 3×2; fixed weights 85/36/35): all +1.47 [+0.88, +2.09] → deficit (intransitive-side; cells 2×2 (62/23), 2×3 (8/28), 3×2 (15/20)); trainable +1.49 [+0.84, +2.18] → deficit (intransitive-side; cells 2×2 (46/18), 2×3 (6/23), 3×2 (11/12)); heldout +1.40 [-0.00, +2.90] → unresolved (no deficit (contributions descriptive); cells 2×2 (16/5), 2×3 (2/5), 3×2 (4/8)).
- DAS-unexposed (held-out) band L: Head +4.54 [+2.76, +6.11], Tail +2.79 [+2.17, +3.41], XTail +1.95 [+1.21, +2.69]. Trainable pairs, four-cell ΔL_tok: +1.66 [+0.99, +2.30] → deficit.

- Per verb (single-token): Head AUC(a_V, T vs I) 0.94; D_L = (ā_T,Head − ā_I,Head) − (ā_T,Tail − ā_I,Tail) +1.51 [+0.45, +2.45] (D_T +0.27 [-0.32, +0.82], D_I -1.24 [-2.07, -0.38]; n {'T_head': 44, 'T_tail': 24, 'I_head': 45, 'I_tail': 20}).

All prefix sets (ΔL_tok; reading; side; band means Head / Tail / XTail):

| set | ΔL_tok | reading | side | ΔL Head − XTail | band means |
|---|---|---|---|---|---|
| freq | +1.66 [+1.08, +2.22] | deficit | intransitive-side | +1.64 [+0.94, +2.34] | see freq table above |
| been | +1.62 [+1.09, +2.18] | deficit | intransitive-side | +1.30 [+0.58, +2.00] | L +3.47 / +3.10 / +2.17; Δ_pass(T) +1.07 / +1.28 / +1.39; Δ_pass(I) -2.40 / -1.82 / -0.78; M_has -0.64 / -0.89 / -0.47; L₁ +3.36 / +0.99 / +0.18; L_rest +0.11 / +2.11 / +1.99 |
| cur | +1.58 [+1.04, +2.13] | deficit | intransitive-side | +1.65 [+0.99, +2.33] | L +3.56 / +2.94 / +1.91; Δ_pass(T) +0.78 / +0.92 / +1.14; Δ_pass(I) -2.78 / -2.02 / -0.77; M_has -0.66 / -0.90 / -0.53; L₁ +3.46 / +1.06 / +0.21; L_rest +0.10 / +1.88 / +1.71 |
| they | +1.05 [+0.47, +1.65] | deficit | intransitive-side | +1.51 [+0.87, +2.10] | L +3.29 / +2.30 / +1.78; Δ_pass(T) -0.07 / -0.19 / +0.55; Δ_pass(I) -3.37 / -2.50 / -1.23; M_has -0.32 / -0.30 / -0.17; L₁ +3.25 / +0.53 / +0.09; L_rest +0.04 / +1.77 / +1.69 |

## pythia-1.4b

Checks: first-subtoken reconstruction 3.5e-05, identity 4.1e-05 (tolerance 0.001); max |logit| at the slot 17.7; round-5 reproduction (94346 rows in common) max |Δ lp| 0.0e+00.

### freq (primary): band means

| Quantity | Head | Tail | XTail | Head − Tail | Head − XTail | Tail − XTail, token-standardized |
|---|---|---|---|---|---|---|
| L | +3.37 [+2.74, +4.00] | +2.56 [+2.21, +2.93] | +1.96 [+1.53, +2.37] | +0.81 [+0.16, +1.46] | +1.41 [+0.71, +2.15] | +0.99 [+0.43, +1.53] |
| Δ_pass(T) | +0.48 [-0.08, +1.02] | +0.33 [-0.06, +0.76] | +0.84 [+0.43, +1.27] | +0.15 [-0.32, +0.59] | -0.36 [-0.84, +0.10] | -0.41 [-0.78, -0.06] |
| Δ_pass(I) | -2.89 [-3.43, -2.33] | -2.23 [-2.64, -1.77] | -1.12 [-1.58, -0.65] | -0.66 [-1.21, -0.11] | -1.77 [-2.38, -1.19] | -1.40 [-1.83, -0.96] |
| M_has | -0.75 [-1.16, -0.30] | -0.93 [-1.22, -0.64] | -0.48 [-0.82, -0.14] | +0.18 [-0.29, +0.65] | -0.27 [-0.79, +0.23] | -0.56 [-1.00, -0.07] |
| L₁ | +3.30 [+2.66, +3.95] | +0.77 [+0.45, +1.09] | +0.18 [-0.07, +0.42] | +2.53 [+1.83, +3.22] | +3.12 [+2.44, +3.79] | +0.54 [+0.13, +0.97] |
| L_rest | +0.07 [-0.02, +0.24] | +1.80 [+1.44, +2.19] | +1.78 [+1.29, +2.27] | -1.73 [-2.12, -1.32] | -1.71 [-2.21, -1.19] | +0.45 [-0.22, +1.07] |

By token cell (Tail vs XTail; L, Δ_pass(T), Δ_pass(I)):

| Cell | Tail pairs | XTail pairs | L Tail | L XTail | Δ_pass(T) Tail | Δ_pass(T) XTail | Δ_pass(I) Tail | Δ_pass(I) XTail |
|---|---:|---:|---|---|---|---|---|---|
| 2×2 | 62 | 23 | +2.53 [+2.07, +3.01] | +1.86 [+1.35, +2.42] | +0.27 [-0.21, +0.77] | +0.96 [+0.45, +1.49] | -2.27 [-2.79, -1.74] | -0.90 [-1.40, -0.41] |
| 2×3 | 8 | 28 | +3.48 [+2.39, +4.55] | +2.16 [+1.40, +2.97] | +1.17 [+0.61, +1.84] | +0.77 [+0.20, +1.38] | -2.30 [-3.49, -1.15] | -1.39 [-2.10, -0.71] |
| 3×2 | 15 | 20 | +2.37 [+1.30, +3.44] | +1.72 [+0.76, +2.68] | +0.05 [-0.81, +0.93] | +1.00 [+0.31, +1.67] | -2.32 [-3.19, -1.53] | -0.73 [-1.29, -0.14] |
| 3×3 | 5 | 17 | +4.57 [+3.08, +6.06] | +2.37 [+1.18, +3.57] | +0.80 [-0.05, +1.70] | +0.63 [-0.12, +1.38] | -3.77 [-4.75, -2.68] | -1.74 [-2.87, -0.68] |

- **Primary ΔL_tok (Tail − XTail, token-standardized): +0.99 [+0.43, +1.53] → deficit**; ΔΔ_pass(T) -0.41 [-0.78, -0.06], −ΔΔ_pass(I) +1.40 [+0.96, +1.83] → **intransitive-side**. ΔM_has (token-standardized) -0.56 [-1.00, -0.07] → surplus; reading unresolved.
- Sensitivities: OLS (L ~ band + T tokens + I tokens, multi-multi) +0.82 [+0.28, +1.36]; 2×2 only +0.67 [-0.02, +1.36].
- Head − XTail (token-confounded): ΔL +1.41 [+0.71, +2.15] → deficit; ΔM_has -0.27 [-0.79, +0.23]; reading unresolved. Head − Tail: ΔL +0.81 [+0.16, +1.46] → deficit.
- Three-cell contrast (2×2, 2×3, 3×2; fixed weights 85/36/35): all +0.82 [+0.22, +1.38] → deficit (intransitive-side; cells 2×2 (62/23), 2×3 (8/28), 3×2 (15/20)); trainable +0.97 [+0.31, +1.63] → deficit (intransitive-side; cells 2×2 (46/18), 2×3 (6/23), 3×2 (11/12)); heldout +0.53 [-0.66, +1.63] → unresolved (no deficit (contributions descriptive); cells 2×2 (16/5), 2×3 (2/5), 3×2 (4/8)).
- DAS-unexposed (held-out) band L: Head +4.54 [+2.11, +6.39], Tail +2.20 [+1.48, +2.86], XTail +1.71 [+0.97, +2.45]. Trainable pairs, four-cell ΔL_tok: +1.09 [+0.44, +1.72] → deficit.

- Per verb (single-token): Head AUC(a_V, T vs I) 0.93; D_L = (ā_T,Head − ā_I,Head) − (ā_T,Tail − ā_I,Tail) +1.56 [+0.58, +2.47] (D_T +0.31 [-0.26, +0.85], D_I -1.25 [-2.01, -0.45]; n {'T_head': 44, 'T_tail': 24, 'I_head': 45, 'I_tail': 20}).

All prefix sets (ΔL_tok; reading; side; band means Head / Tail / XTail):

| set | ΔL_tok | reading | side | ΔL Head − XTail | band means |
|---|---|---|---|---|---|
| freq | +0.99 [+0.43, +1.53] | deficit | intransitive-side | +1.41 [+0.71, +2.15] | see freq table above |
| been | +1.35 [+0.80, +1.93] | deficit | intransitive-side | +1.23 [+0.50, +1.94] | L +3.38 / +2.91 / +2.15; Δ_pass(T) +1.23 / +1.11 / +1.27; Δ_pass(I) -2.15 / -1.80 / -0.87; M_has -0.75 / -0.93 / -0.48; L₁ +3.28 / +1.08 / +0.24; L_rest +0.10 / +1.83 / +1.91 |
| cur | +1.12 [+0.58, +1.67] | deficit | intransitive-side | +1.40 [+0.70, +2.11] | L +3.46 / +2.73 / +2.06; Δ_pass(T) +0.89 / +0.71 / +1.08; Δ_pass(I) -2.57 / -2.02 / -0.99; M_has -0.74 / -0.96 / -0.50; L₁ +3.37 / +1.00 / +0.25; L_rest +0.09 / +1.73 / +1.81 |
| they | +0.89 [+0.21, +1.60] | deficit | intransitive-side | +1.24 [+0.54, +1.88] | L +2.91 / +2.05 / +1.68; Δ_pass(T) -0.47 / -0.61 / +0.14; Δ_pass(I) -3.38 / -2.66 / -1.54; M_has -0.12 / -0.20 / +0.04; L₁ +2.86 / +0.33 / +0.10; L_rest +0.05 / +1.72 / +1.58 |

## pythia-2.8b

Checks: first-subtoken reconstruction 3.4e-05, identity 5.6e-05 (tolerance 0.01); max |logit| at the slot 17.7; round-5 reproduction (94346 rows in common) max |Δ lp| 0.0e+00.

### freq (primary): band means

| Quantity | Head | Tail | XTail | Head − Tail | Head − XTail | Tail − XTail, token-standardized |
|---|---|---|---|---|---|---|
| L | +3.33 [+2.71, +3.96] | +2.79 [+2.42, +3.19] | +2.13 [+1.72, +2.53] | +0.54 [-0.11, +1.23] | +1.20 [+0.50, +1.93] | +0.89 [+0.23, +1.52] |
| Δ_pass(T) | +0.35 [-0.16, +0.86] | +0.22 [-0.15, +0.62] | +0.64 [+0.26, +1.05] | +0.13 [-0.36, +0.60] | -0.29 [-0.78, +0.19] | -0.41 [-0.88, +0.05] |
| Δ_pass(I) | -2.98 [-3.49, -2.47] | -2.57 [-2.93, -2.19] | -1.48 [-1.87, -1.08] | -0.41 [-0.97, +0.15] | -1.49 [-2.07, -0.91] | -1.30 [-1.75, -0.84] |
| M_has | -0.64 [-1.08, -0.19] | -0.81 [-1.10, -0.52] | -0.60 [-0.90, -0.30] | +0.16 [-0.32, +0.64] | -0.05 [-0.54, +0.49] | -0.19 [-0.63, +0.26] |
| L₁ | +3.25 [+2.62, +3.89] | +0.86 [+0.54, +1.18] | +0.17 [-0.08, +0.43] | +2.40 [+1.73, +3.08] | +3.08 [+2.40, +3.76] | +0.52 [+0.10, +0.98] |
| L_rest | +0.08 [-0.01, +0.24] | +1.93 [+1.53, +2.34] | +1.96 [+1.49, +2.44] | -1.86 [-2.29, -1.44] | -1.88 [-2.37, -1.38] | +0.36 [-0.32, +1.03] |

By token cell (Tail vs XTail; L, Δ_pass(T), Δ_pass(I)):

| Cell | Tail pairs | XTail pairs | L Tail | L XTail | Δ_pass(T) Tail | Δ_pass(T) XTail | Δ_pass(I) Tail | Δ_pass(I) XTail |
|---|---:|---:|---|---|---|---|---|---|
| 2×2 | 62 | 23 | +2.86 [+2.36, +3.35] | +2.50 [+1.76, +3.22] | +0.15 [-0.33, +0.63] | +1.14 [+0.53, +1.78] | -2.71 [-3.17, -2.25] | -1.36 [-1.89, -0.88] |
| 2×3 | 8 | 28 | +3.71 [+2.65, +4.78] | +2.18 [+1.43, +2.96] | +1.24 [+0.50, +2.11] | +0.47 [-0.09, +1.00] | -2.47 [-3.69, -1.36] | -1.71 [-2.42, -1.01] |
| 3×2 | 15 | 20 | +2.78 [+1.54, +4.00] | +1.96 [+0.99, +2.89] | +0.06 [-0.80, +0.87] | +0.69 [-0.09, +1.44] | -2.72 [-3.77, -1.75] | -1.28 [-1.83, -0.73] |
| 3×3 | 5 | 17 | +3.96 [+1.81, +6.19] | +1.97 [+0.97, +2.97] | +0.33 [-1.29, +1.67] | +0.13 [-0.44, +0.74] | -3.63 [-4.97, -2.29] | -1.84 [-2.69, -1.02] |

- **Primary ΔL_tok (Tail − XTail, token-standardized): +0.89 [+0.23, +1.52] → deficit**; ΔΔ_pass(T) -0.41 [-0.88, +0.05], −ΔΔ_pass(I) +1.30 [+0.84, +1.75] → **intransitive-side**. ΔM_has (token-standardized) -0.19 [-0.63, +0.26] → unresolved; reading unresolved.
- Sensitivities: OLS (L ~ band + T tokens + I tokens, multi-multi) +0.76 [+0.11, +1.39]; 2×2 only +0.36 [-0.50, +1.27].
- Head − XTail (token-confounded): ΔL +1.20 [+0.50, +1.93] → deficit; ΔM_has -0.05 [-0.54, +0.49]; reading unresolved. Head − Tail: ΔL +0.54 [-0.11, +1.23] → unresolved.
- Three-cell contrast (2×2, 2×3, 3×2; fixed weights 85/36/35): all +0.73 [+0.06, +1.39] → deficit (intransitive-side; cells 2×2 (62/23), 2×3 (8/28), 3×2 (15/20)); trainable +0.83 [+0.07, +1.58] → deficit (intransitive-side; cells 2×2 (46/18), 2×3 (6/23), 3×2 (11/12)); heldout +0.46 [-1.06, +1.98] → unresolved (no deficit (contributions descriptive); cells 2×2 (16/5), 2×3 (2/5), 3×2 (4/8)).
- DAS-unexposed (held-out) band L: Head +4.21 [+2.33, +5.96], Tail +2.34 [+1.67, +3.00], XTail +2.07 [+1.28, +2.93]. Trainable pairs, four-cell ΔL_tok: +0.98 [+0.23, +1.66] → deficit.

- Per verb (single-token): Head AUC(a_V, T vs I) 0.93; D_L = (ā_T,Head − ā_I,Head) − (ā_T,Tail − ā_I,Tail) +1.45 [+0.47, +2.39] (D_T +0.20 [-0.37, +0.71], D_I -1.24 [-2.04, -0.44]; n {'T_head': 44, 'T_tail': 24, 'I_head': 45, 'I_tail': 20}).

All prefix sets (ΔL_tok; reading; side; band means Head / Tail / XTail):

| set | ΔL_tok | reading | side | ΔL Head − XTail | band means |
|---|---|---|---|---|---|
| freq | +0.89 [+0.23, +1.52] | deficit | intransitive-side | +1.20 [+0.50, +1.93] | see freq table above |
| been | +1.12 [+0.51, +1.72] | deficit | intransitive-side | +1.13 [+0.41, +1.84] | L +3.42 / +2.98 / +2.30; Δ_pass(T) +1.12 / +0.95 / +1.25; Δ_pass(I) -2.30 / -2.03 / -1.05; M_has -0.64 / -0.81 / -0.60; L₁ +3.31 / +1.06 / +0.18; L_rest +0.11 / +1.92 / +2.12 |
| cur | +1.00 [+0.41, +1.61] | deficit | intransitive-side | +1.15 [+0.53, +1.82] | L +3.38 / +2.97 / +2.23; Δ_pass(T) +0.81 / +0.67 / +0.98; Δ_pass(I) -2.58 / -2.31 / -1.25; M_has -0.67 / -0.89 / -0.67; L₁ +3.29 / +1.09 / +0.27; L_rest +0.10 / +1.88 / +1.96 |
| they | +0.72 [+0.04, +1.40] | deficit | intransitive-side | +1.01 [+0.33, +1.66] | L +2.91 / +2.31 / +1.90; Δ_pass(T) -0.09 / -0.26 / +0.26; Δ_pass(I) -3.01 / -2.56 / -1.65; M_has -0.13 / -0.32 / -0.24; L₁ +2.86 / +0.42 / +0.09; L_rest +0.05 / +1.89 / +1.81 |

## pythia-6.9b

Checks: first-subtoken reconstruction 4.8e-05, identity 6.6e-05 (tolerance 0.01); max |logit| at the slot 19.2; round-5 reproduction (94346 rows in common) max |Δ lp| 0.0e+00.

### freq (primary): band means

| Quantity | Head | Tail | XTail | Head − Tail | Head − XTail | Tail − XTail, token-standardized |
|---|---|---|---|---|---|---|
| L | +3.50 [+2.91, +4.09] | +2.64 [+2.27, +3.02] | +1.86 [+1.44, +2.27] | +0.86 [+0.23, +1.48] | +1.64 [+0.93, +2.33] | +1.04 [+0.47, +1.57] |
| Δ_pass(T) | +0.23 [-0.25, +0.72] | +0.13 [-0.21, +0.49] | +0.52 [+0.13, +0.93] | +0.10 [-0.36, +0.53] | -0.29 [-0.78, +0.18] | -0.48 [-0.88, -0.10] |
| Δ_pass(I) | -3.27 [-3.77, -2.75] | -2.51 [-2.85, -2.16] | -1.34 [-1.75, -0.94] | -0.76 [-1.31, -0.21] | -1.93 [-2.50, -1.33] | -1.51 [-1.95, -1.10] |
| M_has | -0.67 [-1.08, -0.23] | -0.80 [-1.08, -0.53] | -0.35 [-0.65, -0.04] | +0.13 [-0.34, +0.61] | -0.32 [-0.81, +0.18] | -0.47 [-0.88, -0.02] |
| L₁ | +3.42 [+2.80, +4.01] | +0.76 [+0.47, +1.07] | +0.17 [-0.09, +0.43] | +2.66 [+1.99, +3.32] | +3.25 [+2.60, +3.88] | +0.37 [-0.02, +0.79] |
| L_rest | +0.09 [-0.03, +0.27] | +1.89 [+1.50, +2.28] | +1.69 [+1.23, +2.13] | -1.80 [-2.21, -1.38] | -1.61 [-2.07, -1.11] | +0.66 [+0.06, +1.24] |

By token cell (Tail vs XTail; L, Δ_pass(T), Δ_pass(I)):

| Cell | Tail pairs | XTail pairs | L Tail | L XTail | Δ_pass(T) Tail | Δ_pass(T) XTail | Δ_pass(I) Tail | Δ_pass(I) XTail |
|---|---:|---:|---|---|---|---|---|---|
| 2×2 | 62 | 23 | +2.60 [+2.10, +3.10] | +2.21 [+1.59, +2.84] | +0.02 [-0.42, +0.46] | +1.00 [+0.47, +1.54] | -2.58 [-3.05, -2.12] | -1.20 [-1.66, -0.76] |
| 2×3 | 8 | 28 | +3.87 [+3.07, +4.70] | +1.65 [+0.95, +2.36] | +0.90 [+0.23, +1.64] | +0.19 [-0.31, +0.69] | -2.97 [-3.73, -2.24] | -1.47 [-2.14, -0.83] |
| 3×2 | 15 | 20 | +2.43 [+1.35, +3.49] | +1.23 [+0.44, +2.01] | -0.22 [-1.09, +0.57] | +0.55 [-0.10, +1.24] | -2.64 [-3.49, -1.89] | -0.68 [-1.20, -0.13] |
| 3×3 | 5 | 17 | +4.03 [+2.44, +5.52] | +2.69 [+1.68, +3.75] | +0.34 [-0.51, +1.22] | +0.35 [-0.39, +1.14] | -3.69 [-5.16, -2.18] | -2.34 [-3.36, -1.39] |

- **Primary ΔL_tok (Tail − XTail, token-standardized): +1.04 [+0.47, +1.57] → deficit**; ΔΔ_pass(T) -0.48 [-0.88, -0.10], −ΔΔ_pass(I) +1.51 [+1.10, +1.95] → **intransitive-side**. ΔM_has (token-standardized) -0.47 [-0.88, -0.02] → surplus; reading unresolved.
- Sensitivities: OLS (L ~ band + T tokens + I tokens, multi-multi) +1.00 [+0.45, +1.53]; 2×2 only +0.39 [-0.38, +1.22].
- Head − XTail (token-confounded): ΔL +1.64 [+0.93, +2.33] → deficit; ΔM_has -0.32 [-0.81, +0.18]; reading unresolved. Head − Tail: ΔL +0.86 [+0.23, +1.48] → deficit.
- Three-cell contrast (2×2, 2×3, 3×2; fixed weights 85/36/35): all +0.99 [+0.40, +1.56] → deficit (intransitive-side; cells 2×2 (62/23), 2×3 (8/28), 3×2 (15/20)); trainable +1.01 [+0.32, +1.66] → deficit (intransitive-side; cells 2×2 (46/18), 2×3 (6/23), 3×2 (11/12)); heldout +1.05 [-0.08, +2.19] → unresolved (no deficit (contributions descriptive); cells 2×2 (16/5), 2×3 (2/5), 3×2 (4/8)).
- DAS-unexposed (held-out) band L: Head +4.46 [+2.58, +5.89], Tail +2.45 [+1.79, +3.07], XTail +1.71 [+1.03, +2.45]. Trainable pairs, four-cell ΔL_tok: +1.04 [+0.37, +1.64] → deficit.

- Per verb (single-token): Head AUC(a_V, T vs I) 0.94; D_L = (ā_T,Head − ā_I,Head) − (ā_T,Tail − ā_I,Tail) +1.40 [+0.44, +2.29] (D_T +0.00 [-0.58, +0.56], D_I -1.39 [-2.12, -0.64]; n {'T_head': 44, 'T_tail': 24, 'I_head': 45, 'I_tail': 20}).

All prefix sets (ΔL_tok; reading; side; band means Head / Tail / XTail):

| set | ΔL_tok | reading | side | ΔL Head − XTail | band means |
|---|---|---|---|---|---|
| freq | +1.04 [+0.47, +1.57] | deficit | intransitive-side | +1.64 [+0.93, +2.33] | see freq table above |
| been | +1.46 [+0.89, +2.03] | deficit | intransitive-side | +1.46 [+0.75, +2.16] | L +3.62 / +3.05 / +2.16; Δ_pass(T) +1.14 / +1.06 / +1.19; Δ_pass(I) -2.48 / -1.99 / -0.98; M_has -0.67 / -0.80 / -0.35; L₁ +3.49 / +0.99 / +0.12; L_rest +0.13 / +2.06 / +2.04 |
| cur | +1.17 [+0.66, +1.66] | deficit | intransitive-side | +1.55 [+0.89, +2.23] | L +3.54 / +2.85 / +1.99; Δ_pass(T) +0.73 / +0.63 / +0.88; Δ_pass(I) -2.81 / -2.21 / -1.11; M_has -0.70 / -0.85 / -0.37; L₁ +3.44 / +0.99 / +0.21; L_rest +0.10 / +1.86 / +1.78 |
| they | +0.90 [+0.22, +1.55] | deficit | intransitive-side | +1.37 [+0.68, +2.03] | L +3.12 / +2.31 / +1.75; Δ_pass(T) -0.03 / -0.15 / +0.41; Δ_pass(I) -3.15 / -2.46 / -1.33; M_has -0.26 / -0.34 / -0.07; L₁ +3.08 / +0.51 / +0.02; L_rest +0.04 / +1.80 / +1.73 |

## pythia-12b

Checks: first-subtoken reconstruction 4.0e-05, identity 7.5e-05 (tolerance 0.01); max |logit| at the slot 18.6; round-5 reproduction (94346 rows in common) max |Δ lp| 0.0e+00.

### freq (primary): band means

| Quantity | Head | Tail | XTail | Head − Tail | Head − XTail | Tail − XTail, token-standardized |
|---|---|---|---|---|---|---|
| L | +3.54 [+2.92, +4.18] | +2.75 [+2.38, +3.15] | +2.15 [+1.77, +2.54] | +0.79 [+0.13, +1.48] | +1.39 [+0.68, +2.09] | +0.94 [+0.38, +1.48] |
| Δ_pass(T) | +0.49 [-0.00, +1.01] | +0.41 [+0.07, +0.77] | +0.93 [+0.57, +1.32] | +0.08 [-0.41, +0.57] | -0.44 [-0.95, +0.04] | -0.48 [-0.91, -0.08] |
| Δ_pass(I) | -3.05 [-3.56, -2.54] | -2.34 [-2.68, -1.98] | -1.21 [-1.58, -0.83] | -0.71 [-1.25, -0.16] | -1.83 [-2.40, -1.26] | -1.42 [-1.85, -0.99] |
| M_has | -0.71 [-1.11, -0.28] | -0.78 [-1.06, -0.50] | -0.45 [-0.74, -0.16] | +0.07 [-0.39, +0.53] | -0.26 [-0.74, +0.25] | -0.43 [-0.87, +0.03] |
| L₁ | +3.43 [+2.79, +4.07] | +0.77 [+0.47, +1.08] | +0.10 [-0.16, +0.37] | +2.66 [+1.99, +3.35] | +3.33 [+2.66, +3.99] | +0.53 [+0.11, +0.97] |
| L_rest | +0.11 [+0.00, +0.28] | +1.97 [+1.59, +2.37] | +2.05 [+1.63, +2.48] | -1.87 [-2.27, -1.47] | -1.94 [-2.38, -1.50] | +0.41 [-0.22, +1.03] |

By token cell (Tail vs XTail; L, Δ_pass(T), Δ_pass(I)):

| Cell | Tail pairs | XTail pairs | L Tail | L XTail | Δ_pass(T) Tail | Δ_pass(T) XTail | Δ_pass(I) Tail | Δ_pass(I) XTail |
|---|---:|---:|---|---|---|---|---|---|
| 2×2 | 62 | 23 | +2.76 [+2.26, +3.24] | +2.33 [+1.87, +2.81] | +0.28 [-0.16, +0.72] | +1.25 [+0.76, +1.75] | -2.48 [-2.92, -2.02] | -1.08 [-1.46, -0.66] |
| 2×3 | 8 | 28 | +3.69 [+2.68, +4.76] | +2.17 [+1.50, +2.87] | +1.18 [+0.47, +1.96] | +0.83 [+0.30, +1.35] | -2.51 [-3.49, -1.53] | -1.35 [-2.01, -0.66] |
| 3×2 | 15 | 20 | +2.62 [+1.50, +3.69] | +1.65 [+0.75, +2.48] | +0.33 [-0.62, +1.20] | +1.02 [+0.35, +1.64] | -2.28 [-3.08, -1.58] | -0.64 [-1.16, -0.08] |
| 3×3 | 5 | 17 | +4.29 [+2.63, +5.93] | +2.38 [+1.34, +3.48] | +0.93 [-0.38, +2.18] | +0.58 [-0.18, +1.36] | -3.36 [-4.76, -1.99] | -1.80 [-2.71, -0.96] |

- **Primary ΔL_tok (Tail − XTail, token-standardized): +0.94 [+0.38, +1.48] → deficit**; ΔΔ_pass(T) -0.48 [-0.91, -0.08], −ΔΔ_pass(I) +1.42 [+0.99, +1.85] → **intransitive-side**. ΔM_has (token-standardized) -0.43 [-0.87, +0.03] → unresolved; reading unresolved.
- Sensitivities: OLS (L ~ band + T tokens + I tokens, multi-multi) +0.83 [+0.27, +1.36]; 2×2 only +0.43 [-0.27, +1.11].
- Head − XTail (token-confounded): ΔL +1.39 [+0.68, +2.09] → deficit; ΔM_has -0.26 [-0.74, +0.25]; reading unresolved. Head − Tail: ΔL +0.79 [+0.13, +1.48] → deficit.
- Three-cell contrast (2×2, 2×3, 3×2; fixed weights 85/36/35): all +0.80 [+0.23, +1.36] → deficit (intransitive-side; cells 2×2 (62/23), 2×3 (8/28), 3×2 (15/20)); trainable +1.05 [+0.36, +1.69] → deficit (intransitive-side; cells 2×2 (46/18), 2×3 (6/23), 3×2 (11/12)); heldout +0.07 [-0.99, +1.10] → unresolved (no deficit (contributions descriptive); cells 2×2 (16/5), 2×3 (2/5), 3×2 (4/8)).
- DAS-unexposed (held-out) band L: Head +4.50 [+2.60, +6.27], Tail +2.43 [+1.81, +3.03], XTail +2.49 [+1.70, +3.28]. Trainable pairs, four-cell ΔL_tok: +1.16 [+0.51, +1.77] → deficit.

- Per verb (single-token): Head AUC(a_V, T vs I) 0.94; D_L = (ā_T,Head − ā_I,Head) − (ā_T,Tail − ā_I,Tail) +1.50 [+0.52, +2.43] (D_T +0.12 [-0.50, +0.69], D_I -1.38 [-2.16, -0.61]; n {'T_head': 44, 'T_tail': 24, 'I_head': 45, 'I_tail': 20}).

All prefix sets (ΔL_tok; reading; side; band means Head / Tail / XTail):

| set | ΔL_tok | reading | side | ΔL Head − XTail | band means |
|---|---|---|---|---|---|
| freq | +0.94 [+0.38, +1.48] | deficit | intransitive-side | +1.39 [+0.68, +2.09] | see freq table above |
| been | +1.30 [+0.79, +1.81] | deficit | intransitive-side | +1.24 [+0.51, +1.92] | L +3.43 / +2.97 / +2.19; Δ_pass(T) +1.27 / +1.29 / +1.47; Δ_pass(I) -2.16 / -1.68 / -0.72; M_has -0.71 / -0.78 / -0.45; L₁ +3.30 / +0.97 / +0.04; L_rest +0.13 / +2.00 / +2.16 |
| cur | +0.92 [+0.40, +1.44] | deficit | intransitive-side | +1.27 [+0.61, +1.92] | L +3.47 / +2.78 / +2.20; Δ_pass(T) +0.85 / +0.74 / +1.15; Δ_pass(I) -2.62 / -2.04 / -1.05; M_has -0.68 / -0.79 / -0.51; L₁ +3.36 / +0.95 / +0.19; L_rest +0.11 / +1.83 / +2.01 |
| they | +0.56 [-0.09, +1.24] | unresolved | no deficit (contributions descriptive) | +1.01 [+0.38, +1.63] | L +3.10 / +2.31 / +2.08; Δ_pass(T) -0.16 / -0.40 / +0.37; Δ_pass(I) -3.26 / -2.72 / -1.71; M_has -0.13 / -0.12 / -0.05; L₁ +3.03 / +0.35 / +0.10; L_rest +0.07 / +1.97 / +1.99 |

