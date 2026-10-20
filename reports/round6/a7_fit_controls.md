# Round 6, A7: fit controls and robustness on the D11 pairs

Spec: `plan.md`, A7. Ratings: Gemma-4-31B-it (`rate_fit.py`, two prompts averaged; round-P ratings reused for identical sentences); outcomes from A2's 1.4B cur set. 35406 pair × context rows (281 pairs × 126 curated passive_1 contexts). 95% CIs: pairs within band × contexts within context band, 2,000 draws.

Rating format: digit mass median 1.000; good-fit mean by verb band Head 4.25, Tail 3.16, XTail 2.72; bad-patient mean Head 3.64, Tail 3.36, XTail 3.08.

Rating validation (new sentences): 41047 sentences; digit mass median 1.000, 0.0% below 0.9; prompt agreement (Spearman) 0.957.

Blind audit: Claude rated 60 stratified new sentences before seeing Gemma's ratings; Spearman 0.906.

## 1. Matched-fit regression

| outcome | model | Tail (95% CI) | XTail (95% CI) | good fit | bad patient |
|---|---|---|---|---|---|
| M_was | unmatched | -0.94 [-1.53, -0.39] | -1.15 [-1.74, -0.58] | — | — |
| M_was | good | -0.60 [-1.16, -0.04] | -0.67 [-1.25, -0.09] | +0.31 [+0.26, +0.38] | — |
| M_was | good+bad_patient | -0.63 [-1.20, -0.06] | -0.72 [-1.32, -0.13] | +0.33 [+0.27, +0.39] | -0.13 [-0.18, -0.09] |
| L | unmatched | -0.73 [-1.45, -0.03] | -1.40 [-2.12, -0.64] | — | — |
| L | good | -0.51 [-1.21, +0.16] | -1.09 [-1.82, -0.35] | +0.20 [+0.13, +0.27] | — |
| L | good+bad_patient | -0.50 [-1.19, +0.17] | -1.06 [-1.75, -0.33] | +0.19 [+0.12, +0.26] | +0.09 [+0.03, +0.15] |
| dT | unmatched | -0.18 [-0.59, +0.24] | +0.19 [-0.26, +0.64] | — | — |
| dT | good | -0.01 [-0.41, +0.41] | +0.43 [+0.00, +0.89] | +0.16 [+0.11, +0.21] | — |
| dT | good+bad_patient | -0.03 [-0.45, +0.39] | +0.36 [-0.08, +0.82] | +0.18 [+0.13, +0.22] | -0.17 [-0.22, -0.12] |
| dI | unmatched | +0.55 [-0.07, +1.15] | +1.58 [+0.93, +2.18] | — | — |
| dI | good | +0.51 [-0.12, +1.11] | +1.52 [+0.88, +2.13] | -0.04 [-0.09, +0.02] | — |
| dI | good+bad_patient | +0.47 [-0.14, +1.03] | +1.42 [+0.80, +2.01] | -0.01 [-0.07, +0.04] | -0.26 [-0.31, -0.20] |
| M_has | unmatched | -0.21 [-0.75, +0.30] | +0.25 [-0.33, +0.81] | — | — |
| M_has | good | -0.09 [-0.63, +0.44] | +0.43 [-0.14, +0.98] | +0.12 [+0.06, +0.17] | — |
| M_has | good+bad_patient | -0.13 [-0.65, +0.37] | +0.34 [-0.21, +0.86] | +0.14 [+0.08, +0.19] | -0.22 [-0.27, -0.17] |

- **XTail − Head on M_was at matched fit (good + bad patient): -0.72 [-1.32, -0.13] → survives matched fit** (unmatched -1.15; attenuation -0.42 [-0.62, -0.26]); Tail − Head -0.63 [-1.20, -0.06] → survives matched fit. L: XTail -1.06 [-1.75, -0.33] → survives matched fit; its sides at matched fit: Δ_pass(T) +0.36, −Δ_pass(I) -1.42. Accounting β(M_was) = β(L) + β(M_has): max error 2.2e-15.
- Sensitivities (XTail on M_was): bare-clause I verbs only -0.86 [-1.50, -0.17]; quadratic fit terms -0.72 [-1.32, -0.13]; band-specific fit slopes (at fit = 4) -0.77 [-1.38, -0.18].
- Token-standardized (fit-adjusted Tail − XTail within cells 2×2, 2×3, 3×2, 3×3, weights 85/36/35/22; deficit > 0): M_was +0.32 [-0.18, +0.79] → unresolved; L +0.99 [+0.46, +1.58] → deficit.

## 2. Noun-rarity check: D11 verbs in the 300 released passive_1 contexts (noun bands)

| outcome | verb band | Head-noun contexts | Tail-noun contexts | XTail-noun contexts |
|---|---|---|---|---|
| L | Head | +3.28 | +3.31 | +3.33 |
| L | Tail | +2.67 | +2.72 | +2.81 |
| L | XTail | +1.99 | +2.01 | +2.08 |
| L | effects | verb (XTail − Head) -1.28 [-1.88, -0.67] → Head − XTail deficit | context (XTail − Head) +0.09 [-0.03, +0.21] → negligible | |
| M_pass | Head | +2.39 | +2.43 | +2.42 |
| M_pass | Tail | +1.99 | +2.02 | +2.11 |
| M_pass | XTail | +1.67 | +1.64 | +1.74 |
| M_pass | effects | verb (XTail − Head) -0.72 [-1.28, -0.18] → Head − XTail deficit | context (XTail − Head) +0.07 [-0.03, +0.19] → negligible | |

Secondary: the curated contexts (context band = the band of the context's original verb, not noun frequency; patient Zipf medians 4.96 / 4.70 / 4.45):

| outcome | verb band | Head contexts | Tail contexts | XTail contexts |
|---|---|---|---|---|
| L | Head | +3.12 | +3.46 | +3.63 |
| L | Tail | +2.62 | +2.79 | +2.72 |
| L | XTail | +1.91 | +2.09 | +2.11 |
| L | effects | verb (XTail − Head) -1.37 [-2.09, -0.62] → Head − XTail deficit | context (XTail − Head) +0.27 [+0.07, +0.49] → context effect | |
| M_was | Head | +2.66 | +2.64 | +2.82 |
| M_was | Tail | +1.82 | +1.77 | +1.74 |
| M_was | XTail | +1.56 | +1.57 | +1.56 |
| M_was | effects | verb (XTail − Head) -1.14 [-1.73, -0.56] → Head − XTail deficit | context (XTail − Head) +0.03 [-0.16, +0.22] → negligible | |

## 3. Robustness: every A1 prefix set

| population | set | ΔL Head − XTail | reading | ΔL_tok (Tail − XTail) | reading | side (token-std.) |
|---|---|---|---|---|---|---|
| D11 (1.4B) | freq | +1.41 [+0.71, +2.15] | deficit | +0.99 [+0.43, +1.53] | deficit | intransitive-side |
| D11 (1.4B) | been | +1.23 [+0.50, +1.94] | deficit | +1.35 [+0.80, +1.93] | deficit | intransitive-side |
| D11 (1.4B) | cur | +1.40 [+0.70, +2.11] | deficit | +1.12 [+0.58, +1.67] | deficit | intransitive-side |
| D11 (1.4B) | they | +1.24 [+0.54, +1.88] | deficit | +0.89 [+0.21, +1.60] | deficit | intransitive-side |
| curated 126 (1.4B, round 5) | freq | +2.12 [+1.26, +2.94] | — | — | — | — |
| curated 126 (1.4B, round 5) | been | +1.82 [+0.98, +2.67] | — | — | — | — |
| curated 126 (1.4B, round 5) | cur | +2.12 [+1.27, +2.95] | — | — | — | — |
| curated 126 (1.4B, round 5) | they | +1.73 [+0.87, +2.57] | — | — | — | — |
| D11 (pythia-160m) | freq | +1.52 [+0.92, +2.16] | deficit | +0.97 [+0.40, +1.54] | deficit | intransitive-side |
| D11 (pythia-410m) | freq | +1.65 [+0.97, +2.31] | deficit | +1.26 [+0.66, +1.86] | deficit | intransitive-side |
| D11 (pythia-1b) | freq | +1.64 [+0.94, +2.34] | deficit | +1.66 [+1.08, +2.22] | deficit | intransitive-side |
| D11 (pythia-2.8b) | freq | +1.20 [+0.50, +1.93] | deficit | +0.89 [+0.23, +1.52] | deficit | intransitive-side |
| D11 (pythia-6.9b) | freq | +1.64 [+0.93, +2.33] | deficit | +1.04 [+0.47, +1.57] | deficit | intransitive-side |
| D11 (pythia-12b) | freq | +1.39 [+0.68, +2.09] | deficit | +0.94 [+0.38, +1.48] | deficit | intransitive-side |

