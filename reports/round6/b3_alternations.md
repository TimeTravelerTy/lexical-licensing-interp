# Round 6, B3: object-onset and clause-end expectation after T and I verbs

Spec: `plan.md`, B3. Run: `run_next_sets.py` on `data/round6/alternation`; analysis: `analyze_alternation.py`. Frames: "The N p" (simple past; primary) and "The N has v" (perfect); 24 A1 nouns. O = round 2's object-onset set (an object-onset proxy; strict objective pronouns him / them / me / us as sensitivity); clause end = "." (primary; the punctuation set as sensitivity). M_obj = log P(O|T) − log P(O|I) (all 281 pairs); M_end = log P(.|I) − log P(.|T) on the eligible pairs (T obligatory for both raters and I forms a complete bare clause: 197 pairs). Token-standardized Tail − XTail: cells 2×2, 2×3, 3×2, 3×3, weights 85 / 36 / 35 / 22. Rule (bound 0.25 nats) as A2. Side specificity: the paired difference of class slopes on Zipf. 95% CIs: pairs within band (× cell) × nouns within animacy, 2,000 draws, identical across models.

## Pythia-1.4B, simple past (primary)

| measure | pairs | Head | Tail | XTail | Tail − XTail (token-std.) | reading | side | Head − XTail | reading | side |
|---|---:|---|---|---|---|---|---|---|---|---|
| M_obj | 281 | +3.23 | +3.03 | +2.40 | +0.63 [+0.33, +0.93] | deficit | I-side | +0.82 [+0.39, +1.26] | deficit | I-side |
| M_obj_strict | 281 | +4.59 | +3.93 | +3.29 | +1.12 [+0.53, +1.74] | deficit | I-side | +1.30 [+0.66, +1.93] | deficit | I-side |
| M_end | 197 | +2.35 | +1.93 | +1.08 | +0.98 [+0.46, +1.50] | deficit | mixed | +1.27 [+0.52, +2.02] | deficit | T-side |
| M_end_punct | 197 | +1.75 | +1.34 | +0.81 | +0.63 [+0.22, +1.05] | deficit | mixed | +0.94 [+0.37, +1.51] | deficit | T-side |
| M_end_all_oblig | 219 | +1.67 | +1.76 | +1.04 | +0.80 [+0.27, +1.32] | deficit | mixed | +0.63 [-0.22, +1.49] | unresolved | no deficit (contributions descriptive) |
| M_obj_no_rare_trans_any | 170 | +3.79 | +3.30 | +2.70 | +0.51 [+0.17, +0.86] | deficit | I-side | +1.09 [+0.50, +1.74] | deficit | I-side |
| M_obj_no_rare_trans_both | 262 | +3.37 | +3.10 | +2.47 | +0.63 [+0.33, +0.93] | deficit | I-side | +0.89 [+0.43, +1.33] | deficit | I-side |
| M_end_no_rare_trans_any | 124 | +2.38 | +1.93 | +1.10 | +0.92 [+0.27, +1.60] | deficit | mixed | +1.28 [-0.15, +2.95] | unresolved | no deficit (contributions descriptive) |
| M_end_no_rare_trans_both | 183 | +2.35 | +1.91 | +1.11 | +0.90 [+0.37, +1.44] | deficit | mixed | +1.24 [+0.47, +2.06] | deficit | T-side |

Side parts (positive = contributes to the deficit): M_obj tok: T +0.04 [-0.08, +0.16], I +0.59 [+0.29, +0.89]; M_obj head_xtail: T -0.16 [-0.34, -0.01], I +0.99 [+0.61, +1.39]; M_end tok: T +0.60 [+0.31, +0.91], I +0.38 [-0.09, +0.82]; M_end head_xtail: T +1.54 [+1.02, +2.08], I -0.28 [-0.99, +0.39].

Per-verb slopes on Zipf (nats per Zipf unit; frame-specific log token count as covariate) and side specificity (predicted-side slope minus the other class's slope, same pairs; predicted < 0):

| readout | predicted side | predicted-side slope | reading | other-side slope | reading | difference | side specificity | categorical tokens | multi-token Tail/XTail only |
|---|---|---|---|---|---|---|---|---|---|
| O | I | -0.311 [-0.516, -0.110] | **tracks frequency** | -0.041 [-0.122, +0.039] | flat | -0.270 [-0.485, -0.056] | **side-specific** | side-specific | side-specific |
| O_strict | I | -0.650 [-0.920, -0.374] | **tracks frequency** | +0.065 [-0.217, +0.350] | unresolved | -0.715 [-1.095, -0.346] | **side-specific** | side-specific | side-specific |
| DOT | T | -0.687 [-0.891, -0.483] | **tracks frequency** | -0.075 [-0.391, +0.262] | unresolved | -0.612 [-1.010, -0.242] | **side-specific** | side-specific | side-specific |
| END | T | -0.423 [-0.545, -0.294] | **tracks frequency** | -0.046 [-0.287, +0.225] | flat | -0.377 [-0.675, -0.098] | **side-specific** | side-specific | not side-specific (unresolved) |

Animacy split (nouns): animate: O slope I -0.334 [-0.572, -0.099], T -0.078, side-specific; '.' slope T -0.473 [-0.722, -0.208], I -0.193, not side-specific (unresolved); inanimate: O slope I -0.287 [-0.520, -0.056], T -0.004, side-specific; '.' slope T -0.900 [-1.143, -0.672], I +0.044, side-specific.

log P(PREP) by band (T / I): Head -2.88 / -0.75; Tail -3.04 / -0.91; XTail -2.81 / -0.98.

## All models (simple past)

| model | ΔM_obj Head − XTail | ΔM_obj token-std. | ΔM_end Head − XTail | ΔM_end token-std. | O side specificity | "." side specificity |
|---|---|---|---|---|---|---|
| pythia-160m | +1.26 (deficit; I-side) | +0.70 (deficit) | +0.54 (unresolved; no deficit (contributions descriptive)) | +0.64 (deficit) | side-specific | not side-specific (unresolved) |
| pythia-410m | +0.91 (deficit; I-side) | +0.61 (deficit) | +1.04 (deficit; T-side) | +0.91 (deficit) | side-specific | side-specific |
| pythia-1b | +0.76 (deficit; I-side) | +0.70 (deficit) | +1.32 (deficit; T-side) | +1.00 (deficit) | side-specific | side-specific |
| pythia-1.4b | +0.82 (deficit; I-side) | +0.63 (deficit) | +1.27 (deficit; T-side) | +0.98 (deficit) | side-specific | side-specific |
| pythia-2.8b | +0.58 (deficit; I-side) | +0.52 (deficit) | +1.04 (deficit; T-side) | +0.93 (deficit) | not side-specific (unresolved) | side-specific |
| pythia-6.9b | +0.71 (deficit; I-side) | +0.48 (deficit) | +1.08 (deficit; T-side) | +0.84 (deficit) | not side-specific (unresolved) | side-specific |
| pythia-12b | +0.55 (deficit; I-side) | +0.52 (deficit) | +1.14 (deficit; T-side) | +0.81 (deficit) | not side-specific (unresolved) | side-specific |

Perfect frame ("The N has v"), 1.4B: M_obj Head − XTail +1.10 (deficit, I-side), token-std. +0.81 (deficit); M_end Head − XTail +1.93 (deficit, T-side), token-std. +1.27 (deficit); side specificity O side-specific, O_strict side-specific, DOT side-specific, END side-specific.

## Against counts

A1's counts not available yet (rerun when they are).

## Released FreqBLiMP (Pythia-1.4B LP): accuracy and a descriptive item-level association

Item margin (good − bad LP) on the bad verb's side measure, with band fixed effects and the bad verb's Zipf and log token count as controls; verbs resampled within band. Analysed only with ≥ 3 mapped verb types in every band and ≥ 15 in total; otherwise descriptive. This is an association, not a test of the mechanism (lexical, inflectional and good-verb differences are not controlled).

| paradigm | accuracy Head / Tail / XTail (%, 95% CI) | bad verb | mapped items | verb types H / T / X | coefficient (95% CI) | association |
|---|---|---|---|---|---|---|
| causative | 79.3 [76.7, 81.8] / 86.7 [84.5, 88.7] / 78.8 [76.2, 81.2] | I | 2200 | 23 / 39 / 53 | -0.794 [-1.053, -0.521] | **negative** |
| transitive | 75.3 [72.6, 77.8] / 75.5 [72.8, 78.2] / 69.2 [66.4, 72.1] | I | 2034 | 43 / 117 / 85 | -0.214 [-0.500, +0.043] | **unresolved** |
| inchoative | 87.2 [85.0, 89.2] / 91.5 [89.7, 93.2] / 81.2 [78.7, 83.6] | T | 501 | 4 / 6 / 8 | — | **descriptive only (too few verb types)** |
| drop_argument | 80.0 [77.6, 82.6] / 60.5 [57.5, 63.5] / 71.3 [68.5, 74.2] | T | 377 | 1 / 2 / 4 | — | **descriptive only (too few verb types)** |
| intransitive | 67.5 [64.6, 70.4] / 73.6 [70.8, 76.3] / 69.8 [66.9, 72.6] | T | 938 | 78 / 126 / 119 | -0.254 [-0.516, +0.026] | **unresolved** |

