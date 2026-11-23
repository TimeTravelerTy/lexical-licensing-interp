# Expanded eval set: behavior and passive transfer (Gemma-3-1B)



Rules: `reports/expanded_eval_rules.md`. Scripts: `scripts/build_expanded_eval.py`, `scripts/build_passive_test.py`, `scripts/run_passive_test.py`, `scripts/analyze_expanded_eval.py`.

Reproduction: v's D_bad(by) on the original plain pairs differs from the stored run by at most 1.1e-16 nats.

## Behavior (natural passive *by* gap)

| population | pairs | mean pair gap | share of pairs > 0 | slope on intransitive Zipf |
|---|---:|---|---|---|
| original plain | 59 | 0.37 | 0.61 | +0.41 [+0.03, +0.92] |
| new plain | 74 | 0.45 | 0.66 | -0.10 [-0.44, +0.23] |
| pooled plain | 133 | 0.42 | 0.64 | +0.14 [-0.09, +0.42] |
| original prep | 60 | 0.43 | 0.72 | +0.04 [-0.40, +0.45] |
| new prep | 68 | 0.60 | 0.65 | +0.78 [+0.29, +1.24] |
| pooled prep | 128 | 0.52 | 0.68 | +0.35 [+0.03, +0.67] |
| new plain, no selection-flagged | 62 | 0.47 | 0.68 | -0.11 [-0.55, +0.29] |
| new plain, no FreqBLiMP transitives | 15 | 0.23 | 0.67 | -1.08 [-1.71, -0.57] |
| new plain, single-token | 6 | 0.31 | 0.50 | +2.13 [+0.40, +4.23] |

## Passive transfer, D_bad(*by*) (nats) and D / natural gap

| site | population | pairs | natural gap | v: D_bad(by) | v: D/gap | u_md: D_bad(by) | u_md / v | v: O class |
|---:|---|---:|---|---|---|---|---|---|
| 5 | original plain | 59 | 0.37 [0.07, 0.67] | 0.57 [0.45, 0.70] | 1.54 [0.88, 6.12] | 0.51 [0.33, 0.71] | 0.90 | RISE |
| 5 | new plain | 74 | 0.45 [0.21, 0.68] | 0.46 [0.36, 0.56] | 1.01 [0.67, 2.07] | 0.38 [0.24, 0.54] | 0.83 | RISE |
| 5 | pooled plain | 133 | 0.42 [0.23, 0.62] | 0.51 [0.43, 0.60] | 1.22 [0.83, 2.18] | 0.44 [0.27, 0.61] | 0.86 | RISE |
| 5 | original prep | 60 | 0.43 [0.08, 0.77] | 0.73 [0.57, 0.89] | 1.69 [0.97, 6.95] | 0.58 [0.36, 0.82] | 0.80 | RISE |
| 5 | new prep | 68 | 0.60 [0.28, 0.93] | 0.71 [0.57, 0.86] | 1.18 [0.79, 2.40] | 0.61 [0.38, 0.85] | 0.86 | RISE |
| 5 | pooled prep | 128 | 0.52 [0.29, 0.75] | 0.72 [0.61, 0.84] | 1.38 [0.97, 2.44] | 0.60 [0.38, 0.82] | 0.83 | RISE |
| 5 | new plain, no selection-flagged | 62 | 0.47 [0.20, 0.74] | 0.49 [0.39, 0.61] | 1.06 [0.68, 2.36] | 0.39 [0.24, 0.55] | 0.79 | RISE |
| 5 | new plain, no FreqBLiMP transitives | 15 | 0.23 [-0.15, 0.61] | 0.46 [0.31, 0.66] | 2.00 [-17.14, 16.32] | 0.33 [0.18, 0.52] | 0.72 | RISE |
| 5 | new plain, single-token | 6 | 0.31 [-0.46, 0.99] | 0.53 [0.41, 0.66] | 1.70 [-16.79, 18.12] | 0.61 [0.37, 0.85] | 1.16 | RISE |
| 8 | original plain | 59 | 0.37 [0.07, 0.67] | 0.63 [0.54, 0.73] | 1.71 [0.96, 6.83] | 0.44 [0.32, 0.58] | 0.69 | RISE |
| 8 | new plain | 74 | 0.45 [0.21, 0.68] | 0.54 [0.46, 0.62] | 1.19 [0.79, 2.49] | 0.41 [0.30, 0.53] | 0.76 | RISE |
| 8 | pooled plain | 133 | 0.42 [0.23, 0.62] | 0.58 [0.51, 0.66] | 1.39 [0.95, 2.46] | 0.42 [0.31, 0.54] | 0.73 | RISE |
| 8 | original prep | 60 | 0.43 [0.08, 0.77] | 0.85 [0.72, 0.99] | 1.97 [1.13, 8.65] | 0.58 [0.42, 0.74] | 0.68 | RISE |
| 8 | new prep | 68 | 0.60 [0.28, 0.93] | 0.88 [0.76, 1.01] | 1.47 [0.98, 3.02] | 0.56 [0.40, 0.73] | 0.63 | RISE |
| 8 | pooled prep | 128 | 0.52 [0.29, 0.75] | 0.87 [0.77, 0.98] | 1.66 [1.18, 2.97] | 0.57 [0.42, 0.73] | 0.66 | RISE |
| 8 | new plain, no selection-flagged | 62 | 0.47 [0.20, 0.74] | 0.55 [0.47, 0.64] | 1.18 [0.76, 2.63] | 0.44 [0.32, 0.57] | 0.79 | RISE |
| 8 | new plain, no FreqBLiMP transitives | 15 | 0.23 [-0.15, 0.61] | 0.49 [0.38, 0.62] | 2.12 [-19.86, 18.27] | 0.35 [0.23, 0.48] | 0.70 | RISE |
| 8 | new plain, single-token | 6 | 0.31 [-0.46, 0.99] | 0.71 [0.61, 0.82] | 2.29 [-21.54, 24.81] | 0.46 [0.32, 0.61] | 0.64 | RISE |
| 12 | original plain | 59 | 0.37 [0.07, 0.67] | 0.96 [0.85, 1.09] | 2.61 [1.47, 10.59] | 0.63 [0.49, 0.80] | 0.66 | RISE |
| 12 | new plain | 74 | 0.45 [0.21, 0.68] | 0.81 [0.71, 0.92] | 1.79 [1.21, 3.79] | 0.57 [0.44, 0.71] | 0.70 | RISE |
| 12 | pooled plain | 133 | 0.42 [0.23, 0.62] | 0.88 [0.79, 0.97] | 2.11 [1.45, 3.80] | 0.60 [0.46, 0.74] | 0.68 | RISE |
| 12 | original prep | 60 | 0.43 [0.08, 0.77] | 1.21 [1.04, 1.38] | 2.80 [1.63, 12.27] | 0.79 [0.62, 0.98] | 0.66 | RISE |
| 12 | new prep | 68 | 0.60 [0.28, 0.93] | 1.26 [1.12, 1.42] | 2.11 [1.43, 4.33] | 0.79 [0.61, 0.97] | 0.63 | RISE |
| 12 | pooled prep | 128 | 0.52 [0.29, 0.75] | 1.24 [1.11, 1.37] | 2.38 [1.69, 4.28] | 0.79 [0.62, 0.97] | 0.64 | RISE |
| 12 | new plain, no selection-flagged | 62 | 0.47 [0.20, 0.74] | 0.82 [0.72, 0.93] | 1.76 [1.13, 3.96] | 0.59 [0.45, 0.73] | 0.72 | RISE |
| 12 | new plain, no FreqBLiMP transitives | 15 | 0.23 [-0.15, 0.61] | 0.75 [0.64, 0.89] | 3.26 [-31.04, 26.45] | 0.53 [0.37, 0.71] | 0.70 | RISE |
| 12 | new plain, single-token | 6 | 0.31 [-0.46, 0.99] | 1.05 [0.92, 1.20] | 3.40 [-30.02, 36.35] | 0.71 [0.55, 0.87] | 0.67 | RISE |
| 15 | original plain | 59 | 0.37 [0.07, 0.67] | 0.88 [0.79, 0.98] | 2.39 [1.29, 10.43] | 0.77 [0.65, 0.89] | 0.87 | RISE |
| 15 | new plain | 74 | 0.45 [0.21, 0.68] | 0.82 [0.72, 0.92] | 1.80 [1.21, 3.82] | 0.67 [0.56, 0.78] | 0.82 | RISE |
| 15 | pooled plain | 133 | 0.42 [0.23, 0.62] | 0.85 [0.77, 0.93] | 2.03 [1.41, 3.69] | 0.71 [0.60, 0.82] | 0.84 | RISE |
| 15 | original prep | 60 | 0.43 [0.08, 0.77] | 1.39 [1.22, 1.55] | 3.22 [1.85, 14.14] | 0.88 [0.74, 1.02] | 0.63 | RISE |
| 15 | new prep | 68 | 0.60 [0.28, 0.93] | 1.57 [1.41, 1.74] | 2.61 [1.76, 5.31] | 0.97 [0.81, 1.11] | 0.62 | RISE |
| 15 | pooled prep | 128 | 0.52 [0.29, 0.75] | 1.48 [1.35, 1.62] | 2.85 [2.01, 5.07] | 0.92 [0.79, 1.06] | 0.62 | RISE |
| 15 | new plain, no selection-flagged | 62 | 0.47 [0.20, 0.74] | 0.81 [0.70, 0.92] | 1.74 [1.13, 3.92] | 0.67 [0.56, 0.78] | 0.83 | RISE |
| 15 | new plain, no FreqBLiMP transitives | 15 | 0.23 [-0.15, 0.61] | 0.77 [0.62, 0.91] | 3.32 [-30.31, 27.59] | 0.64 [0.51, 0.77] | 0.83 | RISE |
| 15 | new plain, single-token | 6 | 0.31 [-0.46, 0.99] | 1.09 [0.91, 1.31] | 3.52 [-31.94, 35.62] | 0.95 [0.79, 1.11] | 0.87 | RISE |

## Declared comparison 1: does transfer generalize to new plain verbs?

Paired (one set of context and donor draws). Reading at the rule sites; other sites descriptive.

| instrument | site | D_bad(by) new | D_bad(by) original | ratio new / original | D_new − 0.5·D_orig | reading |
|---|---:|---|---|---|---|---|
| v | 5 | 0.46 [0.36, 0.56] | 0.57 [0.45, 0.69] | 0.81 [0.62, 1.05] | 0.17 [0.08, 0.28] | generalizes |
| v | 8 | 0.54 [0.46, 0.62] | 0.63 [0.53, 0.73] | 0.86 [0.72, 1.03] | 0.23 [0.14, 0.31] | generalizes |
| v | 12 | 0.81 [0.72, 0.91] | 0.96 [0.85, 1.08] | 0.85 [0.74, 0.97] | 0.33 [0.24, 0.42] | descriptive |
| v | 15 | 0.82 [0.72, 0.91] | 0.88 [0.79, 0.98] | 0.93 [0.81, 1.05] | 0.38 [0.28, 0.47] | descriptive |
| umd | 5 | 0.38 [0.23, 0.53] | 0.51 [0.32, 0.72] | 0.74 [0.59, 0.91] | 0.12 [0.04, 0.21] | generalizes |
| umd | 8 | 0.41 [0.30, 0.53] | 0.44 [0.31, 0.58] | 0.95 [0.78, 1.14] | 0.19 [0.12, 0.27] | generalizes |
| umd | 12 | 0.57 [0.43, 0.71] | 0.63 [0.48, 0.79] | 0.90 [0.77, 1.06] | 0.25 [0.17, 0.35] | descriptive |
| umd | 15 | 0.67 [0.56, 0.78] | 0.77 [0.64, 0.89] | 0.88 [0.77, 1.00] | 0.29 [0.21, 0.38] | descriptive |

## Checks

```
{
 "v_passive_checks": {
  "5": {
   "self_patch_max_abs_dM": 0.030130386352539062,
   "heldout_recompute_max_abs_dM": 0.09097576141357422,
   "heldout_recompute_median_abs_dM": 0.0
  },
  "8": {
   "self_patch_max_abs_dM": 0.05850982666015625,
   "heldout_recompute_max_abs_dM": 0.057201385498046875,
   "heldout_recompute_median_abs_dM": 0.0
  },
  "12": {
   "self_patch_max_abs_dM": 0.05901336669921875,
   "heldout_recompute_max_abs_dM": 0.059413909912109375,
   "heldout_recompute_median_abs_dM": 0.0
  },
  "15": {
   "self_patch_max_abs_dM": 0.05316925048828125,
   "heldout_recompute_max_abs_dM": 0.06627273559570312,
   "heldout_recompute_median_abs_dM": 0.0
  }
 },
 "umd_passive_checks": {
  "5": {
   "self_patch_max_abs_dM": 0.05915069580078125
  },
  "8": {
   "self_patch_max_abs_dM": 0.07798099517822266
  },
  "12": {
   "self_patch_max_abs_dM": 0.1024332046508789
  },
  "15": {
   "self_patch_max_abs_dM": 0.044551849365234375
  }
 },
 "v_original_D_bad_by_vs_stored_max_abs": 1.1102230246251565e-16,
 "reproduction_ok": true
}
```
