# Expanded eval set: behavior and passive transfer (Pythia-1.4B)



Rules: `reports/expanded_eval_rules.md`. Scripts: `scripts/build_expanded_eval.py`, `scripts/build_passive_test.py`, `scripts/run_passive_test.py`, `scripts/analyze_expanded_eval.py`.

Reproduction: v's D_bad(by) on the original plain pairs differs from the stored run by at most 1.9e-05 nats.

## Behavior (natural passive *by* gap)

| population | pairs | mean pair gap | share of pairs > 0 | slope on intransitive Zipf |
|---|---:|---|---|---|
| original plain | 59 | 0.67 | 0.71 | +0.09 [-0.38, +0.68] |
| new plain | 74 | 0.75 | 0.76 | +0.23 [-0.19, +0.59] |
| pooled plain | 133 | 0.71 | 0.74 | +0.11 [-0.19, +0.45] |
| original prep | 60 | 0.93 | 0.77 | +0.11 [-0.33, +0.53] |
| new prep | 68 | 1.13 | 0.75 | +0.61 [+0.06, +1.12] |
| pooled prep | 128 | 1.03 | 0.76 | +0.31 [-0.04, +0.64] |
| new plain, no selection-flagged | 62 | 0.70 | 0.74 | +0.24 [-0.23, +0.65] |
| new plain, no FreqBLiMP transitives | 15 | 0.49 | 0.73 | -1.09 [-1.95, -0.30] |
| new plain, single-token | 6 | 1.20 | 0.67 | +2.50 [+0.35, +5.42] |

## Passive transfer, D_bad(*by*) (nats) and D / natural gap

| site | population | pairs | natural gap | v: D_bad(by) | v: D/gap | u_md: D_bad(by) | u_md / v | v: O class |
|---:|---|---:|---|---|---|---|---|---|
| 8 | original plain | 59 | 0.67 [0.32, 1.02] | 0.71 [0.61, 0.81] | 1.05 [0.70, 2.13] | 0.62 [0.54, 0.72] | 0.88 | NO RISE |
| 8 | new plain | 74 | 0.75 [0.48, 1.02] | 0.72 [0.62, 0.82] | 0.96 [0.71, 1.46] | 0.60 [0.52, 0.70] | 0.84 | unresolved |
| 8 | pooled plain | 133 | 0.71 [0.49, 0.93] | 0.71 [0.64, 0.80] | 1.00 [0.76, 1.43] | 0.61 [0.54, 0.69] | 0.86 | NO RISE |
| 8 | original prep | 60 | 0.93 [0.57, 1.29] | 1.31 [1.15, 1.49] | 1.41 [1.03, 2.29] | 1.03 [0.90, 1.18] | 0.79 | RISE |
| 8 | new prep | 68 | 1.13 [0.79, 1.46] | 1.32 [1.15, 1.51] | 1.17 [0.92, 1.61] | 1.02 [0.88, 1.17] | 0.77 | RISE |
| 8 | pooled prep | 128 | 1.03 [0.76, 1.28] | 1.32 [1.18, 1.47] | 1.28 [1.04, 1.69] | 1.02 [0.91, 1.15] | 0.78 | RISE |
| 8 | new plain, no selection-flagged | 62 | 0.70 [0.40, 1.01] | 0.75 [0.65, 0.87] | 1.07 [0.77, 1.85] | 0.63 [0.53, 0.73] | 0.84 | unresolved |
| 8 | new plain, no FreqBLiMP transitives | 15 | 0.49 [-0.02, 0.97] | 0.72 [0.49, 0.97] | 1.48 [-3.09, 8.55] | 0.58 [0.37, 0.82] | 0.81 | unresolved |
| 8 | new plain, single-token | 6 | 1.20 [0.74, 1.65] | 0.94 [0.84, 1.06] | 0.79 [0.57, 1.26] | 0.80 [0.69, 0.91] | 0.85 | RISE |
| 12 | original plain | 59 | 0.67 [0.32, 1.02] | 0.79 [0.70, 0.88] | 1.17 [0.77, 2.39] | 0.89 [0.78, 1.00] | 1.12 | RISE |
| 12 | new plain | 74 | 0.75 [0.48, 1.02] | 0.80 [0.71, 0.88] | 1.07 [0.80, 1.63] | 0.87 [0.77, 0.97] | 1.09 | RISE |
| 12 | pooled plain | 133 | 0.71 [0.49, 0.93] | 0.79 [0.72, 0.86] | 1.11 [0.85, 1.58] | 0.88 [0.79, 0.96] | 1.10 | RISE |
| 12 | original prep | 60 | 0.93 [0.57, 1.29] | 0.99 [0.89, 1.10] | 1.07 [0.78, 1.75] | 0.94 [0.84, 1.06] | 0.95 | RISE |
| 12 | new prep | 68 | 1.13 [0.79, 1.46] | 1.07 [0.95, 1.19] | 0.95 [0.74, 1.30] | 1.02 [0.91, 1.15] | 0.96 | RISE |
| 12 | pooled prep | 128 | 1.03 [0.76, 1.28] | 1.03 [0.94, 1.13] | 1.00 [0.81, 1.33] | 0.99 [0.89, 1.08] | 0.96 | RISE |
| 12 | new plain, no selection-flagged | 62 | 0.70 [0.40, 1.01] | 0.82 [0.73, 0.91] | 1.17 [0.84, 1.98] | 0.88 [0.77, 0.99] | 1.07 | RISE |
| 12 | new plain, no FreqBLiMP transitives | 15 | 0.49 [-0.02, 0.97] | 0.76 [0.58, 0.93] | 1.57 [-3.23, 9.17] | 0.81 [0.58, 1.03] | 1.06 | RISE |
| 12 | new plain, single-token | 6 | 1.20 [0.74, 1.65] | 0.80 [0.57, 1.07] | 0.67 [0.42, 1.22] | 1.06 [0.91, 1.25] | 1.33 | RISE |
| 17 | original plain | 59 | 0.67 [0.32, 1.02] | -0.93 [-1.10, -0.76] | -1.38 [-2.87, -0.88] | 0.42 [0.35, 0.50] | -0.45 | RISE |
| 17 | new plain | 74 | 0.75 [0.48, 1.02] | -0.95 [-1.12, -0.78] | -1.28 [-1.99, -0.90] | 0.37 [0.31, 0.43] | -0.38 | RISE |
| 17 | pooled plain | 133 | 0.71 [0.49, 0.93] | -0.94 [-1.11, -0.79] | -1.32 [-1.94, -0.95] | 0.39 [0.33, 0.45] | -0.41 | RISE |
| 17 | original prep | 60 | 0.93 [0.57, 1.29] | -0.68 [-0.84, -0.53] | -0.74 [-1.25, -0.49] | 0.44 [0.37, 0.52] | -0.65 | RISE |
| 17 | new prep | 68 | 1.13 [0.79, 1.46] | -0.66 [-0.80, -0.52] | -0.58 [-0.86, -0.41] | 0.49 [0.41, 0.57] | -0.74 | RISE |
| 17 | pooled prep | 128 | 1.03 [0.76, 1.28] | -0.67 [-0.81, -0.54] | -0.65 [-0.94, -0.48] | 0.47 [0.40, 0.53] | -0.70 | RISE |
| 17 | new plain, no selection-flagged | 62 | 0.70 [0.40, 1.01] | -0.95 [-1.13, -0.78] | -1.36 [-2.44, -0.92] | 0.38 [0.31, 0.45] | -0.40 | RISE |
| 17 | new plain, no FreqBLiMP transitives | 15 | 0.49 [-0.02, 0.97] | -1.04 [-1.30, -0.79] | -2.13 [-14.59, 5.21] | 0.32 [0.22, 0.41] | -0.31 | RISE |
| 17 | new plain, single-token | 6 | 1.20 [0.74, 1.65] | -1.34 [-1.59, -1.11] | -1.12 [-2.00, -0.71] | 0.36 [0.21, 0.54] | -0.27 | RISE |

## Declared comparison 1: does transfer generalize to new plain verbs?

Paired (one set of context and donor draws). Reading at the rule sites; other sites descriptive.

| instrument | site | D_bad(by) new | D_bad(by) original | ratio new / original | D_new − 0.5·D_orig | reading |
|---|---:|---|---|---|---|---|
| v | 8 | 0.72 [0.62, 0.82] | 0.71 [0.62, 0.80] | 1.01 [0.88, 1.17] | 0.36 [0.27, 0.46] | generalizes |
| v | 12 | 0.80 [0.71, 0.88] | 0.79 [0.71, 0.88] | 1.01 [0.90, 1.14] | 0.40 [0.32, 0.48] | generalizes |
| v | 17 | -0.95 [-1.14, -0.78] | -0.93 [-1.10, -0.77] | 1.03 [0.90, 1.16] | -0.49 [-0.61, -0.37] | descriptive |
| umd | 8 | 0.60 [0.52, 0.70] | 0.62 [0.54, 0.71] | 0.96 [0.82, 1.13] | 0.29 [0.21, 0.37] | generalizes |
| umd | 12 | 0.87 [0.77, 0.97] | 0.89 [0.78, 1.00] | 0.98 [0.86, 1.11] | 0.42 [0.33, 0.52] | generalizes |
| umd | 17 | 0.37 [0.30, 0.43] | 0.42 [0.35, 0.50] | 0.86 [0.71, 1.07] | 0.15 [0.09, 0.22] | descriptive |

## Checks

```
{
 "v_passive_checks": {
  "8": {
   "self_patch_max_abs_dM": 0.0,
   "heldout_recompute_max_abs_dM": 0.2882232666015625,
   "heldout_recompute_median_abs_dM": 4.440892098500626e-16
  },
  "12": {
   "self_patch_max_abs_dM": 0.0,
   "heldout_recompute_max_abs_dM": 0.36195945739746094,
   "heldout_recompute_median_abs_dM": 0.0005016326904296875
  },
  "17": {
   "self_patch_max_abs_dM": 0.0,
   "heldout_recompute_max_abs_dM": 0.4295167922973633,
   "heldout_recompute_median_abs_dM": 0.00437307357788097
  }
 },
 "umd_passive_checks": {
  "8": {
   "self_patch_max_abs_dM": 0.0
  },
  "12": {
   "self_patch_max_abs_dM": 0.0
  },
  "17": {
   "self_patch_max_abs_dM": 0.0
  }
 },
 "v_original_D_bad_by_vs_stored_max_abs": 1.9311904907226562e-05,
 "reproduction_ok": true
}
```
