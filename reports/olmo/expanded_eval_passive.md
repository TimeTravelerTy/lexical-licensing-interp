# Expanded eval set: behavior and passive transfer (OLMo-2-1B)



Rules: `reports/expanded_eval_rules.md`. Scripts: `scripts/build_expanded_eval.py`, `scripts/build_passive_test.py`, `scripts/run_passive_test.py`, `scripts/analyze_expanded_eval.py`.

Reproduction: v's D_bad(by) on the original plain pairs differs from the stored run by at most 5.6e-17 nats.

## Behavior (natural passive *by* gap)

| population | pairs | mean pair gap | share of pairs > 0 | slope on intransitive Zipf |
|---|---:|---|---|---|
| original plain | 59 | 0.45 | 0.66 | +0.13 [-0.27, +0.61] |
| new plain | 74 | 0.46 | 0.68 | +0.17 [-0.19, +0.53] |
| pooled plain | 133 | 0.46 | 0.67 | +0.12 [-0.13, +0.42] |
| original prep | 60 | 0.75 | 0.78 | -0.05 [-0.48, +0.32] |
| new prep | 68 | 0.99 | 0.72 | +0.75 [+0.24, +1.26] |
| pooled prep | 128 | 0.88 | 0.75 | +0.28 [-0.05, +0.60] |
| new plain, no selection-flagged | 62 | 0.46 | 0.68 | +0.11 [-0.23, +0.41] |
| new plain, no FreqBLiMP transitives | 15 | 0.43 | 0.60 | -1.16 [-2.10, -0.41] |
| new plain, single-token | 4 | 1.56 | 1.00 | +1.18 [-24.56, +5.24] |

## Passive transfer, D_bad(*by*) (nats) and D / natural gap

| site | population | pairs | natural gap | v: D_bad(by) | v: D/gap | u_md: D_bad(by) | u_md / v | v: O class |
|---:|---|---:|---|---|---|---|---|---|
| 9 | original plain | 59 | 0.45 [0.15, 0.75] | 0.44 [0.38, 0.50] | 0.98 [0.56, 2.74] | 0.42 [0.36, 0.48] | 0.95 | RISE |
| 9 | new plain | 74 | 0.46 [0.23, 0.72] | 0.37 [0.31, 0.43] | 0.79 [0.52, 1.58] | 0.34 [0.29, 0.40] | 0.93 | RISE |
| 9 | pooled plain | 133 | 0.46 [0.26, 0.66] | 0.40 [0.35, 0.45] | 0.87 [0.60, 1.50] | 0.37 [0.33, 0.43] | 0.94 | RISE |
| 9 | original prep | 60 | 0.75 [0.42, 1.09] | 0.94 [0.81, 1.08] | 1.25 [0.89, 2.17] | 0.74 [0.64, 0.86] | 0.79 | RISE |
| 9 | new prep | 68 | 0.99 [0.64, 1.35] | 0.97 [0.82, 1.13] | 0.98 [0.73, 1.44] | 0.76 [0.64, 0.89] | 0.78 | RISE |
| 9 | pooled prep | 128 | 0.88 [0.61, 1.13] | 0.96 [0.85, 1.07] | 1.09 [0.86, 1.52] | 0.75 [0.66, 0.85] | 0.78 | RISE |
| 9 | new plain, no selection-flagged | 62 | 0.46 [0.19, 0.72] | 0.39 [0.33, 0.45] | 0.85 [0.53, 1.96] | 0.36 [0.30, 0.42] | 0.92 | RISE |
| 9 | new plain, no FreqBLiMP transitives | 15 | 0.43 [-0.11, 0.97] | 0.33 [0.22, 0.45] | 0.76 [-4.20, 6.54] | 0.30 [0.20, 0.42] | 0.91 | RISE |
| 9 | new plain, single-token | 4 | 1.56 [0.81, 2.72] | 0.48 [0.28, 0.72] | 0.31 [0.26, 0.42] | 0.61 [0.42, 0.87] | 1.25 | RISE |
| 11 | original plain | 59 | 0.45 [0.15, 0.75] | 0.15 [0.08, 0.21] | 0.33 [0.15, 0.95] | 0.44 [0.38, 0.51] | 3.00 | RISE |
| 11 | new plain | 74 | 0.46 [0.23, 0.72] | 0.13 [0.08, 0.18] | 0.28 [0.15, 0.60] | 0.36 [0.31, 0.42] | 2.80 | RISE |
| 11 | pooled plain | 133 | 0.46 [0.26, 0.66] | 0.14 [0.08, 0.19] | 0.30 [0.17, 0.56] | 0.40 [0.35, 0.45] | 2.89 | RISE |
| 11 | original prep | 60 | 0.75 [0.42, 1.09] | 0.52 [0.43, 0.62] | 0.69 [0.47, 1.21] | 0.64 [0.54, 0.75] | 1.23 | RISE |
| 11 | new prep | 68 | 0.99 [0.64, 1.35] | 0.54 [0.43, 0.65] | 0.54 [0.39, 0.82] | 0.68 [0.56, 0.79] | 1.26 | RISE |
| 11 | pooled prep | 128 | 0.88 [0.61, 1.13] | 0.53 [0.46, 0.61] | 0.60 [0.46, 0.86] | 0.66 [0.58, 0.75] | 1.25 | RISE |
| 11 | new plain, no selection-flagged | 62 | 0.46 [0.19, 0.72] | 0.14 [0.09, 0.20] | 0.32 [0.16, 0.78] | 0.37 [0.32, 0.43] | 2.58 | RISE |
| 11 | new plain, no FreqBLiMP transitives | 15 | 0.43 [-0.11, 0.97] | 0.09 [-0.02, 0.21] | 0.21 [-0.86, 1.90] | 0.31 [0.22, 0.40] | 3.36 | RISE |
| 11 | new plain, single-token | 4 | 1.56 [0.81, 2.72] | -0.02 [-0.22, 0.20] | -0.01 [-0.28, 0.18] | 0.55 [0.47, 0.64] | -31.02 | RISE |
| 12 | original plain | 59 | 0.45 [0.15, 0.75] | -0.74 [-0.88, -0.60] | -1.65 [-4.78, -0.98] | 0.19 [0.11, 0.26] | -0.26 | RISE |
| 12 | new plain | 74 | 0.46 [0.23, 0.72] | -0.81 [-0.94, -0.68] | -1.75 [-3.51, -1.14] | 0.05 [-0.00, 0.11] | -0.07 | RISE |
| 12 | pooled plain | 133 | 0.46 [0.26, 0.66] | -0.78 [-0.90, -0.66] | -1.70 [-2.87, -1.17] | 0.11 [0.06, 0.16] | -0.15 | RISE |
| 12 | original prep | 60 | 0.75 [0.42, 1.09] | -0.62 [-0.74, -0.50] | -0.82 [-1.44, -0.56] | -0.02 [-0.09, 0.06] | 0.03 | RISE |
| 12 | new prep | 68 | 0.99 [0.64, 1.35] | -0.65 [-0.77, -0.53] | -0.65 [-0.98, -0.48] | -0.01 [-0.09, 0.06] | 0.02 | RISE |
| 12 | pooled prep | 128 | 0.88 [0.61, 1.13] | -0.63 [-0.74, -0.54] | -0.72 [-1.03, -0.55] | -0.02 [-0.07, 0.04] | 0.02 | RISE |
| 12 | new plain, no selection-flagged | 62 | 0.46 [0.19, 0.72] | -0.84 [-0.98, -0.69] | -1.84 [-4.27, -1.18] | 0.05 [-0.01, 0.11] | -0.06 | RISE |
| 12 | new plain, no FreqBLiMP transitives | 15 | 0.43 [-0.11, 0.97] | -0.79 [-1.03, -0.55] | -1.82 [-15.04, 10.01] | 0.02 [-0.09, 0.15] | -0.03 | RISE |
| 12 | new plain, single-token | 4 | 1.56 [0.81, 2.72] | -0.71 [-0.99, -0.50] | -0.46 [-0.95, -0.19] | 0.07 [-0.04, 0.16] | -0.09 | RISE |
| 13 | original plain | 59 | 0.45 [0.15, 0.75] | -1.10 [-1.27, -0.93] | -2.45 [-7.09, -1.44] | -0.09 [-0.17, -0.02] | 0.08 | RISE |
| 13 | new plain | 74 | 0.46 [0.23, 0.72] | -1.05 [-1.22, -0.90] | -2.28 [-4.63, -1.47] | -0.12 [-0.18, -0.05] | 0.11 | RISE |
| 13 | pooled plain | 133 | 0.46 [0.26, 0.66] | -1.07 [-1.23, -0.92] | -2.35 [-4.09, -1.61] | -0.11 [-0.17, -0.05] | 0.10 | RISE |
| 13 | original prep | 60 | 0.75 [0.42, 1.09] | -0.81 [-0.95, -0.68] | -1.08 [-1.90, -0.72] | -0.01 [-0.07, 0.04] | 0.02 | RISE |
| 13 | new prep | 68 | 0.99 [0.64, 1.35] | -0.87 [-1.01, -0.74] | -0.88 [-1.38, -0.62] | -0.07 [-0.15, 0.01] | 0.08 | RISE |
| 13 | pooled prep | 128 | 0.88 [0.61, 1.13] | -0.84 [-0.96, -0.73] | -0.96 [-1.40, -0.73] | -0.04 [-0.10, 0.01] | 0.05 | RISE |
| 13 | new plain, no selection-flagged | 62 | 0.46 [0.19, 0.72] | -1.07 [-1.24, -0.91] | -2.34 [-5.49, -1.48] | -0.11 [-0.18, -0.04] | 0.10 | RISE |
| 13 | new plain, no FreqBLiMP transitives | 15 | 0.43 [-0.11, 0.97] | -1.06 [-1.31, -0.81] | -2.44 [-20.22, 13.75] | -0.15 [-0.26, -0.02] | 0.14 | RISE |
| 13 | new plain, single-token | 4 | 1.56 [0.81, 2.72] | -1.08 [-1.29, -0.92] | -0.70 [-1.37, -0.38] | -0.18 [-0.33, -0.05] | 0.17 | RISE |

## Declared comparison 1: does transfer generalize to new plain verbs?

Paired (one set of context and donor draws). Reading at the rule sites; other sites descriptive.

| instrument | site | D_bad(by) new | D_bad(by) original | ratio new / original | D_new − 0.5·D_orig | reading |
|---|---:|---|---|---|---|---|
| v | 9 | 0.37 [0.31, 0.42] | 0.44 [0.38, 0.50] | 0.84 [0.69, 1.01] | 0.15 [0.09, 0.21] | generalizes |
| v | 11 | 0.13 [0.08, 0.18] | 0.15 [0.08, 0.21] | 0.88 [0.49, 1.57] | 0.06 [-0.00, 0.11] | criterion not met: inconclusive |
| v | 12 | -0.81 [-0.96, -0.67] | -0.74 [-0.87, -0.59] | 1.10 [0.94, 1.32] | -0.44 [-0.56, -0.33] | descriptive |
| v | 13 | -1.05 [-1.22, -0.89] | -1.10 [-1.26, -0.92] | 0.96 [0.86, 1.09] | -0.51 [-0.63, -0.40] | descriptive |
| umd | 9 | 0.34 [0.29, 0.40] | 0.42 [0.36, 0.47] | 0.82 [0.69, 0.98] | 0.13 [0.08, 0.19] | generalizes |
| umd | 11 | 0.36 [0.31, 0.41] | 0.44 [0.38, 0.51] | 0.82 [0.69, 0.98] | 0.14 [0.09, 0.19] | generalizes |
| umd | 12 | 0.05 [-0.00, 0.11] | 0.19 [0.12, 0.27] | 0.28 [-0.00, 0.63] | -0.04 [-0.10, 0.02] | descriptive |
| umd | 13 | -0.12 [-0.18, -0.05] | -0.09 [-0.17, -0.02] | 1.25 [0.59, 4.23] | -0.07 [-0.13, -0.01] | descriptive |

## Checks

```
{
 "v_passive_checks": {
  "9": {
   "self_patch_max_abs_dM": 0.0,
   "heldout_recompute_max_abs_dM": 0.03632831573486328,
   "heldout_recompute_median_abs_dM": 0.0
  },
  "11": {
   "self_patch_max_abs_dM": 0.0,
   "heldout_recompute_max_abs_dM": 0.0375523567199707,
   "heldout_recompute_median_abs_dM": 0.0
  },
  "12": {
   "self_patch_max_abs_dM": 0.0,
   "heldout_recompute_max_abs_dM": 0.038712978363038,
   "heldout_recompute_median_abs_dM": 0.0
  },
  "13": {
   "self_patch_max_abs_dM": 0.0,
   "heldout_recompute_max_abs_dM": 0.028995037078857866,
   "heldout_recompute_median_abs_dM": 0.0
  }
 },
 "umd_passive_checks": {
  "9": {
   "self_patch_max_abs_dM": 0.0
  },
  "11": {
   "self_patch_max_abs_dM": 0.0
  },
  "12": {
   "self_patch_max_abs_dM": 0.0
  },
  "13": {
   "self_patch_max_abs_dM": 0.0
  }
 },
 "v_original_D_bad_by_vs_stored_max_abs": 5.551115123125783e-17,
 "reproduction_ok": true
}
```
