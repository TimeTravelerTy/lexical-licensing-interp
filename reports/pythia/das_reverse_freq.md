# Reverse frequency transfer: DAS trained only on rare verbs (Pythia-1.4B)

Rules: `reports/pythia/das_reverse_freq_rules.md` (with its Nov 23 amendment). Scripts: `scripts/run_das_reverse_freq.py`, `scripts/run_passive_test.py`, `scripts/analyze_das_reverse_freq.py`.

**Overall (sites 8 and 12, Head band): F invalid; U invalid; B invalid.** Problems: passive checks for U at site 12: {'self_patch_max_abs_dM': 0.0, 'heldout_recompute_max_abs_dM': 0.3627977371215816, 'heldout_recompute_median_abs_dM': 0.0010113716125488281}

Training pairs: 43 rare plain pairs (F keeps 42 after the behavior filter; dropped: tweeze/teethe); B, both sides rare: 28 pairs, 27 after the filter. Failing verbs (items on the class side, of 7): teethe (I, 3).
Excluded overlap pairs (primary): F: new/xtail/besmirch/reemerge, new/xtail/cajole/saunter; U: new/xtail/besmirch/reemerge, new/xtail/cajole/saunter; B: new/xtail/besmirch/reemerge

- **Site 8, F: transfers.** D_bad(by) on Head 1.09 [0.96, 1.22] (v 1.08 [0.96, 1.20]; ratio 1.00 [0.89, 1.14]; D_rare − 0.5·D_v 0.54 [0.43, 0.67]); all bands ratio 1.02 [0.93, 1.13]. Held-out active IIA on v's frequent pairs 0.63 [0.55, 0.70] (v 0.78 [0.72, 0.84]; difference -0.16 [-0.23, -0.09]). Median |cos| with v 0.803.
- **Site 8, U: transfers.** D_bad(by) on Head 1.12 [0.99, 1.25] (v 1.08 [0.96, 1.20]; ratio 1.03 [0.91, 1.16]; D_rare − 0.5·D_v 0.57 [0.46, 0.69]); all bands ratio 1.03 [0.93, 1.13]. Held-out active IIA on v's frequent pairs 0.62 [0.54, 0.70] (v 0.78 [0.72, 0.84]; difference -0.16 [-0.23, -0.09]). Median |cos| with v 0.804.
- **Site 8, B: transfers.** D_bad(by) on Head 1.10 [0.96, 1.23] (v 1.08 [0.96, 1.20]; ratio 1.01 [0.89, 1.15]; D_rare − 0.5·D_v 0.55 [0.43, 0.68]); all bands ratio 1.00 [0.90, 1.10]. Held-out active IIA on v's frequent pairs 0.59 [0.51, 0.67] (v 0.78 [0.72, 0.84]; difference -0.19 [-0.27, -0.12]). Median |cos| with v 0.790.
- **Site 12, F: transfers.** D_bad(by) on Head 0.99 [0.88, 1.10] (v 1.05 [0.95, 1.15]; ratio 0.94 [0.85, 1.04]; D_rare − 0.5·D_v 0.46 [0.37, 0.56]); all bands ratio 0.92 [0.85, 0.98]. Held-out active IIA on v's frequent pairs 0.81 [0.74, 0.87] (v 0.90 [0.85, 0.94]; difference -0.09 [-0.14, -0.04]). Median |cos| with v 0.828.
- **Site 12, U: transfers.** D_bad(by) on Head 0.99 [0.88, 1.10] (v 1.05 [0.95, 1.15]; ratio 0.94 [0.86, 1.03]; D_rare − 0.5·D_v 0.46 [0.38, 0.56]); all bands ratio 0.91 [0.84, 0.97]. Held-out active IIA on v's frequent pairs 0.82 [0.75, 0.88] (v 0.90 [0.85, 0.94]; difference -0.08 [-0.13, -0.04]). Median |cos| with v 0.830.
- **Site 12, B: transfers.** D_bad(by) on Head 0.97 [0.85, 1.08] (v 1.05 [0.96, 1.15]; ratio 0.92 [0.82, 1.02]; D_rare − 0.5·D_v 0.44 [0.34, 0.54]); all bands ratio 0.91 [0.84, 0.97]. Held-out active IIA on v's frequent pairs 0.80 [0.73, 0.86] (v 0.90 [0.85, 0.94]; difference -0.10 [-0.15, -0.05]). Median |cos| with v 0.823.
- **Site 17, F: descriptive (signed retention).** D_bad(by) on Head -0.71 [-0.91, -0.53] (v -0.90 [-1.10, -0.71]; ratio 0.79 [0.63, 0.98]; D_rare − 0.5·D_v -0.26 [-0.42, -0.12]); all bands ratio 0.98 [0.85, 1.11]. Held-out active IIA on v's frequent pairs 0.96 [0.94, 0.98] (v 0.97 [0.96, 0.99]; difference -0.01 [-0.03, 0.00]). Median |cos| with v 0.830.
- **Site 17, U: descriptive (signed retention).** D_bad(by) on Head -0.74 [-0.93, -0.57] (v -0.90 [-1.12, -0.70]; ratio 0.83 [0.68, 1.01]; D_rare − 0.5·D_v -0.29 [-0.44, -0.17]); all bands ratio 1.05 [0.93, 1.17]. Held-out active IIA on v's frequent pairs 0.96 [0.94, 0.98] (v 0.97 [0.96, 0.99]; difference -0.01 [-0.03, 0.00]). Median |cos| with v 0.828.
- **Site 17, B: descriptive (signed retention).** D_bad(by) on Head -0.55 [-0.71, -0.39] (v -0.90 [-1.10, -0.71]; ratio 0.61 [0.48, 0.74]; D_rare − 0.5·D_v -0.10 [-0.22, 0.02]); all bands ratio 0.83 [0.73, 0.94]. Held-out active IIA on v's frequent pairs 0.95 [0.92, 0.97] (v 0.97 [0.96, 0.99]; difference -0.02 [-0.04, -0.01]). Median |cos| with v 0.811.

| site | cond | band | pairs | median \|cos\| | D_bad(by) rare | D_bad(by) v | ratio |
|---:|---|---|---:|---|---|---|---|
| 8 | F | head | 9 | 0.803 | 1.09 [0.96, 1.22] | 1.08 [0.96, 1.20] | 1.00 [0.89, 1.14] |
| 8 | F | tail | 28 | 0.803 | 0.73 [0.61, 0.86] | 0.68 [0.56, 0.81] | 1.07 [0.95, 1.20] |
| 8 | F | xtail | 25 | 0.803 | 0.59 [0.46, 0.72] | 0.60 [0.47, 0.72] | 0.98 [0.87, 1.09] |
| 8 | F | all | 62 | 0.803 | 0.72 [0.63, 0.82] | 0.70 [0.62, 0.80] | 1.02 [0.93, 1.13] |
| 8 | U | head | 9 | 0.804 | 1.12 [0.99, 1.25] | 1.08 [0.96, 1.20] | 1.03 [0.91, 1.16] |
| 8 | U | tail | 28 | 0.804 | 0.73 [0.61, 0.85] | 0.68 [0.55, 0.81] | 1.08 [0.96, 1.21] |
| 8 | U | xtail | 25 | 0.804 | 0.58 [0.46, 0.71] | 0.60 [0.48, 0.73] | 0.97 [0.87, 1.07] |
| 8 | U | all | 62 | 0.804 | 0.73 [0.63, 0.83] | 0.70 [0.62, 0.80] | 1.03 [0.93, 1.13] |
| 8 | B | head | 9 | 0.790 | 1.10 [0.96, 1.23] | 1.08 [0.96, 1.20] | 1.01 [0.89, 1.15] |
| 8 | B | tail | 28 | 0.790 | 0.72 [0.60, 0.84] | 0.68 [0.54, 0.81] | 1.06 [0.94, 1.20] |
| 8 | B | xtail | 26 | 0.790 | 0.56 [0.45, 0.69] | 0.62 [0.50, 0.74] | 0.91 [0.81, 1.02] |
| 8 | B | all | 63 | 0.790 | 0.71 [0.61, 0.81] | 0.71 [0.62, 0.81] | 1.00 [0.90, 1.10] |
| 12 | F | head | 9 | 0.828 | 0.99 [0.88, 1.10] | 1.05 [0.95, 1.15] | 0.94 [0.85, 1.04] |
| 12 | F | tail | 28 | 0.828 | 0.75 [0.64, 0.86] | 0.79 [0.69, 0.91] | 0.94 [0.87, 1.01] |
| 12 | F | xtail | 25 | 0.828 | 0.61 [0.47, 0.75] | 0.70 [0.57, 0.83] | 0.87 [0.79, 0.95] |
| 12 | F | all | 62 | 0.828 | 0.73 [0.64, 0.82] | 0.79 [0.71, 0.88] | 0.92 [0.85, 0.98] |
| 12 | U | head | 9 | 0.830 | 0.99 [0.88, 1.10] | 1.05 [0.95, 1.15] | 0.94 [0.86, 1.03] |
| 12 | U | tail | 28 | 0.830 | 0.74 [0.64, 0.84] | 0.79 [0.69, 0.90] | 0.93 [0.87, 1.00] |
| 12 | U | xtail | 25 | 0.830 | 0.60 [0.47, 0.74] | 0.70 [0.58, 0.83] | 0.85 [0.78, 0.93] |
| 12 | U | all | 62 | 0.830 | 0.72 [0.63, 0.81] | 0.79 [0.71, 0.88] | 0.91 [0.84, 0.97] |
| 12 | B | head | 9 | 0.823 | 0.97 [0.85, 1.08] | 1.05 [0.96, 1.15] | 0.92 [0.82, 1.02] |
| 12 | B | tail | 28 | 0.823 | 0.74 [0.64, 0.85] | 0.79 [0.68, 0.91] | 0.94 [0.86, 1.01] |
| 12 | B | xtail | 26 | 0.823 | 0.62 [0.49, 0.75] | 0.71 [0.59, 0.83] | 0.87 [0.79, 0.94] |
| 12 | B | all | 63 | 0.823 | 0.72 [0.63, 0.82] | 0.80 [0.71, 0.88] | 0.91 [0.84, 0.97] |
| 17 | F | head | 9 | 0.830 | -0.71 [-0.91, -0.53] | -0.90 [-1.10, -0.71] | 0.79 [0.63, 0.98] |
| 17 | F | tail | 28 | 0.830 | -0.89 [-1.12, -0.66] | -0.94 [-1.16, -0.74] | 0.94 [0.81, 1.08] |
| 17 | F | xtail | 25 | 0.830 | -1.01 [-1.24, -0.79] | -0.92 [-1.12, -0.74] | 1.09 [0.96, 1.25] |
| 17 | F | all | 62 | 0.830 | -0.91 [-1.12, -0.73] | -0.93 [-1.10, -0.77] | 0.98 [0.85, 1.11] |
| 17 | U | head | 9 | 0.828 | -0.74 [-0.93, -0.57] | -0.90 [-1.12, -0.70] | 0.83 [0.68, 1.01] |
| 17 | U | tail | 28 | 0.828 | -0.95 [-1.19, -0.72] | -0.94 [-1.16, -0.74] | 1.00 [0.88, 1.13] |
| 17 | U | xtail | 25 | 0.828 | -1.10 [-1.33, -0.87] | -0.92 [-1.11, -0.75] | 1.19 [1.05, 1.33] |
| 17 | U | all | 62 | 0.828 | -0.98 [-1.18, -0.78] | -0.93 [-1.10, -0.77] | 1.05 [0.93, 1.17] |
| 17 | B | head | 9 | 0.811 | -0.55 [-0.71, -0.39] | -0.90 [-1.10, -0.71] | 0.61 [0.48, 0.74] |
| 17 | B | tail | 28 | 0.811 | -0.72 [-0.91, -0.54] | -0.94 [-1.16, -0.76] | 0.77 [0.66, 0.88] |
| 17 | B | xtail | 26 | 0.811 | -0.91 [-1.11, -0.72] | -0.93 [-1.11, -0.75] | 0.98 [0.86, 1.10] |
| 17 | B | all | 63 | 0.811 | -0.78 [-0.93, -0.61] | -0.93 [-1.10, -0.77] | 0.83 [0.73, 0.94] |

Per seed (point estimates, plan rows of split k): `results/pythia/das_reverse_freq/per_seed.csv`.
Forward direction (v on Tail / XTail, R2-ptest) is the `D_bad(by) v` column at those bands.

## Checks

```
{
 "runner_settings": {
  "model_type": "gpt_neox",
  "bos": false,
  "fp32_unembed": false,
  "site_check_max_abs_sites_0_to_L-1": 0.0,
  "site_check_last_site": 77.1875
 },
 "n_rare_pairs": 43,
 "n_kept_F": 42,
 "dropped_F": [
  "tweeze/teethe"
 ],
 "failing_verbs": [
  {
   "verb": "teethe",
   "cls": 0,
   "n_correct": 3
  }
 ],
 "n_both_rare": 28,
 "n_kept_B": 27,
 "stored_recompute_site8": {
  "max": 0.08890914916992188,
  "median": 0.0,
  "median_bar": 0.001,
  "pass": true
 },
 "stored_recompute_site12": {
  "max": 0.10953044891357377,
  "median": 0.0,
  "median_bar": 0.001,
  "pass": true
 },
 "stored_recompute_site17": {
  "max": 0.04273700714111328,
  "median": 0.0,
  "median_bar": 0.006559610366821456,
  "pass": true
 }
}
```
