# DAS stability under more training verbs (Pythia-1.4B)

Rules: `reports/pythia/das_expanded_stability_rules.md`. Scripts: `scripts/run_das_expanded_stability.py`, `scripts/run_passive_test.py`, `scripts/analyze_das_expanded_stability.py`.

**Overall: invalid.** Problems: passive-test checks failed for P at site 8: {'self_patch_max_abs_dM': 0.0, 'heldout_recompute_max_abs_dM': 0.3475656509399414, 'heldout_recompute_median_abs_dM': 5.551115123125783e-17}; passive-test checks failed for E at site 8: {'self_patch_max_abs_dM': 0.0, 'heldout_recompute_max_abs_dM': 0.22687339782714844, 'heldout_recompute_median_abs_dM': 6.661338147750939e-16}; passive-test checks failed for S at site 8: {'self_patch_max_abs_dM': 0.0, 'heldout_recompute_max_abs_dM': 0.38833045959472656, 'heldout_recompute_median_abs_dM': 4.440892098500626e-16}; stored held-out recompute at site 8; passive-test checks failed for P at site 12: {'self_patch_max_abs_dM': 0.0, 'heldout_recompute_max_abs_dM': 0.36397266387939453, 'heldout_recompute_median_abs_dM': 0.0}; passive-test checks failed for E at site 12: {'self_patch_max_abs_dM': 0.0, 'heldout_recompute_max_abs_dM': 0.35914611816406206, 'heldout_recompute_median_abs_dM': 0.0007848739624023438}; passive-test checks failed for S at site 12: {'self_patch_max_abs_dM': 0.0, 'heldout_recompute_max_abs_dM': 0.21761417388916016, 'heldout_recompute_median_abs_dM': 0.0006680488586425504}; stored held-out recompute at site 12

- **Site 8: partly stable** (P, primary; E: partly stable; reference: stored v). Median |cos| with v: E 0.887, P 0.914, R 1.000, S 0.996. D_bad(by): stored 0.72 [0.62, 0.82], P 0.79 [0.70, 0.89], E 0.79 [0.71, 0.89], S 0.72 [0.62, 0.81]. Ratio to stored: P 1.10 [1.05, 1.17], E 1.11 [1.03, 1.19], S 1.00 [0.99, 1.01].
- **Site 12: partly stable** (P, primary; E: partly stable; reference: stored v). Median |cos| with v: E 0.891, P 0.922, R 1.000, S 0.994. D_bad(by): stored 0.81 [0.72, 0.89], P 0.86 [0.76, 0.95], E 0.82 [0.73, 0.91], S 0.80 [0.71, 0.89]. Ratio to stored: P 1.06 [1.02, 1.11], E 1.02 [0.97, 1.07], S 0.99 [0.99, 1.00].

| site | condition | median \|cos\| | min \|cos\| | held-out IIA (cross) | median pc frac | D_bad(by) | ratio to stored |
|---:|---|---|---|---|---|---|---|
| 8 | stored | nan | nan | 0.782 | 0.78 | 0.72 [0.62, 0.82] | — |
| 8 | R | 1.000 | 1.000 | 0.782 | 0.78 | — | — |
| 8 | S | 0.996 | 0.994 | 0.779 | 0.77 | 0.72 [0.62, 0.81] | 1.00 [0.99, 1.01] |
| 8 | P | 0.914 | 0.903 | 0.804 | 0.80 | 0.79 [0.70, 0.89] | 1.10 [1.05, 1.17] |
| 8 | E | 0.887 | 0.872 | 0.775 | 0.75 | 0.79 [0.71, 0.89] | 1.11 [1.03, 1.19] |
| 12 | stored | nan | nan | 0.898 | 0.87 | 0.81 [0.72, 0.89] | — |
| 12 | R | 1.000 | 0.999 | 0.900 | 0.87 | — | — |
| 12 | S | 0.994 | 0.991 | 0.900 | 0.87 | 0.80 [0.71, 0.89] | 0.99 [0.99, 1.00] |
| 12 | P | 0.922 | 0.915 | 0.911 | 0.92 | 0.86 [0.76, 0.95] | 1.06 [1.02, 1.11] |
| 12 | E | 0.891 | 0.878 | 0.899 | 0.88 | 0.82 [0.73, 0.91] | 1.02 [0.97, 1.07] |

## Checks

```
{
 "dropped_stored": [],
 "dropped_E": [
  "arraign/intercede",
  "explicate/clam",
  "tweeze/teethe"
 ],
 "n_new_pairs_kept_E": 120,
 "dropped_P": [
  "tweeze/teethe"
 ],
 "n_new_pairs_kept_P": 61,
 "stored_items_equal_site8": true,
 "stored_heldout_recompute_max_abs_site8": 0.08890914916992188,
 "stored_items_equal_site12": true,
 "stored_heldout_recompute_max_abs_site12": 0.10953044891357377,
 "R_median_abs_cos_site8": 0.9998922348022461,
 "R_median_abs_cos_site12": 0.9998115301132202
}
```
