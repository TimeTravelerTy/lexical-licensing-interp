# DAS stability, extended: Pythia-1.4B

Rules: `reports/das_stability_ext_rules.md` (with its Nov 23 amendment; recompute bar: Amendment 2 of `reports/pythia/das_expanded_stability_rules.md`). Scripts: `scripts/run_das_stability_ext.py`, `scripts/run_passive_test.py`, `scripts/analyze_das_stability_ext.py`.

**INVALID.** Function (P vs stored, clean population): equivalent, equivalent, unresolved. Direction (P vs the seed range): below seed range, below seed range, below seed range. Problems: passive checks for S2 at site 17: {'self_patch_max_abs_dM': 0.0, 'heldout_recompute_max_abs_dM': 0.6621379852294922, 'heldout_recompute_median_abs_dM': 0.0039033889770508368}

Clean population: primary (plain) minus 5 pairs sharing a verb with P's training pairs (new/tail/pillage/waver, new/tail/raze/relent, new/tail/requisition/flinch, new/xtail/besmirch/reemerge, new/xtail/cajole/saunter).

- **Site 8: function equivalent, direction below seed range** (item-14 label partly stable). Median |cos| with v: P 0.914 (P seeds 0.914–0.916); stored-pair seeds 0.996–0.996; L (stored pairs, P's update count) 0.995; R 1.000. D_bad(by) ratio P / stored: clean 1.11 [1.05, 1.18], full 1.10 [1.05, 1.17]; seed range (clean) 0.99–1.00.
- **Site 12: function equivalent, direction below seed range** (item-14 label partly stable). Median |cos| with v: P 0.922 (P seeds 0.922–0.923); stored-pair seeds 0.993–0.994; L (stored pairs, P's update count) 0.992; R 1.000. D_bad(by) ratio P / stored: clean 1.06 [1.02, 1.11], full 1.06 [1.02, 1.11]; seed range (clean) 0.99–1.00.
- **Site 17: function unresolved, direction below seed range** (item-14 label partly stable). Median |cos| with v: P 0.904 (P seeds 0.902–0.904); stored-pair seeds 0.936–0.938; L (stored pairs, P's update count) 0.958; R 0.999. D_bad(by) ratio P / stored: clean 1.31 [1.22, 1.41], full 1.32 [1.23, 1.41]; seed range (clean) 0.94–1.01.

| site | condition | median \|cos\| | min \|cos\| | held-out IIA (cross) | median pc frac | D_bad(by), clean | ratio, clean | ratio, full |
|---:|---|---|---|---|---|---|---|---|
| 8 | stored | nan | nan | 0.782 | 0.78 | 0.71 [0.61, 0.81] | — | — |
| 8 | R | 1.000 | 1.000 | 0.782 | 0.78 | — | — | — |
| 8 | S1 | 0.996 | 0.994 | 0.779 | 0.77 | 0.71 [0.61, 0.81] | 1.00 [0.99, 1.01] | 1.00 [0.99, 1.01] |
| 8 | S2 | 0.996 | 0.994 | 0.778 | 0.78 | 0.71 [0.61, 0.81] | 1.00 [0.99, 1.01] | 1.00 [0.99, 1.01] |
| 8 | S3 | 0.996 | 0.994 | 0.780 | 0.77 | 0.71 [0.61, 0.80] | 1.00 [0.99, 1.00] | 0.99 [0.99, 1.00] |
| 8 | S4 | 0.996 | 0.995 | 0.777 | 0.77 | 0.70 [0.61, 0.80] | 0.99 [0.99, 1.00] | 0.99 [0.99, 1.00] |
| 8 | S5 | 0.996 | 0.994 | 0.780 | 0.77 | 0.70 [0.61, 0.80] | 0.99 [0.99, 1.00] | 0.99 [0.99, 1.00] |
| 8 | P | 0.914 | 0.903 | 0.804 | 0.80 | 0.79 [0.69, 0.89] | 1.11 [1.05, 1.18] | 1.10 [1.05, 1.17] |
| 8 | P2 | 0.915 | 0.906 | 0.804 | 0.80 | — | — | — |
| 8 | P3 | 0.916 | 0.906 | 0.811 | 0.80 | — | — | — |
| 8 | L | 0.995 | 0.993 | 0.777 | 0.78 | — | — | — |
| 12 | stored | nan | nan | 0.898 | 0.87 | 0.79 [0.71, 0.88] | — | — |
| 12 | R | 1.000 | 0.999 | 0.900 | 0.87 | — | — | — |
| 12 | S1 | 0.994 | 0.991 | 0.900 | 0.87 | 0.78 [0.70, 0.87] | 0.99 [0.98, 1.00] | 0.99 [0.99, 1.00] |
| 12 | S2 | 0.994 | 0.991 | 0.899 | 0.87 | 0.79 [0.70, 0.87] | 0.99 [0.99, 1.00] | 0.99 [0.99, 1.00] |
| 12 | S3 | 0.994 | 0.990 | 0.902 | 0.87 | 0.79 [0.70, 0.87] | 0.99 [0.99, 1.00] | 0.99 [0.99, 1.00] |
| 12 | S4 | 0.993 | 0.990 | 0.899 | 0.87 | 0.78 [0.70, 0.87] | 0.99 [0.99, 1.00] | 0.99 [0.99, 1.00] |
| 12 | S5 | 0.993 | 0.991 | 0.900 | 0.87 | 0.79 [0.71, 0.88] | 1.00 [0.99, 1.01] | 1.00 [0.99, 1.01] |
| 12 | P | 0.922 | 0.915 | 0.911 | 0.92 | 0.84 [0.74, 0.94] | 1.06 [1.02, 1.11] | 1.06 [1.02, 1.11] |
| 12 | P2 | 0.923 | 0.913 | 0.917 | 0.92 | — | — | — |
| 12 | P3 | 0.923 | 0.905 | 0.919 | 0.92 | — | — | — |
| 12 | L | 0.992 | 0.989 | 0.898 | 0.88 | — | — | — |
| 17 | stored | nan | nan | 0.973 | 1.07 | -0.93 [-1.09, -0.77] | — | — |
| 17 | R | 0.999 | 0.998 | 0.973 | 1.06 | — | — | — |
| 17 | S1 | 0.937 | 0.930 | 0.975 | 1.06 | -0.87 [-1.04, -0.72] | 0.94 [0.92, 0.96] | 0.94 [0.91, 0.96] |
| 17 | S2 | 0.936 | 0.932 | 0.973 | 1.06 | -0.90 [-1.06, -0.74] | 0.97 [0.94, 0.99] | 0.97 [0.94, 0.99] |
| 17 | S3 | 0.936 | 0.931 | 0.974 | 1.06 | -0.93 [-1.10, -0.77] | 1.01 [0.97, 1.04] | 1.00 [0.97, 1.04] |
| 17 | S4 | 0.937 | 0.931 | 0.972 | 1.06 | -0.89 [-1.06, -0.72] | 0.96 [0.93, 0.99] | 0.96 [0.93, 0.99] |
| 17 | S5 | 0.938 | 0.932 | 0.974 | 1.06 | -0.91 [-1.08, -0.75] | 0.98 [0.95, 1.01] | 0.98 [0.95, 1.01] |
| 17 | P | 0.904 | 0.896 | 0.986 | 1.17 | -1.22 [-1.43, -1.01] | 1.31 [1.22, 1.41] | 1.32 [1.23, 1.41] |
| 17 | P2 | 0.903 | 0.896 | 0.985 | 1.17 | — | — | — |
| 17 | P3 | 0.902 | 0.892 | 0.985 | 1.17 | — | — | — |
| 17 | L | 0.958 | 0.954 | 0.982 | 1.13 | — | — | — |

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
 "dropped_stored": [],
 "dropped_P_new": [
  "tweeze/teethe"
 ],
 "n_new_pairs_kept_P": 61,
 "stored_items_equal_site8": true,
 "stored_recompute_site8": {
  "max": 0.08890914916992188,
  "median": 0.0,
  "median_bar": 0.001,
  "pass": true
 },
 "stored_items_equal_site12": true,
 "stored_recompute_site12": {
  "max": 0.10953044891357377,
  "median": 0.0,
  "median_bar": 0.001,
  "pass": true
 },
 "stored_items_equal_site17": true,
 "stored_recompute_site17": {
  "max": 0.04273700714111328,
  "median": 0.0,
  "median_bar": 0.006559610366821456,
  "pass": true
 },
 "R_median_abs_cos_site8": 0.9998922348022461,
 "R_median_abs_cos_site12": 0.9998115301132202,
 "R_median_abs_cos_site17": 0.9989653825759888
}
```
