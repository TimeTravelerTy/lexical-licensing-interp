# DAS stability, extended: OLMo-2-1B

Rules: `reports/das_stability_ext_rules.md` (with its Nov 23 amendment; recompute bar: Amendment 2 of `reports/pythia/das_expanded_stability_rules.md`). Scripts: `scripts/run_das_stability_ext.py`, `scripts/run_passive_test.py`, `scripts/analyze_das_stability_ext.py`.

**Valid.** Function (P vs stored, clean population): equivalent, larger, equivalent, larger. Direction (P vs the seed range): below seed range, below seed range, below seed range, below seed range.

Clean population: primary (plain) minus 5 pairs sharing a verb with P's training pairs (new/tail/pillage/waver, new/tail/raze/relent, new/tail/requisition/flinch, new/xtail/besmirch/reemerge, new/xtail/cajole/saunter).

- **Site 9: function equivalent, direction below seed range** (item-14 label partly stable). Median |cos| with v: P 0.921 (P seeds 0.919–0.922); stored-pair seeds 0.993–0.994; L (stored pairs, P's update count) 0.993; R 1.000. D_bad(by) ratio P / stored: clean 1.16 [1.10, 1.23], full 1.16 [1.11, 1.24]; seed range (clean) 1.00–1.02.
- **Site 11: function larger, direction below seed range** (item-14 label changed). Median |cos| with v: P 0.923 (P seeds 0.921–0.923); stored-pair seeds 0.994–0.994; L (stored pairs, P's update count) 0.990; R 1.000. D_bad(by) ratio P / stored: clean 1.82 [1.48, 2.65], full 1.86 [1.53, 2.73]; seed range (clean) 0.98–1.06.
- **Site 12: function equivalent, direction below seed range** (item-14 label partly stable). Median |cos| with v: P 0.908 (P seeds 0.904–0.908); stored-pair seeds 0.994–0.994; L (stored pairs, P's update count) 0.979; R 1.000. D_bad(by) ratio P / stored: clean 1.02 [0.92, 1.12], full 1.02 [0.93, 1.12]; seed range (clean) 0.96–0.99.
- **Site 13: function larger, direction below seed range** (item-14 label changed). Median |cos| with v: P 0.830 (P seeds 0.818–0.830); stored-pair seeds 0.992–0.993; L (stored pairs, P's update count) 0.947; R 1.000. D_bad(by) ratio P / stored: clean 1.57 [1.47, 1.69], full 1.58 [1.48, 1.71]; seed range (clean) 0.95–1.00.

| site | condition | median \|cos\| | min \|cos\| | held-out IIA (cross) | median pc frac | D_bad(by), clean | ratio, clean | ratio, full |
|---:|---|---|---|---|---|---|---|---|
| 9 | stored | nan | nan | 0.807 | 0.80 | 0.44 [0.38, 0.50] | — | — |
| 9 | R | 1.000 | 1.000 | 0.807 | 0.80 | — | — | — |
| 9 | S1 | 0.993 | 0.989 | 0.808 | 0.80 | 0.44 [0.38, 0.50] | 1.00 [0.99, 1.01] | 1.00 [0.99, 1.01] |
| 9 | S2 | 0.994 | 0.991 | 0.810 | 0.79 | 0.44 [0.38, 0.51] | 1.02 [1.01, 1.02] | 1.02 [1.01, 1.03] |
| 9 | S3 | 0.994 | 0.991 | 0.807 | 0.79 | 0.44 [0.38, 0.51] | 1.00 [1.00, 1.01] | 1.00 [1.00, 1.01] |
| 9 | S4 | 0.994 | 0.991 | 0.806 | 0.80 | 0.44 [0.38, 0.51] | 1.00 [1.00, 1.01] | 1.00 [1.00, 1.01] |
| 9 | S5 | 0.993 | 0.991 | 0.810 | 0.80 | 0.45 [0.38, 0.51] | 1.02 [1.01, 1.03] | 1.02 [1.01, 1.03] |
| 9 | P | 0.921 | 0.909 | 0.831 | 0.83 | 0.51 [0.45, 0.57] | 1.16 [1.10, 1.23] | 1.16 [1.11, 1.24] |
| 9 | P2 | 0.922 | 0.913 | 0.837 | 0.83 | — | — | — |
| 9 | P3 | 0.919 | 0.915 | 0.838 | 0.84 | — | — | — |
| 9 | L | 0.993 | 0.991 | 0.814 | 0.80 | — | — | — |
| 11 | stored | nan | nan | 0.926 | 0.94 | 0.15 [0.08, 0.22] | — | — |
| 11 | R | 1.000 | 1.000 | 0.926 | 0.94 | — | — | — |
| 11 | S1 | 0.994 | 0.991 | 0.929 | 0.93 | 0.15 [0.09, 0.22] | 1.05 [1.00, 1.12] | 1.05 [1.00, 1.13] |
| 11 | S2 | 0.994 | 0.976 | 0.928 | 0.94 | 0.15 [0.09, 0.22] | 1.03 [0.98, 1.11] | 1.03 [0.98, 1.11] |
| 11 | S3 | 0.994 | 0.991 | 0.924 | 0.93 | 0.14 [0.08, 0.22] | 0.98 [0.94, 1.02] | 0.99 [0.94, 1.02] |
| 11 | S4 | 0.994 | 0.992 | 0.930 | 0.94 | 0.15 [0.08, 0.22] | 1.00 [0.95, 1.05] | 1.00 [0.95, 1.06] |
| 11 | S5 | 0.994 | 0.991 | 0.929 | 0.93 | 0.16 [0.09, 0.23] | 1.06 [1.01, 1.16] | 1.06 [1.01, 1.15] |
| 11 | P | 0.923 | 0.909 | 0.941 | 0.99 | 0.27 [0.21, 0.33] | 1.82 [1.48, 2.65] | 1.86 [1.53, 2.73] |
| 11 | P2 | 0.922 | 0.917 | 0.946 | 0.99 | — | — | — |
| 11 | P3 | 0.921 | 0.915 | 0.945 | 0.99 | — | — | — |
| 11 | L | 0.990 | 0.982 | 0.934 | 0.95 | — | — | — |
| 12 | stored | nan | nan | 0.970 | 1.06 | -0.74 [-0.88, -0.60] | — | — |
| 12 | R | 1.000 | 1.000 | 0.970 | 1.06 | — | — | — |
| 12 | S1 | 0.994 | 0.991 | 0.966 | 1.05 | -0.71 [-0.85, -0.57] | 0.96 [0.94, 0.98] | 0.96 [0.94, 0.98] |
| 12 | S2 | 0.994 | 0.993 | 0.970 | 1.05 | -0.72 [-0.86, -0.58] | 0.97 [0.95, 0.99] | 0.97 [0.95, 0.99] |
| 12 | S3 | 0.994 | 0.991 | 0.967 | 1.05 | -0.73 [-0.87, -0.59] | 0.99 [0.97, 1.02] | 0.99 [0.96, 1.02] |
| 12 | S4 | 0.994 | 0.993 | 0.969 | 1.05 | -0.73 [-0.87, -0.59] | 0.99 [0.97, 1.01] | 0.99 [0.97, 1.01] |
| 12 | S5 | 0.994 | 0.992 | 0.970 | 1.05 | -0.73 [-0.87, -0.59] | 0.99 [0.97, 1.01] | 0.99 [0.97, 1.01] |
| 12 | P | 0.908 | 0.884 | 0.973 | 1.13 | -0.75 [-0.90, -0.60] | 1.02 [0.92, 1.12] | 1.02 [0.93, 1.12] |
| 12 | P2 | 0.907 | 0.900 | 0.977 | 1.15 | — | — | — |
| 12 | P3 | 0.904 | 0.895 | 0.979 | 1.14 | — | — | — |
| 12 | L | 0.979 | 0.975 | 0.978 | 1.10 | — | — | — |
| 13 | stored | nan | nan | 0.983 | 1.12 | -1.10 [-1.26, -0.93] | — | — |
| 13 | R | 1.000 | 1.000 | 0.983 | 1.12 | — | — | — |
| 13 | S1 | 0.992 | 0.990 | 0.984 | 1.11 | -1.04 [-1.20, -0.88] | 0.95 [0.92, 0.97] | 0.94 [0.92, 0.97] |
| 13 | S2 | 0.993 | 0.989 | 0.984 | 1.11 | -1.06 [-1.22, -0.90] | 0.97 [0.95, 0.99] | 0.97 [0.95, 0.99] |
| 13 | S3 | 0.992 | 0.991 | 0.983 | 1.12 | -1.09 [-1.26, -0.93] | 1.00 [0.98, 1.02] | 1.00 [0.98, 1.02] |
| 13 | S4 | 0.992 | 0.989 | 0.982 | 1.11 | -1.09 [-1.26, -0.93] | 1.00 [0.98, 1.01] | 1.00 [0.98, 1.01] |
| 13 | S5 | 0.992 | 0.990 | 0.982 | 1.11 | -1.09 [-1.25, -0.93] | 0.99 [0.97, 1.01] | 0.99 [0.97, 1.01] |
| 13 | P | 0.830 | 0.809 | 0.990 | 1.28 | -1.72 [-1.93, -1.50] | 1.57 [1.47, 1.69] | 1.58 [1.48, 1.71] |
| 13 | P2 | 0.828 | 0.809 | 0.992 | 1.31 | — | — | — |
| 13 | P3 | 0.818 | 0.807 | 0.991 | 1.30 | — | — | — |
| 13 | L | 0.947 | 0.936 | 0.990 | 1.21 | — | — | — |

## Checks

```
{
 "runner_settings": {
  "model_type": "olmo2",
  "bos": false,
  "fp32_unembed": true,
  "site_check_max_abs_sites_0_to_L-1": 0.0,
  "site_check_last_site": 45.875
 },
 "dropped_stored": [],
 "dropped_P_new": [
  "counsel/reverberate",
  "sod/patter"
 ],
 "n_new_pairs_kept_P": 60,
 "stored_items_equal_site9": true,
 "stored_recompute_site9": {
  "max": 8.881784197001252e-16,
  "median": 0.0,
  "median_bar": 0.001,
  "pass": true
 },
 "stored_items_equal_site11": true,
 "stored_recompute_site11": {
  "max": 8.881784197001252e-16,
  "median": 0.0,
  "median_bar": 0.001,
  "pass": true
 },
 "stored_items_equal_site12": true,
 "stored_recompute_site12": {
  "max": 8.881784197001252e-16,
  "median": 0.0,
  "median_bar": 0.001,
  "pass": true
 },
 "stored_items_equal_site13": true,
 "stored_recompute_site13": {
  "max": 8.881784197001252e-16,
  "median": 0.0,
  "median_bar": 0.001,
  "pass": true
 },
 "R_median_abs_cos_site9": 1.0000001192092896,
 "R_median_abs_cos_site11": 1.0,
 "R_median_abs_cos_site12": 1.0,
 "R_median_abs_cos_site13": 1.0000001192092896
}
```
