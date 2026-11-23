# DAS stability, extended: Gemma-3-1B

Rules: `reports/das_stability_ext_rules.md` (with its Nov 23 amendment; recompute bar: Amendment 2 of `reports/pythia/das_expanded_stability_rules.md`). Scripts: `scripts/run_das_stability_ext.py`, `scripts/run_passive_test.py`, `scripts/analyze_das_stability_ext.py`.

**Valid.** Function (P vs stored, clean population): unresolved, unresolved, equivalent, equivalent. Direction (P vs the seed range): below seed range, below seed range, below seed range, below seed range.

Clean population: primary (plain) minus 5 pairs sharing a verb with P's training pairs (new/tail/pillage/waver, new/tail/raze/relent, new/tail/requisition/flinch, new/xtail/besmirch/reemerge, new/xtail/cajole/saunter).

- **Site 5: function unresolved, direction below seed range** (item-14 label partly stable). Median |cos| with v: P 0.724 (P seeds 0.724–0.726); stored-pair seeds 0.850–0.891; L (stored pairs, P's update count) 0.946; R 1.000. D_bad(by) ratio P / stored: clean 1.19 [1.07, 1.34], full 1.21 [1.09, 1.36]; seed range (clean) 0.87–1.08.
- **Site 8: function unresolved, direction below seed range** (item-14 label partly stable). Median |cos| with v: P 0.898 (P seeds 0.898–0.898); stored-pair seeds 0.993–0.994; L (stored pairs, P's update count) 0.989; R 1.000. D_bad(by) ratio P / stored: clean 1.18 [1.11, 1.26], full 1.19 [1.12, 1.27]; seed range (clean) 1.00–1.02.
- **Site 12: function equivalent, direction below seed range** (item-14 label partly stable). Median |cos| with v: P 0.897 (P seeds 0.896–0.897); stored-pair seeds 0.990–0.991; L (stored pairs, P's update count) 0.979; R 1.000. D_bad(by) ratio P / stored: clean 1.05 [1.00, 1.11], full 1.06 [1.00, 1.11]; seed range (clean) 0.99–1.01.
- **Site 15: function equivalent, direction below seed range** (item-14 label partly stable). Median |cos| with v: P 0.848 (P seeds 0.842–0.848); stored-pair seeds 0.952–0.957; L (stored pairs, P's update count) 0.952; R 1.000. D_bad(by) ratio P / stored: clean 1.03 [0.97, 1.09], full 1.03 [0.97, 1.09]; seed range (clean) 0.98–1.01.

| site | condition | median \|cos\| | min \|cos\| | held-out IIA (cross) | median pc frac | D_bad(by), clean | ratio, clean | ratio, full |
|---:|---|---|---|---|---|---|---|---|
| 5 | stored | nan | nan | 0.851 | 1.09 | 0.57 [0.45, 0.70] | — | — |
| 5 | R | 1.000 | 1.000 | 0.851 | 1.09 | — | — | — |
| 5 | S1 | 0.891 | 0.798 | 0.843 | 1.10 | 0.51 [0.38, 0.65] | 0.89 [0.80, 0.97] | 0.90 [0.81, 0.97] |
| 5 | S2 | 0.862 | 0.784 | 0.862 | 1.11 | 0.50 [0.37, 0.63] | 0.87 [0.77, 0.97] | 0.87 [0.75, 0.97] |
| 5 | S3 | 0.853 | 0.094 | 0.810 | 1.01 | 0.58 [0.46, 0.72] | 1.01 [0.89, 1.12] | 1.01 [0.89, 1.13] |
| 5 | S4 | 0.863 | 0.749 | 0.858 | 1.11 | 0.61 [0.49, 0.74] | 1.08 [1.02, 1.14] | 1.08 [1.03, 1.14] |
| 5 | S5 | 0.850 | 0.616 | 0.845 | 1.08 | 0.57 [0.46, 0.70] | 1.00 [0.95, 1.07] | 1.00 [0.95, 1.06] |
| 5 | P | 0.724 | 0.607 | 0.940 | 1.17 | 0.68 [0.54, 0.83] | 1.19 [1.07, 1.34] | 1.21 [1.09, 1.36] |
| 5 | P2 | 0.726 | 0.646 | 0.941 | 1.19 | — | — | — |
| 5 | P3 | 0.724 | 0.645 | 0.931 | 1.16 | — | — | — |
| 5 | L | 0.946 | 0.934 | 0.867 | 1.25 | — | — | — |
| 8 | stored | nan | nan | 0.875 | 0.90 | 0.63 [0.53, 0.73] | — | — |
| 8 | R | 1.000 | 1.000 | 0.875 | 0.90 | — | — | — |
| 8 | S1 | 0.994 | 0.989 | 0.875 | 0.90 | 0.63 [0.53, 0.74] | 1.01 [1.00, 1.02] | 1.01 [1.00, 1.02] |
| 8 | S2 | 0.994 | 0.990 | 0.876 | 0.90 | 0.64 [0.54, 0.75] | 1.02 [1.01, 1.03] | 1.02 [1.01, 1.03] |
| 8 | S3 | 0.994 | 0.991 | 0.874 | 0.90 | 0.63 [0.53, 0.74] | 1.01 [1.00, 1.02] | 1.01 [1.00, 1.02] |
| 8 | S4 | 0.993 | 0.992 | 0.872 | 0.90 | 0.63 [0.53, 0.73] | 1.00 [0.99, 1.01] | 1.00 [0.99, 1.01] |
| 8 | S5 | 0.993 | 0.991 | 0.875 | 0.90 | 0.63 [0.53, 0.74] | 1.01 [0.99, 1.02] | 1.01 [1.00, 1.02] |
| 8 | P | 0.898 | 0.882 | 0.922 | 0.98 | 0.74 [0.63, 0.86] | 1.18 [1.11, 1.26] | 1.19 [1.12, 1.27] |
| 8 | P2 | 0.898 | 0.882 | 0.922 | 0.97 | — | — | — |
| 8 | P3 | 0.898 | 0.879 | 0.921 | 0.97 | — | — | — |
| 8 | L | 0.989 | 0.984 | 0.879 | 0.90 | — | — | — |
| 12 | stored | nan | nan | 0.944 | 1.03 | 0.96 [0.84, 1.09] | — | — |
| 12 | R | 1.000 | 1.000 | 0.944 | 1.03 | — | — | — |
| 12 | S1 | 0.990 | 0.984 | 0.945 | 1.03 | 0.96 [0.84, 1.09] | 1.00 [0.99, 1.01] | 1.00 [0.99, 1.01] |
| 12 | S2 | 0.991 | 0.987 | 0.945 | 1.04 | 0.97 [0.85, 1.10] | 1.01 [1.00, 1.02] | 1.01 [1.00, 1.02] |
| 12 | S3 | 0.991 | 0.986 | 0.946 | 1.03 | 0.95 [0.84, 1.08] | 0.99 [0.98, 1.00] | 0.99 [0.98, 1.00] |
| 12 | S4 | 0.991 | 0.987 | 0.945 | 1.03 | 0.95 [0.84, 1.08] | 0.99 [0.98, 1.00] | 0.99 [0.98, 1.00] |
| 12 | S5 | 0.990 | 0.987 | 0.947 | 1.04 | 0.96 [0.84, 1.09] | 1.00 [0.99, 1.01] | 1.00 [0.99, 1.01] |
| 12 | P | 0.897 | 0.872 | 0.961 | 1.09 | 1.01 [0.89, 1.14] | 1.05 [1.00, 1.11] | 1.06 [1.00, 1.11] |
| 12 | P2 | 0.896 | 0.879 | 0.962 | 1.09 | — | — | — |
| 12 | P3 | 0.897 | 0.867 | 0.958 | 1.09 | — | — | — |
| 12 | L | 0.979 | 0.964 | 0.950 | 1.06 | — | — | — |
| 15 | stored | nan | nan | 0.979 | 1.18 | 0.88 [0.78, 0.98] | — | — |
| 15 | R | 1.000 | 1.000 | 0.979 | 1.18 | — | — | — |
| 15 | S1 | 0.954 | 0.923 | 0.981 | 1.18 | 0.87 [0.77, 0.97] | 0.98 [0.97, 1.00] | 0.99 [0.97, 1.00] |
| 15 | S2 | 0.955 | 0.912 | 0.979 | 1.17 | 0.87 [0.77, 0.97] | 0.99 [0.97, 1.00] | 0.99 [0.97, 1.00] |
| 15 | S3 | 0.952 | 0.920 | 0.981 | 1.18 | 0.89 [0.79, 1.00] | 1.01 [1.00, 1.03] | 1.01 [1.00, 1.03] |
| 15 | S4 | 0.957 | 0.916 | 0.982 | 1.18 | 0.89 [0.79, 0.99] | 1.01 [1.00, 1.02] | 1.01 [1.00, 1.02] |
| 15 | S5 | 0.953 | 0.923 | 0.978 | 1.17 | 0.87 [0.78, 0.98] | 0.99 [0.98, 1.01] | 0.99 [0.98, 1.01] |
| 15 | P | 0.848 | 0.807 | 0.985 | 1.30 | 0.91 [0.81, 1.02] | 1.03 [0.97, 1.09] | 1.03 [0.97, 1.09] |
| 15 | P2 | 0.842 | 0.818 | 0.984 | 1.29 | — | — | — |
| 15 | P3 | 0.846 | 0.815 | 0.988 | 1.30 | — | — | — |
| 15 | L | 0.952 | 0.940 | 0.984 | 1.27 | — | — | — |

## Checks

```
{
 "runner_settings": {
  "model_type": "gemma3_text",
  "bos": true,
  "fp32_unembed": true,
  "site_check_max_abs_sites_0_to_L-1": 0.0,
  "site_check_last_site": 57855.87109375
 },
 "dropped_stored": [
  "transplant/ensue"
 ],
 "dropped_P_new": [
  "bewail/mew",
  "certify/retreat",
  "counsel/reverberate",
  "outmatch/cohere",
  "outrace/gab"
 ],
 "n_new_pairs_kept_P": 57,
 "stored_items_equal_site5": true,
 "stored_recompute_site5": {
  "max": 8.881784197001252e-16,
  "median": 0.0,
  "median_bar": 0.001,
  "pass": true
 },
 "stored_items_equal_site8": true,
 "stored_recompute_site8": {
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
 "stored_items_equal_site15": true,
 "stored_recompute_site15": {
  "max": 8.881784197001252e-16,
  "median": 0.0,
  "median_bar": 0.001,
  "pass": true
 },
 "R_median_abs_cos_site5": 1.0,
 "R_median_abs_cos_site8": 0.9999998807907104,
 "R_median_abs_cos_site12": 1.0000001192092896,
 "R_median_abs_cos_site15": 1.0000001192092896
}
```
