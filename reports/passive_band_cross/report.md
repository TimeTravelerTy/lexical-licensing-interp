# Verb band x context band passive cross

107,352 scored pairs. Released contexts: 100 sampled per band and paradigm, each crossed with all 26/50/50 verb pairs. Intervals: two-way cluster bootstrap (verb pairs within verb band, contexts within context band).

## `passive_1`

**Accuracy (%)**: rows = verb band, columns = context band

| Verb \ Context | head | tail | xtail |
|---|---:|---:|---:|
| head | 84.5 [75.8, 91.1] | 84.3 [76.4, 90.4] | 84.5 [75.5, 91.6] |
| tail | 79.2 [74.1, 84.3] | 81.3 [76.6, 85.6] | 82.2 [77.6, 86.4] |
| xtail | 71.4 [65.9, 76.9] | 70.7 [65.5, 75.8] | 72.2 [66.7, 77.8] |

**Mean `whole_margin`**: rows = verb band, columns = context band

| Verb \ Context | head | tail | xtail |
|---|---:|---:|---:|
| head | 3.15 [2.34, 3.82] | 3.17 [2.39, 3.85] | 3.32 [2.47, 4.02] |
| tail | 2.79 [2.23, 3.37] | 3.02 [2.45, 3.57] | 3.14 [2.59, 3.70] |
| xtail | 1.97 [1.45, 2.49] | 1.85 [1.34, 2.38] | 1.93 [1.43, 2.46] |

**Mean `verb_margin`**: rows = verb band, columns = context band

| Verb \ Context | head | tail | xtail |
|---|---:|---:|---:|
| head | 2.59 [2.02, 3.17] | 2.67 [2.13, 3.22] | 2.62 [2.04, 3.22] |
| tail | 2.06 [1.55, 2.58] | 2.16 [1.67, 2.68] | 2.27 [1.76, 2.79] |
| xtail | 1.53 [1.06, 2.01] | 1.45 [1.02, 1.93] | 1.57 [1.08, 2.06] |

**Contrasts (XTail - Head)**

| Contrast | Accuracy (pp) | Whole margin | Verb margin |
|---|---:|---:|---:|
| diagonal_xtail_minus_head | -12.3 [-21.0, -2.2] | -1.21 [-2.09, -0.27] | -1.03 [-1.79, -0.27] |
| verb_xtail_minus_head | -13.0 [-21.1, -3.5] | -1.30 [-2.11, -0.39] | -1.11 [-1.82, -0.41] |
| context_xtail_minus_head | 1.3 [-0.9, 3.4] | 0.16 [-0.04, 0.36] | 0.09 [-0.04, 0.24] |
| interaction_xtail_head | 0.8 [-3.7, 5.6] | -0.21 [-0.57, 0.17] | 0.01 [-0.30, 0.34] |

**Curated contexts: own vs other verbs' vs released (accuracy %)**

| Verb band | Own curated | Other curated | Released (all bands) | Own - other | Other - released |
|---|---:|---:|---:|---:|---:|
| head | 100.0 [100.0, 100.0] | 81.4 [72.9, 89.3] | 84.4 [76.6, 91.1] | 18.6 [10.7, 27.1] | -3.1 [-8.6, 1.4] |
| tail | 98.0 [94.0, 100.0] | 77.9 [72.0, 83.4] | 80.9 [76.4, 85.1] | 20.1 [13.7, 26.6] | -3.0 [-6.5, 0.6] |
| xtail | 98.0 [94.0, 100.0] | 73.3 [68.3, 78.6] | 71.4 [66.0, 76.4] | 24.7 [18.0, 30.8] | 1.9 [-3.3, 7.3] |

**Context-level regression, XTail-context coefficient on whole margin (all verbs)**

| Model | XTail vs Head context | R^2 |
|---|---:|---:|
| band_only | 0.16 [-0.00, 0.32] | 0.012 |
| band_plus_frame | 0.11 [-0.06, 0.26] | 0.104 |
| band_plus_zipf_plus_frame | 0.48 [-0.31, 1.22] | 0.110 |

## `passive_2`

**Accuracy (%)**: rows = verb band, columns = context band

| Verb \ Context | head | tail | xtail |
|---|---:|---:|---:|
| head | 85.7 [75.7, 92.9] | 84.9 [75.0, 92.4] | 83.8 [74.1, 91.5] |
| tail | 83.7 [78.7, 88.5] | 85.8 [81.1, 90.1] | 85.2 [80.4, 89.5] |
| xtail | 71.0 [64.3, 77.6] | 71.0 [63.7, 78.4] | 71.4 [64.5, 78.3] |

**Mean `whole_margin`**: rows = verb band, columns = context band

| Verb \ Context | head | tail | xtail |
|---|---:|---:|---:|
| head | 2.85 [2.00, 3.64] | 2.84 [2.01, 3.66] | 2.69 [1.88, 3.44] |
| tail | 2.80 [2.29, 3.37] | 3.08 [2.56, 3.66] | 2.90 [2.40, 3.47] |
| xtail | 1.69 [1.15, 2.24] | 1.73 [1.14, 2.33] | 1.66 [1.12, 2.22] |

**Mean `verb_margin`**: rows = verb band, columns = context band

| Verb \ Context | head | tail | xtail |
|---|---:|---:|---:|
| head | 2.68 [2.12, 3.29] | 2.67 [2.07, 3.28] | 2.47 [1.90, 3.05] |
| tail | 2.07 [1.61, 2.63] | 2.29 [1.81, 2.84] | 2.14 [1.68, 2.66] |
| xtail | 1.48 [1.00, 1.98] | 1.53 [1.01, 2.05] | 1.48 [1.03, 1.97] |

**Contrasts (XTail - Head)**

| Contrast | Accuracy (pp) | Whole margin | Verb margin |
|---|---:|---:|---:|
| diagonal_xtail_minus_head | -14.3 [-24.7, -2.0] | -1.19 [-2.18, -0.17] | -1.20 [-1.95, -0.44] |
| verb_xtail_minus_head | -13.6 [-24.0, -1.9] | -1.10 [-2.07, -0.09] | -1.11 [-1.85, -0.36] |
| context_xtail_minus_head | 0.0 [-1.9, 1.8] | -0.03 [-0.18, 0.12] | -0.05 [-0.18, 0.08] |
| interaction_xtail_head | 2.2 [-2.1, 6.5] | 0.14 [-0.13, 0.40] | 0.22 [-0.04, 0.50] |

**Curated contexts: own vs other verbs' vs released (accuracy %)**

| Verb band | Own curated | Other curated | Released (all bands) | Own - other | Other - released |
|---|---:|---:|---:|---:|---:|
| head | 100.0 [100.0, 100.0] | 82.8 [73.9, 90.3] | 84.8 [74.8, 92.6] | 17.2 [9.7, 26.1] | -2.0 [-5.8, 1.8] |
| tail | 98.0 [94.0, 100.0] | 81.6 [76.3, 86.8] | 84.9 [80.2, 89.3] | 16.4 [10.1, 22.5] | -3.3 [-7.0, 0.3] |
| xtail | 96.0 [90.0, 100.0] | 69.8 [62.9, 76.0] | 71.2 [64.1, 77.5] | 26.2 [17.8, 34.4] | -1.4 [-6.2, 3.6] |

**Context-level regression, XTail-context coefficient on whole margin (all verbs)**

| Model | XTail vs Head context | R^2 |
|---|---:|---:|
| band_only | -0.01 [-0.14, 0.12] | 0.018 |
| band_plus_frame | -0.04 [-0.15, 0.06] | 0.329 |
| band_plus_zipf_plus_frame | 0.25 [-0.04, 0.54] | 0.337 |

