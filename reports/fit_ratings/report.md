# Fit ratings: gemma4_31b_it

## Validation

- **Format**: digit mass median 1.000; 0.0% of ratings below 0.9.

- **Prompt agreement** (Spearman, all items): 0.956

- **Direction checks**
  - own > mean of other contexts for 99.6% of verb pairs (mean own 6.93, other 3.21)
  - own > role reversal (passive_1) for 92.9% of verb pairs (mean difference 4.24); reversal is not always implausible

- **Rarity check**: mean rating by verb band (own and reversal should be flat)

| Paradigm | Verb band | Own | Other | Reversal | n |
|---|---|---:|---:|---:|---:|
| passive_1 | head | 6.77 | 3.82 | 2.53 | 26 |
| passive_1 | tail | 6.95 | 2.86 | 2.68 | 50 |
| passive_1 | xtail | 6.92 | 2.74 | 2.72 | 50 |
| passive_2 | head | 7.00 | 4.38 | - | 26 |
| passive_2 | tail | 7.00 | 3.30 | - | 50 |
| passive_2 | xtail | 6.92 | 3.03 | - | 50 |

- Spearman(own rating, participle Zipf): 0.182; Spearman(other rating, participle Zipf): 0.282

- **Second rater agreement**: Spearman 0.726 on 28971 items

## Pythia margin vs fit (curated cross)

`fit` is the rating minus 4 (per rating point); band terms are relative to Head verbs.

**passive_1**

| Term | Whole margin | Verb margin | Accuracy |
|---|---:|---:|---:|
| intercept | 3.04 [2.34, 3.64] | 2.90 [2.37, 3.44] | 0.82 [0.75, 0.89] |
| fit | 0.55 [0.39, 0.72] | 0.33 [0.17, 0.47] | 0.05 [0.02, 0.07] |
| tail | 0.50 [-0.41, 1.41] | -0.37 [-1.07, 0.35] | 0.01 [-0.08, 0.09] |
| xtail | -0.32 [-1.19, 0.58] | -1.05 [-1.79, -0.39] | -0.04 [-0.12, 0.05] |
| fit:tail | 0.13 [-0.10, 0.34] | 0.15 [-0.04, 0.34] | -0.00 [-0.03, 0.03] |
| fit:xtail | -0.12 [-0.37, 0.14] | -0.01 [-0.21, 0.21] | -0.01 [-0.04, 0.02] |

Own-context accuracy advantage after controlling fit: 5.0 pp [0.3, 9.1]

Accuracy (%) by fit bin, other contexts only:

| Fit bin | head | tail | xtail |
|---|---:|---:|---:|
| [1, 3] | 75.2 (n=1415) | 74.4 (n=3896) | 68.0 (n=4078) |
| [3, 5] | 85.7 (n=572) | 84.2 (n=948) | 75.1 (n=767) |
| [5, 7] | 90.9 (n=1263) | 91.7 (n=1406) | 85.8 (n=1405) |

**passive_2**

| Term | Whole margin | Verb margin | Accuracy |
|---|---:|---:|---:|
| intercept | 2.42 [1.69, 3.28] | 2.69 [2.21, 3.21] | 0.81 [0.73, 0.89] |
| fit | 0.41 [0.22, 0.60] | 0.40 [0.25, 0.55] | 0.05 [0.02, 0.08] |
| tail | 0.57 [-0.37, 1.45] | -0.37 [-1.09, 0.33] | 0.03 [-0.06, 0.13] |
| xtail | -0.55 [-1.51, 0.34] | -0.95 [-1.63, -0.34] | -0.08 [-0.18, 0.02] |
| fit:tail | 0.09 [-0.14, 0.33] | 0.06 [-0.12, 0.27] | -0.01 [-0.05, 0.02] |
| fit:xtail | -0.11 [-0.37, 0.14] | -0.10 [-0.31, 0.10] | -0.02 [-0.05, 0.02] |

Own-context accuracy advantage after controlling fit: 7.1 pp [0.3, 13.1]

Accuracy (%) by fit bin, other contexts only:

| Fit bin | head | tail | xtail |
|---|---:|---:|---:|
| [1, 3] | 69.7 (n=1148) | 77.1 (n=3409) | 65.8 (n=3731) |
| [3, 5] | 84.9 (n=486) | 84.2 (n=868) | 72.6 (n=798) |
| [5, 7] | 90.3 (n=1616) | 91.2 (n=1973) | 81.3 (n=1721) |

