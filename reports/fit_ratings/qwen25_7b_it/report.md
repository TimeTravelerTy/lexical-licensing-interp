# Fit ratings: qwen25_7b_it

## Validation

- **Format**: digit mass median 1.000; 0.0% of ratings below 0.9.

- **Prompt agreement** (Spearman, all items): 0.899

- **Direction checks**
  - own > mean of other contexts for 100.0% of verb pairs (mean own 5.34, other 2.71)
  - own > role reversal (passive_1) for 92.9% of verb pairs (mean difference 3.26); reversal is not always implausible

- **Rarity check**: mean rating by verb band (own and reversal should be flat)

| Paradigm | Verb band | Own | Other | Reversal | n |
|---|---|---:|---:|---:|---:|
| passive_1 | head | 6.39 | 3.42 | 2.57 | 26 |
| passive_1 | tail | 5.89 | 2.83 | 2.45 | 50 |
| passive_1 | xtail | 5.81 | 2.86 | 3.03 | 50 |
| passive_2 | head | 4.98 | 2.82 | - | 26 |
| passive_2 | tail | 4.74 | 2.37 | - | 50 |
| passive_2 | xtail | 4.57 | 2.35 | - | 50 |

- Spearman(own rating, participle Zipf): 0.141; Spearman(other rating, participle Zipf): 0.164

- **Human agreement**: Spearman 0.697 on 132 items

- **Second rater agreement**: Spearman 0.726 on 28971 items

## Pythia margin vs fit (curated cross)

`fit` is the rating minus 4 (per rating point); band terms are relative to Head verbs.

**passive_1**

| Term | Whole margin | Verb margin | Accuracy |
|---|---:|---:|---:|
| intercept | 3.35 [2.68, 3.97] | 3.08 [2.58, 3.59] | 0.85 [0.78, 0.90] |
| fit | 0.72 [0.47, 0.99] | 0.42 [0.26, 0.59] | 0.06 [0.02, 0.10] |
| tail | 0.55 [-0.32, 1.50] | -0.30 [-1.01, 0.43] | 0.01 [-0.07, 0.09] |
| xtail | -0.33 [-1.17, 0.53] | -1.04 [-1.74, -0.39] | -0.04 [-0.11, 0.04] |
| fit:tail | 0.25 [-0.10, 0.59] | 0.25 [0.02, 0.48] | 0.01 [-0.04, 0.05] |
| fit:xtail | 0.02 [-0.35, 0.39] | 0.10 [-0.14, 0.35] | 0.01 [-0.04, 0.05] |

Own-context accuracy advantage after controlling fit: 1.8 pp [-4.1, 6.9]

Accuracy (%) by fit bin, other contexts only:

| Fit bin | head | tail | xtail |
|---|---:|---:|---:|
| [1, 3] | 75.8 (n=1291) | 73.3 (n=3370) | 67.2 (n=3332) |
| [3, 5] | 84.6 (n=1354) | 84.6 (n=2410) | 77.1 (n=2463) |
| [5, 7] | 95.9 (n=605) | 96.6 (n=470) | 93.0 (n=455) |

**passive_2**

| Term | Whole margin | Verb margin | Accuracy |
|---|---:|---:|---:|
| intercept | 3.51 [2.73, 4.45] | 3.60 [3.03, 4.22] | 0.92 [0.87, 0.97] |
| fit | 0.79 [0.36, 1.23] | 0.65 [0.40, 0.91] | 0.08 [0.02, 0.14] |
| tail | 0.66 [-0.45, 1.64] | -0.37 [-1.25, 0.48] | 0.01 [-0.06, 0.08] |
| xtail | -0.44 [-1.52, 0.51] | -1.08 [-1.89, -0.35] | -0.05 [-0.13, 0.02] |
| fit:tail | 0.15 [-0.36, 0.64] | 0.11 [-0.20, 0.43] | -0.01 [-0.08, 0.06] |
| fit:xtail | 0.12 [-0.39, 0.60] | 0.00 [-0.35, 0.31] | 0.02 [-0.05, 0.09] |

Own-context accuracy advantage after controlling fit: 1.5 pp [-5.2, 7.0]

Accuracy (%) by fit bin, other contexts only:

| Fit bin | head | tail | xtail |
|---|---:|---:|---:|
| [1, 3] | 78.4 (n=1580) | 77.0 (n=3834) | 65.0 (n=3927) |
| [3, 5] | 88.3 (n=1495) | 90.7 (n=2250) | 78.9 (n=2224) |
| [5, 7] | 98.3 (n=175) | 96.9 (n=166) | 90.3 (n=99) |

