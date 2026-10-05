# DAS round 2: held-out active results

Site 17 (output of layer 16), 1 epochs, rank 1 (rule: rank 1 unless a larger rank raises mean held-out median pc_frac by > 0.1; mean held-out median positive-control fraction by rank {1: 1.067, 2: 1.087, 4: 1.098}). Primary set: 29 pairs ([] dropped by the behaviour filter). 5-fold CV over pairs x 3 split seeds; held-out pairs only; CIs = bootstrap over pairs. No passive has been evaluated.

## Primary run (29 pairs)

| Metric (held-out pairs) | Intransitive base <- transitive source | Transitive base <- intransitive source |
|---|---|---|
| IIA (patched M on the source's side) | 0.97 [0.94, 0.99] | 0.98 [0.96, 0.99] |
| Fraction of the natural gap in M (median per pair) | 1.06 [1.01, 1.11] | 0.93 [0.87, 0.99] |
| Fraction of the natural gap in log P(O) | 1.07 [1.05, 1.10] | 0.89 [0.79, 0.99] |
| Delta log P(O) | 2.86 [2.52, 3.19] | -2.18 [-2.39, -1.96] |
| Delta log P(determiners) | 2.66 [2.33, 2.99] | -2.14 [-2.34, -1.93] |
| Delta log P(pronouns) | 4.30 [3.90, 4.68] | -2.94 [-3.22, -2.68] |
| Delta log P(reflexives) | 3.66 [3.37, 3.94] | -2.63 [-2.88, -2.40] |
| Delta log P(I) | -2.24 [-2.47, -2.01] | 2.16 [1.96, 2.39] |

Per fold (mean over 15 fold x split runs):

| Condition | IIA cross | IIA same-class | Positive-control fraction (median) | IIA cross, held-out subject David |
|---|---|---|---|---|
| DAS | 0.973 | 0.987 | 1.067 | 0.964 |
| random | 0.019 | 0.981 | 0.000 | 0.018 |
| random_normmatched | 0.019 | 0.982 | 0.012 | 0.018 |
| shuffled_labels | 0.341 | 0.983 | 0.419 | 0.330 |

DAS beats the 95th percentile of norm-matched random subspaces on the positive-control fraction in 15 of 15 fold x split runs.

By source and token count (intransitive base <- transitive source; exploratory):

| Source | Tokens | Pairs | IIA | Gap fraction |
|---|---|---|---|---|
| expansion | multi | 15 | 0.99 [0.98, 1.00] | 1.09 [1.03, 1.15] |
| expansion | single | 6 | 0.94 [0.83, 1.00] | 1.02 [0.92, 1.10] |
| orig_head | single | 8 | 0.96 [0.88, 1.00] | 1.04 [0.94, 1.14] |

## Sensitivity run (39 pairs)

| Metric (held-out pairs) | Intransitive base <- transitive source | Transitive base <- intransitive source |
|---|---|---|
| IIA (patched M on the source's side) | 0.97 [0.95, 0.99] | 0.98 [0.96, 0.99] |
| Fraction of the natural gap in M (median per pair) | 1.08 [1.04, 1.12] | 0.91 [0.87, 0.96] |
| Fraction of the natural gap in log P(O) | 1.07 [1.05, 1.09] | 0.86 [0.79, 0.93] |
| Delta log P(O) | 3.10 [2.80, 3.40] | -2.40 [-2.59, -2.19] |
| Delta log P(determiners) | 2.89 [2.60, 3.20] | -2.34 [-2.52, -2.14] |
| Delta log P(pronouns) | 4.59 [4.26, 4.93] | -3.23 [-3.44, -3.00] |
| Delta log P(reflexives) | 3.68 [3.40, 3.97] | -2.60 [-2.80, -2.39] |
| Delta log P(I) | -2.37 [-2.56, -2.17] | 2.16 [1.99, 2.34] |

Per fold (mean over 15 fold x split runs):

| Condition | IIA cross | IIA same-class | Positive-control fraction (median) | IIA cross, held-out subject David |
|---|---|---|---|---|
| DAS | 0.976 | 0.991 | 1.084 | 0.968 |
| random | 0.014 | 0.986 | -0.000 | 0.013 |
| random_normmatched | 0.014 | 0.987 | 0.010 | 0.013 |
| shuffled_labels | 0.432 | 0.987 | 0.504 | 0.421 |

DAS beats the 95th percentile of norm-matched random subspaces on the positive-control fraction in 15 of 15 fold x split runs.

By source and token count (intransitive base <- transitive source; exploratory):

| Source | Tokens | Pairs | IIA | Gap fraction |
|---|---|---|---|---|
| expansion | multi | 21 | 0.98 [0.95, 1.00] | 1.09 [1.03, 1.15] |
| expansion | single | 8 | 0.97 [0.91, 1.00] | 1.09 [0.99, 1.18] |
| orig_head | single | 10 | 0.97 [0.95, 0.99] | 1.06 [1.00, 1.12] |

## Directions

- primary: pairwise cosine across its 15 fold x split bases: median 0.917 (min 0.899)
- sensitivity: pairwise cosine across its 15 fold x split bases: median 0.948 (min 0.934)
- primary vs sensitivity: cosine of mean directions 0.979; all primary x sensitivity base pairs median 0.916 (min 0.888).
- held-out transfer agreement on the 26 intransitive-base pairs present in both runs: gap fraction primary 1.07 vs sensitivity 1.11, Pearson r 0.77.
