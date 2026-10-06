# DAS round 2: held-out active results

Site 8 (output of layer 7), 3 epochs, rank 1 (rule: rank 1 unless a larger rank raises mean held-out median pc_frac by > 0.1; mean held-out median positive-control fraction by rank {1: 0.777}). Primary set: 29 pairs ([] dropped by the behaviour filter). 5-fold CV over pairs x 3 split seeds; held-out pairs only; CIs = bootstrap over pairs. No passive has been evaluated.

## Primary run (29 pairs)

| Metric (held-out pairs) | Intransitive base <- transitive source | Transitive base <- intransitive source |
|---|---|---|
| IIA (patched M on the source's side) | 0.76 [0.65, 0.85] | 0.81 [0.71, 0.89] |
| Fraction of the natural gap in M (median per pair) | 0.75 [0.68, 0.82] | 0.64 [0.57, 0.71] |
| Fraction of the natural gap in log P(O) | 0.92 [0.87, 0.97] | 0.48 [0.40, 0.57] |
| Delta log P(O) | 2.46 [2.12, 2.79] | -1.23 [-1.45, -1.02] |
| Delta log P(determiners) | 2.20 [1.88, 2.54] | -1.15 [-1.36, -0.96] |
| Delta log P(pronouns) | 3.82 [3.44, 4.18] | -1.80 [-2.10, -1.53] |
| Delta log P(reflexives) | 3.73 [3.36, 4.10] | -1.41 [-1.63, -1.20] |
| Delta log P(I) | -1.21 [-1.41, -1.02] | 1.80 [1.55, 2.07] |

Per fold (mean over 15 fold x split runs):

| Condition | IIA cross | IIA same-class | Positive-control fraction (median) | IIA cross, held-out subject David |
|---|---|---|---|---|
| DAS | 0.782 | 0.990 | 0.777 | 0.780 |
| random | 0.018 | 0.982 | -0.000 | 0.018 |
| random_normmatched | 0.018 | 0.982 | 0.009 | 0.018 |
| shuffled_labels | 0.199 | 0.983 | 0.374 | 0.183 |

DAS beats the 95th percentile of norm-matched random subspaces on the positive-control fraction in 15 of 15 fold x split runs.

By source and token count (intransitive base <- transitive source; exploratory):

| Source | Tokens | Pairs | IIA | Gap fraction |
|---|---|---|---|---|
| expansion | multi | 15 | 0.77 [0.63, 0.89] | 0.73 [0.64, 0.83] |
| expansion | single | 6 | 0.75 [0.44, 0.97] | 0.80 [0.66, 0.90] |
| orig_head | single | 8 | 0.74 [0.58, 0.90] | 0.75 [0.64, 0.87] |

## Directions

- primary: pairwise cosine across its 15 fold x split bases: median 0.964 (min 0.947)
