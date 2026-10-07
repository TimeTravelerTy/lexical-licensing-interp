# DAS round 2: held-out active results

Site 10 (output of layer 9), 2 epochs, rank 1 (rule: rank 1 unless a larger rank raises mean held-out median pc_frac by > 0.1; mean held-out median positive-control fraction by rank {1: 0.788}). Primary set: 29 pairs ([] dropped by the behaviour filter). 5-fold CV over pairs x 3 split seeds; held-out pairs only; CIs = bootstrap over pairs. No passive has been evaluated.

## Primary run (29 pairs)

| Metric (held-out pairs) | Intransitive base <- transitive source | Transitive base <- intransitive source |
|---|---|---|
| IIA (patched M on the source's side) | 0.79 [0.68, 0.88] | 0.86 [0.78, 0.93] |
| Fraction of the natural gap in M (median per pair) | 0.77 [0.71, 0.84] | 0.69 [0.62, 0.76] |
| Fraction of the natural gap in log P(O) | 0.94 [0.89, 0.98] | 0.53 [0.45, 0.62] |
| Delta log P(O) | 2.50 [2.17, 2.83] | -1.32 [-1.54, -1.12] |
| Delta log P(determiners) | 2.26 [1.93, 2.60] | -1.28 [-1.50, -1.07] |
| Delta log P(pronouns) | 3.88 [3.50, 4.25] | -1.90 [-2.18, -1.63] |
| Delta log P(reflexives) | 3.62 [3.26, 3.98] | -1.31 [-1.50, -1.13] |
| Delta log P(I) | -1.28 [-1.48, -1.08] | 1.89 [1.64, 2.14] |

Per fold (mean over 15 fold x split runs):

| Condition | IIA cross | IIA same-class | Positive-control fraction (median) | IIA cross, held-out subject David |
|---|---|---|---|---|
| DAS | 0.822 | 0.991 | 0.788 | 0.818 |
| random | 0.018 | 0.982 | -0.000 | 0.018 |
| random_normmatched | 0.019 | 0.982 | 0.009 | 0.018 |
| shuffled_labels | 0.205 | 0.985 | 0.348 | 0.209 |

DAS beats the 95th percentile of norm-matched random subspaces on the positive-control fraction in 15 of 15 fold x split runs.

By source and token count (intransitive base <- transitive source; exploratory):

| Source | Tokens | Pairs | IIA | Gap fraction |
|---|---|---|---|---|
| expansion | multi | 15 | 0.81 [0.67, 0.92] | 0.77 [0.67, 0.85] |
| expansion | single | 6 | 0.74 [0.43, 0.97] | 0.78 [0.67, 0.87] |
| orig_head | single | 8 | 0.78 [0.64, 0.92] | 0.78 [0.68, 0.90] |

## Directions

- primary: pairwise cosine across its 15 fold x split bases: median 0.965 (min 0.939)
