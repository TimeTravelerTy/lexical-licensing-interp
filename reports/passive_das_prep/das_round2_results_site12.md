# DAS round 2: held-out active results

Site 12 (output of layer 11), 2 epochs, rank 1 (rule: rank 1 unless a larger rank raises mean held-out median pc_frac by > 0.1; mean held-out median positive-control fraction by rank {1: 0.87}). Primary set: 29 pairs ([] dropped by the behaviour filter). 5-fold CV over pairs x 3 split seeds; held-out pairs only; CIs = bootstrap over pairs. No passive has been evaluated.

## Primary run (29 pairs)

| Metric (held-out pairs) | Intransitive base <- transitive source | Transitive base <- intransitive source |
|---|---|---|
| IIA (patched M on the source's side) | 0.87 [0.80, 0.94] | 0.93 [0.85, 0.97] |
| Fraction of the natural gap in M (median per pair) | 0.86 [0.80, 0.92] | 0.83 [0.75, 0.91] |
| Fraction of the natural gap in log P(O) | 0.99 [0.95, 1.03] | 0.74 [0.63, 0.86] |
| Delta log P(O) | 2.64 [2.30, 2.97] | -1.80 [-2.05, -1.55] |
| Delta log P(determiners) | 2.37 [2.04, 2.71] | -1.74 [-2.00, -1.49] |
| Delta log P(pronouns) | 4.26 [3.86, 4.66] | -2.53 [-2.84, -2.24] |
| Delta log P(reflexives) | 3.40 [3.05, 3.73] | -1.48 [-1.71, -1.26] |
| Delta log P(I) | -1.52 [-1.74, -1.32] | 2.05 [1.82, 2.30] |

Per fold (mean over 15 fold x split runs):

| Condition | IIA cross | IIA same-class | Positive-control fraction (median) | IIA cross, held-out subject David |
|---|---|---|---|---|
| DAS | 0.898 | 0.990 | 0.870 | 0.894 |
| random | 0.018 | 0.982 | -0.000 | 0.018 |
| random_normmatched | 0.019 | 0.982 | 0.011 | 0.018 |
| shuffled_labels | 0.242 | 0.983 | 0.352 | 0.231 |

DAS beats the 95th percentile of norm-matched random subspaces on the positive-control fraction in 15 of 15 fold x split runs.

By source and token count (intransitive base <- transitive source; exploratory):

| Source | Tokens | Pairs | IIA | Gap fraction |
|---|---|---|---|---|
| expansion | multi | 15 | 0.91 [0.81, 0.98] | 0.87 [0.78, 0.95] |
| expansion | single | 6 | 0.83 [0.65, 0.98] | 0.86 [0.78, 0.93] |
| orig_head | single | 8 | 0.84 [0.71, 0.94] | 0.84 [0.73, 0.96] |

## Directions

- primary: pairwise cosine across its 15 fold x split bases: median 0.966 (min 0.942)
