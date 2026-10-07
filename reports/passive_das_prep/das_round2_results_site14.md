# DAS round 2: held-out active results

Site 14 (output of layer 13), 2 epochs, rank 1 (rule: rank 1 unless a larger rank raises mean held-out median pc_frac by > 0.1; mean held-out median positive-control fraction by rank {1: 0.959}). Primary set: 29 pairs ([] dropped by the behaviour filter). 5-fold CV over pairs x 3 split seeds; held-out pairs only; CIs = bootstrap over pairs. No passive has been evaluated.

## Primary run (29 pairs)

| Metric (held-out pairs) | Intransitive base <- transitive source | Transitive base <- intransitive source |
|---|---|---|
| IIA (patched M on the source's side) | 0.94 [0.89, 0.98] | 0.96 [0.90, 0.99] |
| Fraction of the natural gap in M (median per pair) | 0.95 [0.90, 1.00] | 0.89 [0.82, 0.96] |
| Fraction of the natural gap in log P(O) | 1.03 [1.00, 1.07] | 0.82 [0.72, 0.93] |
| Delta log P(O) | 2.76 [2.41, 3.09] | -2.00 [-2.23, -1.78] |
| Delta log P(determiners) | 2.48 [2.15, 2.82] | -1.94 [-2.18, -1.70] |
| Delta log P(pronouns) | 4.48 [4.06, 4.87] | -2.79 [-3.07, -2.50] |
| Delta log P(reflexives) | 3.22 [2.91, 3.53] | -1.82 [-2.02, -1.61] |
| Delta log P(I) | -1.84 [-2.07, -1.61] | 2.13 [1.92, 2.36] |

Per fold (mean over 15 fold x split runs):

| Condition | IIA cross | IIA same-class | Positive-control fraction (median) | IIA cross, held-out subject David |
|---|---|---|---|---|
| DAS | 0.947 | 0.990 | 0.959 | 0.938 |
| random | 0.019 | 0.981 | -0.000 | 0.018 |
| random_normmatched | 0.019 | 0.982 | 0.009 | 0.018 |
| shuffled_labels | 0.274 | 0.983 | 0.360 | 0.269 |

DAS beats the 95th percentile of norm-matched random subspaces on the positive-control fraction in 15 of 15 fold x split runs.

By source and token count (intransitive base <- transitive source; exploratory):

| Source | Tokens | Pairs | IIA | Gap fraction |
|---|---|---|---|---|
| expansion | multi | 15 | 0.97 [0.93, 1.00] | 0.96 [0.89, 1.03] |
| expansion | single | 6 | 0.89 [0.72, 0.99] | 0.94 [0.84, 1.03] |
| orig_head | single | 8 | 0.92 [0.80, 0.99] | 0.93 [0.82, 1.03] |

## Directions

- primary: pairwise cosine across its 15 fold x split bases: median 0.964 (min 0.945)
