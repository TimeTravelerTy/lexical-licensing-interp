# DAS round 2: held-out active results

Site 6 (output of layer 5), 2 epochs, rank 1 (rule: rank 1 unless a larger rank raises mean held-out median pc_frac by > 0.1; mean held-out median positive-control fraction by rank {1: 0.72}). Primary set: 29 pairs ([] dropped by the behaviour filter). 5-fold CV over pairs x 3 split seeds; held-out pairs only; CIs = bootstrap over pairs. No passive has been evaluated.

## Primary run (29 pairs)

| Metric (held-out pairs) | Intransitive base <- transitive source | Transitive base <- intransitive source |
|---|---|---|
| IIA (patched M on the source's side) | 0.66 [0.54, 0.77] | 0.65 [0.54, 0.76] |
| Fraction of the natural gap in M (median per pair) | 0.70 [0.62, 0.77] | 0.52 [0.45, 0.59] |
| Fraction of the natural gap in log P(O) | 0.88 [0.81, 0.94] | 0.36 [0.28, 0.44] |
| Delta log P(O) | 2.33 [1.99, 2.65] | -0.95 [-1.16, -0.74] |
| Delta log P(determiners) | 2.09 [1.75, 2.42] | -0.89 [-1.08, -0.70] |
| Delta log P(pronouns) | 3.62 [3.27, 3.95] | -1.42 [-1.68, -1.19] |
| Delta log P(reflexives) | 3.45 [3.09, 3.80] | -0.98 [-1.19, -0.79] |
| Delta log P(I) | -1.06 [-1.26, -0.87] | 1.58 [1.34, 1.84] |

Per fold (mean over 15 fold x split runs):

| Condition | IIA cross | IIA same-class | Positive-control fraction (median) | IIA cross, held-out subject David |
|---|---|---|---|---|
| DAS | 0.656 | 0.991 | 0.720 | 0.650 |
| random | 0.018 | 0.982 | -0.000 | 0.018 |
| random_normmatched | 0.019 | 0.982 | 0.008 | 0.018 |
| shuffled_labels | 0.168 | 0.981 | 0.320 | 0.160 |

DAS beats the 95th percentile of norm-matched random subspaces on the positive-control fraction in 15 of 15 fold x split runs.

By source and token count (intransitive base <- transitive source; exploratory):

| Source | Tokens | Pairs | IIA | Gap fraction |
|---|---|---|---|---|
| expansion | multi | 15 | 0.67 [0.50, 0.81] | 0.68 [0.57, 0.78] |
| expansion | single | 6 | 0.64 [0.34, 0.91] | 0.74 [0.62, 0.86] |
| orig_head | single | 8 | 0.66 [0.49, 0.85] | 0.69 [0.53, 0.86] |

## Directions

- primary: pairwise cosine across its 15 fold x split bases: median 0.958 (min 0.934)
