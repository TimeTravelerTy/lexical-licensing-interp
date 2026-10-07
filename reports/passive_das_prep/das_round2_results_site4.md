# DAS round 2: held-out active results

Site 4 (output of layer 3), 3 epochs, rank 1 (rule: rank 1 unless a larger rank raises mean held-out median pc_frac by > 0.1; mean held-out median positive-control fraction by rank {1: 0.621}). Primary set: 29 pairs ([] dropped by the behaviour filter). 5-fold CV over pairs x 3 split seeds; held-out pairs only; CIs = bootstrap over pairs. No passive has been evaluated.

## Primary run (29 pairs)

| Metric (held-out pairs) | Intransitive base <- transitive source | Transitive base <- intransitive source |
|---|---|---|
| IIA (patched M on the source's side) | 0.54 [0.40, 0.66] | 0.36 [0.24, 0.48] |
| Fraction of the natural gap in M (median per pair) | 0.60 [0.52, 0.69] | 0.33 [0.27, 0.39] |
| Fraction of the natural gap in log P(O) | 0.78 [0.70, 0.87] | 0.18 [0.13, 0.24] |
| Delta log P(O) | 2.09 [1.75, 2.42] | -0.49 [-0.64, -0.36] |
| Delta log P(determiners) | 1.87 [1.54, 2.20] | -0.45 [-0.58, -0.34] |
| Delta log P(pronouns) | 3.26 [2.93, 3.59] | -0.82 [-1.05, -0.63] |
| Delta log P(reflexives) | 3.12 [2.78, 3.46] | -0.58 [-0.74, -0.44] |
| Delta log P(I) | -0.86 [-1.07, -0.66] | 1.17 [0.94, 1.42] |

Per fold (mean over 15 fold x split runs):

| Condition | IIA cross | IIA same-class | Positive-control fraction (median) | IIA cross, held-out subject David |
|---|---|---|---|---|
| DAS | 0.446 | 0.986 | 0.621 | 0.425 |
| random | 0.018 | 0.982 | -0.000 | 0.018 |
| random_normmatched | 0.019 | 0.982 | 0.006 | 0.018 |
| shuffled_labels | 0.129 | 0.979 | 0.272 | 0.111 |

DAS beats the 95th percentile of norm-matched random subspaces on the positive-control fraction in 15 of 15 fold x split runs.

By source and token count (intransitive base <- transitive source; exploratory):

| Source | Tokens | Pairs | IIA | Gap fraction |
|---|---|---|---|---|
| expansion | multi | 15 | 0.54 [0.35, 0.71] | 0.60 [0.49, 0.71] |
| expansion | single | 6 | 0.52 [0.19, 0.83] | 0.65 [0.49, 0.80] |
| orig_head | single | 8 | 0.55 [0.34, 0.77] | 0.57 [0.39, 0.75] |

## Directions

- primary: pairwise cosine across its 15 fold x split bases: median 0.951 (min 0.928)
