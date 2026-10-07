# DAS round 2: held-out active results

Site 16 (output of layer 15), 2 epochs, rank 1 (rule: rank 1 unless a larger rank raises mean held-out median pc_frac by > 0.1; mean held-out median positive-control fraction by rank {1: 1.045}). Primary set: 29 pairs ([] dropped by the behaviour filter). 5-fold CV over pairs x 3 split seeds; held-out pairs only; CIs = bootstrap over pairs. No passive has been evaluated.

## Primary run (29 pairs)

| Metric (held-out pairs) | Intransitive base <- transitive source | Transitive base <- intransitive source |
|---|---|---|
| IIA (patched M on the source's side) | 0.96 [0.93, 0.99] | 0.97 [0.93, 0.99] |
| Fraction of the natural gap in M (median per pair) | 1.03 [0.98, 1.08] | 0.94 [0.87, 1.01] |
| Fraction of the natural gap in log P(O) | 1.06 [1.03, 1.09] | 0.89 [0.79, 1.00] |
| Delta log P(O) | 2.83 [2.49, 3.16] | -2.20 [-2.43, -1.96] |
| Delta log P(determiners) | 2.58 [2.25, 2.92] | -2.13 [-2.37, -1.90] |
| Delta log P(pronouns) | 4.46 [4.04, 4.87] | -2.86 [-3.17, -2.55] |
| Delta log P(reflexives) | 3.50 [3.22, 3.77] | -2.49 [-2.73, -2.27] |
| Delta log P(I) | -2.18 [-2.43, -1.93] | 2.18 [1.96, 2.40] |

Per fold (mean over 15 fold x split runs):

| Condition | IIA cross | IIA same-class | Positive-control fraction (median) | IIA cross, held-out subject David |
|---|---|---|---|---|
| DAS | 0.966 | 0.990 | 1.045 | 0.953 |
| random | 0.019 | 0.981 | -0.000 | 0.018 |
| random_normmatched | 0.019 | 0.982 | 0.005 | 0.018 |
| shuffled_labels | 0.308 | 0.984 | 0.398 | 0.298 |

DAS beats the 95th percentile of norm-matched random subspaces on the positive-control fraction in 15 of 15 fold x split runs.

By source and token count (intransitive base <- transitive source; exploratory):

| Source | Tokens | Pairs | IIA | Gap fraction |
|---|---|---|---|---|
| expansion | multi | 15 | 0.98 [0.97, 1.00] | 1.06 [1.00, 1.11] |
| expansion | single | 6 | 0.94 [0.84, 1.00] | 1.00 [0.91, 1.08] |
| orig_head | single | 8 | 0.94 [0.85, 1.00] | 1.00 [0.90, 1.12] |

## Directions

- primary: pairwise cosine across its 15 fold x split bases: median 0.965 (min 0.949)
