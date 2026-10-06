# Task 3: single-prompt readout

log-odds(" by" vs ".") after the participle within one prompt, against the two-sentence *by* margin log P(by | good) - log P(by | bad). Released `passive_1` contexts (37,800 items). Item-level CIs: two-way cluster bootstrap (200 draws); verb-pair-level CIs: bootstrap over pairs (1000 draws).

Sanity: readout log P(by | good) vs full-sentence `good_by_lp`: median |diff| 0.036, 99th pct 0.268 (bf16, different batching).

## Raw log-probs after the participle (band means, item level)

| Quantity | Head | Tail | XTail | XTail - Head |
|---|---:|---:|---:|---:|
| log P(" by" \| good) | -3.01 [-3.57, -2.55] | -2.54 [-2.85, -2.22] | -2.68 [-2.96, -2.40] | 0.33 [-0.21, 0.94] |
| log P("." \| good) | -3.76 [-4.19, -3.39] | -3.05 [-3.26, -2.84] | -3.26 [-3.55, -3.00] | 0.50 [0.02, 0.98] |
| log-odds by vs . (good) | 0.75 [0.23, 1.24] | 0.51 [0.22, 0.82] | 0.58 [0.28, 0.87] | -0.17 [-0.72, 0.38] |
| log P(" by" \| bad) | -4.12 [-4.57, -3.73] | -3.60 [-3.86, -3.34] | -3.16 [-3.39, -2.94] | 0.96 [0.51, 1.45] |
| log P("." \| bad) | -3.93 [-4.42, -3.50] | -3.79 [-4.08, -3.52] | -3.44 [-3.66, -3.19] | 0.49 [0.02, 1.04] |
| log-odds by vs . (bad) | -0.19 [-0.61, 0.26] | 0.19 [-0.05, 0.41] | 0.28 [0.08, 0.47] | 0.47 [-0.01, 0.92] |
| two-sentence by margin | 1.11 [0.32, 1.90] | 1.06 [0.70, 1.42] | 0.47 [0.11, 0.83] | -0.63 [-1.49, 0.19] |

## Spearman with the two-sentence *by* margin

| Readout | Item | Verb pair (all) | Pair: Head | Pair: Tail | Pair: XTail |
|---|---:|---:|---:|---:|---:|
| log-odds (good prompt) | 0.41 [0.33, 0.49] | 0.54 [0.39, 0.65] | 0.49 [0.11, 0.74] | 0.47 [0.22, 0.66] | 0.59 [0.38, 0.75] |
| log-odds (bad prompt) | -0.25 [-0.33, -0.16] | -0.24 [-0.39, -0.07] | -0.52 [-0.76, -0.11] | -0.06 [-0.36, 0.27] | -0.22 [-0.47, 0.06] |
| log-odds good - bad | 0.55 [0.48, 0.64] | 0.56 [0.42, 0.68] | 0.66 [0.37, 0.83] | 0.40 [0.08, 0.64] | 0.56 [0.34, 0.72] |
| log P(" by" \| good) alone | 0.67 [0.60, 0.74] | 0.72 [0.62, 0.80] | 0.77 [0.51, 0.90] | 0.68 [0.46, 0.82] | 0.80 [0.65, 0.88] |
| log P("." \| good) alone | 0.17 [0.07, 0.29] | 0.24 [0.07, 0.40] | 0.25 [-0.19, 0.61] | 0.34 [0.03, 0.58] | 0.19 [-0.10, 0.48] |

Curated `passive_1`, other contexts (15,750 items), item-level Spearman with the curated *by* margin: log-odds (good) 0.38 [0.29, 0.46]; good - bad 0.49 [0.40, 0.58].
