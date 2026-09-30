# Bad-side fit: bad_gemma4_31b_it

## Validation

- **Format**: digit mass median 1.000; 0.0% below 0.9.
- **Prompt agreement**: Spearman 0.966
- **Mean bad-side rating by bad-verb band**: head 4.71, tail 4.12, xtail 3.84
- **Second rater agreement**: Spearman 0.860
- **Human agreement**: Spearman 0.831 on 60 items

Lowest rated: The money chuckled.; The money slouched.; The gate hibernated.; The tail glared.; The secret bragged.; The beer lumbered.; The statue quacked.; The statue thrived.

Highest rated: The burglar appeared.; The heretic seethed.; The pickpocket lied.; The winemaker sulked.; The rival glared.; The burglar slouched.; The soldier replied.; The government functioned.

- Correlation of good fit with bad-patient fit (Pearson): passive_1 0.16, passive_2 0.23

## Margin models

Fit terms are per rating point (rating - 4). Accuracy in pp.

**passive_1**, full model (good+bad)

| Term | correct | whole_margin | verb_margin | by_margin | suffix_margin |
|---|---:|---:|---:|---:|---:|
| good_fit | 4.57 [3.54, 5.67] | 0.57 [0.48, 0.68] | 0.41 [0.34, 0.49] | 0.09 [0.03, 0.16] | 0.16 [0.09, 0.25] |
| bad_patient | -1.03 [-1.82, -0.28] | -0.07 [-0.14, -0.01] | -0.14 [-0.20, -0.09] | -0.00 [-0.05, 0.04] | 0.07 [0.02, 0.12] |
| bad_agent | -1.85 [-2.86, -0.85] | -0.22 [-0.34, -0.11] | -0.20 [-0.29, -0.12] | 0.13 [0.05, 0.21] | -0.02 [-0.14, 0.09] |
| tail | -0.66 [-9.22, 8.47] | 0.22 [-0.63, 1.09] | -0.63 [-1.33, 0.05] | 0.17 [-0.58, 0.94] | 0.85 [0.16, 1.59] |
| xtail | -5.44 [-13.75, 3.54] | -0.38 [-1.22, 0.50] | -1.22 [-1.87, -0.58] | -0.26 [-1.06, 0.56] | 0.85 [0.19, 1.51] |

XTail - Head across models (passive_1):

| Model | correct | whole_margin | verb_margin | by_margin | suffix_margin |
|---|---:|---:|---:|---:|---:|
| unmatched | -8.01 [-17.25, 1.74] | -0.76 [-1.64, 0.11] | -1.39 [-2.07, -0.69] | -0.45 [-1.29, 0.40] | 0.63 [-0.02, 1.31] |
| good | -3.39 [-12.16, 6.11] | -0.17 [-1.02, 0.70] | -0.98 [-1.69, -0.34] | -0.35 [-1.18, 0.47] | 0.82 [0.15, 1.48] |
| good+bad_patient | -4.38 [-12.92, 4.79] | -0.25 [-1.10, 0.63] | -1.11 [-1.78, -0.46] | -0.33 [-1.16, 0.49] | 0.86 [0.21, 1.53] |
| good+bad | -5.44 [-13.75, 3.54] | -0.38 [-1.22, 0.50] | -1.22 [-1.87, -0.58] | -0.26 [-1.06, 0.56] | 0.85 [0.19, 1.51] |

Own-context advantage (passive_1):

| Model | correct | whole_margin | verb_margin | by_margin | suffix_margin |
|---|---:|---:|---:|---:|---:|
| own | 21.60 [16.04, 26.23] | 5.56 [4.38, 6.63] | 3.79 [2.93, 4.67] | -0.04 [-0.49, 0.47] | 1.77 [0.93, 2.66] |
| good+own | 5.01 [0.25, 9.15] | 3.51 [2.26, 4.65] | 2.38 [1.44, 3.22] | -0.44 [-0.97, 0.09] | 1.13 [0.16, 2.12] |

**passive_2**, full model (good+bad)

| Term | correct | whole_margin | verb_margin | suffix_margin |
|---|---:|---:|---:|---:|
| good_fit | 4.38 [3.19, 5.56] | 0.46 [0.37, 0.55] | 0.43 [0.36, 0.51] | 0.02 [-0.05, 0.09] |
| bad_patient | -2.57 [-3.36, -1.75] | -0.23 [-0.29, -0.17] | -0.21 [-0.27, -0.16] | -0.02 [-0.07, 0.03] |
| tail | 1.90 [-6.26, 10.91] | 0.42 [-0.47, 1.27] | -0.50 [-1.21, 0.22] | 0.92 [0.13, 1.73] |
| xtail | -9.34 [-19.05, -0.22] | -0.60 [-1.51, 0.25] | -1.00 [-1.66, -0.34] | 0.40 [-0.37, 1.14] |

XTail - Head across models (passive_2):

| Model | correct | whole_margin | verb_margin | suffix_margin |
|---|---:|---:|---:|---:|
| unmatched | -12.92 [-22.78, -2.50] | -1.00 [-1.95, -0.18] | -1.39 [-2.11, -0.74] | 0.39 [-0.40, 1.13] |
| good | -7.83 [-17.82, 2.12] | -0.46 [-1.38, 0.41] | -0.87 [-1.53, -0.24] | 0.41 [-0.36, 1.15] |
| good+bad_patient | -9.34 [-19.05, -0.22] | -0.60 [-1.51, 0.25] | -1.00 [-1.66, -0.34] | 0.40 [-0.37, 1.14] |
| good+bad | -9.34 [-19.05, -0.22] | -0.60 [-1.51, 0.25] | -1.00 [-1.66, -0.34] | 0.40 [-0.37, 1.14] |

Own-context advantage (passive_2):

| Model | correct | whole_margin | verb_margin | suffix_margin |
|---|---:|---:|---:|---:|
| own | 20.45 [13.70, 25.88] | 3.80 [2.87, 4.67] | 3.79 [2.88, 4.70] | 0.02 [-0.46, 0.52] |
| good+own | 7.12 [0.28, 13.08] | 2.41 [1.47, 3.30] | 2.46 [1.55, 3.33] | -0.05 [-0.57, 0.51] |

