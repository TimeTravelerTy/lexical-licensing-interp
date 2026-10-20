# Round 6, part 0: FreqBLiMP human validation, passive paradigms

Spec: `plan.md`, part 0. Script: `analyze_human_passives.py`. Source: the merged three-annotator export (corrected 2026-08-25; sha256 `d5fe7ee580560f88…`, outside the repo); per-item table `../../data/round6/human/passive_items.csv`.

**Format.** Forced choice between the two sentences of a pair (A / B / can't decide, with reasons), then a 1–5 naturalness rating of the intended acceptable sentence only. There are no per-sentence acceptability ratings, so bad-item acceptance cannot be computed; per the round-6 rule, accuracy by band is reported, descriptively. 15 pairs per paradigm × condition, each judged by all three annotators (45 judgments). 95% CIs: items resampled within cell (2,000 draws, seed 17); with 15 items per cell they are wide.

## `passive_1`

| Condition | judgments | correct among decided (%) | can't decide (%) | accuracy, can't decide = ½ (%) | items with 2-of-3 support | naturalness (good sentence) | Pythia-1.4B LP, same items (%) |
|---|---:|---|---|---|---:|---|---|
| Original | 45 | 97.6 [91.4, 100.0] | 8.9 [0.0, 22.2] | 93.3 [82.2, 100.0] | 13 of 15 | 4.07 [3.69, 4.38] | — |
| Head | 45 | 94.6 [85.3, 100.0] | 17.8 [8.8, 28.9] | 86.7 [75.6, 95.6] | 13 of 15 | 2.51 [1.91, 3.13] | 80.0 [60.0, 100.0] |
| Tail | 45 | 88.2 [74.3, 100.0] | 24.4 [11.1, 37.8] | 78.9 [68.9, 88.9] | 10 of 15 | 2.36 [2.00, 2.73] | 66.7 [40.0, 86.7] |
| XTail | 45 | 85.7 [75.0, 96.3] | 37.8 [24.4, 51.1] | 72.2 [65.6, 80.0] | 7 of 15 | 1.87 [1.51, 2.29] | 73.3 [53.3, 93.3] |

## `passive_2`

| Condition | judgments | correct among decided (%) | can't decide (%) | accuracy, can't decide = ½ (%) | items with 2-of-3 support | naturalness (good sentence) | Pythia-1.4B LP, same items (%) |
|---|---:|---|---|---|---:|---|---|
| Original | 45 | 100.0 [100.0, 100.0] | 4.4 [0.0, 11.1] | 97.8 [94.4, 100.0] | 15 of 15 | 4.67 [4.51, 4.80] | — |
| Head | 45 | 97.3 [91.2, 100.0] | 17.8 [6.7, 31.1] | 88.9 [80.0, 95.6] | 12 of 15 | 2.98 [2.33, 3.60] | 80.0 [60.0, 100.0] |
| Tail | 45 | 83.8 [71.1, 94.7] | 17.8 [8.8, 28.9] | 77.8 [66.7, 87.8] | 11 of 15 | 2.80 [2.22, 3.36] | 80.0 [60.0, 100.0] |
| XTail | 45 | 91.2 [79.4, 100.0] | 24.4 [11.1, 37.8] | 81.1 [68.9, 91.1] | 10 of 15 | 2.82 [2.38, 3.27] | 80.0 [60.0, 100.0] |

Can't-decide reasons (counts; an annotator could give several):

| paradigm | condition | both_correct | both_wrong | other | unknown_words |
|---|---|---:|---:|---:|---:|
| passive_1 | Head | 5 | 3 | 1 | 0 |
| passive_1 | Original | 4 | 0 | 0 | 0 |
| passive_1 | Tail | 9 | 4 | 1 | 0 |
| passive_1 | XTail | 14 | 2 | 0 | 7 |
| passive_2 | Head | 4 | 4 | 1 | 0 |
| passive_2 | Original | 2 | 0 | 0 | 0 |
| passive_2 | Tail | 4 | 4 | 1 | 0 |
| passive_2 | XTail | 7 | 4 | 1 | 2 |

Items where 2 of 3 chose the intended unacceptable sentence: 2 (`tail:passive_1:495`: The deliberation is tinged by the neuroscientists. / *The deliberation is mused by the neuroscientists.*; `tail:passive_2:970`: The machinist isn't derided. / *The machinist isn't branched.*).
