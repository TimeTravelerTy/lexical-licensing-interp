# Task 4: Yes/No readout

Prompt: FreqBLiMP base-model Yes/No prompt (blimp-rare `YES_NO_BASE_PROMPT`); score = P(Yes)/(P(Yes)+P(No)), Yes = {"Yes", " Yes"}, No = {"No", " No"}; a pair is correct if the good sentence scores higher. `yn_margin` = logit(score good) - logit(score bad). Curated cross, both paradigms. 95% CIs: two-way cluster bootstrap (1000 draws).

## Accuracy (%) by verb band

Head rows first answer the gating question (chance = 50%).

| Paradigm | Contexts | Verb band | Yes/No acc | LP acc | Yes/No margin |
|---|---|---|---:|---:|---:|
| passive_1 | other | head | 60.8 [50.5, 69.8] | 81.4 [72.9, 88.9] | 0.1 [0.0, 0.1] |
| passive_1 | other | tail | 51.5 [43.1, 59.4] | 77.9 [72.5, 83.7] | -0.0 [-0.1, 0.0] |
| passive_1 | other | xtail | 51.5 [44.2, 59.2] | 73.3 [68.5, 78.4] | 0.0 [-0.0, 0.0] |
| passive_1 | other | xtail-head | -9.3 [-21.0, 2.8] | -8.1 [-16.8, 1.9] | -0.1 [-0.1, 0.0] |
| passive_1 | own | head | 46.2 [16.7, 79.2] | 100.0 [100.0, 100.0] | -0.0 [-0.1, 0.2] |
| passive_1 | own | tail | 56.0 [34.0, 77.8] | 98.0 [89.5, 100.0] | 0.0 [-0.1, 0.1] |
| passive_1 | own | xtail | 56.0 [34.3, 78.6] | 98.0 [87.2, 100.0] | 0.0 [-0.1, 0.1] |
| passive_1 | own | xtail-head | 9.8 [-30.0, 45.1] | -2.0 [-12.8, 0.0] | 0.0 [-0.2, 0.2] |
| passive_2 | other | head | 57.6 [46.2, 68.0] | 82.8 [75.0, 89.9] | 0.1 [-0.0, 0.1] |
| passive_2 | other | tail | 47.8 [39.4, 56.9] | 81.6 [75.8, 86.5] | -0.0 [-0.1, 0.0] |
| passive_2 | other | xtail | 52.4 [43.1, 61.1] | 69.8 [63.0, 75.7] | 0.0 [-0.0, 0.1] |
| passive_2 | other | xtail-head | -5.2 [-19.1, 8.8] | -13.0 [-22.9, -2.7] | -0.1 [-0.1, 0.0] |
| passive_2 | own | head | 53.8 [20.7, 84.0] | 100.0 [100.0, 100.0] | -0.0 [-0.2, 0.1] |
| passive_2 | own | tail | 48.0 [25.4, 72.6] | 98.0 [89.3, 100.0] | 0.0 [-0.1, 0.1] |
| passive_2 | own | xtail | 54.0 [32.3, 76.9] | 96.0 [82.7, 100.0] | 0.0 [-0.1, 0.1] |
| passive_2 | own | xtail-head | 0.2 [-39.2, 41.8] | -4.0 [-17.3, 0.0] | 0.0 [-0.1, 0.2] |

Yes+No probability mass on good prompts: median 0.443 (10th pct 0.397). Mean P(Yes) share, good vs bad: 0.581 vs 0.580. Spearman(yn_margin, whole_margin) over items: -0.028.

## Fit effects: Yes/No vs LP

WLS `y ~ good_fit + bad_patient + band` (the `good+bad_patient` spec of `analyze_bad_fit.py`), all curated items (own included, as there). Fit per rating point; accuracy in pp.

| Paradigm | Term | Yes/No acc | LP acc | Yes/No margin | LP whole margin |
|---|---|---:|---:|---:|---:|
| passive_1 | good_fit | 0.11 [-1.04, 1.27] | 4.52 [3.49, 5.52] | 0.00 [-0.00, 0.01] | 0.57 [0.47, 0.67] |
| passive_1 | bad_patient | 0.17 [-0.61, 0.95] | -1.37 [-2.15, -0.56] | 0.00 [-0.00, 0.00] | -0.11 [-0.19, -0.04] |
| passive_1 | xtail | -8.88 [-20.87, 3.41] | -4.38 [-13.37, 5.04] | -0.05 [-0.11, 0.01] | -0.25 [-1.10, 0.62] |
| passive_2 | good_fit | -0.19 [-1.40, 1.04] | 4.38 [3.10, 5.59] | -0.00 [-0.01, 0.01] | 0.46 [0.36, 0.55] |
| passive_2 | bad_patient | 0.36 [-0.59, 1.29] | -2.57 [-3.40, -1.79] | 0.00 [-0.00, 0.01] | -0.23 [-0.30, -0.18] |
| passive_2 | xtail | -5.07 [-19.88, 9.22] | -9.34 [-19.06, 0.85] | -0.05 [-0.14, 0.03] | -0.60 [-1.55, 0.24] |
