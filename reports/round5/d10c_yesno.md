# Round 5, D10c: FreqBLiMP passive_1 Yes/No readout

Spec: `plan.md`, D10c. Run: `run_yesno.py` (TSUBAME job 9009081, fp32). Analysis: `analyze_yesno.py`. Prompt: task 4's `YES_NO_BASE_PROMPT`; score = LSE(Yes, " Yes") − LSE(No, " No") at the answer position; a pair is correct if the good sentence scores higher (ties half). Items: released `passive_1`, 1,000 per band. CIs: item bootstrap (2,000 draws); sensitivity: good-verb clusters.

| model | band | accuracy [95% CI] | 90% CI | reading | good-verb clusters: 95% CI, reading | margin [95% CI] | median Yes+No mass |
|---|---|---|---|---|---|---|---:|
| 1.4b | head | 0.499 [0.468, 0.530] | [0.473, 0.525] | **practically at chance** | [0.460, 0.542], practically at chance | +0.007 [-0.001, +0.015] | 0.471 |
| 6.9b | head | 0.450 [0.420, 0.481] | [0.425, 0.476] | **below chance** | [0.416, 0.484], below chance | -0.019 [-0.028, -0.010] | 0.675 |
| 12b | head | 0.398 [0.368, 0.428] | [0.373, 0.424] | **below chance** | [0.364, 0.433], below chance | -0.045 [-0.058, -0.032] | 0.687 |
| 1.4b | tail | 0.512 [0.480, 0.544] | [0.486, 0.538] | **practically at chance** | [0.473, 0.551], practically at chance | +0.005 [-0.001, +0.012] | 0.483 |
| 6.9b | tail | 0.460 [0.428, 0.490] | [0.433, 0.486] | **below chance** | [0.421, 0.500], unresolved | -0.019 [-0.028, -0.011] | 0.690 |
| 12b | tail | 0.474 [0.443, 0.505] | [0.448, 0.501] | **unresolved** | [0.442, 0.507], unresolved | -0.012 [-0.023, -0.000] | 0.673 |
| 1.4b | xtail | 0.546 [0.516, 0.576] | [0.520, 0.571] | **above chance** | [0.500, 0.591], unresolved | +0.008 [+0.002, +0.015] | 0.478 |
| 6.9b | xtail | 0.537 [0.505, 0.568] | [0.510, 0.563] | **above chance** | [0.492, 0.587], unresolved | +0.010 [+0.002, +0.018] | 0.700 |
| 12b | xtail | 0.485 [0.454, 0.516] | [0.459, 0.511] | **practically at chance** | [0.446, 0.527], practically at chance | -0.005 [-0.015, +0.005] | 0.670 |

**Model comparisons** (accuracy difference, paired item draws, 95% CI):

| band | 6.9B − 1.4B | 12B − 1.4B | 12B − 6.9B |
|---|---|---|---|
| head | -0.049 [-0.092, -0.006] | -0.101 [-0.142, -0.058] | -0.052 [-0.088, -0.018] |
| tail | -0.052 [-0.097, -0.009] | -0.038 [-0.079, +0.001] | +0.014 [-0.023, +0.051] |
| xtail | -0.009 [-0.051, +0.031] | -0.061 [-0.102, -0.017] | -0.052 [-0.088, -0.015] |

**Descriptive (not declared).** Share of sentences scored Yes > No, and the accuracy CI with the *bad* participle's clusters resampled (Head has only 20 bad verbs):

| model | band | Yes > No: good / bad | bad verbs | bad-verb clusters, 95% CI |
|---|---|---|---:|---|
| 1.4b | head | 0.98 / 0.97 | 20 | [0.401, 0.596] |
| 6.9b | head | 1.00 / 1.00 | 20 | [0.355, 0.543] |
| 12b | head | 0.99 / 0.99 | 20 | [0.315, 0.474] |
| 1.4b | tail | 0.99 / 0.99 | 80 | [0.451, 0.573] |
| 6.9b | tail | 1.00 / 1.00 | 80 | [0.410, 0.507] |
| 12b | tail | 1.00 / 1.00 | 80 | [0.431, 0.519] |
| 1.4b | xtail | 0.99 / 0.99 | 99 | [0.480, 0.611] |
| 6.9b | xtail | 1.00 / 1.00 | 99 | [0.483, 0.590] |
| 12b | xtail | 1.00 / 1.00 | 99 | [0.441, 0.530] |

