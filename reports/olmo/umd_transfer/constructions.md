# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +2.42 [+2.10, +2.76] → **passes**; numerator +1.43 [+1.16, +1.72], PREP -0.99 [-1.15, -0.82], O +1.66 [+1.36, +1.98].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.53 / 0.48 | +1.34 [+1.17, +1.52] RISE | +1.55 [+1.39, +1.71] RISE | +1.12 [+0.99, +1.25] | -0.43 [-0.51, -0.36] | +0.06 [-0.08, +0.19] | +0.82 [+0.73, +0.93] | 0.53 / 0.84 | **mixed** |
| 11 | 0.58 / 0.48 | +1.66 [+1.48, +1.85] RISE | +1.36 [+1.23, +1.49] RISE | +0.95 [+0.84, +1.07] | -0.41 [-0.47, -0.35] | +0.07 [-0.08, +0.21] | +0.50 [+0.44, +0.58] | 0.49 / 0.87 | **mixed** |
| 12 | 0.62 / 0.48 | +2.52 [+2.33, +2.73] RISE | +0.50 [+0.40, +0.60] RISE | -0.04 [-0.11, +0.02] | -0.54 [-0.63, -0.46] | -0.01 [-0.14, +0.12] | +0.10 [+0.05, +0.15] | 0.48 / 0.88 | **mixed** |
| 13 | 0.62 / 0.48 | +2.87 [+2.63, +3.09] RISE | +0.86 [+0.77, +0.96] RISE | -0.12 [-0.19, -0.05] | -0.98 [-1.10, -0.86] | -0.46 [-0.62, -0.32] | +0.27 [+0.23, +0.32] | 0.42 / 0.89 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0822, ' to' -0.0775, ' in' -0.0741, ' is' +0.0390, ' the' +0.0361, ' into' -0.0307
- site 9 T - I: ' was' +0.0555, ' in' -0.0450, ' to' -0.0341, ' is' +0.0244, ' with' -0.0197, ' the' +0.0175
- site 11 T - I: ' in' -0.0482, ' was' +0.0408, ' the' +0.0300, ' to' -0.0265, ' is' +0.0235, ' on' -0.0169
- site 12 T - I: ' the' +0.0806, ' in' -0.0730, ' his' +0.0369, ' her' +0.0301, ' to' -0.0269, ' on' -0.0184
- site 13 T - I: ' the' +0.1362, ' in' -0.0913, ' his' +0.0468, ' her' +0.0389, ' to' -0.0354, ' with' -0.0338

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +1.31 [+1.03, +1.60] → **passes**; numerator +0.47 [+0.14, +0.77], PREP -0.85 [-1.10, -0.61], O +0.06 [-0.24, +0.36].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.53 / 0.26 | +1.17 [+1.03, +1.32] RISE | +0.97 [+0.83, +1.12] RISE | +0.44 [+0.34, +0.55] | -0.53 [-0.60, -0.46] | +0.15 [+0.05, +0.25] | +0.39 [+0.32, +0.47] | 0.38 / 0.84 | **mixed** |
| 11 | 0.58 / 0.26 | +1.43 [+1.28, +1.60] RISE | +0.86 [+0.74, +0.99] RISE | +0.35 [+0.26, +0.46] | -0.51 [-0.58, -0.45] | +0.15 [+0.03, +0.26] | +0.29 [+0.23, +0.34] | 0.35 / 0.87 | **mixed** |
| 12 | 0.62 / 0.26 | +2.46 [+2.27, +2.65] RISE | +0.33 [+0.26, +0.39] RISE | -0.28 [-0.34, -0.21] | -0.60 [-0.70, -0.51] | +0.06 [-0.04, +0.15] | +0.05 [+0.03, +0.08] | 0.33 / 0.88 | **mixed** |
| 13 | 0.62 / 0.26 | +2.86 [+2.65, +3.06] RISE | +0.24 [+0.16, +0.31] unresolved | -0.66 [-0.77, -0.56] | -0.90 [-1.01, -0.79] | -0.07 [-0.17, +0.02] | +0.03 [+0.01, +0.06] | 0.25 / 0.89 | **unresolved** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 9 0.99, site 11 1.00, site 12 1.00, site 13 1.00

Tokens driving the effects (probability change):

- natural good - bad: ',' +0.0586, ' in' -0.0490, '.' +0.0400, ' through' -0.0267, ' and' +0.0261, ' around' -0.0241
- site 9 T - I: ',' +0.0430, ' in' -0.0408, '.' +0.0223, ' on' -0.0208, ' and' +0.0189, ' the' +0.0184
- site 11 T - I: ' in' -0.0453, ',' +0.0332, ' the' +0.0225, ' and' +0.0220, ' on' -0.0168, ' at' -0.0151
- site 12 T - I: ' the' +0.0630, ' in' -0.0565, ' a' +0.0217, ',' -0.0203, '.' -0.0179, ' on' -0.0144
- site 13 T - I: ' the' +0.1117, ' in' -0.0642, ',' -0.0404, '.' -0.0362, ' a' +0.0301, ' on' -0.0257

