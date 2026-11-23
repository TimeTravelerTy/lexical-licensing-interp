# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +2.62 [+2.28, +2.96] → **passes**; numerator +1.75 [+1.48, +2.01], PREP -0.87 [-1.01, -0.74], O +1.16 [+0.92, +1.41].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 8 | 0.49 / 0.52 | +0.97 [+0.82, +1.14] RISE | +1.84 [+1.67, +2.04] RISE | +1.41 [+1.26, +1.58] | -0.43 [-0.50, -0.37] | +0.40 [+0.25, +0.55] | +1.26 [+1.13, +1.39] | 0.64 / 0.75 | **mixed** |
| 12 | 0.53 / 0.52 | +1.33 [+1.19, +1.49] RISE | +1.25 [+1.12, +1.39] RISE | +0.94 [+0.83, +1.05] | -0.31 [-0.36, -0.26] | +0.48 [+0.36, +0.61] | +0.78 [+0.71, +0.86] | 0.62 / 0.80 | **mixed** |
| 17 | 0.57 / 0.52 | +2.68 [+2.44, +2.91] RISE | +1.12 [+0.99, +1.27] RISE | +0.25 [+0.16, +0.35] | -0.87 [-1.00, -0.75] | +0.03 [-0.10, +0.15] | +0.54 [+0.47, +0.62] | 0.53 / 0.87 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0722, ' with' -0.0662, ' in' -0.0653, ' into' -0.0466, ' over' -0.0369, ',' +0.0353

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +2.09 [+1.75, +2.41] → **passes**; numerator +0.98 [+0.76, +1.21], PREP -1.10 [-1.28, -0.93], O +0.36 [+0.10, +0.63].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 8 | 0.49 / 0.42 | +0.60 [+0.48, +0.72] RISE | +1.46 [+1.29, +1.64] RISE | +0.81 [+0.68, +0.95] | -0.66 [-0.76, -0.56] | +0.22 [+0.08, +0.36] | +0.71 [+0.61, +0.81] | 0.44 / 0.75 | **mixed** |
| 12 | 0.53 / 0.42 | +0.91 [+0.78, +1.04] RISE | +1.17 [+1.05, +1.31] RISE | +0.66 [+0.55, +0.77] | -0.52 [-0.60, -0.44] | +0.37 [+0.26, +0.48] | +0.53 [+0.46, +0.59] | 0.43 / 0.80 | **mixed** |
| 17 | 0.57 / 0.42 | +3.07 [+2.80, +3.32] RISE | +0.52 [+0.44, +0.61] RISE | -0.06 [-0.13, +0.01] | -0.58 [-0.67, -0.49] | +0.15 [+0.07, +0.24] | +0.15 [+0.12, +0.18] | 0.31 / 0.87 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 8 0.99, site 12 1.00, site 17 0.99

Tokens driving the effects (probability change):

- natural good - bad: ',' +0.1022, '.' +0.0873, ' with' -0.0558, ' in' -0.0492, ' around' -0.0373, ' through' -0.0334

