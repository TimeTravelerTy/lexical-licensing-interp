# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +2.62 [+2.28, +2.96] → **passes**; numerator +1.75 [+1.48, +2.01], PREP -0.87 [-1.01, -0.74], O +1.16 [+0.92, +1.41].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 8 | 0.49 / 0.52 | +1.53 [+1.34, +1.73] RISE | +2.27 [+2.06, +2.48] RISE | +1.62 [+1.44, +1.81] | -0.64 [-0.73, -0.56] | +0.36 [+0.18, +0.55] | +1.74 [+1.57, +1.91] | 0.70 / 0.78 | **mixed** |
| 12 | 0.53 / 0.52 | +2.59 [+2.39, +2.81] RISE | +1.11 [+0.99, +1.23] RISE | +0.44 [+0.33, +0.55] | -0.67 [-0.76, -0.58] | +0.18 [+0.05, +0.30] | +0.74 [+0.66, +0.83] | 0.66 / 0.81 | **mixed** |
| 17 | 0.57 / 0.52 | +3.51 [+3.25, +3.77] RISE | +1.19 [+1.03, +1.35] RISE | -1.00 [-1.15, -0.84] | -2.18 [-2.46, -1.92] | -1.88 [-2.12, -1.64] | +0.29 [+0.24, +0.35] | 0.40 / 0.85 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0722, ' with' -0.0662, ' in' -0.0653, ' into' -0.0466, ' over' -0.0369, ',' +0.0353

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +2.09 [+1.75, +2.41] → **passes**; numerator +0.98 [+0.76, +1.21], PREP -1.10 [-1.28, -0.93], O +0.36 [+0.10, +0.63].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 8 | 0.49 / 0.42 | +1.02 [+0.88, +1.17] RISE | +1.85 [+1.63, +2.06] RISE | +0.95 [+0.80, +1.12] | -0.89 [-1.01, -0.78] | +0.15 [-0.03, +0.33] | +1.12 [+0.99, +1.26] | 0.51 / 0.78 | **mixed** |
| 12 | 0.53 / 0.42 | +2.14 [+1.94, +2.32] RISE | +1.09 [+0.97, +1.23] RISE | +0.46 [+0.37, +0.57] | -0.63 [-0.71, -0.56] | +0.24 [+0.13, +0.35] | +0.56 [+0.49, +0.63] | 0.47 / 0.81 | **mixed** |
| 17 | 0.57 / 0.42 | +4.13 [+3.82, +4.42] RISE | +0.35 [+0.25, +0.46] unresolved | -1.17 [-1.34, -1.01] | -1.52 [-1.74, -1.33] | -1.31 [-1.50, -1.12] | +0.02 [+0.00, +0.04] | 0.22 / 0.85 | **unresolved** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 8 1.00, site 12 1.00, site 17 1.00

Tokens driving the effects (probability change):

- natural good - bad: ',' +0.1022, '.' +0.0873, ' with' -0.0558, ' in' -0.0492, ' around' -0.0373, ' through' -0.0334

