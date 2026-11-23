# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +2.38 [+2.03, +2.74] → **passes**; numerator +1.40 [+1.11, +1.72], PREP -0.98 [-1.15, -0.81], O +1.68 [+1.36, +2.00].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.53 / 0.48 | +1.75 [+1.56, +1.95] RISE | +1.93 [+1.75, +2.12] RISE | +1.31 [+1.15, +1.47] | -0.62 [-0.72, -0.53] | +0.02 [-0.13, +0.17] | +1.20 [+1.06, +1.34] | 0.57 / 0.86 | **mixed** |
| 11 | 0.58 / 0.48 | +2.56 [+2.32, +2.81] RISE | +1.23 [+1.09, +1.35] RISE | +0.57 [+0.45, +0.69] | -0.65 [-0.76, -0.56] | -0.24 [-0.37, -0.11] | +0.49 [+0.41, +0.57] | 0.52 / 0.88 | **mixed** |
| 12 | 0.62 / 0.48 | +3.42 [+3.17, +3.68] RISE | +0.29 [+0.16, +0.42] NO RISE | -1.05 [-1.20, -0.89] | -1.34 [-1.56, -1.14] | -1.14 [-1.36, -0.93] | -0.08 [-0.14, -0.01] | 0.47 / 0.88 | **surface** |
| 13 | 0.62 / 0.48 | +3.61 [+3.34, +3.88] RISE | +1.10 [+0.96, +1.24] RISE | -1.03 [-1.18, -0.88] | -2.13 [-2.39, -1.88] | -1.51 [-1.72, -1.30] | +0.30 [+0.24, +0.35] | 0.42 / 0.89 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0810, ' to' -0.0774, ' in' -0.0763, ' is' +0.0374, ' the' +0.0362, ' into' -0.0340

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +1.27 [+0.97, +1.59] → **passes**; numerator +0.43 [+0.09, +0.74], PREP -0.84 [-1.13, -0.58], O +0.08 [-0.23, +0.37].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.53 / 0.25 | +1.58 [+1.41, +1.75] RISE | +1.22 [+1.03, +1.41] RISE | +0.51 [+0.38, +0.66] | -0.71 [-0.80, -0.62] | +0.10 [-0.02, +0.21] | +0.56 [+0.46, +0.67] | 0.40 / 0.86 | **mixed** |
| 11 | 0.58 / 0.25 | +2.18 [+1.97, +2.40] RISE | +0.95 [+0.82, +1.08] RISE | +0.29 [+0.17, +0.41] | -0.66 [-0.74, -0.58] | +0.03 [-0.10, +0.14] | +0.34 [+0.29, +0.41] | 0.36 / 0.88 | **mixed** |
| 12 | 0.62 / 0.25 | +3.45 [+3.20, +3.69] RISE | +0.21 [+0.14, +0.28] unresolved | -1.05 [-1.20, -0.91] | -1.26 [-1.44, -1.09] | -0.84 [-1.01, -0.69] | -0.00 [-0.02, +0.02] | 0.32 / 0.88 | **unresolved** |
| 13 | 0.62 / 0.25 | +3.50 [+3.23, +3.75] RISE | +0.39 [+0.26, +0.49] RISE | -1.71 [-1.89, -1.53] | -2.09 [-2.31, -1.88] | -1.25 [-1.44, -1.08] | +0.05 [+0.02, +0.08] | 0.25 / 0.89 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 9 1.00, site 11 1.00, site 12 1.00, site 13 1.00

Tokens driving the effects (probability change):

- natural good - bad: ',' +0.0535, ' in' -0.0491, '.' +0.0386, ' through' -0.0267, ' around' -0.0240, ' and' +0.0236

