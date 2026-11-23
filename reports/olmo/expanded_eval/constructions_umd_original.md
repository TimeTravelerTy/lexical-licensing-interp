# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +2.38 [+2.03, +2.74] → **passes**; numerator +1.40 [+1.11, +1.72], PREP -0.98 [-1.15, -0.81], O +1.68 [+1.36, +2.00].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.53 / 0.48 | +1.31 [+1.15, +1.49] RISE | +1.53 [+1.37, +1.69] RISE | +1.10 [+0.97, +1.24] | -0.43 [-0.50, -0.35] | +0.03 [-0.10, +0.18] | +0.80 [+0.70, +0.91] | 0.52 / 0.83 | **mixed** |
| 11 | 0.58 / 0.48 | +1.62 [+1.44, +1.80] RISE | +1.36 [+1.22, +1.49] RISE | +0.95 [+0.83, +1.07] | -0.41 [-0.47, -0.35] | +0.06 [-0.09, +0.21] | +0.51 [+0.44, +0.58] | 0.48 / 0.86 | **mixed** |
| 12 | 0.62 / 0.48 | +2.51 [+2.31, +2.71] RISE | +0.51 [+0.40, +0.62] RISE | -0.04 [-0.11, +0.03] | -0.55 [-0.63, -0.46] | -0.02 [-0.16, +0.12] | +0.11 [+0.06, +0.16] | 0.47 / 0.86 | **mixed** |
| 13 | 0.62 / 0.48 | +2.85 [+2.63, +3.06] RISE | +0.86 [+0.76, +0.96] RISE | -0.13 [-0.21, -0.05] | -0.98 [-1.12, -0.86] | -0.49 [-0.64, -0.33] | +0.27 [+0.23, +0.32] | 0.42 / 0.87 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0810, ' to' -0.0774, ' in' -0.0763, ' is' +0.0374, ' the' +0.0362, ' into' -0.0340

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +1.27 [+0.97, +1.59] → **passes**; numerator +0.43 [+0.09, +0.74], PREP -0.84 [-1.13, -0.58], O +0.08 [-0.23, +0.37].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.53 / 0.25 | +1.14 [+0.99, +1.29] RISE | +0.94 [+0.79, +1.09] RISE | +0.43 [+0.32, +0.54] | -0.51 [-0.58, -0.44] | +0.15 [+0.04, +0.25] | +0.37 [+0.30, +0.44] | 0.38 / 0.83 | **mixed** |
| 11 | 0.58 / 0.25 | +1.40 [+1.24, +1.56] RISE | +0.84 [+0.72, +0.97] RISE | +0.34 [+0.24, +0.45] | -0.50 [-0.57, -0.44] | +0.15 [+0.03, +0.26] | +0.28 [+0.23, +0.34] | 0.35 / 0.86 | **mixed** |
| 12 | 0.62 / 0.25 | +2.44 [+2.24, +2.63] RISE | +0.33 [+0.25, +0.40] RISE | -0.28 [-0.35, -0.22] | -0.61 [-0.71, -0.51] | +0.06 [-0.04, +0.15] | +0.06 [+0.03, +0.08] | 0.33 / 0.86 | **mixed** |
| 13 | 0.62 / 0.25 | +2.85 [+2.64, +3.05] RISE | +0.22 [+0.15, +0.30] unresolved | -0.68 [-0.78, -0.58] | -0.90 [-1.01, -0.79] | -0.08 [-0.18, +0.01] | +0.03 [+0.01, +0.06] | 0.25 / 0.87 | **unresolved** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 9 0.99, site 11 1.00, site 12 1.00, site 13 1.00

Tokens driving the effects (probability change):

- natural good - bad: ',' +0.0535, ' in' -0.0491, '.' +0.0386, ' through' -0.0267, ' around' -0.0240, ' and' +0.0236

