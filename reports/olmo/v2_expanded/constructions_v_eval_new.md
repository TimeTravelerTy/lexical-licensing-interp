# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +3.13 [+2.68, +3.61] → **passes**; numerator +1.92 [+1.62, +2.25], PREP -1.21 [-1.44, -0.99], O +1.43 [+1.07, +1.82].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.53 / 0.63 | +1.86 [+1.61, +2.14] RISE | +1.94 [+1.77, +2.11] RISE | +1.32 [+1.18, +1.48] | -0.61 [-0.74, -0.50] | -0.04 [-0.26, +0.16] | +1.24 [+1.09, +1.38] | 0.62 / 0.87 | **mixed** |
| 11 | 0.58 / 0.63 | +2.67 [+2.42, +2.93] RISE | +1.07 [+0.87, +1.26] RISE | +0.41 [+0.22, +0.61] | -0.66 [-0.79, -0.54] | -0.43 [-0.59, -0.27] | +0.40 [+0.28, +0.51] | 0.55 / 0.88 | **mixed** |
| 12 | 0.62 / 0.63 | +3.39 [+3.12, +3.69] RISE | +0.14 [-0.02, +0.31] NO RISE | -1.03 [-1.21, -0.86] | -1.17 [-1.41, -0.97] | -1.04 [-1.30, -0.81] | -0.14 [-0.23, -0.07] | 0.46 / 0.87 | **surface** |
| 13 | 0.62 / 0.63 | +3.68 [+3.37, +4.00] RISE | +0.92 [+0.77, +1.08] RISE | -0.87 [-1.05, -0.70] | -1.79 [-2.07, -1.54] | -1.30 [-1.55, -1.05] | +0.20 [+0.15, +0.26] | 0.39 / 0.89 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.1116, ' in' -0.0779, ' with' -0.0663, ' is' +0.0592, ' to' -0.0496, ' into' -0.0491

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +2.12 [+1.65, +2.63] → **passes**; numerator +0.97 [+0.70, +1.28], PREP -1.15 [-1.40, -0.92], O -0.02 [-0.32, +0.29].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.53 / 0.42 | +1.85 [+1.64, +2.06] RISE | +1.61 [+1.40, +1.85] RISE | +0.70 [+0.54, +0.89] | -0.90 [-1.04, -0.77] | -0.02 [-0.23, +0.19] | +0.64 [+0.54, +0.74] | 0.39 / 0.87 | **mixed** |
| 11 | 0.58 / 0.42 | +2.46 [+2.25, +2.68] RISE | +1.15 [+1.01, +1.31] RISE | +0.35 [+0.19, +0.51] | -0.81 [-0.93, -0.69] | -0.24 [-0.45, -0.05] | +0.35 [+0.29, +0.41] | 0.35 / 0.88 | **mixed** |
| 12 | 0.62 / 0.42 | +3.70 [+3.45, +3.96] RISE | +0.18 [+0.06, +0.29] NO RISE | -1.03 [-1.19, -0.87] | -1.21 [-1.42, -1.00] | -0.93 [-1.13, -0.72] | -0.04 [-0.07, -0.01] | 0.30 / 0.87 | **surface** |
| 13 | 0.62 / 0.42 | +3.72 [+3.42, +4.01] RISE | +0.55 [+0.41, +0.70] RISE | -1.45 [-1.67, -1.24] | -2.00 [-2.26, -1.76] | -1.12 [-1.33, -0.91] | +0.03 [-0.00, +0.06] | 0.21 / 0.89 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 9 1.00, site 11 1.00, site 12 1.00, site 13 1.00

Tokens driving the effects (probability change):

- natural good - bad: ',' +0.1082, '.' +0.0694, ' in' -0.0692, ' with' -0.0498, ' around' -0.0496, ' on' -0.0491

