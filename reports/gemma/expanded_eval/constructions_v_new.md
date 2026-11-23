# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +2.17 [+1.82, +2.51] → **passes**; numerator +1.63 [+1.35, +1.91], PREP -0.54 [-0.65, -0.44], O +1.18 [+0.87, +1.46].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 5 | 0.54 / 0.43 | +2.34 [+2.02, +2.65] RISE | +1.48 [+1.29, +1.67] RISE | +0.75 [+0.55, +0.94] | -0.74 [-0.89, -0.60] | +0.28 [+0.06, +0.51] | +1.17 [+1.01, +1.33] | 0.61 / 0.64 | **mixed** |
| 8 | 0.53 / 0.43 | +1.73 [+1.52, +1.96] RISE | +1.57 [+1.38, +1.76] RISE | +1.09 [+0.93, +1.24] | -0.48 [-0.56, -0.41] | +0.40 [+0.19, +0.60] | +1.32 [+1.17, +1.47] | 0.64 / 0.75 | **mixed** |
| 12 | 0.56 / 0.43 | +2.46 [+2.21, +2.70] RISE | +1.65 [+1.46, +1.84] RISE | +0.91 [+0.76, +1.06] | -0.74 [-0.85, -0.64] | +0.58 [+0.36, +0.80] | +0.96 [+0.84, +1.08] | 0.53 / 0.81 | **mixed** |
| 15 | 0.58 / 0.43 | +3.13 [+2.86, +3.38] RISE | +2.37 [+2.13, +2.60] RISE | +1.36 [+1.16, +1.57] | -1.01 [-1.14, -0.89] | +0.50 [+0.26, +0.73] | +1.01 [+0.89, +1.14] | 0.40 / 0.80 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0620, ' with' -0.0519, ' on' -0.0419, ' into' -0.0408, ' over' -0.0404, ' is' +0.0350

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +1.07 [+0.79, +1.37] → **passes**; numerator +0.45 [+0.28, +0.62], PREP -0.63 [-0.78, -0.47], O +0.60 [+0.37, +0.85].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 5 | 0.54 / 0.21 | +1.19 [+1.02, +1.38] RISE | +0.84 [+0.69, +1.00] RISE | +0.33 [+0.24, +0.43] | -0.51 [-0.60, -0.43] | +0.25 [+0.13, +0.38] | +0.58 [+0.47, +0.70] | 0.57 / 0.64 | **mixed** |
| 8 | 0.53 / 0.21 | +1.31 [+1.16, +1.46] RISE | +0.83 [+0.71, +0.97] RISE | +0.29 [+0.20, +0.38] | -0.55 [-0.62, -0.47] | +0.08 [-0.02, +0.18] | +0.51 [+0.43, +0.60] | 0.56 / 0.75 | **mixed** |
| 12 | 0.56 / 0.21 | +1.96 [+1.77, +2.15] RISE | +0.97 [+0.84, +1.10] RISE | +0.25 [+0.16, +0.36] | -0.72 [-0.80, -0.63] | +0.19 [+0.09, +0.29] | +0.40 [+0.34, +0.46] | 0.41 / 0.81 | **mixed** |
| 15 | 0.58 / 0.21 | +2.57 [+2.35, +2.80] RISE | +1.48 [+1.32, +1.64] RISE | +0.48 [+0.35, +0.63] | -1.00 [-1.10, -0.89] | +0.04 [-0.07, +0.16] | +0.44 [+0.37, +0.51] | 0.27 / 0.80 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 5 0.97, site 8 0.99, site 12 1.00, site 15 1.00

Tokens driving the effects (probability change):

- natural good - bad: '.' +0.0593, ',' +0.0506, ' in' -0.0366, ' around' -0.0240, ' through' -0.0228, ' into' -0.0225

