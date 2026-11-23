# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +2.17 [+1.82, +2.51] → **passes**; numerator +1.63 [+1.35, +1.91], PREP -0.54 [-0.65, -0.44], O +1.18 [+0.87, +1.46].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 5 | 0.54 / 0.43 | +0.73 [+0.41, +1.06] RISE | +0.99 [+0.63, +1.36] RISE | +0.73 [+0.45, +1.02] | -0.26 [-0.36, -0.17] | +0.27 [+0.14, +0.43] | +0.58 [+0.38, +0.79] | 0.52 / 0.78 | **mixed** |
| 8 | 0.53 / 0.43 | +0.83 [+0.61, +1.06] RISE | +0.93 [+0.70, +1.17] RISE | +0.71 [+0.52, +0.91] | -0.22 [-0.28, -0.17] | +0.24 [+0.12, +0.36] | +1.22 [+1.03, +1.42] | 1.00 / 0.71 | **mixed** |
| 12 | 0.56 / 0.43 | +1.15 [+0.90, +1.40] RISE | +0.99 [+0.76, +1.22] RISE | +0.75 [+0.58, +0.94] | -0.23 [-0.30, -0.17] | +0.36 [+0.22, +0.51] | +1.01 [+0.86, +1.16] | 0.93 / 0.86 | **mixed** |
| 15 | 0.58 / 0.43 | +1.54 [+1.32, +1.77] RISE | +1.93 [+1.64, +2.22] RISE | +1.52 [+1.29, +1.76] | -0.41 [-0.50, -0.33] | +0.48 [+0.31, +0.66] | +0.76 [+0.61, +0.91] | 0.35 / 0.91 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0620, ' with' -0.0519, ' on' -0.0419, ' into' -0.0408, ' over' -0.0404, ' is' +0.0350

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +1.07 [+0.79, +1.37] → **passes**; numerator +0.45 [+0.28, +0.62], PREP -0.63 [-0.78, -0.47], O +0.60 [+0.37, +0.85].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 5 | 0.54 / 0.21 | +0.53 [+0.33, +0.73] unresolved | +0.43 [+0.26, +0.59] RISE | +0.16 [+0.09, +0.25] | -0.26 [-0.35, -0.17] | +0.12 [+0.05, +0.19] | +0.30 [+0.17, +0.43] | 0.34 / 0.78 | **unresolved (gate)** |
| 8 | 0.53 / 0.21 | +0.69 [+0.52, +0.84] RISE | +0.44 [+0.33, +0.56] RISE | +0.16 [+0.10, +0.23] | -0.28 [-0.35, -0.21] | +0.07 [+0.01, +0.14] | +0.38 [+0.28, +0.48] | 0.68 / 0.71 | **unresolved (gate)** |
| 12 | 0.56 / 0.21 | +1.00 [+0.77, +1.24] RISE | +0.42 [+0.31, +0.54] RISE | +0.13 [+0.08, +0.19] | -0.29 [-0.37, -0.22] | +0.16 [+0.09, +0.24] | +0.29 [+0.22, +0.36] | 0.64 / 0.86 | **mixed** |
| 15 | 0.58 / 0.21 | +1.46 [+1.25, +1.68] RISE | +0.92 [+0.77, +1.08] RISE | +0.35 [+0.26, +0.47] | -0.57 [-0.66, -0.47] | +0.09 [-0.00, +0.18] | +0.22 [+0.13, +0.31] | 0.16 / 0.91 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 5 0.65 (fails), site 8 0.73 (fails), site 12 0.84, site 15 0.96

Tokens driving the effects (probability change):

- natural good - bad: '.' +0.0593, ',' +0.0506, ' in' -0.0366, ' around' -0.0240, ' through' -0.0228, ' into' -0.0225

