# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +1.75 [+1.41, +2.12] → **passes**; numerator +1.25 [+0.94, +1.60], PREP -0.50 [-0.64, -0.36], O +1.14 [+0.79, +1.51].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 5 | 0.54 / 0.35 | +0.68 [+0.37, +1.01] RISE | +0.90 [+0.55, +1.28] RISE | +0.69 [+0.41, +0.98] | -0.22 [-0.31, -0.13] | +0.38 [+0.23, +0.57] | +0.58 [+0.34, +0.81] | 0.50 / 0.74 | **mixed** |
| 8 | 0.53 / 0.35 | +0.77 [+0.54, +1.00] RISE | +0.81 [+0.58, +1.05] RISE | +0.61 [+0.42, +0.82] | -0.20 [-0.26, -0.14] | +0.29 [+0.19, +0.41] | +1.18 [+0.96, +1.44] | 0.98 / 0.66 | **mixed** |
| 12 | 0.56 / 0.35 | +1.09 [+0.83, +1.35] RISE | +0.98 [+0.75, +1.21] RISE | +0.73 [+0.56, +0.91] | -0.25 [-0.32, -0.18] | +0.47 [+0.32, +0.63] | +0.96 [+0.79, +1.14] | 0.91 / 0.88 | **mixed** |
| 15 | 0.58 / 0.35 | +1.44 [+1.22, +1.65] RISE | +1.83 [+1.55, +2.12] RISE | +1.42 [+1.19, +1.67] | -0.41 [-0.49, -0.32] | +0.59 [+0.43, +0.77] | +0.75 [+0.59, +0.92] | 0.36 / 0.95 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' to' -0.0695, ' was' +0.0538, ' over' -0.0410, ' into' -0.0300, ' is' +0.0272, ' in' -0.0240

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +0.63 [+0.41, +0.87] → **passes**; numerator +0.19 [-0.01, +0.39], PREP -0.44 [-0.67, -0.23], O +0.46 [+0.19, +0.71].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 5 | 0.54 / 0.13 | +0.56 [+0.35, +0.76] RISE | +0.40 [+0.25, +0.56] RISE | +0.16 [+0.09, +0.24] | -0.25 [-0.33, -0.15] | +0.16 [+0.09, +0.25] | +0.22 [+0.11, +0.34] | 0.36 / 0.74 | **unresolved (gate)** |
| 8 | 0.53 / 0.13 | +0.68 [+0.50, +0.85] RISE | +0.39 [+0.28, +0.50] RISE | +0.08 [+0.02, +0.15] | -0.31 [-0.39, -0.23] | +0.13 [+0.06, +0.20] | +0.63 [+0.51, +0.77] | 1.33 / 0.66 | **mixed** |
| 12 | 0.56 / 0.13 | +1.04 [+0.80, +1.27] RISE | +0.42 [+0.32, +0.52] RISE | +0.06 [+0.00, +0.12] | -0.35 [-0.45, -0.27] | +0.23 [+0.15, +0.31] | +0.37 [+0.29, +0.46] | 0.92 / 0.88 | **mixed** |
| 15 | 0.58 / 0.13 | +1.42 [+1.20, +1.64] RISE | +0.76 [+0.63, +0.89] RISE | +0.24 [+0.15, +0.33] | -0.52 [-0.61, -0.43] | +0.11 [+0.02, +0.19] | +0.11 [+0.03, +0.19] | 0.09 / 0.95 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 5 0.72 (fails), site 8 0.83, site 12 0.87, site 15 0.91

Tokens driving the effects (probability change):

- natural good - bad: ' in' -0.0420, '.' +0.0403, ' around' -0.0204, ',' +0.0164, ' through' -0.0123, ' the' +0.0111

