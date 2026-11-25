# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +2.64 [+2.36, +2.94] → **passes**; numerator +1.58 [+1.34, +1.81], PREP -1.06 [-1.21, -0.93], O +1.59 [+1.33, +1.87].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.54 / 0.53 | +1.83 [+1.65, +2.04] RISE | +2.06 [+1.88, +2.24] RISE | +1.40 [+1.27, +1.55] | -0.66 [-0.75, -0.57] | +0.07 [-0.08, +0.21] | +1.47 [+1.33, +1.62] | 0.66 / 0.94 | **mixed** |
| 11 | 0.59 / 0.53 | +2.83 [+2.61, +3.05] RISE | +1.31 [+1.19, +1.43] RISE | +0.53 [+0.42, +0.64] | -0.78 [-0.88, -0.68] | -0.32 [-0.44, -0.20] | +0.56 [+0.49, +0.64] | 0.57 / 0.93 | **mixed** |
| 12 | 0.63 / 0.53 | +3.53 [+3.30, +3.76] RISE | +0.69 [+0.55, +0.83] RISE | -1.01 [-1.16, -0.88] | -1.70 [-1.92, -1.50] | -1.42 [-1.63, -1.22] | +0.03 [-0.02, +0.09] | 0.50 / 0.93 | **mixed** |
| 13 | 0.67 / 0.53 | +3.88 [+3.62, +4.13] RISE | +1.17 [+1.05, +1.30] RISE | -1.81 [-2.01, -1.61] | -2.98 [-3.26, -2.68] | -2.60 [-2.86, -2.33] | +0.29 [+0.25, +0.35] | 0.44 / 0.94 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0913, ' in' -0.0768, ' to' -0.0680, ' is' +0.0448, ' into' -0.0391, ' with' -0.0357

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +1.56 [+1.28, +1.85] → **passes**; numerator +0.62 [+0.36, +0.86], PREP -0.94 [-1.16, -0.75], O +0.04 [-0.22, +0.30].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.54 / 0.31 | +1.69 [+1.54, +1.85] RISE | +1.46 [+1.28, +1.66] RISE | +0.65 [+0.53, +0.79] | -0.81 [-0.91, -0.72] | +0.17 [+0.04, +0.30] | +0.76 [+0.65, +0.87] | 0.45 / 0.94 | **mixed** |
| 11 | 0.59 / 0.31 | +2.50 [+2.32, +2.68] RISE | +1.10 [+0.98, +1.23] RISE | +0.33 [+0.23, +0.43] | -0.77 [-0.86, -0.69] | +0.01 [-0.12, +0.13] | +0.39 [+0.34, +0.45] | 0.37 / 0.93 | **mixed** |
| 12 | 0.63 / 0.31 | +3.72 [+3.48, +3.93] RISE | +0.53 [+0.45, +0.60] RISE | -0.95 [-1.07, -0.83] | -1.47 [-1.65, -1.31] | -0.89 [-1.04, -0.74] | +0.07 [+0.05, +0.09] | 0.32 / 0.93 | **mixed** |
| 13 | 0.67 / 0.31 | +3.87 [+3.61, +4.11] RISE | +0.61 [+0.53, +0.70] RISE | -2.17 [-2.38, -1.95] | -2.78 [-3.02, -2.52] | -2.08 [-2.30, -1.84] | +0.08 [+0.06, +0.10] | 0.23 / 0.94 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 9 1.00, site 11 1.00, site 12 1.00, site 13 1.00

Tokens driving the effects (probability change):

- natural good - bad: ',' +0.0719, ' in' -0.0559, '.' +0.0490, ' around' -0.0326, ' and' +0.0313, ' on' -0.0277

