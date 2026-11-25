# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +2.64 [+2.36, +2.94] → **passes**; numerator +1.58 [+1.34, +1.81], PREP -1.06 [-1.21, -0.93], O +1.59 [+1.33, +1.87].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.53 / 0.53 | +1.78 [+1.60, +1.98] RISE | +1.94 [+1.78, +2.10] RISE | +1.31 [+1.19, +1.45] | -0.62 [-0.71, -0.54] | -0.00 [-0.13, +0.12] | +1.21 [+1.10, +1.33] | 0.58 / 0.86 | **mixed** |
| 11 | 0.58 / 0.53 | +2.59 [+2.37, +2.83] RISE | +1.17 [+1.05, +1.30] RISE | +0.52 [+0.40, +0.63] | -0.65 [-0.75, -0.57] | -0.30 [-0.41, -0.20] | +0.46 [+0.38, +0.53] | 0.53 / 0.88 | **mixed** |
| 12 | 0.62 / 0.53 | +3.41 [+3.17, +3.66] RISE | +0.24 [+0.13, +0.37] NO RISE | -1.04 [-1.18, -0.90] | -1.29 [-1.49, -1.10] | -1.11 [-1.31, -0.92] | -0.10 [-0.16, -0.05] | 0.46 / 0.87 | **surface** |
| 13 | 0.62 / 0.53 | +3.64 [+3.37, +3.90] RISE | +1.04 [+0.92, +1.17] RISE | -0.97 [-1.11, -0.84] | -2.01 [-2.25, -1.80] | -1.44 [-1.64, -1.25] | +0.26 [+0.22, +0.31] | 0.41 / 0.89 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0913, ' in' -0.0768, ' to' -0.0680, ' is' +0.0448, ' into' -0.0391, ' with' -0.0357

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +1.56 [+1.28, +1.85] → **passes**; numerator +0.62 [+0.36, +0.86], PREP -0.94 [-1.16, -0.75], O +0.04 [-0.22, +0.30].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.53 / 0.31 | +1.67 [+1.51, +1.84] RISE | +1.35 [+1.19, +1.53] RISE | +0.57 [+0.46, +0.70] | -0.77 [-0.87, -0.69] | +0.06 [-0.06, +0.17] | +0.59 [+0.51, +0.67] | 0.40 / 0.86 | **mixed** |
| 11 | 0.58 / 0.31 | +2.28 [+2.10, +2.47] RISE | +1.02 [+0.90, +1.14] RISE | +0.31 [+0.21, +0.41] | -0.71 [-0.79, -0.64] | -0.06 [-0.19, +0.06] | +0.34 [+0.30, +0.39] | 0.36 / 0.88 | **mixed** |
| 12 | 0.62 / 0.31 | +3.54 [+3.30, +3.76] RISE | +0.20 [+0.14, +0.27] NO RISE | -1.04 [-1.17, -0.91] | -1.24 [-1.41, -1.08] | -0.87 [-1.03, -0.72] | -0.01 [-0.03, +0.01] | 0.31 / 0.87 | **surface** |
| 13 | 0.62 / 0.31 | +3.57 [+3.32, +3.83] RISE | +0.44 [+0.34, +0.54] RISE | -1.62 [-1.79, -1.45] | -2.06 [-2.27, -1.86] | -1.21 [-1.38, -1.04] | +0.05 [+0.02, +0.07] | 0.23 / 0.89 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 9 1.00, site 11 1.00, site 12 1.00, site 13 1.00

Tokens driving the effects (probability change):

- natural good - bad: ',' +0.0719, ' in' -0.0559, '.' +0.0490, ' around' -0.0326, ' and' +0.0313, ' on' -0.0277

