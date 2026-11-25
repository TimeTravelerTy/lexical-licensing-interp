# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +2.09 [+1.79, +2.41] → **passes**; numerator +1.54 [+1.27, +1.84], PREP -0.55 [-0.66, -0.44], O +1.15 [+0.86, +1.45].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 5 | 0.54 / 0.42 | +2.32 [+2.01, +2.63] RISE | +1.30 [+1.11, +1.50] RISE | +0.65 [+0.46, +0.85] | -0.65 [-0.79, -0.52] | +0.34 [+0.13, +0.54] | +1.18 [+1.03, +1.35] | 0.66 / 0.69 | **mixed** |
| 8 | 0.53 / 0.42 | +1.71 [+1.50, +1.94] RISE | +1.40 [+1.22, +1.58] RISE | +0.96 [+0.82, +1.11] | -0.44 [-0.51, -0.37] | +0.46 [+0.28, +0.63] | +1.28 [+1.14, +1.43] | 0.65 / 0.76 | **mixed** |
| 12 | 0.56 / 0.42 | +2.41 [+2.19, +2.66] RISE | +1.45 [+1.28, +1.64] RISE | +0.75 [+0.61, +0.91] | -0.70 [-0.80, -0.60] | +0.64 [+0.45, +0.82] | +0.95 [+0.84, +1.07] | 0.54 / 0.83 | **mixed** |
| 15 | 0.58 / 0.42 | +3.04 [+2.79, +3.29] RISE | +2.22 [+2.00, +2.44] RISE | +1.22 [+1.03, +1.43] | -0.99 [-1.12, -0.88] | +0.58 [+0.38, +0.78] | +1.02 [+0.90, +1.15] | 0.42 / 0.84 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0635, ' to' -0.0554, ' over' -0.0392, ' into' -0.0345, ' is' +0.0336, ' with' -0.0290

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +0.86 [+0.65, +1.08] → **passes**; numerator +0.31 [+0.15, +0.47], PREP -0.55 [-0.72, -0.39], O +0.45 [+0.25, +0.66].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 5 | 0.54 / 0.17 | +1.15 [+0.98, +1.32] RISE | +0.77 [+0.65, +0.89] RISE | +0.28 [+0.21, +0.36] | -0.49 [-0.57, -0.42] | +0.25 [+0.11, +0.38] | +0.63 [+0.52, +0.74] | 0.68 / 0.69 | **mixed** |
| 8 | 0.53 / 0.17 | +1.30 [+1.14, +1.46] RISE | +0.75 [+0.65, +0.85] RISE | +0.20 [+0.13, +0.28] | -0.55 [-0.61, -0.49] | +0.08 [-0.01, +0.16] | +0.55 [+0.47, +0.63] | 0.63 / 0.76 | **mixed** |
| 12 | 0.56 / 0.17 | +1.95 [+1.77, +2.15] RISE | +0.89 [+0.79, +0.99] RISE | +0.14 [+0.05, +0.22] | -0.75 [-0.84, -0.67] | +0.13 [+0.05, +0.21] | +0.43 [+0.37, +0.50] | 0.46 / 0.83 | **mixed** |
| 15 | 0.58 / 0.17 | +2.54 [+2.32, +2.77] RISE | +1.36 [+1.23, +1.49] RISE | +0.33 [+0.21, +0.44] | -1.03 [-1.14, -0.93] | -0.05 [-0.14, +0.04] | +0.44 [+0.38, +0.50] | 0.28 / 0.84 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 5 0.97, site 8 0.98, site 12 0.99, site 15 0.99

Tokens driving the effects (probability change):

- natural good - bad: '.' +0.0522, ' in' -0.0427, ',' +0.0335, ' around' -0.0304, ' through' -0.0144, ' because' +0.0141

