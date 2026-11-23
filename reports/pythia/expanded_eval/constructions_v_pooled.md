# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +2.45 [+2.21, +2.71] → **passes**; numerator +1.60 [+1.40, +1.82], PREP -0.85 [-0.95, -0.74], O +1.32 [+1.11, +1.54].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 8 | 0.49 / 0.49 | +1.47 [+1.30, +1.64] RISE | +2.19 [+1.99, +2.38] RISE | +1.55 [+1.39, +1.72] | -0.64 [-0.71, -0.57] | +0.37 [+0.23, +0.52] | +1.75 [+1.60, +1.90] | 0.71 / 0.81 | **mixed** |
| 12 | 0.53 / 0.49 | +2.61 [+2.41, +2.81] RISE | +1.13 [+1.01, +1.25] RISE | +0.43 [+0.33, +0.54] | -0.70 [-0.80, -0.61] | +0.17 [+0.07, +0.28] | +0.79 [+0.71, +0.87] | 0.68 / 0.84 | **mixed** |
| 17 | 0.57 / 0.49 | +3.54 [+3.28, +3.79] RISE | +1.22 [+1.08, +1.36] RISE | -1.05 [-1.20, -0.92] | -2.27 [-2.53, -2.01] | -1.97 [-2.19, -1.75] | +0.32 [+0.27, +0.37] | 0.42 / 0.87 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0669, ' in' -0.0579, ' into' -0.0445, ' with' -0.0387, ' over' -0.0379, ',' +0.0305

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +1.80 [+1.57, +2.05] → **passes**; numerator +0.79 [+0.60, +0.99], PREP -1.01 [-1.20, -0.85], O +0.45 [+0.24, +0.66].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 8 | 0.49 / 0.36 | +0.96 [+0.84, +1.09] RISE | +1.71 [+1.53, +1.89] RISE | +0.86 [+0.74, +1.00] | -0.85 [-0.95, -0.75] | +0.13 [-0.00, +0.27] | +1.08 [+0.96, +1.20] | 0.51 / 0.81 | **mixed** |
| 12 | 0.53 / 0.36 | +2.07 [+1.91, +2.24] RISE | +1.05 [+0.94, +1.16] RISE | +0.42 [+0.34, +0.50] | -0.63 [-0.69, -0.56] | +0.24 [+0.15, +0.33] | +0.56 [+0.50, +0.63] | 0.50 / 0.84 | **mixed** |
| 17 | 0.57 / 0.36 | +4.09 [+3.79, +4.35] RISE | +0.37 [+0.29, +0.45] RISE | -1.23 [-1.40, -1.08] | -1.60 [-1.80, -1.41] | -1.38 [-1.57, -1.19] | +0.03 [+0.02, +0.05] | 0.24 / 0.87 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 8 1.00, site 12 1.00, site 17 1.00

Tokens driving the effects (probability change):

- natural good - bad: ',' +0.0860, '.' +0.0740, ' in' -0.0464, ' with' -0.0429, ' around' -0.0319, ' through' -0.0298

