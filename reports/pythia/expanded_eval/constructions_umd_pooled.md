# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +2.45 [+2.21, +2.71] → **passes**; numerator +1.60 [+1.40, +1.82], PREP -0.85 [-0.95, -0.74], O +1.32 [+1.11, +1.54].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 8 | 0.49 / 0.49 | +0.95 [+0.82, +1.09] RISE | +1.84 [+1.68, +2.00] RISE | +1.39 [+1.25, +1.53] | -0.45 [-0.51, -0.39] | +0.40 [+0.29, +0.52] | +1.30 [+1.19, +1.41] | 0.65 / 0.79 | **mixed** |
| 12 | 0.53 / 0.49 | +1.34 [+1.21, +1.49] RISE | +1.28 [+1.16, +1.40] RISE | +0.95 [+0.85, +1.05] | -0.32 [-0.37, -0.28] | +0.51 [+0.41, +0.61] | +0.83 [+0.76, +0.90] | 0.64 / 0.84 | **mixed** |
| 17 | 0.57 / 0.49 | +2.73 [+2.50, +2.95] RISE | +1.13 [+1.01, +1.26] RISE | +0.23 [+0.15, +0.31] | -0.90 [-1.02, -0.79] | +0.02 [-0.07, +0.13] | +0.55 [+0.49, +0.61] | 0.55 / 0.88 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0669, ' in' -0.0579, ' into' -0.0445, ' with' -0.0387, ' over' -0.0379, ',' +0.0305

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +1.80 [+1.57, +2.05] → **passes**; numerator +0.79 [+0.60, +0.99], PREP -1.01 [-1.20, -0.85], O +0.45 [+0.24, +0.66].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 8 | 0.49 / 0.36 | +0.55 [+0.45, +0.65] RISE | +1.38 [+1.23, +1.53] RISE | +0.74 [+0.63, +0.86] | -0.64 [-0.72, -0.55] | +0.22 [+0.10, +0.33] | +0.70 [+0.62, +0.78] | 0.45 / 0.79 | **mixed** |
| 12 | 0.53 / 0.36 | +0.85 [+0.73, +0.96] RISE | +1.14 [+1.03, +1.26] RISE | +0.62 [+0.54, +0.71] | -0.52 [-0.59, -0.45] | +0.38 [+0.29, +0.47] | +0.54 [+0.48, +0.59] | 0.46 / 0.84 | **mixed** |
| 17 | 0.57 / 0.36 | +3.05 [+2.80, +3.27] RISE | +0.51 [+0.43, +0.58] RISE | -0.09 [-0.15, -0.03] | -0.60 [-0.68, -0.52] | +0.14 [+0.07, +0.23] | +0.16 [+0.13, +0.18] | 0.32 / 0.88 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 8 0.99, site 12 1.00, site 17 0.99

Tokens driving the effects (probability change):

- natural good - bad: ',' +0.0860, '.' +0.0740, ' in' -0.0464, ' with' -0.0429, ' around' -0.0319, ' through' -0.0298

