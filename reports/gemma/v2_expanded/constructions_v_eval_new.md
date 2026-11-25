# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +2.76 [+2.26, +3.28] → **passes**; numerator +2.11 [+1.71, +2.53], PREP -0.65 [-0.84, -0.49], O +1.15 [+0.73, +1.59].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 5 | 0.54 / 0.55 | +2.49 [+2.13, +2.86] RISE | +1.61 [+1.37, +1.88] RISE | +0.83 [+0.55, +1.11] | -0.78 [-1.01, -0.59] | +0.09 [-0.21, +0.41] | +1.40 [+1.15, +1.67] | 0.64 / 0.65 | **mixed** |
| 8 | 0.53 / 0.55 | +1.84 [+1.57, +2.12] RISE | +1.68 [+1.47, +1.93] RISE | +1.18 [+0.98, +1.39] | -0.51 [-0.62, -0.41] | +0.31 [-0.01, +0.62] | +1.54 [+1.30, +1.78] | 0.68 / 0.76 | **mixed** |
| 12 | 0.56 / 0.55 | +2.56 [+2.28, +2.87] RISE | +1.76 [+1.55, +1.98] RISE | +0.98 [+0.77, +1.20] | -0.78 [-0.94, -0.63] | +0.46 [+0.15, +0.78] | +1.09 [+0.92, +1.27] | 0.57 / 0.83 | **mixed** |
| 15 | 0.58 / 0.55 | +3.26 [+2.95, +3.60] RISE | +2.51 [+2.25, +2.81] RISE | +1.44 [+1.17, +1.71] | -1.08 [-1.24, -0.92] | +0.31 [-0.04, +0.66] | +1.12 [+0.94, +1.31] | 0.43 / 0.84 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0828, ' with' -0.0583, ' on' -0.0503, ' is' +0.0461, ' into' -0.0433, ' over' -0.0356

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +1.32 [+0.89, +1.76] → **passes**; numerator +0.55 [+0.33, +0.77], PREP -0.77 [-1.00, -0.54], O +0.45 [+0.17, +0.75].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 5 | 0.54 / 0.26 | +1.23 [+1.01, +1.45] RISE | +0.96 [+0.75, +1.19] RISE | +0.36 [+0.23, +0.51] | -0.60 [-0.71, -0.50] | +0.12 [-0.03, +0.27] | +0.73 [+0.54, +0.94] | 0.64 / 0.65 | **mixed** |
| 8 | 0.53 / 0.26 | +1.35 [+1.18, +1.54] RISE | +0.89 [+0.73, +1.07] RISE | +0.31 [+0.18, +0.45] | -0.58 [-0.68, -0.50] | -0.02 [-0.16, +0.11] | +0.58 [+0.46, +0.73] | 0.62 / 0.76 | **mixed** |
| 12 | 0.56 / 0.26 | +2.03 [+1.81, +2.27] RISE | +1.01 [+0.86, +1.18] RISE | +0.25 [+0.12, +0.40] | -0.76 [-0.87, -0.66] | +0.05 [-0.09, +0.18] | +0.43 [+0.35, +0.52] | 0.46 / 0.83 | **mixed** |
| 15 | 0.58 / 0.26 | +2.69 [+2.43, +2.94] RISE | +1.49 [+1.31, +1.68] RISE | +0.45 [+0.29, +0.63] | -1.04 [-1.17, -0.92] | -0.14 [-0.27, -0.00] | +0.46 [+0.35, +0.56] | 0.28 / 0.84 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 5 0.99, site 8 1.00, site 12 1.00, site 15 1.00

Tokens driving the effects (probability change):

- natural good - bad: '.' +0.0754, ',' +0.0670, ' around' -0.0500, ' in' -0.0443, ' on' -0.0301, ' and' +0.0291

