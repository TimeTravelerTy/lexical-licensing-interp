# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +2.24 [+1.90, +2.60] → **passes**; numerator +1.43 [+1.12, +1.75], PREP -0.81 [-0.96, -0.65], O +1.53 [+1.22, +1.85].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 8 | 0.49 / 0.45 | +1.38 [+1.20, +1.59] RISE | +2.09 [+1.86, +2.33] RISE | +1.46 [+1.26, +1.66] | -0.63 [-0.74, -0.53] | +0.37 [+0.20, +0.55] | +1.76 [+1.57, +1.98] | 0.72 / 0.85 | **mixed** |
| 12 | 0.53 / 0.45 | +2.62 [+2.39, +2.87] RISE | +1.16 [+1.01, +1.31] RISE | +0.42 [+0.28, +0.57] | -0.73 [-0.89, -0.61] | +0.16 [+0.02, +0.30] | +0.85 [+0.75, +0.97] | 0.71 / 0.89 | **mixed** |
| 17 | 0.57 / 0.45 | +3.58 [+3.30, +3.86] RISE | +1.25 [+1.09, +1.45] RISE | -1.13 [-1.29, -0.97] | -2.38 [-2.69, -2.09] | -2.10 [-2.35, -1.85] | +0.35 [+0.29, +0.42] | 0.45 / 0.90 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0602, ' to' -0.0600, ' in' -0.0485, ' into' -0.0419, ' over' -0.0392, ' at' -0.0288

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +1.44 [+1.11, +1.77] → **passes**; numerator +0.54 [+0.19, +0.84], PREP -0.90 [-1.23, -0.61], O +0.56 [+0.27, +0.85].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 8 | 0.49 / 0.29 | +0.89 [+0.75, +1.05] RISE | +1.54 [+1.33, +1.75] RISE | +0.75 [+0.61, +0.91] | -0.79 [-0.91, -0.68] | +0.12 [-0.06, +0.28] | +1.02 [+0.86, +1.20] | 0.52 / 0.85 | **mixed** |
| 12 | 0.53 / 0.29 | +2.00 [+1.80, +2.21] RISE | +0.99 [+0.86, +1.14] RISE | +0.37 [+0.26, +0.47] | -0.62 [-0.71, -0.55] | +0.24 [+0.12, +0.36] | +0.57 [+0.48, +0.67] | 0.52 / 0.89 | **mixed** |
| 17 | 0.57 / 0.29 | +4.04 [+3.73, +4.32] RISE | +0.39 [+0.30, +0.49] RISE | -1.31 [-1.49, -1.15] | -1.71 [-1.92, -1.51] | -1.46 [-1.68, -1.26] | +0.04 [+0.02, +0.07] | 0.27 / 0.90 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 8 1.00, site 12 1.00, site 17 1.00

Tokens driving the effects (probability change):

- natural good - bad: ',' +0.0656, '.' +0.0573, ' in' -0.0429, ' with' -0.0266, ' through' -0.0253, ' around' -0.0252

