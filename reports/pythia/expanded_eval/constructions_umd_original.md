# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +2.24 [+1.90, +2.60] → **passes**; numerator +1.43 [+1.12, +1.75], PREP -0.81 [-0.96, -0.65], O +1.53 [+1.22, +1.85].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 8 | 0.49 / 0.45 | +0.91 [+0.77, +1.08] RISE | +1.82 [+1.63, +2.03] RISE | +1.35 [+1.18, +1.52] | -0.47 [-0.56, -0.39] | +0.40 [+0.26, +0.55] | +1.34 [+1.19, +1.52] | 0.67 / 0.84 | **mixed** |
| 12 | 0.53 / 0.45 | +1.36 [+1.19, +1.54] RISE | +1.31 [+1.17, +1.46] RISE | +0.97 [+0.85, +1.10] | -0.34 [-0.40, -0.28] | +0.54 [+0.42, +0.66] | +0.89 [+0.79, +0.99] | 0.68 / 0.89 | **mixed** |
| 17 | 0.57 / 0.45 | +2.79 [+2.54, +3.03] RISE | +1.14 [+1.00, +1.29] RISE | +0.19 [+0.11, +0.28] | -0.94 [-1.10, -0.80] | +0.01 [-0.11, +0.14] | +0.56 [+0.48, +0.65] | 0.56 / 0.90 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0602, ' to' -0.0600, ' in' -0.0485, ' into' -0.0419, ' over' -0.0392, ' at' -0.0288

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +1.44 [+1.11, +1.77] → **passes**; numerator +0.54 [+0.19, +0.84], PREP -0.90 [-1.23, -0.61], O +0.56 [+0.27, +0.85].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 8 | 0.49 / 0.29 | +0.50 [+0.38, +0.63] RISE | +1.27 [+1.09, +1.45] RISE | +0.66 [+0.53, +0.80] | -0.61 [-0.71, -0.51] | +0.21 [+0.07, +0.35] | +0.69 [+0.57, +0.81] | 0.47 / 0.84 | **mixed** |
| 12 | 0.53 / 0.29 | +0.77 [+0.64, +0.91] RISE | +1.09 [+0.96, +1.23] RISE | +0.58 [+0.47, +0.70] | -0.51 [-0.60, -0.43] | +0.40 [+0.28, +0.51] | +0.55 [+0.47, +0.64] | 0.49 / 0.89 | **mixed** |
| 17 | 0.57 / 0.29 | +3.03 [+2.77, +3.28] RISE | +0.49 [+0.42, +0.57] RISE | -0.13 [-0.20, -0.07] | -0.62 [-0.72, -0.54] | +0.13 [+0.05, +0.22] | +0.16 [+0.13, +0.20] | 0.34 / 0.90 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 8 0.99, site 12 1.00, site 17 0.99

Tokens driving the effects (probability change):

- natural good - bad: ',' +0.0656, '.' +0.0573, ' in' -0.0429, ' with' -0.0266, ' through' -0.0253, ' around' -0.0252

