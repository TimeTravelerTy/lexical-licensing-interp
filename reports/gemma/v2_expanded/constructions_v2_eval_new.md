# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +2.76 [+2.26, +3.28] → **passes**; numerator +2.11 [+1.71, +2.53], PREP -0.65 [-0.84, -0.49], O +1.15 [+0.73, +1.59].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 5 | 0.57 / 0.55 | +2.72 [+2.35, +3.10] RISE | +1.34 [+1.07, +1.61] RISE | +0.54 [+0.27, +0.82] | -0.80 [-1.03, -0.60] | +0.11 [-0.26, +0.51] | +1.49 [+1.25, +1.76] | 0.73 / 0.77 | **mixed** |
| 8 | 0.55 / 0.55 | +2.06 [+1.78, +2.37] RISE | +1.82 [+1.60, +2.05] RISE | +1.27 [+1.08, +1.47] | -0.55 [-0.67, -0.45] | +0.42 [+0.07, +0.77] | +1.78 [+1.52, +2.04] | 0.77 / 0.89 | **mixed** |
| 12 | 0.57 / 0.55 | +2.85 [+2.56, +3.16] RISE | +1.86 [+1.63, +2.11] RISE | +1.07 [+0.86, +1.29] | -0.80 [-0.96, -0.66] | +0.57 [+0.23, +0.92] | +1.41 [+1.21, +1.63] | 0.65 / 0.96 | **mixed** |
| 15 | 0.58 / 0.55 | +3.43 [+3.11, +3.77] RISE | +2.52 [+2.26, +2.80] RISE | +1.34 [+1.06, +1.62] | -1.18 [-1.37, -1.01] | +0.24 [-0.11, +0.60] | +1.40 [+1.18, +1.63] | 0.51 / 0.93 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0828, ' with' -0.0583, ' on' -0.0503, ' is' +0.0461, ' into' -0.0433, ' over' -0.0356

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +1.32 [+0.89, +1.76] → **passes**; numerator +0.55 [+0.33, +0.77], PREP -0.77 [-1.00, -0.54], O +0.45 [+0.17, +0.75].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 5 | 0.57 / 0.26 | +1.25 [+1.03, +1.47] RISE | +0.90 [+0.68, +1.13] RISE | +0.34 [+0.22, +0.48] | -0.56 [-0.69, -0.43] | +0.20 [+0.01, +0.39] | +0.92 [+0.70, +1.15] | 0.73 / 0.77 | **mixed** |
| 8 | 0.55 / 0.26 | +1.40 [+1.22, +1.59] RISE | +1.01 [+0.83, +1.21] RISE | +0.36 [+0.23, +0.51] | -0.65 [-0.75, -0.56] | +0.05 [-0.11, +0.20] | +0.84 [+0.67, +1.03] | 0.71 / 0.89 | **mixed** |
| 12 | 0.57 / 0.26 | +2.06 [+1.84, +2.29] RISE | +1.09 [+0.93, +1.27] RISE | +0.27 [+0.14, +0.42] | -0.81 [-0.93, -0.71] | +0.14 [-0.03, +0.29] | +0.60 [+0.47, +0.74] | 0.53 / 0.96 | **mixed** |
| 15 | 0.58 / 0.26 | +2.76 [+2.51, +3.02] RISE | +1.53 [+1.34, +1.74] RISE | +0.42 [+0.26, +0.61] | -1.11 [-1.25, -0.98] | -0.14 [-0.30, +0.01] | +0.59 [+0.46, +0.73] | 0.34 / 0.93 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 5 1.00, site 8 1.00, site 12 1.00, site 15 1.00

Tokens driving the effects (probability change):

- natural good - bad: '.' +0.0754, ',' +0.0670, ' around' -0.0500, ' in' -0.0443, ' on' -0.0301, ' and' +0.0291

