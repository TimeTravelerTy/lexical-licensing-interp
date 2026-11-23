# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +2.50 [+2.26, +2.76] → **passes**; numerator +1.49 [+1.31, +1.69], PREP -1.01 [-1.14, -0.89], O +1.50 [+1.29, +1.72].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.53 / 0.50 | +1.78 [+1.60, +1.97] RISE | +1.90 [+1.75, +2.05] RISE | +1.27 [+1.16, +1.39] | -0.63 [-0.70, -0.56] | -0.02 [-0.12, +0.09] | +1.14 [+1.06, +1.22] | 0.56 / 0.84 | **mixed** |
| 11 | 0.58 / 0.50 | +2.54 [+2.36, +2.75] RISE | +1.18 [+1.06, +1.29] RISE | +0.51 [+0.41, +0.62] | -0.66 [-0.75, -0.59] | -0.31 [-0.41, -0.22] | +0.45 [+0.39, +0.51] | 0.51 / 0.85 | **mixed** |
| 12 | 0.62 / 0.50 | +3.33 [+3.12, +3.55] RISE | +0.25 [+0.16, +0.34] NO RISE | -0.97 [-1.10, -0.85] | -1.22 [-1.38, -1.07] | -1.04 [-1.21, -0.87] | -0.09 [-0.13, -0.05] | 0.44 / 0.85 | **surface** |
| 13 | 0.62 / 0.50 | +3.59 [+3.35, +3.84] RISE | +1.00 [+0.89, +1.12] RISE | -0.94 [-1.07, -0.82] | -1.95 [-2.14, -1.75] | -1.38 [-1.55, -1.20] | +0.24 [+0.20, +0.27] | 0.38 / 0.86 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0882, ' in' -0.0727, ' to' -0.0500, ' with' -0.0450, ' is' +0.0426, ' into' -0.0369

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +1.60 [+1.35, +1.87] → **passes**; numerator +0.68 [+0.48, +0.88], PREP -0.91 [-1.08, -0.75], O +0.16 [-0.07, +0.39].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.53 / 0.32 | +1.71 [+1.55, +1.86] RISE | +1.42 [+1.26, +1.61] RISE | +0.63 [+0.51, +0.76] | -0.80 [-0.88, -0.72] | +0.11 [-0.00, +0.21] | +0.57 [+0.50, +0.65] | 0.37 / 0.84 | **mixed** |
| 11 | 0.58 / 0.32 | +2.32 [+2.16, +2.49] RISE | +1.07 [+0.96, +1.18] RISE | +0.35 [+0.26, +0.45] | -0.71 [-0.78, -0.64] | -0.01 [-0.13, +0.09] | +0.34 [+0.30, +0.38] | 0.33 / 0.85 | **mixed** |
| 12 | 0.62 / 0.32 | +3.60 [+3.38, +3.82] RISE | +0.19 [+0.13, +0.25] NO RISE | -0.99 [-1.11, -0.87] | -1.18 [-1.34, -1.04] | -0.79 [-0.93, -0.66] | -0.02 [-0.03, -0.00] | 0.29 / 0.85 | **surface** |
| 13 | 0.62 / 0.32 | +3.66 [+3.40, +3.91] RISE | +0.46 [+0.36, +0.56] RISE | -1.55 [-1.73, -1.40] | -2.01 [-2.22, -1.83] | -1.14 [-1.31, -0.98] | +0.04 [+0.02, +0.06] | 0.21 / 0.86 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 9 1.00, site 11 1.00, site 12 1.00, site 13 1.00

Tokens driving the effects (probability change):

- natural good - bad: ',' +0.0772, ' in' -0.0561, '.' +0.0490, ' on' -0.0320, ' with' -0.0296, ' and' +0.0271

