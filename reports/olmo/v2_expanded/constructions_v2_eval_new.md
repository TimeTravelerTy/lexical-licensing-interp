# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +3.13 [+2.68, +3.61] → **passes**; numerator +1.92 [+1.62, +2.25], PREP -1.21 [-1.44, -0.99], O +1.43 [+1.07, +1.82].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.54 / 0.63 | +1.96 [+1.69, +2.26] RISE | +2.12 [+1.92, +2.32] RISE | +1.43 [+1.28, +1.60] | -0.68 [-0.81, -0.57] | -0.01 [-0.25, +0.24] | +1.62 [+1.42, +1.82] | 0.72 / 0.98 | **mixed** |
| 11 | 0.59 / 0.63 | +2.94 [+2.68, +3.20] RISE | +1.28 [+1.11, +1.46] RISE | +0.45 [+0.26, +0.64] | -0.83 [-0.99, -0.69] | -0.48 [-0.68, -0.28] | +0.54 [+0.42, +0.67] | 0.60 / 0.96 | **mixed** |
| 12 | 0.63 / 0.63 | +3.55 [+3.27, +3.82] RISE | +0.68 [+0.51, +0.85] RISE | -1.03 [-1.22, -0.84] | -1.70 [-1.94, -1.47] | -1.45 [-1.74, -1.18] | -0.01 [-0.11, +0.07] | 0.52 / 0.95 | **mixed** |
| 13 | 0.67 / 0.63 | +3.94 [+3.63, +4.27] RISE | +1.14 [+1.01, +1.28] RISE | -1.77 [-2.00, -1.52] | -2.91 [-3.23, -2.58] | -2.57 [-2.93, -2.19] | +0.24 [+0.19, +0.30] | 0.43 / 0.95 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.1116, ' in' -0.0779, ' with' -0.0663, ' is' +0.0592, ' to' -0.0496, ' into' -0.0491

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +2.12 [+1.65, +2.63] → **passes**; numerator +0.97 [+0.70, +1.28], PREP -1.15 [-1.40, -0.92], O -0.02 [-0.32, +0.29].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.54 / 0.42 | +1.90 [+1.69, +2.12] RISE | +1.78 [+1.55, +2.05] RISE | +0.81 [+0.63, +1.01] | -0.97 [-1.12, -0.84] | +0.11 [-0.14, +0.35] | +0.92 [+0.78, +1.07] | 0.48 / 0.98 | **mixed** |
| 11 | 0.59 / 0.42 | +2.70 [+2.47, +2.92] RISE | +1.28 [+1.12, +1.45] RISE | +0.38 [+0.22, +0.55] | -0.90 [-1.04, -0.78] | -0.16 [-0.38, +0.06] | +0.44 [+0.36, +0.51] | 0.38 / 0.96 | **mixed** |
| 12 | 0.63 / 0.42 | +3.89 [+3.63, +4.15] RISE | +0.60 [+0.49, +0.72] RISE | -0.95 [-1.12, -0.79] | -1.55 [-1.77, -1.34] | -1.01 [-1.25, -0.79] | +0.05 [+0.02, +0.08] | 0.32 / 0.95 | **mixed** |
| 13 | 0.67 / 0.42 | +3.98 [+3.69, +4.27] RISE | +0.75 [+0.64, +0.86] RISE | -2.21 [-2.49, -1.94] | -2.96 [-3.29, -2.65] | -2.23 [-2.54, -1.93] | +0.07 [+0.04, +0.10] | 0.22 / 0.95 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 9 1.00, site 11 1.00, site 12 1.00, site 13 1.00

Tokens driving the effects (probability change):

- natural good - bad: ',' +0.1082, '.' +0.0694, ' in' -0.0692, ' with' -0.0498, ' around' -0.0496, ' on' -0.0491

