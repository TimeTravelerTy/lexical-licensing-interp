# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +1.75 [+1.41, +2.12] → **passes**; numerator +1.25 [+0.94, +1.60], PREP -0.50 [-0.64, -0.36], O +1.14 [+0.79, +1.51].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 5 | 0.54 / 0.35 | +2.23 [+1.90, +2.56] RISE | +1.15 [+0.93, +1.38] RISE | +0.56 [+0.33, +0.79] | -0.58 [-0.73, -0.45] | +0.46 [+0.26, +0.68] | +1.07 [+0.90, +1.26] | 0.67 / 0.70 | **mixed** |
| 8 | 0.53 / 0.35 | +1.65 [+1.41, +1.90] RISE | +1.25 [+1.06, +1.45] RISE | +0.85 [+0.69, +1.02] | -0.40 [-0.48, -0.33] | +0.53 [+0.36, +0.72] | +1.15 [+0.99, +1.31] | 0.63 / 0.76 | **mixed** |
| 12 | 0.56 / 0.35 | +2.34 [+2.08, +2.59] RISE | +1.29 [+1.10, +1.51] RISE | +0.64 [+0.48, +0.81] | -0.66 [-0.77, -0.55] | +0.73 [+0.54, +0.92] | +0.88 [+0.74, +1.02] | 0.52 / 0.83 | **mixed** |
| 15 | 0.58 / 0.35 | +2.92 [+2.68, +3.18] RISE | +2.06 [+1.84, +2.31] RISE | +1.11 [+0.91, +1.33] | -0.95 [-1.09, -0.82] | +0.72 [+0.51, +0.93] | +0.97 [+0.82, +1.14] | 0.41 / 0.84 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' to' -0.0695, ' was' +0.0538, ' over' -0.0410, ' into' -0.0300, ' is' +0.0272, ' in' -0.0240

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +0.63 [+0.41, +0.87] → **passes**; numerator +0.19 [-0.01, +0.39], PREP -0.44 [-0.67, -0.23], O +0.46 [+0.19, +0.71].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 5 | 0.54 / 0.13 | +1.11 [+0.93, +1.29] RISE | +0.68 [+0.55, +0.81] RISE | +0.24 [+0.16, +0.33] | -0.44 [-0.52, -0.36] | +0.31 [+0.17, +0.45] | +0.57 [+0.47, +0.68] | 0.69 / 0.70 | **mixed** |
| 8 | 0.53 / 0.13 | +1.28 [+1.10, +1.46] RISE | +0.68 [+0.59, +0.79] RISE | +0.15 [+0.07, +0.23] | -0.53 [-0.60, -0.47] | +0.13 [+0.03, +0.21] | +0.53 [+0.44, +0.62] | 0.64 / 0.76 | **mixed** |
| 12 | 0.56 / 0.13 | +1.91 [+1.71, +2.13] RISE | +0.83 [+0.73, +0.94] RISE | +0.08 [-0.02, +0.18] | -0.75 [-0.84, -0.66] | +0.17 [+0.07, +0.26] | +0.43 [+0.36, +0.52] | 0.46 / 0.83 | **mixed** |
| 15 | 0.58 / 0.13 | +2.47 [+2.23, +2.71] RISE | +1.29 [+1.16, +1.43] RISE | +0.26 [+0.13, +0.40] | -1.03 [-1.15, -0.93] | -0.00 [-0.11, +0.10] | +0.42 [+0.36, +0.50] | 0.29 / 0.84 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 5 0.97, site 8 0.98, site 12 0.99, site 15 0.99

Tokens driving the effects (probability change):

- natural good - bad: ' in' -0.0420, '.' +0.0403, ' around' -0.0204, ',' +0.0164, ' through' -0.0123, ' the' +0.0111

