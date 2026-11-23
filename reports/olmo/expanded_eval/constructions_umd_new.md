# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +2.60 [+2.25, +2.93] → **passes**; numerator +1.57 [+1.32, +1.80], PREP -1.03 [-1.19, -0.89], O +1.35 [+1.10, +1.60].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.53 / 0.52 | +1.26 [+1.12, +1.41] RISE | +1.42 [+1.28, +1.57] RISE | +1.01 [+0.90, +1.12] | -0.41 [-0.48, -0.35] | -0.06 [-0.18, +0.06] | +0.71 [+0.64, +0.79] | 0.51 / 0.78 | **mixed** |
| 11 | 0.58 / 0.52 | +1.58 [+1.42, +1.74] RISE | +1.22 [+1.09, +1.36] RISE | +0.83 [+0.71, +0.94] | -0.40 [-0.46, -0.34] | -0.12 [-0.24, -0.01] | +0.41 [+0.35, +0.47] | 0.45 / 0.81 | **mixed** |
| 12 | 0.62 / 0.52 | +2.37 [+2.19, +2.56] RISE | +0.46 [+0.37, +0.56] unresolved | -0.04 [-0.11, +0.03] | -0.50 [-0.58, -0.43] | -0.06 [-0.16, +0.03] | +0.08 [+0.04, +0.12] | 0.43 / 0.81 | **unresolved** |
| 13 | 0.62 / 0.52 | +2.78 [+2.57, +3.00] RISE | +0.78 [+0.70, +0.87] RISE | -0.09 [-0.16, -0.03] | -0.88 [-0.98, -0.78] | -0.47 [-0.60, -0.35] | +0.21 [+0.18, +0.24] | 0.37 / 0.83 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0939, ' in' -0.0699, ' with' -0.0648, ' is' +0.0467, ' on' -0.0397, ' into' -0.0392

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +1.86 [+1.50, +2.21] → **passes**; numerator +0.88 [+0.67, +1.09], PREP -0.98 [-1.17, -0.79], O +0.22 [-0.06, +0.52].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.53 / 0.37 | +1.30 [+1.17, +1.43] RISE | +1.19 [+1.03, +1.36] RISE | +0.58 [+0.47, +0.70] | -0.61 [-0.70, -0.53] | +0.13 [+0.01, +0.23] | +0.36 [+0.30, +0.42] | 0.32 / 0.78 | **mixed** |
| 11 | 0.58 / 0.37 | +1.61 [+1.47, +1.75] RISE | +1.05 [+0.91, +1.19] RISE | +0.47 [+0.36, +0.58] | -0.58 [-0.65, -0.51] | +0.07 [-0.05, +0.18] | +0.26 [+0.21, +0.31] | 0.28 / 0.81 | **mixed** |
| 12 | 0.62 / 0.37 | +2.57 [+2.39, +2.75] RISE | +0.33 [+0.26, +0.40] unresolved | -0.21 [-0.28, -0.14] | -0.53 [-0.62, -0.45] | +0.06 [-0.04, +0.15] | +0.03 [+0.01, +0.05] | 0.26 / 0.81 | **unresolved** |
| 13 | 0.62 / 0.37 | +3.05 [+2.83, +3.26] RISE | +0.32 [+0.25, +0.40] unresolved | -0.50 [-0.60, -0.40] | -0.82 [-0.92, -0.72] | -0.01 [-0.10, +0.09] | +0.02 [+0.01, +0.04] | 0.19 / 0.83 | **unresolved** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 9 0.99, site 11 1.00, site 12 1.00, site 13 1.00

Tokens driving the effects (probability change):

- natural good - bad: ',' +0.0961, ' in' -0.0616, '.' +0.0573, ' with' -0.0470, ' on' -0.0440, ' and' +0.0300

