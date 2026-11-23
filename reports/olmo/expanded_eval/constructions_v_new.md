# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +2.60 [+2.25, +2.93] → **passes**; numerator +1.57 [+1.32, +1.80], PREP -1.03 [-1.19, -0.89], O +1.35 [+1.10, +1.60].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.53 / 0.52 | +1.80 [+1.60, +2.00] RISE | +1.87 [+1.71, +2.03] RISE | +1.24 [+1.11, +1.36] | -0.63 [-0.72, -0.55] | -0.05 [-0.18, +0.08] | +1.09 [+0.99, +1.19] | 0.56 / 0.82 | **mixed** |
| 11 | 0.58 / 0.52 | +2.53 [+2.31, +2.74] RISE | +1.14 [+1.00, +1.27] RISE | +0.46 [+0.33, +0.59] | -0.67 [-0.77, -0.58] | -0.37 [-0.49, -0.26] | +0.42 [+0.34, +0.49] | 0.50 / 0.83 | **mixed** |
| 12 | 0.62 / 0.52 | +3.26 [+3.03, +3.49] RISE | +0.21 [+0.10, +0.32] NO RISE | -0.92 [-1.05, -0.79] | -1.13 [-1.30, -0.98] | -0.95 [-1.14, -0.78] | -0.10 [-0.15, -0.05] | 0.43 / 0.82 | **surface** |
| 13 | 0.62 / 0.52 | +3.57 [+3.31, +3.84] RISE | +0.93 [+0.81, +1.05] RISE | -0.88 [-1.00, -0.75] | -1.81 [-2.02, -1.60] | -1.27 [-1.47, -1.09] | +0.19 [+0.15, +0.23] | 0.36 / 0.84 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0939, ' in' -0.0699, ' with' -0.0648, ' is' +0.0467, ' on' -0.0397, ' into' -0.0392

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +1.86 [+1.50, +2.21] → **passes**; numerator +0.88 [+0.67, +1.09], PREP -0.98 [-1.17, -0.79], O +0.22 [-0.06, +0.52].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.53 / 0.37 | +1.80 [+1.64, +1.96] RISE | +1.59 [+1.39, +1.81] RISE | +0.72 [+0.57, +0.87] | -0.87 [-0.98, -0.77] | +0.11 [-0.02, +0.24] | +0.58 [+0.50, +0.67] | 0.35 / 0.82 | **mixed** |
| 11 | 0.58 / 0.37 | +2.43 [+2.25, +2.62] RISE | +1.16 [+1.02, +1.31] RISE | +0.41 [+0.29, +0.54] | -0.75 [-0.84, -0.67] | -0.05 [-0.18, +0.08] | +0.33 [+0.28, +0.38] | 0.31 / 0.83 | **mixed** |
| 12 | 0.62 / 0.37 | +3.72 [+3.48, +3.95] RISE | +0.18 [+0.10, +0.26] NO RISE | -0.95 [-1.09, -0.81] | -1.12 [-1.29, -0.97] | -0.76 [-0.91, -0.60] | -0.03 [-0.05, -0.01] | 0.26 / 0.82 | **surface** |
| 13 | 0.62 / 0.37 | +3.79 [+3.51, +4.05] RISE | +0.52 [+0.40, +0.64] RISE | -1.43 [-1.62, -1.25] | -1.95 [-2.18, -1.75] | -1.05 [-1.24, -0.87] | +0.03 [+0.01, +0.05] | 0.19 / 0.84 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 9 1.00, site 11 1.00, site 12 1.00, site 13 1.00

Tokens driving the effects (probability change):

- natural good - bad: ',' +0.0961, ' in' -0.0616, '.' +0.0573, ' with' -0.0470, ' on' -0.0440, ' and' +0.0300

