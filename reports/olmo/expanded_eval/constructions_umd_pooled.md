# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +2.50 [+2.26, +2.76] → **passes**; numerator +1.49 [+1.31, +1.69], PREP -1.01 [-1.14, -0.89], O +1.50 [+1.29, +1.72].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.53 / 0.50 | +1.28 [+1.15, +1.43] RISE | +1.47 [+1.34, +1.61] RISE | +1.05 [+0.95, +1.16] | -0.42 [-0.48, -0.36] | -0.02 [-0.11, +0.08] | +0.75 [+0.69, +0.81] | 0.52 / 0.80 | **mixed** |
| 11 | 0.58 / 0.50 | +1.59 [+1.46, +1.74] RISE | +1.28 [+1.16, +1.41] RISE | +0.88 [+0.78, +0.98] | -0.40 [-0.45, -0.36] | -0.04 [-0.14, +0.06] | +0.45 [+0.40, +0.50] | 0.46 / 0.83 | **mixed** |
| 12 | 0.62 / 0.50 | +2.43 [+2.27, +2.61] RISE | +0.49 [+0.40, +0.57] unresolved | -0.04 [-0.09, +0.02] | -0.52 [-0.59, -0.46] | -0.04 [-0.14, +0.05] | +0.09 [+0.06, +0.13] | 0.45 / 0.83 | **unresolved** |
| 13 | 0.62 / 0.50 | +2.81 [+2.62, +3.02] RISE | +0.82 [+0.74, +0.90] RISE | -0.11 [-0.17, -0.05] | -0.92 [-1.02, -0.83] | -0.48 [-0.60, -0.37] | +0.24 [+0.20, +0.27] | 0.39 / 0.85 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0882, ' in' -0.0727, ' to' -0.0500, ' with' -0.0450, ' is' +0.0426, ' into' -0.0369

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +1.60 [+1.35, +1.87] → **passes**; numerator +0.68 [+0.48, +0.88], PREP -0.91 [-1.08, -0.75], O +0.16 [-0.07, +0.39].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.53 / 0.32 | +1.23 [+1.11, +1.35] RISE | +1.08 [+0.94, +1.22] RISE | +0.51 [+0.42, +0.62] | -0.56 [-0.63, -0.50] | +0.13 [+0.05, +0.22] | +0.36 [+0.32, +0.41] | 0.34 / 0.80 | **mixed** |
| 11 | 0.58 / 0.32 | +1.52 [+1.39, +1.65] RISE | +0.96 [+0.85, +1.08] RISE | +0.41 [+0.33, +0.51] | -0.54 [-0.60, -0.49] | +0.10 [+0.01, +0.20] | +0.27 [+0.23, +0.31] | 0.31 / 0.83 | **mixed** |
| 12 | 0.62 / 0.32 | +2.51 [+2.34, +2.68] RISE | +0.33 [+0.27, +0.38] RISE | -0.24 [-0.29, -0.19] | -0.57 [-0.64, -0.49] | +0.06 [-0.02, +0.14] | +0.04 [+0.02, +0.06] | 0.29 / 0.83 | **mixed** |
| 13 | 0.62 / 0.32 | +2.96 [+2.76, +3.15] RISE | +0.28 [+0.22, +0.34] unresolved | -0.58 [-0.66, -0.49] | -0.85 [-0.95, -0.76] | -0.04 [-0.12, +0.04] | +0.03 [+0.01, +0.04] | 0.22 / 0.85 | **unresolved** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 9 0.99, site 11 1.00, site 12 1.00, site 13 1.00

Tokens driving the effects (probability change):

- natural good - bad: ',' +0.0772, ' in' -0.0561, '.' +0.0490, ' on' -0.0320, ' with' -0.0296, ' and' +0.0271

