# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +2.09 [+1.79, +2.41] → **passes**; numerator +1.54 [+1.27, +1.84], PREP -0.55 [-0.66, -0.44], O +1.15 [+0.86, +1.45].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 5 | 0.57 / 0.42 | +2.43 [+2.12, +2.76] RISE | +1.10 [+0.90, +1.31] RISE | +0.46 [+0.26, +0.66] | -0.64 [-0.78, -0.51] | +0.40 [+0.17, +0.62] | +1.23 [+1.07, +1.40] | 0.73 / 0.76 | **mixed** |
| 8 | 0.55 / 0.42 | +1.91 [+1.70, +2.15] RISE | +1.57 [+1.41, +1.74] RISE | +1.10 [+0.97, +1.24] | -0.47 [-0.54, -0.40] | +0.59 [+0.39, +0.80] | +1.44 [+1.27, +1.61] | 0.70 / 0.84 | **mixed** |
| 12 | 0.57 / 0.42 | +2.72 [+2.48, +2.96] RISE | +1.56 [+1.39, +1.75] RISE | +0.86 [+0.72, +1.01] | -0.70 [-0.80, -0.61] | +0.75 [+0.55, +0.95] | +1.15 [+1.01, +1.30] | 0.58 / 0.90 | **mixed** |
| 15 | 0.58 / 0.42 | +3.23 [+2.98, +3.49] RISE | +2.17 [+1.95, +2.39] RISE | +1.12 [+0.93, +1.32] | -1.04 [-1.18, -0.92] | +0.52 [+0.32, +0.73] | +1.16 [+1.02, +1.32] | 0.46 / 0.89 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0635, ' to' -0.0554, ' over' -0.0392, ' into' -0.0345, ' is' +0.0336, ' with' -0.0290

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +0.86 [+0.65, +1.08] → **passes**; numerator +0.31 [+0.15, +0.47], PREP -0.55 [-0.72, -0.39], O +0.45 [+0.25, +0.66].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 5 | 0.57 / 0.17 | +1.12 [+0.94, +1.29] RISE | +0.65 [+0.53, +0.77] RISE | +0.25 [+0.18, +0.32] | -0.40 [-0.48, -0.33] | +0.32 [+0.17, +0.46] | +0.68 [+0.57, +0.80] | 0.74 / 0.76 | **mixed** |
| 8 | 0.55 / 0.17 | +1.36 [+1.22, +1.52] RISE | +0.84 [+0.74, +0.95] RISE | +0.25 [+0.18, +0.33] | -0.59 [-0.65, -0.53] | +0.15 [+0.05, +0.25] | +0.67 [+0.57, +0.77] | 0.65 / 0.84 | **mixed** |
| 12 | 0.57 / 0.17 | +1.98 [+1.80, +2.18] RISE | +0.93 [+0.83, +1.04] RISE | +0.16 [+0.08, +0.25] | -0.77 [-0.85, -0.69] | +0.23 [+0.14, +0.32] | +0.50 [+0.43, +0.58] | 0.47 / 0.90 | **mixed** |
| 15 | 0.58 / 0.17 | +2.63 [+2.40, +2.84] RISE | +1.38 [+1.25, +1.51] RISE | +0.31 [+0.19, +0.43] | -1.07 [-1.17, -0.97] | -0.04 [-0.14, +0.06] | +0.49 [+0.42, +0.57] | 0.31 / 0.89 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 5 0.99, site 8 0.99, site 12 0.99, site 15 0.99

Tokens driving the effects (probability change):

- natural good - bad: '.' +0.0522, ' in' -0.0427, ',' +0.0335, ' around' -0.0304, ' through' -0.0144, ' because' +0.0141

