# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +2.55 [+2.27, +2.87] → **passes**; numerator +1.66 [+1.43, +1.92], PREP -0.89 [-1.01, -0.76], O +1.39 [+1.13, +1.64].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 8 | 0.49 / 0.51 | +1.45 [+1.28, +1.65] RISE | +2.18 [+1.98, +2.39] RISE | +1.55 [+1.38, +1.73] | -0.63 [-0.72, -0.55] | +0.34 [+0.19, +0.50] | +1.84 [+1.68, +2.01] | 0.74 / 0.86 | **mixed** |
| 12 | 0.53 / 0.51 | +2.65 [+2.44, +2.88] RISE | +1.14 [+1.01, +1.28] RISE | +0.43 [+0.30, +0.55] | -0.71 [-0.84, -0.61] | +0.16 [+0.04, +0.28] | +0.83 [+0.73, +0.93] | 0.71 / 0.88 | **mixed** |
| 17 | 0.57 / 0.51 | +3.59 [+3.30, +3.86] RISE | +1.28 [+1.13, +1.46] RISE | -1.11 [-1.27, -0.95] | -2.38 [-2.68, -2.11] | -2.09 [-2.34, -1.86] | +0.34 [+0.29, +0.40] | 0.45 / 0.91 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0703, ' in' -0.0593, ' into' -0.0470, ' to' -0.0446, ' over' -0.0399, ',' +0.0312

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +1.73 [+1.46, +2.00] → **passes**; numerator +0.71 [+0.46, +0.95], PREP -1.01 [-1.26, -0.81], O +0.42 [+0.16, +0.67].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 8 | 0.49 / 0.35 | +0.96 [+0.82, +1.10] RISE | +1.63 [+1.45, +1.83] RISE | +0.81 [+0.68, +0.95] | -0.83 [-0.94, -0.73] | +0.11 [-0.05, +0.26] | +1.08 [+0.95, +1.22] | 0.54 / 0.86 | **mixed** |
| 12 | 0.53 / 0.35 | +2.07 [+1.89, +2.26] RISE | +1.02 [+0.90, +1.14] RISE | +0.39 [+0.30, +0.48] | -0.63 [-0.71, -0.56] | +0.24 [+0.14, +0.34] | +0.57 [+0.50, +0.65] | 0.52 / 0.88 | **mixed** |
| 17 | 0.57 / 0.35 | +4.08 [+3.77, +4.35] RISE | +0.39 [+0.30, +0.49] RISE | -1.30 [-1.48, -1.13] | -1.70 [-1.91, -1.50] | -1.45 [-1.66, -1.26] | +0.04 [+0.02, +0.06] | 0.26 / 0.91 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 8 1.00, site 12 1.00, site 17 1.00

Tokens driving the effects (probability change):

- natural good - bad: ',' +0.0817, '.' +0.0704, ' in' -0.0439, ' around' -0.0382, ' with' -0.0335, ' and' +0.0272

