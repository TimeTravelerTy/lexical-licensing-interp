# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +2.55 [+2.27, +2.87] → **passes**; numerator +1.66 [+1.43, +1.92], PREP -0.89 [-1.01, -0.76], O +1.39 [+1.13, +1.64].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 8 | 0.50 / 0.51 | +1.36 [+1.18, +1.55] RISE | +2.28 [+2.07, +2.50] RISE | +1.60 [+1.42, +1.78] | -0.68 [-0.77, -0.59] | +0.38 [+0.22, +0.55] | +2.01 [+1.82, +2.20] | 0.80 / 0.93 | **mixed** |
| 12 | 0.54 / 0.51 | +2.70 [+2.48, +2.95] RISE | +1.38 [+1.25, +1.54] RISE | +0.59 [+0.47, +0.71] | -0.79 [-0.91, -0.69] | +0.20 [+0.09, +0.33] | +1.09 [+0.99, +1.20] | 0.78 / 0.95 | **mixed** |
| 17 | 0.59 / 0.51 | +3.67 [+3.37, +3.95] RISE | +1.69 [+1.51, +1.89] RISE | -1.33 [-1.52, -1.14] | -3.02 [-3.37, -2.68] | -2.47 [-2.75, -2.19] | +0.54 [+0.47, +0.61] | 0.49 / 0.95 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0703, ' in' -0.0593, ' into' -0.0470, ' to' -0.0446, ' over' -0.0399, ',' +0.0312

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +1.73 [+1.46, +2.00] → **passes**; numerator +0.71 [+0.46, +0.95], PREP -1.01 [-1.26, -0.81], O +0.42 [+0.16, +0.67].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 8 | 0.50 / 0.35 | +0.87 [+0.73, +1.01] RISE | +1.77 [+1.56, +1.97] RISE | +0.87 [+0.73, +1.02] | -0.90 [-1.01, -0.79] | +0.18 [+0.01, +0.33] | +1.26 [+1.10, +1.42] | 0.60 / 0.93 | **mixed** |
| 12 | 0.54 / 0.35 | +2.07 [+1.88, +2.27] RISE | +1.21 [+1.07, +1.36] RISE | +0.49 [+0.39, +0.59] | -0.72 [-0.81, -0.64] | +0.31 [+0.20, +0.42] | +0.77 [+0.66, +0.86] | 0.58 / 0.95 | **mixed** |
| 17 | 0.59 / 0.35 | +4.23 [+3.92, +4.52] RISE | +0.76 [+0.66, +0.87] RISE | -1.53 [-1.75, -1.32] | -2.30 [-2.57, -2.03] | -1.78 [-2.02, -1.53] | +0.13 [+0.10, +0.15] | 0.28 / 0.95 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 8 1.00, site 12 1.00, site 17 1.00

Tokens driving the effects (probability change):

- natural good - bad: ',' +0.0817, '.' +0.0704, ' in' -0.0439, ' around' -0.0382, ' with' -0.0335, ' and' +0.0272

