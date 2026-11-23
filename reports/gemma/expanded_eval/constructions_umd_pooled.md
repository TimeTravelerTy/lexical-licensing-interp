# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +1.98 [+1.73, +2.24] → **passes**; numerator +1.46 [+1.24, +1.69], PREP -0.52 [-0.61, -0.44], O +1.16 [+0.91, +1.41].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 5 | 0.54 / 0.40 | +0.70 [+0.39, +1.01] RISE | +0.95 [+0.58, +1.30] RISE | +0.71 [+0.43, +0.99] | -0.24 [-0.33, -0.16] | +0.32 [+0.19, +0.47] | +0.58 [+0.42, +0.74] | 0.51 / 0.76 | **mixed** |
| 8 | 0.53 / 0.40 | +0.81 [+0.59, +1.03] RISE | +0.88 [+0.65, +1.10] RISE | +0.67 [+0.49, +0.85] | -0.21 [-0.27, -0.16] | +0.26 [+0.17, +0.37] | +1.21 [+1.05, +1.36] | 0.99 / 0.68 | **mixed** |
| 12 | 0.56 / 0.40 | +1.12 [+0.87, +1.38] RISE | +0.98 [+0.76, +1.21] RISE | +0.74 [+0.58, +0.92] | -0.24 [-0.31, -0.18] | +0.40 [+0.28, +0.55] | +0.99 [+0.87, +1.10] | 0.92 / 0.87 | **mixed** |
| 15 | 0.58 / 0.40 | +1.49 [+1.29, +1.70] RISE | +1.89 [+1.61, +2.16] RISE | +1.48 [+1.25, +1.70] | -0.41 [-0.49, -0.34] | +0.53 [+0.39, +0.68] | +0.76 [+0.64, +0.89] | 0.36 / 0.92 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0584, ' over' -0.0407, ' into' -0.0360, ' with' -0.0351, ' to' -0.0330, ' is' +0.0316

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +0.88 [+0.68, +1.08] → **passes**; numerator +0.33 [+0.20, +0.47], PREP -0.54 [-0.67, -0.41], O +0.53 [+0.35, +0.72].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 5 | 0.54 / 0.18 | +0.54 [+0.34, +0.74] RISE | +0.42 [+0.26, +0.58] RISE | +0.16 [+0.09, +0.23] | -0.26 [-0.35, -0.16] | +0.14 [+0.08, +0.20] | +0.27 [+0.18, +0.36] | 0.35 / 0.76 | **unresolved (gate)** |
| 8 | 0.53 / 0.18 | +0.68 [+0.52, +0.84] RISE | +0.42 [+0.31, +0.52] RISE | +0.13 [+0.08, +0.18] | -0.29 [-0.36, -0.22] | +0.10 [+0.04, +0.15] | +0.49 [+0.41, +0.58] | 0.97 / 0.68 | **unresolved (gate)** |
| 12 | 0.56 / 0.18 | +1.02 [+0.79, +1.24] RISE | +0.42 [+0.32, +0.52] RISE | +0.10 [+0.06, +0.15] | -0.32 [-0.40, -0.24] | +0.19 [+0.13, +0.26] | +0.33 [+0.27, +0.38] | 0.76 / 0.87 | **mixed** |
| 15 | 0.58 / 0.18 | +1.44 [+1.24, +1.66] RISE | +0.85 [+0.71, +0.99] RISE | +0.30 [+0.22, +0.39] | -0.55 [-0.63, -0.46] | +0.10 [+0.03, +0.17] | +0.17 [+0.11, +0.23] | 0.13 / 0.92 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 5 0.68 (fails), site 8 0.77 (fails), site 12 0.84, site 15 0.94

Tokens driving the effects (probability change):

- natural good - bad: '.' +0.0509, ' in' -0.0390, ',' +0.0354, ' around' -0.0224, ' through' -0.0181, ' into' -0.0155

