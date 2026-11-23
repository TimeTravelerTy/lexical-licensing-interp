# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +1.98 [+1.73, +2.24] → **passes**; numerator +1.46 [+1.24, +1.69], PREP -0.52 [-0.61, -0.44], O +1.16 [+0.91, +1.41].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 5 | 0.54 / 0.40 | +2.29 [+1.99, +2.59] RISE | +1.33 [+1.17, +1.51] RISE | +0.67 [+0.49, +0.83] | -0.67 [-0.80, -0.55] | +0.36 [+0.18, +0.53] | +1.13 [+1.00, +1.26] | 0.64 / 0.67 | **mixed** |
| 8 | 0.53 / 0.40 | +1.70 [+1.50, +1.91] RISE | +1.43 [+1.27, +1.59] RISE | +0.98 [+0.85, +1.11] | -0.45 [-0.51, -0.39] | +0.46 [+0.30, +0.61] | +1.25 [+1.13, +1.37] | 0.64 / 0.75 | **mixed** |
| 12 | 0.56 / 0.40 | +2.40 [+2.18, +2.63] RISE | +1.49 [+1.33, +1.66] RISE | +0.79 [+0.67, +0.92] | -0.70 [-0.80, -0.61] | +0.65 [+0.48, +0.81] | +0.92 [+0.83, +1.02] | 0.53 / 0.82 | **mixed** |
| 15 | 0.58 / 0.40 | +3.04 [+2.81, +3.28] RISE | +2.24 [+2.03, +2.45] RISE | +1.25 [+1.08, +1.43] | -0.98 [-1.10, -0.88] | +0.60 [+0.42, +0.77] | +1.00 [+0.89, +1.11] | 0.41 / 0.82 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0584, ' over' -0.0407, ' into' -0.0360, ' with' -0.0351, ' to' -0.0330, ' is' +0.0316

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +0.88 [+0.68, +1.08] → **passes**; numerator +0.33 [+0.20, +0.47], PREP -0.54 [-0.67, -0.41], O +0.53 [+0.35, +0.72].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 5 | 0.54 / 0.18 | +1.16 [+1.00, +1.32] RISE | +0.77 [+0.65, +0.88] RISE | +0.29 [+0.22, +0.36] | -0.48 [-0.54, -0.41] | +0.28 [+0.16, +0.39] | +0.58 [+0.49, +0.66] | 0.62 / 0.67 | **mixed** |
| 8 | 0.53 / 0.18 | +1.29 [+1.15, +1.45] RISE | +0.77 [+0.67, +0.87] RISE | +0.22 [+0.16, +0.30] | -0.54 [-0.60, -0.48] | +0.10 [+0.02, +0.18] | +0.52 [+0.45, +0.59] | 0.59 / 0.75 | **mixed** |
| 12 | 0.56 / 0.18 | +1.94 [+1.76, +2.12] RISE | +0.90 [+0.81, +1.01] RISE | +0.17 [+0.09, +0.26] | -0.73 [-0.81, -0.66] | +0.18 [+0.10, +0.25] | +0.41 [+0.36, +0.46] | 0.43 / 0.82 | **mixed** |
| 15 | 0.58 / 0.18 | +2.53 [+2.31, +2.74] RISE | +1.40 [+1.26, +1.53] RISE | +0.39 [+0.28, +0.50] | -1.01 [-1.10, -0.92] | +0.02 [-0.06, +0.11] | +0.44 [+0.38, +0.49] | 0.27 / 0.82 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 5 0.97, site 8 0.99, site 12 0.99, site 15 0.99

Tokens driving the effects (probability change):

- natural good - bad: '.' +0.0509, ' in' -0.0390, ',' +0.0354, ' around' -0.0224, ' through' -0.0181, ' into' -0.0155

