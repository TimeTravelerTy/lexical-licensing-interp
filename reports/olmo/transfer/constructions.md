# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +2.42 [+2.10, +2.76] → **passes**; numerator +1.43 [+1.16, +1.72], PREP -0.99 [-1.15, -0.82], O +1.66 [+1.36, +1.98].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.53 / 0.48 | +1.77 [+1.57, +1.98] RISE | +1.97 [+1.79, +2.16] RISE | +1.33 [+1.18, +1.50] | -0.64 [-0.74, -0.55] | +0.05 [-0.11, +0.18] | +1.24 [+1.10, +1.38] | 0.58 / 0.87 | **mixed** |
| 11 | 0.58 / 0.48 | +2.60 [+2.35, +2.86] RISE | +1.23 [+1.11, +1.36] RISE | +0.57 [+0.45, +0.69] | -0.66 [-0.76, -0.57] | -0.22 [-0.34, -0.10] | +0.49 [+0.42, +0.57] | 0.53 / 0.89 | **mixed** |
| 12 | 0.62 / 0.48 | +3.44 [+3.19, +3.70] RISE | +0.29 [+0.16, +0.42] NO RISE | -1.05 [-1.21, -0.91] | -1.34 [-1.56, -1.14] | -1.13 [-1.35, -0.93] | -0.08 [-0.15, -0.02] | 0.47 / 0.89 | **surface** |
| 13 | 0.62 / 0.48 | +3.63 [+3.36, +3.91] RISE | +1.11 [+0.96, +1.25] RISE | -1.01 [-1.16, -0.86] | -2.12 [-2.37, -1.87] | -1.49 [-1.70, -1.29] | +0.29 [+0.25, +0.35] | 0.42 / 0.91 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0822, ' to' -0.0775, ' in' -0.0741, ' is' +0.0390, ' the' +0.0361, ' into' -0.0307
- site 9 T - I: ' was' +0.0717, ' in' -0.0621, ' to' -0.0456, ' is' +0.0306, ' with' -0.0271, ' on' -0.0268
- site 11 T - I: ' in' -0.0647, ' the' +0.0589, ' to' -0.0385, ' his' +0.0347, ' her' +0.0309, ' on' -0.0277
- site 12 T - I: ' it' +0.2268, ' in' -0.0967, ' the' +0.0862, ' him' +0.0708, ' to' -0.0583, ' her' +0.0482
- site 13 T - I: ' the' +0.1970, ' it' +0.1286, ' in' -0.1107, ' to' -0.0760, ' her' +0.0679, ' his' +0.0629

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +1.31 [+1.03, +1.60] → **passes**; numerator +0.47 [+0.14, +0.77], PREP -0.85 [-1.10, -0.61], O +0.06 [-0.24, +0.36].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.53 / 0.26 | +1.62 [+1.44, +1.80] RISE | +1.26 [+1.09, +1.45] RISE | +0.52 [+0.40, +0.67] | -0.74 [-0.83, -0.65] | +0.10 [-0.02, +0.22] | +0.58 [+0.48, +0.69] | 0.40 / 0.87 | **mixed** |
| 11 | 0.58 / 0.26 | +2.22 [+2.01, +2.43] RISE | +0.98 [+0.85, +1.11] RISE | +0.29 [+0.18, +0.41] | -0.69 [-0.76, -0.61] | +0.03 [-0.10, +0.14] | +0.35 [+0.29, +0.41] | 0.36 / 0.89 | **mixed** |
| 12 | 0.62 / 0.26 | +3.47 [+3.22, +3.72] RISE | +0.22 [+0.14, +0.29] unresolved | -1.06 [-1.21, -0.93] | -1.28 [-1.47, -1.11] | -0.85 [-1.02, -0.69] | -0.00 [-0.03, +0.02] | 0.32 / 0.89 | **unresolved** |
| 13 | 0.62 / 0.26 | +3.51 [+3.24, +3.77] RISE | +0.42 [+0.30, +0.53] RISE | -1.70 [-1.89, -1.53] | -2.12 [-2.34, -1.90] | -1.25 [-1.44, -1.08] | +0.05 [+0.03, +0.08] | 0.25 / 0.91 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 9 1.00, site 11 1.00, site 12 1.00, site 13 1.00

Tokens driving the effects (probability change):

- natural good - bad: ',' +0.0586, ' in' -0.0490, '.' +0.0400, ' through' -0.0267, ' and' +0.0261, ' around' -0.0241
- site 9 T - I: ' in' -0.0535, ',' +0.0494, ' on' -0.0303, '.' +0.0281, ' the' +0.0258, ' and' +0.0193
- site 11 T - I: ' in' -0.0517, ' the' +0.0406, ' on' -0.0274, ',' +0.0243, ' at' -0.0156, ' it' +0.0150
- site 12 T - I: ' it' +0.2267, ' him' +0.0732, ' in' -0.0671, ' the' +0.0660, ',' -0.0654, '.' -0.0396
- site 13 T - I: ' it' +0.2224, ' the' +0.1483, ',' -0.0833, ' in' -0.0774, '.' -0.0495, ' him' +0.0400

