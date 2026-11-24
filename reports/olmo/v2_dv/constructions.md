# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +2.42 [+2.10, +2.76] → **passes**; numerator +1.43 [+1.16, +1.72], PREP -0.99 [-1.15, -0.82], O +1.66 [+1.36, +1.98].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.53 / 0.48 | +1.80 [+1.60, +2.00] RISE | +2.06 [+1.85, +2.27] RISE | +1.40 [+1.24, +1.57] | -0.66 [-0.76, -0.56] | +0.13 [-0.04, +0.29] | +1.45 [+1.28, +1.63] | 0.64 / 0.93 | **mixed** |
| 11 | 0.58 / 0.48 | +2.81 [+2.57, +3.07] RISE | +1.33 [+1.21, +1.45] RISE | +0.57 [+0.46, +0.69] | -0.76 [-0.87, -0.65] | -0.22 [-0.35, -0.08] | +0.58 [+0.50, +0.66] | 0.55 / 0.93 | **mixed** |
| 12 | 0.62 / 0.48 | +3.54 [+3.31, +3.80] RISE | +0.70 [+0.55, +0.86] RISE | -1.01 [-1.16, -0.87] | -1.71 [-1.94, -1.50] | -1.39 [-1.61, -1.18] | +0.05 [-0.02, +0.12] | 0.50 / 0.93 | **mixed** |
| 13 | 0.62 / 0.48 | +3.87 [+3.61, +4.13] RISE | +1.19 [+1.05, +1.33] RISE | -1.81 [-2.03, -1.60] | -3.00 [-3.31, -2.70] | -2.59 [-2.89, -2.32] | +0.32 [+0.27, +0.38] | 0.45 / 0.95 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0822, ' to' -0.0775, ' in' -0.0741, ' is' +0.0390, ' the' +0.0361, ' into' -0.0307
- site 9 T - I: ' was' +0.0776, ' in' -0.0651, ' to' -0.0410, ' is' +0.0303, ' on' -0.0298, ' with' -0.0265
- site 11 T - I: ' in' -0.0713, ' the' +0.0692, ' his' +0.0525, ' her' +0.0414, ' to' -0.0361, ' on' -0.0322
- site 12 T - I: ' the' +0.1480, ' it' +0.1289, ' in' -0.1029, ' him' +0.0849, ' her' +0.0746, ' his' +0.0720
- site 13 T - I: ' it' +0.6801, ' in' -0.1152, ' to' -0.0820, ' with' -0.0511, ' the' +0.0506, ' on' -0.0450

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +1.31 [+1.03, +1.60] → **passes**; numerator +0.47 [+0.14, +0.77], PREP -0.85 [-1.10, -0.61], O +0.06 [-0.24, +0.36].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 9 | 0.53 / 0.26 | +1.62 [+1.46, +1.78] RISE | +1.34 [+1.16, +1.54] RISE | +0.59 [+0.46, +0.74] | -0.75 [-0.85, -0.66] | +0.21 [+0.07, +0.34] | +0.70 [+0.58, +0.83] | 0.44 / 0.93 | **mixed** |
| 11 | 0.58 / 0.26 | +2.44 [+2.23, +2.64] RISE | +1.03 [+0.91, +1.17] RISE | +0.31 [+0.20, +0.43] | -0.72 [-0.80, -0.65] | +0.09 [-0.04, +0.20] | +0.38 [+0.31, +0.44] | 0.36 / 0.93 | **mixed** |
| 12 | 0.62 / 0.26 | +3.64 [+3.40, +3.87] RISE | +0.50 [+0.43, +0.57] RISE | -0.96 [-1.09, -0.83] | -1.46 [-1.64, -1.29] | -0.85 [-1.00, -0.70] | +0.07 [+0.05, +0.10] | 0.32 / 0.93 | **mixed** |
| 13 | 0.62 / 0.26 | +3.81 [+3.55, +4.06] RISE | +0.57 [+0.48, +0.66] RISE | -2.16 [-2.37, -1.96] | -2.73 [-2.99, -2.48] | -2.03 [-2.26, -1.81] | +0.08 [+0.06, +0.11] | 0.24 / 0.95 | **mixed** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 9 1.00, site 11 1.00, site 12 1.00, site 13 0.99

Tokens driving the effects (probability change):

- natural good - bad: ',' +0.0586, ' in' -0.0490, '.' +0.0400, ' through' -0.0267, ' and' +0.0261, ' around' -0.0241
- site 9 T - I: ' in' -0.0558, ',' +0.0558, '.' +0.0347, ' on' -0.0337, ' the' +0.0240, ' and' +0.0207
- site 11 T - I: ' in' -0.0549, ' the' +0.0485, ' on' -0.0305, ',' +0.0222, ' at' -0.0161, ' a' +0.0158
- site 12 T - I: ' it' +0.1436, ' the' +0.1179, ' in' -0.0706, ' him' +0.0670, ',' -0.0621, '.' -0.0384
- site 13 T - I: ' it' +0.6397, ',' -0.0897, ' in' -0.0801, '.' -0.0546, ' the' +0.0531, ' and' -0.0458

