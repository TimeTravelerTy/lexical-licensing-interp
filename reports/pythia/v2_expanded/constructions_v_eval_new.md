# C9. Object relatives and tough constructions

Spec: `round3_plan.md`, C9. Run: `run_construction_test.py`; analysis: `analyze_constructions.py`. 64 primary pairs × 24 contexts per construction. D = change after a transitive minus an intransitive active donor on the bad item, nats; 95% CIs: three-way bootstrap (pairs within band, contexts, donor pairs). L_OR = log P(MAIN) − log P(PREP); L_TC = log P(END) − log P(PREP).

## Object relatives ("The house that John destroyed")

Natural gate: good − bad L = +3.16 [+2.68, +3.68] → **passes**; numerator +2.13 [+1.78, +2.49], PREP -1.03 [-1.26, -0.82], O +1.12 [+0.75, +1.50].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 8 | 0.49 / 0.63 | +1.59 [+1.34, +1.86] RISE | +2.36 [+2.10, +2.63] RISE | +1.73 [+1.50, +1.96] | -0.64 [-0.76, -0.52] | +0.28 [+0.01, +0.54] | +1.99 [+1.74, +2.24] | 0.76 / 0.87 | **mixed** |
| 12 | 0.53 / 0.63 | +2.70 [+2.44, +3.00] RISE | +1.11 [+0.94, +1.30] RISE | +0.44 [+0.27, +0.61] | -0.67 [-0.81, -0.54] | +0.15 [-0.05, +0.36] | +0.77 [+0.64, +0.91] | 0.70 / 0.88 | **mixed** |
| 17 | 0.57 / 0.63 | +3.59 [+3.27, +3.91] RISE | +1.32 [+1.13, +1.53] RISE | -1.06 [-1.29, -0.85] | -2.38 [-2.70, -2.09] | -2.09 [-2.37, -1.82] | +0.33 [+0.26, +0.40] | 0.44 / 0.93 | **mixed** |

Tokens driving the effects (probability change):

- natural good - bad: ' was' +0.0902, ' in' -0.0804, ' with' -0.0673, ' into' -0.0570, ',' +0.0444, ' over' -0.0413

## Tough constructions ("The house is easy to destroy")

Natural gate: good − bad L = +2.30 [+1.86, +2.77] → **passes**; numerator +1.06 [+0.78, +1.35], PREP -1.24 [-1.46, -1.03], O +0.14 [-0.22, +0.52].

| Site | δ / δ_L | D(O) | D(L) | D(numerator) | D(PREP) | D(" by") | swap good→bad: L | z gap / active | Reading |
|---:|---|---|---|---|---|---|---|---|---|
| 8 | 0.49 / 0.46 | +1.08 [+0.88, +1.28] RISE | +1.82 [+1.58, +2.08] RISE | +0.92 [+0.74, +1.12] | -0.90 [-1.05, -0.76] | +0.10 [-0.15, +0.34] | +1.18 [+1.02, +1.36] | 0.57 / 0.87 | **mixed** |
| 12 | 0.53 / 0.46 | +2.20 [+1.95, +2.46] RISE | +1.08 [+0.93, +1.24] RISE | +0.42 [+0.30, +0.55] | -0.65 [-0.77, -0.55] | +0.24 [+0.10, +0.39] | +0.58 [+0.49, +0.67] | 0.51 / 0.88 | **mixed** |
| 17 | 0.57 / 0.46 | +4.16 [+3.80, +4.50] RISE | +0.40 [+0.27, +0.52] unresolved | -1.28 [-1.48, -1.09] | -1.68 [-1.92, -1.45] | -1.43 [-1.66, -1.22] | +0.03 [+0.00, +0.05] | 0.24 / 0.93 | **unresolved** |

TC base-form gate ("John can read" vs "John can sleep", AUC over verb means along d_s): site 8 1.00, site 12 1.00, site 17 1.00

Tokens driving the effects (probability change):

- natural good - bad: ',' +0.1134, '.' +0.0961, ' around' -0.0637, ' with' -0.0471, ' in' -0.0458, ' on' -0.0408

