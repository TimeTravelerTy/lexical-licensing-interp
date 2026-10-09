# Round 4, Part B stage 1: natural gates

Spec: `plan.md`, B6/B7. Run: `run_fillergap.py natural`; analysis: `analyze_fillergap.py gates`. fill = with the wh filler, nofill = without. L = log P(".") − log P(PREP) (B6) or log P("?") − log P(PREP) (B7). Half A (12 contexts) selects I verbs; half B (12 contexts) tests. CIs: T and I verbs resampled independently, contexts jointly (2,000 draws, seed 17).

## B6 embedded wh ("I know that / what NAME V-ed")

**I-verb gate (half A):** 39 of 50 I verbs kept (filler raises PREP, CI above 0). Dropped: camp, creep, frolic, lounge, lurk, remain, saunter, skulk, sleep, tiptoe, writhe.

| Contexts, I set | n I | interaction on L | interaction on O |
|---|---:|---|---|
| half B, kept I | 39 | +1.55 [+1.13, +1.97] | +0.63 [+0.33, +0.93] |
| all contexts, kept I | 39 | +1.58 [+1.17, +1.96] | +0.63 [+0.34, +0.93] |
| half B, all I | 50 | +1.21 [+0.76, +1.61] | +0.77 [+0.49, +1.06] |

**Natural gate (half B, kept I verbs): passes.**

Per matrix verb / auxiliary (half B, kept I): forgot: L +1.63 [+1.22, +2.06], O +0.43 [+0.13, +0.74]; heard: L +1.13 [+0.75, +1.49], O +0.80 [+0.48, +1.09]; know: L +2.14 [+1.71, +2.58], O +0.54 [+0.22, +0.86]; remember: L +1.31 [+0.92, +1.69], O +0.77 [+0.46, +1.07]

Prepositions the filler raises most after kept I verbs (half B, probability change): ' for' +0.057, ' on' +0.052, ' about' +0.041, ' to' +0.027, ' over' +0.025, ' of' +0.015, ' from' +0.014, ' through' +0.010

Class × frame means (half B, log-probs):

| class | frame | O | PREP | . or ? | L | END |
|---|---|---:|---:|---:|---:|---:|
| I | fill | -4.48 | -0.45 | -4.44 | -3.98 | -3.50 |
| I | nofill | -3.30 | -0.72 | -4.50 | -3.78 | -3.34 |
| T | fill | -2.40 | -1.34 | -2.73 | -1.39 | -1.91 |
| T | nofill | -0.44 | -4.01 | -6.40 | -2.39 | -5.08 |

Natural z along d_s (all contexts; 0 = active intransitive, 1 = transitive level):

| site | T nofill | T fill | I nofill | I fill |
|---:|---:|---:|---:|---:|
| 4 | 0.99 | 0.89 | 0.02 | -0.01 |
| 6 | 0.97 | 0.85 | 0.04 | 0.00 |
| 8 | 1.00 | 0.82 | 0.07 | 0.01 |
| 10 | 1.00 | 0.75 | 0.05 | -0.06 |
| 12 | 1.02 | 0.73 | 0.05 | -0.09 |
| 14 | 1.00 | 0.60 | 0.02 | -0.11 |
| 16 | 1.01 | 0.48 | 0.00 | -0.11 |
| 17 | 1.03 | 0.44 | -0.01 | -0.08 |

## B7 matrix wh ("AUX NAME V" / "What aux NAME V")

**I-verb gate (half A):** 45 of 50 I verbs kept (filler raises PREP, CI above 0). Dropped: camp, creep, lumber, remain, whiz.

| Contexts, I set | n I | interaction on L | interaction on O |
|---|---:|---|---|
| half B, kept I | 45 | +1.45 [+1.04, +1.88] | +0.96 [+0.68, +1.24] |
| all contexts, kept I | 45 | +1.41 [+1.04, +1.80] | +0.95 [+0.70, +1.21] |
| half B, all I | 50 | +1.20 [+0.75, +1.64] | +0.97 [+0.70, +1.26] |

**Natural gate (half B, kept I verbs): passes.**

Per matrix verb / auxiliary (half B, kept I): Can: L +1.24 [+0.79, +1.71], O +1.27 [+0.82, +1.70]; Did: L +1.87 [+1.43, +2.33], O +0.92 [+0.64, +1.19]; Will: L +0.97 [+0.59, +1.33], O +0.89 [+0.63, +1.17]; Would: L +1.72 [+1.27, +2.20], O +0.76 [+0.48, +1.03]

Prepositions the filler raises most after kept I verbs (half B, probability change): ' for' +0.057, ' about' +0.048, ' to' +0.039, ' on' +0.033, ' as' +0.022, ' at' +0.020, ' into' +0.020, ' over' +0.013

Class × frame means (half B, log-probs):

| class | frame | O | PREP | . or ? | L | END |
|---|---|---:|---:|---:|---:|---:|
| I | fill | -4.58 | -0.53 | -3.94 | -3.41 | -3.19 |
| I | nofill | -3.61 | -1.07 | -3.92 | -2.85 | -2.42 |
| T | fill | -2.53 | -1.30 | -2.11 | -0.81 | -1.79 |
| T | nofill | -0.59 | -3.84 | -5.28 | -1.44 | -4.01 |

Natural z along d_s (all contexts; 0 = active intransitive, 1 = transitive level):

| site | T nofill | T fill | I nofill | I fill |
|---:|---:|---:|---:|---:|
| 4 | 0.84 | 0.83 | 0.02 | -0.00 |
| 6 | 0.88 | 0.82 | 0.05 | -0.01 |
| 8 | 0.89 | 0.78 | 0.04 | 0.01 |
| 10 | 0.91 | 0.76 | 0.04 | -0.04 |
| 12 | 0.94 | 0.76 | 0.06 | -0.04 |
| 14 | 0.94 | 0.62 | 0.02 | -0.10 |
| 16 | 0.95 | 0.50 | 0.02 | -0.07 |
| 17 | 0.96 | 0.47 | 0.01 | -0.03 |

B7 base-form gate ("NAME can V", T vs I verbs, AUC over verb means along d_s): site 4 1.00, site 6 1.00, site 8 1.00, site 10 1.00, site 12 1.00, site 14 1.00, site 16 1.00, site 17 1.00

