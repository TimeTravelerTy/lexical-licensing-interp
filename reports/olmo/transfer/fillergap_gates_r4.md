# Round 4, Part B stage 1: natural gates

Spec: `plan.md`, B6/B7. Run: `run_fillergap.py natural`; analysis: `analyze_fillergap.py gates`. fill = with the wh filler, nofill = without. L = log P(".") − log P(PREP) (B6) or log P("?") − log P(PREP) (B7). Half A (12 contexts) selects I verbs; half B (12 contexts) tests. CIs: T and I verbs resampled independently, contexts jointly (2,000 draws, seed 17).

## B6 embedded wh ("I know that / what NAME V-ed")

**I-verb gate (half A):** 18 of 50 I verbs kept (filler raises PREP, CI above 0). Dropped: camp, crawl, creep, emigrate, frolic, hitchhike, languish, linger, loiter, lounge, lumber, lurk, mosey, nap, perish, reappear, reemerge, remain, saunter, skulk, sleep, slouch, slump, sneeze, tarry, tiptoe, toddle, toil, tremble, waddle, whiz, writhe.

| Contexts, I set | n I | interaction on L | interaction on O |
|---|---:|---|---|
| half B, kept I | 18 | +2.04 [+1.54, +2.49] | +0.57 [+0.24, +0.90] |
| all contexts, kept I | 18 | +2.01 [+1.54, +2.45] | +0.57 [+0.22, +0.90] |
| half B, all I | 50 | +1.29 [+0.93, +1.66] | +0.92 [+0.67, +1.16] |

**Natural gate (half B, kept I verbs): passes.**

Per matrix verb / auxiliary (half B, kept I): forgot: L +1.93 [+1.46, +2.37], O +0.46 [+0.13, +0.82]; heard: L +1.62 [+1.07, +2.18], O +0.80 [+0.35, +1.19]; know: L +2.44 [+1.95, +2.91], O +0.54 [+0.22, +0.87]; remember: L +2.17 [+1.63, +2.64], O +0.49 [+0.13, +0.86]

Prepositions the filler raises most after kept I verbs (half B, probability change): ' to' +0.062, ' about' +0.032, ' on' +0.028, ' for' +0.023, ' of' +0.020, ' upon' +0.015, ' at' +0.013, ' through' +0.013

Class × frame means (half B, log-probs):

| class | frame | O | PREP | . or ? | L | END |
|---|---|---:|---:|---:|---:|---:|
| I | fill | -4.27 | -0.63 | -3.94 | -3.31 | -2.79 |
| I | nofill | -3.45 | -0.70 | -4.38 | -3.68 | -3.25 |
| T | fill | -2.25 | -1.45 | -2.84 | -1.39 | -1.82 |
| T | nofill | -0.50 | -3.86 | -6.91 | -3.05 | -5.33 |

Natural z along d_s (all contexts; 0 = active intransitive, 1 = transitive level):

| site | T nofill | T fill | I nofill | I fill |
|---:|---:|---:|---:|---:|
| 9 | 1.00 | 0.78 | 0.07 | -0.02 |
| 11 | 1.02 | 0.68 | 0.05 | -0.02 |
| 12 | 1.00 | 0.57 | 0.02 | -0.06 |
| 13 | 1.03 | 0.47 | -0.00 | -0.09 |

## B7 matrix wh ("AUX NAME V" / "What aux NAME V")

**I-verb gate (half A):** 32 of 50 I verbs kept (filler raises PREP, CI above 0). Dropped: camp, creep, decamp, defect, frolic, loiter, lumber, lurk, mosey, nap, perish, remain, saunter, slog, tiptoe, waddle, whiz, writhe.

| Contexts, I set | n I | interaction on L | interaction on O |
|---|---:|---|---|
| half B, kept I | 32 | +1.92 [+1.58, +2.30] | +1.03 [+0.74, +1.33] |
| all contexts, kept I | 32 | +1.93 [+1.55, +2.29] | +1.03 [+0.77, +1.28] |
| half B, all I | 50 | +1.34 [+0.97, +1.73] | +1.06 [+0.78, +1.35] |

**Natural gate (half B, kept I verbs): passes.**

Per matrix verb / auxiliary (half B, kept I): Can: L +1.99 [+1.57, +2.42], O +1.58 [+1.31, +1.85]; Did: L +2.18 [+1.78, +2.60], O +0.89 [+0.65, +1.14]; Will: L +2.01 [+1.59, +2.42], O +0.96 [+0.72, +1.21]; Would: L +1.52 [+1.15, +1.87], O +0.68 [+0.40, +0.96]

Prepositions the filler raises most after kept I verbs (half B, probability change): ' for' +0.048, ' to' +0.040, ' on' +0.034, ' about' +0.025, ' at' +0.024, ' from' +0.018, ' of' +0.010, ' into' +0.009

Class × frame means (half B, log-probs):

| class | frame | O | PREP | . or ? | L | END |
|---|---|---:|---:|---:|---:|---:|
| I | fill | -4.98 | -0.76 | -3.59 | -2.83 | -3.10 |
| I | nofill | -3.86 | -0.91 | -3.10 | -2.19 | -2.67 |
| T | fill | -2.76 | -1.35 | -2.37 | -1.03 | -2.13 |
| T | nofill | -0.58 | -4.30 | -6.03 | -1.73 | -5.22 |

Natural z along d_s (all contexts; 0 = active intransitive, 1 = transitive level):

| site | T nofill | T fill | I nofill | I fill |
|---:|---:|---:|---:|---:|
| 9 | 0.92 | 0.72 | 0.01 | -0.04 |
| 11 | 0.96 | 0.63 | 0.04 | -0.00 |
| 12 | 0.96 | 0.52 | 0.04 | -0.04 |
| 13 | 0.98 | 0.43 | 0.02 | -0.05 |

B7 base-form gate ("NAME can V", T vs I verbs, AUC over verb means along d_s): site 9 1.00, site 11 1.00, site 12 1.00, site 13 1.00

