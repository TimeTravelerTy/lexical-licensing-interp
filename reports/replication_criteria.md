# Replication criteria T1–T3 and S1, per model

Criteria declared in `reports/olmo/finals_rules.md` §3 (2026-11-16, before any OLMo transfer run), applied to all three
models from their stored tables. Classes are the declared ones of each source table (passive test: RISE / NO RISE /
unresolved against δ_s; filler-gap: round-4 reading rule). Nothing is recomputed here.

Declared secondary sites (0.8 / 0.9 rules): Pythia 8 / 12, Gemma 5 / 8, OLMo 9 / 11. For T3 the declared sites are
the primary and all declared secondaries (Pythia 8, 12, 17; Gemma 5, 8, 12, 15; OLMo 9, 11, 12, 13). Exploratory
sites never count.

## Summary

| Criterion | Pythia-1.4B | Gemma-3-1B | OLMo-2-1B |
|---|---|---|---|
| T1 passive *by* RISE at both rule sites | **pass** | **pass** | **fail** (NO RISE at 9 and 11) |
| T2 OR and TC D(O) RISE at both rule sites | **pass** | **pass** | **pass** |
| T3 filler-gap licensing-like, in range, at a declared site | **pass** (matrix, 8) | **fail** | **pass** (embedded and matrix, 9) |
| S1 staging with v | **pass** | **pass** (no surface site to 58% depth) | **pass** |
| S1 staging with the split-edit on-axis piece | **pass** | not run | not run |

Paper rule: a model "replicates" the transfer claim if T1 and T2 pass. Pythia and Gemma do. OLMo does not (T1 fails),
so the declared fallback applies: "Pythia and Gemma, with a partial third-family check".

## Evidence

### T1 and T2 (`reports/gemma/transfer.md`; OLMo `reports/olmo/transfer/passive_test.md`, `constructions.md`. D on bad
bases, T − I donors, nats, 95% CI)

| Model | Site | D log P(" by") | D(O) OR | D(O) TC |
|---|---:|---|---|---|
| Pythia-1.4B | 8 | +0.72 [+0.63, +0.82] RISE | +1.41 [+1.23, +1.61] RISE | +0.92 [+0.78, +1.08] RISE |
| Pythia-1.4B | 12 | +0.81 [+0.72, +0.89] RISE | +2.66 [+2.42, +2.90] RISE | +2.05 [+1.84, +2.25] RISE |
| Gemma-3-1B | 5 | +0.56 [+0.44, +0.67] RISE | +2.27 [+1.94, +2.59] RISE | +1.14 [+0.96, +1.32] RISE |
| Gemma-3-1B | 8 | +0.63 [+0.53, +0.74] RISE | +1.67 [+1.43, +1.89] RISE | +1.30 [+1.12, +1.48] RISE |
| OLMo-2-1B | 9 | +0.43 [+0.37, +0.49] NO RISE | +1.77 [+1.57, +1.98] RISE | +1.62 [+1.44, +1.80] RISE |
| OLMo-2-1B | 11 | +0.14 [+0.08, +0.21] NO RISE | +2.60 [+2.35, +2.86] RISE | +2.22 [+2.01, +2.43] RISE |

### T3 (filler-gap, primary population: non-DAS, half B, fp32)

Natural gates pass in all three models and both experiments (`reports/round5/results.md` C8;
`reports/gemma/transfer/fillergap_gates.md`; `reports/olmo/transfer/fillergap_gates.md`).

In-range readings at declared sites:

| Model | Site | embedded wh | matrix wh |
|---|---:|---|---|
| Pythia-1.4B | 8 | mixed | **licensing-like** |
| Pythia-1.4B | 12 | surface | surface |
| Pythia-1.4B | 17 | surface | surface |
| Gemma-3-1B | 5 | surface | surface |
| Gemma-3-1B | 8 | mixed | surface |
| Gemma-3-1B | 12 | mixed | surface |
| Gemma-3-1B | 15 | mixed | surface |
| OLMo-2-1B | 9 | **licensing-like** | **licensing-like** |
| OLMo-2-1B | 11 | mixed | mixed |
| OLMo-2-1B | 12 | mixed | mixed |
| OLMo-2-1B | 13 | mixed | mixed |

Embedded wh is licensing-like in range at no declared site in Pythia or Gemma; in OLMo it is, at site 9 (fill O level
vs T −0.01 [−0.50, +0.38], not above). Outside the declared sites:
- Pythia: matrix licensing-like at 4 and 10 (sites added after the declaration).
- Gemma: licensing-like only at exploratory sites (embedded and matrix in range at 2; T − I at 1).

### S1 (passive depth curve, `reports/gemma/depth_curve.md`; Pythia 14–17 from `reports/gemma/transfer.md`; OLMo
`reports/olmo/depth_curve/olmo_passive_test.md` and `reports/olmo/transfer/passive_test.md`)

| Model | abstract | mixed | surface | unresolved |
|---|---|---|---|---|
| Pythia-1.4B (v) | 1–8 | 9–13 | 14, 16, 17 | 0 |
| Pythia-1.4B (on-axis, `reports/pythia/split_edit.md`) | 6, 8, 12 | — | 17 | 4 |
| Gemma-3-1B (v) | 1, 3 | 4–15 | none | 0, 2 |
| OLMo-2-1B (v) | 0, 4 | 5, 6 | 7–15 | 1, 3 (2: nothing moves) |

Gemma meets S1 because it has abstract sites and no surface site, so the ordering condition holds trivially. Every
Gemma site from 4 to 15 (to 58% depth) is mixed.

Depth (site / layers): Pythia abstract to 0.33, surface from 0.58 (24 layers); Gemma abstract at 0.04 and 0.12
(26 layers); OLMo abstract at 0.00 and 0.25, surface from 0.44 (16 layers).

### OLMo-2-1B, the T1 numbers

- δ_s is 0.2 × the site's active O effect (2.65 nats at 9, 2.88 at 11), so δ = 0.53 and 0.58.
- D(by) at 9 is +0.43 [+0.37, +0.49]: the CI is above 0, and it is 0.90 of the natural good − bad gap in log P(" by")
  (0.48 nats). The 90% CI lies within ±δ, so the declared class is NO RISE.
- D(by) at 11 is +0.14 (0.30 of the natural gap). At 12 and 13 it is FALL (−0.74, −1.09).
- Over the depth curve, *by* is RISE at sites 0, 4, 5 and 6 (0.24–0.57 nats, 0.50–1.19 of the natural gap) and NO
  RISE from 7 to 11.
- For comparison (same ratio): Pythia 1.01 at 8 and 1.13 at 12; Gemma 1.36 at 5 and 1.53 at 8.

Runs: TSUBAME jobs 9415867–9, 9417175–9 (finals), 9417181 (prep), 9417183 (main), 9417187 (fg), 9417188 (depth
curve); checkout on /gs/bs after the /gs/fs group quota filled. Finals gate: all 16 sites within 0.03 of the sweep
(max 0.019, site 8; `results/olmo/das/transfer/finals_gate.json`).
