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
| S1 staging with the split-edit on-axis piece | **pass** (abstract 6, 8, 12; surface 17) | **fail** (no abstract site) | **fail** (no abstract site) |
| S1 staging with u_md | **pass** (abstract 4–12; surface 17) | **pass** (abstract 0; surface 6–10, 13) | **fail** (no abstract site) |
| T1 with the on-axis piece | **pass** | **fail** (NO RISE at 5 and 8) | **fail** (NO RISE at 9 and 11) |
| T1 with u_md | **pass** | **fail** (unresolved at 5 and 8) | **fail** (NO RISE at 9 and 11) |
| R-A1 off-axis carries v's object push (rule sites) | held (Pythia A1) | **held** (10 / 12 RISE sites) | **not held** (2 / 11) |
| R-A2 on-axis carries v's *by* (rule sites) | held (Pythia A2) | **not held** (5 / 15) | n/a at rule sites; 3 / 4 elsewhere |
| FI decision at the rule sites | mixed (0.54 / 0.50) | mixed (0.17 / 0.70) | low (0.34 / 0.33) |

Paper rule: a model "replicates" the transfer claim if T1 and T2 pass. Pythia and Gemma do. OLMo does not (T1 fails),
so the declared fallback applies: "Pythia and Gemma, with a partial third-family check". The secondary T1 instruments
(on-axis, u_md) do not rescue OLMo under the declared margin, so the paper sentence reserved for that case ("passive *by*
transfer holds along the class axis…") is not used. See "Split edit and u_md" below for what the curves show.

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

## Split edit and u_md (rules: `reports/split_edit_replication_rules.md`, declared before the run)

Reports: `reports/gemma/split_edit.md`, `reports/olmo/split_edit.md` (Pythia: `reports/pythia/split_edit.md` and
`results/pythia/das_direction_check`). TSUBAME jobs 9420462 (Gemma), 9420463 (OLMo); sites 0–15. Every check
passes in both models: held-out active IIA reproduces the stored finals exactly (max diff 0.0000), passive D(O) and
D(by) with full v match the stored values (all within 0.001 nats), write = read is bitwise, a·u + w = v to 1e-17.
Gemma's run drops one pair (transplant/ensue), as in its stored passive test.

### Passive D per condition at the rule sites (nats, 95% CI, declared class against δ_s = 0.2 × v's active O effect)

| Model | Site | δ_s | D(by) v | on | u_md | D(O) v | on | u_md |
|---|---:|---:|---|---|---|---|---|---|
| Gemma | 5 | 0.54 | +0.56 RISE | +0.18 NO RISE | +0.49 [+0.31, +0.69] unres. | +1.50 RISE | +0.09 NO RISE | +0.43 unres. |
| Gemma | 8 | 0.53 | +0.63 RISE | +0.38 NO RISE | +0.44 [+0.31, +0.57] unres. | +1.38 RISE | +0.40 NO RISE | +0.73 RISE |
| OLMo | 9 | 0.53 | +0.43 NO RISE | +0.32 NO RISE | +0.41 [+0.36, +0.47] NO RISE | +1.71 RISE | +0.73 RISE | +1.04 RISE |
| OLMo | 11 | 0.58 | +0.14 NO RISE | +0.31 NO RISE | +0.43 [+0.37, +0.50] NO RISE | +2.40 RISE | +0.80 RISE | +1.22 RISE |

### What the curves show

- **Gemma: the object push rides on w, as in Pythia (R-A1 held).** The off-axis share of v's D(O) is 0.72–0.84 at
  every RISE site, and on-axis O is NO RISE through site 11. Unlike Pythia, w also carries much of *by*: the on-axis
  share of D(by) is 0.16–0.54 (R-A2 not held). Gemma's v is less aligned with u (|a| ≈ 0.4–0.7), so the on-axis piece
  is small. u_md moves *by* by +0.36 to +0.50 at every site 1–10 (0.65–0.96 of v's), but straddles δ at the rule sites.
- **OLMo: the class axis itself predicts the object from site 7–8 (R-A1 not held).** On-axis and u_md D(O) are RISE
  from 8 and 7, and the off-axis share of O is only ≈ 0.5. So OLMo's surface behaviour is not mainly an off-axis
  artefact, unlike Pythia's. What is off-axis is v's deep *by* FALL: v's D(by) drops from +0.43 (9) to +0.14 (11) and
  FALL at 12–15, while u_md stays at +0.41 / +0.43 at 9 / 11 (3× v's at 11) and on-axis stays positive to 15. The
  off-axis piece alone is −0.32 to −0.79 at 12–15.
- **OLMo early sites (1–6): *by* leads O along the class axis.** u_md D(by) +0.22 to +0.49 vs D(O) +0.01 to +0.40, and
  on-axis D(by) exceeds D(O) at every site 0–6. But D(by) stays under v's δ, so the declared class is NO RISE /
  unresolved.
- **FI.** OLMo falls from ≈ 1 at 0–3 to 0.33 at 11 and 0.10 at 15: the same shape as Pythia (≈ 1 at 0–4, 0.54 / 0.50
  at 8 / 12), lower at the rule sites (decision "low": the paper says "one direction that every construction reads").
  Gemma is not monotone: FI 0.17 at 5 but 0.70–0.87 at 6–9, ≈ 0 at 14–15. B1 (cos above sign-flip 97.5th pct) fails
  at 14 / 26 Gemma sites and at 13–15 in OLMo.

### Exploratory: a margin scaled to each condition (`reports/selfscaled_passive_classes.md`)

Chosen after seeing the results, so labelled exploratory. The declared δ holds the on-axis piece and u_md to v's active
O effect, although they push less on actives too (u_md's active D(O) is 0.35–0.76 of v's). Re-classing with
δ_c = 0.2 × the condition's own active D(O), same classifier, stored CIs (full v reproduces the declared outcomes):

| Model | on-axis | u_md |
|---|---|---|
| Pythia | abstract 4–12, mixed 17 | abstract 4–12, surface 17 |
| Gemma | abstract 0–4, mixed 5–15 | abstract 0, mixed 2–15 |
| OLMo | abstract 1–5, mixed 6–10, surface 11–15 | abstract 1–4, mixed 6–11, surface 12–15 |

Under this margin, OLMo's class axis stages the way Pythia's does (abstract → mixed → surface), at shallower relative
depth (abstract to 0.25–0.31, surface from 0.69–0.75 of depth; Pythia abstract to 0.50). u_md's *by* is RISE at both
OLMo rule sites. This is reported as a description of OLMo's class axis, not as T1 passing.
