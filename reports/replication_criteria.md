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
| S1 staging with the split-edit on-axis piece | **pass** (abstract 6, 8, 12; surface 17; sampled 4, 6, 8, 12, 17) | **fail** (no abstract site) | **fail** (no abstract site) |
| S1 staging with u_md | **pass** (abstract 4, 6, 8, 12; surface 17) | **pass**, thin (abstract only at 0; surface 6–10, 13) | **fail** (no abstract site) |
| T1 with the on-axis piece | **pass** | **fail** (NO RISE at 5 and 8) | **fail** (NO RISE at 9 and 11) |
| T1 with u_md | **pass** | **fail** (unresolved at 5 and 8) | **fail** (NO RISE at 9 and 11) |
| R-A1 off-axis carries v's object push (rule sites) | held (Pythia A1) | **held** (10 / 12 RISE sites) | **not held** (2 / 11) |
| R-A2 on-axis carries v's *by* (rule sites) | held (Pythia A2) | **not held** (5 / 15) | n/a at rule sites; 3 / 4 elsewhere |
| FI decision at the rule sites | mixed (0.54 / 0.50) | mixed (0.17 / 0.70) | low (0.34 / 0.33) |
| T2 with u_md: OR and TC D(O) RISE at both rule sites | **pass** | **pass** | **pass** |
| T2 with u_md, own-effect margin (hi95) | **pass** | **pass** | **pass** |
| T3 with u_md: in-range licensing-like at a declared site | **fail** (mixed at 8, 12, 17) | **pass** (embedded 12, 15; matrix 15) | **pass** (embedded 9) |

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

Labels: "surface" = O RISE and *by* not RISE, which includes *by* unresolved; "fails T1" means *by* missed the declared
margin at the rule sites, not that *by* did not move (see the CIs).

- **Gemma: the object push rides mostly on w, as in Pythia (R-A1 held at 5 and 8).** The off-axis share of v's D(O)
  is 0.72–0.84 at every RISE site, and on-axis O is NO RISE through site 11. Unlike Pythia, w also carries much of
  *by*: the on-axis share of D(by) is 0.16–0.71 (0.33 at 5, 0.61 at 8; R-A2 not held). Gemma's v is less aligned with
  u (|a| ≈ 0.4–0.7 from the per-basis |a|), so the on-axis piece is small. u_md moves *by* by +0.36 to +0.50 at every
  site 1–10 (0.65–0.96 of v's), but straddles δ at the rule sites. Gemma's u_md O is RISE from 6.
- **OLMo: the object rise is not mainly off-axis (R-A1 not held at 9 and 11).** Writes along u raise O too: u_md O is
  RISE from 7, on-axis O from 8. The off-axis share of v's O is 0.48–0.60 at 7–11 and grows with depth (0.66 at 12–13,
  0.84–0.90 at 14–15). So both pieces raise O, with depth-dependent shares; OLMo's surface outcome is not an off-axis
  artefact the way Pythia's site-12 O rise is. A u_md write is not shown to change only the class variable, so this is
  a statement about the u direction, not about a pure class feature.
- **OLMo's deep *by* FALL is mostly non-additive.** v's D(by) drops from +0.43 (9) to +0.14 (11) and FALLs at 12–15.
  Alone, the on-axis piece stays positive (+0.07 to +0.17 at 12–15) and the off-axis piece is negative (−0.32 to
  −0.79), but full − on − off is −0.59 to −0.91 [CIs below 0] at 12–15: most of the FALL appears only when both
  pieces are written together. u_md (read and write u) keeps *by* at +0.41 / +0.43 at 9 / 11 (3× v's at 11) and
  +0.18 / −0.09 / −0.06 at 12–14 (−0.33 at 15).
- **OLMo early sites (0–6): *by* exceeds O along the class axis.** u_md D(by) +0.22 to +0.49 vs D(O) +0.01 to +0.40
  at 1–6, and on-axis D(by) exceeds D(O) at every site 0–6. D(by) stays under v's δ, so the declared class is NO RISE
  / unresolved.
- **FI** (an approximate disattenuation; it exceeds 1 where split-half reliability is low, e.g. OLMo 1.51 at 0,
  Pythia 1.69 / 1.43 at 0 / 1, so it is not a calibrated share; raw cosines and split-half reliabilities are in
  `summary_frame.csv`). OLMo falls from 0.85–1.5 at 0–3 to 0.33 at 11 and 0.10 at 15, the shape Pythia shows (0.98–1.69 at
  0–4, 0.54 / 0.50 at 8 / 12). Decision at OLMo's rule sites: "low". Gemma is not monotone: 0.17 at 5, 0.70–0.87 at
  6–9, ≈ 0 at 14–15. B1 fails at 14 / 26 Gemma sites and at 13–15 in OLMo. Active and passive axes come from
  different verb sets and frames, so the cosine is not a pure frame comparison; the "one direction every
  construction reads" wording rests on transfer, not on FI.

### Exploratory: a margin scaled to each condition (`reports/selfscaled_passive_classes.md`)

Chosen after seeing the results, so labelled exploratory. The declared δ holds the on-axis piece and u_md to v's active
O effect, although they push less on actives too (u_md's active D(O) is 0.38–0.69 of v's in Gemma, 0.37–0.78 in OLMo,
0.59–0.79 in Pythia). Re-classing with δ_c = 0.2 × the condition's own active D(O), same classifier, stored CIs (full v
reproduces the declared outcomes). Pythia was sampled at 4, 6, 8, 12, 17 only.

| Model | on-axis | u_md |
|---|---|---|
| Pythia | abstract 4, 6, 8, 12; mixed 17 | abstract 4, 6, 8, 12; surface 17 |
| Gemma | abstract 0–4, mixed 5–15 | abstract 0, mixed 2–15 |
| OLMo | abstract 1–5, mixed 6–10, surface 11–15 | abstract 1–4, mixed 6–11, surface 12–15 |

Caveats (Codex review, 2026-11-17, checked):
- δ_c treats the condition's active point estimate as exact. At OLMo u_md site 11, *by* (+0.4349) clears δ_c
  (0.4345) by 0.0004 nats; a multiplier of 0.201 instead of 0.200 makes it surface. Site 9 clears by 0.023.
- Where a condition's active effect is tiny, δ_c is tiny (Gemma on-axis at 1: active D(O) 0.09, δ_c 0.019), which
  makes RISE easy (and NO RISE hard). The Gemma on-axis "abstract 0–4" rests on such margins.
- The reading is transfer relative to each condition's own active effect, not comparable absolute strength.

So the OLMo statement is limited to: along u, OLMo's passive outcome goes abstract-leaning (1–4) → mixed → surface
(12–15), losing the abstract reading earlier than Pythia (OLMo after 0.25–0.31 of depth; Pythia abstract at 0.50).
Its first surface site under this margin (12/16 = 0.75) is not shallower than Pythia's (17/24 = 0.71). u_md *by* at
OLMo 11 is a margin-edge case, not a RISE to report.

## u_md transfer to OR, TC and filler-gap (rules: `reports/umd_transfer_rules.md`, declared before the run)

Reports: `reports/{pythia,gemma,olmo}/umd_transfer/` (`readings.md`, `constructions.md`, `inrange.md`,
`fillergap.md`). TSUBAME jobs 9429477–80 (runs), 9429530–2 (readings). Every check passes: the u_md bases map onto
v's bases exactly; the natural passes (constructions, in-range, filler-gap) reproduce v's runs with max difference 0.0;
the filler-gap `gates.json` equals v's; the re-analysed v patches reproduce the stored v D_bad(O) to 1e-15, so the
u_md / v ratio is paired on identical bootstrap draws. Pythia was run at 4, 6, 8, 12, 17 (where u was stored).

### T2: OR and TC transfer along u_md

u_md moves O on OR and TC bad bases at every site in all three models (CIs above 0). D(O) is RISE against v's δ at
both rule sites everywhere, and also against the own-effect margin taken at its upper bound.

| Model | Site | OR: u_md D(O) | u_md / v | TC: u_md D(O) | u_md / v |
|---|---:|---|---|---|---|
| Pythia | 8 | +0.93 [+0.78, +1.09] RISE | 0.66 [0.62, 0.70] | +0.52 [+0.41, +0.65] RISE | 0.57 [0.50, 0.62] |
| Pythia | 12 | +1.38 [+1.21, +1.56] RISE | 0.52 [0.48, 0.56] | +0.81 [+0.68, +0.96] RISE | 0.40 [0.35, 0.44] |
| Gemma | 5 | +0.68 [+0.37, +0.99] RISE | 0.30 [0.17, 0.42] | +0.56 [+0.35, +0.78] RISE | 0.49 [0.33, 0.65] |
| Gemma | 8 | +0.78 [+0.56, +1.02] RISE | 0.47 [0.36, 0.58] | +0.70 [+0.52, +0.88] RISE | 0.54 [0.42, 0.66] |
| OLMo | 9 | +1.34 [+1.17, +1.52] RISE | 0.75 [0.72, 0.79] | +1.17 [+1.03, +1.32] RISE | 0.72 [0.69, 0.76] |
| OLMo | 11 | +1.66 [+1.48, +1.85] RISE | 0.64 [0.61, 0.66] | +1.43 [+1.28, +1.60] RISE | 0.64 [0.61, 0.67] |

Over all sites the ratio is 0.40–0.84 (Pythia), 0.30–1.10 (Gemma), 0.64–0.82 (OLMo), in the range of u_md's own
active ratio to v (0.59–0.79, 0.38–0.69, 0.37–0.78). So the training-free class direction carries the OR / TC
transfer at about the strength it has on actives. Outside the rule sites, TC is weaker: Pythia TC at 4 and 6 and Gemma
TC at 0–4 are NO RISE / unresolved against v's δ (positive CIs). Against the own-effect margin, Gemma TC at 1–4
becomes RISE; Pythia TC at 4 (unresolved) and 6 (NO RISE) and Gemma TC at 0 (unresolved) do not.

### T3: filler-gap along u_md

| Model | v (in range, declared sites) | u_md (in range, declared sites) |
|---|---|---|
| Pythia | matrix licensing-like at 8 | mixed / unresolved at 8, 12, 17 (both experiments) |
| Gemma | none | embedded licensing-like at 12, 15; matrix at 15 |
| OLMo | embedded and matrix at 9 | embedded at 9 |

The flips follow from push size against a two-sided rule, not from a different mechanism. "Licensing-like" needs the
unfilled frame's end punctuation to FALL against v's δ (a floor) and the filled frame's O level to stay not above T's
(a ceiling). u_md pushes less than v:
- Pythia: u_md's unfilled "." drop is −0.17 / −0.20 at 8 (NO RISE; v −0.71 FALL), so the floor fails.
- Gemma: v's filled O level is above T at 5 (embedded) and at 5 and 8 (matrix), so v reads surface there; u_md
  stays under the ceiling, and its "." drop is FALL at 12 and 15, so the reading becomes licensing-like.
Report T3 per instrument, with this explanation, and do not read the u_md flips as evidence for or against licensing.

### In-range profile (passive, OR, TC; `inrange.md`)

u_md mostly does not reproduce the natural good − bad profile in range: signs match almost everywhere, but its
effects are mostly 0.1–0.7 of the natural gap (outside the declared [0.5, 1.5]). Exceptions that reproduce
it: Gemma passive at 8 and 12 and TC at 8. OLMo's passive O sign fails at every site (u_md raises O on in-range
passives), consistent with u_md's passive O RISE from site 7 in the split edit.
