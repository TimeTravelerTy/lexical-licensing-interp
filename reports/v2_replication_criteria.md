# Replication criteria with v2, read three ways (Nov 25; rules: `reports/v2_retrain_rules.md`)

Criteria as declared in `reports/olmo/finals_rules.md` §3 and `reports/split_edit_replication_rules.md`; v's readings
are the stored ones (`reports/replication_criteria.md`). Each v2 reading is given under v2's own δ (the paper's
primary reading, declared), under the stored δ_v, and as the raw effect (*by* as a share of the natural good − bad
gap). Runs: `jobs/v2_chain.sh` (own δ: roots `results/{m}/das_v2`, reports `reports/{m}/v2/`) and `jobs/v2_dv.sh` +
the chain's analysis steps only (δ_v: roots `results/{m}/das_v2_dv`, reports `reports/{m}/v2_dv/`). Compact tables:
`results/v2_summary/passive_rule_sites.csv`, `results/{m}/das_v2*/{split_edit,fillergap,constructions,nonce}/`.

Rule sites: Pythia 8, 12; Gemma 5, 8; OLMo 9, 11 (T3 also at the primary and declared secondaries: Pythia 17; Gemma
12, 15; OLMo 12, 13).

## Checks

All pass in all three models:
- Natural (unpatched) readouts of every rerun equal the stored run's, max |Δ| = 0: passive, constructions, in-range
  and filler-gap, under both margins. Gemma's passive first failed (0.15 on 493 rows) because it ran at batch 512
  against the stored 1024; it was rerun at 1024.
- Filler-gap natural gates equal the stored gates (`cmp`).
- Split-edit passive D(O) and D(*by*) with full v2 equal the v2 passive run.
- Under δ_v, the u_md readings (u_md is not retrained) reproduce the stored u_md readings site for site (Gemma, OLMo).
- The δ_v copies differ from the own-δ roots only in each frozen config's margin block (`delta_v_stored.json`).
- Pythia's v2 split-edit analysis first ran without `--rule-sites 8,12` (the T1 and rule-site R-A1 / R-A2 fields were
  empty). It was rerun with them (jobs 9558145, 9558146); no other field changed.

Margins (δ = 0.2 × the instrument's held-out active O effect at the site):

| Model | Site | δ_v | δ_v2 (own) |
|---|---:|---:|---:|
| Pythia | 8 / 12 | 0.49 / 0.53 | 0.50 / 0.54 |
| Gemma | 5 / 8 | 0.54 / 0.53 | 0.57 / 0.55 |
| OLMo | 9 / 11 | 0.53 / 0.58 | 0.54 / 0.59 |

## Summary

| Criterion | Pythia: v / v2 own / v2 δ_v | Gemma: v / v2 own / v2 δ_v | OLMo: v / v2 own / v2 δ_v |
|---|---|---|---|
| T1 passive *by* RISE at both rule sites | pass / **pass** / pass | pass / **fail** / fail | fail / **fail** / fail |
| T2 OR and TC D(O) RISE at both rule sites | pass / **pass** / pass | pass / **pass** / pass | pass / **pass** / pass |
| T3 filler-gap licensing-like in range at a declared site | pass / **fail** / fail | fail / **fail** / fail | pass / **fail** / fail |
| S1 staging, full direction | pass / **pass** / pass | pass / **pass** / pass | pass / **pass** / pass |
| S1 staging, on-axis piece | pass / **pass** / pass | fail / **fail** / fail | fail / **fail** / fail |
| T1, on-axis piece | pass / **fail** / fail | fail / **fail** / fail | fail / **fail** / fail |
| R-A1 off-axis carries the object push | held / **held** / held | held / **held** / held | not held / **not held** / not held |
| R-A2 on-axis carries *by* | held / **held** / held | not held / **held at 8, n/a at 5** / same | n/a / **n/a** / n/a |
| Nonce: declared decisions at the declared sites | same in all three models (below) | | |

Own δ and δ_v give the same pass / fail everywhere. The margin is not what separates v and v2; the effects are.

**Under the declared rule** (v2 is primary; where v2 fails a criterion v passed, the claim narrows to v2 and v's pass
is reported beside it; `reports/v2_retrain_rules.md`), three readings change:

1. **Gemma T1 fails.** *by* at site 5 is +0.52 [+0.40, +0.65] with v2, against δ_v2 0.57 and δ_v 0.54: unresolved
   under both. v read +0.56 [+0.44, +0.67], RISE by 0.02 nats over its δ. Site 8 is RISE with both (v2 +0.72, v
   +0.63). So "T1 and T2 pass" (the paper's replication rule) holds with v2 in Pythia only; with v it held in Pythia
   and Gemma. The miss is narrow. Relative to the natural gap, Gemma 5 moves *by* by 1.26 of the gap with v2 (v:
   1.36), above Pythia's 1.09 at 8. Under the exploratory gap-scaled margin (below), v2 passes T1 in all three
   models.
2. **T3 fails in Pythia and OLMo.** The licensing-like reading needs the filled-frame O level after the in-range
   write to stay below T's natural level ("not above": hi90 < δ). With v2 the write overshoots slightly.

   | Model, site, frame | fill O level vs T: v | v2 |
   |---|---|---|
   | Pythia 8, matrix | +0.14 [−0.19, +0.52], not above | +0.36 [+0.01, +0.74], unresolved |
   | OLMo 9, embedded | −0.01 [−0.50, +0.38], not above | +0.24 [−0.29, +0.66], unresolved |
   | OLMo 9, matrix | −0.00 [−0.37, +0.36], not above | +0.41 [+0.04, +0.79], unresolved |

   The other conditions of the licensing-like reading still hold at these sites (nofill "." FALL, nofill O RISE,
   fill L RISE, fill PREP FALL). Pythia's undeclared licensing-like sites (matrix 4, 10 with v) are also gone with v2
   (matrix 4 mixed; 10 not in the v2 filler-gap site list). So with v2, T3 is licensing-like at no declared site in
   any model. Gemma has one exploratory site (embedded, site 1).
3. **Pythia T1 with the on-axis piece fails.** The on-axis *by* at 8 is unresolved with v2 (RISE with v). The
   on-axis share of v2's *by* at 8 is 0.59 (v: 0.71), at 12 it is 0.72 (v: 0.80), so R-A2 still holds. The on-axis
   S1 passes, with abstract at 12 only (v: 6, 8, 12; with v2, 6 and 8 are unresolved).

And one that does not change but needs a note:

4. **Gemma R-A2 "held at the rule sites" with v2 is not a gain.** R-A2 applies only where the full direction's *by*
   is RISE. With v2 that is site 8 alone (on-axis share 0.51, v: 0.61); at site 5 the on-axis share is 0.32 (v: 0.33),
   which would fail. Report it as "held at 8 (0.51), not applicable at 5", next to v's "not held".

## Evidence

### T1 and T2 (D on bad bases, T − I donors, nats, 95% CI; class under own δ, then δ_v)

| Model | Site | D(*by*) v | D(*by*) v2 | *by* / natural gap, v → v2 | D(O) OR, v2 | D(O) TC, v2 |
|---|---:|---|---|---|---|---|
| Pythia | 8 | +0.72 RISE | +0.78 [+0.68, +0.88] RISE / RISE | 1.01 → 1.09 | +1.30 [+1.12, +1.50] RISE | +0.81 [+0.67, +0.97] RISE |
| Pythia | 12 | +0.81 RISE | +0.83 [+0.74, +0.92] RISE / RISE | 1.13 → 1.17 | +2.66 [+2.41, +2.91] RISE | +2.01 [+1.81, +2.21] RISE |
| Gemma | 5 | +0.56 RISE | +0.52 [+0.40, +0.65] unres. / unres. | 1.36 → 1.26 | +2.33 [+2.00, +2.66] RISE | +1.08 [+0.90, +1.25] RISE |
| Gemma | 8 | +0.63 RISE | +0.72 [+0.61, +0.83] RISE / RISE | 1.53 → 1.75 | +1.87 [+1.63, +2.11] RISE | +1.36 [+1.20, +1.54] RISE |
| OLMo | 9 | +0.43 NO RISE | +0.48 [+0.42, +0.54] NO RISE / unres. | 0.90 → 1.01 | +1.80 [+1.60, +2.00] RISE | +1.62 [+1.46, +1.78] RISE |
| OLMo | 11 | +0.14 NO RISE | +0.26 [+0.20, +0.32] NO RISE / NO RISE | 0.30 → 0.54 | +2.81 [+2.57, +3.07] RISE | +2.44 [+2.23, +2.64] RISE |

T2 D(O) with v (stored): Pythia OR 1.41 / 2.66, TC 0.92 / 2.05; Gemma OR 2.27 / 1.67, TC 1.14 / 1.30; OLMo OR 1.77 /
2.60, TC 1.62 / 2.22. All RISE with both instruments under both margins.

Natural *by* gaps (same items): Pythia 0.71, Gemma 0.41, OLMo 0.48 nats. Relative to them, v2 moves *by* more than v
at 5 of 6 rule sites; the exception is Gemma 5. At OLMo 11 the effect nearly doubles (0.30 → 0.54) but stays under δ,
which exceeds OLMo's whole natural *by* gap.

**Exploratory, gap-scaled margin** (δ_g = 0.2 × the natural *by* gap; chosen after OLMo's T1 failed, as in
`reports/gapscaled_by_classes.md`): v2's *by* is RISE at all six rule sites, also with δ from the gap's upper 95%
bound (Pythia 0.21, Gemma 0.14, OLMo 0.15). With v, OLMo 11 was unresolved at the hi95 margin.

Gemma site 5 also changes in O: v2's passive D(O) there is +0.86 (v: +1.50); the active-trained effect is larger
(δ_v2 > δ_v). Site 5 is where v2 and v differ most (|cos| median 0.71, `reports/v2/instrument_measures.md`).

### T3 (filler-gap, primary population: non-DAS, half B, fp32; in-range readings at declared sites)

| Model | Site | embedded: v / v2 | matrix: v / v2 |
|---|---:|---|---|
| Pythia | 8 | mixed / surface | **licensing-like** / mixed |
| Pythia | 12, 17 | surface / surface | surface / surface |
| Gemma | 5 | surface / surface | surface / surface |
| Gemma | 8, 12, 15 | mixed / surface | surface / surface |
| OLMo | 9 | **licensing-like** / mixed | **licensing-like** / mixed |
| OLMo | 11, 12 | mixed / mixed | mixed / mixed |
| OLMo | 13 | mixed / mixed | mixed / surface |

Identical under δ_v. Gemma's matrix base-form gate is inconclusive along v2 at 1–3 (exploratory sites; check 6
exempts this gate).

### S1 (passive depth curve, full direction; Gemma and OLMo from the split edit at 0–15, Pythia from the depth curve)

| Model | abstract | mixed | surface | unresolved |
|---|---|---|---|---|
| Pythia, v | 1–8 | 9–13 | 14, 16, 17 | 0 |
| Pythia, v2 (own = δ_v) | 0–9 | 11–13 | 14, 16, 17 | 10 |
| Gemma, v | 1, 3 | 4–15 | none | 0, 2 |
| Gemma, v2 own | 0–4 | 6, 8–15 | 5, 7 | none |
| Gemma, v2 δ_v | 0–4 | 6–15 | 5 | none |
| OLMo, v | 0, 4 | 5, 6 | 7–15 | 1, 3 (2: nothing moves) |
| OLMo, v2 own | 1, 3, 4 | 6 | 7–15 | 0, 2, 5 |
| OLMo, v2 δ_v | 0, 1, 4 | 6 | 7–15 | 2, 3, 5 |

S1 passes in every model with both instruments. Gemma's v2 curve now has a surface site (5, 0.19 of depth) after
the abstract run 0–4 (0.00–0.15), so "no surface site to 58% depth" no longer describes it. Gemma site 5 is surface
because *by* is unresolved (above).

On-axis piece: Pythia abstract 12 / surface 17 (v: abstract 6, 8, 12); Gemma surface at 15 only (v: 14, 15); OLMo
surface 9–15 (v: 8–15), no abstract site in either. u_md (declared with δ_v) is unchanged; its δ_v2 version is a
recalibration and is not used for a claim.

### R-A1 / R-A2 at the rule sites

| Model | R-A1 off-axis share of D(O) (on-axis O class): v → v2 | R-A2 on-axis share of D(*by*): v → v2 |
|---|---|---|
| Pythia | 12: 0.67 (NO RISE) → 0.71 (NO RISE), held | 8: 0.71 → 0.59; 12: 0.80 → 0.72, held |
| Gemma | 5: 0.84 → 0.79; 8: 0.72 → 0.71 (NO RISE), held | 5: 0.33 → n/a (0.32); 8: 0.61 → 0.51 |
| OLMo | 9: 0.48 → 0.50; 11: 0.60 → 0.64 (on-axis O RISE), not held | n/a (*by* not RISE at 9, 11) |

Same under δ_v. With v2 the on-axis piece carries a smaller share of *by* in Pythia and Gemma than with v (by
0.08–0.12), and the off-axis piece a slightly larger share of O in Pythia. The split-edit illusion defense (on-axis
carries most of *by*, off-axis the O rise) still holds in Pythia, at 0.59–0.72 instead of 0.71–0.80.

### Nonce (declared sites; v2 bases, stored data and settings)

Declared decisions are the same with v and v2 in all three models:
- Pythia 6, 8: context-sensitive, verb-specific and raises *by* (Δ_matched v2 +0.127 / +0.165, v +0.129 / +0.151);
  C9 NP-driven at both.
- Gemma 1, 3, 5: no / no / *by* yes; C9 neither.
- OLMo 3: no. OLMo 5, 9: yes, yes, yes; C9 NP-driven.

Onset (descriptive, matched T − I z on passive probes, all v2 sites):
- Pythia: rises at 5–6.
- Gemma: jumps at 6 (0.23 of depth).
- OLMo: jumps at 5 (0.31), and the dip at 8 that fails OLMo's declared criterion (Settled 18) is reproduced (+0.016).

## What this changes in the paper (`docs/paper_outline.md`, C1 + C2)

- Transfer (T1 + T2) replicates with the primary instrument in Pythia. Gemma misses T1 at site 5 by 0.02–0.06 nats
  of margin, with v's pass beside it; OLMo misses as before, with a larger effect (0.54–1.01 of the natural gap).
  T2 holds in all three models with v2.
- Filler-gap T3: with v2, licensing-like at no declared site. The paper reports v's passes (Pythia matrix 8, OLMo 9)
  beside it and says what differs: v2's in-range write overshoots the natural O level of transitive verbs in the
  filled frame.
- Staging S1, R-A1 and the Pythia R-A2 illusion defense hold with v2. Nonce is unchanged.
