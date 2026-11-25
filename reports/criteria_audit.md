# Criteria audit: do the criteria test the claims? (Nov 25, before any expanded v2 output was read)

Review: Codex (gpt-6.2-sol, xhigh), with its factual points checked against the files (below). Prompted by three
readings of the 126-pair v / v2 chain (`reports/v2_replication_criteria.md`):
- OLMo's T1 "fails" while D(*by*) is 0.90–1.01 of OLMo's natural *by* gap.
- Gemma's T1 flips on 0.02 nats.
- T3 fails when a stronger write slightly overshoots transitives' natural O level.

This amendment replaces the criteria of `reports/olmo/finals_rules.md` §3 and the R-A rules of
`reports/split_edit_replication_rules.md` for the paper's main text. It applies to every instrument (v2, v, u_md,
the on- and off-axis pieces) and every model at once. The old criteria and their outcomes stay in the appendix
unchanged.

## What was wrong, claim by claim

| Claim | Old criterion | Problem |
|---|---|---|
| Passive *by* transfer (T1) | D(*by*) est ≥ δ and lo95 > 0, δ = 0.2 × the instrument's active **O** effect | Wrong unit. In OLMo δ (0.53–0.59) exceeds the whole natural *by* gap (0.48), so a patch can reproduce the full natural difference and fail. The point-estimate test flips on hundredths of a nat (Gemma 5). |
| OR / TC (T2) | D(O) RISE against the same δ | Ignores the construction's own natural behavior (amended Nov 25, below). Unpatched intransitive items mostly predict a preposition (Pythia OR: PREP 0.63, closure 0.04, O 0.02); transitives raise closure *and* O. The natural O gap in OR is large (+1.14 to +1.66 nats; TC +0.06 to +0.56), so an O rise is not evidence against licensing by itself; only O beyond the natural gap is. The licensing readout L (closure vs PREP) was secondary. |
| Filler-gap licensing (T3) | A conjunction of classes plus a ceiling: the filled-frame O level after the write may not exceed T's natural level ("not above") | Never requires the frame interaction itself: OLMo embedded 9 passed with v at an interaction of −0.04 [−0.31, +0.22]. Penalizes a stronger write. "Any declared site in either experiment" is a search over 8–16 cells. |
| Staging (S1) | ≥ 1 abstract site, every surface site deeper | Passes with zero surface sites (Gemma, v). "Surface" lumps *by* unchanged, unresolved and suppressed. |
| Split-edit dissociation (R-A1 / R-A2) | Standalone share of the full effect ≥ 0.5 (point estimate), only at sites where the full effect is RISE | The shares are standalone recoveries and overlap. At Pythia 8 with v2 the off-axis write alone recovers *more* of *by* (0.66) than the on-axis write (0.59). The pieces interact (full − on − off ≈ −0.20 nats). Eligibility depends on the outcome: Gemma 5 left R-A2 with v2 while its on-axis share stayed 0.32. |
| "A model replicates if T1 and T2 pass" | Binary verdict per model | Throws away effect sizes, and a verdict that flips on 0.02 nats is not a finding. |
| D / natural gap as "reproduces" | — | D is a T − I donor contrast. Recovery on a bad base is (T-patched − unpatched) / gap (OLMo v2 site 9: D ≈ 1.0 gap, T arm ≈ 0.9 gap). |

## Criteria (main text)

**Units.** *by* is scaled by its own natural good − bad gap G on the same items. G and D come from the same pair and
context draws, and D also resamples donors. In OR / TC, L and O are both read against the construction's own natural gaps (claim 2). Elsewhere O is scaled by the instrument's held-out active O effect A (δ_O = 0.2 ×
A, the stored δ). O's natural gap is about 0, and the O claims are about how much of the edit's own object push
reaches another frame. Every effect is also shown in nats. L readouts keep δ_L = 0.2 × their natural gap, as
declared.

**Classes (one classifier for all instruments):**
- Checked in this order (erratum below):
- *by*:
  - **RISE** if lo95(D − 0.2 G) > 0;
  - **FALL** (suppressed) if hi95(D) < 0;
  - **NO RISE** (equivalent) if hi90(D − 0.2 G) < 0 and lo90(D + 0.2 G) > 0;
  - **positive, magnitude unresolved** if lo95(D) > 0 otherwise;
  - **unresolved** otherwise.
- O:
  - **RISE** if lo95(D) > δ_O;
  - **FALL** if hi95(D) < −δ_O;
  - **NO RISE** if the 90% CI is within ±δ_O;
  - **positive, magnitude unresolved** if lo95(D) > 0 otherwise;
  - **unresolved** otherwise.
- Differences from before: the margin is tested on the CI, not the point estimate, and *by* gets its own unit.
- **Erratum (Nov 25, found while reading the expanded output).** As first committed, this list and `analyze_v2_expanded.py`
  checked "positive, magnitude unresolved" before NO RISE. A shift that is significantly above 0 but whose 90% CI lies
  inside the margin was then never NO RISE (OLMo v2 site 1: O 0.05 [0.01, 0.10] against δ_O 0.31). That contradicts
  the line above (the old classifier, `apt.classify`, tests equivalence before anything but RISE / FALL) and the
  class's name: inside the margin the magnitude is resolved. The order is now equivalence first. Only O cells change
  (and with them some outcomes: "unresolved" becomes "abstract" at Pythia v2 sites 4–9 and v site 8, Gemma v2 0–4 except 2,
  OLMo v2 1–4). This flips S1 for OLMo v2 and Pythia v from failing to holding. Codex argues the frozen order should stay primary, since the frozen text and code agreed. **Decision (Nov 25, author): equivalence first is primary**, because it is what the class means; the frozen order is reported as a sensitivity reading (`reports/v2_expanded_readings.md` §2 shows both), and the amendment is disclosed as made after the output was read.

**Per claim:**
1. **Passive *by* transfer.** Per (model, rule site, instrument), the *by* class plus D / G with its joint-draw CI. D / G is shown only where lo95(G) > 0; otherwise nats only. Also the T-arm recovery (T-patched − unpatched) / G, the quantity behind any "reproduces" wording. The text states the cells; no model-level verdict.
2. **OR / TC: does the edit reproduce the construction's natural profile?** (amended Nov 25, before any expanded output was read.) Per construction, rule site and instrument, with G_L and G_O the natural good − bad gaps on the same items, from the same draws as D:
   - **Licensing readout:** L RISE if lo95(D_L − 0.2 G_L) > 0. As before, it is read only where the natural L gate passes (lo95(G_L) > 0) and, for TC, the base-form gate passes.
   - **Object relative to natural:** O is *beyond natural* if lo95(D_O − G_O) > 0, and *not beyond natural* otherwise.
   - **Readings:**
     - *licensing-like (natural profile)*: L RISE, O not beyond natural;
     - *licensing plus object overshoot*: L RISE, O beyond natural;
     - *surface*: O beyond natural, L not RISE;
     - otherwise *unresolved*.
   - **Shown:** D_L / G_L and D_O / G_O with joint-draw CIs (a ratio only where lo95(G) > 0; OLMo's TC O gap is about 0, so there any O rise is beyond natural), and D(O) against δ_O as the transfer of the edit's object push.
   - **Why:** an unpatched intransitive repairs toward a prepositional gap ("the house that John emerged *from*"); transitives take closure. A transitive edit that lowers PREP and raises closure licenses the gap at the verb. An O rise up to the natural O gap is what the model's own transitives do in these frames, partly because "that" can open a complement clause.
3. **Filler-gap (own item set; not on the expanded set).**
   - Primary readout: the frame interaction J_O = D_nofill(O) − D_fill(O) at each declared site and experiment, with its paired CI. The filled frame should attenuate the object push. "Licensing-like" requires lo95(J_O) > 0.
   - Shown beside it: fill L, PREP and "." separately.
   - The filled-frame O level vs T's natural level is reported continuously. "Matches T's level" needs its 90% CI within ±δ_O, as a separate calibration statement.
   - Every declared cell is shown; a claim names its cells. No any-cell verdict.
4. **Staging (S1).**
   - Per site:
     - abstract = *by* RISE and O NO RISE;
     - mixed = both RISE;
     - surface = O RISE with *by* split into NO RISE (unchanged), FALL (suppressed) or other (unresolved).
   - S1 holds if the curve has ≥ 1 abstract and ≥ 1 surface site and every abstract site is shallower than every surface site.
   - The curve uses every frozen site of the run (no site selection). It is a depth profile of intervention responses, with directions refit per layer.
5. **Split-edit dissociation.**
   - At each split-edit site, two paired contrasts of standalone recovery:
     - Δ_by = share_on(*by*) − share_off(*by*), where the full *by* is RISE;
     - Δ_O = share_off(O) − share_on(O), where the full O is RISE.
   - The dissociation holds at a site if every applicable contrast has lo95 > 0 and at least one applies.
   - Shown beside it: both standalone shares and the interaction full − on − off. An exact split of the write need not split the response.
6. **New verbs.**
   - Claim 1 applied to the new-only plain cohort.
   - Retention, separately: the joint-draw D_new − 0.5 D_orig (amendment 4 of `expanded_eval_rules.md`); *inconclusive* if its CI crosses 0, *below half* if it lies below.
   - The natural gaps of both cohorts are reported.

**Thresholds.** 0.2 × G mirrors the declared δ_L. 0.2 × A is the stored δ. Both are conventions, stated as such, and the
sensitivity rows use the upper 95% bound of G.

**C3 wording (LEACE), unchanged numbers.** The headline values are pooled reductions of the natural *by* gap under
cross-fitted erasure. They are shown with the natural and erased gaps. "About half or more" describes the point
estimates; the Pythia and OLMo lower bounds are below half.

## Where this differs from Codex

- **O margin.** Codex proposed a fixed ±0.20-nat O margin. In passives (claim 1, S1) we keep 0.2 × the instrument's active O effect, tested on the CI: S1 is a claim about how much of an edit's object push survives a frame change. In OR / TC, O is read against the construction's natural O gap (claim 2). Codex's "natural O gaps are about 0" holds for OLMo TC only. A fixed nat margin would treat an O shift that is 4% of a large active effect and one that is 40% of a small one alike. The passive natural O gap is about 0 (−0.22 in Pythia), so passive O cannot be gap-scaled; the active effect is the edit's own reference.
- **R-A dissociation.** Codex suggested testing "removing the on-axis piece costs more than half" as well. We use the on − off standalone contrast, which needs no further runs, and show the interaction.

## Forking paths (disclosed in the paper's appendix)

- The revision was informed by:
  - the 126-pair v and v2 readings;
  - the exploratory gap-scaled *by* margin (`reports/gapscaled_by_classes.md`, chosen after OLMo's T1 failed);
  - the expanded v / u_md readings (`reports/expanded_eval_rules.md`).
- The expanded v2 outputs (`results/{m}/das_v2/expanded_eval/`) had not been read when this was committed.
- Not fully protected:
  - 59 of v2's 89 primary expanded pairs are originals whose 126-pair readings were seen.
  - 7 of the 30 new plain pairs share a lemma with the earlier 19 evaluation pairs, which were in the 126-pair chain. That leaves 23 lemma-fresh pairs, shown as their own row.
- So the expanded v2 reading is "held out from instrument fitting" for all 224 pairs and "unseen during rule revision" only as stated here.
- The old criteria's outcomes for v and v2 (126-pair chain) are in `reports/replication_criteria.md` and `reports/v2_replication_criteria.md`, and they go to the appendix beside the new ones.

## Checked from the files before adopting

- OLMo TC natural O gap +0.06 [−0.24, +0.36] (`reports/olmo/transfer/constructions.md`).
- The OLMo embedded site-9 interaction with v is −0.04 [−0.31, +0.22] while classed licensing-like. With v2, OLMo matrix 9 is −0.50 [−0.77, −0.24] and Pythia matrix 8 −0.19 [−0.40, +0.02] (`reports/*/v2/fillergap.md`).
- Standalone shares at Pythia v2 site 8: *by* on 0.59, off 0.66 (`reports/pythia/v2/split_edit.md`).
- OLMo v2 site 9: the T arm raises *by* by +0.43 against a 0.48 gap (`reports/olmo/v2/passive_test.md`).
- 7 of 30 new plain pairs share a lemma with `data/verb_expansion/eval_pairs.csv`.

## Codex points noted, not acted on now

- Simultaneous inference over the depth curve; we use a fixed contrast per site and an ordering claim.
- An fp32 check of Gemma's bf16 self-patch deviations.
- A fresh lemma-disjoint cohort to confirm the revised rules.
- Intervals condition on the fitted bases; this is a limitation sentence.
