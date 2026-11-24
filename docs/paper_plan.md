# Transitivity mechanism paper: plan to ARR Jan 4

Started Nov 16, 2026; revised the same day against the repo (branch `paper`). This is the working guide for the rest of the project, and it is meant to change. Update the status table as results land, strike items that stop holding, and note which decision rule fired. Each experiment still gets its own plan file with rules and predictions, committed before it runs; this file sits above those.

## Goal and core claim

Submit a long paper (8 pages) to ARR by **Jan 4, 2027 AoE** (Jan 5, 20:59 JST) for ACL 2027, with Pythia-1.4B as the main model and Gemma-3-1B and OLMo-2-1B as the cross-family replications.

**Claim.** Language models compute verb transitivity as one shared, shallow feature at the verb, roughly "an NP follows this verb", set from both lexical identity and context. Construction-specific components then read it. Rare verbs get the feature right, and their deficit sits on the output side.

**Framing: H1/H2 answered as a split, not a winner.** The intro sets up lookup (H1) against computation (H2). The answer:

- The *format* is shared and computed. One direction is read by every construction, it is set from context for nonce verbs, and it encodes something shallower than argument structure.
- The *value* for a known verb is partly retrieved per verb. That is the part only an in-sample eraser removes.

With this framing, the necessity numbers answer the intro question instead of qualifying it.

**One number for "how shared"** (abstract): an eraser fit on other verbs removes half or more of a new verb's *by* gap: 0.53 [0.44, 0.67] in Pythia, 0.57 [0.45, 0.74] in OLMo and 0.83 [0.61, 1.22] in Gemma, with 268 verb pairs (Settled 16, 20; with the original 126: 0.41 / 0.44 / 0.76). An eraser fit on the verbs themselves removes almost all of it, in all three models.

## Status per claim

Evidence is on branch `paper` (merged Nov 16 from `olmo-replication`, `leace-necessity` and `das-direction-check`). Paths are relative to `reports/`.

| Claim | Pythia-1.4B | Gemma-3-1B | OLMo-2-1B | Evidence (Pythia / Gemma) | Left to do |
| --- | --- | --- | --- | --- | --- |
| Shared direction transfers across 5 constructions (passive, OR, TC, embedded and matrix wh) | done | done | done: OR/TC and both wh transfer, passive *by* does not (T1 fails; Settled 9) | R2-ptest, R4-A2, R4-B, R5-C8 / `gemma/transfer.md`; criteria `replication_criteria.md` | u_md transfer done (Settled 11: T2-umd passes in all three); OLMo passive *by* reported as curves (Settled 10) |
| Shallow, context-derived class (nonce onset ~site 6, NP-driven) | done | done | done: onset 0.31 of depth (site 5), NP-driven; declared criterion fails on a dip at site 8 (Settled 18) | R4-C8/C9, round 3 nonce / `gemma/nonce/summary.md` | OLMo nonce, sites 0–8 |
| Abstract → surface staging | done, redrawn with the on-axis piece: abstract 6–12, surface 17 (Settled 2) | **weak**: v mixed 4–15; u_md abstract only at 0, surface 6–10 and 13 (S1-umd pass, thin); on-axis has no abstract site (Settled 10) | v: abstract 0 and 4, surface from 7; on-axis and u_md: no abstract site under v's δ; with a self-scaled δ (exploratory) u_md abstract 1–4 / mixed 6–11 / surface 12–15 | R2-depth, `pythia/split_edit.md`; `{gemma,olmo}/split_edit.md`; `olmo/depth_curve/` | Decide the Figure 1 instrument per model |
| Necessity: verb-general + additional in-sample removal (LEACE) | done; cross-frame: active-fit erasers remove 0.82 (held-out verbs, matched centering) / 0.87 (in-sample) of the passive-fit removal (Settled 13); 268-pair set: removal 0.53 (Settled 16) | done: N1–N3 hold, removal 0.76; cross-frame majority, ρ 0.82 (Settled 17); 268-pair set: removal 0.83 (Settled 20) | done: N1–N3 hold, removal 0.44; cross-frame majority, ρ 0.94 (Settled 17); 268-pair set: removal 0.57 (Settled 20) | `pythia/leace_necessity.md`, `leace_permutation.md`, `leace_learning_curve.md`, `pythia/leace_crossframe.md`; R5-B5; R3-B5 (reverse DAS: a passive-trained direction is aligned with v early, \|cos\| 0.57 at site 4 falling to 0.22 at 17, and raises O in actives at 4–6, which converges with cross-frame LEACE) | Rank-k erasure (Pythia); LEACE on Gemma, OLMo, with the cross-frame test |
| Downstream readers: heads and neurons (how the class is read late) | done (with stored v at site 8): heads switch the *by* MLPs 11–17 between voices; conjunction neurons in MLPs 11 / 14 carry 0.65 of the *by* switch; also read the class in wh-gap frames; not necessary (R5-B5) | not run | not run | R3-B7, R4-A5, R4-D10, R4-D11, R4-E13T, R5-B5, R5-B6, R5-C9, R5-B7 | Work item 16: consolidate into one section and figure; no new runs required ("components", not "circuit") |
| Rare verbs: feature right, output-side deficit + edit | done; reverse DAS: a rare-only direction transfers to frequent verbs' passive *by* at 0.94–1.00 of v (Settled 15) | not run | not run | R6-C4 (31/32 rare singletons classified from z), R5-A1/A2, R5-A4; R2-freq / R3-A1 (the passive site-8 gap barely tracks Zipf, slope +0.065, and not behavior, with split-half R ≥ 0.985, so not a reliability failure); R4-D12 (translation gain tracks the released *by* margin, partial r +0.57, not frequency) | Second model, or scope it to Pythia |
| *by* transfer passes the illusion check | done | not run | not run | `pythia/das_direction_check.md`; R4-A4 (part of a passive-trained direction's *by* effect is a generic "by lever": shuffled direction +0.36, projecting it out halves the effect; cite as a caveat) | Rerun the direction check per model |
| Scale (Pythia-6.9B; behavior also 160M–12B) | behavior done: rare-verb deficit ≈ 2 nats at every size, intransitive side (R5-D10b, R6-D11, R6-A6); DAS only timed (R5-D10), never fit | — | — | `round5/d10b_suite.md`, `round5/d10_scaleup.md`, `round6/results.md` (D11) | 6.9B mechanism check before the freeze (work item 12) |
| Training dynamics (emergence order) | behavior done (R6-A6, R7-A8: 16 checkpoints); verb-site measures not run | n/a (no public checkpoints) | not run | `round6/a6_checkpoints*.md`, `round7/a8_trajectories.md` | Prediction, then 16 checkpoints per model |

Side splits: committed in `pythia/split_edit.md` §D (Settled 8).

## Settled (Nov 16)

1. **The necessity numbers hold up, with wide CIs.** Cross-fitted (5-fold) scrub 0–12 leaves 0.59 [0.35, 0.70] of the natural *by* gap. LOO leaves 0.55 [0.29, 0.67], and the learning curve is flat from n = 75 to 125 by point estimate. The CIs at n = 75 and 100 are wide ([0.03, 0.72]), so quote LOO and cross-fitted, not the curve, for the shared share. In-sample scrub 0–12 leaves 0.04 [−0.46, 0.28]. That result is credible because of the permutation control: all 20 label permutations leave 0.96–1.00, and the true-label CI lies below their 5th percentile. Always print the CI next to 0.04.
2. **Split edit (Pythia, `reports/pythia/split_edit.md`): staging survives, redrawn with the on-axis piece.** v's patch splits exactly into an on-axis part (along u_md) and an off-axis part. All checks pass, the stored v numbers reproduce exactly, and A1 and A2 held.
   - The off-axis part carries the O rise. At site 12 it carries 0.67 [0.64, 0.70] of v's passive D(O). The on-axis part's D(O) is +0.11 (NO RISE).
   - The on-axis part carries most of the *by* effect: 0.66–0.80 of v's at sites 4–12.
   - On-axis classes: abstract (O NO RISE, *by* RISE) at sites 6, 8 and 12, with *by* unresolved at 4. Surface (O RISE, *by* NO RISE) at 17. v alone gave mixed at 10 and 12.
   - The off-axis push is 2–4 natural sd along ŵ, and T and I barely separate along ŵ in actives (standardized difference ≤ 0.22). That is the illusion pattern: the O rise rides on a direction the classes do not use.
   - Decision rule fired: Figure 1 draws staging with the on-axis piece (u_md), with v as an upper bound. The same test is needed on Gemma (v mixed at every declared site) and OLMo.
3. **u_md is the weaker causal instrument on actives.** Its held-out active IIA is 0.30–0.41 of v's at sites 4–12, and its Δ log P(O) is 0.59–0.70 of v's. Use u_md as co-primary for *direction* claims (passive *by* transfer, staging) and report ratios to v; do not present its IIA as a replacement for v's. Mean ablation: v removes 0.37–0.41 of the passive *by* gap at sites 4–12, u_md 0.27–0.37, random directions about 0.
4. **Gemma ties input and output embeddings** (`scripts/model_arch.py`; `--fp32-unembed` unties the head). In Gemma, "the deficit lives in the output rows" means editing rows that are also input embeddings. Port the output-side edit on the untied head only, and say so.
5. **The LEACE / u_md cosine gap was a frame difference, and the class axis becomes frame-specific with depth.**
   - B0: the LEACE in-sample edit vector is the passive class mean difference (|cos| ≥ 0.998 at every site).
   - cos(u_act, u_pas) is 0.24–0.51 at every site, always above the sign-flip reference (B1 held).
   - Against the split-half ceilings, FI 0.98–1.69 at sites 0–4 (an approximate correction; values > 1 mean it is not a calibrated share), 0.78 at 6, 0.54 at 8, 0.50 at 12, 0.37 at 17 and 0.30 at 23.
   - The declared decision is "mixed" (site 12 is 0.497).
   - Reading for the paper: the class axis is shared across frames at the early (abstract) sites and becomes increasingly frame-specific with depth. Say "one direction every construction reads, frame-invariant early". Do not say "one class axis everywhere".
   - Patching along the passive axis (C) moves *by* by +0.23–0.33 with no O effect at any site, but it never reaches RISE, and it is weak on actives. The causal transfer evidence stays with v and u_md.
6. **Single-token and multi-token verbs separate sharply at site 20** (cross-fitted edit |cos| 0.69 vs 0.03). So the single-token × frequency confound is a concrete limitation, not a hypothetical one.
7. **TSUBAME:** `gpu_h` (half a GPU) is enough for every 1B-scale job and starts much sooner than `gpu_1`. Paper runs use the `lli_paper` checkout (branch `paper`). The group's /gs/fs share (3 TB) filled on Nov 16 (other members' data; ours is 69 GB), so new runs use `/gs/bs/tga-sip_arase/tyrone/lli_paper` (HDD, ~4.5 TB free).
8. **Suppression framing holds (D1).** Bad passives gaining *by* account for most of every gap closure:
   - 0.71–0.84 of the v and u_md mean ablation at sites 4–12;
   - 0.71–0.81 of in-sample LEACE;
   - 0.78–0.89 of cross-fitted LEACE (0.89 [0.83, 0.94] for scrub 0–12).

   So class information mainly suppresses *by* after intransitives, and this goes in the paper as the reading of the necessity result.
9. **Replication criteria applied (`reports/replication_criteria.md`): OLMo's passive *by* effect misses the declared margin (T1 fails).**
   - Pythia passes T1, T2, T3 and S1. Gemma passes T1, T2 and S1 (trivially: abstract at 1 and 3, no surface site, so no observed abstract → surface transition) and fails T3.
   - OLMo (all 16 finals within 0.03 of the sweep) passes T2, T3 and S1 and fails T1. D(*by*) is +0.43 [0.37, 0.49] at site 9 (NO RISE against δ = 0.53; 0.90 of the natural gap) and +0.14 at 11; it FALLs at 12 and 13. *by* is RISE only at sites 0 and 4–6.
   - From site 7 on, the OLMo passive outcome is surface: the v patch's O push dominates early (abstract at 0 and 4, surface from relative depth 0.44, vs 0.58 in Pythia).
   - Declared fallback fires: the transfer claim is "Pythia and Gemma, with a partial third-family check" (OLMo: OR, TC and both wh transfer; passive *by* is positive with CIs above 0 at 0–11 but clears δ only at 0 and 4–6). Write "misses the declared margin", not "does not transfer".
   - Embedded wh is licensing-like in range at a declared site only in OLMo (site 9, matrix too). So the embedded-wh failure is not consistent across models.
   - The OLMo split edit was declared (`reports/split_edit_replication_rules.md`) and run: Settled 10.
10. **Split edit and u_md on Gemma and OLMo (`reports/replication_criteria.md`, "Split edit and u_md"; reviewed by Codex 2026-11-17).** All checks pass in both (stored IIA and passive D reproduce exactly; write = read bitwise).
   - **Gemma: R-A1 held** at 5 and 8 (off-axis share of v's O 0.72–0.84; on-axis O NO RISE through 11), as in Pythia. **R-A2 not held:** the on-axis share of *by* is 0.16–0.71 (0.33 at 5, 0.61 at 8), so w carries much of *by* too (per-basis |cos(v, u)| ≈ 0.4–0.7). u_md *by* is +0.36 to +0.50 at sites 1–10 (0.65–0.96 of v's), unresolved against δ at 5 and 8. T1-on and T1-umd fail; S1-umd passes thinly (abstract only at 0; surface 6–10, 13); S1-on fails. Gemma staging stays weak with every instrument. **Illusion pattern, Gemma (from `gemma/split_edit/summary_push.csv`, Nov 18): only half holds.** T and I separate along ŵ in actives even less than in Pythia (|standardized difference| ≤ 0.11 at every site, against ≤ 0.22). But the off-axis push is not consistently large: the median is 1.7–2.0 natural sd at sites 0–4 and 1.3–2.4 at 12–15, and only 0.6–0.8 at 8–11, where almost no swaps exceed 2 sd. Gemma's ŵ sd is also heavy-tailed, so the sd yardstick is weaker. And the off-axis piece carries much of *by* too (R-A2 not held). Paper wording for Gemma: the O rise rides on a direction the classes do not separate on in actives. Do not call it an off-distribution push. (OLMo: separation ≤ 0.20, push 1.2–6.6 sd, so the push statistics match Pythia's, but R-A1 fails there.)
   - **OLMo: T1 fails with every instrument** (u_md D(by) +0.41 / +0.43 at 9 / 11, CIs above 0, under δ). **R-A1 not held:** writes along u raise O too (u_md O RISE from 7, on-axis from 8). The off-axis share of v's O is 0.48–0.60 at 7–11 and rises to 0.84–0.90 at 14–15. So OLMo's object rise is not mainly off-axis; both pieces raise it, with depth-dependent shares.
   - **OLMo's deep *by* FALL (12–15) is mostly non-additive:** on alone +0.07 to +0.17, off alone −0.32 to −0.79, full − on − off −0.59 to −0.91. Do not attribute the FALL to the off-axis piece. u_md keeps *by* at +0.41 / +0.43 at 9 / 11.
   - FI: OLMo falls from 0.85–1.5 at 0–3 to 0.33 at 11 (decision "low"), the same shape as Pythia. Gemma is non-monotone (0.17 at 5, 0.70–0.87 at 6–9, ≈ 0 at 14–15). FI is approximate (> 1 at some early sites) and compares different verb sets; "one direction every construction reads" rests on transfer.
   - Exploratory (margin scaled to each condition's own active effect; `reports/selfscaled_passive_classes.md`): along u, OLMo goes abstract-leaning (1–4) → mixed → surface (12–15). It loses the abstract reading earlier than Pythia, but its first surface site (0.75 of depth) is not shallower than Pythia's (0.71). u_md *by* at OLMo 11 clears the scaled margin by 0.0004 nats, so it is not a RISE to report; the Gemma on-axis margins are tiny (δ_c ≈ 0.02). Descriptive only.
   - Push table: `a` and the w-separations are now averaged after aligning each basis to a ≥ 0 (v's sign is arbitrary); the earlier signed means cancelled at sites with mixed signs (e.g. Pythia 12: mean a 0.27 signed, 0.81 aligned). Pythia's "mean |separation along ŵ| ≤ 0.22" in Settled 2 is unaffected.
   - Reading for the paper: the "off-axis object push" story fits Pythia and Gemma; in OLMo, writes along the class mean difference also raise O from site 7. Say so as a model difference, with the u-direction caveat (a u_md write is not shown to change only the class variable).
11. **u_md transfer to OR, TC and filler-gap (`reports/umd_transfer_rules.md`; readings in `replication_criteria.md`).** All checks pass (natural passes identical to v's; paired bootstrap with v's draws reproducing the stored v to 1e-15).
   - **T2-umd passes in all three models**, under v's margin and the own-effect margin (hi95). u_md / v for OR and TC D(O) is 0.40–0.84 (Pythia), 0.30–1.10 (Gemma), 0.64–0.82 (OLMo), about u_md's own active ratio. The training-free class direction carries the OR / TC transfer: this is the Kumon & Yanaka defense.
   - **T3-umd:** Pythia fails (v passed: matrix at 8), Gemma passes (embedded 12 and 15, matrix 15; v failed), OLMo passes (embedded 9, as v). The flips come from push size against a two-sided rule (a FALL floor on the unfilled "." and a not-above-T ceiling on the filled O): u_md misses the floor in Pythia and stays under the ceiling in Gemma. Report per instrument with that explanation; do not read the flips as evidence about licensing.
   - In-range: u_md's effects are mostly 0.1–0.7 of the natural gap, so the in-range profile is mostly "does not reproduce" (signs match).
12. **Passive *by* against its own natural gap (exploratory; `reports/gapscaled_by_classes.md`, `replication_criteria.md`).** The declared δ for *by* is set by v's active O effect; in OLMo it exceeds the natural *by* gap (δ 0.53 / 0.58 vs gap 0.48 [0.18, 0.77]). With δ = 0.2 × the natural *by* gap (the convention the rules use for L readouts), OLMo's T1 passes at the point gap and fails at its upper bound (11 unresolved); T1-on and T1-umd pass in Gemma and OLMo. D(*by*) / gap at the rule sites: Pythia 1.01 / 1.13, Gemma 1.36 / 1.53, OLMo 0.90 / 0.30. Along u_md, OLMo keeps 0.86 / 0.91 at 9 / 11 with no deep FALL. Use D / gap as the cross-model quantity in the paper; the declared T1 result stands.
13. **Cross-frame LEACE (Pythia; `reports/pythia/leace_crossframe.md`, rules `pythia/leace_crossframe_rules.md`, amended before the run; reviewed by Codex 2026-11-18): erasers fit only on actives remove most of what passive-fit erasers remove.** Erasers are fit on active verb states ("The AGENT VERB-ed", same verbs and contexts) and applied to the necessity run's passive items, re-centered on the passive mean (primary). All checks pass within tolerance: both references reproduce (max |Δ| ≤ 1e-16), identity and offline checks are 0, and scrub replay residuals are ≤ 1.1e-7.
   - **Verb-held-out (governs the claim).** Active-fit erasers at scrub 0–12 leave 0.66 [0.47, 0.76] of the passive *by* gap, removing 0.34 [0.24, 0.53] of it (0.26 [0.21, 0.31] nats). That is ρ = 0.82 [0.74, 0.91] of what the passive-fit cross-fitted eraser removes, with matched centering (P-cf leaves 0.59; 0.81 against the stored reference). All 20 label permutations leave ≥ 0.92. Declared reading: majority transfer, with the translation flag (below).
   - **In-sample.** Active-fit erasers leave 0.16 [−0.27, 0.37] (P-in 0.04), removing 0.84 [0.63, 1.27] of the gap (0.65 [0.57, 0.73] nats); ρ 0.87 [0.84, 0.90]. The additional removal from fitting on the evaluated verbs also transfers across frames: 0.50 of the gap with active fits, 0.55 with passive fits (point estimates). This is not an additive "verb-specific component": in-sample and cross-fitted fits differ in training size, covariance and the whole scrub sequence.
   - **Curves.** Single-site ρ is 0.90–1.12 at sites 0–6 and 0.83–0.99 at 7–12 in both pairings. It falls from 15 and is 0.39–0.52 at 18–23, next to the readout. Scrub-prefix ρ (verb-held-out) is 0.80–1.13 at k = 0–12 and 0.78–0.93 beyond.
   - **Geometry (descriptive).** The active and passive edit vectors diverge with depth (|cos| 0.99 at site 1, 0.85 at 4, 0.49 at 8, 0.44 at 12), as in Settled 5, yet the active-fit erasers remove nearly as much. LEACE is oblique (read vector and edit size matter too), so the edit cosine does not measure how much information is common. The active-fit edits are closer to u_md than the passive-fit ones (|cos| 0.51 / 0.61 / 0.70 / 0.74 at 4 / 8 / 12 / 17, vs 0.49 / 0.36 / 0.37 / 0.29). This does not identify the causal direction.
   - **Active side.** The erasers are not inert in their own frame. Scrub 0–12 leaves 0.39 [0.34, 0.44] of the active T − I log P(object start) gap in-sample and 0.80 [0.76, 0.83] verb-held-out. These are different readouts, so they are not compared with the passive shares.
   - **Translation flag.** As-fit erasers (active mean) agree: 0.63 / 0.14 remaining. The constant mean shift alone removes 0.03 [0.01, 0.05] (0.02 in-sample). The declared flag fires because its CI excludes 0, so by the amendment the paper does not call this "shared". The primary erasers preserve the local passive mean. The shift-only effect is small, but it is not an additive bound on nonlinear side effects.
   - **Centering mismatch: resolved (`reports/pythia/leace_pcf_recentered.md`).** The stored P-cf erasers kept their training folds' passive mean, while A-cf (c) is centered on the evaluated fold. The stored P-cf erasers were rebuilt (reproducing the stored run to 1.8e-15) and re-centered the same way. P-cf then leaves 0.59 [0.36, 0.71] (stored 0.59 [0.35, 0.70]), and matched ρ is 0.82 [0.74, 0.91] (stored-reference 0.81). The reading is unchanged (majority transfer). Quote the matched ρ.
   - **Paper wording (intervention first; no "shared", no "the feature is necessary").**
     - "Erasers fit only on active clauses of other verbs, applied at the verb in passives, reduce the passive *by* gap by 0.34 [0.24, 0.53], 0.82 [0.74, 0.91] of the reduction from erasers fit on passives (0.84 [0.63, 1.27] with the evaluated verbs in the fit). This shows that the passive *by* preference is sensitive to an intervention learned from actives; it does not establish a single shared direction."
     - The direction-specific evidence stays with the v / u_md ablation (Settled 3).
     - For the cross-verb split of Settled 1, write "verb-general" (cross-fitted) and "additional in-sample removal" rather than "shared + lexical".

14. **Training-pool ceiling (Nov 23; `reports/pythia/train_pool_ceiling.md`).** 197 frequent intransitive candidates: 73 fail review, 68 prepositional, 2 unpairable, 54 plain (all pass the behavior filter). v's 29 pairs used 29 of the 30 usable in October; the Nov 18–19 spot-checks raise the supply with frequent partners to about 40 pairs. So "29 was the available ceiling" holds for the pool v was trained from, not for today's.
15. **Reverse frequency transfer (Nov 23; `reports/pythia/das_reverse_freq.md`, rules amended after Codex).** A rank-1 DAS direction trained only on 42 rare plain pairs transfers to frequent verbs: Head D_bad(*by*) 1.00 [0.89, 1.14] of v's at site 8, 0.94 [0.85, 1.04] at 12 (both sides rare: 1.01 / 0.92); D_rare − 0.5·D_v CI above 0; seeds agree. Active IIA on frequent verbs is lower (0.63 vs 0.78 at 8); |cos| with v 0.80–0.83. Reading: the class feature the passive reads is shared across frequency; the rare-verb deficit is not a missing feature. (Unfiltered condition invalid at site 12 by a 1e-5 recompute miss.)
16. **Expanded eval set, Pythia (Nov 23; `reports/expanded_eval_rules.md` Result, `reports/pythia/expanded_eval_*.md`).** Transfer generalizes to 74 new plain verbs (v new / original 1.01 at 8 and 12; u_md 0.96 / 0.98); OR / TC RISE for new verbs. Cross-fitted LEACE with erasers fit on 268 pairs removes more: on the same 126 original items 0.42 vs 0.59 remaining (paired −0.16 [−0.26, −0.09]); pooled 0.47 [0.33, 0.56], removed 0.53. By the declared rule the pooled number replaces 0.41 as the headline. The 126-pair learning curve was flat by point estimate; the verb-general share was underestimated by the small verb set.
17. **LEACE replication (Nov 23; `reports/leace_replication.md`).** Gemma and OLMo: N1–N3 hold (cross-fitted removal 0.76 / 0.44; in-sample removes 0.49 more, paired CI below 0; true in-sample CI below every permutation). Cross-frame: majority transfer in all three (ρ cf 0.82 / 0.94). The translation flag fires in both, in the opposite direction from Pythia (the shift alone slightly widens the gap).
18. **OLMo nonce and controls (Nov 23; `reports/olmo/nonce_controls.md`).** Nonce onset at 0.31 of depth (Pythia 0.12–0.25, Gemma 0.23), NP-driven (C9) at 5 and 9; the declared criterion fails because site 8 dips to +0.004 z. Controls: trained IIA above random at every site; probe ≥ 0.99 from site 3.
19. **DAS stability, all rule sites and three models (Nov 23; `reports/das_stability_ext_rules.md` Result).** Retraining v with 61 more plain verbs (P) rotates it below the seed range at every site in every model (cos ≈ 0.90–0.92; Gemma 5: 0.72), reproducibly across P's seeds and not because of longer training (update-matched L stays near the seed range). P's passive *by* transfer is never smaller (1.0–1.2× at the rule sites). **OLMo site 11: P transfers 1.82× v's *by*** (0.27 vs 0.15 nats), where OLMo's T1 failed; report next to T1. Pythia's declared reading is invalid on one seed-band recompute (max 0.66 > 0.5); numbers exploratory.
20. **Expanded eval set, Gemma and OLMo (Nov 23; `reports/expanded_eval_rules.md` Result, `reports/{gemma,olmo}/expanded_eval_*.md`).** Transfer generalizes to the 74 new plain verbs in 7 of 8 declared readings (new / original 0.74–0.95; the exception is v at OLMo site 11, where v's *by* is already small, 0.15: inconclusive); OR / TC D(O) RISE for new verbs. Cross-fitted LEACE with 268 pairs removes more on the same original items, as in Pythia (paired difference Gemma −0.27 [−0.64, −0.15], OLMo −0.16 [−0.30, −0.08]). Headline removal: Gemma 0.83 [0.61, 1.22] (was 0.76; pooled CI compatible with the stored point), OLMo 0.57 [0.45, 0.74] (was 0.44; lower share remaining). The three-model "one number" is now 0.53 / 0.83 / 0.57.

## Work, ordered by risk

Run whatever could break the claim first, then polish.

1. **Split-edit test, then the u_md staging curve.** For each v patch, δ = v·h_s − v·h_b. Run the full edit (δv), the on-axis piece (δa·u_md) and the off-axis piece (δb·ŵ), and report D(by), D(O) and Δ log P(O) for each. Compare the size of the ŵ push with the natural spread of ŵ coordinates. Pythia first, building on `scripts/run_das_direction_check.py`; then Gemma and OLMo. The prediction to declare is Settled 2.
   - *Nov 16:* Pythia done (Settled 2). Next: Gemma and OLMo (OLMo after its finals).
2. **OLMo final runs.** Run final DAS, the passive test at all 16 sites, nonce at sites 0–8 and transfer at the rule-picked sites. This fills the empty column and settles relative depth vs absolute layer. Queue early, because the bottleneck is queue time, not code.
   - *Nov 16:* rules `reports/olmo/finals_rules.md`. Primary site 13; declared secondaries 9 (IIA ≥ 0.8), 11 (≥ 0.9) and 12 (within 0.05 of max). Finals at all sites 0–15, then the transfer chain and the passive depth curve, are queued. Nonce and controls get their own rules after the finals.
   - *Nov 16, done:* finals, transfer chain and depth curve (Settled 9). Left: nonce at sites 0–8, controls, LEACE. Split edit done Nov 16 (Settled 10).
3. **Frame invariance of the class axis** (Pythia done, Settled 5; rerun per model with the split edit). Per site: cos(u_md^active, u_md^passive); a one-line check that the LEACE edit vector equals the passive mean difference; transfer with u_md^passive vs u_md^active. A high cosine supports one feature. A low cosine combined with good transfer means the frames share a readable direction but not the class mean, which needs to be said in the paper.
4. **Rank-k erasure** (closes the main objection to "lexical"). Rank-1 LEACE removes one direction per site by construction. A reviewer can read the ~0.55 residue as shared class information in a few more directions rather than verb-specific information. The held-out probe still reads class at 0.67–0.78 after cross-fitted erasure, so the residue is real. Run cross-fitted rank-k erasure (iterated LEACE or INLP, k = 1…8) at site 8 and for scrub 0–12, with a random-direction reference.
5. **Replication criteria, declared once per claim.** State what counts as replicated in Gemma and OLMo before their remaining runs: sign, CI excluding 0, the site the declared rule picks, and whether depth is mapped by relative depth or absolute layer. The three-model table then reads as pass/fail against stated criteria.
   - *Nov 16, drafted:* criteria for transfer (T1 passive *by*, T2 OR/TC, T3 filler-gap) and staging (S1), applied to all three models, in `reports/olmo/finals_rules.md` §3. Still to declare: necessity (for the Gemma/OLMo LEACE runs) and nonce.
   - *Nov 16, applied:* `reports/replication_criteria.md` (Settled 9).
6. **Training dynamics on 16 checkpoints per model.**
   - Write the prediction first (candidates below).
   - Pythia: the same 16 checkpoints as R6-A6 (step 64 to 143000, already used for behavior). Run u_md, the held-out probe and the passive *by* readout at the verb at all 16, plus DAS at all 16. DAS is about 1 min per site for 5 folds, so 16 checkpoints × a few sites is cheap.
   - Code: add `--revision` to `run_das_round2.Runner` (OLMo `rules.md` notes it is missing).
   - OLMo: 16 checkpoints matched to Pythia's by tokens seen, drawn from the 37 early-training checkpoints and the 195 stage-1 branches.
   - Check the TSUBAME HDD cache before downloading anything.
7. **LEACE on Gemma and OLMo.** In-sample, cross-fitted and the permutation control, reusing the Pythia pipeline. "Shared + lexical" is too central to rest on one model.
8. **Side splits.** Pythia done (Settled 8). Add to the Gemma / OLMo LEACE runs.
9. **Robustness.** DAS seed variance and random-subspace baselines for every reported number, plus the single-token × frequency confound (Settled 6). Pythia-6.9B is work item 12.
10. **Writing, starting now.** Write the outline, figure list and methods now; results go in as each piece lands.
11. **Embedded wh in Pythia.** Pythia embedded wh is mixed at site 8 and surface at site 6 (R5-C8); Gemma's in-range result is mixed or unresolved. Hard stop Nov 18, then it goes to limitations. OLMo's is licensing-like at site 9 (Settled 9), so the paper can report it as model-dependent rather than as a uniform failure.
12. **Pythia-6.9B mechanism check (moved before the freeze, Nov 18; freeze now Dec 11).** Behavior is already done at 6.9B (and across 160M–12B): the rare-verb deficit and its intransitive source hold at every size. No mechanism result exists at 6.9B; R5-D10 only timed DAS with random bases. In order, stopping when the time runs out:
   - **Behavior at the paper's items.** Natural passive *by* gap and OR / TC gaps on the same items as the 1.4B runs (forward passes only).
   - **u_md transfer** (training-free): passive *by*, OR and TC at the matched-depth sites (R5-D10: 5, 8, 11, 13, 16, 19, 21, 23 of 32), reported as D / natural gap.
   - **Cross-fitted LEACE** scrub over the matched-depth range plus the permutation control. This tests whether "verb-general + additional in-sample removal" holds at scale.
   - **DAS** sweep at the matched-depth sites, the final at the rule-picked site, then the transfer chain under T1–T3 / S1 with relative depth.
   - **Cost:** R5-D10 measured about 3 GPU-hours for the DAS sweep plus final on one H100 in fp32 (DAS peak 27.5 GiB, weights 25.5 GiB). Check whether `gpu_h` fits this; otherwise use `gpu_1`. Keep fp32, as the 1.4B runs do. Weights for 2.8B, 6.9B and 12B are already in the HDD cache.
   - **Rules:** declare before running, reusing the replication criteria with relative depth. A null at 6.9B is reported, not dropped.
13. **More verb pairs (decided Nov 18).** 126 pairs is enough for the transfer and direction claims. It is thin for three things:
   - the verb-general share (learning-curve CIs at 75 / 100 pairs are [0.03, 0.72]);
   - the 27 single-token pairs;
   - per-band splits (head 26 pairs, about 5 per fold).

   **Supply.** The binding class is intransitives, and their scarcity is at the frequent end (Zipf), not the rare end.
   - The earlier expansion (`data/verb_expansion/`, unused, never spot-checked) found only 7 clean head intransitives, all with prepositional complements. Most clean intransitives sit at participle Zipf 3.2–3.5, between Tail and Head ("near-head").
   - Included after Codex's adversarial review: 99 intransitives (head 7, near-head 53, tail 9, xtail 30) and 220 transitives. Of these, 54 near-head pairs were reserved for DAS training and 19 for evaluation.
   - FreqBLiMP's strict-intransitive inventory (`freq-blimp/vocabulary_overlay.csv`, 253 participles) adds about 136 not yet used, mostly rare: 49 xtail, 25 tail, 21 between the bands and 31 below xtail. Only about 10 reach Zipf ≥ 3.2, and several of those are unusable (adjectival or with ordinary objects: *deceased*, *spoken*, *sat*).
   - Transitives are plentiful.

   **Plan.**
   - Pool the vetted intransitives from both sources. Pair each with an unused transitive by summed-lemma Zipf (within 0.25) and token count, as `das_train_pairs.csv` did. Aim for about 250 pairs in total (roughly double), with as many single-token pairs as supply allows.
   - Evaluate the new pairs in the existing curated contexts, in the same cross design (each verb in the other pairs' contexts). So no new context curation is needed. The new pairs' own contexts are optional.
   - Use them only for the training-free analyses: behavior, u_md projection and transfer, and LEACE (cross-fitted, learning curve, cross-frame). DAS is unchanged: v is trained on the 29 round-2 pairs (21 near-head expansion + 8 original head) and evaluated on the 126. The 24 expanded pairs sharing a verb with those 29 are flagged `das_train_overlap`.
   - **Spot-check (Tyrone):** a manual pass over `data/verb_expansion/intrans_spotcheck.csv` before any run. It lists 239 candidates (99 from the reviewed expansion, 140 from FreqBLiMP), with Zipf, tokens, Pile counts where available and automatic flags. This is the one step that cannot be automated. Pair homograph-flagged verbs by participle Zipf, not summed-lemma Zipf. *Nov 18:* done by Claude and Codex (gpt-6.2-sol) independently. They agreed on 199 of 239 verbs. Most disagreements were speech and manner verbs, dropped for consistency with the earlier speech-object and clause exclusions. Kept: 163, then 160 after the declared weather-verb exclusion (*gusted*, *sleeted*, *squalled*; no other selectional restrictions, since the cross design shares semantic mismatch across classes). Single-token: 25. About 286 pairs in all. A `selection` flag (inanimate-preferred 23, non-human animal 1) gives a with/without robustness line. The decisions and both reviewers' reasons are in the file. *Eval set (Nov 19):* every analysis uses the expanded pairs minus the 24 that share a verb with v's 29 training pairs (`das_train_overlap`), so one set serves all. After reconciling with the round-5 list (R5-D11: 42 unjudged intransitives spot-checked the same way, 20 kept, including 5 head intransitives; addendum in `pairing_rules.md`), that is **142 new pairs, 268 with the originals**: 24 single-token, 11 head intransitives. *Pairing (Nov 18):* 147 new pairs (`data/verb_expansion/expanded_pairs.csv`; rules and result in `pairing_rules.md`), 273 in all. 25 are single-token. 13 rare intransitives are unpaired. 100 of the transitives come from FreqBLiMP's passivizable inventory, screened by Claude, because the reviewed transitives run out at the rare end.

   **Frequency bands.** For the mechanism paper, drop the three-band split. Treat frequency as a continuous covariate (log participle and summed-lemma Zipf), with matched T / I pairs. Reasons:
   - the mechanism claims are verb-general;
   - the near-head verbs fall outside the bands anyway;
   - the band cells are too small to split.

   The rare-verb claim becomes a regression or binned curve over Zipf. Keep the band labels only to cite the FreqBLiMP behavior results (P1) and as a robustness row.

   **Rules:** declare before running, including the pairing calipers, exclusions and which analyses use the expanded set. The 126-pair results stay the primary numbers unless the expanded-set analyses are declared as replacements before running.

14. **DAS stability on the expanded pairs (added Nov 18; rules `reports/pythia/das_expanded_stability_rules.md`).** v is trained on only 29 pairs. Does it change with 5× more training verbs? At the rule sites 8 and 12, retrain the 15 cross-fitted bases with the 123 non-overlapping expanded pairs added (E), alongside a reproduction (R) and a seed baseline (S). Compare each with v on three things: the cosine; held-out active IIA; and passive *by* transfer D_bad(*by*) on the stored plan, paired bootstrap. About 2–3 GPU-hours, mostly the passive patches. *Result (Nov 18):* the declared reading is invalid, because the held-out recompute tolerance I set (max |ΔM| < 1e-2) was tighter than the stored pipeline's own bf16 precision (max 0.29–0.36, median ≈ 0); every other check passes. Exploratory reading, partly stable: plain-only P rotates v by about 23° (|cos| 0.91–0.92, against 0.99 for a seed change) with equal held-out IIA and 1.06–1.10× the passive *by* transfer. v is not an artifact of its 29 verbs; a plain retrain gives a nearby, slightly stronger direction. **Decision (Nov 19, pending sensei's agreement at the Nov 19 meeting): keep the stored v.** Reasons: the retrain is close and at least as effective; every v result in all three models and the head / neuron analyses (item 16, done with the stored site-8 v) stay valid; a retrain and chain rerun would cost about a week before the freeze. Report the check as robustness (appendix row plus one sentence). If sensei prefers a retrain: plain pairs only (stored 29 + new plain + plain originals, cross-fitted against the passive test), Pythia first. Declared before running. The stored v stays the instrument either way. A different direction is reported as a finding about v's verb dependence, not swapped in. A full DAS retrain on the expanded set is not planned, because staging and transfer lean on u_md and the on-axis piece, which are recomputed on all 273 pairs anyway.

15. **Reverse frequency transfer (added Nov 19; scoped).** v is trained on frequent verbs (head and near-head) and transfers to rare verbs. The reverse test: does a direction trained only on *rare* verbs transfer to frequent ones? If it does, the class feature is shared across frequency, so the rare-verb deficit is not a missing feature (complements R6-C4 and the u_md / LEACE results).
   - **Rare by Pythia's training counts, not Zipf (Nov 19).** Use the Pile count of the participle (`c_total` as in R6-A1: infini-gram `v4_piletrain_llama`, all case and line-start variants), since that is what Pythia saw.
     - Coverage: counts exist for 95 of the 147 new intransitives, 35 of their transitives, all 281 D11 pairs, and 37 of the 58 verbs in v's 29 pairs. Fetch the rest with `fetch_pile_counts_r6.py`, overall-count slots only: a few thousand queries.
     - The rare cutoff, a count quantile, is declared before training. The "frequent" test verbs are v's 29 pairs, whose counts are also checked.
   - **Train:** rank-1 DAS at the rule sites 8 and 12, same recipe as v, on rare plain pairs only (by that cutoff), with matched transitives. No frequent verb in training, and none of v's 29.
   - **Test on frequent verbs:**
     - held-out active IIA on v's 29 frequent pairs (none in training, so one basis per seed, no folds needed);
     - passive *by* transfer D_bad(*by*) on the stored passive plan, by band (head is the frequent test);
     - |cos| with v.
   - **Compare** with v on the same frequent items. Also give the forward direction: v's transfer to the tail / xtail passive bands, already in R2-ptest.
   - **Scope:** about 1 GPU-hour, reusing `run_das_expanded_stability.py` / `run_passive_test.py`. Rules before running, including what counts as transfer (for example, D_bad(*by*) on head at least half of v's).
16. **Head and neuron level: bring back the downstream-readers results (added Nov 19; writing, no new runs required).** All Pythia-1.4B, using the stored site-8 v (one more reason to keep it; item 14).
   - **Heads route voice into the *by* MLPs** (R3-B7 path patching). Switching the auxiliary from "was" to "has", joint attention heads carry 67–115% of the switch in the *by* MLPs 11–17. Earlier MLPs carry the late object MLPs 15 / 18 / 21 / 22.
     - The top heads are not previous-token heads, and keep their role under adverb and *got* frames (R4-A5).
     - The switch is a voice switch: "has been V" vs "has V" reproduces it in 9 of 9 MLPs (R4-D10).
   - **Conjunction neurons** ("transitive AND passive", R4-D11): in MLPs 11 and 14, a passive-conjunction set selected on half the pairs carries 0.65 [0.60, 0.70] of the *by* switch on the other half, with sign agreement 0.99. Their response holds from Head to XTail (XTail / Head 0.84).
   - **Serial chain** (R5-B6): heads → mid MLPs → late object MLPs. Freezing leaves 0.14–0.26 of the late switch; the chain holds for 2 of 4 targets (mixed).
   - **Shared readers** (R5-C9): the same switch neurons read the verb class in wh-gap frames, but only with a filler present (ρ 1.4–1.9).
   - **In the wild** (R5-B7, Pile text): the top neurons fire on passives with *by* next (MLP14.n2114 / n5407) or on transitive verbs with a missing object (MLP11.n3462).
   - **Limits:** not necessary (R5-B5: the top-50 neurons remove 0.09 of *by*); restoring them at the translation stage beats random (+0.169 [0.107, 0.234]), but the declared claim fails on the Head control and deficit closure (R4-E13T).
   - **Paper role:** one section, "how the class is read", between transfer and necessity, with one figure (heads → MLP switch shares, plus the conjunction-neuron share). Wording: "components", not "circuit"; Pythia-only, stated.
   - **Also bring back (reconciled with `results_index.md`, Nov 19).** These are earlier site-8 results the plan had not cited.
     - **Why passives get no object rise** (R2-mech, main): a signed decomposition at site 8. Object survival is 0.46 [0.40, 0.50]; the object MLPs 15 / 18 / 21 / 22 vanish in passives, while MLPs 11, 13, 14, 16 and 17 flip sign on *by*. This is the entry point for the head / neuron story.
     - **Dose response** (R2-dose): *by* is linear along v at site 8 (slope 0.92 [0.81, 1.04]); the object response is intermediate (0.43). The object acceleration at large doses is mostly the final-LN scale (R3-A3).
     - **Persistence** (R3-A2): the passive site-8 separation stays flat to site 23 (retention 0.98–1.16).
     - **Frame generality** (R3-B6): the site-8 switch carries over from "was" to "got" passives.
   - **Optional, if time: find conjunction neurons in a second model** (OLMo-2-1B first, since its passive *by* behaves differently; or Gemma). Reuse R4-D11's procedure: run_conjunction.py with half-A selection and half-B tests, at the MLPs that the split edit and path patching would flag. A second model with the same "transitive AND passive" neurons would make item 16 more than a Pythia case study. It needs a port of the MLP hooks and a model-specific site, about 1 GPU-hour plus porting time, so it comes after the freeze-critical work. Also cheap: check that the Pythia conjunction neurons respond to the expanded eval verbs (forward passes only).

### Dynamics prediction candidates (pick before running)

- **Feature before use, rare = frequent.** The class feature at the verb (u_md separation, held-out probe) appears at about the same checkpoint for rare and frequent verbs, while the rare-verb behavioral deficit (R6-A6: from step 256, peaking around steps 1000–3000) persists. Then the deficit is not a late-feature problem, which links "rare verbs get the feature right" to emergence. This also differs from Desai & Nair, who track sharing, not rare vs frequent onset of a feature vs its use.
- **Shared before lexical.** The cross-fitted (shared) share removed appears before the in-sample minus cross-fitted (lexical) share.
- **Abstract before surface.** The u_md *by* effect appears before the late O effect, following COMPASS.

## Possible framing: class information suppresses *by*

Confirmed in Pythia (Settled 8): the gap closes mostly from the bad side under mean ablation and both LEACE fittings. Class information mainly suppresses *by* after intransitives, and the default after a passive participle is to expect *by*. That ties into the output-side result: rare intransitives being under-rejected is weaker suppression.

## Paper shape

Eight pages for six claims plus dynamics is tight. The main-text spine is one figure per claim:

1. Transfer: 5 constructions × 3 models, v and u_md side by side.
2. Shallow: nonce onset by depth.
3. Shared + lexical: LEACE in-sample vs cross-fitted (with rank-k if it lands).
4. Rare verbs: feature right (classify-from-z), output side wrong (restoration).
5. Emergence order.

Staging now has a clean Pythia result (on-axis piece: abstract 6–12, surface 17), so it can carry Figure 1 if Gemma's split edit agrees. The downstream readers (heads and neurons, item 16) get a short section with one figure; details go in the appendix. If more has to shrink, shrink the edit section (sensei asked for dynamics).

## Related work that shapes the framing

- [Kumon & Yanaka](https://arxiv.org/abs/2604.22166) (ACL 2026) find that activation-patching mechanisms hold out of distribution while DAS overfits narrow linguistic data. Two defenses. Cross-construction transfer is itself an out-of-distribution test (trained on actives, tested on four other constructions). And u_md is training-free, so report its transfer next to v's.
- [Desai & Nair](https://preview.aclanthology.org/ingest-acl/2026.findings-acl.1737/) (Findings ACL 2026) run DAS over BabyLM checkpoints and call filler-gap sharing "shared, yet item-sensitive". That is close to "shared + lexical", so cite it. This paper quantifies the split and places the item-sensitive part (rare verbs, output side).
- [COMPASS](https://arxiv.org/html/2511.20910v2) reports semantic-role circuit structure stabilizing before it becomes functionally needed. It's the nearest training-dynamics comparison.
- [Boguraev, Potts & Mahowald](https://arxiv.org/pdf/2505.16002) (EMNLP 2025) used transitive/intransitive prediction as a control task. This paper makes their control the object of study.

## Cheap analyses worth adding

- **Alternating verbs as a graded test.** They were excluded by design. Projecting them onto u_md and correlating with corpus transitivity bias (Pile counts from R6-A1) would show the feature is graded. No training needed.
- **Gemma Scope 2 lookup.** Take the cosine of u_md with SAE decoder latents at the same sites. This gives unsupervised support without the attribution-graph work saved for the thesis.
- **More verbs for the training-free analyses:** promoted to work item 13 (Nov 18).

## Timeline and decision rules

Freeze experiments on **Dec 11** (moved from Dec 4 on Nov 18; writing runs alongside), then write the full draft quickly; the date it goes to sensei is to be set (about a week after the freeze). Coding is fast. Queue time on TSUBAME, reading results and sensei's turnaround are not.

| Week | Experiments | Writing |
| --- | --- | --- |
| Nov 16–22 | Split-edit + frame invariance (Pythia); side splits; queue OLMo finals; declare replication criteria and the dynamics prediction; embedded wh stop (Nov 18) | Outline, figure list, methods |
| Nov 23–29 | OLMo transfer / nonce / split-edit; Gemma split-edit; rank-k erasure; LEACE on Gemma, OLMo; checkpoint runs; verb expansion: spot-check, pairing, rules (item 13) | Results sections as they land |
| Nov 30–Dec 4 | Expanded-verb behavior / u_md / LEACE runs (item 13); reverse frequency transfer (item 15); dynamics analysis | Intro, related work; heads / neurons section (item 16) |
| Dec 5–11 | Seeds and random-subspace baselines; Pythia-6.9B mechanism check (item 12; rules and behavior can go earlier); **freeze Dec 11** | Results and discussion |
| Dec 12–18 | — | Full draft to sensei (date to set) |
| Dec 19–Jan 4 | — | Revisions, appendix, checklist, submit **Jan 4 AoE** |

The week rows are a proposal; the original timeline chart did not survive the export. Fixed dates: Nov 18, freeze Dec 11, Jan 4.

**How declared rules are used**

The declared criteria are a confirmatory floor, not the whole report. Each claim is shown as curves over all sites and instruments (v, on-axis, u_md), with CIs and the rule sites marked. Each model's mechanism is described as it is, and model differences are a finding, not a failure. Anything chosen after seeing the data is labelled exploratory. Nothing is dropped for failing a rule. What would count against the claims is counter-evidence in the curves (for example *by* falling where the class says it should rise), not a single class flip at one threshold. Keep three things apart in the text: a positive effect (CI above 0), a declared meaningful effect (RISE against δ), and the diagnostic outcome labels ("surface" includes O RISE with *by* unresolved). An exploratory margin never replaces the declared one.

In the paper itself the rules are method, not story (decided 2026-11-17). The main text states each claim with its effect sizes and CIs, in units that compare across models (for *by*, D / natural gap). The declared sites, margins and pass/fail table go in one methods paragraph and an appendix. A declared failure is still reported where the claim is stated (e.g. "OLMo misses the declared margin"), in one clause, next to what the curves show.

**Decision rules**

- **Off-axis piece carries the passive O rise:** redraw staging with u_md (or the on-axis piece) in every model, and report v as an upper bound.
- **On-axis piece carries it too:** keep staging as drawn with v, and note u_md's later boundary.
- **OLMo transfer fails:** the paper becomes "Pythia and Gemma, with a partial third-family check". *Fired Nov 16 (T1 fails; T2 passes).* Read with the curves: OLMo's *by* effect is positive at every site 0–11 (0.30–1.19 of the natural gap) but clears δ only at 0 and 4–6. The split edit and u_md did not rescue T1 (Settled 10): v's deep *by* FALL is mostly a non-additive on × off interaction, and writes along u raise O from site 7–8.
- **Rank-k residue keeps falling with k:** reword "lexical" as "shared, low-rank plus verb-specific", and report where it plateaus.
- **cos(u_md^active, u_md^passive) is low:** drop "one feature" in favor of "one direction every construction reads", and keep transfer as the evidence.
- **Training dynamics not clean by the freeze:** report emergence order for the measures that are in, on one model if needed.
- **Embedded wh not solved by Nov 18:** it goes in limitations.
- **New idea after the freeze:** it goes in the thesis file.

## Admin checklist

- [x] Merge the result branches into one paper branch (`paper`, Nov 16)
- [ ] OpenReview profiles for both authors. Check the ARR CfP for the ORCID requirement and reviewer registration for all authors.
- [ ] Anonymized code link (repo scrubbed of names, TSUBAME paths and lab references)
- [ ] Cite FreqBLiMP in the third person
- [ ] Limitations section: embedded wh, single-token × frequency confound (Settled 6), English only, models of 1–7B
- [ ] Responsible NLP checklist

## Sources

- [ACL 2027 call for papers (via AI Weekly)](https://aiweekly.co/node/16101)
- [Kumon & Yanaka, Fine-Grained Analysis of Shared Syntactic Mechanisms in Language Models](https://arxiv.org/abs/2604.22166)
- [Desai & Nair, Filling in the Mechanisms](https://preview.aclanthology.org/ingest-acl/2026.findings-acl.1737/)
- [Emergence and Localisation of Semantic Role Circuits in LLMs](https://arxiv.org/html/2511.20910v2)
- [Boguraev, Potts & Mahowald, Causal Interventions Reveal Shared Structure Across English Filler-Gap Constructions](https://arxiv.org/pdf/2505.16002)
- In the repo: `reports/results_index.md` (R-numbers above), `reports/pythia/` (LEACE, direction check), `reports/gemma/`, `reports/olmo/`
