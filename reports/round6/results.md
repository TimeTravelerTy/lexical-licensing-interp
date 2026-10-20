# Round 6: results (paper 1: frequency and entrenchment)

Plan, predictions and decision rules: `plan.md`, each part committed before
its runs (part 0: `01f6f23`; A1–A2: `0b1d505`, code fixes after a second
review `d875bc7`, A2 run clarification in `02461c1`; B3, C4, C5: `02461c1`;
A6–A7: `37e842c`). An outside reviewer (Codex) reviewed every part's draft
and code before its runs, twice per part. Generated reports, one per
experiment:
- 0: `../results_index.md` (paper tags), `human_passives.md`;
- A1: `a1_exposure.md`; A2: `a2_suite_d11.md`;
- B3: `b3_alternations.md`; C4: `c4_classify.md`; C5: `c5_preemption.md`;
- A6: `a6_checkpoints.md`; A7: `a7_fit_controls.md`.

TSUBAME jobs: 9069457 (A2, seven models; 160M stopped at a check), 9069909
(A2, 160M rerun under the declared clarification), 9069908 (B3, C4; C5
stopped at its check), 9070274 (C5 rerun under the declared clarification),
9070046 (A6 checkpoints and A7 released-context scores), 9070047 (A7 fit
ratings). Counts (A1) and training-stream slices (A6) were fetched locally
from the infini-gram API and the Hugging Face CDN.

Disclosures, also in `plan.md`: (1) the C4 classification of round-5 A4's
32 rare verbs was seen in a code smoke test before C4 was declared; (2)
mock analyses used to test the A1, A2 and A7 code mixed real round-5 scores
of the 433 D11 verbs that round 5 had already scored with random numbers
for the rest, and so printed partly real band contrasts before A2's
analysis; they were not used and the mocks were replaced.

## Summary

- **Part 0.** Round 5 was already in the results index; every row now
  carries a paper tag (P1 frequency / P2 mechanism / both) and the parked
  paper-2 items are listed. The FreqBLiMP human study is forced choice, so
  only accuracy by band is reported: annotators also find rare passives
  harder (`passive_1` correct among decided 94.6 → 85.7%, can't-decide 18
  → 38% from Head to XTail), on 15 items per cell (descriptive).
- **A2, the D11 frame contrast across the Pythia suite.** The rare-verb
  deficit replicates on the 281-pair inventory and survives token-count
  standardization.
  - At equal token counts, Tail pairs have a larger was–has contrast L
    than XTail pairs at every size from 160M to 12B (ΔL_tok +0.89 to
    +1.66 nats; 1.4B +0.99 [+0.43, +1.53]), so the deficit is not only a
    tokenization artifact of the Head − XTail comparison.
  - It is intransitive-side at every size: rare intransitive participles
    are rejected less after "was" (1.4B: −ΔΔ_pass(I) +1.40), while rare
    transitives are, if anything, accepted more (ΔΔ_pass(T) −0.41).
  - The held-out (DAS-unexposed) pairs point the same way at every size
    but are too few to resolve (1.4B three-cell +0.53 [−0.66, +1.63]).
- **B3, other alternations.** The symmetric entrenchment prediction holds
  in both frames tested on D11. After "The N V-ed", rarer intransitives
  leave more room for an object (log P(O | I) falls with Zipf, −0.31 per
  unit; transitives flat) and rarer obligatory transitives leave more room
  for a period (log P(. | T) −0.69; intransitives unresolved). Both are
  side-specific; the Head − XTail deficits are on the predicted side
  (I-side for objects, T-side for clause ends). Side specificity is not
  established at every suite size. Count analyses wait for A1.
- **C4, classify-from-z restore.** Rare single-token verbs' own site-8 z
  gives their class (31 of 32 correct; θ fit on Head verbs), and an edit to the Head
  class target driven by that label matches the oracle on the curated
  affected pairs (+0.53 nats) without changing Head pairs, unaffected
  items or Pile-10k loss (NLL at the edited tokens rises by +0.08). This
  concerns single-token participles only.
- **C5, in-context preemption: not supported, reversed.** Repeating a rare
  intransitive's own past form in the context makes "was V-ed" relatively
  more probable (Δ_pass(I) +1.17 against filler), not less; exact-form
  mentions do so more; nominal (-ing) mentions show no resolved effect.
  Context effects, not preemption, dominate.
- **A6, training dynamics.** The deficit appears early (1.4B: Head − XTail
  from step 256, token-standardized from 1000) and is largest, about 3
  nats Head − XTail, around steps 1000–3000: frequent intransitive
  participles are by then much less probable after "was" than after
  "has", while for rare ones Δ_pass(I) is still positive. The gap then
  narrows and over the last ~90% of training shows no trend (1.21–1.63
  nats). For 1.4B intransitives, whether one exposure curve predicts all
  verbs is unresolved (curve collapse neither holds nor fails; at 6.9B,
  with 6 steps, it fails). The stationarity diagnostic (lowercase
  single-token participles) finds no drift in the training stream.
- **A7, fit controls.** The D11 deficit survives matched plausibility (M_was
  XTail − Head −0.72 [−1.32, −0.13] at matched fit), does not come from
  rare nouns in the context (noun effect negligible), and reads a deficit
  under every prefix set and model size.

## Part 0: housekeeping

- `../results_index.md`: a Paper column (P1 / P2 / P1+P2) on all 62 rows of
  rounds 1–5 and a note listing the parked paper-2 items (LEACE necessity,
  gap-neuron mediation, human annotation of B7, 6.9B DAS). Round-6 rows are
  added at the end of the round.
- Human data (`human_passives.md`): the study is forced choice (A / B /
  can't decide, then a naturalness rating of the good sentence only), so
  bad-item acceptance cannot be computed. Per the rule, accuracy by band,
  descriptive (15 pairs × 3 annotators per cell):

| paradigm | band | correct among decided (%) | can't decide (%) | 2-of-3 support | Pythia-1.4B LP, same items (%) |
|---|---|---|---|---:|---|
| `passive_1` | Original | 97.6 | 8.9 | 13 / 15 | — |
| | Head | 94.6 | 17.8 | 13 / 15 | 80.0 |
| | Tail | 88.2 | 24.4 | 10 / 15 | 66.7 |
| | XTail | 85.7 | 37.8 | 7 / 15 | 73.3 |
| `passive_2` | Original | 100.0 | 4.4 | 15 / 15 | — |
| | Head | 97.3 | 17.8 | 12 / 15 | 80.0 |
| | Tail | 83.8 | 17.8 | 11 / 15 | 80.0 |
| | XTail | 91.2 | 24.4 | 10 / 15 | 80.0 |

  The rarity decline in humans shows mostly as more "can't decide"
  responses (XTail `passive_1`: of 17 non-decisions, 14 give "both
  acceptable" and 7 "unknown words" as a reason; several reasons allowed). CIs are in the report; with 15 items per
  cell none of the band differences is resolved.

## A2. The D11 frame contrast across the Pythia suite

Spec: `plan.md`, A2; report `a2_suite_d11.md`. All checks passed (1.4B
reconstruction 3.5e-5, identity 4.1e-5; 160M under the declared 1e-2
tolerances: 1.6e-3, 2.9e-3, h deviation 9.2e-4); every verb that round 5
had scored reproduces round 5's log-probs (1.4B: exactly; 160M: 2.7e-3).

**Primary, token-standardized Tail − XTail (freq prefixes):**

| model | ΔL_tok | reading | ΔΔ_pass(T) / −ΔΔ_pass(I) | side | held-out three-cell | ΔL Head − XTail | Head AUC | singleton D_L |
|---|---|---|---|---|---|---|---|---|
| 160M | +0.97 [+0.40, +1.54] | deficit | −0.22 / +1.19 | intransitive | +1.12 [−0.39, +2.64] | +1.52 [+0.92, +2.16] | 0.94 | +1.28 [+0.51, +2.01] |
| 410M | +1.26 [+0.66, +1.86] | deficit | −0.22 / +1.47 | intransitive | +0.64 [−0.72, +2.02] | +1.65 [+0.97, +2.31] | 0.95 | +1.39 [+0.47, +2.31] |
| 1B | +1.66 [+1.08, +2.22] | deficit | +0.04 / +1.62 | intransitive | +1.40 [−0.00, +2.90] | +1.64 [+0.94, +2.34] | 0.94 | +1.51 [+0.45, +2.45] |
| **1.4B** | **+0.99 [+0.43, +1.53]** | **deficit** | −0.41 / +1.40 | intransitive | +0.53 [−0.66, +1.63] | +1.41 [+0.71, +2.15] | 0.93 | +1.56 [+0.58, +2.47] |
| 2.8B | +0.89 [+0.23, +1.52] | deficit | −0.41 / +1.30 | intransitive | +0.46 [−1.06, +1.98] | +1.20 [+0.50, +1.93] | 0.93 | +1.45 [+0.47, +2.39] |
| 6.9B | +1.04 [+0.47, +1.57] | deficit | −0.48 / +1.51 | intransitive | +1.05 [−0.08, +2.19] | +1.64 [+0.93, +2.33] | 0.94 | +1.40 [+0.44, +2.29] |
| 12B | +0.94 [+0.38, +1.48] | deficit | −0.48 / +1.42 | intransitive | +0.07 [−0.99, +1.10] | +1.39 [+0.68, +2.09] | 0.94 | +1.50 [+0.52, +2.43] |

Held-out readings are "unresolved" at every size. 1.4B details:
- Band means of L: Head 3.37, Tail 2.56, XTail 1.96; Δ_pass(I) −2.89 /
  −2.23 / −1.12; Δ_pass(T) +0.48 / +0.33 / +0.84. Within every token cell,
  Tail intransitives are rejected more after "was" than XTail ones (2×2:
  −2.27 vs −0.90).
- Sensitivities: OLS adjustment for token counts +0.82 [+0.28, +1.36]; the
  2×2 cell alone +0.67 [−0.02, +1.36]. Every prefix set reads a deficit,
  intransitive-side (been +1.35, cur +1.12, they +0.89).
- Head − XTail (token-confounded) +1.41 [+0.71, +2.15], smaller than on
  the curated 126 (+2.12). Head − Tail +0.81. ΔM_has (token-standardized)
  −0.56: a surplus, so the reading of the deficit as was–has-specific is
  unresolved, as in round 5.
- Single-token verbs: Head AUC 0.93; the singleton contrast D_L +1.56
  [+0.58, +2.47] comes from the intransitive rows (D_I −1.25; D_T +0.31).
- Three-cell contrast: all pairs +0.82 [+0.22, +1.38] and trainable pairs
  +0.97 [+0.31, +1.63] are deficits; held-out pairs +0.53 [−0.66, +1.63].

**Against the predictions.** 1.4B: deficit in the predicted range (+0.5 to
+2.0) and intransitive-side — **held**. Every size ≥ 410M: deficit and
intransitive-side — **held**; size within ±50% of 1.4B's — held at five of
six sizes (1B: +1.66, 68% above), so not held as stated. 160M: deficit — held. Held-out at
1.4B: positive, deficit or unresolved — **held** (unresolved). Head AUC ≥
0.9 and singleton D_L above 0 — **held**.

## B3. Other alternations: object onset and clause end

Spec: `plan.md`, B3; report `b3_alternations.md`. Run in job 9069908. The
runner file on TSUBAME had been updated to the declared clarification
(`f8ff21b`: tolerance 1e-3, checked after saving) before the B3 step began
(pull before 14:03; B3 from 14:22), so B3 ran under it: batched-vs-single
differences 3.1e-5 to 9.0e-4, four models above the original 1e-4 (160M
9.0e-4, 410M 2.2e-4, 2.8B 1.2e-4, 12B 1.3e-4). The count analyses wait for
A1's counts.

Inventory: D11 (T = non-alternating transitives, I = non-alternating
intransitives). Frames "The N p" (primary) and "The N has v"; O = the
object-onset proxy set, "." = clause end (on the 197 eligible pairs).

**Pythia-1.4B, simple past (primary): side test.**

| readout | predicted side | its slope on Zipf | other side's slope | difference (side specificity) | B3 prediction |
|---|---|---|---|---|---|
| object onset O | log P(O \| I) | −0.31 [−0.52, −0.11], tracks frequency | T −0.04, flat | −0.27 [−0.49, −0.06], side-specific | **held** (strict pronouns agree: −0.72 [−1.10, −0.35]) |
| clause end "." | log P(. \| T) | −0.69 [−0.89, −0.48], tracks frequency | I −0.07, unresolved | −0.61 [−1.01, −0.24], side-specific | **held** |

- Band contrasts: ΔM_obj Head − XTail +0.82 [+0.39, +1.26], deficit,
  I-side (I part +0.99, T part −0.16); token-standardized Tail − XTail
  +0.63 [+0.33, +0.93], deficit, I-side. ΔM_end Head − XTail +1.27 [+0.52,
  +2.02], deficit, T-side (T part +1.54, I part −0.28); token-standardized
  +0.98 [+0.46, +1.50], deficit with mixed sides (T +0.60, I +0.38
  unresolved).
- In words: after "The N V-ed", rare intransitives leave more room for an
  object, and rare obligatory transitives leave more room for a period. By
  the primary side test and the Head − XTail side parts, each frame's
  deficit sits on the verb that the frame does not suit, as in the passive
  (rare intransitives after "was"). The exception is the token-standardized
  clause-end contrast, whose sides are mixed (the I part is unresolved, not
  shown to be absent).
- Sensitivities: with the token count as categories, all four readouts are
  side-specific. On multi-token Tail and XTail pairs only, O, the strict
  pronouns and "." are side-specific (the punctuation set is unresolved).
  Excluding I verbs with a rare-transitive flag from any rater leaves the
  token-standardized deficits (O +0.51; "." +0.92). By animacy, O is
  side-specific for animate and for inanimate nouns; "." is side-specific
  for inanimate nouns and unresolved for animate ones (T slope −0.47, I
  −0.19).
- Perfect frame ("The N has v"): the same pattern, larger (Head − XTail
  M_obj +1.10, M_end +1.93; all four side tests side-specific).
- Also in the report: all obligatory-T pairs for "." (219 pairs:
  token-standardized +0.80, deficit; Head − XTail +0.63, unresolved); log
  P(PREP) by band (T −2.88 / −3.04 / −2.81, I −0.75 / −0.91 / −0.98, Head /
  Tail / XTail: band differences of at most 0.23); released accuracies with item CIs.

**Suite (descriptive).** At every size ΔM_obj Head − XTail is an I-side
deficit (+0.55 to +1.26) and both token-standardized contrasts are deficits
(M_obj +0.48 to +0.70; M_end +0.64 to +1.00). ΔM_end Head − XTail is a
T-side deficit at 410M–12B (unresolved at 160M). Side specificity: the
strict pronoun set at every size; the broad O set at 160M–1.4B (unresolved
at 2.8B–12B, differences −0.16 to −0.20); "." at 410M–12B (unresolved at
160M).

**Released FreqBLiMP (descriptive association, 1.4B).** Item margin on the
bad verb's side measure: `causative` −0.79 [−1.05, −0.52], negative (items
whose bad intransitive leaves more room for an object have smaller
good − bad margins); `transitive` −0.21 [−0.50, +0.04] and `intransitive`
−0.25 [−0.52, +0.03], unresolved; `inchoative` and `drop_argument`
descriptive only (too few verb types).

**Against the predictions.** Object frame (tracks frequency, side-specific,
strict set agrees; Head − XTail I-side deficit; token-standardized deficit):
**held**. Clause end (tracks frequency, side-specific; Head − XTail T-side
deficit): **held**, although it was predicted with low confidence. Released
items: `causative` negative, held; `transitive` unresolved, so not
supported. Suite, "the same side specificity at every size": **not
established** (side-specific at 7 of 7 sizes for the strict pronouns, 6 of
7 for ".", 4 of 7 for the broad O set; unresolved sizes give no support).
Counts: pending A1.

## C4. Classify-from-z restore

Spec: `plan.md`, C4; report `c4_classify.md`. Job 9069908. All checks
passed: z reproduces A4's to 2e-16, affinities to 8e-7, edit targets to
6e-7; unedited rows are unchanged.

- **Classification.** θ = 0.514 from A4's Head singletons (282 T, 30 I;
  balanced accuracy 0.988 in sample, 0.971 leave-one-out; the sensitivity
  θ with D11's extra Head singletons is the same value). A4's 32 rare
  singletons: 31 correct, balanced accuracy 0.944 [0.833, 1.000] (T 1.00
  [1.00, 1.00], I 0.89 [0.67, 1.00]), above chance (seen before declaration; see the
  disclosure). The one error is
  *mutated* (I, z = 0.56, just above θ). D11's Tail singletons (secondary;
  28 of the 44 new): 44 of 44 correct, 1.000 [0.950, 1.000] (T 1.00
  [1.00, 1.00], I 1.00 [0.90, 1.00]).
- **Restoration.** On the curated passives, classify equals the oracle
  exactly, because *mutated* is in no curated pair. Affected `passive_1`
  margin +0.53 [+0.24, +0.83] (+6.5 pp accuracy); `passive_2` +0.53
  [+0.25, +0.83]; Head pairs unchanged. classify − 0.5 × oracle +0.26
  [+0.12, +0.42]; classify − oracle 0.000 (90% CI within ±0.1). The share
  of the curated deficit closed is not resolved (`passive_1` +0.49 [−2.07,
  +2.94]): the affected pairs' deficit is small and noisy.
- **Released items and specificity.** Tail `passive_2` affected items: 77
  → 81% (oracle 82%). Tail `passive_1` affected items: 73 → 73% (oracle
  75%); this is where *mutated* items occur, and the misclassified verb's T
  target cancels the gain. Unaffected items ±0.00 pp; Pile-10k Δ NLL
  0.0000 per token. NLL at the edited tokens on Pile-validation: classify
  +0.080 [+0.029, +0.137], oracle +0.058.
- **Against the predictions.** Improves: **held**. Half the oracle:
  **held**. It matches the oracle on the curated pairs, consistent with the
  conditional prediction (the misclassified verb is in no affected pair; it
  costs the gain on released `passive_1` items). Specific: held. D11 Tail
  balanced accuracy above 0.5: held.
- **Reading.** A rare verb's own site-8 class coordinate carries its class
  label: a threshold calibrated on Head verbs reads it correctly for 31 of
  32 rare singletons, and an external edit to the Head class targets
  driven by that label matches the oracle edit on the 12 + 12 affected
  curated pairs (not on released `passive_1` items, where the one error
  sits). This shows that the information is present early; it does not
  show a native causal path from z to the output rows. Round 5's linear
  prediction of the target from z did not establish half the oracle
  (Δ_LF − 0.5 Δ_oracle −0.02 [−0.22, +0.19]).

## C5. In-context preemption

Spec: `plan.md`, C5; report `c5_preemption.md`. Job 9070274 (rerun under
the declared clarification; batched vs single 1.7e-4). At k = 0 the run
reproduces A1's D11 scores exactly.

**Primary** (233 rare dynamic intransitives; "was" / "has"; k = 8; each
arm minus the length-matched filler arm):

| arm | E (95% CI) |
|---|---|
| preemptive ("The S V-ed.") | **+1.17 [+0.89, +1.43]: raises** |
| exposure-only (nominal -ing) | +0.09 [−0.23, +0.42] |
| exact-form mention | +2.30 [+1.94, +2.65] |
| another intransitive's preemptive sentences | +0.74 [+0.54, +0.93] |
| present "V-s" | +0.58 [+0.28, +0.87] |
| "is V-ing" | +0.01 [−0.30, +0.32] |
| T control (rare T in transitive use, vs its filler) | −0.08 [−0.39, +0.26] |

- **Decisions.** E_pre raises Δ_pass(I); the prediction was that it
  lowers it. Beyond exposure: no (E_pre − E_exp +1.08). Beyond exact-form
  mention: yes (−1.13 [−1.40, −0.86]). Verb-specific: no (E_pre − E_other
  +0.43, above 0). T control does not fall: not established (lower bound
  −0.39). **In-context preemption is not claimed.** Rare verbs move away
  from the Head level, not toward it (distance grows by 1.17).
- **What happens instead.** Seeing the verb's own past form in the context
  raises its log-prob after both auxiliaries by 11–12 nats, slightly more
  after "was" (+12.4) than after "has" (+11.2), so Δ_pass(I) ends closer
  to 0 than under matched filler; the pre − filler difference grows by
  +0.32 [+0.23, +0.41] per doubling of k. (In absolute terms Δ_pass(pre)
  goes from −1.74 at k = 0 to −2.39 at k = 8, not monotonically, because
  the filler arm alone lowers Δ_pass with k: rare I −1.74 → −3.56; Head I
  −2.81 → −4.44; rare T +0.66 → −0.54. Every contrast is taken against
  filler for this reason.) The exact-form mention arm raises Δ_pass most
  (+2.30), through the largest boost after "was" (+13.1; after "has"
  +10.8, less than the preemptive arm's +11.2).
- **Not a simple compression.** Large boosts do not always shrink Δ_pass:
  the "is V-ing" arm raises the participle by +10.9 after both auxiliaries
  with Δ_pass near 0 (+0.01 [−0.30, +0.32], unresolved), and the T
  control's +13.7 / +13.8 comes with E_T −0.08 [−0.39, +0.26]
  (unresolved); another verb's intransitive sentences raise
  Δ_pass(I) (+0.74) without repeating the target. So the result is form
  repetition favouring "was" plus arm- and frame-dependent context effects
  (the secondary frame behaves differently, below). In-context copying is
  a plausible part of it, which the plan's low-confidence caveat
  anticipated; these controls do not identify the mechanism.
- **The one contrast in the predicted direction.** Intransitive uses leave
  Δ_pass(I) 1.13 nats lower than metalinguistic mentions of the same form
  (declared "beyond mention": held). It combines less facilitation after
  "was" (pre − mention −0.73) with more after "has" (+0.40). The two arms
  also differ in sentence type, and another verb's intransitive sentences
  raise Δ_pass(I) by +0.74 on their own, so this relative difference does
  not identify preemption.
- Breakdowns (descriptive): Tail +1.09, XTail +1.27; no complement +1.20,
  with complement +0.94; single-token rare I (20 verbs) −0.43 [−0.98,
  +0.05], multi-token (213) +1.32 [+1.03, +1.59]. Head I E_pre +0.26
  [−0.06, +0.58]. Including statives: +1.16. Theme subjects (the
  affected-entity reading) +1.28 [+0.64, +1.90]. Head level (Head I under
  filler, k = 8) −4.44; rare − Head gap G +0.88 [+0.31, +1.41]; share
  closed −1.33 [−3.93, −0.72] (the gap widens).
- Secondary frame ("has been" / "has"): E_pre +0.60 raises; beyond
  exposure here (E_exp +1.27); the T control falls (−0.95 [−1.24, −0.67]).
- **Against the predictions.** Lowers: **failed (reversed)**. Beyond
  exposure: failed. Beyond mention: held. Verb-specific: failed. T control:
  not established. Moves toward Head: failed. The mismatch arms lower
  Δ_pass less than the preemptive arm: failed (reversed; E_pre − E_vs
  +0.59, E_pre − E_ing +1.15: the preemptive arm raises Δ_pass more than
  the present-tense arm, which raises it, and the "is V-ing" arm, whose
  effect is near 0 and unresolved).
  Raw log-probs rise after both auxiliaries in the preemptive and mention
  arms: held. Less after "was" in the preemptive arm: failed (more).

## A6. Checkpoints: when the deficit appears, and whether exposure predicts it

Spec: `plan.md`, A6; report `a6_checkpoints.md` (Δ_pass(T) version,
descriptive: `a6_checkpoints_T.md`). Job 9070046: all 16 + 16 + 6
checkpoints scored, none missing, every check passed (reconstruction ≤
4.5e-5, h deviation 0); the final checkpoints reproduce A2. The analysis
used A1's c_total (the four c_total query families, complete) before A1's
other counts had finished; the values are those of the final count file.

**Stationarity (diagnostic).** Slice-count rate ratio, last to first: Head
single-token participles 1.010 [0.996, 1.025], Tail 1.015 [0.957, 1.077];
stationary within ±20% (the ten most frequent Head verbs individually
0.96–1.05).

**Trajectories, 1.4B** (L by band, Head / Tail / XTail; ΔL Head − XTail;
ΔL_tok):

| step | L | ΔL Head − XTail | ΔL_tok | Δ_pass(I) Head / XTail | Head AUC |
|---|---|---|---|---|---|
| 64 | +0.07 / −0.05 / +0.02 | +0.06, negligible | −0.02, negligible | +0.06 / +0.05 | 0.54 |
| 256 | +0.28 / −0.07 / −0.09 | +0.37 [+0.14, +0.62], deficit, **T-side** | −0.02, negligible | +0.25 / +0.28 | 0.69 |
| 512 | +1.78 / +0.06 / +0.02 | +1.76, deficit, I-side | −0.11, unresolved | −1.19 / +1.09 | 0.90 |
| 1000 | +3.19 / +1.06 / +0.17 | +3.03, deficit, I-side | +0.66 [+0.21, +1.12], deficit | −2.95 / +0.40 | 0.95 |
| 4000 | +3.32 / +2.21 / +0.94 | +2.38, deficit, I-side | +1.52, deficit | −2.95 / −0.12 | 0.93 |
| 16000 | +3.07 / +2.44 / +1.70 | +1.37, deficit, I-side | +1.07, deficit | −2.79 / −1.03 | 0.92 |
| 143000 | +3.37 / +2.56 / +1.96 | +1.41 [+0.70, +2.15], deficit, I-side | +0.99 [+0.44, +1.55], deficit | −2.89 / −1.12 | 0.93 |

- The deficit reads "deficit" at every 1.4B checkpoint from step 256
  (Head − XTail) and from step 1000 (token-standardized); 410M from 512 and
  2000; 6.9B at every checkpoint scored (from 512; token-standardized from
  2000). Descriptive onset, no simultaneous inference.
- Shape (descriptive; no longitudinal test): by step 1000 Head
  intransitive participles are much less probable after "was" than after
  "has" (Δ_pass(I) −2.95), while for XTail intransitives Δ_pass(I) is
  still positive at steps 512–3000 (+0.4 to +1.1). The Head − XTail gap is
  largest at steps 1000–3000 (about 3 nats) and then narrows as XTail
  Δ_pass(I) falls (+0.49 at 3000 → −1.03 at 16000); from step 16000 to the
  end (about 90% of training) it shows no trend (+1.21 to +1.63; XTail
  Δ_pass(I) −0.82 to −1.30). The token-standardized gap develops later and
  is largest at step 10000 (+1.63). 410M: Head − XTail largest at
  2000–3000 (about 3 nats), +1.99 at 16000 → +1.65 at the end; 6.9B
  (6 steps): +2.66 at 2000, +1.40 to +1.64 from step 8000.

**Curve collapse (Δ_pass(I), 1.4B, primary).** CV MSE: exposure-only M_E
2.123, frequency + step M_Cg 2.077, free 2D M_2D 2.029, step-only M_g
2.237.
- CC1 (M_E → M_2D): +0.044 [+0.005, +0.071] → **unresolved** (neither
  holds nor fails).
- CC2 (M_E → M_Cg): +0.021 [−0.014, +0.049] → unresolved.
- Usefulness (M_g → M_E): +0.051 [−0.006, +0.105] → **not established**.
- **Curve collapse is not supported** (and not rejected). Step terms add
  beyond exposure (M_E → M_Eg +0.065 [+0.041, +0.088], diagnostic); at
  matched exposure, later checkpoints have slightly higher Δ_pass(I) (mean
  +0.08 over 226,550 matched cell pairs).
- Steps ≥ 1000: M_E useful (+0.079 [+0.026, +0.128]); CC1 +0.022 [−0.021,
  +0.052] and CC2 unresolved. Steps ≥ 4000: CC1 +0.034 [−0.011, +0.071],
  unresolved; M_E not useful by the 2% rule (+0.054 [+0.003, +0.101]).
  c_marked in place of c_total: the same to three decimals (the newline
  forms add 1.7% to c_total on average; r = 0.99997 on the log scale).
- 410M (replication): M_E useful (+0.079 [+0.025, +0.133]); CC1 +0.025
  [−0.010, +0.047] and CC2 +0.002 unresolved; the same for steps ≥ 1000
  and ≥ 4000 (M_E useful in both). 6.9B (6 steps, descriptive): collapse
  fails (+0.086 [+0.043, +0.110]), frequency + step more predictive
  (+0.070 [+0.031, +0.097]), M_E not useful; from step 1000 (5 steps), CC1
  and CC2 unresolved and M_E useful.
- Bootstrap correction: the first run resampled pairs within band ×
  held-out split for the collapse CIs (the trajectories' A2 scheme); the
  declared collapse scheme is within band, and the figures above use it.
  Every collapse verdict is the same under both (first run, 1.4B: CC1
  +0.044 [+0.007, +0.070]; usefulness +0.051 [−0.005, +0.107]).

Descriptive sensitivities (not gated):
- Δ_pass(T) (`a6_checkpoints_T.md`): exposure predicts Δ_pass(T) worse
  than step terms alone (1.4B M_g → M_E −0.030 [−0.054, −0.012]; 410M
  −0.017), as expected when Δ_pass(T) has little frequency trend.
- L (`a6_checkpoints_L.md`; pair exposure = geometric mean of the two
  verbs' counts): all steps, CC1 +0.022 [−0.014, +0.042] and CC2
  unresolved, M_E not useful (+0.014). From step 4000, CC1 reads "collapse
  holds" (−0.005 [−0.041, +0.017]) and exposure is as predictive as
  frequency + step (CC2 −0.007 [−0.043, +0.017]), but M_E does not beat
  step-only by 2% (+0.022 [−0.005, +0.047]). 410M: unresolved throughout.
- The report now also gives, per checkpoint, band means of Δ_pass(T) and
  Δ_pass(I) and the numerical side parts (added after the first draft, at
  the review's request; the plan lists them).

**Against the predictions.**
- 1.4B deficit from step 2000 onward: **held** (from 256). ΔL_tok deficit
  at the final step: held. Intransitive-side wherever it reads deficit:
  **failed at one checkpoint** (step 256, Head − XTail +0.37, T-side);
  held at every step from 512. Head AUC above 0.9 by step 4000: held (0.90
  at 512, 0.95 at 1000).
- All steps: M_E useful — **not established**; CC1 collapse fails —
  **not established** (unresolved); steps ≥ 4000 collapse holds — not
  established (unresolved); CC2 unresolved — as predicted.
- 410M, the same readings: trajectory readings held (deficit from 512,
  intransitive-side wherever it reads deficit, AUC 0.94 by 1000); on the
  collapse, 410M differs from 1.4B in that M_E is useful, and CC1 is
  unresolved in both. 6.9B deficit at every step ≥ 2000: held.
  Stationarity holds: held.

## A7. Fit controls and robustness

Spec: `plan.md`, A7; report `a7_fit_controls.md`. Ratings: job 9070047
(Gemma-4-31B-it; 41,047 new sentences; digit-mass median 1.000; prompt
agreement Spearman 0.957; Spearman 0.906 against a blind audit of 60
sentences that Claude rated before any rating existed). Released-context
scores: job 9070046.

- **Matched fit** (1.4B, curated prefixes, 281 pairs × 126 contexts).
  XTail − Head on M_was: −1.15 [−1.74, −0.58] unmatched, −0.72 [−1.32,
  −0.13] at matched good + bad-patient fit, so it **survives matched fit**
  (attenuation −0.42 [−0.62, −0.26]). Tail − Head −0.63 [−1.20, −0.06]
  also survives. On L, XTail −1.06 [−1.75, −0.33] survives (Tail −0.50
  [−1.19, +0.17], unresolved); at matched fit
  the XTail deficit is on the intransitive side (−Δ_pass(I) −1.42,
  Δ_pass(T) +0.36). Sensitivities agree (bare-clause I verbs only −0.86;
  quadratic −0.72; band-specific slopes −0.77). Token-standardized and
  fit-adjusted Tail − XTail: L +0.99 [+0.46, +1.58], deficit; M_was +0.32
  [−0.18, +0.79], unresolved. The accounting β(M_was) = β(L) + β(M_has)
  holds to 2e-15; M_has has no resolved band coefficient (XTail +0.34
  [−0.21, +0.86]).
- **Noun rarity** (300 released `passive_1` contexts, which carry noun
  bands). The context (noun) effect on L is negligible (+0.09 [−0.03,
  +0.21]) and the verb effect is a deficit (−1.28 [−1.88, −0.67]); M_pass
  likewise (+0.07; −0.72). Secondary, on the curated contexts: the verb
  effect is a deficit, and the context effect on L is +0.27 [+0.07, +0.49]
  (there the context band is the band of the context's original verb, not
  noun frequency).
- **Robustness** (table in the report, with side parts). Every prefix set
  (freq, been, cur, they) reads a deficit Head − XTail (+1.23 to +1.41) and
  token-standardized (+0.89 to +1.35), intransitive-side, as does every
  suite size; the curated 126 (round 5) read Head − XTail deficits for
  every set (+1.73 to +2.12), with the I part above the T part. The side
  parts and the curated rows' readings were added to the table after the
  first draft of the report. A2 stored no CIs for its Head − XTail parts
  (shown as point estimates); round 5 stored 95% CIs only and no paired
  draws, so the curated readings use the 95% part of A2's rule and their
  sides are a descriptive comparison (non-overlapping intervals), not A2's
  paired side rule.
- **Against the predictions.** M_was survives, estimate in −0.5 to −2.0:
  **held** (−0.72). L survives: held. Token-standardized deficit or
  unresolved: held (L deficit, M_was unresolved). Noun effect negligible,
  verb effect deficit: held.
