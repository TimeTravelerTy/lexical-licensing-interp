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
