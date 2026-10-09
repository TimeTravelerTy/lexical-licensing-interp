# Round 4: declared plan

Builds on round 3 (`../passive_das_prep/round3_plan.md`,
`../passive_das_prep/round3_results.md`). Each part is written and committed
before any of its runs. Predictions are stated before the data exist; where a
prediction leans on round-3 outputs (dose curves, existing D values), that is
said. Notation as in round 3:
- site s = residual stream output of layer s − 1 (0 = embeddings);
- d_s = the active-trained rank-1 DAS direction at site s (15 fold × split
  bases, `results/das_round2/bases_rank1.npz`); d_p = the passive-trained
  direction of B5;
- z = the sign-aligned coordinate on d_s, 0 = held-out active intransitive
  level, 1 = held-out active transitive level;
- primary population = the 64 plain-bad-verb pairs; cross-fitting as in
  `passive_test_plan.md`;
- D = mean change after a transitive (T) minus an intransitive (I) active
  donor on the bad item;
- CIs are 95%, 2,000 draws, seed 17, unless stated otherwise.

Layout: plan and reports in `reports/round4/`, regenerable inputs in
`data/round4/<part>/`, outputs in `results/round4/<part>/`.

# Part A: confirm current claims (declared 2026-10-09)

Revised after an outside review (Codex) of the first draft, before any run.

## A1. Dose check for C9 (no model)

**Question.** At site 8 the T − I donor patch reproduces each construction's
own natural transitive-vs-intransitive profile, plus excess object mass in
passives and tough constructions (round 3, C9 post hoc). T and I donors set
the coordinate to the active levels (z ≈ 1 and ≈ 0), while each
construction's natural good/bad span is narrower. Does the excess object mass
track how far the donors overshoot each construction's natural range?

**Already known.** The natural z gaps and their ratio to the active gap were
printed in `round3_constructions.md` (column "z gap / active"), and the D(O)
and natural O gaps in `round3_results.md`. From those: ratio at site 8 =
0.42 (passive), 0.85 (OR), 0.62 (TC); excess object mass ≈ +0.58, −0.08,
+0.36. A1 adds intervals and paired comparisons; it is a confirmatory
re-analysis, not a blind test. The causal test is A2.

**Measures, per construction c ∈ {passive, OR, TC} and site (4–17).**
Items: bad and good items of the 64 primary pairs (passive-test items for
passives; C9 items for OR and TC). From existing outputs only.
- Natural levels z̄_good,c and z̄_bad,c (cross-fitted z as in round 3), gap
  g_c and ratio ρ_c = g_c / (active gap of the same verbs).
- Donor levels z̄_T and z̄_I over the plan rows of bad bases.
- Overshoot: upper u_c = z̄_T − z̄_good,c (primary); lower l_c = z̄_bad,c −
  z̄_I; span excess (z̄_T − z̄_I) − g_c.
- Excess object mass E_c = D(O) − (natural good − bad log P(O)), same items.

**Inference.** One value per pair (mean over its contexts and donors); a
pair bootstrap with pairs resampled within band, jointly for the three
constructions, so differences between constructions are paired. This is
inference over pairs, conditional on the sampled contexts, donors and bases.
Sensitivity (site 8): the same quantities with contexts and donor pairs also
resampled (independently per construction, pairs jointly).

**Decision rule (site 8).**
- **Dose differs by construction** if ρ_passive < ρ_TC < ρ_OR, with both
  paired differences' 95% CIs excluding 0 (descriptive of the natural span).
- **The excess tracks the overshoot** if u_passive > u_TC > u_OR and
  E_passive > E_TC > E_OR, each with both paired differences' CIs excluding
  0. E is in log-probability units (excess object log-probability).
- If u is ordered but E is not: "overshoot differs but does not explain the
  excess". Otherwise unresolved. With three constructions this is an
  association; the causal test is A2.
- Per-pair relation (descriptive only): within each construction, the slope
  of E on u across pairs. It is confounded: the good verb enters both u
  (through z̄_good) and E (through the natural O gap), so a positive slope
  can arise without any dose mechanism.
- A1 and A2 use all 64 pairs; A3 is a separate check (if A3 reads
  "affected", A1/A2 OR results are rerun without that pair as a
  sensitivity).

**Prediction.** ρ_OR 0.8–0.9 (≈ 1 is not reached), ρ_TC 0.55–0.7, ρ_passive
≈ 0.4; the dose ordering holds, and u is ordered the same way (site 8 levels:
u ≈ 0.55, 0.40, 0.23). E_passive > E_OR and E_TC > E_OR exclude 0;
E_passive > E_TC may not, so "tracks" may fail on that one difference.

## A2. In-range patch

**Question.** If the coordinate is moved only within each construction's own
natural range, does the patch reproduce that construction's natural profile
(in size and sign), and does the excess object mass disappear?

**Items.** Bad items of the 64 primary pairs: passives (passive-test items,
126 curated contexts, ~125 per pair), OR and TC (C9 items, 24 contexts).

**Patch (coordinate-setting, as A3 of round 3).** At site s, on the verb's
last token: h ← h + (t − h·d) d, d = the item's cross-fitted fold basis for
split k (the existing plans' fold assignments; all 3 splits). Two targets,
in raw projection units of the same basis:
- t_T = mean projection of the construction's **good** items in the same
  context, over primary pairs other than the item's own and outside the
  basis's DAS training pairs;
- t_I = the same mean over **bad** items.
So t_T − t_I is the construction's natural span in that context, under the
basis used for the patch.

Good and bad target means use the same eligible pairs. Every context has
61–64 primary pairs, so no target is missing. Targets are computed once from
the natural pass and treated as fixed; inference conditions on them.

**Estimand.** D_in = R(t_T) − R(t_I) on the bad item, per readout: the
change from setting the coordinate at the construction's intransitive class
mean to its transitive class mean, in the same context. The rest of the
state is the bad verb's, so comparing D_in with the natural good − bad gap
tests whether the d coordinate alone, moved within range, produces the
construction's profile (calibration). It does not measure the share of the
natural difference mediated by d. Also reported: R(t_T) − R(unpatched) and
R(t_I) − R(unpatched). Bootstrap: pairs within band × contexts (as the C9
natural gap); per draw, the ratio of population means D_in / natural.

**Readouts** (log-probs at the verb's last token; C9 sets): O, " by", ".",
PREP, PREP without " by", MAIN, END; L_OR = MAIN − PREP, L_TC = END − PREP.
PREP contains " by", so for passives it is descriptive only.

| Construction | Sign readouts | Ratio readouts |
|---|---|---|
| passive | " by", ".", O | " by" |
| OR | MAIN, PREP, O, L_OR | MAIN, PREP |
| TC | END, PREP, O, L_TC | END, PREP |

**Decision rule (sites 6 and 8, per construction; other sites
descriptive).** **Reproduces the profile** iff
- (a) for every sign readout whose natural gap has a 95% CI excluding 0 and
  |estimate| ≥ 0.1 nats, D_in's 95% CI excludes 0 on the side of the natural
  gap; and
- (b) for every ratio readout, D_in / natural is within [0.5, 1.5] (point
  estimate; CI reported). A ratio readout whose natural gap is below 0.2
  nats in magnitude makes the result unresolved.

**Excess removed** (only for constructions whose T − I excess E was positive
in A1, i.e. passive and TC): the upper 95% bound of D_in(O) − natural O is
≤ 0.2 nats. **Excess remains**: its lower bound is > 0.2. Otherwise
unresolved.

**Prediction** (from the round-3 site-8 dose curve and the T − I D values,
scaled by each construction's span; at site 8 the spans are z 0.09–0.45
(passive), 0.06–0.60 (TC), 0.04–0.77 (OR)).
- Passive: D_in(" by") / natural ≈ 0.45 (0.35–0.6); D_in(O) ≈ +0.1 while
  the natural O gap is negative (−0.22): if that natural gap's CI excludes
  0, the sign fails. **Does not reproduce** at 6 or 8, and excess object
  mass remains.
- OR: MAIN ≈ 0.75, PREP ≈ 0.6, O ≈ 0.6–0.7 of natural, signs match:
  **reproduces** at 6 and 8.
- TC: END ≈ 0.7, PREP ≈ 0.45–0.55 (borderline), O ≈ 0.6–0.9: borderline;
  excess removed.

## A3. OR verb forms (no model)

**Check.** Pairs whose verbs have simple past ≠ participle, so the OR item
("The N that John V-past") puts a form under d that d was not trained on.
Among the 64 primary pairs (both verbs, `verb_forms.csv`) there is exactly
one: tail/outdo/recede (good past "outdid", participle "outdone"). A
base-form-style gate on one verb is not meaningful, so the pair is dropped.

**Rerun.** OR at site 8 from the existing outputs without that pair: D(O),
D(L), D(MAIN), D(PREP), natural gaps, reading (same machinery as C9).

**Rule.** The C9 site-8 OR result is **unaffected** if every estimate moves
by < 0.05 and the reading is unchanged. **Prediction:** unaffected.

## A4. B5 by-lever

**Question.** In B5, d_p raised " by" in actives at every site. Shuffled-
label passive DAS also finds a weaker copy of the passive effect (+0.5 to
+1.0 Δ M_p), and its direction shares |cos| 0.50–0.55 with d_p (median per
site) but only 0.03–0.23 with d_a (computed from the saved bases). Is the
" by" rise in actives a generic "by lever" that needs no verb-class labels?

**Run.** The B5 active test (`data/das_round2/reverse_test/plan.csv.gz`,
hash-checked), bad active bases, arms 1 (G vs B passive donors) and 3 (T vs I
active donors), all eight sites, with three new interventions per site,
split and fold (unit-norm bases):
- **s**: interchange along the shuffled-label passive basis
  (`shuf_s{k}_f{f}`, saved by the B5 final runs);
- **d_p⊥s**: interchange along e = (d_p − (d_p·s) s) / ‖·‖. This recomputes
  the donor − base coordinate along e, so direction and dose both change;
- **d_p − s-part** (dose-preserving control): d_p's own displacement with
  its component along s removed, Δ = ((h_src − h)·d_p)(d_p − (d_p·s) s).
Per-row displacement norms are saved for every intervention. Nulls: 100
random rank-1 directions on the split-0 G/B rows of bad bases, norm-matched
row by row to the s displacement, and another 100 norm-matched to the d_p⊥s
displacement. Readouts and bootstrap as B5; δ_s = the active-DAS bound
(0.42–0.57).

**Decision rules.**
- **The shuffled direction raises " by" in actives** at site s if its arm-1
  D(" by") has a 95% CI above 0, an estimate ≥ 0.2 nats, and its split-0
  point estimate lies beyond the 95th percentile of its own null on the
  same rows.
- Only at those sites are d_p⊥s and d_p − s-part read (all are run in one
  job; results at other sites are reported, not interpreted). Per site,
  with paired differences against d_p on the same rows:
  - **by lever separable** if arm-1 D(" by") under d_p⊥s is ≤ 50% of d_p's
    and, where d_p's arm-1 or arm-3 D(O) was a RISE, D(O) under d_p⊥s is
    still a RISE and beyond its own null's 95th percentile;
  - **the by lever carries the transfer** if, where d_p's D(O) was a RISE,
    the paired reduction D(O)[d_p] − D(O)[d_p⊥s] has a 95% CI above 0 and an
    estimate ≥ δ_s, together with a " by" reduction of ≥ 50%;
  - otherwise mixed.
  The dose-preserving control is read the same way; if the two disagree,
  the difference is attributed to the change in dose and reported.
- Reported alongside: |cos(d_p, s)|, |cos(d_a, s)|, |cos(d_p⊥s, d_a)|;
  displacement norms; s's arm-1 and arm-3 D(O) with the B5 classification.
- Limit: s comes from one label permutation per fold; random directions do
  not control for training on the *by* readout. More permutations would need
  new DAS training and are not run in Part A.

**Prediction.** The shuffled direction raises " by" in actives at every
site (≥ 0.3 nats, below d_p's), with D(O) NO RISE. After projecting it out,
" by" falls by ≥ 50% at every site, and the O RISE at 4–6 stays: **by lever
separable** (both versions).

## A5. Auxiliary heads

**(a) Generic previous-token scores.** For every head, the mean attention
from position t to t − 1 (t ≥ 1) on (i) 256 random-token sequences of 64
tokens (ids uniform in [1000, 50000), seed 17) and (ii) 256 natural-text
passages of ≥ 48 tokens (good sentences from the 67 BLiMP paradigms,
concatenated; `data/round4/aux_heads/natural_text.txt`, seed 17). A head is a
**previous-token head** if its score is ≥ 0.5 on both. Reported for all 384
heads, with ranks, for the B7 top 15.

**(b) B7 rerun in two new frames.** Same items and rows as B7 (primary bad
passives, 32 contexts per pair, seed 17; split-0 basis; all T donors;
site-8 T-donor interchange in every frame; fp32):
- **adverb:** base "The N was quickly V", counterfactual "The N has quickly
  V" (sensitivity "had quickly", for S only). " quickly" is one token.
- **got:** base "The N got V", counterfactual "The N has V" (sensitivity
  "had").
Quantities as B7: S_l, group and single-component PEs, interaction; heads
ranked on half A (as B7). On half B: the **fixed B7 top 5** (L9H7, L10H2,
L3H3, L1H6, L8H4) and the newly selected top 5 replaced jointly; per head
set, the value-only replacements at the auxiliary, at the adverb (adverb
frame), at all positions, and pattern-only. Base-run attention from the
participle's last token to the auxiliary, to the adverb and to the previous
token, for every head.

**Token checks.** Frames are built from token ids, not by string search:
in every frame the context tokens, the auxiliary position and the
participle tokens are checked, and the auxiliary and adverb indices are
recomputed. Value-only replacements are aligned within each frame pair
(base and counterfactual have the same length). Attention and value-only
results are also reported separately for single-token and multi-token
participles (auxiliary 1 vs ≥ 2 tokens back in the plain frames).

**Decision rules, per frame.**
- **Gate** (as B7): Δ has the step-1 active sign for ≥ 7 of 9 flagged MLPs.
- **Eligible MLPs** (as B7): |S_l| ≥ 0.05 in that frame. Fractions are
  computed only where the signed all-heads PE (sign(S)·PE) is ≥ 0.05; if
  fewer than 3 " by" MLPs qualify, the head criteria are unresolved.
- **H1** (as B7): joint heads ≥ 0.5 |S| for at least half of the eligible
  MLPs.
- **The B7 heads retain their role** if all three hold: (i) H1 holds for the
  " by" MLPs (11, 13, 14, 16, 17; ≥ 3 of the qualifying ones); (ii) the fixed
  B7 top 5's signed joint PE has a 95% CI (pair bootstrap, half B) above 0,
  and as a fraction of the all-heads PE is ≥ 0.5 × its B7 half-B fraction,
  for ≥ 3 qualifying " by" MLPs; (iii) ≥ 3 of the B7 top 5 are in the new
  frame's top 10 (ranked on half A). Because B7's own fraction was 0.34–0.78,
  (ii) allows the five heads to carry as little as ~20% of the head effect:
  this criterion says the same heads keep a share, not that they carry most
  of the switch.
- **Read the auxiliary across the adverb** (adverb frame): ≥ 3 of the fixed
  top 5 attend ≥ 0.3 to the auxiliary, and the auxiliary-value-only
  replacement of the fixed top 5 gives ≥ 50% of their joint PE for at least
  half of the qualifying MLPs (as B7).
- Caveats: generic previous-token scores cannot rule out task-dependent
  local attention; got → has also changes aspect and interpretation, and
  was → has the subject's role (as B7).

**Prediction.** (a) At most one of the B7 top 5 (L1H6 or L3H3) is a
previous-token head; L8H4, L9H7 and L10H2 are not. (b) got: the gate, H1 and
"retain their role" all hold. Adverb: the gate and H1 hold, the B7 heads
retain their role, and they read the auxiliary across the adverb (L8H4,
L9H7, L10H2 keep ≥ 0.3 attention to it); an early head may shift its
attention to the adverb.

## Pre-run clarifications (Part A, from a code review, before any GPU run)

- A4: "beyond its own null" is applied in every arm it is used for, so nulls
  are generated for arm 1 and arm 3 of d_p⊥s and of the dose-preserving
  control (each norm-matched to its own displacement), not only for arm 1 of
  d_p⊥s. "Carries the transfer" needs the paired O reduction in every arm
  where d_p's O effect was a RISE. Where d_p has no O RISE in either arm, the
  O condition of "separable" is vacuous and the reading rests on the " by"
  reduction (reported as such).
- A5: "qualifying" MLPs are the eligible MLPs with signed half-B all-heads PE
  ≥ 0.05; the read-the-auxiliary rule counts an MLP where the fixed-five joint
  PE is ≥ 0.05 and the auxiliary value gives ≥ 50% of it, out of the
  qualifying MLPs, and is unresolved with fewer than 3 qualifying " by"
  MLPs. H1 with no eligible MLP is unresolved.
- Plans in use are checked against the committed ones by rebuilding them
  (`verify_plan.py`): the builders hash the in-memory plan, which a CSV read
  does not reproduce exactly.
- A1 and A3 ran on committed outputs right after the plan commit
  (`a1_a3_dose.md`).

## Order (Part A)

1. Commit this part.
2. A1 and A3 locally on committed outputs.
3. A2, A4, A5 on TSUBAME (one job each).
4. Report in `results.md`.

# Part B: filler-gap test (declared 2026-10-09)

Written after Part A's A1–A3 and A5 results and before any Part B run;
revised after an outside review (Codex) of the first draft and its code,
also before any run. From now on this is the main gap-construction test. Directions: the active-trained
d_s, sites 4–17 (decisions at 6 and 8; later sites for the depth profile).

## Verbs (both experiments)

Candidate pools (`data/round4/fillergap/candidates.csv`): T = the
inventory's transitive verbs, the primary good verbs and the DAS
transitives; I = the round-2 `plain` intransitives plus a hand-added set of
frequent intransitives (arrive, sleep, sit, stay, wander, …). Two raters
(Claude; Codex blind to the other's labels) judged each verb against
written criteria (in the file header of `verbs.csv`):
- **T:** obligatorily transitive ("John V-ed." alone is incomplete; no
  common object drop); typical object inanimate, so "what John V-ed" is
  natural; no particle and not mainly clause-taking; not jargon.
- **I:** intransitive with no common transitive use; natural with a human
  subject; a common adjunct-like PP that strands naturally ("what John
  emerged from", "slept in", "arrived at"); not a prepositional verb with a
  selected preposition ("rely on", "look at", "reside in") and not
  particle + P.
A verb is used only if **both raters accept it** and its simple past equals
its participle (d was trained on participles). The final lists are in
`data/round4/fillergap/verbs.csv`. DAS training verbs are cross-fitted as
before (their pair's held-out fold only).

## B6. Embedded wh, one-word toggle

**Items.** "I know that NAME V-ed" (**that**) vs "I know what NAME V-ed"
(**what**); the two differ in one token (" that" / " what"), and the verb's
tokens are checked to be identical. 24 contexts: matrix verb {know,
remember, forgot, heard} × NAME {John, Mary, Tom, Sarah, Peter, Anna}
("I forgot what John destroyed"). Every T and I verb in every context and
both frames.

**Readouts** (log-probs at the verb's last token): "." ; PREP (the 26-token
C9 set); O (the 27 object-start tokens). **L = log P(".") − log P(PREP)**
(main). Secondary: END (".", ",", "!", "?", ";") in place of ".".

**Context halves.** The 24 contexts are split once (seed 17) into half A
and half B, 3 names per matrix verb in each. Half A selects I verbs; half B
carries the natural gate and the primary patch analysis, so neither is
computed on the data that selected the verbs. All 24 contexts are
reported as secondary.

**Stage 1 (natural pass, no patching).**
- **I-verb gate (half A).** An I verb is kept only if the filler raises
  PREP after it: per verb, log P(PREP | what) − log P(PREP | that), mean
  over the 12 half-A contexts, > 0 with a context-bootstrap 95% CI
  excluding 0.
- **Natural gate (half B; kept I verbs, all T verbs):** filler × class
  interaction, on L: [L_T − L_I]_what − [L_T − L_I]_that > 0, and on O:
  [O_T − O_I]_that − [O_T − O_I]_what > 0; both with 95% CIs above 0.
  Bootstrap: T verbs and I verbs resampled independently, contexts jointly
  (2,000 draws, seed 17). Also reported on all contexts and on all I verbs.
  **If the natural gate fails, stop: no patching.**
- Descriptive: the interaction per matrix verb; the prepositions the filler
  raises most after I verbs (top tokens by probability change).
- The gates are computed and committed before stage 2 is run.

**Stage 2 (patches, kept I verbs as bases, both frames, all eight sites).**
- **T − I donors**, as C9: per verb × context × split the cross-fitted
  fold, shared by the two frames; donors = that fold's held-out DAS pairs
  (excluding the verb's own), each contributing its transitive and
  intransitive active ("She has destroyed" / "She has emerged") with one
  shared random subject, also shared by the two frames; interchange at the
  verb's last token. D = after T − after I.
- **In-range** (A2 rule): coordinate set to t_T = the mean projection of
  the T verbs in the same frame and context, and to t_I = the mean over the
  kept I verbs other than the base verb (both excluding the basis's DAS
  training verbs). D_in = R(t_T) − R(t_I).
- Bootstrap: base verbs × contexts × donor pairs (donors only for T − I);
  2,000 draws, seed 17. Primary on half-B contexts; all contexts secondary.
- Fail-fast: stage 2 refuses to run if the natural gate did not pass, if
  the bases differ from stage 1's, or if fewer than 5 eligible verbs enter
  any in-range target; a self-set patch (coordinate set to the item's own
  value) must reproduce the natural readout.

**Bounds.** O: δ_s (the active-DAS bound, 0.42–0.57). ".", PREP and L:
δ_R = 0.2 × |the natural T − I gap of that readout in that frame|, used
only if that natural gap is ≥ 0.2 nats in magnitude (otherwise the
readout is unresolved there). Classification as before (RISE / FALL with
the 95% CI and the bound; NO RISE by the 90% CI within ±bound; otherwise
unresolved). The bounds condition on the natural gaps (not resampled).
Also reported: the patch × frame interaction D_what − D_that per readout.

**O against the natural T level (what frame).** Δ_T = O of the patched I
base (after a T donor, or at t_T) − the mean natural O of T verbs in the
same frame and context; the T verbs are resampled too (same context
draws). **Not above** if the 90% CI upper bound of Δ_T is < δ_s; **above**
if its 95% CI is above 0 and the estimate ≥ δ_s; otherwise unresolved.
This is a calibration reference: the patched state keeps the I verb's
other coordinates, so it can fall short of or exceed T for reasons other
than licensing.

**Reading, per site and patch type.**

| that frame | what frame | Reading |
|---|---|---|
| "." FALL and O RISE | L RISE, with D(".") 95% CI above 0 (closure rises, not only PREP falling) and D(PREP) FALL, and O **not above** the T level | **licensing-like crossover** |
| O RISE | O RISE and O **above** the T level | **surface** ("object next") |
| anything else | | mixed / unresolved, with CIs |

The crossover shows that the patched value controls continuations in a
filler-sensitive way. It is what licensing predicts, but a lexical-class
signal feeding a learned continuation policy ("transitive + what → close;
transitive + that → object") would produce the same pattern; the reading
does not separate the two.

**Sensitivities** (sites 6 and 8, both patch types): non-DAS verbs only;
multi-token verbs only (tokenization differs between pools: 52 of 106 T
and 10 of 50 I past forms are single tokens); each matrix verb.

**Prediction.**
- The natural gate passes; the I gate keeps most I verbs (≥ 70%).
- In-range, sites 6–8: **licensing-like crossover**.
- T − I donors, sites 6–8: the that-frame and what-frame L/PREP parts as
  licensing, but O may land above the T level in the what frame because the
  donors overshoot the natural range (A1/A2): licensing-like or mixed.
- Sites 14–17, both patch types: **surface**.

## B7. Matrix wh

**Items.** "AUX NAME V" vs "What AUX NAME V" with base forms, AUX ∈ {Did,
Will, Can, Would} (sentence-initial; "did" etc. after "What") × the same 6
names = 24 contexts. Same verbs.

**Readouts.** "?" ; PREP; O. **L = log P("?") − log P(PREP).**

**Base-form gate** (as the TC gate): "NAME can V" references for every B7
verb; at each site, T vs I verbs must separate along d_s (AUC over verb
means ≥ 0.8). Where they do not, B7 results at that site are inconclusive.

**Stages, halves, patches, bounds, reading:** as B6, with "?" in place of
".". The I gate and the natural gate are computed separately for B7. The
base-form gate checks that d separates the base forms; it does not show
that the intervention acts the same way in questions.

**Prediction.** As B6. The base-form gate passes at every site (as for TC).

## Order (Part B)

1. Commit this part with `verbs.csv`.
2. Stage 1 (natural pass, B6 and B7 together, GPU); gates analysed and
   committed.
3. Stage 2 only where the natural gate passes (GPU).
4. Report in `results.md`.

# Part C: nonce cue tests (declared 2026-10-09)

Written before any Part C run; revised after an outside review (Codex),
also before any run. Same 80 nonce lemmas, 4 slots, nouns,
agents and passive probe as round-3 C8 ("The N was A-ed", z on each site's
d_s as a fixed ruler; 0 = active intransitive, 1 = transitive level; the
probe is a token suffix of every prompt, checked). Declared sites: 6 and 8.
Per-lemma means over slots; lemma bootstrap (2,000 draws, seed 17).
Minimum effects as C8: 0.05 z, 0.1 nats. Measures at the probe's last
token: z (sites 4–17), log P(" by"), log P(O), log P(PREP without " by"),
log P(".").

## C8. Inflection mismatch

**Question.** In C8 a transitive context moved the later passive participle
toward the transitive side of d and raised " by" and objects; the object
rise was probably in-context copying ("dakked the" in the context). If the
context uses a different form of the verb, so that the probe's final token
never occurs in it, which effects survive?

**Contexts** (three sentences, the C8 lead schemas and agents; objects as
C8):
- **ed** (reference, the C8 matched contexts): "In the lab, the artist
  dakked the plate." / "… dakked."
- **ing**: "In the lab, the artist is dakking the plate." / "… is dakking."
- **s**: "In the lab, the artist daks the plate." / "… daks."
- balanced AB / BA in every form (four sentences, A and B alternating; in
  AB only A has objects; ed = the C8 balanced contexts);
- mismatched-lemma T / I in the ing and s forms (the C8 partner B in place
  of A), for the generic effect of a transitive context in that form.
Forms: -ing drops a final "e"; -s adds "s" (no lemma ends in a sibilant);
the past form is C8's. The ing and s forms also change aspect / tense and
add "is", so a smaller object rise there is consistent with, not proof of,
exact-token copying; shared stem tokens remain.

**Token check.** For ing and s, the probe's final token id must not occur
anywhere in the context. A lemma failing it is dropped from that form's
analysis (listed).

**Contrasts per form.** Δ_matched = T − I; Δ_balanced = AB − BA;
mismatched-lemma T − I. Paired differences ed − ing and ed − s, and the ed
reference itself, on each form's retained lemmas.

**Decision rules** (sites 6 and 8, per form).
- **d shift survives:** Δ_matched z 95% CI above 0 and estimate ≥ 0.05.
- **" by" survives:** Δ_matched log P(" by") 95% CI above 0 and ≥ 0.1.
- **The object rise mostly disappears:** Δ_matched log P(O) ≤ 0.5 × the ed
  value on the same lemmas, with the paired ed − form difference's 95% CI
  above 0.
- Verb-specificity in the mismatched forms: Δ_balanced z and " by" with the
  C8 rules (for " by": CI above 0, ≥ 0.1, and above Δ_balanced of PREP
  without " by").
- A form with no retained lemma is unresolved.

**Prediction** (not exact-token copying): the d shift and " by" survive in
both forms (at 50–100% of the ed values); the object rise mostly
disappears.

## C9. Cue 2 × 2: transitivity × adjacent NP

**Question.** Which cue in the context drives the shift: that the verb is
used transitively, or that a noun phrase follows it?

**Contexts** (three sentences each, lemma A, past form unless stated):

| | adjacent NP | no adjacent NP |
|---|---|---|
| **transitive** | "the artist dakked the plate." (= C8 matched T) | relative: "the plate that the artist dakked fell."; question: "what did the artist dak?" (base form) |
| **intransitive** | NP adjunct: "the artist dakked the whole night." | bare: "the artist dakked." (= C8 matched I) |

Structure-matched controls, so each cue is tested with the other held
fixed:
- **intransitive relative** "the plate near which the artist dakked
  fell." (same relative structure and second verb, PP gap);
- **intransitive question** "did the artist dak?";
- **duration PP** "the artist dakked throughout the whole night." (same
  duration content as the NP adjunct, but no adjacent NP).
Adjunct NPs: "the whole night / morning / afternoon / evening" (all start
with " the", like the objects). Relative-clause objects as C8; second
verbs "fell", "broke", "stayed there".

**Contrasts** (per lemma):
- **transitivity without an adjacent NP:** relative T − I (primary),
  question T − I (secondary);
- **adjacent NP without transitivity:** NP adjunct − duration PP
  (primary), NP adjunct − bare (secondary);
- descriptive: T+NP − bare, relative T − bare, and the 2 × 2 main effects.

**Decision rule** (sites 6 and 8 for z; " by" and O): a cue **drives** a
measure if its primary contrast has a 95% CI above 0 and an estimate ≥ the
minimum effect. Reading: transitivity-driven, NP-driven, both, or neither;
the same with the secondary contrasts.

**Prediction.** The d shift and " by" are transitivity-driven (T−NP ≈
T+NP > I+NP ≈ bare); the object rise is NP-driven (copying "dakked the").
Caveat declared: an unknown verb followed by "the whole night" may be read
as transitive; then the NP adjunct moves toward T+NP on every measure, and
an "NP-driven" result can reflect inferred transitivity rather than
adjacency. The duration-PP control limits but does not remove this.

## Order (Part C)

1. Commit this part.
2. Builder (no model), token checks and natural pass on TSUBAME (one job).
3. Report in `results.md`.

# Part D: translation stage, mid MLPs (declared 2026-10-09)

Written after Parts A–C ran and before any Part D run; revised after an
outside review (Codex) of the draft and its code, also before any run. "Translation
stage" = the mid-layer MLPs that turn the early verb-class value into
frame-specific output (round 3 B7/A3: MLPs 11–17 raise " by" in passives;
MLPs 15, 18, 21, 22 write objects in actives).

## D10. Clean voice contrast

**Frames.** "The N has been V" (passive perfect; base) vs "The N has V"
(active perfect; counterfactual); only "been" differs (an insertion, so
the two prompts differ in length by one token). Sensitivity: "The N was V"
(a second passive) in place of the counterfactual, S_sens = Δ_was −
Δ_has-been. Same items, rows, basis, donors, site-8 T-donor interchange
and quantities as B7 / A5 (fp32; path effects at the participle's last
position, which exist in every frame). Head-internal value / pattern
replacements need aligned positions and are not computed here; the fixed
B7 top 5 and the newly selected top 5 are replaced jointly as in A5.
Base-run attention to "has" and "been" is reported. The run is repeated
with the I donors of the same rows: Δ − Δ for T vs I donors sets the
coordinate to the same two values in both frames, so S_D = S_T − S_I
compares the frames' responses to the same coordinate change (S_T, the
B7-comparable quantity, also depends on where each frame's natural
coordinate starts).

**Decision rules.**
- **Gate** (as B7): Δ_has has the step-1 active sign for ≥ 7 of 9 MLPs.
- **Voice switch:** S = Δ_has − Δ_has-been (T donors) has B7's sign and
  ≥ 50% of B7's |S| for ≥ 7 of 9 MLPs, and S_D has the same sign for those
  MLPs. (The final-LN reference scale differs between this run and B7.)
- **Not an auxiliary-token effect:** |S_sens| < 0.5 |S| for ≥ 7 of 9
  MLPs (passive vs passive moves the MLPs little).
- **Two-stage routing replicates** if, among eligible MLPs (|S| ≥ 0.05),
  H1 (joint heads ≥ 0.5 |S|) holds for ≥ 3 of the 5 " by" MLPs and the
  earlier-MLP alternative (joint MLPs ≥ 0.5 |S|) for ≥ 3 of the 4 object
  MLPs.
- **B7 heads retain their role:** the A5 criteria (i)–(iii).

**Prediction.** All five hold: the switch is a voice switch, routed as in
B7 (heads into the " by" MLPs, earlier MLPs into the object MLPs), with
the B7 heads.

## D11. Conjunction neurons in MLPs 11 and 14

**Question.** Do single neurons in MLPs 11 and 14 respond to the verb-class
value only in the passive frame ("transitive AND passive")?

**Run.** The B7 items and rows (32 contexts per pair, split-0 basis), now
with both T and I donors, in the was and has frames; site-8 interchange.
For every neuron of MLPs 11 and 14 (2 × 8,192), the post-activation at the
participle's last token; direct effects on " by" = activation × the
neuron's output weight on the centered " by" readout, through one reference
final-LN scale per pair (the mean scale of its was-frame patched runs).

**Per neuron.** Interaction I_n = (a[was, T] − a[was, I]) − (a[has, T] −
a[has, I]), per pair (mean over its rows and contexts). The 64 pairs are
split once into halves within band (seed 17): **half A selects, half B
tests**, so the test is on pairs not used for selection.

**Categories** (half A, sign-aligned to I_n): **passive-conjunction** if
the was effect (T − I) is in the direction of I_n and |has effect| < 0.5 ×
|was effect|; active-conjunction (the reverse); graded (same sign in both
frames); opposite signs.

**Decision rules.**
- **Selection (half A):** one-sample t-test of I_n across pairs; BH-FDR
  q < 0.05 over the 16,384 neurons.
- **Conjunction neurons exist** if ≥ 10 selected neurons are
  passive-conjunction and, on half B, ≥ 80% of them keep the sign of I_n,
  their sign-aligned mean I_n and was effect have 95% CIs (pair bootstrap)
  above 0, and |mean has effect| < 0.5 × the was effect.
- **They carry the " by" switch** if, on half B, their share of the MLPs'
  switch (the direct effect of I_n on " by", MLPs 11 and 14 together) is ≥
  25%, CI above 0. Their share of the passive T − I effect is reported too.
- **Generalization** (descriptive): the passive-conjunction set's
  sign-aligned activation difference (i) natural good − bad passives by
  frequency band (Head, Tail, XTail; all primary items, no patching), with
  the XTail / Head ratio; (ii) the round-3 C8 nonce probes, matched T − I
  and balanced AB − BA (lemma bootstrap).

**Prediction.** Conjunction neurons exist (tens of them, mostly in MLP14);
they carry ≥ 25% of the " by" switch; they respond to natural good − bad
passives in every band, weaker for XTail (ratio 0.5–0.8), and to nonce
probes in the matched contrast.

## D12. Translation gain per pair

**Measures** (natural pass, no patching, fp32; all primary passive-test
items):
- **TG** = the direct effect of MLPs 11–17 on " by" (centered readout
  through the final-LN scale), good − bad, mean over the pair's contexts;
  also per MLP.
- **gain** = TG / (the pair's site-8 z gap), for pairs whose z gap is ≥
  0.1; and **TG_res** = TG − β · z gap (β from OLS across pairs;
  descriptive, reliability conditional on β).
- Participle Zipf (pair mean, as `analyze_frequency.py`).

**Analyses** (as A1): split-half reliability of TG, gain and TG_res
(contexts, stratified by band; 1,000 splits); slopes on participle Zipf
(pair bootstrap); correlations with the three behaviours (single-prompt
" by" preference, released *by* margin, curated LP margin), with the A1
disattenuation and verdict rule.

TG is natural good − bad, so it reflects every way the two verbs differ,
not only d; D12 is observational and does not locate a weakness causally.

**Decision rules.**
- **TG tracks frequency** if its Zipf slope has a 95% CI above 0.
- **Frequency dependence beyond the d gap** if the Zipf coefficient in TG
  ~ z gap + Zipf (refit in every pair-bootstrap draw) has a 95% CI above 0.
- **TG predicts behaviour beyond the d gap** if, for a behaviour, both the
  paired difference r(TG) − r(z gap) and the partial correlation r(TG,
  behaviour | z gap) have 95% CIs above 0. The single-prompt " by"
  preference contains TG as a component (part–whole), so only the released
  *by* margin and the LP margin count for this rule.
- Reading: "rare verbs translate less per unit of d gap" if the second
  rule holds; otherwise "no evidence of a frequency dependence of
  translation beyond the d gap" (not evidence that there is none).

**Prediction.** TG tracks frequency (because the d gap does); TG_res shows
no frequency slope (CI includes 0); TG correlates with the single-prompt
" by" preference (r ≈ 0.3–0.5) but weakly with the LP margin.

## Order (Part D)

1. Commit this part with Part E.
2. D10, D11, D12 on TSUBAME (one job).
3. Report in `results.md`.

# Part E: restoration (declared 2026-10-09)

Revised after the same outside review, before any run.

**What a patch at d can reach.** On the primary pairs, most of the
head − XTail whole-sentence LP deficit sits in the participle's own
tokens (curated `passive_2` verb margin 3.29 vs 1.39), which are predicted
before the participle's last token and so cannot change under a patch
there. What a patch at d can change is the continuation: on curated
`passive_1` ("The N was V by the X."), the " by" margin is 0.95 (Head) vs
0.42 (XTail). The **by margin** is therefore the primary target; the
whole-sentence and suffix margins are reported, with the deficit-share of
the whole margin as a secondary quantity.

## E13. Oracle at d

**Items.** Curated `passive_1` and `passive_2` sentences of the primary
pairs (the band-cross contexts, as `pythia14b_scores.csv`).

**Patch.** At sites 6 and 8 (one at a time), on the participle's last
token, with the item's cross-fitted fold basis for each split: the
coordinate of an XTail pair's **good** sentence is set to the mean raw
projection, under that same basis, of the **Head-band good** participles in
the same context, and of its **bad** sentence to the Head-band bad mean
(true class, passive range). The Head-band pool is all 26 curated Head
pairs, excluding the item's own pair and any pair that is a DAS training
pair of the basis (≥ 2 pairs required; the minimum is reported). The rest of the sentence is scored as
`score_matched_passives.py` (whole, verb, suffix and " by" log-probs).
- **Specificity control:** the same patch on Head pairs (each verb set to
  the Head mean of its class, excluding its own pair).
- **Reverse control** (secondary): Head pairs set to the XTail means.

**Quantities.** Δ by margin, Δ suffix margin, Δ whole margin (= Δ suffix,
checked), per band; **share closed** = Δ(XTail by margin) / (Head by
margin − XTail by margin), natural values from the same run; the same for
the whole margin. Bootstrap: pairs within band × contexts, the context
draws shared by all bands; inference conditional on the targets.

**Decision rule.** E13 **closes a real share** at site 6 or 8 if the
by-margin deficit has a 95% CI above 0, the share is ≥ 0.25 with a 95% CI
above 0, and the Head control's Δ by margin has a 95% CI within ±0.1 nats.
The Head control is weak specificity evidence (with leave-one-out class
means its mean change is near zero under any common linear response); the
reverse control is reported alongside.

**Prediction** (from the site-8 band levels and the dose-curve slopes):
the by-margin share is ≈ 0.2–0.3; the whole-margin share ≤ 0.1; the Head
control moves < 0.1; the reverse control lowers the Head by margin.

## E14. Label-free scaling (only if E13 closes a real share)

Outline only. If E13's rule is met, each verb's own coordinate is scaled
about a label-free centre (z' = c + α (z − c), α ∈ {1.5, 2, 3}, capped to
the active range), and the XTail by-margin deficit closed, the effect on
Head verbs, the LP accuracy of other FreqBLiMP paradigms and held-out LM
loss are measured. The centre c and the patch positions outside passive
contexts (other paradigms, generic text) are not yet defined; they are
fixed in a separate commit before any E14 run. If E13 closes little, E13 is
repeated at the translation stage (MLPs 11–17 outputs set to the Head mean
of the true class), using D's results, also declared first.

## Order (Part E)

1. Commit with Part D.
2. E13 on TSUBAME; E14 only if E13's rule is met.
3. Report in `results.md`.

## E13-T. Oracle at the translation stage (declared 2026-10-09, after E13)

Written after E13 and D10–D12 ran (`results.md`) and before any E13-T run;
revised after an outside review (Codex) of the draft and its code, also
before any run. E13 did not close a real share, so per E14's outline E13 is
repeated at the translation stage, using D's results.

**What is already known.** On all 126 curated band-cross pairs, the
natural Head − XTail " by" margin deficit is +0.45 [−0.33, +1.27]
(`e13_deficit.md`). Natural scores are the same under every intervention,
so E13's "closes a real share" rule (which needs that deficit's CI above 0)
cannot be met here. It is applied for continuity and cannot trigger E14.
The operative rule below is about raising rare verbs' " by" margin.

**Items.** Curated `passive_1` and `passive_2` sentences of all 126
band-cross pairs (26 Head, 50 Tail, 50 XTail). No DAS basis is involved, so
nothing restricts the pairs. The primary pairs (as E13) are reported as a
subset for comparison with E13. fp32.

**Interventions** (at the participle's last token, one at a time).
Targets are the mean over the Head-band pairs of the same paradigm,
context and true class (good → Head good mean, bad → Head bad mean), own
pair excluded (25 pairs; the minimum is reported). The means come from the
natural run.
- **`mlp`:** the output vectors of MLPs 11–17, all seven set jointly to
  their Head means. Each later MLP gets the natural Head-mean value, not its
  response to the earlier replacements.
- **`neurons`:** the post-activations of D11's top 50 switch neurons are
  set to their Head means; everything else is unchanged. These are ranks
  1–50 of `conjunction_switch_neurons.csv`: 16 in MLP11, 34 in MLP14,
  carrying 0.86 of the switch on half B. They were selected on the
  passive-test pairs' patched interaction, not on curated scores.
- **`random`** (control for `neurons`): five sets of 50 other neurons with
  the same layer split (16 in MLP11, 34 in MLP14), drawn uniformly with
  seed 17 from neurons outside the top 50, each set to their Head means.
  `random` is the mean over the five sets (per bootstrap draw).
- **`wmatched`** (secondary control): 50 neurons that D11 did not select,
  each matched one-to-one (same layer, in rank order) to a top-50 neuron on
  |w_by|, the " by" output weight. It tests whether a switch neuron does
  more than a non-switch neuron that writes as strongly onto " by". The
  strongest switch neurons have no equally strong unselected partner (mean
  |w_by| 0.18 vs 0.23). The sets are in
  `data/round4/oracle_translation/neurons.csv` (`build_oracle_neurons.py`).

**Checks** (in the run, fail-fast): own natural values as targets
reproduce the natural scores. With distinct targets, the patched entries
equal the targets, nothing else in the hooked tensors changes, and the
scores change. After each intervention, the verb's log-prob and whole −
suffix are unchanged.

Groups as E13: XTail and Tail items at the Head target; Head items at the
Head target (specificity control; with leave-one-out means its mean change
is near zero under a common linear response); Head items at the XTail mean
(reverse control).

**Quantities.** As E13: Δ by margin (the " by" token of `passive_1`),
Δ suffix and Δ whole margins, the share of the natural Head − XTail by
deficit closed (unstable where the deficit's draws cross 0), and the
patched Head − XTail gap. Bootstrap: pairs within band × contexts, with
the same draws for every intervention, so differences between
interventions are paired. Subsets: all 126 pairs (decisions), the primary
pairs (as E13), and the 67 pairs outside D11 (not primary passive-test
pairs; D11 selected and explored on the primary pairs, so these are the
independent ones).

**Decision rules** (per intervention, all 126 pairs).
- **Closes a real share:** E13's rule, unchanged (expected not met; see
  above).
- **Raises rare verbs' by margin** if:
  - XTail Δ by margin has a 95% CI above 0 and is ≥ 0.25 × the natural
    Head − XTail by gap (point estimate);
  - the Head control's Δ by margin has a 95% CI within ±0.1 nats;
  - for `neurons` only: XTail Δ by exceeds `random`'s (paired difference
    with the five-set mean, CI above 0). The difference from `wmatched` is
    reported.
- **Comparison with d** (descriptive): on the primary pairs, XTail Δ by
  under `mlp` and `neurons` minus E13's site-8 Δ by (mean over splits),
  paired by sentence (the E13 scores are required).
- **Scope.** A positive `mlp` result shows that this broad, label-conditioned
  replacement helps. A negative one does not rule out a weakness at the
  translation stage, since averaging seven MLP outputs may remove useful
  content. `neurons` > `random` supports this selected set over random
  sets, not conjunction specificity in general.

**Predictions.**
- **`mlp`:** by D12, MLPs 11–17 write about the same good − bad " by" for
  XTail as for Head (TG +1.03 vs +1.04), so the full-output oracle moves
  XTail little: Δ by in [−0.10, +0.15]. Replacing seven MLP outputs with a
  mean also removes verb-specific content, so the Head control moves by
  more than 0.1. Rule: **not met**.
- **`neurons`:** by D11 (exploratory), natural XTail pairs drive the top
  50 less (+0.85 vs Head +1.95 in w_by units, about 0.3 logits after the
  final-LN scale of 3.2). So XTail Δ by is +0.10 to +0.35, CI above 0, and
  above `random`; the Head control stays within ±0.1. Rule: **met**. Above
  `wmatched` too, but by less than above `random`.
- **`random`:** |XTail Δ by| < 0.05.

**Order (E13-T).** Commit; one TSUBAME job (`run_oracle_translation.py`);
analysis (`analyze_oracle_translation.py`); report in `results.md`.
