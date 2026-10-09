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

## Order (Part A)

1. Commit this part.
2. A1 and A3 locally on committed outputs.
3. A2, A4, A5 on TSUBAME (one job each).
4. Report in `results.md`.
