# DAS round 2: classes, new-frame readout, filter, matching, training plan

The original 126-pair set and the existing result files are unchanged.

- Full tables:
  - `round2_analysis.md` (primary, strict classes);
  - `round2_analysis_named_only.md` (sensitivity set).
  Both are written by `scripts/analyze_das_round2.py`.
- Data: `data/das_round2/`.
- Readout: TSUBAME job 8900510 (commit `e6d1caa`, Pythia-1.4B bf16),
  `results/das_round2/pythia14b_readout.csv` (SHA-256 `884b48ab…b6f7`, not
  committed). It covers 658 verbs (all expansion candidates and all 252
  original verbs) in two frames:
  - "<Subj> has/have <participle>", with 7 subjects;
  - "The AGENT <past>", with 90 agents.

## Decisions (2026-10-05)

- **Class rule.** A verb whose PP is an argument (a reciprocal *with*-partner
  or a lexically selected P) is `prep_object`. A verb with locative-only PPs
  is `plain`. Applied to all 225 intransitives (below).
- **prep_object subtypes:** `pseudo_passive_ok` (rely on, tamper with,
  intervene in) vs `pseudo_passive_bad` (belong to, result in, succumb to).
  Both are excluded from training and both are kept for the projection
  analysis.
- **contaminated_bad:** jut, scram, crackle, resound, pee, hibernate, reel.
  All evals are reported with and without them.
- **excel** is dropped: it puts reflexive mass after the verb ("excelled
  herself", up to 0.19).
- The 6 new Head eval pairs, whose bad verbs are all prep_object, form a
  **separate eval group**.
- **Primary DAS run:** the strict set, 5-fold CV over pairs, repeated with 3
  fold splits, CIs over verb pairs. It has 28 pairs by the readout filter
  and **29 in training**: under the final target (which adds " you", `."`,
  `,"`), *ensue* passes 4 of 7 items, at the boundary.
- **Secondary run:** the named-only set, 38 pairs by the readout filter and
  39 in training (only chat, quarrel, bicker and compete moved, excel
  dropped). It uses the same layer, rank and epochs, all chosen
  on actives in the primary run before any passive is looked at. Report the
  cosine between the two runs' directions, and whether transfer agrees.
- **Object-start set** adds " you", " herself", " himself", " themselves".
- **Equivalence bound, fixed now:** in the passive test, a rise in
  object-start mass smaller than **20% of the patch's effect on held-out
  actives** counts as "no rise". Reported as a TOST alongside the point
  estimate.

## Classes

Files in `data/das_round2/`:
- `intrans_classes_final.csv`, written by `scripts/build_intrans_classes.py`;
- `class_disagreements.csv`, for review.

**Sources**

1. **FreqBLiMP verb inventory**
   (`freq-blimp/generation_projects/blimp/verb_inventory.json`, frame
   labels `intr`, `trans`, `intr_pp`:P, …). It is the primary source where
   it is informative.
   - A verb with `intr_pp` frames but no bare `intr` frame has an obligatory
     preposition, so it is `prep_object`.
   - Beyond that, the inventory cannot implement the rule. `intr_pp` means
     "a natural verb+P combination where an NP follows", adjuncts included:
     *die* in/of/from/for/with…, *exist* for, *vanish* with.
   - A preposition-type test on the inventory (argument-type P listed →
     prep_object) would mark 55 of the plain verbs as prep_object, including
     die, exist, occur, remain and vanish. That would leave almost no
     training intransitives.
2. **The rule-based classification** decides wherever the inventory lists a
   bare `intr` frame, and for the 36 verbs not in the inventory. It is my
   hand classification, cross-checked with Codex blind, plus the rule moves.

**Rule moves beyond your four** (reciprocal *with*, or a selected typical P):
- coincide, converse, differ, feud, intervene, succumb, triumph, excel;
- elope, hobnob, intercede, luxuriate, cohabit, fraternize;
- original verbs: abstain, banter, bask, belong, clash, coexist, collude,
  cooperate, correspond, desist, diverge, eventuate, exult, result, teem,
  wallow.

**Inventory overrides** (obligatory P, which I had as plain): hesitate,
abound, dally.

| Source | plain | prep_object (ok / bad) | contaminated_bad |
|---|---:|---:|---:|
| original | 59 | 60 (45 / 15) | 7 |
| new | 43 | 56 (43 / 13) | 0 |

**For your review** (`class_disagreements.csv`):
- the 3 inventory overrides;
- 52 verbs the preposition-type heuristic would call prep_object but the
  rule leaves plain (mostly adjunct PPs: die for, exist for, kneel to);
- 27 verbs the rule calls prep_object but whose listed Ps look spatial
  (result in, belong in/on, bask in).

**contaminated_bad vs the bad-side-high *by* group: no overlap.**
- prise/jut and daub/resound are negative but "good side low";
- espy/scram and placate/crackle are reliably positive;
- pee, hibernate and reel are not in the negative-*by* group.

In the active frame only *jut* looks transitive ("She has jutted" → M =
+1.8).

## Is there enough data?

**Training: 28 pairs (strict), 38 (sensitivity).**
- Strict: 20 expansion pairs (near-Head intransitives) and 8 original Head
  pairs, all with both verbs passing the behaviour filter. That is 56 verbs
  × 7 subjects = 392 items.
- With 5-fold CV, about 22 pairs train each fold and about 6 are held out.
  Repeating with 3 split seeds stabilises the held-out estimates. CIs come
  from bootstrapping verb pairs.

**Passive evaluation: Tail/XTail are fine; Head is the bottleneck.**

| Bad-verb class | Head | Tail | XTail |
|---|---:|---:|---:|
| plain, original pairs | 9 (8 without bet/appear; 7 with a reliable *by* margin) | 25 | 25 |
| plain, new eval pairs | 0 | 3 | 2 |
| prep_object, original + new (separate group) | 17 + 6 | 23 + 5 | 20 + 3 |
| contaminated_bad | 0 | 2 | 5 |

- Head passive claims rest on 8 pairs, and those verbs are also the
  orig_head training pairs. The Head passive test is therefore cross-fitted:
  each verb is tested only with directions trained without it.
- The strict rule moved belong, compete and result out of the Head plain
  set.

## 1. Readout on the new frame (strict pool)

Target (option 3): M = log P(O) − log P(I) at the verb's last token.
- **O**: determiners, possessives, accusative pronouns and reflexives.
- **I**: punctuation, " and", prepositions, adverbs.
- " that", " by" and particles are in neither set.

" you" was not tracked in this readout. It is added in the DAS run, which
has full logits.

| Verbs | Class | "She has" | 7 subjects | Old frame (90 agents) |
|---|---|---:|---:|---:|
| expansion, Head transitives (3 multi, 1 single) | trans | 100% | 100% | 85%, 96% |
| near_head, multi | trans / intrans | 100% / 100% | 100% / 100% | 95% / 100% |
| near_head, single | trans / intrans | 80% / 83% | 91% / 90% | 83% / 99% |
| orig_head, single | trans / intrans | 100% / 100% | 96% / 100% | 94% / 100% |

- Pair-level ordering (same subject) is ≈ 100% in the new frame.
- The natural class gap in M is about 5 nats; this is the denominator for
  the positive control.
- In the old frame, Head multi-token transitives were the weak cell,
  consistent with the reduced-relative reading. The new frame fixes this.

**Reflexive check.** Plain intransitives put almost no mass on reflexives
after the verb (median 0.0008, 90th percentile 0.006).
- The outliers are *lumber* (0.20, "lumbered himself with"), which is not a
  training verb, and *excel* (0.19), which is dropped.
- Every other training intransitive is at 0.01 or below.

## 2. Behaviour filter

- An item (verb, subject) passes if M is on its class's side of 0. A verb
  passes if at least 4 of its 7 items pass.
- Pass rates are 100% except in near_head single-token (about 90%) and
  orig_head transitives (96%).
- The one reject is still *ensue* ("She has ensued" → " a", " the"). It goes
  into the projection list.

Projection groups (`projection_groups.csv`), "She has" M:

| Group | Verbs | Mean M | M > 0 |
|---|---:|---:|---:|
| included transitives | 220 | +1.80 | 98% |
| plain intransitives | 102 | −2.59 | 3% |
| prep_object | 116 | −3.69 | 1% |
| contaminated_bad | 7 | −0.65 | 1 of 7 |
| bad-side-high *by* | 11 | −2.54 | 0% |

prep_object verbs look intransitive here by construction: their selected
preposition is in I.

## 3. Familiarity (participle Zipf)

- Strict expansion pairs have participle Zipf 3.35 (transitive) vs 2.86
  (intransitive). Only 7 of 21 pairs are within 0.35.
- Rematching with both calipers gives **15 pairs** (6 single-token, 9
  multi-token), at 3.04 vs 2.99. This is the robustness set:
  `train_pairs_participle_matched.csv`.
- The orig_head pairs are already participle-matched.

## 4. Token count within near_head (strict)

| | Transitive | Intransitive | Pairs (both verbs pass) |
|---|---:|---:|---:|
| single-token | 6 | 6 | 5 |
| multi-token | 15 | 15 | 15 |

That is too few single-token pairs for a held-out comparison. Token count
is reported only as an exploratory per-pair covariate.

## 5. DAS training plan

**Data**
- Primary: the 28 strict pairs.
- Secondary: the 38-pair set (`data/das_round2/sensitivity_named/`).
- Frames: "<Subj> has/have <participle>", training on 6 subjects. *David*
  is held out for a template check.

**Intervention**
- Residual stream (output of layer L) at the verb's last subtoken, which is
  also the readout position.
- A learned orthonormal rank-k subspace; interchange: h ← h + Rᵀ(R h_src − R h_base).

**Objective**
- BCE on σ(M), with the source verb's class as the label.
- O = the, a, an, his, her, their, its, my, our, your, this, these, those,
  some, every, each, several, him, them, it, me, us, you, himself, herself,
  themselves, itself.
- I as in section 1, plus `."` and `,"`.

**Swaps**
- Base and source share the subject frame.
- Cross-class in both directions, plus same-class swaps whose label is
  unchanged.
- Donors come from any pair in the training fold.

**CV**
- 5 folds over pairs, stratified by source (expansion / orig_head) and
  token count.
- Repeated with 3 split seeds, one model per fold and split.

**Selection on actives only (primary run), then freeze**
- Layer: sweep the residual layers, rank 1, choosing by held-out IIA.
- Rank: 1, 2 or 4; take rank 1 unless a larger rank raises the held-out
  positive-control fraction by more than 0.1.
- Epochs: fixed from the held-out curve in the sweep.
- The frozen configuration is written and committed before any passive is
  evaluated.

**Held-out active metrics**
- IIA.
- **Positive control:** ΔM (intransitive base ← transitive source) as a
  fraction of that pair's natural class gap, with the split into
  determiner, pronoun and reflexive mass.
- Same-class swaps.

**Active-side controls**
- 100 random rank-matched subspaces, raw and with displacement norm matched
  to d.
- Shuffled-label DAS: class labels permuted across verbs within the fold,
  same budget.

**Passive transfer (after freezing; next step)**
- Cross-fitted bad passive bases (plain bad verbs, other contexts).
- Primary contrast: D = E[ΔR | transitive donor] − E[ΔR | intransitive
  donor].
- Readouts: log P(" by"), object-start mass split into determiners and
  pronouns, log P(".").
- TOST on object-start mass, with the bound at 20% of the active-side
  effect.
- Controls: good → good passive, and the voice-change baseline
  (intransitive active → bad passive).
- Separate groups:
  - the 6 Head prep_object eval pairs;
  - prep_object bad verbs, by subtype;
  - contaminated_bad (with and without);
  - bad-side-high;
  - reliable vs negative *by*;
  - the familiarity robustness set.

**Secondary run**
- Same configuration, on the 38-pair set.
- Report the cosine between primary and secondary directions (per fold and
  split, and for the mean direction).
- Report agreement of held-out active metrics, and later of passive
  transfer.

**Projection (later)**
Project onto d:
- prep_object (both subtypes);
- *ensue*;
- contaminated_bad;
- bad-side-high;
- the Codex transitive-homonym candidates.

## Files

All in `data/das_round2/`:

| File | Contents |
|---|---|
| `intrans_classes_final.csv` | final classes (`intrans_classes.csv` is the pre-rule hand + Codex version) |
| `class_disagreements.csv` | rule vs inventory disagreements, for review |
| `train_pairs.csv` | strict training pairs, before the filter |
| `sensitivity_named/train_pairs.csv` | 38-pair sensitivity set |
| `train_pairs_participle_matched.csv` | 15-pair familiarity robustness set |
| `behavior_filter_items.csv`, `behavior_filter_verbs.csv` | filter results, rejects kept |
| `projection_groups.csv` | projection groups |

## Training results (2026-10-05; actives only, no passive evaluated)

Full report: `das_round2_results.md` (`scripts/analyze_das_round2_results.py`).

**Runs**
- Sweep: job 8901679.
- Final: job 8901812, commit `f0ba8b8`.
- Outputs in `results/das_round2/`; `heldout_swaps.csv.gz` is not committed.

**Frozen configuration** (`results/das_round2/frozen_config_final.json`)
- **Site 17** (output of layer 16 of 24).
  - Held-out IIA rises smoothly with depth: 0.29 at site 3, 0.80 at site 8,
    0.92 at site 12, and 0.99 from site 17 on.
  - The earliest-site rule picks 17.
  - Full curve: `results/das_round2/sweep_by_site.csv`.
- **1 epoch.**
- **Rank 1.** Mean held-out positive-control fraction is 1.07 / 1.09 / 1.10
  for ranks 1 / 2 / 4, so the rank rule keeps rank 1.

**Held-out actives, primary run** (29 pairs; held-out pairs, 3 splits; CI
over pairs)

| | Intransitive base ← transitive source | Transitive base ← intransitive source |
|---|---|---|
| IIA | 0.97 [0.94, 0.99] | 0.98 [0.96, 0.99] |
| Fraction of the natural gap in M | 1.06 [1.01, 1.11] | 0.93 [0.87, 0.99] |
| Δ log P(O) | +2.86 [2.52, 3.19] | −2.18 [−2.39, −1.96] |
| Δ log P(determiners / pronouns / reflexives) | +2.66 / +4.30 / +3.66 | −2.14 / −2.94 / −2.63 |
| Δ log P(I) | −2.24 [−2.47, −2.01] | +2.16 [1.96, 2.39] |

- **Positive control: passed.** Patching held-out intransitive actives
  closes the full natural class gap (≈1.06×). The rise is largest for
  pronoun objects.
- **Same-class swaps leave M on its side:** 0.99.
- **The held-out subject (David)** reaches IIA 0.96.
- **Controls (per fold, 15 fold × split runs):**
  - Random rank-1 subspaces: IIA 0.02, gap fraction 0.00. Norm-matched:
    IIA 0.02, gap fraction 0.01. DAS beats the 95th percentile of
    norm-matched random subspaces in 15 of 15 runs.
  - Shuffled-label DAS: IIA 0.34, gap fraction 0.42 (range 0.05–0.66).
    Permuting class labels across verbs leaves about half the verbs
    correctly labelled, so a weaker version of the same direction is still
    found. The true-label direction is far stronger.
- **Exploratory, by source and token count:** gap fractions are
  - expansion multi-token (15 pairs): 1.09;
  - expansion single-token (6 pairs): 1.02;
  - orig_head (8 pairs): 1.04.

  The CIs overlap; the single-token cells are small.

**Sensitivity run (39 pairs), same configuration**
- Held-out IIA 0.97 / 0.98 and gap fraction 1.08 / 0.91, essentially the
  same as the primary run.
- **Directions agree:**
  - cosine of the mean primary and sensitivity directions: 0.98;
  - all primary × sensitivity fold bases: median 0.92 (min 0.89);
  - within-run stability: median 0.92 (primary) and 0.95 (sensitivity).
- **Held-out transfer agrees** on the 26 shared intransitive-base pairs:
  mean gap fraction 1.07 vs 1.11, Pearson r = 0.77.

**Passive test settings, fixed now**
- TOST bound δ = 0.2 × 2.86 = **0.57 nats** on the passive contrast D in
  Δ log P(O).
- "No rise" if the 90% CI of D lies within ±0.57. The point estimate is
  reported alongside.

## Passive test: declaration (2026-10-06, before any passive evaluation)

Full spec: `passive_test_plan.md`.

- **Site 17 stays primary.** Sites 8 and 12 are added as declared
  secondary analyses: rank 1, the same recipe, epochs by the frozen epoch
  rule (3 and 2).
  - Configs: `results/das_round2/frozen_config_site{8,12}.json`.
- **Step 1:** project natural good and bad passive participles onto d (no
  patching).
- **Step 2:** passive transfer at all three sites:
  - contrast D;
  - TOST on object-start mass, with δ_s = 20% of each site's active effect
    (0.57 at site 17);
  - log P(" by") and log P(".");
  - good→good, voice-change and passive-swap controls;
  - the separate groups listed above.

## Passive test: results (2026-10-06)

Full tables: `passive_test_results.md` (written by
`scripts/analyze_passive_test.py`). Declared rules from
`passive_test_plan.md`; primary population of 64 plain-bad-verb pairs.

| Site | Role | D log P(O) | D log P(" by") | Outcome |
|---:|---|---|---|---|
| 17 | primary | +3.58 [3.36, 3.79], RISE | −0.93 [−1.10, −0.77], FALL | **surface** |
| 12 | secondary | +1.17 [1.01, 1.33], RISE | +0.81 [0.72, 0.89], RISE | **mixed** |
| 8 | secondary | +0.36 [0.30, 0.44], NO RISE (δ 0.49) | +0.72 [0.63, 0.82], RISE | **abstract** |

This is the depth pattern the plan called the most interesting one:
"takes an object" early, "object next" near the output. Site 17 carries
the primary claim, and there d is surface.

**Site 17 (primary).**
- A transitive value turns "The house was emerged" into an
  object-expecting context: P(O) goes from 0.016 to 0.47, and P(" by")
  from 0.069 to 0.034.
- The verb's own active value does the same to its good passive (+3.2 nats
  O). So at layer 16, d is a voice-specific "object next" value.

**Site 8 (secondary).**
- The transitive value moves P(" by") on bad passives from 0.065 to 0.117.
  The good passives sit at 0.158, so this is 1.01× the natural log gap.
- P(O) moves only from 0.016 to 0.023, and log P(".") does not move.

**Checks that hold at every site.**
- The voice-change baselines are about 0 on bad bases (intransitive donor,
  and the verb's own active). D comes from the transitive value, not from
  inserting an active state.
- The result holds across bands, original-only pairs, without bet/appear,
  for the cross-fitted orig_head pairs, and with participle-matched donors.

**Step 1.**
- Good and bad passives separate along d at all three sites (paired win
  rate 0.94-1.00), but compressed: 0.36 / 0.34 / 0.18 of the same verbs'
  active gap at sites 8 / 12 / 17.
- At site 17 good passives sit near the intransitive level (z = 0.20).
- The passive good → bad swap raises " by" at sites 8 and 12 (+0.36,
  +0.34) but not at 17 (+0.05).

**Caveats on the site-8 "abstract" call.**
- The O class rests on TOST. The rise is reliably above 0 (90% CI
  [0.31, 0.42]).
- Pronoun, reflexive and " him" mass still rise by 0.9-1.4 nats. In
  probability that is small: pronouns go from 0.0004 to 0.0012.
- Site 8 controls actives less well (IIA 0.76, 0.75 of the gap).
- prep_object bad verbs are less selective at site 8 (pseudo_passive_ok:
  O +1.01, "." +0.72).
- No random-direction or shuffled-label control was run on passives. That
  is the obvious next check for the site-8 " by" effect.
