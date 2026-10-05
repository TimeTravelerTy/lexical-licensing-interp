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
- **Primary DAS run:** the strict set (28 pairs), 5-fold CV over pairs,
  repeated with 3 fold splits, CIs over verb pairs.
- **Secondary run:** the 38-pair set (only chat, quarrel, bicker and compete
  moved, excel dropped). It uses the same layer, rank and epochs, all chosen
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
