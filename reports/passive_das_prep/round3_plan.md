# Round 3: declared plan (2026-10-08)

Builds on `followup_plan.md` / `followup_results.md` (depth, site-8 mechanism,
frequency). Each part is written and committed before any of its runs.
Predictions are stated before the data exist. Notation as before:
- site s = residual stream output of layer s − 1 (0 = embeddings);
- d_s = the rank-1 DAS direction at site s (15 fold × split bases);
- z = the sign-aligned coordinate on d_s, 0 = held-out active intransitive
  level, 1 = held-out active transitive level;
- primary population = the 64 plain-bad-verb pairs; cross-fitting as in
  `passive_test_plan.md` (DAS verbs only use bases where their pair is held
  out);
- CIs are 95%, 2,000 draws, seed 17, unless stated otherwise.

# Part A: cheap checks (declared 2026-10-08)

## A1. Split-half reliability of per-pair measures

**Question.** The per-pair passive gap along d tracks participle Zipf but
not Pythia's behavioural passive margins (`frequency_results.md`). Is that
null "no link" or "couldn't measure"?

**Measures** (one value per pair = mean over that pair's contexts):

| Measure | Units averaged | Per pair | Source |
|---|---|---:|---|
| passive gap at site s (s = 4, 6, 8, 10, 12, 14, 16, 17) | curated passive contexts, z(good) − z(bad) | ~125 | `projections_site{s}.parquet` |
| active gap at site s (secondary) | subjects | 7 | same |
| single-prompt " by" preference | curated passive contexts, log P(" by" \| good) − log P(" by" \| bad) | ~125 | `natural.parquet` |
| released *by* margin | released `passive_1` contexts | 300 | `passive_band_cross/pythia14b_scores.csv` |
| curated `passive_2` LP margin | curated contexts, `lp_whole_margin_p2` | ~124 | `passive_das/passives.jsonl` |

The released and LP margins exist for the 59 original pairs; the others for
all 64.

**Procedure.**
- Split each pair's contexts at random into two halves, stratified by
  context band (subjects for the active gap: 3 vs 4).
- Compute the per-pair mean in each half; Pearson r across pairs between
  halves; Spearman-Brown R = 2r / (1 + r).
- 1,000 random splits (seed 17): report the median R and its 2.5–97.5%
  range over splits.
- Uncertainty over pairs: a pair bootstrap (2,000 draws), each draw with a
  fresh random split; the 95% CI of R. Draws with R ≤ 0 are undefined for
  disattenuation; their frequency is reported.
- Spearman-based R reported as secondary.
- Disattenuated correlation for each of the 24 site × behaviour tests:
  r_obs / sqrt(R_gap · R_beh), with both R computed on that test's exact
  pair population (59 or 64), and the pair bootstrap above (each draw
  recomputes r_obs and both reliabilities). Pearson is primary here because
  Spearman-Brown is defined for it; the Spearman versions are reported too.
- Ceiling on any observable correlation: sqrt(R_gap · R_beh).

**What this does and does not measure.** Context-split reliability is the
stability of each pair's value across sampled contexts. Both halves share
the verb pair, its Zipf and band, and the DAS bases. So two secondary
checks:
- **Reliability beyond frequency:** in each split, residualize both halves
  on participle Zipf and band (OLS across pairs), then correlate the
  residual halves. This is the reliability of the part of the gap that
  frequency does not explain.
- **Consistency across bases:** the per-pair passive gap computed with
  split-0, split-1 and split-2 bases only (cross-fitted within each split);
  mean pairwise Pearson r across pairs, and its Spearman-Brown value for 3.
Classical disattenuation assumes errors uncorrelated across measures. The
passive gap and the single-prompt " by" preference use the same contexts,
so that one correlation may be biased; the two other margins use different
contexts.

**Decision rule, per site × behaviour.** Significance and equivalence are
reported separately.
- **Measurement-limited:** median R < 0.6 for either measure.
- **Significance:** the 95% CI of r_obs excludes 0 (disattenuation keeps
  each draw's sign, so this is the same as for r_obs).
- **No moderate link:** the whole disattenuated 95% CI lies inside
  [−0.3, 0.3].
- Otherwise: **no detectable link; a moderate one is not excluded.**

**Prediction.**
- Every per-pair measure averaged over ≥ 100 contexts is stable: R ≥ 0.9
  for the passive gap at every site, the by preference and the released
  by margin; R ≥ 0.8 for the LP margin. The active gap (7 subjects) is
  lower but ≥ 0.7.
- The Zipf-residualized passive gap stays reliable (R ≥ 0.8), and the
  across-bases consistency is ≥ 0.9.
- So the null is not a measurement failure. With 59–64 pairs, most tests
  land in **no detectable link** (upper bounds 0.3–0.5), not in **no
  moderate link**. Site 4 may be significant.

## A2. Is the site-8 signal still present late?

**Question.** Does the transitive/intransitive separation along d_8 persist
at later layers, and in which voice?

**Run.** No patching. One forward pass (bf16, as in the passive test) over
every passive-test prompt; last-token residual at sites 4–23. Project onto
each of the 15 fold bases of d_8 and of d_17 at every site, and onto the
own-site d_s at the DAS sites (descriptive). Sign alignment: site-8
training items for d_8, site-17 training items for d_17, fixed across
sites (a sign flip with depth then shows up as a negative gap).

**Two fixed read-outs, the same at every site.**
- **d_8:** the raw projection x_s · d_8.
- **d_8⊥17:** e = (d_8 − c d_17) / sqrt(1 − c²), c = d_8 · d_17 (per fold;
  |c| ≈ 0.59). This removes the one direction where the late "object next"
  variable overlaps d_8. It does not remove every object-related
  direction, so the result is read as **retained linear separation**, not
  proof that the same functional variable persists.

**Metrics per site, read-out and frame** (cross-fitted as in step 1; good −
bad for passives, transitive − intransitive for actives):
- raw gap (projection units) and its retention, gap_s / gap_8, for each
  voice separately;
- standardized separation d' = gap / pooled within-class SD;
- AUC over items and over verb means;
- passive/active ratio = mean passive raw gap / mean active raw gap over
  pairs (one aggregation everywhere; the own-site ratios of
  `frequency_results.md` were means of per-pair ratios, so the own-site
  ratio is recomputed here the same way for comparison).

Populations: primary passives (good vs bad, ~125 contexts) and the same
verbs in the active frame (7 subjects); DAS held-out actives as a reference.
CIs: pair bootstrap of per-pair means.

Declared sites: 12, 14, 16, 17, 20, 23 (plus 8 as the reference). The
other sites are saved and plotted.

**Decision rule.**
- The separation is **present** at site s in a voice if the AUC over verb
  means is ≥ 0.8 and the raw-gap CI excludes 0.
- **Early separation is retained alongside the conversion** if, at sites
  12–17, the 95% CI of (ratio_s − ratio_8) on d_8⊥17 lies within ±0.1,
  while the own-site ratio falls below its site-8 value (CI below 0).
- **The early separation is itself converted** if the d_8⊥17 ratio falls
  with depth (CI of ratio_17 − ratio_8 below −0.1).
- Otherwise unresolved, reported with its CI.

**Prediction.**
- Present in both voices at every declared site through 17 (passive AUC
  over verb means ≥ 0.9). At 20 and 23 still present (≥ 0.8).
- Active raw gap along d_8 is retained or grows through 17 (≥ 0.8 × site
  8), because late writes overlap d_8.
- The d_8 passive/active ratio declines with depth (late "object next"
  writes are active-heavy) but stays above the own-site ratio at 12–17.
- The d_8⊥17 ratio stays near its site-8 value through 17: **early
  separation is retained alongside the conversion.**

## A3. Decomposing the gradual push

**Question.** In the site-8 dose-response on bad passives, objects creep up
over the passive range [0.10, 0.45] and accelerate above it, while " by"
flattens (`site8_mechanism_results.md`). Is the creep mostly the carry?
Which components produce the acceleration?

**Run.** The step-1 decomposition (fp32; carry + 256 heads + 16 MLPs in
layers 8–23; centered logits through the final LN) applied to
coordinate-setting patches instead of donor patches:
- bases: the primary bad passives (every item, split-0 basis, as in the
  dose-response);
- grid: z = 0.10, 0.275, 0.45, 0.675, 0.90, 1.20, 1.50;
- each consecutive interval [z_a, z_b] is decomposed as "z_b patched vs
  z_a reference", with the LN scale frozen at the z_b run; LN-scale term and
  residual kept separately, as in step 1;
- readouts: " by", ".", " the", " him", Ō; plus actual Δ log P(O),
  log P(" by"), log P(pron) for reference;
- log P(O) = LSE_O − LSE_all over centered logits; both terms are saved
  per grid point, so a change in log P(O) can be split into the object-set
  term and the partition-function term;
- per-z rates: each component's contribution / (z_b − z_a), averaged within
  pair, then over pairs; pair-bootstrap CIs;
- each interval freezes the LN scale at its own upper run, so even a
  linear term changes rate with 1/σ(z_b). A second version divides every
  interval by one fixed per-item scale (σ at z = 0.10); both are reported;
- downstream positive and negative totals (P, N) are reported next to the
  net.
- Reference (not in the decision): intransitive actives (DAS items,
  split-0 held-out basis) on the same grid.

**Declared intervals.** Inside the range R_p = [0.10, 0.45] (two
sub-intervals pooled); above it, [0.45, 0.90] and [0.90, 1.50].

**Decision rules.**
- **Is there acceleration in the logits?** On Ō: A = rate([0.90, 1.50]) −
  rate(R_p). The acceleration is **in the logits** if rate(R_p) > 0 (CI),
  A > 0 (CI) and rate([0.90, 1.50]) / rate(R_p) ≥ 1.25. If not, the
  acceleration in log P(O) comes from the read-out (the LSE_O term
  weighting the fastest-rising object tokens, or the partition function),
  and no component is credited with it; the LSE split says which.
- **Creep = carry** if the carry is ≥ 50% of the net Ō change in R_p and
  the downstream P and N are each smaller than the carry (no large
  cancelling writes).
- **Component shares** of A are computed only if A > 0 (CI): each
  component's own rate difference / A, with absolute rate differences
  alongside. **Late-MLP contribution** is supported if MLPs 15, 18, 21 and
  22 together give ≥ 50% of A. This is a direct-effect attribution; it is
  not called gating without a causal test (B7 tests the voice switch).
- For " by", the same rate differences locate the flattening (no
  threshold; descriptive).

**Prediction.**
- The Ō rate rises above the range (ratio ≥ 1.25, A > 0): the
  acceleration is in the logits, not only a read-out effect.
- The creep in R_p is mostly carry (≥ 60%): downstream components add
  little while z is in the passive range.
- The acceleration comes from the active object writers: MLPs 15, 18, 21,
  22 give ≥ 50% of A, with MLP23 opposing.
- " by" flattens because the mid-layer MLPs that raise " by" (11, 13, 14,
  16, 17) stop growing and MLP23 keeps lowering it.

## A4 (optional). Robustness of low early IIA

**Rank.** `run_das_round2.py final` at sites 4 and 6 with ranks 1, 2 and
4 (same recipe, epochs 3 and 2, same seeds), from explicit configs with no
fixed rank (`rank_sweep_config_site{4,6}.json`). Output:
`rank_sweep_site{4,6}/` (not `final_strict*`, so `export_bases.py` does not
pick them up). These runs do not replace the rank-1 bases.
- The declared rank rule (a larger rank only if it raises the mean
  held-out positive-control fraction by more than 0.1) is applied.
- Separately: the held-out cross-class IIA gain of ranks 2 and 4 over rank
  1, paired by fold × split (15 pairs of runs), with its CI.
- Conclusions are limited to the tested training budget (the rank-1
  epochs).

**Token count.** Held-out cross-class IIA and gap fraction split by the
pair's token count, at all eight sites, from the existing
`heldout_swaps.csv.gz` (rank 1), both swap directions reported.
- All 15 multi-token pairs are expansion pairs; the 14 single-token pairs
  are 6 expansion and 8 orig_head. **Primary comparison: expansion pairs
  only (6 single vs 15 multi).** All pairs is secondary.
- Each base pair's value is the mean over its swaps (all donors equally
  weighted), then pairs are averaged; CI: pair bootstrap.
- The natural class margin (|M| of the unpatched items) is reported by
  group, because IIA also depends on how far the base is from 0.

**Prediction.**
- The rank rule keeps rank 1 at both sites, and the paired IIA gain of
  rank 4 over rank 1 is < 0.05: low early IIA is not a rank limit at this
  training budget.
- At sites 4 and 6, within expansion pairs, multi-token IIA is below
  single-token IIA by ≥ 0.1; by site 12 the difference is < 0.05. This
  would fit a detokenization account (the last subtoken of a multi-token
  verb does not yet carry its class), which stays tentative: token count
  also differs in frequency and natural margin.

## Order (Part A)

1. Commit this plan.
2. A1 (CPU, local).
3. A2, A3, A4 on TSUBAME (one job each).
4. Report in `round3_results.md`.

# Part B: generalization and mechanism (declared 2026-10-08)

Written after Part A ran (results: `round3_results.md`) and before any Part
B run; revised after an outside review (Codex) of the first draft, also
before any run.

## B5. Reverse DAS: train on passives, test on actives

**Question.** Does a direction learned from the passive *by* preference
line up with the active-trained d_s and raise objects in actives (early
layers)? Or does it find a passive-only "agent / *by* next" variable that
does not (late layers)?

**Training data.**
- The same 29 strict pairs and the same fold ids (`pairs_folds.csv`), so
  each fold's passive and active directions are trained on (mostly) the
  same verbs.
- Frame "The N was <participle>", curated `passive_2` contexts. 15 contexts
  drawn once (seed 17), 5 per context band: 12 training contexts (4 per
  band) and 3 held-out contexts (1 per band). 58 verbs × 15 = 870 items.
- **Read-out, declared: *by* vs rest,** M_p = z_by − logsumexp(z without
  " by") (= log-odds of " by"). The by-vs-"." ratio is not used: task 3
  found that "." adds noise and does not differ between good and bad
  passives.
- Each fold has a threshold τ = the midpoint of the mean M_p of its
  retained training good and bad items (training contexts), fixed before
  training.
- **Behaviour filter:** a pair is kept if its good passive has the higher
  M_p in ≥ 8 of the 12 training contexts. Dropped pairs are excluded from
  training, held-out metrics and donors, and listed. Each fold should keep
  ≥ 3 held-out pairs as donors; folds with fewer are reported.

**Training.** As `run_das_round2.py`, with only the read-out changed: rank
1, interchange at the participle's last token, h ← h + ((h_src − h)·d) d,
BCE on σ(M_p − τ) with the source's **class** as the label (class-forcing,
as in the active run; natural good and bad passives overlap at the item
level, so this pushes the patched value toward the class side rather than
toward the donor's own value). Base and source share the context; 4
cross-class swaps per base and 2 same-class swaps; batch 64, lr 1e-2, the
same seeds.
- **Sites:** 4, 6, 8, 10, 12, 14, 16, 17.
- **Epochs:** a sweep on split 0 (8 epochs, held-out passive IIA per
  epoch); the frozen epoch rule (smallest epoch within 0.01 of the site's
  best mean held-out cross-class IIA). Written to
  `results/das_round2/reverse/frozen_config_site{s}.json` and committed
  before any active evaluation.
- **Final:** 3 splits × 5 folds at the frozen epochs. Passive-side controls
  per run: 100 random rank-1 directions (raw and norm-matched) and
  shuffled-label DAS.

**Held-out passive metrics** (held-out pairs; all contexts and the 3
untouched contexts separately).
- **Primary, continuous:** Δ M_p for bad base ← good source; and its
  fraction of the pair's natural gap, only for pairs whose natural gap is
  ≥ 0.2.
- Both swap directions and same-class preservation; IIA ((M_p,patched − τ
  > 0) == source class) with natural threshold accuracy as its reference.
- **Robustness gate per site:** held-out Δ M_p (bad ← good) beats the
  norm-matched random 95th percentile in ≥ 12 of 15 runs (the 15 runs are
  not independent; this is a robustness criterion, not a test).

**Active test (mirror of the passive test).**
- **Bases:** "<Subj> has <participle>" for the bad verbs of the 64 primary
  pairs (7 subjects; "She has emerged"); good verbs as a secondary base.
  Cross-fitting as before: the 8 orig_head pairs use only the fold where
  they are held out; other items one random fold per (item, split), seed 17.
- **Donors:** for item × split, the kept held-out pairs of the assigned
  fold (excluding the item's own pair). Each contributes its good passive
  (**G**) and bad passive (**B**) once, in one shared context drawn from
  the 111 curated contexts not used in passive training (seed 17); and, for
  the dose control, its transitive (**T**) and intransitive (**I**) active
  once, with one shared random subject.
- **Three arms on the same rows:**
  1. **d_p with passive donors** (G vs B): the primary test.
  2. **d_a with the same passive donors:** the active-trained basis of the
     same site, split and fold. Same donors, so arm 1 vs arm 2 compares the
     two directions directly.
  3. **d_p with active donors** (T vs I): a dose control. Passive donors
     are compressed along d (0.2–0.4 of the active separation), so a small
     arm-1 effect may only mean a small passive donor dose.
- Also the base verb's own passive (voice change only) and the active swap.
- **Contrast:** D = mean Δ after G (or T) donors − after B (or I) donors, on
  bad active bases, averaged within pair, then over pairs. Three-way
  bootstrap over base pairs (within band), subjects and donor pairs; 2,000
  draws, seed 17.
- **Readouts:** log P(O) (primary), log P(" by"), log P("."), log P(I),
  determiners / pronouns / reflexives, " the", " him", M.
- **Classification:** δ_s = the active-DAS bound of the same site (0.2 ×
  the active held-out Δ log P(O); 0.42–0.57), used as a practical effect
  size. RISE / FALL: 95% CI beyond 0 and |estimate| ≥ δ_s; NO RISE: 90% CI
  within ±δ_s; otherwise unresolved.
- **Null:** norm-matched random rank-1 directions (100 draws) on the
  split-0 G/B rows of bad bases, compared with DAS D on the same rows; a
  RISE / FALL counts only if DAS lies beyond the null's 95th (5th)
  percentile.

**Directions.** Per site: |cos(d_p, d_a)| for the same split and fold (15
values; median and range); the cosine of the sign-aligned mean directions;
within-run stability of d_p; the random baseline (~0.02). Natural
projection onto d_p: AUC of held-out DAS actives (transitive vs
intransitive) and of the primary pairs' actives (good vs bad verbs).

**Decision rule, per site.**

| Result | Reading |
|---|---|
| median \|cos\| ≥ 0.5, arm-1 D(O) RISE, active AUC along d_p ≥ 0.8 | **aligned, with cross-frame causal transfer** |
| median \|cos\| < 0.3, arm-1 and arm-3 D(O) both NO RISE | **passive-specific**: an "agent / *by* next" variable |
| arm-1 NO RISE but arm-3 RISE | **small transfer at the passive donor dose** (not passive-specific) |
| anything else | mixed / unresolved, reported with CIs |

Neither reading establishes that the two directions carry the same
variable; cos ≥ 0.5 is ≥ 25% shared variance.

**Prediction.**
- Aligned with transfer at 4–10; passive-specific at 14–17; 12 in between.
- Arm 1 is smaller than arm 2 at the early sites (d_a uses the passive
  donors' compressed values too, but is the better object writer).
- D(" by") on actives is ≤ 0 at 4–8 and positive at 14–17 (a "by next"
  value leaks into actives).
- Passive-side training is weaker than active training (the *by* signal
  is ~0.7 nats against ~5 for M) but passes the robustness gate at every
  site.

## B6. Get-passive transfer of the existing directions

**Question.** Does the voice switch generalize from "was" to "got"? A
failure would mean it does not generalize to get-passives, not that it
recognizes the literal token "was".

**Items.** The passive test unchanged except that every passive prompt has
"was" replaced by "got" ("The house got destroyed" / "The house got
emerged"). " was" and " got" are single tokens, and every passive prompt
keeps its length and differs by exactly one token, so prompt ids, plan
rows, bases, donors and cross-fitting are identical. Every patch is paired
with its was-passive counterpart.

**Eventive subset (declared now).** Get-passives favour dynamic events with
an affected patient. Excluded from the subset (good participle reads oddly
or statively after "got"): earned, bet, exerted, awed, forested, uttered,
blurted, eschewed, precluded, wadded, larded, tithed, blabbed, blasphemed,
disabused, edified. That leaves 48 of the 64 primary pairs. Results are
reported for all 64 (primary) and for the subset.

**Run.** `run_passive_test.py` (natural pass, projections, patches) at all
eight sites with the got prompts; analysis `analyze_passive_test.py
passive` with the same rules and δ_s.

**Paired difference.** D_got − D_was per readout and site, on the same plan
rows, with the same bootstrap. Also the natural projection gap ratio for
got-passives at each site.

**Behaviour gate.** The natural good − bad gap in log P(" by") on
got-passives. If it is < 0.2 nats, the *by* dimension is unresolved and no
ratio to it is reported.

**Decision rule, at site 8** (the abstract site for was-passives), from
the paired difference:
- **Generalizes to got:** the 90% CI of D(O)_got − D(O)_was lies within
  ±δ_8, and the *by* transfer is preserved: D(" by")_got 95% CI > 0 and
  its estimate ≥ 0.5 × D(" by")_was.
- **Does not generalize:** D(O)_got − D(O)_was has its 95% CI above 0 and
  estimate ≥ δ_8 (object leakage grows), or D(" by")_got's CI includes 0
  while the natural got gap is ≥ 0.2.
- Otherwise unresolved. The full depth pattern and the per-site
  classifications are reported alongside.

**Prediction.** The natural *by* gap is smaller after "got" (≈ 0.3–0.5
nats). At site 8 the switch generalizes (D(O) within ±δ_8 of the was
value; " by" preserved at ≥ half). The depth pattern (abstract → mixed →
surface) is the same.

## B7. Path patching: what switches the flagged MLPs between frames?

**Auxiliary/frame pair.** "The N was V" (passive) vs "The N has V" (same
tokens except the auxiliary). This is a minimal token contrast, not an
isolated voice counterfactual: tense/aspect, the subject's role and
plausibility change with it. Sensitivity frame: "The N had V"
(tense-matched), used for the gate and S_l only. Items: the primary bad
passives, 32 contexts per pair (seed 17), split-0 basis, all T donors of
the passive-test plan. The site-8 T-donor interchange is applied in every
frame with the same donor and basis. fp32.

**Quantity.** F_l(x) = MLP_l(LN_l(x)) on the pre-LN residual x at layer l
(Pythia's parallel block: MLP_l reads the residual before layer l). For
each flagged MLP l and its readout r (Ō for 15, 18, 21, 22; " by" for 11,
13, 14, 16, 17): Δ_l^f = r · (F_l(x^{f,patched}) − F_l(x^{f,unpatched})) /
σ_ref in frame f; σ_ref is the final-LN scale of the was-frame patched
run, the only frozen scale. The **switch** is S_l = Δ_l^has − Δ_l^was.

**Gate.** Δ_l^has has the sign of the step-1 active mean (held-out DAS
actives, a different population) for ≥ 7 of the 9 flagged MLPs. Otherwise
was → has does not capture the switch, and the path analysis is
descriptive only. S_l^had is reported next to S_l^has. Ratio tests below
use only MLPs with |S_l| ≥ 0.05 ("eligible").

**Path patching.** Upstream components of MLP_l: the site-8 **carry** (the
interchange displacement, patched minus unpatched site-8 state; zero in
unpatched runs), every attention head (l', h) and every MLP l' with l' <
l. Components below layer 8 are identical between patched and unpatched
runs within a frame but differ across frames. Replacing a component means:
in both the patched and the unpatched was-runs, add (its has-run output −
its was-run output) to MLP_l's pre-LN input only; LN_l and the MLP are
recomputed; everything else stays at the was-run.
- PE_l(C) = Δ_l(C replaced) − Δ_l^was, for single components and for
  joint groups: all heads, all MLPs, the carry.
- Replacing every component gives x^has exactly, so it must reproduce S_l
  (per-row check, tolerance 1e-3).
- Reported per MLP: S_l; Σ of single-component PE next to S_l; group PEs;
  the interaction residual S_l − (PE_heads + PE_MLPs + PE_carry); top
  components.

**Hypothesis (declared): attention reading "was".** Contexts are split in
two halves by context (seed 17); half A selects, half B tests.
- **H1, heads carry the switch:** sign(S_l) · PE_l(all heads, joint) ≥
  0.5 |S_l| for at least half of the eligible MLPs.
- **Alternative:** the same with all MLPs jointly (the switch arrives
  through earlier MLPs at the participle).
- **H2, few heads:** on half A, heads are ranked by Σ_l sign(S_l) PE_l(h) /
  |S_l| over eligible MLPs; on half B, the top 5 replaced jointly give
  sign(S_l) · PE_l(top 5) ≥ 0.5 × sign(S_l) · PE_l(all heads) for at least
  half of the eligible MLPs.
- **H3, they read the auxiliary:** on half B, each top-5 head's mean
  was-run attention from the participle's last token to the auxiliary is
  ≥ 0.3, and the **auxiliary-value-only** replacement of the top 5, W_O
  α_aux^{was,c} (V_aux^{has,c} − V_aux^{was,c}) per condition c, gives
  ≥ 50% of their joint PE for at least half of the eligible MLPs. Also
  reported: all-position value-only and pattern-only (Σ_t (α_t^has −
  α_t^was) V_t^was) replacements. A value-only failure leaves H3
  unresolved; it is not evidence against auxiliary mediation (fixed
  attention ignores routing through queries and keys).

**Prediction.** The gate passes. H1 holds for the mid-layer " by" MLPs
(11–17) and fails for the late object MLPs (18, 21, 22), where earlier MLPs
carry much of the switch. H2 and H3 hold: a handful of heads in layers
4–10 that attend to "was" carry the frame signal.

## Order (Part B)

1. Commit this part.
2. B6 and B7 (GPU, one job each).
3. B5: sweep and final passive training (GPU); commit the frozen configs
   and passive-side results; then the active test.
4. Report in `round3_results.md`.

# Part C: extensions (declared 2026-10-08)

Written while Part B was running and before any Part C run; revised after an
outside review (Codex), also before any run. Directions are the existing
active-trained d_s (sites 4–17).

**Nouns (both experiments), chosen by hand from the curated contexts:**
concrete or animate, broadly plausible patients, no temporal or event heads
(which would allow adjunct relatives such as "the day that John emerged").
Four come from contexts curated for each verb band: house, letter, suspect,
king (Head-verb contexts); child, horse, statue, thief (Tail-verb
contexts); baby, ship, suitcase, wallet (XTail-verb contexts). The band is
the verb band a context was written for, not the noun's frequency: these
are common nouns (patient Zipf 3.55–5.71; baby 5.26, ship 4.93), and noun
frequency is not a variable here. Where the analysis resamples contexts
"within band", it is this context-source band.

## C8. Nonce verbs: context-sensitive separation along d in passives

**Lemmas.** The 80 nonce lemmas selected in July (`nonce_passive/
lemmas.csv`), with their regular past forms ("dakked"). The July screen
checked the base form, so the past forms are re-checked here: the probe must
be a token-level suffix of every prompt it appears in.

**Context sentences.** The July lead schemas ("In the lab, the AGENT
PAST …") with the same three animate agents in every condition.

**Conditions** (probe = the test sentence, always with lemma A):

| Condition | Context | Purpose |
|---|---|---|
| matched T / matched I | 3 sentences, A with / without objects | main contrast |
| mismatched T / I | 3 sentences with partner B (two fixed derangements, seed 17) | generic priming |
| **balanced AB / BA** | 4 sentences, A and B alternating (A, B, A, B); in AB, A has objects and B none; in BA the roles swap | the same tokens and objects in both conditions; only which verb takes objects changes |
| none | probe alone | baseline |

**Probes.** Passive "The N was A-ed" (main), and active "SUBJ has A-ed"
(reference), 4 slots per lemma. Nouns are counterbalanced (slot j of lemma
i uses noun (i + 3j) mod 12); subjects likewise over the 7 DAS subjects.
**Real-verb reference:** the good and bad passives of the 64 primary pairs
after the "none" context and after a fixed neutral context (a mismatched-I
context), 4 slots each, as the scale for long prompts.

**Measures at the probe's last token.** Natural projection onto d_s at
sites 4–17 (declared: 6 and 8), raw and in z units (a fixed ruler: the
held-out DAS active scale; all 15 bases averaged); log P(" by"); log P(O),
log P("."), and log P(PREP without " by") so that *by* can be compared
with other PP starts.

**Contrasts per lemma** (mean over slots; lemma bootstrap, 2,000 draws,
seed 17):
- Δ_matched = matched T − matched I; Δ_mismatched likewise (each mapping);
- **Δ_balanced = probe A after AB − after BA** (verb-specific, context
  held constant);
- for z and log P(" by"), and log P(PREP\by) for comparison.

**Causal check (secondary).** At sites 6 and 8, patch the matched-T
probe's coordinate along d_s (split-0 bases, the passive-test interchange)
into the matched-I probe, and the reverse; report Δ log P(" by").

**Decision rules (passive probe, sites 6 and 8).** Minimum effects: 0.05 z
and 0.1 nats.
- **Context-sensitive separation along d:** Δ_matched z has its 95% CI
  above 0 and its estimate ≥ 0.05.
- **Verb-specific:** Δ_balanced z has its 95% CI above 0 and estimate
  ≥ 0.05.
- **Raises *by*, verb-specifically:** Δ_balanced log P(" by") has its 95%
  CI above 0, estimate ≥ 0.1, and exceeds Δ_balanced log P(PREP\by).
- Scale: Δ as a fraction of the real-verb passive gap in the same long
  contexts, and of the active-probe Δ.

**Prediction.** At sites 6–8 there is context-sensitive separation along d
in passives (Δ_matched ≈ 0.1–0.2 z), it is verb-specific (Δ_balanced > 0,
smaller than Δ_matched), and the balanced contrast raises " by" more than
other prepositions. The active-probe Δ is larger (≥ 2× the passive one).

## C9. Object relatives and tough constructions

**Items.** The 64 primary pairs × 24 contexts per construction:
- **Object relative (OR):** "The N that NAME V-past" ("The house that John
  destroyed" / "*The house that John emerged"); the 12 nouns × {John,
  Mary}. Past forms from `verb_forms.csv`.
- **Tough (TC):** "The N is ADJ to V" ("The book is easy to read" / "*The
  book is easy to sleep"); the 12 nouns × {easy, hard}. Base forms.
- **Base-form reference:** "NAME can V" for the same verbs (John, Mary),
  for the projection gate below.

**Readout sets, declared** (single tokens, checked at build time):
- **O:** the 27 object-start tokens.
- **PREP (its own set):** " to", " in", " with", " on", " at", " for",
  " from", " into", " over", " about", " of", " upon", " against",
  " through", " after", " under", " around", " across", " toward",
  " without", " during", " behind", " near", " onto", " by", " as".
- **MAIN (OR), agreement-compatible main-clause continuation:** " was",
  " is", " has", " had", " will", " would", " could", " can", " must",
  " should", " might", " did", " does".
- **END (TC), clause end:** primary ".", ",", "!", "?", ";"; broad END
  adds " and", " but", " because", " if", " when", " so" (secondary).
- **Licensing readout:** L_OR = log P(MAIN) − log P(PREP); L_TC =
  log P(END) − log P(PREP). It contrasts clause closure (a direct-object
  gap is complete) with a PP continuation (which could host a gap, as in
  "that John emerged from"). L_c can rise just because PREP falls, so the
  numerator and PREP are always reported separately.

**Natural gate.** The natural good − bad difference in L_c must be > 0
(95% CI over pairs); otherwise L_c is not used for that construction. The
tokens driving each natural and patched effect are listed (top 5 by
probability change).

**TC base-form gate.** At each site, the base-form references ("John can
read" vs "John can sleep") must separate along d_s (AUC over verb means ≥
0.8). Where they do not, TC results at that site are inconclusive.

**Patches.** As the passive test, at all eight sites, on the bad item
(primary) and the good item (secondary):
- **T vs I active donors** from the held-out DAS pairs of the cross-fitted
  fold (one shared random subject per donor pair), at the verb's last
  token: D = mean Δ after T − after I.
- **Within-construction swap:** the good item's verb state, same context,
  into the bad item (and the reverse).
- Three-way bootstrap over base pairs (within band), contexts and donor
  pairs; 2,000 draws, seed 17.
- Natural projection of good vs bad items onto d_s (z gap; its ratio to
  the active gap of the same verbs), as step 1 of the passive test.

**Classification per site and construction.** O: RISE / FALL / NO RISE
with δ_s (as the passive test). L_c: RISE / FALL / NO RISE with δ_L = 0.2 ×
the construction's natural good − bad L_c gap (a readout-specific bound).

| O | L_c | Reading |
|---|---|---|
| NO RISE | RISE, with the numerator's D 95% CI above 0 | **consistent with direct-gap licensing** |
| RISE | NO RISE or FALL | **surface**: "object next" |
| RISE | RISE | mixed |
| NO RISE | NO RISE | nothing moves |
| FALL, or any other combination, or a failed gate | | unresolved |

**Prediction.**
- Both natural gates pass.
- OR: consistent with direct-gap licensing at 4–8, surface at 14–17.
- TC: the base-form gate passes at most sites; licensing at 4–8 is less
  certain than for OR (the verb is a bare infinitive; d was trained on
  participles); surface at 14–17.
- The within-construction swap raises L_c at sites 4–12 in both.

## Order (Part C)

1. Commit this part.
2. C8 and C9 builders and runners (tokenization checks on TSUBAME), GPU
   jobs.
3. Report in `round3_results.md`.
