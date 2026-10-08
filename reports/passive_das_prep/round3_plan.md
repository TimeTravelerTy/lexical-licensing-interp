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
