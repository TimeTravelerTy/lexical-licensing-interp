# Round 7: declared plan (paper 1: frequency and entrenchment)

Builds on rounds 1–6 (`../results_index.md`). Same rules as rounds 5–6:
each part is written and committed before any of its runs or analyses,
with definitions, predictions and decision rules stated before the
outcomes are computed; where a prediction leans on earlier outputs, that is
said. An outside reviewer (Codex) reviews each part's draft and code before
the analysis.

Notation as in round 6 (`../round6/plan.md`): T / I = transitive /
intransitive verb of a D11 pair; Δ_pass(V) = log P(V | "The N was") −
log P(V | "The N has") (full participle log-prob, summed over subtokens);
L = Δ_pass(T) − Δ_pass(I); a_V = u_V1 · Δh̄; c_total = A1's boundary-safe
Pile count. CIs are 95%, 2,000 draws, seed 17, unless stated otherwise.
Layout: plan and reports in `reports/round7/` (figures in
`reports/round7/figures/`), outputs in `results/round7/<part>/`.


# Part A8: per-verb training trajectories (declared 2026-10-21)

A reanalysis of A6's checkpoint scores; no new GPU jobs. Script
`scripts/analyze_trajectories.py`; report `a8_trajectories.md`; outputs
`results/round7/trajectories/`. Revised after an outside review (Codex) of
the first draft and its code, before any real outcome was computed (the
changes are listed at the end of this part).

**Disclosures.**
1. The predictions below were formed after seeing A6's band-level
   trajectories (`../round6/a6_checkpoints.md`: band means of Δ_pass(I),
   Δ_pass(T) and L at every checkpoint, the curve-collapse comparisons, the
   M_Eg step offsets, and the matched-exposure diagnostic, +0.080 at 1.4B,
   +0.133 at 410M, +0.179 at 6.9B). This part is confirmatory only for the
   per-verb and exposure-aligned versions below, not for the band means
   already reported. In particular, with every verb observed at the same
   steps, the point estimate of β_w (part 2) is the time slope of the
   all-verb mean trajectory, which follows from A6's band means and band
   sizes; and the late band means show the sign of β_b. So the β_w and
   β_b predictions are informed by seen aggregates; the per-verb
   quantities (onsets and β_on, the between-verb slope's size, β_b − β_w
   with its calibration, γ, and the exposure-aligned D(E)) are the
   confirmatory ones.
2. Per-verb trajectories have not been looked at. A6's per-verb cell file
   (`results/round6/checkpoints/a6_cells.csv`) exists; only its header and
   two step-64 rows were seen, while checking its layout. Before this
   declaration, outcome-free design checks were run on the counts only
   (the quintile count ranges and the exposure range each bin covers,
   quoted in part 3).
3. The script was tested on synthetic scores before it read the real ones
   (results at the end of this part); the test showed that the
   crossers-only onset fit is biased towards the prediction (part 1).

## Data and definitions

- **Scores.** A6's runs (`results/round6/checkpoints/pythia-<size>/step<N>/`
  `frame_scores.parquet`): the 281 D11 pairs in the 24 freq frames
  ("The N was / has", 12 animate and 12 inanimate nouns). Every verb occurs
  in exactly one pair, so a per-verb value is a per-pair value.
- **Models and steps.** Pythia-1.4B (primary) and 410M (replication):
  steps 64, 128, 256, 512, 1000, 2000, 3000, 4000, 6000, 10000, 16000,
  25000, 40000, 63000, 100000, 143000. Pythia-6.9B (descriptive): 512,
  2000, 8000, 23000, 63000, 143000.
- **Per-verb outcome.** Δ_pass(V)_{v,t} = the mean over the 24 contexts
  (A6's values; noun-weighted in bootstrap draws). Δ_pass(I) is primary;
  Δ_pass(T) and L are used in part 4 (descriptive).
- **Counts and exposure.** x_v = log10(c_total,v + 0.5) (A1's final count
  file, `data/round6/counts/pile_counts.csv`). Exposure as in A6's code:
  E_{v,t} = (c_total,v + 0.5) × t / 143000. The spec writes c_total; the
  +0.5 is A6's (and A1's) smoothing and changes x by at most 0.0001 here
  (the smallest c_total is above 3,000). Pythia sees 2,097,152 tokens per
  step, so tokens_seen(t) / tokens_seen(143000) = t / 143000;
  log10 E_{v,t} = x_v + s_t with s_t = log10(t / 143000).
- **Token count.** n_tok = the participle's Pythia subtoken count; log n_tok
  is the covariate where stated.
- **Bootstrap.** A6's declared scheme: pairs (verbs) within band × nouns
  within animacy, 2,000 draws, seed 17, drawn in A6's order so that the
  draws are A6's collapse draws; identical across checkpoints and models;
  draw 0 = the point estimate; percentile intervals. Every statistic is
  recomputed in every draw (onsets, the set of crossers, bin means). A fit
  that is not identified in a draw is dropped from that fit's interval:
  least squares needs at least k + 2 units with positive weight (k
  coefficients) and a full-rank design on them; the censored regressions
  also need at least k + 1 uncensored or interval-censored units with a
  full-rank design on those (an all-censored sample has no finite
  maximum). A censored fit is accepted only if it converges (L-BFGS-B,
  then BFGS; success or a maximum gradient below 1e-4) with σ in (1e-4,
  1e3) and every |coefficient| < 1e3; its start values come from least
  squares on all verbs with censored onsets set to their bounds, so they
  do not depend on the crossers-only fit. Valid-draw counts are reported;
  a point estimate that cannot be computed is labelled "unestimable".

**Reliability.** At each checkpoint, the split-half reliability of
per-verb Δ_pass(I): 200 random splits of the 24 contexts into halves of 12
(6 animate + 6 inanimate per half; seed 17; the same splits at every
checkpoint and model). Per split, the Pearson r across the 281 I verbs
between the two half-means; r_half = the mean over splits; r_SB = 2 r_half
/ (1 + r_half) (Spearman–Brown: the reliability of the 24-context mean,
which is what every per-verb result uses). Reported overall (the flag) and
within band, and for Δ_pass(T) (descriptive). The overall r can be driven
by band separation; the within-band values are set beside any band-level
reading.
- **Flag:** a checkpoint with r_SB < 0.7, or with an undefined r_SB, is
  flagged (†) in every per-verb result: the reliability table; onsets
  (counts of onsets whose interpolation uses a flagged checkpoint, and of
  onsets whose persistence window contains one; per-verb flags in
  `a8_onsets.csv`); the late window (flagged window steps; F3 marks the
  model); D(E) (grid points whose interpolation uses a flagged step); the
  class tables of part 4; and the figures (hollow markers in F1, F2, F3).
- Onset reliability (descriptive): onsets (primary definition) computed
  from each half of the same 200 splits; the Pearson r of log10 onset
  across verbs that cross in both halves, averaged over splits with a
  finite r, with the mean number of contributing verbs and the share of
  verbs whose censoring status agrees between halves.

## 1. Onset: when does each intransitive stop preferring "was"?

**Definition (primary).** onset_v = the first scored checkpoint t_k at
which Δ_pass(I)_{v,t_k} < 0 and stays < 0 at the next two scored
checkpoints. Only checkpoints with two later checkpoints can qualify (1.4B
and 410M: up to step 63000; 6.9B: up to 23000). The checkpoint before t_k
then has Δ_pass(I) ≥ 0 (otherwise it would qualify), and log10 onset_v is
interpolated linearly in log10 step between (t_{k−1}, y_{k−1}) and
(t_k, y_k) at y = 0.
- **Left-censored:** the first scored step qualifies (< 0 there and at the
  next two steps; onset ≤ 64, 6.9B ≤ 512). A verb that is < 0 at the first
  step but ≥ 0 at one of the next two is not left-censored: its onset is
  its later qualifying crossing (the number of such verbs is reported).
  This reads the spec's "already < 0 at the first scored step" with the
  same persistence condition as the onset.
- **Right-censored:** no checkpoint qualifies. Its onset is then later than
  the last checkpoint that could qualify (63000; 6.9B 23000), which is the
  right-censoring bound: any later qualifying crossing is interpolated from
  a checkpoint at or after that bound.
- **Sensitivities:** lenient end (the persistence condition uses the later
  checkpoints that exist, so 100000 and 143000 can qualify; bound 143000);
  (a) first crossing without the persistence condition (left-censored if
  Δ_pass(I) < 0 at the first step; bound 143000); (b) threshold −1 in
  place of 0 (persistence as primary); (c) a censored (Tobit) regression on
  all verbs; (d) without the token-count covariate; (e) **maturity floor**:
  the Tobit with every onset at or before step 512 treated as
  left-censored at 512 (part 3's cut). No verb can cross before the model
  has learned the construction (A6: Head AUC 0.54–0.69 before step 512),
  which bunches the frequent verbs' onsets at that floor and flattens β_on
  whatever rare verbs do; (e) discards the exact timing of onsets at or
  before 512, so the bunched onsets no longer set the slope. It does not
  model developmental gating. (f)
  **checkpoint-interval censoring**: each crosser's onset is only known to
  lie between t_{k−1} and t_k (no interpolation), left- and
  right-censored verbs as in (c); interval-censored normal regression.
- Reported: counts by band of crossers, left- and right-censored verbs for
  each definition; onsets in the last qualifying interval; flag counts.

**Fit.** log10 onset_v = a + β_on x_v + γ log n_tok,v + e: OLS on the
crossers, verbs equally weighted (bootstrap multiplicities as weights).
(c), (e), (f): the same mean model with normal errors, weighted maximum
likelihood, left-censored at log10 of the first scored step (e: at
log10 512), right-censored at log10 of the bound. Dropping censored verbs
biases β_on towards 0 (left-censored verbs are frequent verbs with the
earliest onsets, right-censored verbs rare verbs with the latest), which
is the direction of the prediction. The synthetic test (below) shows the
size of this bias: with data generated under exact count equivalence (true
β_on = −1) and 110 of 281 verbs left-censored, the crossers-only fit gave
−0.86 [−1.00, −0.77], which the primary rule reads "held", while (c), (e)
and (f) gave −0.98, −1.04 and −0.96 (CIs including −1). The censored fits
assume a linear, homoskedastic normal latent onset; they correct
censoring, not developmental gating or misspecification.

**Count equivalence** (onset at a constant own exposure, E_{v,onset} = E*)
gives log10 onset_v = log10 E* + log10 143000 − x_v, i.e. **β_on = −1**.

**Prediction:** −1 < β_on < 0 (rare verbs cross earlier than their own
exposure predicts).

**Decision** (95% CI of β_on; primary definition and fit, 1.4B):
- **held**: the CI lies within (−1, 0);
- **failed (consistent with count equivalence)**: the CI includes −1 and
  excludes 0;
- **no frequency effect on onset**: the CI includes 0 and excludes −1;
- **unresolved**: the CI includes both −1 and 0;
- **failed, steeper than count equivalence**: the CI lies below −1;
- **failed, reversed**: the CI lies above 0.

**Verdict:** the primary label, except that a "held" is reported as
**"held, not robust to censoring"** (naming which) when the Tobit (c) or
the maturity-floor Tobit (e) is not "held". The other sensitivities are
reported with their labels and do not change the verdict. A label that
says a CI includes −1 or 0 is not evidence that the slope equals it.

Also reported: onset exposure log10 E_{v,onset} = x_v + log10(onset_v /
143000), median and interquartile range by band (crossers). In the
crossers sample and with the same covariates, its slope on x_v is
β_on + 1 (0 under count equivalence).

## 2. Late window: between-verb vs within-verb slope

Window: the scored steps ≥ 16000 (1.4B and 410M: 16000, 25000, 40000,
63000, 100000, 143000; 6.9B: 23000, 63000, 143000).
- **Between:** ȳ_v = the mean of Δ_pass(I)_{v,t} over the window; OLS of
  ȳ_v on x_v → **β_b** (identical on the verb's mean log10 E, which differs
  from x_v by a constant). Sensitivity: + log n_tok.
- **Within:** Δ_pass(I)_{v,t} on log10 E_{v,t} with verb fixed effects →
  **β_w**. Within a verb, log10 E varies only through log10 step (x_v is
  absorbed by the fixed effect), so β_w is the shared late time slope, in
  nats per decade of training. (Computed as the weighted mean of per-verb
  OLS slopes on log10 step, which equals the fixed-effects slope because
  every verb has the same steps.)
- **Interaction:** Δ_pass(I)_{v,t} = α_v + β_t s_t + γ (x_v − x̄) s_t, x̄ the
  inventory mean of x → **γ**: does the late time slope depend on
  frequency? (γ > 0: rarer verbs decline more steeply late.)
- Verbs are equally weighted (bootstrap multiplicities); each verb
  contributes every window step; β_b − β_w from paired draws.

Count equivalence predicts β_w = β_b only when the common curve f in
Δ_pass(I) = f(log10 E) is roughly linear over the late exposure range: with
a nonlinear f (e.g. frequent verbs on a floor, rare verbs on a slope),
β_b − β_w ≠ 0 and γ ≠ 0 can arise from one common curve. With f flattening
at high exposure, count equivalence also predicts γ > 0. β_b compares
verbs, and frequency also indexes stable lexical differences between
them; β_w compares each verb with itself over time.

**Calibration against a common nonlinear curve** (qualifier). In each
draw, f̂ = a natural cubic spline in log10 E (A6's basis: knots at the
2.5, 33.3, 66.7 and 97.5 percentiles of the window's log10 E values) fitted
by weighted least squares to all verb × window-step values; β_b, β_w and γ
are recomputed on the fitted values f̂(log10 E_{v,t}) (the values one
common exposure curve implies for this exposure layout), and the
calibrated contrasts are observed − implied: (β_b − β_w) − (β_b − β_w)_f̂
and γ − γ_f̂. This tests departure from the declared spline projection,
not from every common curve: spline approximation error can leave a
nonzero calibrated contrast, and a calibrated interval that includes 0
does not establish that one curve is adequate.

**Predictions** (1.4B; 410M the same):
- **main: β_b − β_w < 0** (95% CI below 0);
- β_b near A1's −0.728: its 95% CI excludes 0 and includes −0.728;
- β_w near 0: its 90% CI lies within ±0.25 nats per decade;
- γ unresolved: its 95% CI includes 0.

**Decision.** Main: **held** if the 95% CI of β_b − β_w lies below 0;
**failed (consistent with count equivalence)** if it includes 0;
**reversed** if it lies above 0. A "held" is reported as **"held, within
what a common nonlinear exposure curve produces"** when the calibrated
contrast's CI does not lie below 0, and as "held (also beyond the fitted
common exposure curve)" when it does. The other three predictions: held or
missed as stated; γ (and calibrated γ) labelled positive / negative when
the CI excludes 0, else unresolved.

## 3. Exposure-aligned curves (crossing test)

**Bins.** The 281 I verbs by c_total quintile on the fixed inventory
(ranked by x ascending, ties broken by the form alphabetically; 57 / 56 /
56 / 56 / 56; Q1 rarest). From the counts only: Q1 log10 c 3.52–4.33 (all
XTail), Q2 4.33–4.86, Q3 4.86–5.23, Q4 5.24–5.77, Q5 5.78–7.22 (43 Head,
13 Tail). Sensitivity: Zipf bands (XTail, Tail, Head) as the bins.

**Bin curves** (declared construction). At step t, the bin mean Δ̄_q(t)
(bootstrap-weighted over the bin's verbs) sits at the bin's exposure
X_q(t) = x̄_q + s_t, where x̄_q is the bin's inventory mean of x (fixed, the
same in every draw). Δ̄_q is interpolated linearly in X onto a grid of
log10 E at multiples of 0.1, within [X_q(t_first), X_q(t_last)] (no
extrapolation).

**Per-verb alignment** (qualifier). Aligning bins at their mean log c can
itself produce a crossing under count equivalence: the bin curve is the
mean of f(X + x_v − x̄_q) over the bin's verbs, not f(X), and with
different spreads of x within bins and a curved f the two bins' curves
can cross. So each verb's own trajectory is also interpolated (linear in
its own log10 E) onto the grid, wherever its exposure range covers the
grid point, and a bin's value at a grid point is the bootstrap-weighted
mean over its covering verbs. A grid point belongs to a bin's support
when at least half of the bin's verbs cover it (fixed by the counts and
steps alone). This removes the bin-spread term, but not all of the null
artefact: linear interpolation between checkpoints leaves an error that
differs between verbs whose checkpoints sit at shifted exposures, so
under count equivalence D(E) is zero only up to interpolation error.

**Common-curve calibration** (qualifier, both alignments). In each draw a
natural cubic spline f̂(log10 E) (A6's basis, knots at the 2.5, 33.3, 66.7
and 97.5 percentiles of the log10 E values of all verbs at the
comparison's steps) is fitted by weighted least squares to every verb ×
step value of that step set; its fitted values are passed through the
same sampling, binning and interpolation, giving D_null(E); the
calibrated contrast is D(E) − D_null(E), read with the same pointwise
rule. This absorbs both artefacts above (bin spread and interpolation)
to the extent that f̂ approximates one common curve; the part-2 caveat on
spline projections applies.

**Comparison.** For a rarer bin q and a more frequent bin r: the grid
points inside both ranges (bin curves) or both supports (per-verb), and
D_qr(E) = Δ̄_q(E) − Δ̄_r(E) at each, with a pointwise percentile 95% CI and
a simultaneous 95% band over the comparison's grid points (sup-t: the 95th
percentile over draws of max_E |D* − D̂| / sd(E), over draws in which every
grid point is defined, their number reported; a grid point with zero
bootstrap SD gets a zero-width band).
- **Primary: Q1 − Q5**, all steps, bin curves.
- Also: adjacent bins (Q1 − Q2, Q2 − Q3, Q3 − Q4, Q4 − Q5); Zipf bands
  (XTail − Head, XTail − Tail, Tail − Head); both alignments.
- **Sensitivity: steps ≥ 512** (every curve starts at step 512), for every
  comparison: before step 512 the Head AUC is 0.54–0.69, so low exposure
  there is confounded with an untrained model.
- Expected overlap, from the counts: Q1 − Q5 covers 1.04 decades with all
  steps (Q5 at steps 64 to about 700 against Q1 at steps about 13000 to
  143000), and 0.14 decades (one or two grid points) with steps ≥ 512. A
  comparison with no common grid point is "no overlap"; with one grid
  point, "insufficient overlap" (a crossing needs two). Under the
  sensitivity this is a limit of the design, not evidence against a
  crossing.

**Prediction** (Q1 − Q5, all steps, bin curves, 1.4B): D(E) < 0 at low E
(rare verbs, seen by a more trained model, reject more than frequent verbs
did at the same exposure) and D(E) > 0 at high E (rare verbs plateau
above).

**Decision** (per comparison): **crossing** if one grid point has D < 0
with its 95% CI below 0 and another has D > 0 with its CI above 0
(pointwise, the spec's rule); "predicted order" if a resolved negative
point lies at lower E than a resolved positive point, "reversed order" if
only the opposite, "non-monotonic: both orders" if both. Otherwise the
sign that holds where resolved is reported (resolved points only
negative, only positive, or none resolved), with the E range of each.
Reported with every comparison: the crossing position(s) in log10 E (where
the point estimate changes sign between grid points, interpolated; across
a run of exact zeros or undefined points, the run's midpoint), the same
reading with the simultaneous band and for the calibrated contrast, and
the number of grid points that use a flagged step (a checkpoint with zero
interpolation weight does not count).

**Verdict:** **held** if the primary comparison shows a crossing in the
predicted order (alone or within "both orders"); reported as "held
pointwise; not robust to …", naming each qualifier that does not show a
predicted-order crossing: simultaneous inference (the simultaneous band),
within-bin spread (the per-verb alignment, pointwise), the common-curve
calibration (bin curves, pointwise); otherwise **missed**. The principal claim is this one comparison (1.4B); the others
are descriptive. Matched exposure here compares frequent verbs early in
training with rare verbs late, so even a robust crossing is an
observational trajectory pattern confounded with model age, not a
measure of learning efficiency.

**Check against A6.** A6's matched-exposure diagnostic is recomputed from
the same scores with A6's rule (verb × step cells with |Δ log10 E| ≤ 0.1
— A6's code uses ≤ although its plan says < — and steps ≥ 4× apart; mean of
later − earlier); its point estimate must reproduce A6's reported value to
three decimals (+0.080, +0.133, +0.179), with a bootstrap CI added. This
is blocking: if it is not reproduced, the script stops before writing any
output, and the discrepancy is investigated before any interpretation. Beside
it, the pooled D̄ = the mean of D over every (pair, grid point) cell of the
10 quintile pairs (bin curves, all steps; rarer − more frequent), with its
CI and the paired difference D̄ − diagnostic. Both average the same kind
of comparison (the rarer verb at a later step against the more frequent
verb at an earlier step, at matched exposure) with different weights: A6
weights by the number of matched cell pairs, which concentrates where
verbs are dense, includes pairs within a bin and requires a 4× step ratio;
D̄ averages bins and weights grid cells equally. They should roughly
agree; the report explains any gap.

## 4. Class-level vs verb-specific (descriptive)

- Per checkpoint and band: **AUC** = P(Δ_pass(T) of a random T verb >
  Δ_pass(I) of a random I verb) within the band (all T × I verb
  combinations, ties counted half, bootstrap multiplicities as weights),
  and the mean Δ_pass(T) − Δ_pass(I) (the band mean of L).
- **Separation step:** the first checkpoint after which the band's AUC CI
  lies above 0.5 at every later checkpoint (descriptive, as A6's onset).
  For XTail it is set beside (i) the median primary onset of the XTail I
  verbs (left- and right-censored verbs counted below and above every
  crosser) and (ii) the first checkpoint after which the XTail band-mean
  Δ_pass(I) is < 0 at every later checkpoint; the order is reported.
- **a_V.** A6 stored, per checkpoint, the final states h for every context
  and frame, the unembedding rows of the verbs' first subtokens and the
  log-sum-exp terms, so a_V = u_V1 · Δh̄ is available without new runs. For
  single-token verbs (Head T 44, I 45; Tail T 24, I 20; the one XTail
  single-token T is not reported): mean a_V by band × class and the
  within-band AUC, per checkpoint. For a single-token verb a_V = Δ_pass(V)
  + a per-checkpoint constant (Pythia has no unembedding bias, so lp = u ·
  h − LSE), so this AUC equals the Δ_pass AUC on the single-token subset;
  the identity is checked numerically and its maximum deviation reported.
- No predictions.

## Models, replication, figures, outputs

All four parts on 1.4B (primary), 410M (replication) and 6.9B
(descriptive; with 6 steps the onsets are coarse, the primary onset can
only qualify up to step 23000, and the late window has 3 steps).
**Replication** compares 410M's verdicts with 1.4B's for the three
principal claims: the onset verdict (part 1, including the censoring
qualifier), the late-window verdict (part 2, including the calibration
qualifier) and the crossing verdict (part 3, including its qualifiers);
"replicates" = the same verdict. Matching labels that are not "held"
(e.g. both unresolved) are matching inconclusiveness, not confirmation.

Figures (`reports/round7/figures/`, PNG and PDF; paper candidates):
- **F1** `a8_f1_exposure_<size>`: bin mean Δ_pass(I) against bin log10
  exposure, one line per count quintile, points coloured by step, flagged
  steps hollow (1.4B; 410M in the appendix).
- **F2** `a8_f2_onset_<size>`: onset step against c_total, log-log;
  crossers as points by band (hollow when the interpolation uses a flagged
  step), censored verbs as triangles at the bounds; the OLS fit and a
  slope −1 reference line through the crossers' mean point.
- **F3** `a8_f3_late_slopes`: β_b, β_w, β_b − β_w and the calibrated
  contrast with 95% CIs, 1.4B and 410M (6.9B in grey, descriptive); a
  hollow marker when the window contains a flagged step.

Outputs (`results/round7/trajectories/`): `a8_decisions.json`,
`a8_summary.csv` (with valid-draw counts), `a8_onsets.csv` (per verb ×
model × definition: onset, censoring, interval, log10 E at onset, flags),
`a8_reliability.csv`, `a8_dgrid.csv` (both alignments), `a8_class.csv`.
Report `a8_trajectories.md`; rows R7-A8 in `../results_index.md` (tag P1).

## Synthetic test

Run before this declaration was committed; files in a scratch directory,
never in `results/`; A6-format scores and states for all three models,
200 draws (the first version of the script, then the revised one).
- **(A) Count equivalence:** Δ_pass(I) = f(log10 E) + a verb offset
  (sd 0.3) + context noise, logistic f, T verbs at a constant offset.
  1.4B: Tobit β_on −0.98 [−1.14, −0.84], maturity-floor Tobit −1.04
  [−1.23, −0.90], interval-censored −0.96 [−1.15, −0.83], threshold −1
  −1.01 [−1.05, −0.94]; crossers-only −0.86 [−1.00, −0.77] (the censoring
  bias above; verdict "held, not robust to censoring"). β_b − β_w +0.07
  [−0.07, +0.23] (count equivalence not rejected), its value under the
  fitted common curve +0.09 [+0.05, +0.13], calibrated −0.01 [−0.18,
  +0.14]; γ +0.58 [+0.35, +0.86] (positive, as count equivalence with a
  flattening f predicts), calibrated −0.04 [−0.23, +0.17]. No resolved
  D(E) point in any quintile comparison, either alignment; with the
  common-curve calibration, no crossing in any comparison of any model
  (isolated single resolved points in 4 of the 92 comparisons with at
  least two grid points: pointwise multiplicity); pooled D̄ −0.007 and the
  recomputed A6 diagnostic +0.023, both CIs including 0; a_V identity to
  2e-7.
- **(B) A planted departure:** Δ_pass(I) = f(s + 0.5 x + 2.6) (true
  β_on = −0.5): Tobit −0.53 [−0.63, −0.36], maturity-floor −0.56,
  interval-censored −0.54, crossers-only −0.41; D(E) < 0 resolved in every
  quintile comparison with at least two grid points, both alignments.

## Changes after the outside review (before any real outcome)

Codex reviewed the first draft and script. Changes: (1) the primary onset
now requires two later checkpoints, as the spec says (the first draft
let the last two steps qualify with fewer; kept as the "lenient end"
sensitivity), with right-censoring at the last checkpoint that can
qualify; left-censoring stated with the persistence condition; (2) the
onset verdict is qualified by the maturity-floor Tobit as well as the
Tobit, and a checkpoint-interval-censored fit is added; (3) the late
window gains the common-curve calibration, and its "held" is qualified by
it; (4) part 3 gains the per-verb alignment, the simultaneous-band
qualifier for the verdict, and "insufficient overlap" for one-point grids;
(5) fits that are unidentified or do not converge are dropped and counted
rather than reported as estimates; undefined reliability is flagged;
flags are carried into the onset file and all three figures; crossing
positions are labelled as log10 E; (6) disclosure 1 now says which
predictions were informed by seen aggregates; (7) the A6 diagnostic must
reproduce A6's value exactly; replication is defined on the verdicts.
Kept as specified: the crossers-only primary onset fit and the pointwise
crossing rule (both now carry declared robustness qualifiers).

A second Codex pass on the revision found: (8) the censored fits checked
convergence but not identification (an all-censored or rank-deficient
sample could "converge"), and their start values came from the
crossers-only fit; (9) the shared interval helper fails when no draw is
finite; (10) the per-verb alignment still leaves interpolation error under
a common curve. Changes: identification and sanity checks and
independent start values for the censored fits; a NaN-safe interval; the
common-curve calibration of D(E) as a third crossing qualifier; narrower
wording of what the maturity floor and the spline calibration establish;
the A6 reproduction made blocking; crossing positions and the sup-t made
safe for undefined points; flags no longer counted for checkpoints with
zero interpolation weight.

## Order (A8)

1. Plan and script reviewed (Codex), revised, committed.
2. Run the analysis (local; no GPU).
3. Report, figures, results index; commit.
