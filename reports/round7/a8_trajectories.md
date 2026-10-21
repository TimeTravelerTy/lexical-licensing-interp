# Round 7, A8: per-verb training trajectories

Spec: `plan.md`, A8. Reanalysis of A6's checkpoint scores (`results/round6/checkpoints/`, freq prefixes; no new runs); analysis `analyze_trajectories.py`. Per-verb Δ_pass = mean over the 24 freq contexts. Exposure E = (c_total + 0.5) × step / 143000 (A6). 95% CIs: A6's draws (pairs within band × nouns within animacy, 2,000, seed 17). † = checkpoint with split-half reliability r_SB < 0.7 (or undefined).

## Summary

Written after the run from the generated sections below (everything from "pythia-1.4b" on is the script's output). Verdicts follow the declared rules (`plan.md`, A8): 1.4B primary, 410M replication, 6.9B descriptive. Figures: `figures/a8_f1_exposure_<size>`, `figures/a8_f2_onset_<size>`, `figures/a8_f3_late_slopes` (PNG and PDF).

| claim (prediction) | pythia-1.4b | pythia-410m | pythia-6.9b |
|---|---|---|---|
| 1. onset: −1 < β_on < 0 | **held**: −0.39 [−0.54, −0.28]; Tobit −0.62 [−0.90, −0.37], maturity floor −0.59 [−0.86, −0.37], interval −0.62 [−0.91, −0.38] | held, not robust to censoring: −0.19 [−0.39, −0.07]; maturity-floor Tobit −0.64 [−1.05, −0.38] | held: −0.17 [−0.29, −0.0006]; Tobit −0.58 [−0.87, −0.29] |
| 2. late window: β_b − β_w < 0 | **held**, also after calibration: −0.73 [−1.04, −0.43]; calibrated −0.73 [−1.05, −0.42] | held, also after calibration: −0.45 [−0.77, −0.11]; calibrated −0.44 [−0.77, −0.09] | failed (consistent with count equivalence): −0.18 [−0.49, +0.15] |
| 3. crossing, Q1 − Q5 | **missed**: only negative where resolved (log10 E 3.0–3.6) | missed: only negative (3.0–3.8) | missed: one grid point |
| replication (410M = 1.4B verdict) | | onset no; late window yes; crossing yes (both missed) | |

**Reliability.** No checkpoint is flagged: r_SB of per-verb Δ_pass(I) is 0.98–1.00 at every step of every model (0.97–1.00 within band), and 0.97–1.00 for Δ_pass(T). High values at the earliest steps mean consistency across nouns, not learned grammar. A check run after the analysis: at step 64 (1.4B) the per-verb means spread with SD 0.38 nats across verbs, but a verb's value varies by only 0.07 across the 24 contexts (the barely trained model hardly conditions on the noun). Onsets are reliable too: r between halves 0.93 / 0.88 / 0.92 (1.4B / 410M / 6.9B), with 93 / 93 / 91% of verbs given the same censoring status by both halves.

**1. Onset.** At 1.4B the prediction held under the primary fit and under all three censored fits, which supports robustness to censoring under those fits' latent-onset assumptions (the synthetic test showed the crossers-only fit alone can read "held" under count equivalence); developmental gating and misspecification remain possible. Rare intransitives stop preferring "was" at much lower own exposure than frequent ones: median log10 E at onset Head 3.57, Tail 3.07, XTail 2.65 (an XTail verb crosses at about one eighth of a Head verb's own exposure), at median steps 310, 807 and 3,272. Two limits:
- The definition matters. Without the persistence condition the slope is +0.00 [−0.21, +0.17] (no frequency effect): 131 intransitives are already below 0 at step 64, and 89 of them are back at or above 0 within two checkpoints. Early values are stable verb-specific preferences of an untrained model (above), so a single early sign change is not an onset. Threshold −1 (−0.44 [−0.56, −0.28]) and the lenient end (−0.45) agree with the primary.
- At 410M the maturity-floor Tobit includes −1, so the 410M verdict is "held, not robust to censoring" and the onset claim does not replicate under the declared rule. 6.9B (6 steps, descriptive) held.

**2. Late window.** At 1.4B all four predictions held. After step 16000:
- The between-verb gradient is about A1's: β_b −0.75 [−0.98, −0.50].
- The average within-verb late slope is near 0: β_w −0.02 [−0.23, +0.18], with its 90% CI within ±0.25.
- So β_b − β_w = −0.73 [−1.04, −0.43]. The declared common-exposure spline implies a contrast of −0.00 [−0.05, +0.05], and the calibrated contrast stays below 0: a departure from that spline projection (not a test against every possible common curve).
- γ +0.03 [−0.14, +0.20] is unresolved: no sign that rarer verbs decline faster late.
- Read with disclosure 1: the β_w estimate follows from A6's flat late band means, so the β_w prediction was informed by seen aggregates; the per-verb contrast and its calibration are the new evidence.
- 410M: the main contrast held (−0.45 [−0.77, −0.11], calibrated −0.44), but 410M keeps declining late (β_w −0.53 [−0.77, −0.32]; "near 0" missed). 6.9B, with only 3 window steps: −0.18 [−0.49, +0.15] (β_w −0.57).
- Summary: late in training the gap between rare and frequent intransitives is not matched by what further exposure does within the same verbs. At 1.4B, the average late change is −0.02 nats per decade of training, while the between-verb count gradient is −0.75 nats per decade of count.

**3. Crossing.** The declared primary missed at every size: 1.4B and 410M have no resolved positive point, and 6.9B has insufficient overlap.
- At 1.4B, Q1 − Q5 is negative and resolved at log10 E 3.0–3.6 (about −1.2 nats). The sole positive point estimate is at the last shared grid point, 3.9 (+0.25 [−0.40, +0.93]), and its CI includes 0; the Q1 curve ends there (Q1's mean log10 c is 3.98), so higher exposures cannot be assessed. At 410M, D is still negative at the end of the grid (−0.43 [−1.11, +0.22] at 3.9); at 6.9B the overlap is a single grid point.
- With Zipf bands (a declared sensitivity, descriptive), XTail reaches higher exposure. XTail − Head shows the predicted-order crossing at 1.4B and 410M (crossing near log10 E 3.8–3.9; 1.4B: −1.46 [−1.88, −0.98] at 3.5, +1.78 [+1.21, +2.38] at 4.2). It survives the simultaneous band, the per-verb alignment and the common-curve calibration, but not the steps ≥ 512 sensitivity (next point).
- That crossing pits XTail verbs late in training (steps ≈ 7,400–117,000 across the grid) against Head verbs at steps ≈ 70–1,100 (crossing near Head step 420). Its negative side comes entirely from Head verbs before step 512: with steps ≥ 512, XTail − Head is only positive and Q1 − Q5 has one grid point. So it shows rare verbs ending above where frequent verbs were at the same exposure, against frequent verbs measured during construction learning. It does not compare matured models.
- Mature-step comparisons (steps ≥ 512) give predicted-order crossings in several adjacent bins pointwise (1.4B: Q2 − Q3, Q4 − Q5, XTail − Tail, Tail − Head). With per-verb alignment most of these lose their negative side; descriptive only.
- The reversed-order segments at low exposure (Q1 − Q2 below log10 E ≈ 2, Q3 − Q4 below ≈ 2.7) compare the rarer bin at steps ≈ 200–1,200 with the more frequent bin at steps 64–270. That is the general early rise of Δ_pass(I) (band means +0.03–0.06 at step 64, +0.25–0.29 at step 256; A6), and it disappears with steps ≥ 512.

**Check against A6.** A6's matched-exposure diagnostic is reproduced exactly at all three sizes (+0.080, +0.133, +0.179). With the bootstrap added, none of its CIs excludes 0 (1.4B +0.080 [−0.168, +0.356]). The pooled D̄ is −0.012 [−0.224, +0.204] at 1.4B, a paired gap of −0.091 [−0.183, −0.003] (410M: +0.042 vs +0.133; 6.9B: +0.185 vs +0.179).
- A post hoc decomposition (not declared; 1.4B; 410M similar) describes how exposure regions contribute to A6's diagnostic. A6's 226,550 matched cell pairs average +0.34 at log10 E < 2.5, −0.53 at 2.5–3.5 and +0.37 at ≥ 3.5, with 55% of the pairs in the last region. Pairs whose earlier cell is before step 512 average −0.36; the rest average +0.26.
- So A6's +0.08 combines positive low- and high-exposure regions with a negative middle region; it does not isolate any one bin comparison. D̄ gives the middle region slightly more weight (35.6% of its grid cells vs 32.0% of A6's pairs; figures from the outside review). The rest of the gap comes from the other differences between the two estimands: bin averaging versus verb pairs, A6's pairs within a bin, and A6's 4× step restriction.

**4. Class-level vs verb-specific (descriptive).**
- T vs I separation appears in band order. Separation step at 1.4B: Head 256, Tail 1000, XTail 3000. At 410M: 256, 1000, 2000. At 6.9B: 512, 512, 2000.
- XTail separation appears together with, and just before, the typical XTail intransitive's own onset: median onset 3,418 at 1.4B (410M 2,316; 6.9B 2,451); band-mean Δ_pass(I) < 0 from step 4000 (410M 4000, 6.9B 2000).
- These descriptive timings put XTail separation near, and just before, the median individual onset; they do not establish whether an earlier class-level template exists (separation and median onset are different statistics). XTail AUC never reaches Head's (1.4B final 0.85 [0.78, 0.89] vs 0.93 [0.87, 0.97]).
- The maximum a_V identity deviation is 6.3e-6 nats (6.9B); the single-token a_V AUCs equal the single-token Δ_pass AUCs, and their Head column matches A6's Head AUC.

**Caveats.**
- Matched exposure compares frequent verbs early with rare verbs late, so part 3 (and the onset slope) is an observational trajectory pattern confounded with model age.
- The censored onset fits assume a linear normal latent onset.
- Frequency also indexes stable lexical differences between verbs (part 2's β_b).
- The exposure proxy assumes each verb's occurrences are spread evenly through training (A6's stationarity diagnostic covers lowercase single-token forms only).

# Results by model (generated)

## pythia-1.4b

### Reliability (split-half, 200 animacy-stratified 12 / 12 context splits)

| step | Δ_pass(I) r_half | r_SB | flag | r_SB within Head / Tail / XTail | Δ_pass(T) r_SB |
|---|---|---|---|---|---|
| 64 | 1.00 | 1.00 |  | 1.00 / 1.00 / 1.00 | 1.00 |
| 128 | 1.00 | 1.00 |  | 1.00 / 1.00 / 1.00 | 1.00 |
| 256 | 1.00 | 1.00 |  | 1.00 / 1.00 / 1.00 | 1.00 |
| 512 | 1.00 | 1.00 |  | 1.00 / 1.00 / 1.00 | 1.00 |
| 1000 | 1.00 | 1.00 |  | 1.00 / 1.00 / 0.99 | 0.99 |
| 2000 | 0.99 | 1.00 |  | 1.00 / 0.99 / 0.99 | 0.99 |
| 3000 | 0.99 | 0.99 |  | 0.99 / 0.99 / 0.99 | 0.99 |
| 4000 | 0.98 | 0.99 |  | 0.99 / 0.99 / 0.98 | 0.98 |
| 6000 | 0.98 | 0.99 |  | 0.99 / 0.99 / 0.98 | 0.98 |
| 10000 | 0.98 | 0.99 |  | 0.99 / 0.99 / 0.99 | 0.97 |
| 16000 | 0.97 | 0.99 |  | 0.99 / 0.99 / 0.98 | 0.98 |
| 25000 | 0.97 | 0.99 |  | 0.99 / 0.98 / 0.98 | 0.97 |
| 40000 | 0.97 | 0.98 |  | 0.99 / 0.98 / 0.98 | 0.97 |
| 63000 | 0.97 | 0.99 |  | 0.99 / 0.98 / 0.98 | 0.97 |
| 100000 | 0.97 | 0.98 |  | 0.99 / 0.98 / 0.98 | 0.97 |
| 143000 | 0.97 | 0.98 |  | 0.99 / 0.98 / 0.98 | 0.97 |

Flagged checkpoints (r_SB < 0.7): none. Onset reliability (primary definition; r of log10 onset between halves over verbs crossing in both): 0.93 (mean over 200 splits with a finite r, 206 verbs on average); the two halves give the same censoring status for 93% of verbs.

### 1. Onset of Δ_pass(I) < 0

Counts (point estimate; Head / Tail / XTail). Right-censored verbs have no qualifying checkpoint up to the bound (their onset is later than the bound).

| definition | right-censoring bound | crossers Head / Tail / XTail | left-censored | right-censored | onsets in the last qualifying interval | interpolation across a † step | persistence uses a † step |
|---|---|---|---|---|---|---|---|
| primary | 63,000 | 37 / 107 / 71 | 8 / 21 / 13 | 2 / 4 / 18 | 3 | 0 | 0 |
| lenient end | 143,000 | 37 / 107 / 75 | 8 / 21 / 13 | 2 / 4 / 14 | 1 | 0 | 0 |
| no persistence | 143,000 | 26 / 69 / 53 | 21 / 63 / 47 | 0 / 0 / 2 | 0 | 0 | 0 |
| threshold −1 | 63,000 | 43 / 105 / 47 | 0 / 0 / 0 | 4 / 27 / 55 | 8 | 0 | 0 |

Verbs with Δ_pass(I) < 0 at the first step that are not left-censored (they return to ≥ 0 within the next two checkpoints): 89.

| fit | verbs (point) | β_on (95% CI) | label | valid draws |
|---|---|---|---|---|
| primary (crossers, + log n_tok) | 215 | -0.39 [-0.54, -0.28] | held | 2000 / 2000 |
| (d) no token covariate | 215 | -0.42 [-0.52, -0.35] | held | 2000 / 2000 |
| lenient end (persistence shortened at the last two steps) | 219 | -0.45 [-0.64, -0.31] | held | 2000 / 2000 |
| (a) no persistence | 148 | +0.00 [-0.21, +0.17] | no frequency effect on onset | 2000 / 2000 |
| (b) threshold −1 | 195 | -0.44 [-0.56, -0.28] | held | 2000 / 2000 |
| (c) Tobit (all verbs, + log n_tok) | 281 | -0.62 [-0.90, -0.37] | held | 2000 / 2000 |
| (e) Tobit, maturity floor (onsets ≤ 512 left-censored) | 281 | -0.59 [-0.86, -0.37] | held | 2000 / 2000 |
| (f) checkpoint-interval censored (all verbs, + log n_tok) | 281 | -0.62 [-0.91, -0.38] | held | 2000 / 2000 |

**Onset verdict (primary): held.** Count equivalence predicts β_on = −1; the prediction is −1 < β_on < 0. A CI that includes a value is not evidence that the slope equals it.

Onset exposure log10 E_onset (crossers, primary): Head (n = 37) median 3.57 [IQR 3.31, 3.82], onset step median 310; Tail (n = 107) median 3.07 [IQR 2.82, 3.40], onset step median 807; XTail (n = 71) median 2.65 [IQR 2.18, 2.89], onset step median 3,272. In the crossers sample and with the same covariates, the slope of log10 E_onset on log10 c_total is β_on + 1.

### 2. Late window (steps 16000, 25000, 40000, 63000, 100000, 143000)

Within a verb, log10 E varies only through log10 step (x is absorbed by the verb fixed effect), so β_w is the shared late time slope (nats per decade of training).

| quantity | estimate (95% CI) | prediction | reading |
|---|---|---|---|
| β_b (window mean on log10 c_total) | -0.75 [-0.98, -0.50] | near A1's −0.728 (CI excludes 0, includes −0.728) | held |
| β_b + log n_tok (sensitivity) | -1.05 [-1.39, -0.69] | | |
| β_w (verb fixed effects, on log10 E) | -0.02 [-0.23, +0.18]; 90% [-0.20, +0.15] | near 0 (90% CI within ±0.25) | held |
| **β_b − β_w** | -0.73 [-1.04, -0.43] | **< 0 (CI below 0)** | **held** |
| γ (log10 c × log10 step) | +0.03 [-0.14, +0.20] | unresolved | unresolved |
| time slope at mean log10 c | -0.02 [-0.22, +0.18] | | |
| calibration: β_b − β_w under the fitted common curve | -0.00 [-0.05, +0.05] | | |
| calibrated β_b − β_w (observed − common curve) | -0.73 [-1.05, -0.42] | | below 0 |
| calibrated γ (observed − common curve) | +0.04 [-0.30, +0.35] | | unresolved |

**Late-window verdict: held (also beyond the fitted common exposure curve).**

### 3. Exposure-aligned curves

Bin curves at the bin's inventory mean exposure (declared construction):

| bins | steps | comparison | grid (log10 E) | points | resolved D < 0 at | resolved D > 0 at | reading (pointwise) | simultaneous band | crossing at log10 E (point estimate) | grid points using a † step | common-curve calibrated (pointwise) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| quintile | all | Q1 − Q5 | 3.0 to 3.9 | 10 | 3.0–3.6 (7) | — | only negative where resolved | only negative where resolved | 3.85 | 0 | only negative where resolved |
| quintile | all | Q1 − Q2 | 1.3 to 3.9 | 27 | 2.6–2.8 (3) | 1.3–1.9 (7) | crossing (reversed order) | crossing (reversed order) | 2.03, 2.27, 2.43, 3.14 | 0 | crossing (reversed order) |
| quintile | all | Q2 − Q3 | 1.7 to 4.6 | 30 | 2.4–3.3 (5) | 1.9–4.4 (8) | crossing (non-monotonic: both orders) | crossing (non-monotonic: both orders) | 2.34, 2.76, 2.99, 3.48 | 0 | crossing (non-monotonic: both orders) |
| quintile | all | Q3 − Q4 | 2.2 to 5.0 | 29 | 2.9–3.1 (3) | 2.2–2.7 (6) | crossing (reversed order) | crossing (reversed order) | 2.75, 3.21, 3.44, 3.89 | 0 | crossing (reversed order) |
| quintile | all | Q4 − Q5 | 3.0 to 5.4 | 25 | 3.2–3.9 (8) | 3.0–5.1 (4) | crossing (non-monotonic: both orders) | crossing (reversed order) | 3.10, 4.09, 4.30, 4.43 | 0 | crossing (non-monotonic: both orders) |
| zipf | all | XTail − Head | 3.0 to 4.2 | 13 | 3.0–3.7 (8) | 4.0–4.2 (3) | crossing (predicted order) | crossing (predicted order) | 3.83 | 0 | crossing (predicted order) |
| zipf | all | XTail − Tail | 1.9 to 4.2 | 24 | 2.7–3.0 (4) | 1.9–4.2 (12) | crossing (non-monotonic: both orders) | crossing (non-monotonic: both orders) | 2.61, 3.09, 3.21, 3.54 | 0 | crossing (non-monotonic: both orders) |
| zipf | all | Tail − Head | 3.0 to 5.2 | 23 | 3.0–3.9 (10) | 4.2–5.2 (9) | crossing (predicted order) | crossing (predicted order) | 4.06 | 0 | crossing (predicted order) |
| quintile | ≥ 512 | Q1 − Q5 | 3.9 to 3.9 | 1 | — | — | insufficient overlap (one grid point) | — | — | 0 | insufficient overlap (one grid point) |
| quintile | ≥ 512 | Q1 − Q2 | 2.2 to 3.9 | 18 | 2.6–2.8 (3) | — | only negative where resolved | only negative where resolved | 2.27, 2.43, 3.14 | 0 | only negative where resolved |
| quintile | ≥ 512 | Q2 − Q3 | 2.7 to 4.6 | 20 | 3.2–3.3 (2) | 3.7–4.4 (4) | crossing (predicted order) | crossing (predicted order) | 2.76, 2.99, 3.48 | 0 | crossing (predicted order) |
| quintile | ≥ 512 | Q3 − Q4 | 3.1 to 5.0 | 20 | 3.1–3.1 (1) | — | only negative where resolved | none resolved | 3.21, 3.44, 3.89 | 0 | only negative where resolved |
| quintile | ≥ 512 | Q4 − Q5 | 3.9 to 5.4 | 16 | 3.9–3.9 (1) | 4.9–5.1 (3) | crossing (predicted order) | only negative where resolved | 4.09, 4.30, 4.43 | 0 | crossing (predicted order) |
| zipf | ≥ 512 | XTail − Head | 3.9 to 4.2 | 4 | — | 4.0–4.2 (3) | only positive where resolved | only positive where resolved | — | 0 | only positive where resolved |
| zipf | ≥ 512 | XTail − Tail | 2.9 to 4.2 | 14 | 2.9–3.0 (2) | 3.7–4.2 (6) | crossing (predicted order) | crossing (predicted order) | 3.09, 3.21, 3.54 | 0 | crossing (predicted order) |
| zipf | ≥ 512 | Tail − Head | 3.9 to 5.2 | 14 | 3.9–3.9 (1) | 4.2–5.2 (9) | crossing (predicted order) | crossing (predicted order) | 4.06 | 0 | crossing (predicted order) |

Per-verb alignment (each verb interpolated on its own log10 E; grid points covered by ≥ 50% of each bin's verbs; qualifier):

| bins | steps | comparison | grid (log10 E) | points | resolved D < 0 at | resolved D > 0 at | reading (pointwise) | simultaneous band | crossing at log10 E (point estimate) | grid points using a † step | common-curve calibrated (pointwise) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| quintile | all | Q1 − Q5 | 3.0 to 4.0 | 11 | 3.0–3.5 (6) | — | only negative where resolved | only negative where resolved | 3.76 | 0 | only negative where resolved |
| quintile | all | Q1 − Q2 | 1.4 to 4.0 | 27 | 2.8–2.9 (2) | 1.4–1.9 (6) | crossing (reversed order) | only positive where resolved | 2.07, 3.08 | 0 | crossing (reversed order) |
| quintile | all | Q2 − Q3 | 1.8 to 4.6 | 29 | 2.6–3.2 (3) | 1.9–4.4 (9) | crossing (non-monotonic: both orders) | only positive where resolved | 2.37, 3.48 | 0 | crossing (non-monotonic: both orders) |
| quintile | all | Q3 − Q4 | 2.1 to 5.0 | 30 | 2.9–3.0 (2) | 2.1–2.7 (7) | crossing (reversed order) | crossing (reversed order) | 2.76, 3.24, 3.31, 3.91 | 0 | crossing (reversed order) |
| quintile | all | Q4 − Q5 | 3.0 to 5.4 | 25 | 3.1–4.2 (12) | — | only negative where resolved | only negative where resolved | 3.01, 4.58 | 0 | only negative where resolved |
| zipf | all | XTail − Head | 3.0 to 4.2 | 13 | 3.0–3.5 (6) | 4.1–4.2 (2) | crossing (predicted order) | crossing (predicted order) | 3.80 | 0 | crossing (predicted order) |
| zipf | all | XTail − Tail | 1.9 to 4.2 | 24 | 2.8–3.0 (3) | 1.9–4.2 (10) | crossing (non-monotonic: both orders) | crossing (non-monotonic: both orders) | 2.35, 3.29 | 0 | crossing (non-monotonic: both orders) |
| zipf | all | Tail − Head | 3.0 to 5.2 | 23 | 3.0–4.0 (11) | 4.7–5.2 (6) | crossing (predicted order) | crossing (predicted order) | 4.30 | 0 | crossing (predicted order) |
| quintile | ≥ 512 | Q1 − Q5 | 3.9 to 4.0 | 2 | — | 3.9–4.0 (2) | only positive where resolved | only positive where resolved | — | 0 | only positive where resolved |
| quintile | ≥ 512 | Q1 − Q2 | 2.3 to 4.0 | 18 | 2.8–2.9 (2) | — | only negative where resolved | none resolved | 3.08 | 0 | only negative where resolved |
| quintile | ≥ 512 | Q2 − Q3 | 2.7 to 4.6 | 20 | 3.2–3.2 (1) | 4.0–4.4 (5) | crossing (predicted order) | none resolved | 3.48 | 0 | crossing (predicted order) |
| quintile | ≥ 512 | Q3 − Q4 | 3.0 to 5.0 | 21 | 3.0–3.0 (1) | — | only negative where resolved | none resolved | 3.17, 3.35, 3.91 | 0 | only negative where resolved |
| quintile | ≥ 512 | Q4 − Q5 | 3.9 to 5.4 | 16 | — | — | none resolved | none resolved | 4.56 | 0 | none resolved |
| zipf | ≥ 512 | XTail − Head | 3.9 to 4.2 | 4 | — | 3.9–4.2 (4) | only positive where resolved | only positive where resolved | — | 0 | only positive where resolved |
| zipf | ≥ 512 | XTail − Tail | 2.8 to 4.2 | 15 | — | 3.6–4.2 (7) | only positive where resolved | only positive where resolved | 3.21 | 0 | only positive where resolved |
| zipf | ≥ 512 | Tail − Head | 3.9 to 5.2 | 14 | — | 4.7–5.2 (6) | only positive where resolved | only positive where resolved | — | 0 | only positive where resolved |

Common-curve calibrated = D − D_null, where D_null passes a natural spline f̂(log10 E) fitted to all verbs and steps of that comparison's step set through the same sampling, binning and interpolation (qualifier).

**Primary (Q1 − Q5, all steps, bin curves): only negative where resolved; simultaneous band: only negative where resolved; per-verb alignment: only negative where resolved; common-curve calibrated: only negative where resolved → crossing verdict: missed.** Matched exposure here compares frequent verbs early in training with rare verbs late: an observational trajectory pattern, confounded with model age.

Check against A6: A6's matched-exposure diagnostic recomputed from the same scores (226550 cell pairs, |Δ log10 E| ≤ 0.1, steps ≥ 4× apart; later − earlier) +0.080 [-0.168, +0.356] (A6 reported +0.080: reproduced). Pooled D̄ over the 10 quintile pairs (bin curves, all steps, grid cells equally weighted) -0.012 [-0.224, +0.204]; difference D̄ − diagnostic -0.091 [-0.183, -0.003] (different weighting of the same comparison; see the plan).

### 4. Class-level vs verb-specific (descriptive)

| step | AUC T vs I Head / Tail / XTail | mean L Head / Tail / XTail | single-token a_V T / I Head | Tail | a_V AUC Head / Tail | identity max dev |
|---|---|---|---|---|---|---|
| 64 | 0.54 [0.43, 0.66] / 0.46 [0.39, 0.54] / 0.50 [0.42, 0.58] | +0.07 / -0.05 / +0.02 | +0.09 / +0.02 | -0.01 / -0.02 | 0.54 / 0.53 | 7.1e-07 |
| 128 | 0.60 [0.48, 0.71] / 0.46 [0.39, 0.53] / 0.52 [0.44, 0.59] | +0.10 / -0.06 / +0.02 | +0.14 / +0.02 | +0.13 / -0.02 | 0.63 / 0.66 | 7.4e-07 |
| 256 | 0.67 [0.56, 0.77] / 0.45 [0.38, 0.52] / 0.46 [0.39, 0.54] | +0.28 / -0.07 / -0.09 | +0.41 / +0.11 | +0.30 / +0.19 | 0.69 / 0.58 | 1.8e-06 |
| 512 | 0.88 [0.82, 0.94] / 0.52 [0.46, 0.59] / 0.50 [0.42, 0.58] | +1.78 / +0.06 / +0.02 | +0.38 / -1.48 | +0.53 / -0.23 | 0.90 / 0.69 | 1.2e-06 |
| 1000 | 0.95 [0.90, 0.99] / 0.75 [0.69, 0.80] / 0.55 [0.47, 0.62] | +3.19 / +1.06 / +0.17 | +0.12 / -3.15 | +0.04 / -1.91 | 0.95 / 0.90 | 1.6e-06 |
| 2000 | 0.94 [0.90, 0.98] / 0.81 [0.76, 0.86] / 0.57 [0.49, 0.65] | +3.23 / +1.57 / +0.33 | +0.42 / -2.82 | +0.04 / -1.76 | 0.94 / 0.84 | 2.9e-06 |
| 3000 | 0.95 [0.90, 0.98] / 0.84 [0.79, 0.89] / 0.63 [0.54, 0.71] | +3.49 / +2.01 / +0.59 | +0.73 / -2.75 | +0.33 / -1.76 | 0.94 / 0.90 | 3.5e-06 |
| 4000 | 0.93 [0.88, 0.97] / 0.87 [0.83, 0.91] / 0.69 [0.61, 0.76] | +3.32 / +2.21 / +0.94 | +0.34 / -2.98 | +0.17 / -1.96 | 0.93 / 0.91 | 4.0e-06 |
| 6000 | 0.94 [0.89, 0.98] / 0.88 [0.83, 0.92] / 0.80 [0.73, 0.86] | +3.33 / +2.32 / +1.52 | -0.02 / -3.38 | -0.31 / -2.44 | 0.93 / 0.94 | 4.2e-06 |
| 10000 | 0.95 [0.90, 0.98] / 0.90 [0.86, 0.94] / 0.77 [0.69, 0.83] | +3.30 / +2.64 / +1.42 | +0.46 / -2.89 | +0.27 / -1.85 | 0.95 / 0.96 | 2.4e-06 |
| 16000 | 0.92 [0.87, 0.97] / 0.88 [0.84, 0.91] / 0.82 [0.75, 0.88] | +3.07 / +2.44 / +1.70 | +0.16 / -3.00 | +0.01 / -1.93 | 0.92 / 0.93 | 1.9e-06 |
| 25000 | 0.94 [0.88, 0.98] / 0.89 [0.85, 0.93] / 0.84 [0.78, 0.89] | +3.12 / +2.47 / +1.90 | +0.13 / -3.06 | -0.15 / -1.97 | 0.94 / 0.94 | 2.5e-06 |
| 40000 | 0.93 [0.87, 0.97] / 0.89 [0.85, 0.92] / 0.85 [0.78, 0.90] | +3.20 / +2.48 / +1.87 | +0.31 / -2.93 | +0.13 / -1.70 | 0.93 / 0.94 | 2.5e-06 |
| 63000 | 0.93 [0.88, 0.97] / 0.87 [0.83, 0.91] / 0.79 [0.73, 0.86] | +3.24 / +2.34 / +1.61 | +0.34 / -2.95 | +0.12 / -1.73 | 0.93 / 0.92 | 1.6e-06 |
| 100000 | 0.92 [0.86, 0.97] / 0.89 [0.85, 0.92] / 0.83 [0.77, 0.88] | +3.16 / +2.49 / +1.88 | -0.05 / -3.30 | -0.26 / -2.06 | 0.92 / 0.90 | 3.0e-06 |
| 143000 | 0.93 [0.87, 0.97] / 0.89 [0.85, 0.93] / 0.85 [0.78, 0.89] | +3.37 / +2.56 / +1.96 | -0.01 / -3.45 | -0.33 / -2.20 | 0.93 / 0.90 | 3.6e-06 |

Separation step (AUC CI above 0.5 at every later checkpoint): Head 256, Tail 1000, XTail 3000. XTail I onset: median primary onset 3,418; band-mean Δ_pass(I) < 0 from step 4000. a_V identity (single-token a_V = Δ_pass + per-checkpoint constant): max deviation 4.2e-06 nats.

## pythia-410m

### Reliability (split-half, 200 animacy-stratified 12 / 12 context splits)

| step | Δ_pass(I) r_half | r_SB | flag | r_SB within Head / Tail / XTail | Δ_pass(T) r_SB |
|---|---|---|---|---|---|
| 64 | 1.00 | 1.00 |  | 1.00 / 1.00 / 1.00 | 1.00 |
| 128 | 1.00 | 1.00 |  | 1.00 / 1.00 / 1.00 | 1.00 |
| 256 | 1.00 | 1.00 |  | 1.00 / 1.00 / 1.00 | 1.00 |
| 512 | 0.99 | 1.00 |  | 1.00 / 1.00 / 1.00 | 1.00 |
| 1000 | 0.99 | 1.00 |  | 1.00 / 1.00 / 0.99 | 0.99 |
| 2000 | 0.99 | 0.99 |  | 0.99 / 0.99 / 0.99 | 0.99 |
| 3000 | 0.99 | 0.99 |  | 0.99 / 0.99 / 0.98 | 0.98 |
| 4000 | 0.98 | 0.99 |  | 0.99 / 0.99 / 0.98 | 0.98 |
| 6000 | 0.98 | 0.99 |  | 0.99 / 0.99 / 0.98 | 0.98 |
| 10000 | 0.98 | 0.99 |  | 0.99 / 0.99 / 0.98 | 0.98 |
| 16000 | 0.98 | 0.99 |  | 0.99 / 0.99 / 0.98 | 0.98 |
| 25000 | 0.98 | 0.99 |  | 0.99 / 0.98 / 0.99 | 0.98 |
| 40000 | 0.98 | 0.99 |  | 0.99 / 0.99 / 0.98 | 0.98 |
| 63000 | 0.97 | 0.99 |  | 0.99 / 0.98 / 0.98 | 0.97 |
| 100000 | 0.97 | 0.99 |  | 0.99 / 0.98 / 0.98 | 0.97 |
| 143000 | 0.97 | 0.99 |  | 0.99 / 0.98 / 0.98 | 0.97 |

Flagged checkpoints (r_SB < 0.7): none. Onset reliability (primary definition; r of log10 onset between halves over verbs crossing in both): 0.88 (mean over 200 splits with a finite r, 200 verbs on average); the two halves give the same censoring status for 93% of verbs.

### 1. Onset of Δ_pass(I) < 0

Counts (point estimate; Head / Tail / XTail). Right-censored verbs have no qualifying checkpoint up to the bound (their onset is later than the bound).

| definition | right-censoring bound | crossers Head / Tail / XTail | left-censored | right-censored | onsets in the last qualifying interval | interpolation across a † step | persistence uses a † step |
|---|---|---|---|---|---|---|---|
| primary | 63,000 | 38 / 111 / 63 | 8 / 15 / 20 | 1 / 6 / 19 | 6 | 0 | 0 |
| lenient end | 143,000 | 39 / 113 / 71 | 8 / 15 / 20 | 0 / 4 / 11 | 1 | 0 | 0 |
| no persistence | 143,000 | 22 / 73 / 54 | 25 / 59 / 47 | 0 / 0 / 1 | 1 | 0 | 0 |
| threshold −1 | 63,000 | 42 / 106 / 43 | 0 / 0 / 0 | 5 / 26 / 59 | 14 | 0 | 0 |

Verbs with Δ_pass(I) < 0 at the first step that are not left-censored (they return to ≥ 0 within the next two checkpoints): 88.

| fit | verbs (point) | β_on (95% CI) | label | valid draws |
|---|---|---|---|---|
| primary (crossers, + log n_tok) | 212 | -0.19 [-0.39, -0.07] | held | 2000 / 2000 |
| (d) no token covariate | 212 | -0.33 [-0.46, -0.26] | held | 2000 / 2000 |
| lenient end (persistence shortened at the last two steps) | 223 | -0.25 [-0.54, -0.13] | held | 2000 / 2000 |
| (a) no persistence | 149 | -0.14 [-0.41, +0.07] | no frequency effect on onset | 2000 / 2000 |
| (b) threshold −1 | 191 | -0.40 [-0.56, -0.25] | held | 2000 / 2000 |
| (c) Tobit (all verbs, + log n_tok) | 281 | -0.57 [-0.97, -0.29] | held | 2000 / 2000 |
| (e) Tobit, maturity floor (onsets ≤ 512 left-censored) | 281 | -0.64 [-1.05, -0.38] | failed (consistent with count equivalence) | 2000 / 2000 |
| (f) checkpoint-interval censored (all verbs, + log n_tok) | 281 | -0.57 [-0.97, -0.29] | held | 2000 / 2000 |

**Onset verdict (primary): held, not robust to censoring (maturity-floor Tobit not 'held').** Count equivalence predicts β_on = −1; the prediction is −1 < β_on < 0. A CI that includes a value is not evidence that the slope equals it.

Onset exposure log10 E_onset (crossers, primary): Head (n = 38) median 3.80 [IQR 3.46, 4.06], onset step median 383; Tail (n = 111) median 3.05 [IQR 2.78, 3.40], onset step median 813; XTail (n = 63) median 2.44 [IQR 2.17, 3.22], onset step median 2,320. In the crossers sample and with the same covariates, the slope of log10 E_onset on log10 c_total is β_on + 1.

### 2. Late window (steps 16000, 25000, 40000, 63000, 100000, 143000)

Within a verb, log10 E varies only through log10 step (x is absorbed by the verb fixed effect), so β_w is the shared late time slope (nats per decade of training).

| quantity | estimate (95% CI) | prediction | reading |
|---|---|---|---|
| β_b (window mean on log10 c_total) | -0.98 [-1.23, -0.73] | near A1's −0.728 (CI excludes 0, includes −0.728) | held |
| β_b + log n_tok (sensitivity) | -1.16 [-1.52, -0.80] | | |
| β_w (verb fixed effects, on log10 E) | -0.53 [-0.77, -0.32]; 90% [-0.73, -0.35] | near 0 (90% CI within ±0.25) | missed |
| **β_b − β_w** | -0.45 [-0.77, -0.11] | **< 0 (CI below 0)** | **held** |
| γ (log10 c × log10 step) | +0.10 [-0.05, +0.26] | unresolved | unresolved |
| time slope at mean log10 c | -0.53 [-0.77, -0.32] | | |
| calibration: β_b − β_w under the fitted common curve | -0.00 [-0.05, +0.04] | | |
| calibrated β_b − β_w (observed − common curve) | -0.44 [-0.77, -0.09] | | below 0 |
| calibrated γ (observed − common curve) | +0.12 [-0.23, +0.47] | | unresolved |

**Late-window verdict: held (also beyond the fitted common exposure curve).**

### 3. Exposure-aligned curves

Bin curves at the bin's inventory mean exposure (declared construction):

| bins | steps | comparison | grid (log10 E) | points | resolved D < 0 at | resolved D > 0 at | reading (pointwise) | simultaneous band | crossing at log10 E (point estimate) | grid points using a † step | common-curve calibrated (pointwise) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| quintile | all | Q1 − Q5 | 3.0 to 3.9 | 10 | 3.0–3.8 (9) | — | only negative where resolved | only negative where resolved | — | 0 | only negative where resolved |
| quintile | all | Q1 − Q2 | 1.3 to 3.9 | 27 | 2.4–2.6 (3) | 1.3–1.9 (7) | crossing (reversed order) | crossing (reversed order) | 2.00, 3.04, 3.73 | 0 | crossing (reversed order) |
| quintile | all | Q2 − Q3 | 1.7 to 4.6 | 30 | 2.7–2.8 (2) | 1.8–4.3 (15) | crossing (non-monotonic: both orders) | only positive where resolved | 2.54, 3.17, 4.50, 4.50 | 0 | crossing (non-monotonic: both orders) |
| quintile | all | Q3 − Q4 | 2.2 to 5.0 | 29 | 3.0–3.2 (3) | 2.4–4.6 (5) | crossing (non-monotonic: both orders) | only negative where resolved | 2.83, 3.83, 4.85, 4.98 | 0 | crossing (non-monotonic: both orders) |
| quintile | all | Q4 − Q5 | 3.0 to 5.4 | 25 | 3.2–4.0 (9) | 3.0–4.4 (2) | crossing (non-monotonic: both orders) | crossing (reversed order) | 3.11, 4.09, 5.29, 5.31 | 0 | crossing (non-monotonic: both orders) |
| zipf | all | XTail − Head | 3.0 to 4.2 | 13 | 3.0–3.7 (8) | 4.0–4.2 (3) | crossing (predicted order) | crossing (predicted order) | 3.89 | 0 | crossing (predicted order) |
| zipf | all | XTail − Tail | 1.9 to 4.2 | 24 | 2.7–3.0 (4) | 1.9–4.2 (14) | crossing (non-monotonic: both orders) | crossing (non-monotonic: both orders) | 2.40, 3.17 | 0 | crossing (non-monotonic: both orders) |
| zipf | all | Tail − Head | 3.0 to 5.2 | 23 | 3.1–4.0 (10) | 4.2–5.0 (9) | crossing (predicted order) | crossing (predicted order) | 4.08 | 0 | crossing (predicted order) |
| quintile | ≥ 512 | Q1 − Q5 | 3.9 to 3.9 | 1 | — | — | insufficient overlap (one grid point) | — | — | 0 | insufficient overlap (one grid point) |
| quintile | ≥ 512 | Q1 − Q2 | 2.2 to 3.9 | 18 | 2.4–2.6 (3) | — | only negative where resolved | only negative where resolved | 3.04, 3.73 | 0 | only negative where resolved |
| quintile | ≥ 512 | Q2 − Q3 | 2.7 to 4.6 | 20 | 2.7–2.8 (2) | 3.6–4.3 (8) | crossing (predicted order) | only positive where resolved | 3.17, 4.50, 4.50 | 0 | crossing (predicted order) |
| quintile | ≥ 512 | Q3 − Q4 | 3.1 to 5.0 | 20 | 3.1–3.2 (2) | 4.5–4.6 (2) | crossing (predicted order) | crossing (predicted order) | 3.83, 4.85, 4.98 | 0 | crossing (predicted order) |
| quintile | ≥ 512 | Q4 − Q5 | 3.9 to 5.4 | 16 | 3.9–4.0 (2) | 4.4–4.4 (1) | crossing (predicted order) | only negative where resolved | 4.09, 5.29, 5.31 | 0 | crossing (predicted order) |
| zipf | ≥ 512 | XTail − Head | 3.9 to 4.2 | 4 | — | 4.0–4.2 (3) | only positive where resolved | only positive where resolved | — | 0 | only positive where resolved |
| zipf | ≥ 512 | XTail − Tail | 2.9 to 4.2 | 14 | 2.9–3.0 (2) | 3.3–4.2 (10) | crossing (predicted order) | crossing (predicted order) | 3.17 | 0 | crossing (predicted order) |
| zipf | ≥ 512 | Tail − Head | 3.9 to 5.2 | 14 | 3.9–4.0 (2) | 4.2–5.0 (9) | crossing (predicted order) | crossing (predicted order) | 4.08 | 0 | crossing (predicted order) |

Per-verb alignment (each verb interpolated on its own log10 E; grid points covered by ≥ 50% of each bin's verbs; qualifier):

| bins | steps | comparison | grid (log10 E) | points | resolved D < 0 at | resolved D > 0 at | reading (pointwise) | simultaneous band | crossing at log10 E (point estimate) | grid points using a † step | common-curve calibrated (pointwise) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| quintile | all | Q1 − Q5 | 3.0 to 4.0 | 11 | 3.0–3.5 (6) | — | only negative where resolved | only negative where resolved | 3.92 | 0 | only negative where resolved |
| quintile | all | Q1 − Q2 | 1.4 to 4.0 | 27 | 2.3–2.7 (5) | 1.4–1.9 (6) | crossing (reversed order) | crossing (reversed order) | 2.12, 3.13, 3.83 | 0 | crossing (reversed order) |
| quintile | all | Q2 − Q3 | 1.8 to 4.6 | 29 | 2.8–2.9 (2) | 1.9–4.3 (12) | crossing (non-monotonic: both orders) | only positive where resolved | 2.55, 3.15 | 0 | crossing (non-monotonic: both orders) |
| quintile | all | Q3 − Q4 | 2.1 to 5.0 | 30 | 3.0–3.3 (4) | 2.3–4.6 (8) | crossing (non-monotonic: both orders) | only positive where resolved | 2.82, 3.71, 4.96 | 0 | crossing (non-monotonic: both orders) |
| quintile | all | Q4 − Q5 | 3.0 to 5.4 | 25 | 3.2–4.0 (9) | — | only negative where resolved | only negative where resolved | 3.07, 4.43 | 0 | only negative where resolved |
| zipf | all | XTail − Head | 3.0 to 4.2 | 13 | 3.0–3.5 (6) | 4.0–4.2 (3) | crossing (predicted order) | crossing (predicted order) | 3.79 | 0 | crossing (predicted order) |
| zipf | all | XTail − Tail | 1.9 to 4.2 | 24 | 2.9–2.9 (1) | 1.9–4.2 (12) | crossing (non-monotonic: both orders) | only positive where resolved | 2.31, 3.15 | 0 | crossing (non-monotonic: both orders) |
| zipf | all | Tail − Head | 3.0 to 5.2 | 23 | 3.1–3.9 (9) | 4.6–4.8 (3) | crossing (predicted order) | only negative where resolved | 4.18 | 0 | crossing (predicted order) |
| quintile | ≥ 512 | Q1 − Q5 | 3.9 to 4.0 | 2 | — | 4.0–4.0 (1) | only positive where resolved | only positive where resolved | — | 0 | only positive where resolved |
| quintile | ≥ 512 | Q1 − Q2 | 2.3 to 4.0 | 18 | 2.4–2.7 (4) | — | only negative where resolved | only negative where resolved | 3.13, 3.83 | 0 | only negative where resolved |
| quintile | ≥ 512 | Q2 − Q3 | 2.7 to 4.6 | 20 | 2.8–2.9 (2) | 3.7–4.3 (7) | crossing (predicted order) | only positive where resolved | 3.15 | 0 | crossing (predicted order) |
| quintile | ≥ 512 | Q3 − Q4 | 3.0 to 5.0 | 21 | 3.1–3.3 (3) | 4.3–4.6 (4) | crossing (predicted order) | none resolved | 3.71, 4.96 | 0 | crossing (predicted order) |
| quintile | ≥ 512 | Q4 − Q5 | 3.9 to 5.4 | 16 | — | — | none resolved | none resolved | 3.98, 4.16, 4.22 | 0 | none resolved |
| zipf | ≥ 512 | XTail − Head | 3.9 to 4.2 | 4 | — | 3.9–4.2 (4) | only positive where resolved | only positive where resolved | — | 0 | only positive where resolved |
| zipf | ≥ 512 | XTail − Tail | 2.8 to 4.2 | 15 | — | 3.3–4.2 (10) | only positive where resolved | only positive where resolved | 3.11 | 0 | only positive where resolved |
| zipf | ≥ 512 | Tail − Head | 3.9 to 5.2 | 14 | — | 4.4–4.8 (5) | only positive where resolved | only positive where resolved | — | 0 | only positive where resolved |

Common-curve calibrated = D − D_null, where D_null passes a natural spline f̂(log10 E) fitted to all verbs and steps of that comparison's step set through the same sampling, binning and interpolation (qualifier).

**Primary (Q1 − Q5, all steps, bin curves): only negative where resolved; simultaneous band: only negative where resolved; per-verb alignment: only negative where resolved; common-curve calibrated: only negative where resolved → crossing verdict: missed.** Matched exposure here compares frequent verbs early in training with rare verbs late: an observational trajectory pattern, confounded with model age.

Check against A6: A6's matched-exposure diagnostic recomputed from the same scores (226550 cell pairs, |Δ log10 E| ≤ 0.1, steps ≥ 4× apart; later − earlier) +0.133 [-0.116, +0.380] (A6 reported +0.133: reproduced). Pooled D̄ over the 10 quintile pairs (bin curves, all steps, grid cells equally weighted) +0.042 [-0.142, +0.239]; difference D̄ − diagnostic -0.091 [-0.183, +0.002] (different weighting of the same comparison; see the plan).

### 4. Class-level vs verb-specific (descriptive)

| step | AUC T vs I Head / Tail / XTail | mean L Head / Tail / XTail | single-token a_V T / I Head | Tail | a_V AUC Head / Tail | identity max dev |
|---|---|---|---|---|---|---|
| 64 | 0.51 [0.39, 0.62] / 0.44 [0.36, 0.51] / 0.49 [0.41, 0.57] | +0.03 / -0.07 / -0.01 | +0.04 / +0.01 | -0.04 / +0.06 | 0.50 / 0.40 | 4.1e-07 |
| 128 | 0.55 [0.42, 0.67] / 0.43 [0.36, 0.50] / 0.48 [0.39, 0.56] | +0.07 / -0.07 / -0.02 | +0.03 / -0.03 | -0.10 / +0.08 | 0.54 / 0.31 | 5.3e-07 |
| 256 | 0.62 [0.50, 0.74] / 0.46 [0.39, 0.54] / 0.55 [0.47, 0.63] | +0.18 / -0.04 / +0.07 | +0.33 / +0.16 | +0.23 / +0.43 | 0.63 / 0.33 | 8.6e-07 |
| 512 | 0.87 [0.80, 0.93] / 0.52 [0.45, 0.60] / 0.61 [0.52, 0.68] | +1.35 / +0.07 / +0.30 | +0.84 / -0.57 | +0.97 / +0.30 | 0.88 / 0.72 | 8.3e-07 |
| 1000 | 0.93 [0.87, 0.98] / 0.66 [0.59, 0.72] / 0.55 [0.48, 0.63] | +2.58 / +0.67 / +0.17 | -0.35 / -3.01 | +0.11 / -1.62 | 0.94 / 0.92 | 1.2e-06 |
| 2000 | 0.95 [0.90, 0.98] / 0.84 [0.79, 0.89] / 0.60 [0.53, 0.66] | +3.32 / +1.94 / +0.36 | -0.00 / -3.35 | -0.26 / -2.44 | 0.95 / 0.91 | 1.3e-06 |
| 3000 | 0.94 [0.89, 0.98] / 0.86 [0.81, 0.90] / 0.66 [0.58, 0.73] | +3.47 / +2.05 / +0.67 | +0.39 / -3.10 | +0.12 / -2.14 | 0.94 / 0.93 | 1.1e-06 |
| 4000 | 0.93 [0.87, 0.97] / 0.86 [0.82, 0.90] / 0.66 [0.58, 0.73] | +3.01 / +1.94 / +0.77 | -0.07 / -3.11 | -0.21 / -2.47 | 0.93 / 0.94 | 1.4e-06 |
| 6000 | 0.93 [0.88, 0.98] / 0.90 [0.86, 0.93] / 0.73 [0.66, 0.80] | +3.22 / +2.30 / +0.98 | +0.30 / -2.97 | +0.03 / -2.33 | 0.93 / 0.97 | 1.2e-06 |
| 10000 | 0.93 [0.88, 0.97] / 0.90 [0.87, 0.94] / 0.78 [0.70, 0.84] | +3.09 / +2.44 / +1.30 | +0.47 / -2.70 | +0.33 / -1.96 | 0.94 / 0.96 | 1.8e-06 |
| 16000 | 0.93 [0.88, 0.97] / 0.87 [0.82, 0.91] / 0.75 [0.68, 0.81] | +3.23 / +2.33 / +1.25 | +0.42 / -2.88 | +0.11 / -1.95 | 0.93 / 0.92 | 1.1e-06 |
| 25000 | 0.93 [0.88, 0.97] / 0.88 [0.84, 0.92] / 0.73 [0.66, 0.81] | +3.15 / +2.34 / +1.34 | +0.14 / -3.07 | -0.18 / -2.04 | 0.94 / 0.86 | 1.4e-06 |
| 40000 | 0.94 [0.89, 0.98] / 0.87 [0.83, 0.91] / 0.77 [0.69, 0.84] | +3.18 / +2.23 / +1.49 | +0.29 / -2.91 | +0.09 / -1.79 | 0.94 / 0.89 | 1.7e-06 |
| 63000 | 0.94 [0.89, 0.97] / 0.88 [0.84, 0.92] / 0.78 [0.71, 0.84] | +3.18 / +2.40 / +1.52 | +0.00 / -3.26 | -0.14 / -2.06 | 0.94 / 0.90 | 1.3e-06 |
| 100000 | 0.94 [0.89, 0.98] / 0.91 [0.87, 0.94] / 0.82 [0.75, 0.87] | +3.31 / +2.71 / +1.79 | -0.52 / -3.92 | -0.70 / -2.75 | 0.94 / 0.90 | 2.1e-06 |
| 143000 | 0.94 [0.90, 0.98] / 0.90 [0.87, 0.93] / 0.82 [0.75, 0.87] | +3.37 / +2.67 / +1.72 | -0.40 / -3.85 | -0.61 / -2.67 | 0.95 / 0.90 | 2.4e-06 |

Separation step (AUC CI above 0.5 at every later checkpoint): Head 256, Tail 1000, XTail 2000. XTail I onset: median primary onset 2,316; band-mean Δ_pass(I) < 0 from step 4000. a_V identity (single-token a_V = Δ_pass + per-checkpoint constant): max deviation 2.4e-06 nats.

## pythia-6.9b

### Reliability (split-half, 200 animacy-stratified 12 / 12 context splits)

| step | Δ_pass(I) r_half | r_SB | flag | r_SB within Head / Tail / XTail | Δ_pass(T) r_SB |
|---|---|---|---|---|---|
| 512 | 0.99 | 1.00 |  | 1.00 / 1.00 / 0.99 | 1.00 |
| 2000 | 0.99 | 1.00 |  | 0.99 / 0.99 / 0.99 | 0.99 |
| 8000 | 0.98 | 0.99 |  | 0.99 / 0.99 / 0.99 | 0.98 |
| 23000 | 0.97 | 0.99 |  | 0.99 / 0.98 / 0.98 | 0.98 |
| 63000 | 0.97 | 0.98 |  | 0.99 / 0.98 / 0.97 | 0.97 |
| 143000 | 0.97 | 0.99 |  | 0.99 / 0.98 / 0.98 | 0.97 |

Flagged checkpoints (r_SB < 0.7): none. Onset reliability (primary definition; r of log10 onset between halves over verbs crossing in both): 0.92 (mean over 200 splits with a finite r, 139 verbs on average); the two halves give the same censoring status for 91% of verbs.

### 1. Onset of Δ_pass(I) < 0

Counts (point estimate; Head / Tail / XTail). Right-censored verbs have no qualifying checkpoint up to the bound (their onset is later than the bound).

| definition | right-censoring bound | crossers Head / Tail / XTail | left-censored | right-censored | onsets in the last qualifying interval | interpolation across a † step | persistence uses a † step |
|---|---|---|---|---|---|---|---|
| primary | 23,000 | 13 / 81 / 53 | 32 / 41 / 18 | 2 / 10 / 31 | 14 | 0 | 0 |
| lenient end | 143,000 | 13 / 85 / 72 | 32 / 41 / 18 | 2 / 6 / 12 | 8 | 0 | 0 |
| no persistence | 143,000 | 15 / 83 / 67 | 32 / 47 / 32 | 0 / 2 / 3 | 0 | 0 | 0 |
| threshold −1 | 23,000 | 27 / 85 / 37 | 13 / 9 / 0 | 7 / 38 / 65 | 8 | 0 | 0 |

Verbs with Δ_pass(I) < 0 at the first step that are not left-censored (they return to ≥ 0 within the next two checkpoints): 20.

| fit | verbs (point) | β_on (95% CI) | label | valid draws |
|---|---|---|---|---|
| primary (crossers, + log n_tok) | 147 | -0.17 [-0.29, -0.00] | held | 2000 / 2000 |
| (d) no token covariate | 147 | -0.23 [-0.30, -0.13] | held | 2000 / 2000 |
| lenient end (persistence shortened at the last two steps) | 170 | -0.35 [-0.53, -0.08] | held | 2000 / 2000 |
| (a) no persistence | 165 | -0.20 [-0.33, -0.02] | held | 2000 / 2000 |
| (b) threshold −1 | 149 | -0.27 [-0.39, -0.12] | held | 2000 / 2000 |
| (c) Tobit (all verbs, + log n_tok) | 281 | -0.58 [-0.87, -0.29] | held | 2000 / 2000 |
| (e) Tobit, maturity floor (onsets ≤ 512 left-censored) | 281 | -0.58 [-0.87, -0.29] | held | 2000 / 2000 |
| (f) checkpoint-interval censored (all verbs, + log n_tok) | 281 | -0.58 [-0.88, -0.28] | held | 2000 / 2000 |

**Onset verdict (primary): held.** Count equivalence predicts β_on = −1; the prediction is −1 < β_on < 0. A CI that includes a value is not evidence that the slope equals it.

Onset exposure log10 E_onset (crossers, primary): Head (n = 13) median 3.66 [IQR 3.47, 4.14], onset step median 570; Tail (n = 81) median 3.09 [IQR 2.82, 3.43], onset step median 839; XTail (n = 53) median 2.54 [IQR 2.00, 2.95], onset step median 1,439. In the crossers sample and with the same covariates, the slope of log10 E_onset on log10 c_total is β_on + 1.

### 2. Late window (steps 23000, 63000, 143000)

Within a verb, log10 E varies only through log10 step (x is absorbed by the verb fixed effect), so β_w is the shared late time slope (nats per decade of training).

| quantity | estimate (95% CI) | prediction | reading |
|---|---|---|---|
| β_b (window mean on log10 c_total) | -0.75 [-0.99, -0.49] | near A1's −0.728 (CI excludes 0, includes −0.728) | held |
| β_b + log n_tok (sensitivity) | -1.10 [-1.44, -0.75] | | |
| β_w (verb fixed effects, on log10 E) | -0.57 [-0.79, -0.36]; 90% [-0.75, -0.40] | near 0 (90% CI within ±0.25) | missed |
| **β_b − β_w** | -0.18 [-0.49, +0.15] | **< 0 (CI below 0)** | **failed (consistent with count equivalence)** |
| γ (log10 c × log10 step) | -0.01 [-0.15, +0.13] | unresolved | unresolved |
| time slope at mean log10 c | -0.57 [-0.79, -0.36] | | |
| calibration: β_b − β_w under the fitted common curve | +0.01 [-0.04, +0.06] | | |
| calibrated β_b − β_w (observed − common curve) | -0.19 [-0.50, +0.14] | | includes 0 |
| calibrated γ (observed − common curve) | -0.06 [-0.39, +0.28] | | unresolved |

**Late-window verdict: failed (consistent with count equivalence).**

### 3. Exposure-aligned curves

Bin curves at the bin's inventory mean exposure (declared construction):

| bins | steps | comparison | grid (log10 E) | points | resolved D < 0 at | resolved D > 0 at | reading (pointwise) | simultaneous band | crossing at log10 E (point estimate) | grid points using a † step | common-curve calibrated (pointwise) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| quintile | all | Q1 − Q5 | 3.9 to 3.9 | 1 | 3.9–3.9 (1) | — | insufficient overlap (one grid point) | — | — | 0 | insufficient overlap (one grid point) |
| quintile | all | Q1 − Q2 | 2.2 to 3.9 | 18 | 2.2–2.6 (5) | — | only negative where resolved | only negative where resolved | 3.01 | 0 | only negative where resolved |
| quintile | all | Q2 − Q3 | 2.7 to 4.6 | 20 | — | 3.8–4.0 (3) | only positive where resolved | none resolved | 3.02 | 0 | only positive where resolved |
| quintile | all | Q3 − Q4 | 3.1 to 5.0 | 20 | 3.1–3.3 (3) | — | only negative where resolved | only negative where resolved | 3.57, 4.69, 4.73 | 0 | only negative where resolved |
| quintile | all | Q4 − Q5 | 3.9 to 5.4 | 16 | 3.9–4.1 (3) | 4.4–4.9 (6) | crossing (predicted order) | crossing (predicted order) | 4.26 | 0 | crossing (predicted order) |
| zipf | all | XTail − Head | 3.9 to 4.2 | 4 | 3.9–3.9 (1) | 4.2–4.2 (1) | crossing (predicted order) | only positive where resolved | 4.01 | 0 | crossing (predicted order) |
| zipf | all | XTail − Tail | 2.9 to 4.2 | 14 | 2.9–3.0 (2) | 3.4–4.2 (9) | crossing (predicted order) | crossing (predicted order) | 3.21 | 0 | crossing (predicted order) |
| zipf | all | Tail − Head | 3.9 to 5.2 | 14 | 3.9–4.1 (3) | 4.3–5.1 (9) | crossing (predicted order) | crossing (predicted order) | 4.20 | 0 | crossing (predicted order) |
| quintile | ≥ 512 | Q1 − Q5 | 3.9 to 3.9 | 1 | 3.9–3.9 (1) | — | insufficient overlap (one grid point) | — | — | 0 | insufficient overlap (one grid point) |
| quintile | ≥ 512 | Q1 − Q2 | 2.2 to 3.9 | 18 | 2.2–2.6 (5) | — | only negative where resolved | only negative where resolved | 3.01 | 0 | only negative where resolved |
| quintile | ≥ 512 | Q2 − Q3 | 2.7 to 4.6 | 20 | — | 3.8–4.0 (3) | only positive where resolved | none resolved | 3.02 | 0 | only positive where resolved |
| quintile | ≥ 512 | Q3 − Q4 | 3.1 to 5.0 | 20 | 3.1–3.3 (3) | — | only negative where resolved | only negative where resolved | 3.57, 4.69, 4.73 | 0 | only negative where resolved |
| quintile | ≥ 512 | Q4 − Q5 | 3.9 to 5.4 | 16 | 3.9–4.1 (3) | 4.4–4.9 (6) | crossing (predicted order) | crossing (predicted order) | 4.26 | 0 | crossing (predicted order) |
| zipf | ≥ 512 | XTail − Head | 3.9 to 4.2 | 4 | 3.9–3.9 (1) | 4.2–4.2 (1) | crossing (predicted order) | only positive where resolved | 4.01 | 0 | crossing (predicted order) |
| zipf | ≥ 512 | XTail − Tail | 2.9 to 4.2 | 14 | 2.9–3.0 (2) | 3.4–4.2 (9) | crossing (predicted order) | crossing (predicted order) | 3.21 | 0 | crossing (predicted order) |
| zipf | ≥ 512 | Tail − Head | 3.9 to 5.2 | 14 | 3.9–4.1 (3) | 4.3–5.1 (9) | crossing (predicted order) | crossing (predicted order) | 4.20 | 0 | crossing (predicted order) |

Per-verb alignment (each verb interpolated on its own log10 E; grid points covered by ≥ 50% of each bin's verbs; qualifier):

| bins | steps | comparison | grid (log10 E) | points | resolved D < 0 at | resolved D > 0 at | reading (pointwise) | simultaneous band | crossing at log10 E (point estimate) | grid points using a † step | common-curve calibrated (pointwise) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| quintile | all | Q1 − Q5 | 3.9 to 4.0 | 2 | — | — | none resolved | none resolved | — | 0 | none resolved |
| quintile | all | Q1 − Q2 | 2.3 to 4.0 | 18 | 2.3–2.5 (3) | — | only negative where resolved | only negative where resolved | 3.00 | 0 | only negative where resolved |
| quintile | all | Q2 − Q3 | 2.7 to 4.6 | 20 | — | 3.6–3.9 (4) | only positive where resolved | none resolved | 3.12 | 0 | only positive where resolved |
| quintile | all | Q3 − Q4 | 3.0 to 5.0 | 21 | 3.0–3.4 (5) | — | only negative where resolved | only negative where resolved | 3.70, 4.87 | 0 | only negative where resolved |
| quintile | all | Q4 − Q5 | 3.9 to 5.4 | 16 | — | 4.7–4.7 (1) | only positive where resolved | none resolved | 4.29 | 0 | only positive where resolved |
| zipf | all | XTail − Head | 3.9 to 4.2 | 4 | — | 3.9–4.2 (4) | only positive where resolved | only positive where resolved | — | 0 | only positive where resolved |
| zipf | all | XTail − Tail | 2.8 to 4.2 | 15 | — | 3.5–4.2 (8) | only positive where resolved | only positive where resolved | 3.14 | 0 | only positive where resolved |
| zipf | all | Tail − Head | 3.9 to 5.2 | 14 | — | 4.5–5.2 (8) | only positive where resolved | only positive where resolved | 4.03 | 0 | only positive where resolved |
| quintile | ≥ 512 | Q1 − Q5 | 3.9 to 4.0 | 2 | — | — | none resolved | none resolved | — | 0 | none resolved |
| quintile | ≥ 512 | Q1 − Q2 | 2.3 to 4.0 | 18 | 2.3–2.5 (3) | — | only negative where resolved | only negative where resolved | 3.00 | 0 | only negative where resolved |
| quintile | ≥ 512 | Q2 − Q3 | 2.7 to 4.6 | 20 | — | 3.6–3.9 (4) | only positive where resolved | none resolved | 3.12 | 0 | only positive where resolved |
| quintile | ≥ 512 | Q3 − Q4 | 3.0 to 5.0 | 21 | 3.0–3.4 (5) | — | only negative where resolved | only negative where resolved | 3.70, 4.87 | 0 | only negative where resolved |
| quintile | ≥ 512 | Q4 − Q5 | 3.9 to 5.4 | 16 | — | 4.7–4.7 (1) | only positive where resolved | none resolved | 4.29 | 0 | only positive where resolved |
| zipf | ≥ 512 | XTail − Head | 3.9 to 4.2 | 4 | — | 3.9–4.2 (4) | only positive where resolved | only positive where resolved | — | 0 | only positive where resolved |
| zipf | ≥ 512 | XTail − Tail | 2.8 to 4.2 | 15 | — | 3.5–4.2 (8) | only positive where resolved | only positive where resolved | 3.14 | 0 | only positive where resolved |
| zipf | ≥ 512 | Tail − Head | 3.9 to 5.2 | 14 | — | 4.5–5.2 (8) | only positive where resolved | only positive where resolved | 4.03 | 0 | only positive where resolved |

Common-curve calibrated = D − D_null, where D_null passes a natural spline f̂(log10 E) fitted to all verbs and steps of that comparison's step set through the same sampling, binning and interpolation (qualifier).

**Primary (Q1 − Q5, all steps, bin curves): insufficient overlap (one grid point); simultaneous band: —; per-verb alignment: none resolved; common-curve calibrated: insufficient overlap (one grid point) → crossing verdict: missed.** Matched exposure here compares frequent verbs early in training with rare verbs late: an observational trajectory pattern, confounded with model age.

Check against A6: A6's matched-exposure diagnostic recomputed from the same scores (30485 cell pairs, |Δ log10 E| ≤ 0.1, steps ≥ 4× apart; later − earlier) +0.179 [-0.133, +0.462] (A6 reported +0.179: reproduced). Pooled D̄ over the 10 quintile pairs (bin curves, all steps, grid cells equally weighted) +0.185 [-0.052, +0.412]; difference D̄ − diagnostic +0.006 [-0.103, +0.116] (different weighting of the same comparison; see the plan).

### 4. Class-level vs verb-specific (descriptive)

| step | AUC T vs I Head / Tail / XTail | mean L Head / Tail / XTail | single-token a_V T / I Head | Tail | a_V AUC Head / Tail | identity max dev |
|---|---|---|---|---|---|---|
| 512 | 0.88 [0.81, 0.95] / 0.59 [0.52, 0.66] / 0.51 [0.44, 0.59] | +1.56 / +0.29 / +0.01 | +0.71 / -0.91 | +1.11 / +0.21 | 0.90 / 0.80 | 2.4e-06 |
| 2000 | 0.94 [0.90, 0.98] / 0.78 [0.72, 0.83] / 0.65 [0.56, 0.73] | +3.31 / +1.64 / +0.65 | -0.64 / -3.97 | -0.96 / -2.84 | 0.94 / 0.77 | 6.3e-06 |
| 8000 | 0.93 [0.87, 0.97] / 0.90 [0.86, 0.93] / 0.80 [0.74, 0.86] | +3.20 / +2.54 / +1.80 | +0.06 / -3.16 | -0.07 / -2.12 | 0.92 / 0.89 | 5.1e-06 |
| 23000 | 0.93 [0.87, 0.97] / 0.89 [0.85, 0.93] / 0.81 [0.73, 0.86] | +3.16 / +2.51 / +1.63 | +0.35 / -2.89 | +0.25 / -1.60 | 0.93 / 0.89 | 3.1e-06 |
| 63000 | 0.93 [0.88, 0.97] / 0.89 [0.85, 0.92] / 0.82 [0.75, 0.87] | +3.24 / +2.50 / +1.71 | +0.14 / -3.15 | +0.22 / -1.76 | 0.94 / 0.92 | 3.1e-06 |
| 143000 | 0.94 [0.89, 0.98] / 0.90 [0.86, 0.93] / 0.83 [0.77, 0.88] | +3.50 / +2.64 / +1.86 | -0.29 / -3.85 | -0.29 / -2.45 | 0.94 / 0.93 | 6.0e-06 |

Separation step (AUC CI above 0.5 at every later checkpoint): Head 512, Tail 512, XTail 2000. XTail I onset: median primary onset 2,451; band-mean Δ_pass(I) < 0 from step 2000. a_V identity (single-token a_V = Δ_pass + per-checkpoint constant): max deviation 6.3e-06 nats.

## Replication (principal claims)

| claim | pythia-1.4b | pythia-410m | pythia-6.9b | 410M replicates 1.4B |
|---|---|---|---|---|
| onset (β_on verdict) | held | held, not robust to censoring (maturity-floor Tobit not 'held') | held | no |
| late window (β_b − β_w verdict) | held (also beyond the fitted common exposure curve) | held (also beyond the fitted common exposure curve) | failed (consistent with count equivalence) | yes |
| crossing (Q1 − Q5 verdict) | missed | missed | missed | yes |

Matching labels that are not 'held' (e.g. unresolved) are matching inconclusiveness, not confirmation.

