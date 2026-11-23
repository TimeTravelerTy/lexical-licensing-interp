# LEACE necessity of verb class at the verb position (Pythia-1.4B)

Scripts: `scripts/run_leace_necessity.py` (GPU), `scripts/analyze_leace_necessity.py` (this report). Run: 30 min, concept-erasure 0.2.4 (wheel), fp32 (TF32 off).

## Setup

- Items: task 1 of `analyze_das_prep.py` (curated `was` prefixes + participle, other verbs' contexts): 126 verb pairs x 125 contexts each (all of the 126 curated contexts but the pair's own) = 15,750 items, a T and an I prompt each (31,500 prompts). Readout: log P(" by") at the participle's last token.
- Erasure position: the participle's last subtoken (= last prompt token, the DAS Runner's anchor; the readout is at the same position). Applied to T and I prompts alike.
- Sites: DAS convention, site L = residual stream entering layer L (site 0 = embedding output), L = 0..23. DAS site = 8.
- Concept: class T (1) vs I (0). LeaceFitter with package defaults (affine, shrinkage on, svd_tol 0.01, trace constraint on), float64 statistics.
- Split: the 29-pair DAS split covers only 8 of these 126 pairs, so the DAS fold procedure (`run_das_round2.folds`: pairs shuffled within strata, dealt round-robin; 5 folds, seed 0) is applied to the 126 pairs, strata = verb band x token pattern (both participles single-token or not). Every verb is in one pair only. Erasers for fold f are fit on the other four folds and applied to fold f (cross-fitted), so every item is erased by an eraser that never saw its verb.
- Fitting samples per eraser (per fold, the same at every site): n = 25,000–25,250 verb-position states (200–202 distinct verbs) vs d = 2048 (n/d = 12.2–12.3); shrinkage left on (package default).
- Conditions: `single_L` = LEACE at site L only, fit on natural states. `scrub_all` = concept scrubbing at sites 0..23: per fold, eraser L is fit on the training items' site-L states computed with that fold's erasers 0..L-1 applied (and L not applied). `scrub_0_8` = erasers 0..8 of `scrub_all` (the sequential fit makes them identical).
- In-sample conditions (`insample_*`, rows marked "in-sample"): the same three kinds of erasure, but one eraser set fit on all items (every verb, both prompts) and applied to those same items, so the evaluated items are the fitting data. Their probes: the same nested procedure with one outer fold, i.e. a 5-group verb-held-out split over all pairs (the eraser has seen the test verbs), and a linear probe trained and scored on all items (fit-set accuracy).
- Probes (nested, verb-held-out): for each outer fold g, the states of fold g's items after erasure by fold g's erasers (fit on the other folds); the fold's pairs are dealt into 5 inner groups, a probe is trained on 4 groups and tested on the 5th, accuracy pooled over all test items; chance = 0.5. So neither the probe's training nor its test items were seen by the eraser, and probe train and test verbs differ. Natural states use the same splits. Linear = L2 logistic regression; MLP = one hidden layer of 256 ReLU units.
- KL = mean over prompts (T and I) of KL(P_natural || P_erased) over the full vocabulary at the verb position, natural recomputed in the same batch.
- CIs: task-1 two-way cluster bootstrap (1000 draws, seed 17), shared across conditions.

## Checks

- Identity hook at site 8 vs same-batch natural: max |Δ log P| = 0.
- Non-anchor positions after the erasure hooks (scrub_all, all sites): max |Δh| = 0.
- single_L: erased states inside the model vs offline erasure of the natural states: max |Δ| = 0.
- Same-batch natural vs the natural capture pass (different batching): max |Δ log P(by)| = 0.
- Scrub replay (training items, all 24 erasers applied): pre-erasure mean at each site vs the mean each eraser was fit on, max |Δ| = 2.2e-06; cross-covariance relative difference max 3.4e-06; post-erasure / pre-erasure cross-covariance norm, max over sites and folds 3.4e-06.
- Erasers: top singular value of W Σ_xz min 0.390 (svd_tol 0.01: 0 fits below it, i.e. no erasure); erased rank [1.0]; trace-constraint branch used in 0 fits; training cross-covariance residual max 2.8e-14.
- Held-out class-mean difference, erased / natural norm: single_L 0.575–0.994; scrub_all 0.704–0.994 (range over sites and folds).
- In-sample run (21 min): n = 31,500 fitting states per eraser (252 verbs; n/d = 15.4); top singular value min 0.394, erased rank [1.0], trace branch used in 0 fits; same checks as above: identity hook 0, non-anchor 0, in-model vs offline 0, scrub replay mean 2.1e-06 / cross-covariance 3.2e-06; natural log P(by) vs the cross-fitted run's max |Δ| 0.
- In-sample, on the evaluated items (= the fitting items): cross-covariance with class after / before erasure, max over scrub sites 3.2e-06, single_L max 2.3e-14; class-mean difference after / before erasure: single_L max 2.8e-05, scrub 0–23 max 2.8e-05, scrub 0–8 (erased sites) max 2.8e-05.

## Flags

- Held-out linear probe after erasure is not at chance: 0.567–0.754 over every erased site and condition (chance 0.5; natural 0.627–0.904).
- Held-out MLP probe after erasure is not at chance: 0.575–0.769 over every erased site and condition (chance 0.5; natural 0.635–0.897).
- Readout proximity: *by* is predicted at the verb position itself, so late sites sit next to the readout, and erasure there partly removes the readout computation itself rather than only verb-class information the readout uses. Sites 0–~8 are the necessity-relevant range; rows for later sites (including the post hoc single-site minimum) do not test necessity in that sense.
- In-sample erasure, linear probe trained and scored on all (= fitting) items: 0.488–0.510 over every erased site and condition (chance 0.5).
- In-sample erasure, verb-held-out linear probe (eraser saw the test verbs): 0.000–0.179 over every erased site and condition (chance 0.5).
- In-sample erasure, verb-held-out MLP probe (eraser saw the test verbs): 0.016–0.592 over every erased site and condition (chance 0.5).
- In-sample verb-held-out probes can fall below chance: the eraser zeroes the class cross-covariance over all verbs together, so whatever class signal remains in the probe's training verbs points the opposite way in its test verbs (the same effect as the pooled cross-fold probe in the first smoke test).

## Main table

Probe columns: held-out accuracy at the erased site (scrubbing: range over the erased sites); for the in-sample rows the test verbs are held out from the probe but not from the eraser. Rows for sites after ~8 (scrub 0–23, the post hoc single-site minimum) include erasure next to the readout (see Flags). The B5 reference rows are from `reports/round5/b5_necessity.md` (frame `hasbeen`, all 64 pairs; different items and frame: B7 items, "The N has been V"), converted to share remaining = 1 − share removed.

| Condition | *by* gap (nats) | share of natural gap remaining | KL (nats) | linear probe | MLP probe |
|---|---:|---:|---:|---:|---:|
| natural | +0.60 [+0.38, +0.83] | 1.00 [1.00, 1.00] | 0.000 |  |  |
| scrub, all sites (0–L−1) | +0.21 [-0.02, +0.44] | 0.35 [-0.06, 0.54] | 0.035 | 0.567–0.754 | 0.575–0.769 |
| scrub, sites 0–8 | +0.34 [+0.11, +0.57] | 0.56 [0.31, 0.70] | 0.024 | 0.567–0.754 | 0.575–0.769 |
| scrub, sites 0–8 | +0.34 [+0.11, +0.57] | 0.56 [0.31, 0.70] | 0.024 | 0.567–0.754 | 0.575–0.769 |
| single site 8 (DAS site) | +0.30 [+0.06, +0.52] | 0.49 [0.17, 0.64] | 0.021 | 0.665 | 0.665 |
| single site 14: lowest of 16 (post hoc; pointwise CI) | +0.09 [-0.14, +0.33] | 0.16 [-0.36, 0.41] | 0.017 | 0.654 | 0.666 |
| in-sample: scrub, all sites (0–L−1) | -0.28 [-0.51, -0.05] | -0.47 [-1.35, -0.06] | 0.107 | 0.000–0.179 | 0.016–0.302 |
| in-sample: scrub, sites 0–8 | +0.04 [-0.18, +0.28] | 0.07 [-0.48, 0.33] | 0.070 | 0.000–0.179 | 0.016–0.210 |
| in-sample: scrub, sites 0–8 | +0.04 [-0.18, +0.28] | 0.07 [-0.48, 0.33] | 0.070 | 0.000–0.179 | 0.016–0.210 |
| in-sample: single site 8 (DAS site) | +0.18 [-0.05, +0.41] | 0.30 [-0.12, 0.50] | 0.032 | 0.011 | 0.349 |
| in-sample: single site 14: lowest of 16 (post hoc; pointwise CI) | -0.14 [-0.38, +0.10] | -0.23 [-1.02, 0.12] | 0.023 | 0.019 | 0.592 |
| ref (B5): d -> I mean (DAS site 8 coordinate) |  | 0.85 [0.69, 0.91] |  |  |  |
| ref (B5): top-50 switch neurons -> I mean |  | 0.91 [0.85, 0.96] |  |  |  |
| ref (B5): both (d + top-50 neurons) |  | 0.81 [0.66, 0.87] |  |  |  |

## Single-site LEACE, every site

Natural linear / MLP probe accuracy at the same site in the last two columns.

| Site | *by* gap | share remaining | KL | linear probe | MLP probe | natural linear | natural MLP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | +0.59 [+0.37, +0.82] | 0.99 [0.96, 1.02] | 0.004 | 0.571 | 0.575 | 0.627 | 0.635 |
| 1 | +0.53 [+0.31, +0.75] | 0.88 [0.80, 0.93] | 0.007 | 0.703 | 0.689 | 0.839 | 0.822 |
| 2 | +0.49 [+0.27, +0.71] | 0.82 [0.70, 0.88] | 0.009 | 0.694 | 0.718 | 0.868 | 0.862 |
| 3 | +0.45 [+0.22, +0.67] | 0.74 [0.58, 0.82] | 0.012 | 0.675 | 0.682 | 0.888 | 0.878 |
| 4 | +0.41 [+0.18, +0.64] | 0.68 [0.49, 0.78] | 0.012 | 0.688 | 0.696 | 0.890 | 0.870 |
| 5 | +0.37 [+0.14, +0.59] | 0.61 [0.38, 0.73] | 0.016 | 0.687 | 0.694 | 0.904 | 0.897 |
| 6 | +0.32 [+0.09, +0.54] | 0.53 [0.23, 0.67] | 0.019 | 0.699 | 0.708 | 0.901 | 0.895 |
| 7 | +0.32 [+0.09, +0.55] | 0.54 [0.25, 0.68] | 0.019 | 0.700 | 0.710 | 0.894 | 0.884 |
| 8 | +0.30 [+0.06, +0.52] | 0.49 [0.17, 0.64] | 0.021 | 0.665 | 0.665 | 0.883 | 0.883 |
| 9 | +0.28 [+0.05, +0.51] | 0.47 [0.15, 0.63] | 0.022 | 0.660 | 0.658 | 0.881 | 0.874 |
| 10 | +0.28 [+0.05, +0.52] | 0.47 [0.13, 0.63] | 0.021 | 0.649 | 0.672 | 0.883 | 0.876 |
| 11 | +0.30 [+0.06, +0.53] | 0.50 [0.16, 0.65] | 0.017 | 0.645 | 0.674 | 0.893 | 0.892 |
| 12 | +0.26 [+0.03, +0.50] | 0.44 [0.08, 0.60] | 0.016 | 0.651 | 0.690 | 0.894 | 0.895 |
| 13 | +0.16 [-0.07, +0.40] | 0.27 [-0.18, 0.49] | 0.017 | 0.648 | 0.668 | 0.884 | 0.883 |
| 14 | +0.09 [-0.14, +0.33] | 0.16 [-0.36, 0.41] | 0.017 | 0.654 | 0.666 | 0.872 | 0.861 |
| 15 | +0.11 [-0.13, +0.34] | 0.17 [-0.33, 0.42] | 0.016 | 0.649 | 0.656 | 0.875 | 0.863 |

## Probe accuracy under scrubbing, by site

`scrub_0_8` is erased at sites 0–8 only; its later sites are unerased (states downstream of the erasure). Probes stop at site 23, the input to the last layer; the state after the last layer (which feeds the readout) is not probed.

| Site | natural lin | natural MLP | scrub 0–23 lin | scrub 0–23 MLP | scrub 0–8 lin | scrub 0–8 MLP |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.627 | 0.635 | 0.567 | 0.575 | 0.567 | 0.575 |
| 1 | 0.839 | 0.822 | 0.706 | 0.695 | 0.706 | 0.695 |
| 2 | 0.868 | 0.862 | 0.714 | 0.728 | 0.714 | 0.728 |
| 3 | 0.888 | 0.878 | 0.715 | 0.718 | 0.715 | 0.718 |
| 4 | 0.890 | 0.870 | 0.740 | 0.736 | 0.740 | 0.736 |
| 5 | 0.904 | 0.897 | 0.753 | 0.751 | 0.753 | 0.751 |
| 6 | 0.901 | 0.895 | 0.754 | 0.769 | 0.754 | 0.769 |
| 7 | 0.894 | 0.884 | 0.751 | 0.764 | 0.751 | 0.764 |
| 8 | 0.883 | 0.883 | 0.728 | 0.734 | 0.728 | 0.734 |
| 9 | 0.881 | 0.874 | 0.734 | 0.729 | 0.753 | 0.747 |
| 10 | 0.883 | 0.876 | 0.736 | 0.744 | 0.762 | 0.769 |
| 11 | 0.893 | 0.892 | 0.730 | 0.747 | 0.775 | 0.785 |
| 12 | 0.894 | 0.895 | 0.744 | 0.761 | 0.791 | 0.799 |
| 13 | 0.884 | 0.883 | 0.732 | 0.738 | 0.789 | 0.786 |
| 14 | 0.872 | 0.861 | 0.723 | 0.714 | 0.778 | 0.762 |
| 15 | 0.875 | 0.863 | 0.717 | 0.712 | 0.782 | 0.762 |

## In-sample erasure, every site

Erasers fit on all items and evaluated on them. Single-site rows: `insample_single_L`; probe columns as in the main table, plus the fit-set linear probe. Last three columns: probe accuracy under in-sample scrubbing 0–23 at that site.

| Site | *by* gap | share remaining | KL | linear probe | MLP probe | fit-set linear | scrub 0–23 fit-set lin | scrub 0–23 lin | scrub 0–23 MLP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | +0.58 [+0.36, +0.80] | 0.96 [0.92, 1.00] | 0.006 | 0.179 | 0.210 | 0.488 | 0.488 | 0.179 | 0.210 |
| 1 | +0.44 [+0.22, +0.66] | 0.73 [0.58, 0.81] | 0.017 | 0.002 | 0.020 | 0.499 | 0.510 | 0.000 | 0.016 |
| 2 | +0.38 [+0.15, +0.61] | 0.63 [0.41, 0.74] | 0.020 | 0.000 | 0.025 | 0.501 | 0.503 | 0.000 | 0.021 |
| 3 | +0.34 [+0.11, +0.57] | 0.56 [0.30, 0.69] | 0.023 | 0.001 | 0.030 | 0.500 | 0.499 | 0.001 | 0.028 |
| 4 | +0.27 [+0.04, +0.50] | 0.44 [0.11, 0.60] | 0.027 | 0.000 | 0.048 | 0.498 | 0.503 | 0.001 | 0.032 |
| 5 | +0.22 [-0.01, +0.45] | 0.37 [-0.02, 0.54] | 0.031 | 0.002 | 0.098 | 0.505 | 0.499 | 0.003 | 0.061 |
| 6 | +0.17 [-0.05, +0.40] | 0.28 [-0.12, 0.49] | 0.035 | 0.004 | 0.212 | 0.496 | 0.503 | 0.005 | 0.098 |
| 7 | +0.20 [-0.02, +0.43] | 0.34 [-0.06, 0.53] | 0.032 | 0.006 | 0.298 | 0.501 | 0.502 | 0.009 | 0.149 |
| 8 | +0.18 [-0.05, +0.41] | 0.30 [-0.12, 0.50] | 0.032 | 0.011 | 0.349 | 0.497 | 0.504 | 0.016 | 0.179 |
| 9 | +0.17 [-0.06, +0.40] | 0.28 [-0.16, 0.48] | 0.032 | 0.010 | 0.375 | 0.497 | 0.497 | 0.018 | 0.211 |
| 10 | +0.18 [-0.06, +0.42] | 0.30 [-0.15, 0.50] | 0.030 | 0.013 | 0.437 | 0.501 | 0.503 | 0.022 | 0.253 |
| 11 | +0.18 [-0.05, +0.42] | 0.30 [-0.14, 0.50] | 0.026 | 0.014 | 0.477 | 0.504 | 0.501 | 0.017 | 0.268 |
| 12 | +0.13 [-0.11, +0.36] | 0.21 [-0.29, 0.43] | 0.024 | 0.014 | 0.555 | 0.499 | 0.492 | 0.015 | 0.282 |
| 13 | -0.02 [-0.26, +0.21] | -0.03 [-0.69, 0.25] | 0.024 | 0.016 | 0.576 | 0.500 | 0.494 | 0.017 | 0.290 |
| 14 | -0.14 [-0.38, +0.10] | -0.23 [-1.02, 0.12] | 0.023 | 0.019 | 0.592 | 0.501 | 0.495 | 0.018 | 0.302 |
| 15 | -0.11 [-0.35, +0.12] | -0.19 [-0.94, 0.15] | 0.023 | 0.026 | 0.580 | 0.497 | 0.497 | 0.024 | 0.296 |

## Single-site vs sequential-scrub eraser at the same site

Each rank-1 eraser maps x to x − ⟨r, x − b⟩ e. |cos| between the single-site eraser at site L (fit on natural site-L states) and the scrub eraser at site L (fit on site-L states after erasure at sites 0..L−1), for the edit direction e (what is removed) and the read direction r (what it is keyed on). Cross-fitted: mean over the 5 folds [min, max]; in-sample: one eraser pair per site.

Site 14 (the post hoc single-site minimum; share remaining single 0.16, scrub 0–14 0.34, scrub 0–23 0.35): |cos| cross-fit: edit 0.252 [0.236, 0.278], read 0.262 [0.234, 0.288]; insample: edit 0.246, read 0.272.

So the scrub eraser at site 14 removes a direction nearly orthogonal to the one the single-site eraser removes (edit |cos| 0.252): after erasure at sites 0..13, the class-correlated direction in the fitting states at site 14 is a different one, and the single-site direction is left in place. Of the gap between scrub 0–23 (0.35) and single site 14 (0.16), the cumulative curve puts 0.18 before site 15 (scrub 0–14 0.34) and +0.01 in sites 15–23.

| Site | cross-fit edit | cross-fit read | insample edit | insample read |
|---:|---:|---:|---:|---:|
| 0 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 | 1.000 |
| 1 | 0.862 [0.850, 0.873] | 0.740 [0.732, 0.750] | 0.875 | 0.738 |
| 2 | 0.554 [0.545, 0.567] | 0.516 [0.495, 0.536] | 0.559 | 0.522 |
| 3 | 0.442 [0.434, 0.449] | 0.280 [0.248, 0.296] | 0.447 | 0.281 |
| 4 | 0.459 [0.444, 0.480] | 0.369 [0.358, 0.392] | 0.461 | 0.369 |
| 5 | 0.418 [0.399, 0.442] | 0.309 [0.273, 0.342] | 0.414 | 0.310 |
| 6 | 0.441 [0.426, 0.454] | 0.346 [0.323, 0.369] | 0.437 | 0.358 |
| 7 | 0.350 [0.341, 0.365] | 0.257 [0.233, 0.285] | 0.346 | 0.265 |
| 8 | 0.237 [0.226, 0.257] | 0.275 [0.264, 0.287] | 0.230 | 0.291 |
| 9 | 0.211 [0.201, 0.217] | 0.239 [0.206, 0.250] | 0.204 | 0.242 |
| 10 | 0.268 [0.256, 0.279] | 0.275 [0.258, 0.308] | 0.263 | 0.280 |
| 11 | 0.325 [0.310, 0.339] | 0.298 [0.262, 0.350] | 0.320 | 0.272 |
| 12 | 0.352 [0.324, 0.386] | 0.350 [0.340, 0.359] | 0.349 | 0.351 |
| 13 | 0.279 [0.266, 0.293] | 0.274 [0.252, 0.288] | 0.274 | 0.263 |
| 14 | 0.252 [0.236, 0.278] | 0.262 [0.234, 0.288] | 0.246 | 0.272 |
| 15 | 0.124 [0.093, 0.143] | 0.155 [0.109, 0.195] | 0.119 | 0.154 |

## By verb band and token pattern

Share of the natural gap remaining (natural gap in parentheses). Pattern: participle token counts of T and I (2+ = two or more).

| Subset | pairs | natural gap | scrub 0–23 | scrub 0–8 | single 8 | single 14 (post hoc) | in-sample scrub 0–23 | in-sample scrub 0–8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| head | 26 | +0.71 [+0.11, +1.31] | 0.29 [-2.75, 0.65] | 0.56 [-1.13, 0.78] | 0.50 [-1.71, 0.76] | 0.06 [-4.74, 0.55] | -0.33 [-6.28, 0.31] | 0.26 [-2.82, 0.63] |
| tail | 50 | +0.62 [+0.27, +0.98] | 0.40 [-0.42, 0.65] | 0.61 [0.09, 0.78] | 0.48 [-0.26, 0.69] | 0.15 [-0.99, 0.48] | -0.36 [-2.11, 0.17] | 0.13 [-1.00, 0.48] |
| xtail | 50 | +0.53 [+0.20, +0.84] | 0.33 [-0.72, 0.62] | 0.51 [-0.25, 0.73] | 0.50 [-0.29, 0.70] | 0.24 [-0.90, 0.52] | -0.70 [-3.24, -0.04] | -0.13 [-1.78, 0.31] |
| pattern T1-I1 | 27 | +0.68 [+0.10, +1.25] | 0.28 [-2.96, 0.65] | 0.56 [-1.19, 0.78] | 0.48 [-1.93, 0.76] | 0.03 [-5.15, 0.54] | -0.38 [-6.98, 0.29] | 0.23 [-3.28, 0.62] |
| pattern T1-I2+ | 2 | +0.43 [-1.75, +2.63] | 0.51 [0.43, 2.12] | 0.80 [0.76, 1.47] | 0.32 [0.20, 2.08] | -0.77 [-1.07, 3.70] | -0.50 [-0.72, 3.52] | 0.25 [0.11, 2.22] |
| pattern T2+-I1 | 2 | -0.46 [-2.17, +1.18] | 3.18 [-13.05, 3.71] | 2.42 [-8.56, 2.78] | 2.37 [-6.27, 2.69] | 3.33 [-12.09, 3.83] | 4.15 [-18.95, 4.89] | 3.13 [-13.34, 3.65] |
| pattern T2+-I2+ | 95 | +0.60 [+0.35, +0.84] | 0.41 [-0.01, 0.59] | 0.59 [0.28, 0.72] | 0.53 [0.19, 0.67] | 0.26 [-0.27, 0.47] | -0.43 [-1.41, -0.01] | 0.07 [-0.59, 0.34] |

Bootstrap draws that resample no pair of a subset are dropped: pattern T1-I2+ 875/1000 valid, pattern T2+-I1 865/1000 valid

## Files

- `reports/pythia/leace_necessity_gaps.csv`: gap, Δ gap, share remaining (with CIs) per condition and subset; KL (all-items rows).
- `reports/pythia/leace_necessity_probes.csv`: probe accuracy per condition, site and probe type.
- `reports/pythia/leace_necessity_fit_diagnostics.csv`: per eraser: n, distinct verbs, top singular value, rank, training cross-covariance residual, held-out class-mean difference before/after.
- `reports/pythia/leace_necessity_references.csv`: the B5 reference numbers.
- `reports/pythia/leace_necessity_cumulative.csv`: the cumulative scrub curve.
- `reports/pythia/leace_necessity_eraser_cosines.csv`: single-site vs scrub eraser |cos| per site.
- `results/olmo/leace_necessity/`: per-prompt log P(by) and KL per condition (`items.csv.gz`), fold assignment, checks, run metadata. `items.csv.gz` is not in git (sha256 70d7097676b42cbf456387bf72125f6be89e994dffd55f1b8e293da2363475e5); it is on TSUBAME at `/gs/fs/tga-sip_arase/tyrone/lli_leace/results/pythia/leace_necessity/` (the first run's file next to it as `items_run1.csv.gz`).
- `results/olmo/leace_insample/`: the same files for the in-sample run (`items.csv.gz` not in git, sha256 0bc45d6d8a9abe34f402b2248a9d5b5b0ad4ff04ed000a786900ea0b5f28dbe9; on TSUBAME under `lli_leace/results/olmo/leace_insample/`).

