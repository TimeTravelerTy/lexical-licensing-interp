# LEACE necessity of verb class at the verb position (Pythia-1.4B)

Scripts: `scripts/run_leace_necessity.py` (GPU), `scripts/analyze_leace_necessity.py` (this report). Run: 41 min, concept-erasure 0.2.4 (wheel), fp32 (TF32 off).

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
- Scrub replay (training items, all 24 erasers applied): pre-erasure mean at each site vs the mean each eraser was fit on, max |Δ| = 0.00046; cross-covariance relative difference max 2.1e-06; post-erasure / pre-erasure cross-covariance norm, max over sites and folds 2e-06.
- Erasers: top singular value of W Σ_xz min 0.409 (svd_tol 0.01: 0 fits below it, i.e. no erasure); erased rank [1.0]; trace-constraint branch used in 0 fits; training cross-covariance residual max 3.7e-14.
- Held-out class-mean difference, erased / natural norm: single_L 0.266–0.989; scrub_all 0.522–0.985 (range over sites and folds).
- In-sample run (30 min): n = 31,500 fitting states per eraser (252 verbs; n/d = 15.4); top singular value min 0.409, erased rank [1.0], trace branch used in 0 fits; same checks as above: identity hook 0, non-anchor 0, in-model vs offline 0, scrub replay mean 0.0004 / cross-covariance 2e-06; natural log P(by) vs the cross-fitted run's max |Δ| 0.
- In-sample, on the evaluated items (= the fitting items): cross-covariance with class after / before erasure, max over scrub sites 2e-06, single_L max 3.1e-14; class-mean difference after / before erasure: single_L max 0.00013, scrub 0–23 max 0.00012, scrub 0–8 (erased sites) max 6.1e-05.

## Flags

- Held-out linear probe after erasure: 0.528–0.695 over the erased sites (chance 0.5).
- Held-out MLP probe after erasure is not at chance: 0.565–0.708 over every erased site and condition (chance 0.5; natural 0.628–0.883).
- Readout proximity: *by* is predicted at the verb position itself, so late sites sit next to the readout, and erasure there partly removes the readout computation itself rather than only verb-class information the readout uses. Sites 0–~13 are the necessity-relevant range; rows for later sites (including the post hoc single-site minimum) do not test necessity in that sense.
- In-sample erasure, linear probe trained and scored on all (= fitting) items: 0.480–0.509 over every erased site and condition (chance 0.5).
- In-sample erasure, verb-held-out linear probe (eraser saw the test verbs): 0.009–0.281 over every erased site and condition (chance 0.5).
- In-sample erasure, verb-held-out MLP probe (eraser saw the test verbs): 0.076–0.676 over every erased site and condition (chance 0.5).
- In-sample verb-held-out probes can fall below chance: the eraser zeroes the class cross-covariance over all verbs together, so whatever class signal remains in the probe's training verbs points the opposite way in its test verbs (the same effect as the pooled cross-fold probe in the first smoke test).

## Main table

Probe columns: held-out accuracy at the erased site (scrubbing: range over the erased sites); for the in-sample rows the test verbs are held out from the probe but not from the eraser. Rows for sites after ~13 (scrub 0–23, the post hoc single-site minimum) include erasure next to the readout (see Flags). The B5 reference rows are from `reports/round5/b5_necessity.md` (frame `hasbeen`, all 64 pairs; different items and frame: B7 items, "The N has been V"), converted to share remaining = 1 − share removed.

| Condition | *by* gap (nats) | share of natural gap remaining | KL (nats) | linear probe | MLP probe |
|---|---:|---:|---:|---:|---:|
| natural | +0.41 [+0.17, +0.64] | 1.00 [1.00, 1.00] | 0.000 |  |  |
| scrub, all sites (0–L−1) | -0.01 [-0.25, +0.21] | -0.02 [-1.33, 0.35] | 0.045 | 0.546–0.695 | 0.565–0.708 |
| scrub, sites 0–13 | +0.10 [-0.13, +0.32] | 0.24 [-0.73, 0.51] | 0.030 | 0.546–0.688 | 0.565–0.708 |
| scrub, sites 0–8 | +0.12 [-0.10, +0.34] | 0.30 [-0.61, 0.55] | 0.026 | 0.546–0.686 | 0.565–0.698 |
| single site 8 (DAS site) | +0.18 [-0.05, +0.40] | 0.44 [-0.28, 0.64] | 0.021 | 0.637 | 0.654 |
| single site 20: lowest of 26 (post hoc; pointwise CI) | +0.08 [-0.16, +0.30] | 0.19 [-0.88, 0.48] | 0.011 | 0.664 | 0.679 |
| in-sample: scrub, all sites (0–L−1) | -0.42 [-0.64, -0.20] | -1.03 [-3.52, -0.33] | 0.095 | 0.009–0.191 | 0.076–0.518 |
| in-sample: scrub, sites 0–13 | -0.10 [-0.32, +0.11] | -0.26 [-1.77, 0.18] | 0.061 | 0.009–0.191 | 0.076–0.438 |
| in-sample: scrub, sites 0–8 | -0.04 [-0.26, +0.17] | -0.10 [-1.40, 0.28] | 0.053 | 0.009–0.191 | 0.076–0.325 |
| in-sample: single site 8 (DAS site) | +0.13 [-0.09, +0.35] | 0.32 [-0.55, 0.55] | 0.021 | 0.035 | 0.453 |
| in-sample: single site 19: lowest of 26 (post hoc; pointwise CI) | -0.09 [-0.32, +0.13] | -0.23 [-1.83, 0.21] | 0.013 | 0.057 | 0.649 |
| ref (B5): d -> I mean (DAS site 8 coordinate) |  | 0.85 [0.69, 0.91] |  |  |  |
| ref (B5): top-50 switch neurons -> I mean |  | 0.91 [0.85, 0.96] |  |  |  |
| ref (B5): both (d + top-50 neurons) |  | 0.81 [0.66, 0.87] |  |  |  |

## Single-site LEACE, every site

Natural linear / MLP probe accuracy at the same site in the last two columns.

| Site | *by* gap | share remaining | KL | linear probe | MLP probe | natural linear | natural MLP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | +0.39 [+0.16, +0.62] | 0.96 [0.87, 1.03] | 0.004 | 0.611 | 0.579 | 0.663 | 0.655 |
| 1 | +0.37 [+0.15, +0.60] | 0.92 [0.77, 1.03] | 0.013 | 0.528 | 0.574 | 0.586 | 0.628 |
| 2 | +0.34 [+0.11, +0.56] | 0.83 [0.63, 0.92] | 0.007 | 0.568 | 0.573 | 0.622 | 0.648 |
| 3 | +0.31 [+0.08, +0.53] | 0.77 [0.47, 0.87] | 0.006 | 0.645 | 0.667 | 0.754 | 0.747 |
| 4 | +0.29 [+0.06, +0.50] | 0.71 [0.36, 0.82] | 0.009 | 0.647 | 0.665 | 0.871 | 0.866 |
| 5 | +0.23 [+0.00, +0.45] | 0.57 [0.00, 0.73] | 0.014 | 0.612 | 0.637 | 0.885 | 0.883 |
| 6 | +0.21 [-0.02, +0.42] | 0.51 [-0.09, 0.69] | 0.016 | 0.644 | 0.661 | 0.864 | 0.864 |
| 7 | +0.20 [-0.02, +0.42] | 0.50 [-0.13, 0.68] | 0.019 | 0.638 | 0.645 | 0.866 | 0.864 |
| 8 | +0.18 [-0.05, +0.40] | 0.44 [-0.28, 0.64] | 0.021 | 0.637 | 0.654 | 0.875 | 0.872 |
| 9 | +0.18 [-0.05, +0.39] | 0.43 [-0.30, 0.64] | 0.022 | 0.646 | 0.670 | 0.879 | 0.874 |
| 10 | +0.19 [-0.04, +0.41] | 0.47 [-0.23, 0.66] | 0.021 | 0.639 | 0.664 | 0.879 | 0.875 |
| 11 | +0.18 [-0.04, +0.41] | 0.45 [-0.26, 0.65] | 0.021 | 0.636 | 0.665 | 0.874 | 0.865 |
| 12 | +0.19 [-0.05, +0.41] | 0.47 [-0.26, 0.66] | 0.021 | 0.635 | 0.662 | 0.866 | 0.859 |
| 13 | +0.23 [-0.00, +0.45] | 0.56 [-0.02, 0.72] | 0.019 | 0.637 | 0.651 | 0.855 | 0.837 |
| 14 | +0.21 [-0.02, +0.43] | 0.52 [-0.12, 0.70] | 0.021 | 0.644 | 0.653 | 0.853 | 0.835 |
| 15 | +0.24 [+0.01, +0.47] | 0.59 [0.06, 0.74] | 0.019 | 0.628 | 0.662 | 0.871 | 0.870 |
| 16 | +0.15 [-0.07, +0.37] | 0.38 [-0.39, 0.60] | 0.018 | 0.639 | 0.665 | 0.867 | 0.865 |
| 17 | +0.10 [-0.12, +0.32] | 0.25 [-0.69, 0.53] | 0.016 | 0.639 | 0.656 | 0.842 | 0.846 |
| 18 | +0.09 [-0.13, +0.31] | 0.23 [-0.72, 0.51] | 0.012 | 0.649 | 0.667 | 0.848 | 0.842 |
| 19 | +0.08 [-0.15, +0.30] | 0.21 [-0.82, 0.49] | 0.011 | 0.658 | 0.679 | 0.855 | 0.841 |
| 20 | +0.08 [-0.16, +0.30] | 0.19 [-0.88, 0.48] | 0.011 | 0.664 | 0.679 | 0.846 | 0.831 |
| 21 | +0.11 [-0.12, +0.34] | 0.28 [-0.66, 0.54] | 0.010 | 0.671 | 0.685 | 0.843 | 0.827 |
| 22 | +0.12 [-0.11, +0.34] | 0.28 [-0.65, 0.53] | 0.010 | 0.671 | 0.686 | 0.840 | 0.828 |
| 23 | +0.13 [-0.10, +0.35] | 0.31 [-0.59, 0.55] | 0.010 | 0.668 | 0.679 | 0.836 | 0.827 |
| 24 | +0.13 [-0.10, +0.35] | 0.31 [-0.59, 0.55] | 0.010 | 0.659 | 0.676 | 0.831 | 0.824 |
| 25 | +0.11 [-0.13, +0.34] | 0.27 [-0.71, 0.53] | 0.011 | 0.660 | 0.673 | 0.829 | 0.823 |

## Probe accuracy under scrubbing, by site

`scrub_0_8` is erased at sites 0–8 only; its later sites are unerased (states downstream of the erasure). Probes stop at site 23, the input to the last layer; the state after the last layer (which feeds the readout) is not probed.

| Site | natural lin | natural MLP | scrub 0–23 lin | scrub 0–23 MLP | scrub 0–8 lin | scrub 0–8 MLP |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.663 | 0.655 | 0.611 | 0.579 | 0.611 | 0.579 |
| 1 | 0.586 | 0.628 | 0.546 | 0.588 | 0.546 | 0.588 |
| 2 | 0.622 | 0.648 | 0.568 | 0.565 | 0.568 | 0.565 |
| 3 | 0.754 | 0.747 | 0.643 | 0.661 | 0.643 | 0.661 |
| 4 | 0.871 | 0.866 | 0.669 | 0.679 | 0.669 | 0.679 |
| 5 | 0.885 | 0.883 | 0.666 | 0.661 | 0.666 | 0.661 |
| 6 | 0.864 | 0.864 | 0.667 | 0.679 | 0.667 | 0.679 |
| 7 | 0.866 | 0.864 | 0.673 | 0.681 | 0.673 | 0.681 |
| 8 | 0.875 | 0.872 | 0.686 | 0.698 | 0.686 | 0.698 |
| 9 | 0.879 | 0.874 | 0.688 | 0.708 | 0.713 | 0.729 |
| 10 | 0.879 | 0.875 | 0.683 | 0.698 | 0.712 | 0.723 |
| 11 | 0.874 | 0.865 | 0.686 | 0.705 | 0.713 | 0.727 |
| 12 | 0.866 | 0.859 | 0.674 | 0.698 | 0.712 | 0.726 |
| 13 | 0.855 | 0.837 | 0.681 | 0.684 | 0.712 | 0.709 |
| 14 | 0.853 | 0.835 | 0.680 | 0.679 | 0.703 | 0.703 |
| 15 | 0.871 | 0.870 | 0.692 | 0.700 | 0.719 | 0.728 |
| 16 | 0.867 | 0.865 | 0.693 | 0.706 | 0.730 | 0.746 |
| 17 | 0.842 | 0.846 | 0.679 | 0.692 | 0.716 | 0.733 |
| 18 | 0.848 | 0.842 | 0.686 | 0.692 | 0.723 | 0.733 |
| 19 | 0.855 | 0.841 | 0.690 | 0.692 | 0.732 | 0.730 |
| 20 | 0.846 | 0.831 | 0.690 | 0.685 | 0.727 | 0.720 |
| 21 | 0.843 | 0.827 | 0.693 | 0.690 | 0.728 | 0.721 |
| 22 | 0.840 | 0.828 | 0.695 | 0.689 | 0.729 | 0.719 |
| 23 | 0.836 | 0.827 | 0.687 | 0.684 | 0.721 | 0.717 |
| 24 | 0.831 | 0.824 | 0.683 | 0.680 | 0.719 | 0.717 |
| 25 | 0.829 | 0.823 | 0.680 | 0.680 | 0.721 | 0.717 |

## In-sample erasure, every site

Erasers fit on all items and evaluated on them. Single-site rows: `insample_single_L`; probe columns as in the main table, plus the fit-set linear probe. Last three columns: probe accuracy under in-sample scrubbing 0–23 at that site.

| Site | *by* gap | share remaining | KL | linear probe | MLP probe | fit-set linear | scrub 0–23 fit-set lin | scrub 0–23 lin | scrub 0–23 MLP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | +0.38 [+0.15, +0.60] | 0.93 [0.82, 1.00] | 0.006 | 0.171 | 0.202 | 0.480 | 0.480 | 0.171 | 0.202 |
| 1 | +0.36 [+0.13, +0.58] | 0.88 [0.73, 0.94] | 0.005 | 0.281 | 0.435 | 0.502 | 0.505 | 0.191 | 0.325 |
| 2 | +0.31 [+0.08, +0.53] | 0.77 [0.47, 0.86] | 0.007 | 0.051 | 0.286 | 0.507 | 0.501 | 0.043 | 0.125 |
| 3 | +0.27 [+0.04, +0.49] | 0.66 [0.22, 0.79] | 0.012 | 0.013 | 0.214 | 0.490 | 0.509 | 0.009 | 0.076 |
| 4 | +0.22 [-0.00, +0.44] | 0.55 [-0.01, 0.71] | 0.015 | 0.012 | 0.298 | 0.495 | 0.503 | 0.013 | 0.133 |
| 5 | +0.17 [-0.05, +0.39] | 0.42 [-0.29, 0.63] | 0.020 | 0.015 | 0.279 | 0.500 | 0.497 | 0.015 | 0.143 |
| 6 | +0.14 [-0.09, +0.35] | 0.34 [-0.51, 0.57] | 0.022 | 0.018 | 0.256 | 0.497 | 0.499 | 0.019 | 0.162 |
| 7 | +0.15 [-0.07, +0.37] | 0.37 [-0.41, 0.59] | 0.021 | 0.019 | 0.327 | 0.502 | 0.503 | 0.023 | 0.209 |
| 8 | +0.13 [-0.09, +0.35] | 0.32 [-0.55, 0.55] | 0.021 | 0.035 | 0.453 | 0.501 | 0.501 | 0.040 | 0.308 |
| 9 | +0.13 [-0.09, +0.35] | 0.33 [-0.54, 0.56] | 0.020 | 0.043 | 0.498 | 0.502 | 0.498 | 0.047 | 0.338 |
| 10 | +0.15 [-0.08, +0.37] | 0.36 [-0.48, 0.59] | 0.020 | 0.044 | 0.601 | 0.504 | 0.502 | 0.047 | 0.391 |
| 11 | +0.13 [-0.10, +0.35] | 0.32 [-0.58, 0.55] | 0.019 | 0.044 | 0.640 | 0.499 | 0.497 | 0.054 | 0.394 |
| 12 | +0.15 [-0.08, +0.36] | 0.36 [-0.47, 0.58] | 0.019 | 0.046 | 0.621 | 0.499 | 0.495 | 0.055 | 0.431 |
| 13 | +0.18 [-0.04, +0.40] | 0.45 [-0.26, 0.64] | 0.018 | 0.051 | 0.583 | 0.500 | 0.501 | 0.059 | 0.438 |
| 14 | +0.16 [-0.07, +0.39] | 0.40 [-0.41, 0.61] | 0.019 | 0.043 | 0.552 | 0.502 | 0.501 | 0.059 | 0.433 |
| 15 | +0.19 [-0.04, +0.41] | 0.47 [-0.23, 0.66] | 0.017 | 0.043 | 0.637 | 0.499 | 0.502 | 0.064 | 0.460 |
| 16 | +0.06 [-0.17, +0.28] | 0.15 [-0.92, 0.45] | 0.018 | 0.046 | 0.676 | 0.501 | 0.505 | 0.059 | 0.452 |
| 17 | -0.03 [-0.26, +0.20] | -0.07 [-1.43, 0.31] | 0.016 | 0.047 | 0.601 | 0.502 | 0.499 | 0.055 | 0.394 |
| 18 | -0.05 [-0.28, +0.17] | -0.12 [-1.59, 0.28] | 0.014 | 0.049 | 0.595 | 0.493 | 0.503 | 0.061 | 0.426 |
| 19 | -0.09 [-0.32, +0.13] | -0.23 [-1.83, 0.21] | 0.013 | 0.057 | 0.649 | 0.497 | 0.496 | 0.074 | 0.491 |
| 20 | -0.09 [-0.32, +0.14] | -0.22 [-1.78, 0.22] | 0.013 | 0.058 | 0.674 | 0.499 | 0.506 | 0.071 | 0.503 |
| 21 | -0.04 [-0.27, +0.19] | -0.11 [-1.52, 0.29] | 0.012 | 0.063 | 0.666 | 0.494 | 0.502 | 0.083 | 0.509 |
| 22 | -0.04 [-0.26, +0.19] | -0.09 [-1.48, 0.31] | 0.012 | 0.069 | 0.673 | 0.496 | 0.508 | 0.091 | 0.518 |
| 23 | -0.03 [-0.25, +0.20] | -0.06 [-1.42, 0.32] | 0.012 | 0.068 | 0.667 | 0.500 | 0.502 | 0.096 | 0.517 |
| 24 | -0.02 [-0.24, +0.21] | -0.04 [-1.37, 0.33] | 0.012 | 0.063 | 0.667 | 0.498 | 0.504 | 0.095 | 0.512 |
| 25 | -0.03 [-0.26, +0.20] | -0.07 [-1.45, 0.31] | 0.013 | 0.059 | 0.654 | 0.504 | 0.500 | 0.094 | 0.498 |

## Single-site vs sequential-scrub eraser at the same site

Each rank-1 eraser maps x to x − ⟨r, x − b⟩ e. |cos| between the single-site eraser at site L (fit on natural site-L states) and the scrub eraser at site L (fit on site-L states after erasure at sites 0..L−1), for the edit direction e (what is removed) and the read direction r (what it is keyed on). Cross-fitted: mean over the 5 folds [min, max]; in-sample: one eraser pair per site.

Site 20 (the post hoc single-site minimum; share remaining single 0.19, scrub 0–20 -0.06, scrub 0–23 -0.02): |cos| cross-fit: edit 0.222 [0.189, 0.244], read 0.191 [0.133, 0.239]; insample: edit 0.216, read 0.193.

So the scrub eraser at site 20 removes a direction nearly orthogonal to the one the single-site eraser removes (edit |cos| 0.222): after erasure at sites 0..19, the class-correlated direction in the fitting states at site 20 is a different one, and the single-site direction is left in place. Of the gap between scrub 0–23 (-0.02) and single site 20 (0.19), the cumulative curve puts -0.25 before site 21 (scrub 0–20 -0.06) and +0.04 in sites 21–23.

| Site | cross-fit edit | cross-fit read | insample edit | insample read |
|---:|---:|---:|---:|---:|
| 0 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 | 1.000 |
| 1 | 0.757 [0.738, 0.785] | 0.920 [0.899, 0.938] | 0.784 | 0.931 |
| 2 | 0.547 [0.500, 0.594] | 0.755 [0.744, 0.767] | 0.558 | 0.772 |
| 3 | 0.660 [0.591, 0.719] | 0.659 [0.638, 0.676] | 0.677 | 0.668 |
| 4 | 0.619 [0.580, 0.695] | 0.556 [0.528, 0.584] | 0.621 | 0.564 |
| 5 | 0.771 [0.755, 0.795] | 0.497 [0.477, 0.529] | 0.776 | 0.503 |
| 6 | 0.667 [0.642, 0.682] | 0.337 [0.280, 0.378] | 0.670 | 0.319 |
| 7 | 0.596 [0.588, 0.624] | 0.234 [0.181, 0.262] | 0.599 | 0.242 |
| 8 | 0.853 [0.841, 0.859] | 0.173 [0.144, 0.216] | 0.856 | 0.163 |
| 9 | 0.918 [0.908, 0.925] | 0.174 [0.101, 0.253] | 0.920 | 0.168 |
| 10 | 0.831 [0.811, 0.843] | 0.154 [0.103, 0.197] | 0.834 | 0.159 |
| 11 | 0.866 [0.862, 0.871] | 0.152 [0.115, 0.215] | 0.868 | 0.153 |
| 12 | 0.897 [0.892, 0.901] | 0.103 [0.043, 0.148] | 0.899 | 0.117 |
| 13 | 0.921 [0.915, 0.926] | 0.165 [0.132, 0.206] | 0.922 | 0.169 |
| 14 | 0.819 [0.812, 0.829] | 0.145 [0.092, 0.193] | 0.821 | 0.187 |
| 15 | 0.711 [0.696, 0.728] | 0.156 [0.138, 0.175] | 0.714 | 0.169 |
| 16 | 0.407 [0.376, 0.435] | 0.124 [0.087, 0.154] | 0.408 | 0.140 |
| 17 | 0.218 [0.177, 0.259] | 0.193 [0.147, 0.269] | 0.209 | 0.190 |
| 18 | 0.061 [0.010, 0.101] | 0.179 [0.137, 0.233] | 0.047 | 0.175 |
| 19 | 0.089 [0.032, 0.120] | 0.225 [0.183, 0.273] | 0.081 | 0.244 |
| 20 | 0.222 [0.189, 0.244] | 0.191 [0.133, 0.239] | 0.216 | 0.193 |
| 21 | 0.097 [0.058, 0.129] | 0.127 [0.077, 0.165] | 0.090 | 0.109 |
| 22 | 0.104 [0.069, 0.144] | 0.147 [0.109, 0.185] | 0.100 | 0.141 |
| 23 | 0.109 [0.065, 0.148] | 0.123 [0.087, 0.140] | 0.106 | 0.123 |
| 24 | 0.024 [0.000, 0.048] | 0.140 [0.093, 0.195] | 0.022 | 0.138 |
| 25 | 0.260 [0.228, 0.289] | 0.143 [0.132, 0.172] | 0.270 | 0.166 |

## By verb band and token pattern

Share of the natural gap remaining (natural gap in parentheses). Pattern: participle token counts of T and I (2+ = two or more).

| Subset | pairs | natural gap | scrub 0–23 | scrub 0–8 | single 8 | single 20 (post hoc) | in-sample scrub 0–23 | in-sample scrub 0–8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| head | 26 | +0.70 [-0.00, +1.38] | -0.10 [-5.68, 2.63] | 0.23 [-3.41, 2.07] | 0.44 [-2.70, 1.78] | 0.41 [-2.67, 2.08] | -0.72 [-9.02, 3.77] | -0.04 [-4.88, 2.53] |
| tail | 50 | +0.47 [+0.13, +0.79] | 0.33 [-1.07, 0.65] | 0.49 [-0.58, 0.73] | 0.49 [-0.57, 0.72] | 0.23 [-1.50, 0.57] | -0.47 [-3.84, 0.17] | 0.21 [-1.49, 0.55] |
| xtail | 50 | +0.19 [-0.12, +0.47] | -0.74 [-17.80, 16.07] | -0.04 [-10.89, 9.04] | 0.31 [-5.30, 7.01] | -0.35 [-13.50, 11.52] | -2.98 [-43.02, 30.81] | -1.01 [-20.23, 14.64] |
| pattern T1-I1 | 41 | +0.62 [+0.13, +1.14] | -0.00 [-2.93, 0.48] | 0.30 [-1.70, 0.62] | 0.44 [-1.32, 0.72] | 0.37 [-1.74, 0.68] | -0.74 [-5.62, 0.06] | -0.01 [-2.86, 0.45] |
| pattern T1-I2+ | 10 | +0.66 [+0.14, +1.25] | 0.61 [-0.36, 0.81] | 0.65 [-0.16, 0.80] | 0.55 [-0.42, 0.81] | 0.32 [-1.49, 0.63] | 0.05 [-2.58, 0.52] | 0.49 [-0.57, 0.71] |
| pattern T2+-I1 | 8 | +0.31 [-0.31, +0.86] | 0.01 [-8.52, 7.80] | 0.28 [-5.87, 6.38] | 0.24 [-6.91, 6.57] | -0.02 [-10.71, 10.01] | -1.46 [-26.69, 19.88] | -0.48 [-14.44, 11.52] |
| pattern T2+-I2+ | 67 | +0.25 [-0.03, +0.51] | -0.31 [-8.19, 4.78] | 0.17 [-4.57, 3.50] | 0.41 [-2.94, 2.86] | -0.10 [-5.77, 4.67] | -1.82 [-17.92, 10.02] | -0.41 [-8.58, 5.18] |

Bootstrap draws that resample no pair of a subset are dropped: none.

## Files

- `reports/pythia/leace_necessity_gaps.csv`: gap, Δ gap, share remaining (with CIs) per condition and subset; KL (all-items rows).
- `reports/pythia/leace_necessity_probes.csv`: probe accuracy per condition, site and probe type.
- `reports/pythia/leace_necessity_fit_diagnostics.csv`: per eraser: n, distinct verbs, top singular value, rank, training cross-covariance residual, held-out class-mean difference before/after.
- `reports/pythia/leace_necessity_references.csv`: the B5 reference numbers.
- `reports/pythia/leace_necessity_cumulative.csv`: the cumulative scrub curve.
- `reports/pythia/leace_necessity_eraser_cosines.csv`: single-site vs scrub eraser |cos| per site.
- `results/gemma/leace_necessity/`: per-prompt log P(by) and KL per condition (`items.csv.gz`), fold assignment, checks, run metadata. `items.csv.gz` is not in git (sha256 34163090c757e4cb0598c490be2de8cfe156f2b2af2c6a17c4d84518a879be21); it is on TSUBAME at `/gs/fs/tga-sip_arase/tyrone/lli_leace/results/pythia/leace_necessity/` (the first run's file next to it as `items_run1.csv.gz`).
- `results/gemma/leace_insample/`: the same files for the in-sample run (`items.csv.gz` not in git, sha256 0e9e23f330d4488f20bc06befa845671265113d9bc62e4086892aec7061ec7a8; on TSUBAME under `lli_leace/results/gemma/leace_insample/`).

