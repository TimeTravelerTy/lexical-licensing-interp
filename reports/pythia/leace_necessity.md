# LEACE necessity of verb class at the verb position (Pythia-1.4B)

Scripts: `scripts/run_leace_necessity.py` (GPU), `scripts/analyze_leace_necessity.py` (this report). Run: 59 min, concept-erasure 0.2.4 (wheel), fp32 (TF32 off).

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
- Scrub replay (training items, all 24 erasers applied): pre-erasure mean at each site vs the mean each eraser was fit on, max |Δ| = 1.1e-13; cross-covariance relative difference max 1.8e-15; post-erasure / pre-erasure cross-covariance norm, max over sites and folds 7.5e-08.
- Erasers: top singular value of W Σ_xz min 0.401 (svd_tol 0.01: 0 fits below it, i.e. no erasure); erased rank [1.0]; trace-constraint branch used in 0 fits; training cross-covariance residual max 3.6e-14.
- Held-out class-mean difference, erased / natural norm: single_L 0.633–0.996; scrub_all 0.747–0.996 (range over sites and folds).
- Natural fp32 log P(by) vs the stored bf16 task-1 readout (`results/passive_das_prep/pythia14b_readout.csv`, sha256 3158ed4b04b5…): median |Δ| 0.047, 99th pct 0.253, max 0.675; gap from the stored readout +0.77 [+0.50, +1.02] vs fp32 +0.77 [+0.51, +1.02].
- This cross-fitted run vs the first run of the same job (per-prompt log P(by) and KL, 53 shared columns; first run's items.csv.gz sha256 e4564ab03750…): max |Δ| = 0.
- In-sample run (35 min): n = 31,500 fitting states per eraser (252 verbs; n/d = 15.4); top singular value min 0.402, erased rank [1.0], trace branch used in 0 fits; same checks as above: identity hook 0, non-anchor 0, in-model vs offline 0, scrub replay mean 3.1e-08 / cross-covariance 9.5e-08; natural log P(by) vs the cross-fitted run's max |Δ| 0.
- In-sample, on the evaluated items (= the fitting items): cross-covariance with class after / before erasure, max over scrub sites 9.6e-08, single_L max 4.2e-14; class-mean difference after / before erasure: single_L max 0.00019, scrub 0–23 max 0.00019, scrub 0–8 (erased sites) max 0.00019.

## Flags

- Held-out linear probe after erasure is not at chance: 0.587–0.793 over every erased site and condition (chance 0.5; natural 0.615–0.932).
- Held-out MLP probe after erasure is not at chance: 0.591–0.778 over every erased site and condition (chance 0.5; natural 0.615–0.922).
- Readout proximity: *by* is predicted at the verb position itself, so late sites sit next to the readout, and erasure there partly removes the readout computation itself rather than only verb-class information the readout uses. Sites 0–~12 are the necessity-relevant range; rows for later sites (including the post hoc single-site minimum) do not test necessity in that sense.
- In-sample erasure, linear probe trained and scored on all (= fitting) items: 0.495–0.505 over every erased site and condition (chance 0.5).
- In-sample erasure, verb-held-out linear probe (eraser saw the test verbs): 0.000–0.234 over every erased site and condition (chance 0.5).
- In-sample erasure, verb-held-out MLP probe (eraser saw the test verbs): 0.023–0.620 over every erased site and condition (chance 0.5).
- In-sample verb-held-out probes can fall below chance: the eraser zeroes the class cross-covariance over all verbs together, so whatever class signal remains in the probe's training verbs points the opposite way in its test verbs (the same effect as the pooled cross-fold probe in the first smoke test).

## Main table

Probe columns: held-out accuracy at the erased site (scrubbing: range over the erased sites); for the in-sample rows the test verbs are held out from the probe but not from the eraser. Rows for sites after ~12 (scrub 0–23, the post hoc single-site minimum) include erasure next to the readout (see Flags). The B5 reference rows are from `reports/round5/b5_necessity.md` (frame `hasbeen`, all 64 pairs; different items and frame: B7 items, "The N has been V"), converted to share remaining = 1 − share removed.

| Condition | *by* gap (nats) | share of natural gap remaining | KL (nats) | linear probe | MLP probe |
|---|---:|---:|---:|---:|---:|
| natural | +0.77 [+0.51, +1.02] | 1.00 [1.00, 1.00] | 0.000 |  |  |
| scrub, sites 0–23 | +0.44 [+0.17, +0.68] | 0.57 [0.33, 0.69] | 0.035 | 0.590–0.793 | 0.596–0.778 |
| scrub, sites 0–12 | +0.45 [+0.17, +0.70] | 0.59 [0.35, 0.70] | 0.024 | 0.590–0.784 | 0.596–0.774 |
| scrub, sites 0–8 | +0.49 [+0.21, +0.74] | 0.63 [0.43, 0.73] | 0.021 | 0.590–0.738 | 0.596–0.739 |
| single site 8 (DAS site) | +0.45 [+0.18, +0.69] | 0.58 [0.34, 0.70] | 0.016 | 0.674 | 0.681 |
| single site 20: lowest of 24 (post hoc; pointwise CI) | +0.22 [-0.05, +0.48] | 0.29 [-0.11, 0.47] | 0.014 | 0.663 | 0.687 |
| in-sample: scrub, sites 0–23 | -0.03 [-0.30, +0.22] | -0.04 [-0.58, 0.22] | 0.102 | 0.000–0.234 | 0.023–0.337 |
| in-sample: scrub, sites 0–12 | +0.03 [-0.24, +0.28] | 0.04 [-0.46, 0.28] | 0.083 | 0.000–0.234 | 0.023–0.230 |
| in-sample: scrub, sites 0–8 | +0.12 [-0.15, +0.37] | 0.16 [-0.27, 0.37] | 0.070 | 0.000–0.234 | 0.023–0.230 |
| in-sample: single site 8 (DAS site) | +0.26 [-0.01, +0.51] | 0.34 [-0.01, 0.51] | 0.034 | 0.001 | 0.200 |
| in-sample: single site 21: lowest of 24 (post hoc; pointwise CI) | -0.04 [-0.32, +0.22] | -0.06 [-0.64, 0.22] | 0.021 | 0.028 | 0.620 |
| ref (B5): d -> I mean (DAS site 8 coordinate) |  | 0.85 [0.69, 0.91] |  |  |  |
| ref (B5): top-50 switch neurons -> I mean |  | 0.91 [0.85, 0.96] |  |  |  |
| ref (B5): both (d + top-50 neurons) |  | 0.81 [0.66, 0.87] |  |  |  |

## Single-site LEACE, every site

Natural linear / MLP probe accuracy at the same site in the last two columns.

| Site | *by* gap | share remaining | KL | linear probe | MLP probe | natural linear | natural MLP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | +0.76 [+0.48, +1.00] | 0.98 [0.96, 0.99] | 0.001 | 0.619 | 0.607 | 0.615 | 0.615 |
| 1 | +0.75 [+0.48, +1.00] | 0.97 [0.94, 0.99] | 0.003 | 0.594 | 0.591 | 0.645 | 0.647 |
| 2 | +0.72 [+0.45, +0.97] | 0.94 [0.88, 0.97] | 0.005 | 0.587 | 0.614 | 0.689 | 0.688 |
| 3 | +0.67 [+0.41, +0.92] | 0.87 [0.79, 0.91] | 0.006 | 0.628 | 0.620 | 0.771 | 0.754 |
| 4 | +0.60 [+0.32, +0.85] | 0.77 [0.63, 0.84] | 0.009 | 0.668 | 0.655 | 0.851 | 0.829 |
| 5 | +0.56 [+0.28, +0.81] | 0.73 [0.56, 0.80] | 0.012 | 0.681 | 0.684 | 0.886 | 0.872 |
| 6 | +0.51 [+0.23, +0.76] | 0.66 [0.46, 0.76] | 0.014 | 0.653 | 0.679 | 0.901 | 0.894 |
| 7 | +0.48 [+0.21, +0.73] | 0.63 [0.41, 0.73] | 0.015 | 0.661 | 0.665 | 0.896 | 0.882 |
| 8 | +0.45 [+0.18, +0.69] | 0.58 [0.34, 0.70] | 0.016 | 0.674 | 0.681 | 0.913 | 0.895 |
| 9 | +0.46 [+0.19, +0.71] | 0.60 [0.37, 0.71] | 0.015 | 0.677 | 0.697 | 0.924 | 0.913 |
| 10 | +0.47 [+0.19, +0.71] | 0.61 [0.38, 0.72] | 0.015 | 0.677 | 0.703 | 0.928 | 0.920 |
| 11 | +0.45 [+0.18, +0.69] | 0.58 [0.35, 0.69] | 0.016 | 0.682 | 0.700 | 0.925 | 0.912 |
| 12 | +0.45 [+0.18, +0.70] | 0.59 [0.36, 0.70] | 0.014 | 0.676 | 0.698 | 0.932 | 0.922 |
| 13 | +0.45 [+0.19, +0.70] | 0.59 [0.37, 0.70] | 0.013 | 0.689 | 0.707 | 0.931 | 0.915 |
| 14 | +0.41 [+0.14, +0.66] | 0.53 [0.28, 0.66] | 0.012 | 0.690 | 0.708 | 0.926 | 0.908 |
| 15 | +0.36 [+0.09, +0.61] | 0.46 [0.17, 0.61] | 0.012 | 0.685 | 0.701 | 0.923 | 0.906 |
| 16 | +0.33 [+0.05, +0.58] | 0.42 [0.11, 0.57] | 0.013 | 0.688 | 0.704 | 0.916 | 0.903 |
| 17 | +0.30 [+0.02, +0.56] | 0.39 [0.05, 0.55] | 0.013 | 0.677 | 0.701 | 0.909 | 0.893 |
| 18 | +0.27 [-0.00, +0.52] | 0.34 [-0.01, 0.52] | 0.013 | 0.660 | 0.690 | 0.902 | 0.888 |
| 19 | +0.25 [-0.02, +0.51] | 0.32 [-0.05, 0.50] | 0.014 | 0.666 | 0.689 | 0.900 | 0.884 |
| 20 | +0.22 [-0.05, +0.48] | 0.29 [-0.11, 0.47] | 0.014 | 0.663 | 0.687 | 0.898 | 0.883 |
| 21 | +0.22 [-0.06, +0.48] | 0.29 [-0.11, 0.48] | 0.014 | 0.662 | 0.684 | 0.896 | 0.880 |
| 22 | +0.24 [-0.04, +0.51] | 0.31 [-0.07, 0.50] | 0.015 | 0.661 | 0.686 | 0.894 | 0.876 |
| 23 | +0.26 [-0.02, +0.52] | 0.33 [-0.04, 0.51] | 0.015 | 0.663 | 0.687 | 0.889 | 0.876 |

## Probe accuracy under scrubbing, by site

`scrub_0_8` is erased at sites 0–8 only; its later sites are unerased (states downstream of the erasure). Probes stop at site 23, the input to the last layer; the state after the last layer (which feeds the readout) is not probed.

| Site | natural lin | natural MLP | scrub 0–23 lin | scrub 0–23 MLP | scrub 0–8 lin | scrub 0–8 MLP |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.615 | 0.615 | 0.619 | 0.607 | 0.619 | 0.607 |
| 1 | 0.645 | 0.647 | 0.590 | 0.596 | 0.590 | 0.596 |
| 2 | 0.689 | 0.688 | 0.599 | 0.620 | 0.599 | 0.620 |
| 3 | 0.771 | 0.754 | 0.640 | 0.629 | 0.640 | 0.629 |
| 4 | 0.851 | 0.829 | 0.683 | 0.668 | 0.683 | 0.668 |
| 5 | 0.886 | 0.872 | 0.690 | 0.690 | 0.690 | 0.690 |
| 6 | 0.901 | 0.894 | 0.693 | 0.710 | 0.693 | 0.710 |
| 7 | 0.896 | 0.882 | 0.717 | 0.715 | 0.717 | 0.715 |
| 8 | 0.913 | 0.895 | 0.738 | 0.739 | 0.738 | 0.739 |
| 9 | 0.924 | 0.913 | 0.761 | 0.763 | 0.779 | 0.777 |
| 10 | 0.928 | 0.920 | 0.769 | 0.774 | 0.790 | 0.790 |
| 11 | 0.925 | 0.912 | 0.771 | 0.761 | 0.798 | 0.782 |
| 12 | 0.932 | 0.922 | 0.784 | 0.773 | 0.819 | 0.803 |
| 13 | 0.931 | 0.915 | 0.787 | 0.771 | 0.825 | 0.803 |
| 14 | 0.926 | 0.908 | 0.793 | 0.778 | 0.828 | 0.809 |
| 15 | 0.923 | 0.906 | 0.792 | 0.775 | 0.832 | 0.808 |
| 16 | 0.916 | 0.903 | 0.793 | 0.776 | 0.830 | 0.813 |
| 17 | 0.909 | 0.893 | 0.787 | 0.769 | 0.828 | 0.808 |
| 18 | 0.902 | 0.888 | 0.780 | 0.767 | 0.820 | 0.806 |
| 19 | 0.900 | 0.884 | 0.777 | 0.766 | 0.817 | 0.803 |
| 20 | 0.898 | 0.883 | 0.771 | 0.763 | 0.814 | 0.798 |
| 21 | 0.896 | 0.880 | 0.770 | 0.762 | 0.813 | 0.798 |
| 22 | 0.894 | 0.876 | 0.766 | 0.759 | 0.810 | 0.794 |
| 23 | 0.889 | 0.876 | 0.759 | 0.756 | 0.806 | 0.792 |

## In-sample erasure, every site

Erasers fit on all items and evaluated on them. Single-site rows: `insample_single_L`; probe columns as in the main table, plus the fit-set linear probe. Last three columns: probe accuracy under in-sample scrubbing 0–23 at that site.

| Site | *by* gap | share remaining | KL | linear probe | MLP probe | fit-set linear | scrub 0–23 fit-set lin | scrub 0–23 lin | scrub 0–23 MLP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | +0.74 [+0.46, +0.99] | 0.95 [0.91, 0.98] | 0.003 | 0.234 | 0.230 | 0.500 | 0.500 | 0.234 | 0.230 |
| 1 | +0.71 [+0.43, +0.95] | 0.91 [0.86, 0.94] | 0.004 | 0.046 | 0.285 | 0.497 | 0.503 | 0.036 | 0.140 |
| 2 | +0.63 [+0.35, +0.87] | 0.81 [0.70, 0.87] | 0.009 | 0.006 | 0.074 | 0.498 | 0.498 | 0.004 | 0.041 |
| 3 | +0.52 [+0.25, +0.76] | 0.68 [0.49, 0.76] | 0.017 | 0.003 | 0.044 | 0.503 | 0.495 | 0.002 | 0.023 |
| 4 | +0.41 [+0.13, +0.66] | 0.53 [0.26, 0.65] | 0.027 | 0.000 | 0.047 | 0.502 | 0.501 | 0.000 | 0.024 |
| 5 | +0.37 [+0.09, +0.61] | 0.47 [0.19, 0.61] | 0.030 | 0.000 | 0.052 | 0.500 | 0.498 | 0.000 | 0.030 |
| 6 | +0.31 [+0.04, +0.56] | 0.40 [0.08, 0.56] | 0.033 | 0.002 | 0.090 | 0.503 | 0.499 | 0.004 | 0.039 |
| 7 | +0.30 [+0.03, +0.55] | 0.38 [0.06, 0.54] | 0.034 | 0.003 | 0.132 | 0.500 | 0.502 | 0.004 | 0.051 |
| 8 | +0.26 [-0.01, +0.51] | 0.34 [-0.01, 0.51] | 0.034 | 0.001 | 0.200 | 0.497 | 0.501 | 0.001 | 0.078 |
| 9 | +0.28 [+0.01, +0.52] | 0.36 [0.02, 0.52] | 0.031 | 0.005 | 0.253 | 0.500 | 0.499 | 0.006 | 0.119 |
| 10 | +0.30 [+0.04, +0.55] | 0.39 [0.08, 0.55] | 0.028 | 0.006 | 0.306 | 0.502 | 0.499 | 0.007 | 0.143 |
| 11 | +0.29 [+0.02, +0.53] | 0.37 [0.04, 0.53] | 0.028 | 0.008 | 0.346 | 0.498 | 0.496 | 0.012 | 0.163 |
| 12 | +0.29 [+0.02, +0.54] | 0.38 [0.04, 0.53] | 0.023 | 0.008 | 0.399 | 0.497 | 0.503 | 0.013 | 0.183 |
| 13 | +0.29 [+0.02, +0.54] | 0.37 [0.04, 0.53] | 0.022 | 0.010 | 0.433 | 0.499 | 0.496 | 0.014 | 0.197 |
| 14 | +0.23 [-0.04, +0.48] | 0.29 [-0.09, 0.47] | 0.020 | 0.014 | 0.486 | 0.499 | 0.500 | 0.014 | 0.228 |
| 15 | +0.15 [-0.13, +0.40] | 0.19 [-0.25, 0.40] | 0.019 | 0.014 | 0.538 | 0.500 | 0.500 | 0.016 | 0.251 |
| 16 | +0.10 [-0.17, +0.36] | 0.13 [-0.35, 0.35] | 0.019 | 0.016 | 0.550 | 0.505 | 0.500 | 0.016 | 0.265 |
| 17 | +0.06 [-0.21, +0.32] | 0.08 [-0.42, 0.32] | 0.020 | 0.019 | 0.566 | 0.495 | 0.501 | 0.019 | 0.278 |
| 18 | +0.02 [-0.25, +0.28] | 0.03 [-0.50, 0.28] | 0.020 | 0.021 | 0.591 | 0.499 | 0.498 | 0.021 | 0.302 |
| 19 | -0.00 [-0.28, +0.26] | -0.00 [-0.55, 0.26] | 0.020 | 0.023 | 0.603 | 0.498 | 0.499 | 0.020 | 0.314 |
| 20 | -0.03 [-0.31, +0.23] | -0.04 [-0.62, 0.23] | 0.021 | 0.026 | 0.613 | 0.501 | 0.499 | 0.021 | 0.319 |
| 21 | -0.04 [-0.32, +0.22] | -0.06 [-0.64, 0.22] | 0.021 | 0.028 | 0.620 | 0.498 | 0.500 | 0.024 | 0.331 |
| 22 | -0.02 [-0.31, +0.24] | -0.03 [-0.62, 0.24] | 0.021 | 0.028 | 0.620 | 0.500 | 0.500 | 0.023 | 0.332 |
| 23 | +0.01 [-0.26, +0.27] | 0.02 [-0.52, 0.26] | 0.022 | 0.027 | 0.612 | 0.498 | 0.496 | 0.024 | 0.337 |

## Cumulative scrub curve

Scrubbing sites 0..k (the first k + 1 erasers of the sequential fit), share of the natural gap remaining and KL, k = 0..23. Sites after ~12 sit next to the readout (see Flags).

| k | cross-fitted: share remaining | cross-fitted: KL | in-sample: share remaining | in-sample: KL |
|---:|---:|---:|---:|---:|
| 0 | 0.98 [0.96, 0.99] | 0.001 | 0.95 [0.91, 0.98] | 0.003 |
| 1 | 0.95 [0.92, 0.98] | 0.003 | 0.89 [0.82, 0.93] | 0.006 |
| 2 | 0.92 [0.86, 0.95] | 0.006 | 0.79 [0.66, 0.85] | 0.011 |
| 3 | 0.85 [0.76, 0.90] | 0.009 | 0.65 [0.45, 0.74] | 0.022 |
| 4 | 0.76 [0.62, 0.83] | 0.013 | 0.46 [0.16, 0.60] | 0.038 |
| 5 | 0.73 [0.58, 0.81] | 0.015 | 0.38 [0.04, 0.53] | 0.047 |
| 6 | 0.69 [0.51, 0.77] | 0.017 | 0.28 [-0.11, 0.46] | 0.057 |
| 7 | 0.66 [0.47, 0.76] | 0.019 | 0.23 [-0.18, 0.42] | 0.063 |
| 8 | 0.63 [0.43, 0.73] | 0.021 | 0.16 [-0.27, 0.37] | 0.070 |
| 9 | 0.62 [0.40, 0.72] | 0.022 | 0.12 [-0.34, 0.34] | 0.074 |
| 10 | 0.61 [0.39, 0.72] | 0.022 | 0.11 [-0.36, 0.33] | 0.077 |
| 11 | 0.60 [0.38, 0.71] | 0.023 | 0.08 [-0.41, 0.30] | 0.079 |
| 12 | 0.59 [0.35, 0.70] | 0.024 | 0.04 [-0.46, 0.28] | 0.083 |
| 13 | 0.59 [0.35, 0.70] | 0.025 | 0.04 [-0.46, 0.27] | 0.086 |
| 14 | 0.56 [0.30, 0.68] | 0.026 | -0.05 [-0.59, 0.21] | 0.090 |
| 15 | 0.54 [0.27, 0.66] | 0.026 | -0.10 [-0.68, 0.17] | 0.092 |
| 16 | 0.53 [0.26, 0.66] | 0.027 | -0.14 [-0.72, 0.14] | 0.096 |
| 17 | 0.53 [0.27, 0.66] | 0.028 | -0.13 [-0.72, 0.15] | 0.098 |
| 18 | 0.52 [0.26, 0.65] | 0.029 | -0.16 [-0.75, 0.13] | 0.099 |
| 19 | 0.52 [0.26, 0.65] | 0.030 | -0.17 [-0.77, 0.11] | 0.101 |
| 20 | 0.52 [0.25, 0.65] | 0.031 | -0.20 [-0.81, 0.10] | 0.102 |
| 21 | 0.52 [0.26, 0.65] | 0.032 | -0.19 [-0.81, 0.10] | 0.103 |
| 22 | 0.54 [0.28, 0.67] | 0.033 | -0.16 [-0.78, 0.13] | 0.102 |
| 23 | 0.57 [0.33, 0.69] | 0.035 | -0.04 [-0.58, 0.22] | 0.102 |

## Single-site vs sequential-scrub eraser at the same site

Each rank-1 eraser maps x to x − ⟨r, x − b⟩ e. |cos| between the single-site eraser at site L (fit on natural site-L states) and the scrub eraser at site L (fit on site-L states after erasure at sites 0..L−1), for the edit direction e (what is removed) and the read direction r (what it is keyed on). Cross-fitted: mean over the 5 folds [min, max]; in-sample: one eraser pair per site.

Site 20 (the post hoc single-site minimum; share remaining single 0.29, scrub 0–20 0.52, scrub 0–23 0.57): |cos| cross-fit: edit 0.012 [0.001, 0.022], read 0.142 [0.111, 0.160]; insample: edit 0.007, read 0.144.

So the scrub eraser at site 20 removes a direction nearly orthogonal to the one the single-site eraser removes (edit |cos| 0.012): after erasure at sites 0..19, the class-correlated direction in the fitting states at site 20 is a different one, and the single-site direction is left in place. Of the gap between scrub 0–23 (0.57) and single site 20 (0.29), the cumulative curve puts 0.23 before site 21 (scrub 0–20 0.52) and +0.05 in sites 21–23.

| Site | cross-fit edit | cross-fit read | insample edit | insample read |
|---:|---:|---:|---:|---:|
| 0 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 | 1.000 |
| 1 | 0.793 [0.773, 0.811] | 0.904 [0.893, 0.923] | 0.796 | 0.918 |
| 2 | 0.661 [0.645, 0.670] | 0.614 [0.591, 0.638] | 0.668 | 0.627 |
| 3 | 0.667 [0.654, 0.686] | 0.534 [0.522, 0.543] | 0.681 | 0.548 |
| 4 | 0.650 [0.639, 0.655] | 0.496 [0.475, 0.525] | 0.662 | 0.502 |
| 5 | 0.471 [0.455, 0.483] | 0.340 [0.311, 0.358] | 0.477 | 0.342 |
| 6 | 0.473 [0.463, 0.480] | 0.335 [0.323, 0.353] | 0.478 | 0.341 |
| 7 | 0.424 [0.410, 0.441] | 0.312 [0.292, 0.347] | 0.424 | 0.317 |
| 8 | 0.394 [0.377, 0.413] | 0.307 [0.281, 0.331] | 0.394 | 0.309 |
| 9 | 0.349 [0.318, 0.365] | 0.239 [0.214, 0.265] | 0.348 | 0.227 |
| 10 | 0.248 [0.238, 0.259] | 0.240 [0.201, 0.270] | 0.246 | 0.232 |
| 11 | 0.238 [0.230, 0.246] | 0.270 [0.247, 0.282] | 0.235 | 0.266 |
| 12 | 0.313 [0.298, 0.347] | 0.283 [0.258, 0.309] | 0.311 | 0.264 |
| 13 | 0.218 [0.198, 0.234] | 0.283 [0.249, 0.326] | 0.214 | 0.280 |
| 14 | 0.203 [0.188, 0.227] | 0.258 [0.247, 0.271] | 0.199 | 0.263 |
| 15 | 0.137 [0.116, 0.169] | 0.219 [0.183, 0.242] | 0.130 | 0.213 |
| 16 | 0.128 [0.111, 0.143] | 0.196 [0.157, 0.233] | 0.123 | 0.197 |
| 17 | 0.114 [0.094, 0.147] | 0.160 [0.148, 0.170] | 0.106 | 0.151 |
| 18 | 0.058 [0.032, 0.080] | 0.130 [0.112, 0.147] | 0.053 | 0.117 |
| 19 | 0.127 [0.108, 0.150] | 0.160 [0.129, 0.189] | 0.122 | 0.156 |
| 20 | 0.012 [0.001, 0.022] | 0.142 [0.111, 0.160] | 0.007 | 0.144 |
| 21 | 0.090 [0.080, 0.105] | 0.125 [0.092, 0.163] | 0.090 | 0.131 |
| 22 | 0.012 [0.002, 0.026] | 0.111 [0.064, 0.156] | 0.005 | 0.104 |
| 23 | 0.104 [0.078, 0.114] | 0.067 [0.026, 0.092] | 0.103 | 0.060 |

## By verb band and token pattern

Share of the natural gap remaining (natural gap in parentheses). Pattern: participle token counts of T and I (2+ = two or more).

| Subset | pairs | natural gap | scrub 0–23 | scrub 0–8 | single 8 | single 20 (post hoc) | in-sample scrub 0–23 | in-sample scrub 0–8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| head | 26 | +0.94 [+0.21, +1.64] | 0.62 [-0.52, 0.86] | 0.61 [-0.71, 0.83] | 0.54 [-0.88, 0.75] | 0.25 [-2.00, 0.60] | 0.14 [-2.37, 0.51] | 0.25 [-2.08, 0.58] |
| tail | 50 | +0.97 [+0.58, +1.34] | 0.58 [0.27, 0.73] | 0.66 [0.42, 0.78] | 0.62 [0.34, 0.74] | 0.42 [-0.00, 0.59] | 0.14 [-0.50, 0.41] | 0.31 [-0.20, 0.52] |
| xtail | 50 | +0.49 [+0.11, +0.82] | 0.49 [-0.78, 0.75] | 0.59 [-0.41, 0.80] | 0.56 [-0.54, 0.78] | 0.06 [-2.32, 0.50] | -0.58 [-4.24, 0.10] | -0.22 [-2.99, 0.31] |
| pattern T1-I1 | 27 | +0.99 [+0.27, +1.65] | 0.65 [-0.15, 0.87] | 0.64 [-0.38, 0.83] | 0.57 [-0.46, 0.76] | 0.28 [-1.65, 0.60] | 0.20 [-1.67, 0.54] | 0.30 [-1.40, 0.60] |
| pattern T1-I2+ | 5 | +0.97 [-0.76, +2.47] | 0.56 [-4.28, 4.60] | 0.54 [-3.84, 4.19] | 0.61 [-3.56, 3.87] | 0.35 [-6.40, 5.34] | 0.07 [-8.62, 7.54] | 0.23 [-6.91, 6.19] |
| pattern T2+-I1 | 6 | +0.89 [-0.28, +1.75] | 0.45 [-3.83, 2.41] | 0.54 [-3.73, 2.50] | 0.45 [-4.93, 2.85] | 0.34 [-6.12, 3.05] | -0.07 [-9.47, 3.75] | 0.07 [-7.80, 3.19] |
| pattern T2+-I2+ | 88 | +0.69 [+0.40, +0.95] | 0.54 [0.21, 0.68] | 0.64 [0.39, 0.76] | 0.60 [0.31, 0.72] | 0.28 [-0.24, 0.50] | -0.15 [-0.95, 0.18] | 0.10 [-0.50, 0.36] |

Bootstrap draws that resample no pair of a subset are dropped: pattern T1-I2+ 995/1000 valid, pattern T2+-I1 999/1000 valid

## Files

- `reports/pythia/leace_necessity_gaps.csv`: gap, Δ gap, share remaining (with CIs) per condition and subset; KL (all-items rows).
- `reports/pythia/leace_necessity_probes.csv`: probe accuracy per condition, site and probe type.
- `reports/pythia/leace_necessity_fit_diagnostics.csv`: per eraser: n, distinct verbs, top singular value, rank, training cross-covariance residual, held-out class-mean difference before/after.
- `reports/pythia/leace_necessity_references.csv`: the B5 reference numbers.
- `reports/pythia/leace_necessity_cumulative.csv`: the cumulative scrub curve.
- `reports/pythia/leace_necessity_eraser_cosines.csv`: single-site vs scrub eraser |cos| per site.
- `results/pythia/leace_necessity/`: per-prompt log P(by) and KL per condition (`items.csv.gz`), fold assignment, checks, run metadata. `items.csv.gz` is not in git (sha256 fc8b9c41f943e5c573f476cfec0e8ab2bcf9b30c8f43bdde75abe3a486eef14a); it is on TSUBAME at `/gs/fs/tga-sip_arase/tyrone/lli_leace/results/pythia/leace_necessity/` (the first run's file next to it as `items_run1.csv.gz`).
- `results/pythia/leace_insample/`: the same files for the in-sample run (`items.csv.gz` not in git, sha256 4a0eb9bfb82e7e96150c1a5802b9f782324ea83c2c75478163fcc88d94fc775d; on TSUBAME under `lli_leace/results/pythia/leace_insample/`).

