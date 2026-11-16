# LEACE learning curve and downstream re-derivation (Pythia-1.4B, exploratory)

Rules, committed before the runs: `reports/pythia/leace_learning_curve_rules.md`. Scripts: `scripts/run_leace_learning_curve.py`, `scripts/run_leace_postscrub.py` (GPU), `scripts/analyze_leace_learning_curve.py` (this report). Numbers only.

## Flags

- None: every check below passes.

## Checks

- Fit on all 126 pairs through the learning-curve code vs the in-sample run, per-prompt log P(by): max |Δ| natural 8.9e-16, single_8 1.8e-15, scrub_0_12 1.8e-15.
- Natural log P(by) vs the in-sample run: max |Δ| 8.9e-16.
- Probe function (per-prompt predictions) vs `probe_cv` on the first subset's natural site-8 split: 0.9351 vs 0.9351.
- Erasers: rank [1] over 147 fits; top singular value of W Σ_xz per n (single 8 / scrub sites 0–12): full 0.402–0.500; loo 0.402–0.500; n100 0.404–0.500; n25 0.432–0.502; n50 0.419–0.501; n75 0.395–0.500.
- LOO: 126 fits, every pair held out exactly once; every prompt covered once.
- Saved erasers loaded for Parts B / C reproduce each run's per-prompt log P(by) of scrub 0–12: insample 1.8e-15 (natural 8.9e-16), crossfit 1.8e-15 (natural 8.9e-16).
- Site-20 refit on all verbs vs the saved erasers, edit-direction |cos|: insample_fold0 single 1.00000 / scrub 1.00000, crossfit_fold0 single 1.00000 / scrub 1.00000, crossfit_fold1 single 1.00000 / scrub 1.00000, crossfit_fold2 single 1.00000 / scrub 1.00000, crossfit_fold3 single 1.00000 / scrub 1.00000, crossfit_fold4 single 1.00000 / scrub 1.00000.
- Stored probe accuracies (natural at sites 12–23 and erased site 12) reproduced with the same splits: max |Δ| insample 0.0000, crossfit 0.0000.

## Part A: learning curve

Erasers fit on n training pairs (both prompts, all 125 contexts) and evaluated on the held-out pairs (all other pairs). n = 25, 50, 75, 100: 5 random subsets each; the value is the mean over the 5 subsets with a 95% CI (bootstrap, subsets fixed), and the range over subsets in parentheses. LOO: 126 fits on 125 pairs, each evaluated on its held-out pair, pooled over all items. References: the 5-fold cross-fitted run and the in-sample run (fit on all 126 pairs, evaluated on the same items).

### Share of the natural *by* gap remaining (held-out pairs)

| n training pairs | held-out pairs per fit | single 8: share remaining | single 8: KL | scrub 0–12: share remaining | scrub 0–12: KL |
|---|---|---:|---:|---:|---:|
| 25 | 101 | 0.66 [0.44, 0.76] (0.54–0.75) | 0.009 | 0.76 [0.61, 0.83] (0.73–0.81) | 0.014 |
| 50 | 76 | 0.69 [0.46, 0.77] (0.50–0.81) | 0.013 | 0.73 [0.55, 0.81] (0.58–0.83) | 0.018 |
| 75 | 51 | 0.59 [0.03, 0.72] (0.36–0.74) | 0.016 | 0.59 [0.04, 0.71] (0.43–0.71) | 0.023 |
| 100 | 26 | 0.57 [0.05, 0.69] (0.35–0.74) | 0.018 | 0.58 [0.11, 0.72] (0.41–0.69) | 0.028 |
| LOO (125) | 1 | 0.56 [0.32, 0.68] | 0.018 | 0.55 [0.29, 0.67] | 0.028 |
| cross-fitted (5-fold, ~101 pairs) | ~25 | 0.58 [0.34, 0.70] | 0.016 | 0.59 [0.35, 0.70] | 0.024 |
| in-sample (126) | — | 0.34 [-0.01, 0.51] | 0.034 | 0.04 [-0.46, 0.28] | 0.083 |

### Held-out linear probe accuracy (class, T vs I) after erasure

n-subsets: the held-out pairs of each fit are dealt into 5 groups; train on 4, test on the 5th (the eraser saw none of these verbs). **LOO and in-sample rows are leaky** (rules): the erased states the probe trains on were erased by erasers that saw the probe's test pairs, which pushes accuracy toward or below chance; they are not comparable to the n-subset rows. The cross-fitted reference uses nested within-fold splits (not leaky).

| n training pairs | single 8, site 8 | scrub 0–12, site 12 | scrub 0–12, site 8 | natural, site 8 | natural, site 12 |
|---|---:|---:|---:|---:|---:|
| 25 | 0.899 [0.871, 0.924] (0.85–0.93) | 0.937 [0.917, 0.955] (0.90–0.96) | 0.925 [0.903, 0.946] (0.89–0.94) | 0.947 [0.924, 0.967] (0.94–0.96) | 0.960 [0.940, 0.978] (0.95–0.97) |
| 50 | 0.847 [0.821, 0.871] (0.82–0.88) | 0.904 [0.880, 0.926] (0.88–0.94) | 0.895 [0.871, 0.916] (0.88–0.91) | 0.943 [0.920, 0.963] (0.94–0.95) | 0.947 [0.923, 0.966] (0.94–0.96) |
| 75 | 0.740 [0.701, 0.774] (0.64–0.81) | 0.859 [0.829, 0.886] (0.80–0.89) | 0.818 [0.782, 0.848] (0.71–0.86) | 0.928 [0.900, 0.953] (0.91–0.95) | 0.945 [0.921, 0.966] (0.94–0.96) |
| 100 | 0.701 [0.653, 0.747] (0.63–0.76) | 0.773 [0.726, 0.819] (0.66–0.88) | 0.776 [0.726, 0.817] (0.70–0.83) | 0.901 [0.857, 0.936] (0.85–0.94) | 0.904 [0.864, 0.939] (0.86–0.94) |
| LOO (125) (leaky) | 0.273 [0.226, 0.322] | 0.761 [0.718, 0.802] | 0.699 [0.652, 0.743] | 0.956 [0.935, 0.974] | 0.966 [0.946, 0.983] |
| cross-fitted (5-fold, ~101 pairs) | 0.674 | 0.784 | 0.738 | 0.913 | 0.932 |
| in-sample (126) (leaky) | 0.001 | 0.013 | 0.001 | 0.956 | 0.966 |

## Part B: site 20, split by verb token count

Subset by the prompt's participle: single-token or multi-token (verb level). Erasers at site 20 refit on the subset: single-site on natural states, post-scrub on states after the saved all-verb erasers 0..19. |cos| of edit and read directions (cross-fitted: mean over the 5 folds [min, max]). Probe: linear, trained on all verbs as in the necessity runs, on site-20 states after scrubbing 0–19 (before eraser 20); accuracy over the subset's test prompts. Cross-fitted probes: nested within-fold splits; in-sample probes: verb-held-out split (leaky).

| Fitting | Subset | T / I verbs | prompts | |cos| edit | |cos| read | probe acc after scrub 0–19 | probe acc, natural |
|---|---|---:|---:|---:|---:|---:|---:|
| cross-fitted | all | 126 / 126 | 31500 | 0.012 [0.001, 0.022] | 0.142 [0.111, 0.160] | 0.775 [0.734, 0.816] | 0.898 [0.865, 0.928] |
| cross-fitted | single | 32 / 33 | 8125 | 0.686 [0.647, 0.717] | 0.635 [0.610, 0.655] | 0.813 [0.720, 0.894] | 0.934 [0.876, 0.978] |
| cross-fitted | multi | 94 / 93 | 23375 | 0.033 [0.009, 0.070] | 0.201 [0.186, 0.239] | 0.761 [0.710, 0.808] | 0.886 [0.845, 0.920] |
| in-sample (leaky probe) | all | 126 / 126 | 31500 | 0.007 | 0.144 | 0.207 [0.174, 0.242] | 0.952 [0.929, 0.971] |
| in-sample (leaky probe) | single | 32 / 33 | 8125 | 0.669 | 0.623 | 0.107 [0.062, 0.159] | 0.993 [0.986, 0.999] |
| in-sample (leaky probe) | multi | 94 / 93 | 23375 | 0.049 | 0.190 | 0.242 [0.200, 0.286] | 0.937 [0.907, 0.964] |

## Part C: class probes downstream of scrub 0–12

In-sample scrub 0–12 leaves 0.04 [-0.46, 0.28] of the *by* gap; cross-fitted scrub 0–12 leaves 0.59 [0.35, 0.70]. States at sites 13–23 after scrub 0–12 (no erasure there); site 12 is the last erased site (for the in-sample split it is the leakage gauge). Linear probe, verb-held-out: in-sample with the in-sample run's 5-group split (leaky), cross-fitted with nested within-fold splits (leak-free supplement). Accuracy with 95% CIs.

| Site | in-sample: after scrub 0–12 | in-sample split: natural | cross-fitted: after scrub 0–12 | cross-fitted split: natural |
|---:|---:|---:|---:|---:|
| 12 (erased) | 0.013 [0.004, 0.024] | 0.966 [0.946, 0.983] | 0.784 [0.745, 0.826] | 0.932 [0.906, 0.955] |
| 13 | 0.167 [0.135, 0.201] | 0.960 [0.938, 0.979] | 0.795 [0.756, 0.836] | 0.931 [0.903, 0.954] |
| 14 | 0.270 [0.229, 0.311] | 0.958 [0.935, 0.977] | 0.809 [0.770, 0.850] | 0.926 [0.896, 0.952] |
| 15 | 0.334 [0.292, 0.375] | 0.956 [0.933, 0.976] | 0.809 [0.768, 0.851] | 0.923 [0.894, 0.950] |
| 16 | 0.401 [0.357, 0.447] | 0.956 [0.933, 0.975] | 0.812 [0.771, 0.853] | 0.916 [0.884, 0.945] |
| 17 | 0.428 [0.385, 0.474] | 0.952 [0.929, 0.971] | 0.808 [0.767, 0.849] | 0.909 [0.877, 0.937] |
| 18 | 0.451 [0.409, 0.497] | 0.952 [0.930, 0.972] | 0.801 [0.760, 0.843] | 0.902 [0.870, 0.931] |
| 19 | 0.465 [0.424, 0.512] | 0.951 [0.930, 0.971] | 0.799 [0.758, 0.841] | 0.900 [0.867, 0.930] |
| 20 | 0.484 [0.442, 0.528] | 0.952 [0.929, 0.971] | 0.793 [0.751, 0.835] | 0.898 [0.865, 0.928] |
| 21 | 0.489 [0.448, 0.533] | 0.951 [0.929, 0.971] | 0.793 [0.753, 0.831] | 0.896 [0.863, 0.926] |
| 22 | 0.492 [0.452, 0.533] | 0.950 [0.927, 0.970] | 0.790 [0.749, 0.829] | 0.894 [0.860, 0.924] |
| 23 | 0.493 [0.454, 0.533] | 0.950 [0.927, 0.971] | 0.786 [0.746, 0.825] | 0.889 [0.855, 0.920] |

## Files

- `reports/pythia/leace_learning_curve_summary.csv`: Part A per n (share remaining, KL, probe accuracy, CIs, subset ranges).
- `reports/pythia/leace_learning_curve_per_fit.csv`: Part A per subset fit.
- `reports/pythia/leace_learning_curve_site20_split.csv`: Part B.
- `reports/pythia/leace_learning_curve_downstream_probes.csv`: Part C.
- `results/pythia/leace_learning_curve/` and `results/pythia/leace_postscrub/`: fits, fit diagnostics, checks, metadata, cosines; per-prompt files (`held.csv.gz`, `loo_probes.csv.gz`, `preds.csv.gz`, `prompts.csv.gz`) and `states.npz` not in git (on TSUBAME under `lli_leace/`).

