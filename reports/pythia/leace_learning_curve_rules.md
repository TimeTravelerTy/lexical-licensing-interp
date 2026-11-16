# LEACE learning curve and downstream re-derivation: rules (Pythia-1.4B, exploratory)

Written and committed before any of these runs. Exploratory: no decision rule rides on these numbers. The reports state
numbers only. They add no interpretation beyond the share of the gap remaining and the probe accuracies.

Shared with `reports/pythia/leace_necessity.md`:

- **Items:** 126 verb pairs × 125 contexts. Each item has a T prompt and an I prompt.
- **Position:** the verb's last token, which is both the erasure position and the readout position.
- **Sites:** site L is the residual stream entering layer L.
- **Readout:** log P(" by").
- **Gap:** the mean of log P(by | T) − log P(by | I) over items.
- **Model settings:** fp32 with TF32 off.
- **LEACE settings:** `LeaceFitter` defaults, float64 statistics, rank 1, concept = class (T = 1, I = 0).
- **Sequential scrub:** eraser L is fit on site-L states with erasers 0..L−1 applied.
- **Linear probe:** `probe_cv`'s linear probe (standardised inputs, logistic regression, L2 1e-3, LBFGS 200 iterations).

## Part A: learning curve (`scripts/run_leace_learning_curve.py`)

**Conditions.** Two conditions, each fit only on the training pairs:

- single-site LEACE at site 8, fit on natural states;
- sequential scrub over sites 0–12.

**Training sets.** Whole T/I pairs are held out.

- n ∈ {25, 50, 75, 100}: 5 random subsets of n pairs per n. Each is a simple random sample without replacement from the 126 pairs, with rng `default_rng(10_000 + 100 n + r)` for r = 0..4.
  - Every pair not in the training subset is held out: 101, 76, 51 and 26 pairs.
- LOO: 126 fits. Fit i trains on the other 125 pairs and holds out pair i.
- Fitting data: both prompts of every training pair in all 125 contexts.
- Batching: sorted by prompt length, as in the in-sample run.

**Measures on held-out pairs.** For each fit:

1. **Share of the *by* gap remaining:**
   - n-subsets: per fit, the mean erased gap / mean natural gap over that fit's held-out items. The per-n value is the mean over the 5 subsets.
   - LOO: every item is erased by the fit that held out its pair, then the erased gap is pooled over all 15,750 items / the natural gap.
   - The natural readout is recomputed in the same batch.
   - Mean KL (natural ‖ erased) at the verb position is also reported.
2. **Held-out linear probe accuracy after erasure (class, T vs I).** Probe sites:
   - single 8: the erased site-8 states;
   - scrub 0–12: the erased site-12 states (primary) and the erased site-8 states (secondary).
   - **n-subsets (leak-free):** a fit's held-out pairs are dealt into 5 groups, with rng `default_rng(1 + fit index)`. The probe trains on 4 groups and tests on the 5th, and accuracy is pooled over the 5 test groups. The eraser saw neither the probe's training verbs nor its test verbs. The per-n value is the mean over the 5 subsets.
   - **LOO (flagged as leaky):** one held-out pair cannot carry a held-out probe. The erased states of all 126 fits are assembled into one set and probed with 5 pair groups over all pairs. Each probe-training item was erased by an eraser that saw the probe's test pairs. This is the leak that pushes the in-sample verb-held-out probes to 0.001 at site 8. The LOO probe is reported with this flag and is not comparable to the n-subset probes.
   - Natural-state probes on the same held-out sets and splits are reported as the no-erasure reference.
3. **95% CIs.** The task-1 two-way cluster bootstrap (`analyze_das_prep.Boot`: verb pairs within verb band × contexts within context band), 1000 draws, seed 17.
   - In each draw, every fit's held-out share and probe accuracy is recomputed under the draw's weights, then averaged over the 5 subsets of each n.
   - Probe accuracy per item is the mean correctness of its T and I prompts.
   - Erasers, subsets and probe predictions are fixed (not resampled). Variation across subsets is reported separately as the range over the 5 subsets.
4. **References.**
   - The in-sample value: erasers fit on all 126 pairs and evaluated on the same items. These are `reports/pythia/leace_necessity_gaps.csv` rows `insample_single_8` and `insample_scrub_0_12`, and the in-sample probes in `leace_necessity_probes.csv`; the probes are reported as stored, including the leak caveat. This is the reference line.
   - The 5-fold cross-fitted run (about 101 training pairs per fold) is shown as a second marker, not a line.

**Checks** (reported):

- Every eraser is rank 1.
- A fit on all 126 pairs, run through this code, reproduces the in-sample run's per-prompt log P(by) for single 8 and scrub 0–12 (max |Δ| < 1e-4).
- Natural log P(by) matches the in-sample run.
- LOO covers every pair exactly once.
- The probe function written for per-item predictions gives the same pooled accuracy as `probe_cv` on one split set.

## Part B: site-20 split by verb token count (`scripts/run_leace_postscrub.py`)

**Split.** Verb level: a prompt belongs to "single-token" if its participle is one token, "multi-token" otherwise. The counts are:

- single-token: 32 T verbs, 33 I verbs;
- multi-token: 94 T verbs, 93 I verbs.

A pair-level split (27 pairs with both participles single-token vs 88 with both multi-token, leaving 11 mixed pairs out) is not used.

**Erasers reused.** The saved rank-1 erasers of the in-sample run and the cross-fitted run (`erasers.npz`). Loading is verified by reproducing each run's per-prompt log P(by) for scrub 0–12 (max |Δ| < 1e-4).

**Measures**, all at site 20:

1. **|cos| between the single-site and post-scrub erasers at site 20, per subset.** Edit and read directions.
   - The single-site eraser is fit on the subset's natural site-20 states.
   - The post-scrub eraser is fit on the subset's site-20 states after the saved all-verb erasers 0..19 are applied.
   - In-sample: one fit per subset on all of the subset's prompts.
   - Cross-fitted: per fold f, the subset's fold-f training prompts, with fold f's erasers 0..19 applied. Reported as the mean over folds [min, max].
   - The all-verbs fit must reproduce the saved erasers (|cos| > 0.999 with the saved single 20 and scrub 20 edit directions).
2. **Linear probe accuracy at site 20 after scrubbing 0–19.** These are the states before eraser 20, i.e. what scrub eraser 20 was fit on. Probes are trained as in the necessity runs, on all verbs, and accuracy is split by the test prompt's subset.
   - Cross-fitted: nested splits, leak-free.
   - In-sample: 5-group verb-held-out split, leaky as above.
   - Natural site-20 probes with the same splits are the reference.
   - 95% CIs from the same cluster bootstrap, over the subset's items.

## Part C: probes downstream of in-sample scrub 0–12 (`scripts/run_leace_postscrub.py`)

**States.** At sites 13–23 after in-sample scrub 0–12, with no erasure at 13–23.

**Probe.** Linear, with the in-sample run's verb-held-out split (5 pair groups over all pairs, seed 1).

**Reported per site, next to the in-sample scrub 0–12 share of the *by* gap remaining (0.04 [−0.46, 0.28]):**

- accuracy with 95% CIs (same bootstrap, predictions fixed);
- the natural-state probe at the same site and split;
- the same probe at site 12 (erased) as the leakage gauge for this split.

**Supplement (leak-free).** The same probes after cross-fitted scrub 0–12 (share remaining 0.59), with the cross-fitted run's nested splits.

**Checks:** with the same splits, the natural probes and the site-12 erased probe reproduce the stored `leace_necessity_probes.csv` values (in-sample and cross-fitted) to 0.005.
