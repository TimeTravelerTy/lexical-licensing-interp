# Gemma replication: exploratory / secondary analyses

**Everything in this file is exploratory or secondary.** It does not replace the declared results in
`controls_transfer.md` and `transfer.md`:
- the declared sites are 15 (frozen rule), 12 (within 0.05 of max) and 5 / 8 (Pythia's secondary-site rule);
- the declared passive outcome is the active-donor test.

## Rules, stated before running (2026-11-06)

### 1. IIA-matched sites

**Rule:**
- Take each of Pythia's abstract-reading sites, 4, 6 and 8.
- Match it to the Gemma site whose sweep best-epoch active IIA is closest to that Pythia site's sweep
  best-epoch active IIA. Ties go to the earlier site.
  - The quantity is the same for both models: the best epoch's fold-mean held-out cross-class IIA (split 0,
    5 folds).
  - Pythia's is read from `results/das_round2/sweep_strict/sweep.csv`, Gemma's from
    `results/gemma/das/sweep_strict/sweep.csv`.
  - Candidate Gemma sites are 0–25; the last site, 26 (post-final-norm states), is excluded.
- Epochs come from the frozen epoch rule: the smallest epoch within 0.01 of the site's best. Rank is 1, as at
  every other Gemma transfer site.

**Result of the rule** (computed before any evaluation; configs `results/gemma/das/transfer/frozen_config_site{0,1,4}.json`):

| Pythia site | Pythia IIA | Gemma site | Gemma IIA | distance | runner-up (distance) | epochs |
|---:|---:|---:|---:|---:|---|---:|
| 4 | 0.4544 | 0 | 0.3572 | 0.0972 | 2 (0.1829) | 2 |
| 6 | 0.6694 | 1 | 0.6847 | 0.0153 | 2 (0.0322) | 1 |
| 8 | 0.8026 | 4 | 0.7933 | 0.0092 | 3 (0.0415) | 2 |

Two flags:
- Gemma site 0 is the embedding output; a rank-1 patch there acts on the participle's input embedding alone.
- Its match is loose: 0.097, against 0.015 and 0.009 for the other two. Gemma has no site between 0.36
  (site 0) and 0.64 (site 2).

**Run:**
- Final runs: `run_das_round2.py final`, 3 splits × 5 folds.
- The full existing chain (`jobs/transfer_chain.sh`):
  - prep: bases, TOST bounds, builders;
  - main: passive test, OR / TC, in range;
  - fg: filler-gap, round-5 C8 protocol.
- The chain runs at all seven Gemma sites 0, 1, 4, 5, 8, 12, 15 with the unchanged runners and analyses, so
  every site shares one table. The declared sites 5 / 8 / 12 / 15 must reproduce their committed numbers
  exactly.

### 2. In-range equivalence test (secondary)

**Rule:** apply the passive test's own classes to the in-range passive O effect, D_in(O) = R(t_T) − R(t_I) on
bad passive items (`inrange_summary.csv`):
- RISE = 95% CI > 0 and estimate ≥ δ_s;
- FALL = 95% CI < 0 and estimate ≤ −δ_s;
- NO RISE = 90% CI within ±δ_s;
- otherwise unresolved.

The code is `analyze_passive_test.classify`.

Details:
- δ_s is the same site-specific bound as the declared passive test (`deltas.json`, 0.2 × the site's held-out
  active ΔlogP(O)).
- The CIs are the in-range analysis's own two-way bootstrap (pairs within band × contexts; 2,000 draws, seed 17).
  The declared test's three-way bootstrap also resamples donor pairs, which the in-range patch has none of.
- Scope: both models, every run site. Pythia: 4, 6, 8, 10, 12, 14, 16, 17. Gemma: 0, 1, 4, 5, 8, 12, 15.
- D_in(*by*) gets the same classes. The pair (O class, *by* class) gives an in-range outcome under the same
  surface / abstract / mixed rule. This is a secondary reading.

### 3. Balanced shuffled-label control

**Rule** (`scripts/run_das_shuffle_balanced.py`): 20 permutations per site.
- **Labels.** In each permutation, for each split × fold of the site's final run, take the fold's training
  verbs. Relabel exactly half of the transitive verbs as intransitive and exactly half of the intransitive
  verbs as transitive.
  - A class with an odd count n flips floor(n/2) or ceil(n/2), chosen at random. This keeps the correlation 0 in
    expectation and |phi| ≤ 0.043 on these folds; phi is recorded per fold.
  - Verbs outside the training set keep their labels.
- **Training.** As the site's final run trains (`run_das_round2.train`), with the same rank, epochs, learning
  rate, swaps and batch.
- **Evaluation.** On the fold's held-out swaps with the true classes (`run_das_round2.evaluate` /
  `summarize`). The items and folds must equal the final run's.
- **Statistic.** Each permutation's held-out cross-class IIA is averaged over its 15 folds. Over the 20
  permutations the report gives the mean and the 95th percentile, next to the final run's unconstrained
  shuffled-label mean and the trained DAS mean.
- **Sites.** Gemma 0, 1, 4, 5, 8, 12, 15; Pythia 8, 12, 17, from Pythia's stored final runs. Pythia's outputs
  go under `results/gemma/das/shuffle_balanced/`; the Pythia results are untouched.

No nonce runs.
