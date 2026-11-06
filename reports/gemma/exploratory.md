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

## Results (run 2026-11-06/07; tables: `exploratory_tables.md`, `transfer.md`)

**Reproduction:** rerunning the chain at seven sites left every declared-site output unchanged: all summary
CSVs match row for row (max diff 0), the base vectors are bit-identical, and the run checks pass at every site
(self-patch / self-set max |dM| 0 in bf16, ≤ 2.3e-5 in fp32; held-out recompute max ≤ 0.091). The declared results stand as
reported in `controls_transfer.md`.

### 1. IIA-matched sites: transfer (exploratory)

Final-run DAS IIA (15 folds): Gemma 0.34 / 0.63 / 0.81 at sites 0 / 1 / 4; Pythia 0.45 / 0.66 / 0.78 at
sites 4 / 6 / 8. So the final runs keep the sweep match at sites 1 and 4 (within 0.03) and widen the gap at site 0
(0.11 below Pythia 4). Norm-matched random directions give ≤ 0.02.

**Passive (*by*), primary population, active donors**, same rule as declared:

| Pythia site → Gemma site | Pythia O / *by* | Pythia outcome | Gemma O / *by* (δ_s) | Gemma outcome |
|---|---|---|---|---|
| 4 → 0 | +0.10 NO RISE / +0.58 RISE | abstract | +0.10 [+0.04, +0.16] NO RISE / +0.26 [+0.18, +0.35] unresolved (0.29) | unresolved |
| 6 → 1 | +0.23 NO RISE / +0.69 RISE | abstract | +0.29 [+0.18, +0.40] NO RISE / +0.43 [+0.30, +0.58] RISE (0.41) | **abstract** |
| 8 → 4 | +0.36 NO RISE / +0.72 RISE | abstract | +0.80 [+0.61, +0.98] RISE / +0.63 [+0.49, +0.77] RISE (0.52) | mixed |

- One of three matched sites reads abstract: Gemma site 1 (depth 0.04, the first layer's output), matched to
  Pythia 6 at distance 0.015. It is a narrow reading:
  - *by* +0.43 clears δ_s (0.41) by 0.02;
  - O +0.29 is NO RISE (90% CI upper 0.38 < 0.41).
- At site 0, *by* rises (95% CI > 0) but stays below δ_s and is not equivalent to 0 either, so the outcome is
  unresolved. This is also the loosest match (0.097 in the sweep, 0.11 in the final run).
- Site 4 is the closest match (0.009), and it reads **mixed**, not abstract.
- O as a fraction of the site's active effect (δ_s / 0.2):

  | | Gemma site 0 | Gemma site 1 | Gemma site 4 | Gemma site 5 (declared) |
  |---|---|---|---|---|
  | O fraction | 0.07 | 0.14 | 0.31 | 0.56 |

  Pythia: 0.05 / 0.10 / 0.15 at sites 4 / 6 / 8. So the O fraction is close at the first two matches, but at
  the closest IIA match Gemma's is about twice Pythia's.
- *By* is smaller in Gemma at every match (+0.26 / +0.43 / +0.63 against +0.58 / +0.69 / +0.72). Gemma's
  natural *by* gap is smaller, so as a fraction of it D(*by*) is 0.64 / 1.05 / 1.53, against Pythia's
  0.81 / 0.96 / 1.01.

**OR / TC (T − I donors).** D(O) and D(L) rise at every matched Gemma site, except TC D(O) at site 1, which is
unresolved. The licensed continuation rises less than in Pythia at every match, and PREP falls less:
- OR MAIN +0.62 / +1.12 / +1.11, against +1.34 / +1.49 / +1.47;
- TC END +0.11 / +0.25 / +0.31, against +0.72 / +0.80 / +0.75;
- PREP: OR −0.17 / −0.33 / −0.46 against −0.57 / −0.64 / −0.63; TC −0.21 / −0.31 / −0.34 against
  −0.69 / −0.78 / −0.80.

TC D(O): Gemma +0.40 RISE / +0.38 unresolved / +0.65 RISE; Pythia +0.30 NO RISE / +0.37 unresolved / +0.92 RISE.

**In range** (ratio to the natural gap, Gemma 0 / 1 / 4 against Pythia 4 / 6 / 8). Gemma's in-range effects
at the matched sites are much smaller than Pythia's until site 4:
- OR MAIN 0.11 / 0.19 / 0.62 against 0.79 / 0.93 / 0.89;
- TC END 0.13 / 0.45 / 0.91 against 0.96 / 1.10 / 0.91;
- passive *by* 0.17 / 0.22 / 0.59 against 0.52 / 0.52 / 0.44.

Passive D_in(O) is near zero in both: +0.02 / +0.01 / +0.10 against +0.04 / +0.06 / +0.09.

**Filler-gap** (round-5 C8, non-DAS primary, half B):

| | Gemma 0 / 1 / 4 | Pythia 4 / 6 / 8 |
|---|---|---|
| embedded, T − I donors | mixed / **licensing-like** / surface | surface / surface / surface |
| embedded, in range | mixed / mixed / mixed | mixed / surface / mixed |
| matrix, T − I donors | mixed / **licensing-like** / surface | mixed / surface / surface |
| matrix, in range | mixed / mixed / surface | **licensing-like** / mixed / **licensing-like** |

- Gemma site 1 reads licensing-like with active donors in both frames. Pythia never does with active donors at
  any stored site (primary population).
- The site-1 labels follow the declared classifier. The O-level condition is "not above the T verbs' natural
  level": the estimates are below it, but both CIs include 0 (−0.11 [−0.63, +0.37] embedded, −0.35 [−0.79, +0.11]
  matrix).
  - In embedded wh the filler raises the patched O (fill − nofill +0.22 [+0.03, +0.44]), so the label does not
    mean O is suppressed.
  - Gemma has 16 / 26 I verbs against Pythia's 28 / 34.
- Pythia's licensing-like readings are in range, in matrix wh; Gemma has none in range at the matched sites.

**Reading (exploratory).**
- One tested IIA-matched site reproduces Pythia's early passive classification: site 1. There *by* rises, O is
  positive but within the equivalence bound (+0.29 [+0.18, +0.40], δ_s 0.41), and the filler-gap donor patches
  read licensing-like.
- It is a single site, at the first layer's output. Its passive reading clears the *by* bound by 0.02. The
  direction's effects on the licensed continuations there (OR MAIN, TC END, in range) are well below Pythia's.
- At the closest IIA match (site 4 ~ Pythia 8), Gemma is mixed, with twice Pythia's O fraction.
- So, among the tested sites, an abstract-looking reading appears in Gemma only at one early, low-IIA site,
  not over a span like Pythia's sites 4–8. Matching by IIA does not show how much IIA contributes to the
  declared depth difference. Gemma sites 2 / 3 (sweep IIA 0.64 / 0.76) were not run through the transfer chain.
  Three exploratory comparisons, not corrected for multiplicity.
- The declared reading ("on the declared test, Gemma does not show Pythia's early abstract profile at the
  tested sites") is unchanged. It concerned the rule-picked sites.

### 2. In-range equivalence test (secondary)

D_in(O) on bad passive items against the declared δ_s:
- **Pythia: NO RISE at every site, 4–17** (+0.04 to +0.45; 90% CI upper ≤ 0.50 against δ_s 0.42–0.57).
- **Gemma: NO RISE at six of seven sites**, 0 / 1 / 4 / 5 / 12 / 15 (+0.02 / +0.01 / +0.10 / +0.27 / +0.40 / +0.43).
- **Unresolved at Gemma 8**: +0.50 [+0.45, +0.55]; 90% CI upper 0.54 > δ_s 0.53, and the estimate is below δ_s,
  so it is not RISE.

So with the coordinate set to the passives' own class means, neither model's direction raises the passive
object continuation past the equivalence bound, except one unresolved site. This supports the donor-scale
caveat in `controls_transfer.md`: the declared O RISE at Gemma 5 / 8 / 12 / 15 needs active-donor values.

Gemma's in-range O is still larger than Pythia's at matched depth (e.g. +0.27 / +0.50 at Gemma 5 / 8 against
+0.04 to +0.09 at Pythia 4–8). It is below the bound, not zero.

**The secondary in-range outcome is degenerate.** D_in(*by*) is NO RISE at every site in both models
(+0.04 to +0.37):
- δ_s is scaled to the active-donor effect on O (0.2 × ΔlogP(O)), and the in-range *by* effects are all smaller
  than that.
- So the in-range outcome is "nothing moves" everywhere, except unresolved at Gemma 8.

This says the in-range *by* effect is small next to an active-donor-scaled bound. It does not say *by* is
unaffected: D_in(*by*) is 0.17–0.89 of the natural *by* gap in Gemma and 0.06–0.52 in Pythia. The in-range outcome cannot tell abstract from
surface with this bound, and it should not be read as either.

### 3. Balanced shuffled-label control

20 permutations per site, each over the final run's 15 folds; |phi| between true and permuted training-verb
classes < 0.044 in every fold (max 0.0435). Held-out cross-class IIA:

| model | site | balanced mean | balanced p95 | unconstrained (final run) | trained DAS |
|---|---:|---:|---:|---:|---:|
| Gemma | 0 (expl.) | 0.11 | 0.12 | 0.12 | 0.34 |
| Gemma | 1 (expl.) | 0.21 | 0.25 | 0.24 | 0.63 |
| Gemma | 4 (expl.) | 0.24 | 0.29 | 0.26 | 0.81 |
| Gemma | 5 | 0.26 | 0.30 | 0.29 | 0.85 |
| Gemma | 8 | 0.31 | 0.34 | 0.35 | 0.88 |
| Gemma | 12 | 0.35 | 0.40 | 0.41 | 0.94 |
| Gemma | 15 | 0.37 | 0.43 | 0.41 | 0.98 |
| Pythia | 8 | 0.24 | 0.27 | 0.20 | 0.78 |
| Pythia | 12 | 0.28 | 0.31 | 0.24 | 0.90 |
| Pythia | 17 | 0.39 | 0.44 | 0.34 | 0.97 |

- **Trained DAS exceeds the balanced p95 at every site:**
  - by 0.22 at Gemma 0;
  - by 0.38–0.58 at the other sites: 0.38 at Gemma 1, 0.51–0.58 at Gemma 4–15 and Pythia 8–17;
  - it also exceeds the balanced maximum over permutations at every site (largest 0.45, Gemma 15 / Pythia 17).
- **Near-zero label correlation cannot fully account for the shuffled-label level.** With labels uncorrelated with the classes,
  the cross-class IIA is still 0.11–0.39 and rises with site in both models, well above norm-matched random
  directions (≤ 0.02). Single folds reach 0.82.
- **Balanced against unconstrained:**
  - Gemma's balanced mean is 0.014–0.065 below its unconstrained one at every site;
  - Pythia's is 0.04–0.05 above.
  - So the unconstrained control's model gap shrinks under balancing:
    - site 12: 0.41 vs 0.24 unconstrained, 0.35 vs 0.28 balanced;
    - at the IIA match Gemma 4 ~ Pythia 8: 0.24 vs 0.24;
    - final sites: Gemma 15 vs Pythia 17, 0.37 vs 0.39.
  - This suggests that the higher shuffled-label IIA reported for Gemma in `controls_transfer.md` partly reflects
    which permutations the unconstrained control drew. It is not established: the unconstrained control is one
    permutation per fold, so it gives no permutation distribution to compare against. Under the balanced control
    the two models are close at matched IIA.

No nonce runs.
