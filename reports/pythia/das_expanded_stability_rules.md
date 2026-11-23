# DAS stability under more training verbs: rules (paper plan, work item 14)

Declared Nov 18, 2026, before any run. Model: Pythia-1.4B.

## Question

The stored direction v (round 2, `results/das_round2/final_strict_site{8,12}`) is a rank-1 DAS basis trained on only 29 transitive / intransitive pairs:
- 21 near-head pairs from the earlier expansion;
- 8 original head pairs.

The training items are actives of the form "<Subj> has/have <participle>". The basis is cross-fitted: 3 split seeds × 5 folds, 15 bases per site. Does v change when roughly five times as many training verbs are added?

## Conditions (Pythia rule sites 8 and 12)

Each condition trains 15 bases, one per stored (split, fold), keeping the stored fold assignment of the 29 pairs.
- **R, reproduction:** the stored 29 pairs, with the stored recipe and seed (`10000 + 100·split + fold`).
- **S, seed baseline:** the stored 29 pairs with the seed offset by +5000. This gives the variation from the seed alone.
- **E, expanded:** for each fold, the stored training pairs plus 123 new pairs. The new pairs are `data/verb_expansion/expanded_pairs.csv` minus the 24 pairs that share a verb with the stored 29. They are always in training and never held out, so each fold's held-out stored pairs stay out of training, and the stored passive test plan (donors = held-out stored pairs) is valid for E.

Recipe for all three, taken from the stored `final_meta.json`:
- epochs from `frozen_config_site{s}.json` (site 8: 3; site 12: 2);
- lr 0.01, batch 64, 4 cross-class and 2 same-class swaps per base item;
- bf16;
- training subjects exclude David.

The behavior filter (a verb is kept only if at least 4 of its 7 items fall on its class's side of 0) is applied to all pairs; dropped new pairs are reported. Epochs are fixed by count, so E takes about 5× more gradient steps. This is a property of the comparison, and it is stated.

## Measures

For each site, the stored v is compared with each of R, S and E:
1. **Direction:** |cos(basis, v)| for each (split, fold); the median and range over the 15.
2. **Held-out active IIA:** cross-class IIA and the median positive-control fraction on the stored held-out pairs of each fold (`run_das_round2.evaluate` / `summarize`), meaned over the 15.
3. **Passive *by* transfer:**
   - `run_passive_test.py` with the condition's bases on the stored plan;
   - D_bad(*by*) on the primary (plain) population, all bands, using `analyze_passive_test`'s estimator and bootstrap (2000 draws, seed 17);
   - computed for stored, E and S with the same draws;
   - reported as the ratio to stored and the difference, each with a 95% CI.

## Checks (all must pass, or the reading is invalid)

- **Stored items:** the stored-pair items equal the stored `items.csv` (same prompts, same order).
- **Recompute:** the stored bases, applied in this pipeline, recompute the stored held-out swaps: max |ΔM| < 1e-2 (bf16).
- **Reproduction:**
  - R reproduces v: median |cos(R, v)| ≥ 0.99.
  - If not, GPU nondeterminism is larger than expected. Then R, not the stored v, becomes the reference for E and S, and this is reported.
- **Passive test:** `run_passive_test.py`'s own checks pass (self-patch = unpatched; held-out recompute).
- **Estimator:** the stored D_bad(*by*) recomputed here matches the stored `transfer_summary.csv` within 1e-4.

## Reading (per site, then overall)

- **Stable:**
  - median |cos(E, v)| ≥ median |cos(S, v)| − 0.05;
  - and the 95% CI of D_E / D_stored lies inside [0.8, 1.25].
- **Changed:** either the 95% CI of D_E / D_stored lies entirely outside [0.8, 1.25], or median |cos(E, v)| < median |cos(S, v)| − 0.15.
- **Partly stable:** anything else, described with the numbers.

Overall: stable if stable at both sites; changed if changed at either; otherwise partly stable.

## Use

The stored v stays the paper's instrument whatever the outcome.
- **Stable:** one sentence in the paper ("v is unchanged when trained on 5× more verbs") plus an appendix row.
- **Changed:** reported as a finding about how much v depends on its training verbs. It is discussed next to the u_md results, which are computed on all 273 pairs and need no training. v is not swapped.

Active IIA differences are descriptive. Expanded pairs that share a verb with the stored 29 are flagged `das_train_overlap` in `expanded_pairs.csv` and must be excluded wherever v's numbers are computed on the expanded set.

## Amendment (Nov 18, before the full run; after a training-only smoke at site 12, split 0 / fold 0, whose cosines were not read)

v's training set followed the round-2 class rule (`reports/passive_das_prep/round2.md`): only *plain* intransitives, never *prep_object* ones (argument PPs such as *rely on*, *succumb to*). Under that rule 61 of the 123 new intransitives are prep_object. So E changes two things at once: more verbs, and a different intransitive class.

Added condition **P**: as E, but the new pairs are restricted to plain intransitives. That gives 62 pairs, mostly rare: xtail 29, between bands 8, below xtail 6, near-head 13, tail 5, head 1. Classes come from `data/das_round2/intrans_classes_final.csv` where listed; the other 50 verbs were classified by Claude under the same rule, in `data/verb_expansion/expanded_intrans_class.csv`, with 10 prep_object: *resonate with*, *converge on*, *balk at*, *feast on*, *despair of*, *hunger for*, *empathize with*, *ruminate on*, *sorrow over*, *clam up*.

- **P is the primary reading:** more verbs, same definition.
- **E is secondary:** more verbs plus prep_object intransitives.

Both get the same measures and the same reading rule. The passive test runs for P, E and S.

## Result (Nov 18, TSUBAME job 9451215)

**Declared reading: invalid.** The cause is a miscalibrated check, not a pipeline error. I set the held-out recompute tolerance at max |ΔM| < 1e-2, and it fails for every condition: the max is 0.22–0.39 for P, E and S, and 0.09–0.11 for the stored bases in this pipeline. That tolerance was tighter than the stored pipeline's own precision. The stored passive run (`results/das_round2/passive_test/run_meta.json`) had max 0.29 / 0.36 at sites 8 / 12, with median ≈ 0. Here too the median |ΔM| is 0 to 8e-4, so the gap comes from a few swaps whose bf16 batch composition differs. The other checks pass:
- R reproduces v (median |cos| 1.000 at both sites);
- self-patch = unpatched (0.0);
- S / stored D_bad(*by*) is 1.00 [0.99, 1.01] and 0.99 [0.99, 1.00].

**Exploratory reading,** taking the stored run's precision (median |ΔM| < 1e-3) as the check: P is **partly stable** at both sites, and so is E.

| site | median \|cos\| with v: S / P / E | D_bad(*by*) ratio to stored: P / E | held-out IIA: stored / P / E |
|---|---|---|---|
| 8 | 0.996 / 0.914 / 0.887 | 1.10 [1.05, 1.17] / 1.11 [1.03, 1.19] | 0.78 / 0.80 / 0.78 |
| 12 | 0.994 / 0.922 / 0.891 | 1.06 [1.02, 1.11] / 1.02 [0.97, 1.07] | 0.90 / 0.91 / 0.90 |

- **Why "partly":**
  - P's transfer ratio is inside [0.8, 1.25], so it passes on transfer.
  - Its cosine is 0.07–0.08 below the seed baseline, which misses the 0.05 cut for "stable" but stays well inside the 0.15 cut for "changed".
- **What that means:** with twice as many plain training verbs, the direction rotates by about 23° (cos ≈ 0.92), more than seed noise. It does at least as well: held-out active IIA is equal or slightly higher, and passive *by* transfer is 6–10% larger.
- **E rotates more** (cos ≈ 0.89) without a larger transfer gain, which fits the prepositional intransitives diluting the class.

Outputs are on TSUBAME (`results/pythia/das_expanded_stability/`, hashes in `large_files_sha256.txt`). Report: `reports/pythia/das_expanded_stability.md`.

## Amendment 2 (Nov 23): the held-out recompute check uses the stored run's own precision

Declared before any of the extension runs (more sites, Gemma / OLMo, seed band, reverse DAS). It replaces the max |ΔM| < 1e-2 check above for every run from here on; the Nov 18 run is re-read under it in the Result section (exploratory there, since it was set after that run).

**The bar.** At each site, a held-out recompute passes if it is no worse than the stored passive run at that site (`results/das_round2/passive_test/run_meta.json`, or the stored run of the same model for Gemma / OLMo):
- median |ΔM| ≤ max(1e-3, 1.5 × the stored median at that site), and
- max |ΔM| ≤ 0.5.

Stored Pythia values: median 4e-16 / 5.0e-4 / 4.4e-3 and max 0.29 / 0.36 / 0.43 at sites 8 / 12 / 17. Self-patch = unpatched stays exact (0.0).

**Why.** The stored pipeline runs in bf16 with batch 1024, and a recompute batches the held-out swaps differently. In bf16 the result of a matmul depends on batch composition (kernel choice and accumulation order), so a few swaps differ by up to 0.3–0.4 nats while the median difference is about 0. The stored run, whose numbers every v result rests on, shows exactly this spread against its own training-time evaluation. A tolerance tighter than that (the 1e-2 above) fails the stored pipeline itself, so it cannot tell a broken pipeline from bf16 noise. The median catches a systematic error (wrong basis, wrong site, wrong token position), and the max bound catches an error confined to a few items beyond the stored spread.
