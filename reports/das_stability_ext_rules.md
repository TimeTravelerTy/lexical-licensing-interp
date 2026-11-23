# DAS stability, extended: all rule sites, Gemma and OLMo, a seed band (paper plan, item 14 extension)

Declared Nov 23, 2026, before any run. Extends `reports/pythia/das_expanded_stability_rules.md` (item 14; read with its Amendment 2, which sets the recompute bar used here). The Nov 18 run covered Pythia sites 8 and 12 with one seed-baseline retrain. This run asks the same question at every rule-picked site, in all three models, against a band of five seeds.

## Question

v is a rank-1 DAS basis trained on 29 transitive / intransitive pairs (Gemma: 28 after its behavior filter drops *transplant / ensue*). When the new plain pairs are added to training (condition P), how far does the direction move, compared with how far it moves under a change of seed alone?

## Models and sites

The declared rule sites (`reports/replication_criteria.md`), with the stored final of each site as the reference:

| Model | Sites | Stored finals |
|---|---|---|
| Pythia-1.4B | 8, 12, 17 | `results/das_round2/final_strict_site{8,12}`, `final_strict` (17) |
| Gemma-3-1B | 5, 8, 12, 15 | `results/gemma/das/transfer/final_strict_site{5,8,12}`, `final_strict` (15) |
| OLMo-2-1B | 9, 11, 12, 13 | `results/olmo/das/transfer/final_strict_site{9,11,12}`, `final_strict` (13) |

Each condition trains 15 bases per site, one per stored (split, fold), keeping the stored fold assignment. Recipe from each stored `final_meta.json` (epochs per site, lr 0.01, batch 64, 4 cross / 2 same swaps, bf16, David held out), with the model's architecture defaults (`model_arch`, as the stored runs).

## Conditions

- **R, reproduction:** stored pairs, stored seed (`10000 + 100·split + fold`).
- **S1–S5, seed band:** stored pairs, seed offset by +5000, +6000, +7000, +8000, +9000. S1 is the Nov 18 S.
- **P, plain expansion:** each fold's stored training pairs plus the 62 new plain pairs of the Nov 18 batch (`expanded_pairs.csv`, `batch == nov18`, no `das_train_overlap`, intransitive `plain` in `expanded_intrans_class.csv`), always in training. This is the Nov 18 P. The 19 pairs added on Nov 19 (R5-D11) are left out so that P is the same set at every site and in every model.
- E (with prepositional intransitives) is not rerun; it stays a Pythia 8 / 12 secondary.

The behavior filter (≥ 4 of 7 items on the class's side) is applied per model to all pairs. The stored pairs that fail must be exactly the stored final's `dropped_pairs`; the new pairs each model drops are reported.

## Measures (per model and site)

1. **Direction:** |cos(basis, v)| per (split, fold); median and range over the 15, per condition.
2. **Held-out active IIA:** cross-class IIA and median positive-control fraction on the stored held-out pairs of each fold, mean over the 15.
3. **Passive *by* transfer:** `run_passive_test.py` with the condition's bases on the model's stored passive plan; D_bad(*by*) on the primary (plain) population, all bands, with `analyze_passive_test`'s estimator. Stored, P and S1–S5 share one set of 2000 bootstrap draws (seed 17) per site, so ratios to stored are paired.

**Seed band.** Per site: the five values of median |cos(S_k, v)| and of the point ratio D_{S_k} / D_stored. S̄ is their mean; the band is [min, max].

## Checks (all must pass, or that model's reading is invalid)

- Stored items equal the stored `items.csv`; stored drops equal the stored `dropped_pairs`.
- Recompute (Amendment 2): the stored bases through this pipeline recompute the stored held-out swaps with median |ΔM| ≤ max(1e-3, 1.5 × the stored passive run's median at that site) and max |ΔM| ≤ 0.5. The same bar applies to each passive run's own recompute; self-patch = unpatched exactly.
- R reproduces v: median |cos(R, v)| ≥ 0.99; otherwise R replaces v as the reference and this is reported.
- The stored D_bad(*by*) recomputed here matches the model's stored `transfer_summary.csv` within 1e-4 (sites without a stored passive run, if any, are reported without this check).

## Reading (per model and site; P is the only condition read)

The item-14 rule, with S replaced by the seed band:
- **Stable:** median |cos(P, v)| ≥ S̄_cos − 0.05, and the 95% CI of D_P / D_stored lies inside [0.8, 1.25].
- **Changed:** the CI lies entirely outside [0.8, 1.25], or median |cos(P, v)| < S̄_cos − 0.15.
- **Partly stable:** anything else, described with the numbers.

Also reported, descriptive: whether P's median |cos| and point ratio fall inside the seed band, and P's cosine gap in units of the band's spread, (S̄_cos − cos_P) / sd_k(cos_{S_k}).

Per model: stable if stable at every site, changed if changed at any, otherwise partly stable.

## Use

As item 14: the stored v stays the instrument whatever the outcome. The result is an appendix table (model × site: cos, IIA, D ratio, seed band) and one sentence in the paper. A "changed" reading in any model is flagged as a finding about v's dependence on its training verbs and discussed next to u_md, which needs no training.

## Cost

Training is cheap (Nov 18: 24 min for 4 conditions × 2 sites in Pythia). The passive patches dominate: 6 conditions (S1–S5, P) × 3–4 sites per model. R is used for cosine and IIA only. `gpu_h`, one job per model.

## Amendment (Nov 23, before any full run; after Codex review of the rules and runners)

Only smoke runs had been made (Pythia site 12 split 0 / fold 0, whose P cosine 0.924 reproduces the Nov 18 value; Gemma and reverse smokes). Changes:

1. **Evaluation overlap.** The stored passive plan contains the 19 earlier eval pairs; 7 of them share a verb with P's training pairs (5 in the primary population: *raze / relent*, *pillage / waver*, *requisition / flinch*, *cajole / saunter*, *besmirch / reemerge*). The reading now uses the **clean primary population**: primary (plain) minus every pair that shares a verb with P's training verbs, applied to stored, S and P alike (same pairs in numerator and reference). The full primary population is reported next to it, for comparability with Nov 18.
2. **Two training-only controls** (cosine and held-out IIA; no passive run), because the seed band alone does not separate a change of vocabulary from a change of objective size and optimization length (P has about 3.6 times the training items and updates per epoch):
   - **P2, P3:** P with seed offsets +5000 and +6000, so P has its own seed band (P, P2, P3).
   - **L:** the stored pairs trained for P's number of updates (epochs scaled per fold by P's / the stored training-item count, rounded), stored seed. If L rotates as far as P, the rotation is about optimization length, not vocabulary.
3. **Reading, split in two** (the three-level item-14 label is still computed, for continuity):
   - **Direction:** P's median |cos| against the seed range of S1–S5 (inside if min ≤ P ≤ max, both bounds), with P's own seed range and L reported next to it. The ranges and the gap in seed-sd units are descriptive, not calibrated intervals.
   - **Function:** the ratio D_P / D_stored on the clean population is *equivalent* if its 95% CI lies inside [0.8, 1.25], *larger* if the CI lies above 1.25, *smaller* if below 0.8, otherwise *unresolved*. Whether the CI excludes 1 is reported.
4. **Reproduction:** if median |cos(R, v)| < 0.99 the site's reading is invalid (the earlier "R becomes the reference" fallback is dropped: R has no passive run).
