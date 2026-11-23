# Reverse frequency transfer: a direction trained only on rare verbs (paper plan, item 15)

Declared Nov 23, 2026, before any run. Model: Pythia-1.4B.

## Question

v is trained on frequent pairs (8 head, 21 near-head) and transfers to rare verbs' passives. Does a direction trained only on rare verbs transfer to frequent ones? If it does, the class feature is shared across frequency, and the rare-verb deficit is not a missing feature (with R6-C4, u_md and LEACE).

## Training set

Rare = the intransitive's FreqBLiMP participle-Zipf tag is xtail (1.2–2.2), below xtail (`out`, < 1.2, including participles wordfreq has not seen), or between the Tail and XTail bands (`gap`, 2.2–2.4). This uses the band tags, as decided Nov 23; the plan's Pile-count split is not used, and Pile counts are reported descriptively where they exist.

The pairs are the plain pairs of the Nov 18 expansion batch with a rare intransitive: 43 pairs (xtail 29, gap 8, out 6), none sharing a verb with v's 29. The matched transitives are of similar summed-lemma Zipf; their participle tags run from out to near-head (one pair). Rarity is defined on the intransitive, the binding class.

Two conditions:
- **F, filtered (primary):** the behavior filter used for v (a verb is kept if at least 4 of its 7 items are on its class's side of 0), applied to the 43 pairs.
- **U, unfiltered:** all 43 pairs. The filter keeps only rare verbs the model already gets right in actives, which biases F toward an easy subset; U removes that bias.

Reported: how many of the 43 pairs, and which verbs, the filter drops, and each failing verb's item count.

## Recipe

Rank-1 DAS at sites 8, 12 and 17 with v's recipe at that site (stored `final_meta.json`: epochs, lr 0.01, batch 64, 4 cross / 2 same swaps, bf16), trained on all items of the condition's pairs except subject David. There is no fold structure: no frequent pair is in training. Three seeds (20000 + k, k = 0, 1, 2). Seed k's basis is stored under every key r1_s{k}_f{0..4}, so the stored passive plan (which pairs split k's bases with split k's held-out donors) runs unchanged and the three seeds are averaged as the three splits are for v. With 43 pairs instead of about 23 per fold, an epoch has about twice as many steps as v's; this is stated, not corrected.

## Measures (compared with v on the same items)

1. **|cos| with v:** over the 3 seeds × 15 v bases; median and range. References: v's own seed band and P (`reports/das_stability_ext_rules.md`, same sites).
2. **Held-out active IIA on frequent verbs:** for each stored (split k, fold f), seed k's basis on the stored fold's held-out items of v's 29 pairs, with exactly v's held-out swaps. Cross-class IIA and median positive-control fraction, mean over the 15 folds, next to v's. Paired difference (rare − v) with a 95% CI from 2000 bootstrap draws over held-out pairs.
3. **Passive *by* transfer:** `run_passive_test.py` on the stored plan (donors: v's 29 frequent pairs). D_bad(*by*) on the primary (plain) population per band (Head, Tail, XTail) and all bands, with `analyze_passive_test`'s estimator. v and the rare direction share one set of 2000 draws (seed 17), so D_rare / D_v is paired. **Head is the frequent test**; Tail and XTail give the within-frequency comparison.

The forward direction (v's transfer to the Tail and XTail bands) is already in R2-ptest and is printed alongside from the stored `transfer_summary.csv`.

## Checks

- Passive test: self-patch = unpatched exactly; held-out recompute against this run's own swaps under the Amendment 2 bar (`das_expanded_stability_rules.md`).
- The stored v's D_bad(*by*) recomputed here matches the stored `transfer_summary.csv` within 1e-4.
- No training verb is in v's 29 pairs or in the stored passive plan's donor pairs (asserted).

## Reading (condition F, per site)

On the Head band:
- **Transfers:** the 95% CI of D_rare(*by*) lies above 0 and the point ratio D_rare / D_v ≥ 0.5.
- **Weak transfer:** the CI lies above 0 and the ratio < 0.5.
- **No transfer:** the CI includes 0 or lies below it.

Overall: transfers if it transfers at 2 or more of the 3 sites; no transfer if it fails to transfer at 2 or more; otherwise mixed. U gets the same reading, reported next to F. Active IIA and cosine are descriptive. If the rare direction transfers on passives but its held-out active IIA on frequent verbs is under 0.6, that is said.

## Use

Transfers: one sentence in the rare-verb section ("a direction learned only from rare verbs moves frequent verbs' passive *by* preference by X of v's"), with the table in the appendix. Weak or none: reported in the same place, as a limit on "the feature is shared across frequency", next to the training-free u_md and LEACE evidence.

## Cost

Training: 2 conditions × 3 seeds × 3 sites, each over 43 pairs. Passive: 2 conditions × 3 sites. Under an hour on `gpu_h`.

## Amendment (Nov 23, before any full run; after Codex review)

Only a smoke run had been started (site 17, not read). Changes:

1. **Correction:** the 43 pairs' transitives are tagged xtail 16, gap 10, tail 12, near-head 3 (*flour*, *counsel*, *hazard*), out 2, not "one near-head". Rarity is guaranteed on the intransitive only. Wording for any result from F or U: "a direction learned from rare intransitives contrasted with transitives of mixed frequency".
2. **Added condition B, both sides rare:** the 28 pairs whose transitive is also xtail, gap or out, behavior-filtered as F. B carries the stronger "trained only on rare verbs" wording.
3. **Evaluation overlap:** 3 pairs of the stored passive plan share a verb with the rare training pairs (*cajole / saunter*, *besmirch / reemerge*, primary; *browbeat / cohabit*, prepositional). Every D and ratio is computed on the primary population minus pairs sharing a verb with that condition's training verbs, for the rare direction and v alike.
4. **Sites:** v's passive *by* effect is positive at 8 and 12 and negative at 17 (D_bad(*by*) on Head −0.90 [−1.10, −0.71]; site 17 is surface). So **8 and 12 are the transfer tests**; 17 is reported as signed retention (D_rare / D_v, same sign or not), descriptive.
5. **Labels (Head band, sites 8 and 12):** *transfers* (CI of D_rare above 0 and point ratio ≥ 0.5); *weak transfer* (CI above 0, ratio < 0.5); *unresolved* (CI includes 0); *opposite* (CI below 0). Also reported: the paired contrast D_rare − 0.5·D_v with its CI ("retains at least half" if above 0). **Overall:** transfers if it transfers at both 8 and 12; *no positive transfer* if neither site has a CI above 0; otherwise mixed.
6. **Seeds:** the pooled D averages the three seeds through the estimator's cells; per-seed D (plan rows of split k only) is reported next to it. The bootstrap is conditional on the fitted bases.
7. **Held-out IIA:** one estimand, swap-weighted over all held-out cross-class swaps (the 15 folds pooled), for both the displayed values and the paired difference.

## Result (Nov 23; TSUBAME jobs 9531586, 9532448; report `reports/pythia/das_reverse_freq.md`)

**Checks.** F and B pass every check (self-patch 0; held-out recompute medians ≤ 3.5e-3 at 17 and ≤ 4e-4 at 8 / 12, maxima ≤ 0.39; the stored v reproduces `transfer_summary.csv`). U misses the Amendment 2 median bar at site 12 by 1e-5 (0.00101 against 0.001). The analyzer applied that failure to all three conditions and printed "invalid" for each. Read per condition, as the check is per passive run: **F (primary) and B are valid; U at site 12 is invalid** (its numbers are shown but not read).

**F (primary), Head band: transfers at both test sites.**
- Site 8: D_bad(*by*) 1.09 [0.96, 1.22], v 1.08; ratio 1.00 [0.89, 1.14]; D_rare − 0.5·D_v 0.54 [0.43, 0.67] (retains at least half).
- Site 12: 0.99 [0.88, 1.10], v 1.05; ratio 0.94 [0.85, 1.04]; contrast 0.46 [0.37, 0.56].
- All bands: ratio 1.02 [0.93, 1.13] at 8 and 0.92 [0.85, 0.98] at 12. The three seeds agree (Head ratio 0.98–1.04 at 8, 0.92–0.96 at 12).
- **B (both sides rare, 27 pairs): transfers at 8 and 12** (Head ratio 1.01 [0.89, 1.15] and 0.92 [0.82, 1.02]).
- Site 17 (signed retention, descriptive): the rare directions reproduce v's *by* FALL on Head at 0.79 (F) and 0.61 (B) of v's size.
- Held-out active IIA on v's frequent pairs is lower than v's own (F: 0.63 vs 0.78 at 8, 0.81 vs 0.90 at 12), and the directions sit at |cos| 0.80–0.83 to v (P, trained on frequent plus rare verbs: 0.91–0.92).

**Reading.** A direction learned only from rare verbs moves frequent verbs' passive *by* preference as much as v does (Head ratio 0.94–1.00 with F; 0.92–1.01 with both sides rare), although it controls their actives somewhat less well. Filtering makes no difference (F vs U, 1 of 43 pairs dropped). So the class feature the passive reads is shared across frequency; the rare-verb deficit is not a missing feature.
