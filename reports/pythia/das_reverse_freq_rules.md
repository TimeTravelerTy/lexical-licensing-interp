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
