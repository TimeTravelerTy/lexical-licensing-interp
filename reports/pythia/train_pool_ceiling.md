# How many frequent plain T / I pairs exist for training v? (item 14 follow-up, Nov 23)

Script: `scripts/analyze_train_pool_ceiling.py` (TSUBAME job 9531497, Pythia-1.4B bf16, the round-2 readout and filter). Ledger: `data/verb_expansion/train_pool_ceiling.csv`, one row per frequent intransitive.

Intransitives are the binding class (transitives are plentiful), so the count is over frequent (head / near-head) intransitives in every candidate source, each followed through the selection steps in the order they were applied. Sources, deduplicated by lemma with the latest judgment winning: the 126 original pairs (Head band), the Codex expansion (`verbs.csv`, the round-2 pool), the FreqBLiMP strict-intransitive spot-check (Nov 18) and the R5-D11 spot-check (Nov 19).

| Step | Frequent intransitives |
|---|---:|
| Candidates (all sources) | 197 |
| Dropped at review / spot-check (ordinary NP objects, clause-taking, copular, homograph-inflated) | 73 |
| Kept | 124 |
| of which prepositional (`prep_object`: argument PP, excluded by the round-2 class rule) | 68 |
| of which unpairable (homograph nouns with an unseen participle: *eddied*, *weekended*) | 2 |
| **Plain** | **54** |
| Plain and passing the behavior filter (≥ 4 of 7 actives on the intransitive side) | 54 |

All 54 plain verbs pass the filter (so do all 68 prepositional ones), so the filter is not what limits the training set; the class rule and the review are.

**When v was trained (Oct 5, the round-2 pool):** the pool held 30 frequent plain intransitives (21 near-head from the expansion, 9 original Head). v uses 29 of them. The 30th, *appear*, was left out because its pair's transitive, *bet*, was excluded (`analyze_das_round2.orig_head`). So **29 was the ceiling of the pool available then.**

**Now:** the Nov 18–19 spot-checks added 24 frequent plain intransitives (FreqBLiMP 16, D11 8). In the pairing they are matched by Zipf, and many are noun homographs (*holiday*, *picnic*, *vacation*, *honeymoon*, *tango*) matched on participle Zipf, so their transitives are often rare. Pairs with both verbs head or near-head: 11 (*sleep, retreat, deteriorate, prosper, breeze, vacation, last, behave, lapse, stroll, lounge*). Two more (*flop*, *shiver*) are paired with a transitive in v's 29 (`das_train_overlap`). So a frequent-only training set could now reach about 40 pairs (29 + 11, up to 43 with re-pairing *flop*, *shiver* and *appear*). The Nov 18 P condition already adds 14 of the new frequent intransitives (with 47 rare or tail ones).

**Sentence for the paper (appendix):** "Of 197 frequent (head or near-head) intransitive candidates, 73 failed review and 68 take an argument PP; all 54 plain ones pass the behavior filter. v's 29 pairs used every usable one available when it was trained (29 of 30); a later spot-check raised the frequent-plain supply to about 40 pairs with frequent partners."
