# Pairing rules for the expanded verb set (declared Nov 18, 2026, before pairing)

Paper plan, work item 13. Written before `scripts/pair_verb_expansion.py` was run.

## Inputs

- **Intransitives:** the 160 kept verbs in `intrans_spotcheck.csv` (`keep == y`), after the Claude / Codex spot-check and the weather-verb exclusion.
- **Transitives, two sources:**
  1. **Reviewed:** the 220 transitives that Codex's adversarial review kept (`verbs.csv`, `class == trans`, `included`).
  2. **FreqBLiMP:** participles with `verb, en, passive, strict_trans` set in `freq-blimp/vocabulary_overlay.csv`. These are BLiMP-curated passivizable transitives that have not been reviewed here. Lemma and past come from lemminflect.

  The second source is needed because the kept intransitives are mostly rare (tail, xtail and below: 79 of 160) and mostly multi-token, while the reviewed transitives are mostly head or near-head.
- **Excluded from both classes:** any lemma or participle in the 126 original pairs, and any lemma in the intransitive spot-check list, whether kept or dropped.
- **Superseded:** the earlier provisional pairs (`das_train_pairs.csv`, `eval_pairs.csv`) were never used. This pairing replaces them.

## Matching

- **Token count:** the Pythia-1.4B token count of " participle" must match exactly.
- **Frequency caliper, within 0.25 Zipf, on one scale per pair:**
  - if either verb is homograph-flagged (summed-lemma Zipf − participle Zipf ≥ 1.5), on participle Zipf;
  - otherwise, on summed-lemma Zipf.

  Both gaps are recorded for every pair.
- **Assignment:** greedy, smallest gap first, each verb used at most once, as for `das_train_pairs.csv`. Ties go to a reviewed transitive over a FreqBLiMP one, then alphabetically.
- **Unmatched intransitives** stay unpaired. They are reported, never forced in with a wider caliper.

## Check after pairing

Before any run, every FreqBLiMP transitive that gets selected is screened by Claude, using the same grounds as the review. It is dropped if it is not passivizable in its ordinary sense, if its participle is mainly adjectival (*tired*, *interested*), or if it is mainly intransitive. A dropped transitive's intransitive is re-paired under the same rules. The screen is recorded in the output file.

## Use

- **Analyses:** training-free only (behavior, u_md projection and transfer, LEACE). The verbs are evaluated in the existing curated contexts with the cross design; DAS keeps its 126-pair splits.
- **Results:**
  - the 126-pair results stay primary;
  - expanded-set results are reported as a replication on more verbs;
  - frequency is analyzed as a continuous covariate.
- **Robustness:** results are reported with and without pairs whose intransitive is `selection`-flagged, and with and without FreqBLiMP-source transitives.

## Result (Nov 18, after pairing)

- **Pairs:** 147 in `expanded_pairs.csv`; 13 intransitives had no transitive within the caliper (`expanded_unpaired.csv`; mostly rare 3-token verbs such as *scintillated*, *palpitated*, *pussyfooted*). With the 126 original pairs that makes 273 in all.
- **Transitive sources:** 47 reviewed, 100 FreqBLiMP.
- **Single-token pairs:** 25.
- **Match scale:** 105 pairs on summed-lemma Zipf, 42 on participle Zipf (homograph-flagged). Mean matched gap is 0.005. On participle Zipf the transitive is higher by 0.31 on average, because transitives use the participle form more, as in the earlier expansion.
- **Intransitive tags:** near-head 73, xtail 42, tail 12, between bands 8, head 6, below xtail 6.
- **Screen:** run in two passes.
  - Pass 1 dropped 18 FreqBLiMP transitives. Mainly adjectival: *related*, *deprived*, *frightened*, *misspent*, *globalized*, *encrusted*, *apprised*, *premised*. Mainly intransitive: *adjusted*, *adapted*, *drummed*, *bluffed*, *overreached*, *thieved*, *supplicated*, *contorted*, *bridled*. Obscure: *balled*.
  - Pass 2 re-paired their intransitives and selected 17 new FreqBLiMP transitives; all passed.
  - Dropped lemmas are passed with `--drop-trans`; the command is in the README.
- **Robustness subsets:** 17 pairs have a `selection`-flagged intransitive (16 inanimate-preferred, 1 animal).
