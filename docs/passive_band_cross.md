# Verb band x context band passive cross

## Question

Released FreqBLiMP passives band nouns together with verbs: median subject
and agent Zipf fall from about 3.9 (Head) to 2.8 (Tail) and 1.7 (XTail), while
negation stays near 49% in every band. The released Head-to-XTail accuracy
decline therefore mixes verb rarity with noun rarity. The same v2 verb pairs
score 71% in released XTail contexts and 99% in curated ones.

This experiment crosses every v2 verb pair (26 Head, 50 Tail, 50 XTail; `was`
frame) with the **same** sampled released contexts from all three bands.
The diagonal cells are built exactly like the off-diagonal ones: no context
keeps the verb it was generated for. The contrasts are:

- **verb effect**: XTail minus Head verbs, averaged over context bands;
- **context effect**: XTail minus Head contexts, averaged over verb bands;
- **interaction**: (XTail verb, XTail ctx) - (XTail verb, Head ctx) - (Head verb, XTail ctx) + (Head verb, Head ctx).

A secondary set crosses every verb pair with every v2 curated context.
`own_context` marks the context written for that verb pair. Other verbs'
curated contexts keep common, well-formed nouns but lose the hand-picked
thematic fit, a rough "common but strained" condition.

## Build, score, analyze

```sh
python3 scripts/build_passive_band_cross.py            # 100 contexts per band x paradigm
python3 scripts/score_matched_passives.py \
  --data data/passive_band_cross/pairs.jsonl \
  --out results/passive_band_cross/pythia14b_scores.csv --batch-size 64
python3 scripts/analyze_passive_band_cross.py
```

`pairs.jsonl` (107,352 pairs, about 117 MB) is not committed. The builder is
deterministic; `manifest.json` records its SHA-256 and `contexts.csv` lists
every sampled context with its covariates (auxiliary, negation, number,
tense, determiner, proper-name flags, head-noun Zipf).

## Statistics

Verb pairs and contexts are both sampled units. Intervals come from a
two-way cluster bootstrap that resamples verb pairs within verb band and
contexts within context band independently. Cell means give equal weight to
each verb pair and each context. `context_regression.csv` fits
context-level OLS (margin averaged over verbs) to check whether the
context-band effect survives frame covariates, and whether head-noun Zipf
accounts for it. Plausibility ratings of the good sentences can be added
later as another context-level covariate without rescoring.
