# Matched-context real-verb passive pilot

## Question and design

Does Pythia 1.4B lose its preference for passivizable verbs as verb frequency
decreases when surrounding words are fixed?

`scripts/build_matched_passives.py` reads the canonical FreqBLiMP `passive_1`
and `passive_2` JSONL files for the `head`, `tail`, and `xtail` bands. It uses
only verb forms actually observed on the good or bad side of those files. The
FreqBLiMP vocabulary tables map participles to lemmas; `wordfreq` English
lemma Zipf values determine frequency bands and within-band pairing. The
generator excludes ambiguous mappings and lemmas outside the nominal band.
This is stricter than the original generator, which bands the realized form
and may backfill candidates when a pool is small.

The primary pilot uses an explicit list of passivizable verbs that can
plausibly take a human patient. Every verb pair appears in the same eight
human-patient frames. `passive_1` has `was VERBed by AGENT`; `passive_2` has
`was VERBed`. The good/bad sentences in a pair differ only in the participle.
The same contexts are crossed with all frequency bands. The two paradigms
remain separate because `by` gives `passive_1` an additional construction cue.

Generate the input locally or on TSUBAME:

```sh
python3 scripts/build_matched_passives.py \
  --freqblimp-root ../freq-blimp \
  --out data/matched_passives/pairs.jsonl \
  --audit-out data/matched_passives/audit.json
```

Use `--inventory all` for an exploratory broad-inventory variant; its shared
frames can be semantically odd for many passivizable verbs, so it is not the
primary result. The default primary input is committed to the repo, and the
audit records source counts and exclusions. The generated input is meant to be
reviewed before scoring. The source files contain a few hyphenated forms
(`baby-sat`) that the single-word pilot intentionally excludes.

## Scoring and analysis

Score unprompted full-sentence log probability with the same Pythia model and
autoregressive log-probability convention as the prior behavioral work: sum
log probabilities for all tokens except the first token. The margin is
`log P(good sentence) - log P(bad sentence)`. Accuracy is the fraction of
positive margins. A tie counts as incorrect. The scorer also saves the verb,
post-verb suffix, and `by` contributions; these locate where a margin arises.

```sh
python3 scripts/score_matched_passives.py \
  --model EleutherAI/pythia-1.4b \
  --data data/matched_passives/pairs.jsonl \
  --out results/matched_passives/pythia14b_scores.csv
python3 scripts/summarize_matched_passives.py \
  --scores results/matched_passives/pythia14b_scores.csv \
  --out-dir reports/matched_passives
```

`summary.csv` reports accuracy, mean margin, and cluster-bootstrap 95% intervals
by paradigm and frequency band. The bootstrap resamples verb pairs, averaging
their eight context results first. `per_verb_pair.csv` and `by_frame.csv`
preserve heterogeneity. Do not interpret 8 frames for one verb pair as 8
independent lexical observations.

## Interpretation limits

Even with identical frames, verb identity, meaning, participle frequency, and
tokenization differ. The input records lemma and participle Zipf values, and
the summary reports mean verb token counts. A trend is an association with
verb frequency under controlled context, not a causal effect of exposure.
Some good passives may still be semantically unusual in particular frames;
inspect `by_frame.csv` and individual errors before drawing a mechanism claim.
The small curated pilot is a gate for a larger study, not a final estimate of
Pythia's passive ability.
