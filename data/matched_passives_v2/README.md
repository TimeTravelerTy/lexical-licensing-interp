# Diverse-context passive behavioral gate

This is the second Pythia-1.4B passive pilot. It uses FreqBLiMP's released
passive good verbs and tail/xtail bad verbs, plus the audited head bad pool on
the `codex/passive-head-coverage` branch of `freqblimp-generation` (commit
`50eabb4`). `scripts/build_diverse_passives.py` makes one-to-one good/bad verb
matches and combines each with a manually checked patient and agent noun.
The model input is `pairs.jsonl`; `audit.json` records the candidate counts
and exclusion decisions.

There are **26 head, 50 tail, and 50 xtail distinct verb pairs**. Each pair
occurs in both `passive_1` (with a *by* phrase) and `passive_2` (without one),
and with *was* and *had been* auxiliaries. This yields 504 sentence pairs.
Good/bad sentences within a pair have exactly the same surrounding words.
Each verb appears once per band and paradigm; the two paradigms reuse the
same lexical pair by design. Patients span people, objects, locations,
events, and abstract nouns, with contexts listed in `contexts.csv`.

The band definition follows the **realised participle** frequencies in the
released FreqBLiMP data. Good and bad verbs are also matched within 0.25
Zipf for their lemmas and 0.35 Zipf for their participles. The tighter
condition in which *both lemmas also fall inside the band's window* contains
25 head, 18 tail, and 23 xtail pairs. The form-only primary analysis and
that lexical sensitivity analysis should be reported separately. `vanish`
has lemma Zipf 3.39 but participle Zipf 3.72, as in released head data.

The head inventory is below the requested 50 because manual review found
plausible passive readings for several released bad verbs, and fewer than 50
clear alternatives within the 3.5–5.5 participle band. The experiment keeps
the audited lexical count. The head review distinguishes simple
intransitives from verbs that select or prefer prepositions; results for
those subgroups should also be inspected.

Patient and agent nouns vary by verb pair. This increases lexical coverage
but does not hold every surrounding word fixed across all frequency bands.
Pair-level margins and leave-one-pair-out checks are therefore central to
interpretation. A flat or noisy result is inconclusive at this size.

To regenerate the input locally:

```bash
python3 scripts/build_diverse_passives.py \
  --freqblimp-root /path/to/freqblimp-generation \
  --head-review /path/to/freqblimp-generation/generation_projects/blimp/passive_head_review.csv
```

Score `pairs.jsonl` with `scripts/score_matched_passives.py`, then run
`scripts/summarize_matched_passives.py --design diverse` and
`scripts/summarize_diverse_sensitivity.py`. Resample **verb pairs**, not
individual templates, for intervals.
