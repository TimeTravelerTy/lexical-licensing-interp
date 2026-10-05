# Verb-set expansion

406 candidates from Codex over 5 generate-check rounds; 319 included. A verb is included if it passed the mechanical checks (participle-Zipf band, Pythia tokens, no duplicate of the 126 original pairs) and Codex's adversarial review kept it; for intransitives I override the review in 96 cases (see OVERRIDES in `scripts/finalize_verb_expansion.py`). Everything is still pending your spot-check.

Tags: `head` / `tail` / `xtail` are FreqBLiMP participle-Zipf bands (used for eval). `near_head` (participle Zipf 3.2-3.5, or summed-lemma Zipf >= 3.5 with the participle below Head) is for DAS training only and is never used in band comparisons.

## Included verbs

| Tag | Participle tokens | Transitive | Intransitive |
|---|---|---:|---:|
| head | multi | 21 | 1 |
| head | single | 48 | 6 |
| near_head | multi | 40 | 36 |
| near_head | single | 39 | 17 |
| tail | multi | 37 | 9 |
| tail | single | 25 | 0 |
| xtail | multi | 7 | 30 |
| xtail | single | 3 | 0 |

## Eval pairs (`eval_pairs.csv`): 19

| Band | Tokens | Pairs |
|---|---|---:|
| head | multi | 1 |
| head | single | 5 |
| tail | multi | 8 |
| xtail | multi | 5 |

## DAS training pairs (`das_train_pairs.csv`): 54

`head` + `near_head` verbs not used in eval pairs; each intransitive matched to one transitive with the same participle token count and summed-lemma Zipf within 0.25.

| Tokens | Pairs | Summed-lemma Zipf, trans | Intrans | Participle Zipf, trans | Intrans |
|---|---:|---|---|---|---|
| multi | 36 | 3.86 (0.33), range 3.51-4.68 | 3.86 (0.32), range 3.51-4.67 | 3.27 (0.32), range 1.96-3.73 | 2.78 (0.37), range 2.01-3.41 |
| single | 18 | 4.15 (0.47), range 3.52-5.47 | 4.16 (0.49), range 3.53-5.62 | 3.55 (0.55), range 2.88-5.03 | 3.23 (0.33), range 2.66-4.11 |
| all | 54 | 3.96 (0.40), range 3.51-5.47 | 3.96 (0.41), range 3.51-5.62 | 3.37 (0.43), range 1.96-5.03 | 2.93 (0.42), range 2.01-4.11 |

Summed-lemma Zipf gap (trans - intrans): mean -0.002, max |gap| 0.15. Tags: transitive {'near_head': 41, 'head': 13}, intransitive {'near_head': 53, 'head': 1}. Participle Zipf is not matched (only summed-lemma Zipf and token count).

Included but unpaired: 173 ({'trans': 147, 'intrans': 26}).

## Excluded

87 verbs; one-line reasons in `verbs.csv`. By source: drop 42, doubtful 41, keep 3, mechanical 1.
