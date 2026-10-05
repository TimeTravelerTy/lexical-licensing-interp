# Verb-set expansion for passive DAS

New transitive / intransitive verbs, kept separate from the 126 original
pairs. Nothing here is used yet; the sets are pending a manual spot-check.

## Files

- `verbs.csv`: all 406 candidates. Columns: forms, tag, participle /
  lemma / summed-lemma Zipf, Pythia tokens, `included`, `use` (`das_train`,
  `eval`, `unpaired`, `excluded`), a one-line `reason` for exclusions, and
  Codex's original justification.
- `das_train_pairs.csv`: 54 transitive / intransitive pairs for DAS
  training on actives.
- `eval_pairs.csv`: 19 new good / bad passive pairs (participle bands).
- `summary.md`: counts and matched distributions.
- `work/`: per-round Codex candidates and checks, and the Codex review.

## How the sets were built

1. **Generate → check, 5 rounds.** Codex (GPT-6.1-Sol) proposed candidates
   with a justification against the inclusion criteria. Prompts are in the
   session scratchpad and summarised in the rounds below.
   `scripts/check_verb_candidates.py` then checked each one:
   - wordfreq Zipf of the participle, the form FreqBLiMP bands by
     (`freq-blimp/utils/frequency.py`), with bands Head 3.5-5.5, Tail
     2.4-3.2 and XTail 1.2-2.2;
   - summed-lemma Zipf;
   - Pythia tokens of " participle" and " past";
   - duplicates.

   Rejects and gaps went back to Codex each round.

   | Round | Asked for | Proposed | Passed checks |
   |---|---|---:|---:|
   | 1 | four lists of ~50 | 168 | 167 |
   | 2 | Head intransitives, Head multi-token, single-token Tail, Tail/XTail transitives | 87 | 87 |
   | 3 | Head and single-token Tail intransitives only | 26 | 26 |
   | 4 | saturation check | 5 | 5 |
   | 5 | near-Head pool, both classes | 120 | 120 |

   The checks rarely reject anything, because Codex quoted Zipf values (it
   could look them up). The real limit was supply.
2. **Adversarial review.** Codex reviewed all 405 verbs as a hostile
   reviewer (keep / doubtful / drop).
3. **Inclusion.**
   - Transitives: included only if the review kept them (220 of 252).
   - Intransitives: included if kept, or doubtful only because of a
     prepositional complement (rely on, tamper with) or an obsolete or
     technical sense. Excluded: every Codex drop (ordinary NP objects such as
     speech objects, location objects, causatives; copular verbs;
     adjectival participles), clause-taking verbs (hope, insist, wonder),
     speech-object verbs, copular/relational verbs, and verbs whose
     summed-lemma Zipf is inflated by a noun or adjective homograph (long,
     major, party). These calls are listed in `OVERRIDES` in
     `scripts/finalize_verb_expansion.py`.

## Tags

- `head`, `tail`, `xtail`: FreqBLiMP participle-Zipf bands. Used for the
  Head label and for every eval / behavioural set.
- `near_head`: participle Zipf in (3.2, 3.5), or summed-lemma Zipf >= 3.5
  with the participle below Head. Used for DAS training only and excluded
  from all band comparisons. Priority: a Head participle stays `head`; a
  Tail participle with summed-lemma Zipf >= 3.5 becomes `near_head` and
  leaves the Tail eval.

## What we learned about supply

- **Head intransitives are nearly exhausted.** Most clean intransitives
  (persisted, ensued, perished, flourished, paused, erupted) have
  participle Zipf 3.2-3.5, in the gap between Tail and Head. The ones that
  reach Head mostly have ordinary object uses (shout, wait, stay, agree) or
  are copular (seem, become). Only 7 Head intransitives survive (care,
  react, rely, stare, step, stumble, subscribe), all with prepositional
  complements.
- **Head multi-token intransitives essentially do not exist** in Pythia's
  BPE: Head-band participles are almost always single tokens. The only one
  found is *subscribed*.
- **Single-token Tail/XTail intransitives are exhausted.** Codex said so,
  and all candidates were later dropped (speech objects) or moved to
  `near_head`. As a result the new eval pairs cannot separate rarity from
  token count either: Tail/XTail eval pairs are all multi-token.

## DAS training pairs

Intransitives tagged `head` or `near_head` and not used in eval pairs, each
matched to one transitive with the same participle token count and
summed-lemma Zipf within 0.25 (greedy, smallest gap first). The two classes
match on summed-lemma Zipf (mean gap -0.002). Participle Zipf is not
matched: transitives average 3.37 and intransitives 2.93, because
transitive verbs use the participle form more. 36 pairs are multi-token and
18 single-token. Almost all intransitives are `near_head` (53 of 54).
