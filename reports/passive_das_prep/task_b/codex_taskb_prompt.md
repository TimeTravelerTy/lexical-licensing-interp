# Research-design question: DAS training target for "verb takes a direct object"

## Setup
Model: Pythia-1.4B. Data: FreqBLiMP-style passive minimal pairs, e.g.
- good passive: "The hat was doffed (by the gentleman)."
- bad passive:  "The hat was salivated (by the gentleman)."
Verb pairs (transitive "good" verb, strictly intransitive "bad" verb) are
banded by participle frequency: Head (common), Tail, XTail (rare).

Plan: Distributed Alignment Search (DAS; Geiger et al.) learns a low-dim
subspace d on **actives** with Head verbs: base = intransitive active
("The gentleman salivated"), source = transitive active with the same
subject ("The gentleman doffed"). Interchange d at the verb position; the
training label is the source's class (transitive => an object should come
next). Then transfer: patch d from a transitive active into a **bad passive**
("The hat was salivated") at the participle. Two readings:
- d = surface "an object comes next": patched passive gains object-start
  tokens (" the", " a"...).
- d = abstract "this verb takes an object" (passivizable): patched passive
  gains log P(" by"), no gain in " the".
Natural baseline: good vs bad passives do NOT differ in " the"/" a" mass
after the participle (|diff| < 0.4 nats), while actives differ by ~4 nats on
" the". Training on actives with an object-next target makes d mean
"object next" in actives by construction; that is intended. The passive is
where the two readings come apart.

Constraints and facts:
- Head has only 25 usable pairs (20 train / 5 held-out); we are expanding.
- All Head verbs are single BPE tokens; almost all Tail/XTail verbs are
  multi-token.
- Subjects: 90 agent nouns ("The gentleman", "The war", "The news"...).
- Negative *by* margins in passives come mostly from transitive verbs that
  select a particle/preposition (summed up, larded with, shooed away).

## Candidate training targets (all on actives, at the verb)
1. log P(" the") - log P(".")  (current)
2. log P(O) - log(1 - P(O)), O = curated object-start set. Accusative
   pronouns (" him", " them", " her", " it", " me", " us") are cleaner object
   signals than determiners, which can start adverbials ("the next day").
3. log P(O) - log P(I), I = curated intransitive-continuation set (".",
   ",", prepositions, adverbs).
4. Match the source run's full next-token distribution (KL). Probably
   encodes far more than transitivity (verb identity, selectional
   preferences).
5. Anything better you can propose.

## Measurements we already have (Pythia-1.4B, next-token log-probs after the verb)
O = the, a, an, his, her, their, its, this, some, him, them, it.
I = ".", ",", "\n", " and", to, in, with, on, at, for, from, as, into, over.
Particles (up, out, off, down) are in neither set. " that" is excluded from O
because after say/complain verbs it starts a clause.
"both" = fraction of (verb pair, subject) items where the transitive score is
> 0 AND the intransitive score is < 0. "order" = transitive score > intransitive.

| split (pairs x 90 subjects) | target | trans mean | intrans mean | trans>0 | intrans<0 | both | order |
|---|---|---:|---:|---:|---:|---:|---:|
| head_dev (6) | 1 the vs . | 4.84 | -0.77 | 100% | 65% | 65% | 99.8% |
| head_dev | 2 O vs rest | 0.06 | -5.13 | 61% | 100% | 61% | 100% |
| head_dev | 3 O vs I | 2.04 | -4.57 | 96% | 100% | 96% | 100% |
| head_dev | pronouns only vs I | -0.28 | -7.01 | 46% | 100% | 46% | 99.6% |
| head_train (20) | 1 | 4.68 | -1.97 | 99% | 85% | 84% | 99% |
| head_train | 2 | -0.28 | -4.48 | 45% | 100% | 45% | 99% |
| head_train | 3 | 2.14 | -4.04 | 93% | 100% | 93% | 99.8% |
| xtail transfer (50) | 1 | 3.56 | -1.53 | 96% | 71% | 68% | 97% |
| xtail transfer | 3 | 1.61 | -2.67 | 84% | 96% | 80% | 99% |

Observations:
- Under target 1, Head intransitives fail mostly because they continue with
  a clause or PP: testified/complained/replied/bragged -> " that" (0.4-0.7),
  " to", " about", " of". These are not object-like.
- Intransitive items where O beats I under target 3 are mostly rare verbs:
  jutted -> " his" 0.33 / " out" 0.19 ("jutted his chin", a body-part
  object); fizzed / scrammed / vegetated / resounded / crackled -> " the"
  0.15-0.28 (unclear: adverbial or noise for a poorly known verb).
- Transitive failures under target 3: " himself" after crowned (reflexives
  are not in O), " by" after targeted (P = 0.10) / insured (0.05). That is,
  "The soldier targeted" can be read as a reduced relative ("The soldier
  targeted by the sniper..."). Mean P(" by") after the verb in actives is
  0.016 for transitives vs 0.011 for intransitives (Head train).
- "bet" is a tokenization artifact (continues as "betrays", "bettered").

## Questions
1. Which target best isolates "takes a direct object" in actives while
   keeping the passive transfer test meaningful? Argue for one, and give
   the strongest argument against your own choice.
2. How should the target interact with the passive readout: should the
   training contrast exclude " by" from both sets, include it in I, or
   neither? Does the reduced-relative reading of actives contaminate d with
   passive information, and how would you design the actives to avoid it
   (e.g. pronoun or proper-name subjects)?
3. Token position for multi-token verbs: last subtoken, first, or all
   subtokens? Consider that in passives we patch at the participle and
   Tail/XTail verbs are multi-token while Head verbs are not.
4. Intransitive items whose natural continuation prefers object-start tokens:
   given the continuations above, which should be excluded, at what level
   (item, verb, or none), and on what basis that is not circular?
5. What would make the passive transfer result uninterpretable under your
   chosen target, and what controls are needed (e.g. random-subspace,
   swapping between two transitives, intransitive -> intransitive)?

Be concrete and critical. Point out weak assumptions in this setup.
