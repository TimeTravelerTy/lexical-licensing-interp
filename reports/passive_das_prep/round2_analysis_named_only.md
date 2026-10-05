# DAS round 2: active-frame readout, behaviour filter, matching

Readout: TSUBAME job 8900510 (Pythia-1.4B, bf16), `results/das_round2/pythia14b_readout.csv`. Target (option 3): M = log P(O) - log P(I) at the verb's last token. O = the, a, an, his, her, their, its, my, our, your, this, these, those, some, every, each, several, him, them, it, me, us, himself, herself, themselves, itself. I = . , newline, and, prepositions (to, in, with, on, at, for, from, as, into, over, about, of, upon, against, through, after, under, around, across, toward, behind, near, onto, during, without, like, until, before, since), adverbs and connectives (here, there, again, already, never, not, so, just, well, together, alone, too, but, or, because, when, while), ! ? ; :. In neither: " that", " by", particles (up, out, off, down, away, back). `r1 sets` = the round-1 sets (12 O / 14 I tokens) for comparison.

## 1. Readout on the new frame

Training pool: 29 expansion pairs (plain intransitives only, rematched) plus 10 original Head pairs with a plain intransitive (`orig_head`, optional). New frame: "She has <participle>" (one item per verb) and the same with 7 subjects pooled (She, He, They, We, I, Maria, David). Old frame: "The AGENT <past>" (90 subjects). Accuracy = % items on the correct side of 0.

| Source | Band | Tokens | Class | Verbs | She has | 7 subj (new) | Old frame | 7 subj, r1 sets | Old, r1 sets | Coverage new | Coverage old |
|---|---|---|---|---|---|---|---|---|---|---|---|
| expansion | head | multi | trans | 4 | 100% | 100% | 84% | 100% | 85% | 0.53 | 0.72 |
| expansion | head | single | trans | 2 | 100% | 100% | 91% | 100% | 92% | 0.65 | 0.67 |
| expansion | near_head | multi | trans | 17 | 100% | 100% | 94% | 100% | 91% | 0.62 | 0.73 |
| expansion | near_head | multi | intrans | 21 | 100% | 100% | 100% | 100% | 100% | 0.79 | 0.84 |
| expansion | near_head | single | trans | 6 | 83% | 93% | 86% | 93% | 85% | 0.68 | 0.73 |
| expansion | near_head | single | intrans | 8 | 88% | 93% | 99% | 93% | 99% | 0.80 | 0.88 |
| orig_head | head | single | trans | 10 | 100% | 97% | 94% | 100% | 96% | 0.60 | 0.67 |
| orig_head | head | single | intrans | 10 | 100% | 100% | 100% | 100% | 100% | 0.76 | 0.83 |

Pair-level ordering (M_trans > M_intrans, same subject): new frame 100%, old frame 100%. Coverage = median P(O) + P(I).

## 2. Behaviour filter (new frame, 7 subjects)

Item = (verb, subject). An item passes if M is on its class's side of 0. A verb passes if at least half its 7 items pass. Rejects are kept (`behavior_filter_verbs.csv`).

| Source | Band | Tokens | Class | Verbs | Item pass | Verbs passing |
|---|---|---|---|---|---|---|
| expansion | head | multi | trans | 4 | 100% | 4/4 |
| expansion | head | single | trans | 2 | 100% | 2/2 |
| expansion | near_head | multi | intrans | 21 | 100% | 21/21 |
| expansion | near_head | multi | trans | 17 | 100% | 17/17 |
| expansion | near_head | single | intrans | 8 | 93% | 7/8 |
| expansion | near_head | single | trans | 6 | 93% | 6/6 |
| orig_head | head | single | intrans | 10 | 100% | 10/10 |
| orig_head | head | single | trans | 10 | 97% | 10/10 |

Class balance after the verb-level filter:

| Source | Pairs | Transitives passing | Intransitives passing | Pairs with both passing |
|---|---|---|---|---|
| expansion | 29 | 29 | 28 | 28 |
| orig_head | 10 | 10 | 10 | 10 |

Rejected verbs (She-has M; for intransitives, M > 0 means the model treats the verb as transitive in this frame):

- intrans: ensue (+0.0, 43%)
- trans: none
  - She has ensued: ` a` 0.19, ` in` 0.09, ` the` 0.07, ` with` 0.05, `,` 0.04, ` an` 0.03

### Projection groups (for later; She-has M)

| Group | Verbs | Mean M | % M > 0 |
|---|---|---|---|
| included transitives (all bands) | 220 | +1.80 | 98% |
| plain intransitive | 134 | -2.89 | 3% |
| prep_object | 84 | -3.63 | 0% |
| contaminated_bad | 7 | -0.65 | 14% |
| bad-side-high *by* (original) | 11 | -2.54 | 0% |

contaminated_bad: crackle (M -1.2), hibernate (M -1.9), jut (M +1.8), pee (M -1.3), reel (M -1.6), resound (M -0.3), scram (M -0.1)

Their *by* margins (released contexts): daub/resound -2.85 (reliable_neg, good side low); espy/scram +0.53 (reliable_pos); placate/crackle +1.19 (reliable_pos); prise/jut -0.32 (reliable_neg, good side low). Overlap with the bad-side-high group: none.

## 3. Familiarity: participle Zipf

Expansion pairs: participle Zipf trans 3.30 (0.37) vs intrans 2.88 (0.37). 10/29 existing pairs already have |participle gap| <= 0.35. Rematching the same pool under both calipers (summed-lemma <= 0.25, participle <= 0.35) gives 21 pairs (8 single-token, 13 multi-token): `data/das_round2/train_pairs_participle_matched.csv`.
Robustness set: participle Zipf trans 3.03 (0.35) vs intrans 3.01 (0.34); summed-lemma 3.85 (0.29) vs 3.88 (0.27).
Pairs: pardon/faint, taunt/relapse, thwart/succumb, hoard/tremble, tow/intervene, censor/coincide, assassinate/perish, obey/defect, lynch/triumph, sabotage/loom, worship/reign, counterfeit/feud, solicit/sneeze, transplant/hesitate, visualize/err, underestimate/persist, withhold/ensue, compute/reside, copyright/arise, rinse/kneel, grip/differ

`orig_head` pairs are matched on lemma and participle Zipf by construction: participle Zipf trans 4.33 (0.52) vs intrans 4.37 (0.56).

## 4. Token count in the expansion pool (near_head intransitives; transitives near_head or Head)

| Participle tokens | Class | Verbs in pairs | Passing filter |
|---|---|---|---|
| multi | intrans | 21 | 21 |
| multi | trans | 21 | 21 |
| single | intrans | 8 | 7 |
| single | trans | 8 | 8 |

Pairs per token group (all / both verbs passing): multi 21 / 21, single 8 / 7.
