# Task B: DAS training target on actives

Question: which target best isolates "takes a direct object" in actives while
keeping the passive transfer test (patch d into "The hat was salivated";
" by" up vs " the" up) meaningful?

## Measurements (existing readout, no new model runs)

`scripts/compare_das_targets.py` scores each candidate target on the
`data/passive_das/actives.jsonl` items ("The AGENT VERB-ed", 90 subjects per
verb pair, readout at the verb's last token). Full table:
`target_comparison.csv`; intransitive items that look object-like:
`intrans_object_like.csv`.

O = the, a, an, his, her, their, its, this, some, him, them, it.
I = ".", ",", "\n", " and", to, in, with, on, at, for, from, as, into, over.
Particles (up, out, off, down) are in neither set. " that" is in neither set
(see below). "Both" = transitive score > 0 and intransitive score < 0 for
the same subject; "order" = transitive score > intransitive score.

| Split | Target | Trans > 0 | Intrans < 0 | Both | Order |
|---|---|---:|---:|---:|---:|
| head_dev (6 pairs) | 1 " the" vs "." | 100% | 65% | 65% | 99.8% |
| | 2 O vs rest | 61% | 100% | 61% | 100% |
| | **3 O vs I** | 96% | 100% | **96%** | 100% |
| | pronouns vs I | 46% | 100% | 46% | 99.6% |
| head_train (20) | 1 | 99% | 85% | 84% | 99.4% |
| | 3 | 93% | 100% | 93% | 99.8% |
| transfer_xtail (50) | 1 | 96% | 71% | 68% | 97.4% |
| | 3 | 84% | 96% | 80% | 98.7% |

What the failures are:

- **Target 1, intransitive side:** *testified / complained / replied /
  bragged* continue with " that" (0.3-0.7), " to", " about", " of", i.e. a
  clause or PP. These are not object-like. " that" was in the first version
  of O and caused the same failures for target 3, so it is now excluded
  from both sets: after say-type verbs it starts a clause, and after
  transitives it can be a demonstrative object ("took that book").
- **Target 3, intransitive side** (rare verbs only): *jutted* → " his" 0.33
  ("jutted his chin", a real transitive use); *fizzed, scrammed, vegetated,
  resounded, crackled* → " the" 0.15-0.28.
- **Target 3, transitive side:** reflexives (*crowned* → " himself") are not
  in O. *targeted* → " by" 0.10 and *insured* → " by" 0.05: "The soldier
  targeted" also reads as a reduced relative ("The soldier targeted by the
  sniper…"). Mean P(" by") after a Head active verb is 0.016 (transitive)
  vs 0.011 (intransitive): small on average, but concentrated in a few verbs.
- *bet* continues as "bet|rays" / "bet|tered" in actives too.

## Codex (GPT-6.1-Sol, xhigh), verdict

Full answer: `codex_taskb_answer.md`.

- **Target 3**, scored as binary cross-entropy on q = σ(log P(O) − log P(I)),
  with the source's class as the label. It fixes the main flaw of target 1
  ("." is a poor stand-in for intransitive continuations). Unlike target 2,
  it does not require O to beat the whole vocabulary, and unlike KL it does
  not reward copying the source verb's lexical preferences. Add " me",
  " us" and reflexives to O. Keep determiner-only and pronoun-only scores as
  diagnostics. Record the coverage C = P(O) + P(I), because a clean q with
  tiny C is a contest between two unlikely options.
- **Strongest argument against its own choice:** target 3 measures an
  immediate continuation preference, not direct-object licensing. It could
  pick up a mix of argument structure, collocation and construction
  preference. Codex suggests validating with complete annotated
  continuations (full objects vs full non-object continuations, including
  determiner-initial temporal adjuncts).
- **" by": in neither set.** With target 3 the softmax normaliser cancels, so
  the " by" logit has no direct effect on the loss. Putting " by" in I
  would mix the transfer readout into training.
- **Reduced relatives:** fix the active frame. Pronoun subjects ("She
  targeted") as the primary frame, and "She has targeted" as a strong
  control, plus proper names and the current ambiguous frame as
  robustness conditions. Hold out templates.
- **Position: last subtoken**, one rank-d intervention, the same site in
  both voices. The first subtoken has not seen the rest of the word; "all
  subtokens" changes intervention strength and capacity. Head data cannot
  decide this (all single-token), so fix the rule in advance and test common
  multi-token verbs as a bridge.
- **Object-like intransitives:** decide on independent lexical grounds, not
  M > 0. *jut, scram, crackle, resound* have dictionary transitive uses
  (jut one's chin, scram a reactor), so they are not strictly intransitive.
  Keep *fizz, vegetate* unless annotation says otherwise. Do not drop items
  just because the model gives them M > 0.
- **Controls:** swaps in both directions; transitive → transitive and
  intransitive → intransitive; many donors per passive base; intransitive
  active → bad passive (including the same verb); random and
  norm-matched subspaces; shuffled-label DAS; good passive → bad passive as
  a positive control for the site and readout; several seeds and lexical
  splits; an equivalence bound for "no rise in ' the'". Freeze target,
  layer, dimension and position on active dev data before looking at
  passives.

## My view

I had settled on target 3 before asking, so the agreement is real but not
independent of the same evidence. I agree with Codex on target 3, keeping
" by" out of both sets, the last subtoken, swaps in both directions, the
coverage check, and the controls list. Codex is right about *jut, scram,
crackle, resound*. Those are bad verbs in the original 126 pairs
(jut/prise, scram/espy, crackle/placate, resound/daub). I have not changed
the set; they should be flagged in the hand check.

## Where I disagree or would go further

1. **Primary active frame: "She has VERB-ed", not "She VERB-ed".** Codex
   makes the perfect a control. I would make it primary:
   - it removes the reduced-relative reading entirely, not just mostly;
   - the patched token is then the **participle**, the same string (and
     the same subtokens) as in the passive test. That removes a train/test
     form mismatch (*outdid* vs *outdone*), and for multi-token verbs it
     makes the verb-final subtoken identical across voices;
   - "has + participle" vs "was + participle" then differs only in the
     auxiliary, which is the cleanest possible voice contrast for transfer.

   The cost is that "has" may lower object expectation slightly, or invite
   "has been". Both are measurable with one cheap readout run before
   choosing. Simple past with pronoun subjects stays as the replication
   frame.
2. **Behavioural filtering of the training set.** Codex warns against
   selecting items by M. For the *evaluation* population I agree. For DAS
   *training* I would keep items only where both unpatched runs behave
   according to their class (as Geiger et al. do). Otherwise the objective
   asks the intervention to produce behaviour the model never shows for
   that item. The passive test is a separate population, so this is not
   circular for the transfer claim. Report the filter rate per split.
3. **Label.** The class label with BCE is fine once training items are
   filtered as above. Without filtering I would use the source run's own q
   as a soft target, since the claim is "d carries what the source run
   carries".
4. **Pronouns.** Pronoun-only mass is too low and too dependent on the
   subject to be the training target (46% of transitive items). Codex
   treats its 99.6% ordering as a reason not to dismiss it. I keep it as a
   readout only.
5. **The " the" side of the passive test needs its own positive control.**
   Codex lists a positive control for the passive site via the " by"
   readout (good → bad passive). We also need to show that the patch *can*
   raise object starts somewhere. Patching d from a transitive active into
   an intransitive active at held-out verbs does this. Without it, "no rise
   in ' the'" in passives cannot be told apart from "d is too weak".

## Recommended setup (pending the readout check in point 1)

- Target: BCE on σ(log P(O) − log P(I)) at the verb-final subtoken.
  O = the, a, an, his, her, their, its, this, some, him, them, it, me, us,
  himself, herself, themselves, itself. I = ".", ",", "\n", " and", to, in,
  with, on, at, for, from, as, into, over. " that", " by" and particles in
  neither set. Also log coverage C.
- Frames: "She/He has VERB-ed" (primary), "She/He VERB-ed" (replication),
  noun subject (robustness). Hold out a frame.
- Training items: both unpatched runs behave per class; swaps in both
  directions.
- Before training: one readout run of the new frames on Head, Tail and
  XTail verbs (no DAS), to check pass rates and C for the new O.
