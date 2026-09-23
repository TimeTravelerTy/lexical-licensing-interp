# Why the curated passive pilot looks easier than FreqBLiMP

The paper reports a large passive accuracy decline from Head to XTail, but the
curated Pythia-1.4B pilot was near ceiling. I tested the same model with the
same full-sentence LP scorer on three item sets:

1. **Released:** all 1,000 FreqBLiMP pairs per band and passive paradigm.
2. **Crossed:** the pilot's *exact good/bad verb pairings* placed into released
   FreqBLiMP contexts wherever the good participle occurred. The good
   sentence is byte-identical to the released item; only its bad participle
   changes. This yields 1,339 pairs and retains all 26/50/50 selected good
   verb types in each paradigm.
3. **Curated:** the 504 pilot pairs with manually chosen, plausible patient
   and agent nouns and the `was` / `had been` templates.

The released and crossed TSUBAME jobs were `8763375` and `8763398`; both had
exit status 0. The released dataset files match the copy used for the paper
byte-for-byte. The code revisions were `d4c8501` and `98c2b56`,
respectively. The released and crossed score file SHA-256 values are
`7da8b88ba8f33a813646727f37beded018d76154d1b593934bf4c83cb5e3cbc0`
and `063c4e11b8e1ff50a92836c1fb39143da7573e94994ae398707e316c01a70191`.
The [FreqBLiMP paper](https://arxiv.org/html/2609.07153v1)
reports LP accuracy averaged over seven Gemma/Llama/Qwen/Mistral base models;
Pythia was not one of those models. All Pythia rows below were scored by
`scripts/score_matched_passives.py` under the same settings.

## Accuracy comparison

Percentages are item-level accuracy. The paper row is a seven-model mean;
the other rows use Pythia-1.4B. The crossed set has more contexts for some
verbs than others, so its item average is descriptive; pair-equal comparisons
follow below.

| Paradigm | Item set | Head | Tail | XTail |
| --- | --- | ---: | ---: | ---: |
| `passive_1` | Paper, seven-model mean | 80.4 | 74.9 | 65.8 |
| `passive_1` | Pythia, released | 78.7 | 79.5 | 70.0 |
| `passive_1` | Pythia, exact pilot verbs in released contexts | 82.4 | 80.6 | 71.4 |
| `passive_1` | Pythia, curated pilot | 100.0 | 99.0 | 99.0 |
| `passive_2` | Paper, seven-model mean | 81.6 | 78.4 | 71.4 |
| `passive_2` | Pythia, released | 81.6 | 79.9 | 72.7 |
| `passive_2` | Pythia, exact pilot verbs in released contexts | 81.8 | 83.9 | 73.5 |
| `passive_2` | Pythia, curated pilot | 100.0 | 99.0 | 96.0 |

Pythia therefore **does exhibit the released benchmark's passive decline**:
Head to XTail is -8.7 percentage points in `passive_1` and -8.9 in
`passive_2`. Model choice does not explain the near-ceiling pilot. Crucially,
the *same selected verb pairings* remain difficult in released contexts. In
XTail, changing from those contexts to the curated ones increases accuracy
from 71.4% to 99.0% in `passive_1` and from 73.5% to 96.0% in `passive_2`.
When each selected verb pair gets equal weight, the curated-minus-crossed
accuracy change is +29.0 points (95% pair-bootstrap interval +21.9 to +36.6)
and +21.4 points (+12.9 to +29.7), respectively. The comparison changes
several context properties together: noun frequency, semantic fit,
determiners, number, auxiliary tense, and negation. It does not isolate one
of those properties as the cause.

## What the limited head bad pool can explain

The released Head files have only **20 unique bad participles** in each
paradigm, versus 80/79 in Tail and 99 in XTail. Head items repeat those forms
roughly 50 times each. Giving each bad participle equal weight instead of
each sentence barely changes Pythia's Head accuracy: 78.7% to 78.5% in
`passive_1`, and 81.6% to 81.6% in `passive_2`. Thus unequal repetition of
the head forms is not producing the measured decline.

Four released Head bad forms have plausible transitive/passive readings in
some settings (`progressed`, `remarked`, `screamed`, `reacted`). Their effect
is mixed in these specific sentences: Pythia accuracy is 64.9% on these 208
`passive_1` items versus 82.3% on the other 792; for `passive_2` it is
85.3% on 217 such items versus 80.6% on the others. Replacing the bad form
with the pilot's matched form in released contexts changes Head accuracy
from 74.5% to 82.4% on the available `passive_1` rows, but from 83.0% to
81.8% on the available `passive_2` rows. This does not support a single,
large head-pool artifact across both paradigms. The small Head inventory
still limits lexical coverage and could affect the size of a Head–XTail
comparison; a counterfactual full-size Head inventory would be needed to
quantify that contribution.

## Why the curated XTail items are easier

The selected verbs alone are not an unusually easy subset: placing their
exact pairings back into released contexts restores XTail accuracy close to
the full released result. Tight within-pair participle-frequency matching
also does not remove the weakness in the released data. Among released XTail
pairs with good/bad participle Zipf gap at most 0.35, Pythia scores 68.6%
in `passive_1` and 72.5% in `passive_2`, close to 70.0% and 72.7% overall.

The pilot gives every good verb a patient and agent that make its passive
plausible, and it holds the surrounding vocabulary common across bands. The
released generator instead samples rare noun phrases under coarse semantic
features. Consider the *same* XTail verb pairing:

> The hat was doffed by the gentleman. / *The hat was salivated by the gentleman.
>
> Some morticians are doffed by that acrophobia. / *Some morticians are salivated by that acrophobia.

Pythia's margin is +7.30 for the curated sentence and -7.18 for the released
context. Other low-margin released-context good sentences include “The
embroiderers were guzzled” and “Most bootlickers aren't mulched by some
gleaners.” Their verbs are passivizable, but the patient/agent combinations
are semantically strained. These are examples, not a formal semantic-fit
rating. The paper's human validation also finds declining naturalness and
more non-decisions in its declining-paradigm group at XTail, although those
figures are not passive-specific.

The participle itself accounts for much of the context change. In
`passive_1` XTail, the average verb-token margin is +1.42 in the crossed
released contexts versus +4.69 in the curated contexts. The good/bad verb
token counts are nearly identical in the crossed set (2.49/2.49), so this
large difference cannot be attributed simply to one side using more tokens.

## Reading for the thesis

The released passive weakness is real for Pythia under the benchmark's full
lexicalization, and it is not explained away by the 20-form Head bad pool or
by selecting particular good/bad verb types. The near-ceiling pilot removes
much of the **rare surrounding-vocabulary and semantic-fit challenge** at
the same time. The remaining Head–XTail decline among crossed verb pairings
shows that these contexts can expose a frequency-sensitive readout, but it
does not distinguish verb knowledge from noun-verb compatibility, rarity of
the surrounding nouns, or the more complex sentence frames.

For a causal passive experiment, the next behavioral design should cross
each selected verb pair with multiple independently rated, plausible noun
contexts at each frequency band, holding the auxiliary frame constant. That
would retain clean grammatical labels while testing whether the current
near-ceiling result depends on a small set of especially supportive
contexts. The released benchmark remains useful as an external behavioral
check, with its context effects reported explicitly.

Reproduction files: `summary.csv`, `head_bad_types.csv`, and
`context_effect.csv` here; adapters in
`scripts/build_freqblimp_original_passives.py` and
`scripts/build_original_contexts_v2_verbs.py`; raw scores in
`results/freqblimp_original_passives/pythia14b_scores.csv` and
`results/freqblimp_original_contexts_v2_verbs/pythia14b_scores.csv`.
