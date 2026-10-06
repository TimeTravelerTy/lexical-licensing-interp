# Passive DAS groundwork (Pythia-1.4B, FreqBLiMP `passive_1` / `passive_2`)

Question: does Pythia represent that a verb takes a direct object, and does it
use that in passives? This folder collects the checks and reference
profiles needed before training DAS. No existing result file was changed.

| Task | Summary file | Tables |
|---|---|---|
| 0 Consistency | `task0_consistency.md` | `task0_own_context.csv`, `task0_by_reliability.csv` |
| 1 Natural baseline | `task1_baseline.md` | `task1_baseline.csv` |
| 2 *by* margin split | `task2_by_split.md` | `task2_by_split_per_verb.csv` |
| 3 Single-prompt readout | `task3_single_prompt.md` | `task3_single_prompt.csv` |
| 4 Yes/No | `task4_yesno.md` | `task4_yesno_cells.csv`, `task4_yesno_fit_models.csv` |
| 5 DAS datasets | `task5_das_datasets.md` | `data/passive_das/*.jsonl` |

## Run

- Prompts: `scripts/build_das_prep_prompts.py` -> `data/passive_das_prep/prompts.jsonl`
  (183,456 prompts, SHA-256 `2b3f8e9f…16a6`).
- Readout: `scripts/run_next_token_readout.py`, TSUBAME job `8829368` (`gpu_h`,
  commit `20e201e`, bf16, same tokenization as the band-cross scores) ->
  `results/passive_das_prep/pythia14b_readout.csv` (SHA-256 `3158ed4b…e142`,
  104 MB, not committed).
- Analysis: `scripts/das_prep_consistency.py` (task 0),
  `scripts/analyze_das_prep.py` (tasks 1-4),
  `scripts/build_das_passive_datasets.py` (task 5; needs `transformers`).
- 95% CIs: two-way cluster bootstrap over verb pairs (within verb band) and
  contexts (within context band), 1000 draws, seed 17, unless stated otherwise.

## Findings

**0. Consistency.** Before and after come from one spec (`analyze_bad_fit.py`:
WLS, shared good-fit slope, band dummies). A from-scratch recompute matches
the committed values exactly. The own-context advantage falls from 21.6
[16.0, 26.2] to 5.0 [0.3, 9.1] pp (`passive_1`) and from 20.5 [13.7, 25.9] to
7.1 [0.3, 13.1] pp (`passive_2`). The old `fit_band_own` model in
`analyze_fit_ratings.py` has the same design, seed and bootstrap order, so the
identical "after" values are expected, not stale. Adding bad-patient fit
lowers them to 4.6 [-0.3, 8.8] and 5.8 [-0.6, 11.8] pp; both intervals now
include 0. *by* reliability (released contexts, CI over 300 contexts):

| Band | Reliable + | n.s. + | n.s. - | Reliable - |
|---|---:|---:|---:|---:|
| Head | 21 | 0 | 0 | 5 |
| Tail | 40 | 0 | 1 | 9 |
| XTail | 35 | 3 | 0 | 12 |

Negative Head pairs: sum/compete, bet/appear, include/happen,
exhibit/proceed, load/respond. The draft counts (21/26, 35/50, 12) are
reproduced only on released contexts. On curated contexts they are 18/26 and
33/50, with 17 negative XTail pairs.

**1. Baseline.**
- Passives do not already favour an object after a transitive participle.
  Good minus bad log P(" the") is +0.12 [-0.55, 0.83] (Head), +0.41
  [0.06, 0.77] (Tail) and +0.41 [0.11, 0.71] (XTail). For " a" it is about 0.
  Total object-start mass does not differ in any band.
- Actives show the contrast clearly: " the" is +4.04 [3.56, 4.50] (Head)
  higher after the transitive verb. The gap is smaller for XTail (-0.88
  [-1.45, -0.28] vs Head), mainly because XTail *intransitives* put more mass
  on " the" (-4.93 vs -6.01).
- After a passive participle the mass goes to " by" and prepositions. " by"
  is favoured after good verbs by about 0.9 nats in Head and Tail and 0.49
  [0.11, 0.82] in XTail.

**2. *by* split.** Negative *by* pairs fall into two groups:
- **Good side low** (4/5 Head, 5/10 Tail, 7/12 XTail). The transitive
  participle wants its own particle or PP instead of *by*: summed **up**,
  shooed **away**, wadded **up**, larded **with**, daubed **with**, swaddled
  **in**, stowed **away**, blurted **out**. This is verb-specific selection,
  not a failure to recognise the passive.
- **Bad side high** (doff/salivate, debug/tingle, disgorge/exult,
  unlatch/burgeon, shoplift/eventuate, exhibit/proceed, ...). The
  intransitive participle gets *by* at or above the reliable-pair level: the
  model treats it as passivisable. These are the pairs that bear on
  argument-structure knowledge.

The two groups should be analysed separately in any DAS evaluation (the
per-verb table has the driver and z-scores).

**3. Single-prompt readout.** Within the good prompt, log-odds(" by" vs ".")
tracks the two-sentence *by* margin only moderately: Spearman 0.41
[0.33, 0.49] per item and 0.54 [0.39, 0.65] per verb pair. Using the
difference between the good and bad prompts raises this to 0.55 / 0.56. Raw
log P(" by" | good) alone correlates better (0.67 / 0.72), but it is one term
of the margin, so part of that is mechanical. log P(".") adds noise (0.17
per item). A single-prompt DAS target is therefore a partial proxy for the
*by* margin; log P(" by") is a cleaner target than the by-vs-"." ratio.

**4. Yes/No.** The gate fails. Head accuracy is 60.8 [50.5, 69.8]
(`passive_1`) and 57.6 [46.2, 68.0] (`passive_2`), versus 81-83% under LP.
The mean P(Yes) share is 0.581 for good sentences and 0.580 for bad ones, and
Yes/No margins do not correlate with LP margins (Spearman -0.03). The
all-band fit model was run for completeness only: good fit and bad-patient
fit have no effect under Yes/No (0.11 [-1.04, 1.27] pp per point vs 4.52
[3.49, 5.52] under LP). Pythia-1.4B does not use this prompt, so there is no
fit effect to test there.

**5. DAS datasets.**
- Train: Head actives, 20 train pairs / 6 held-out dev pairs x 90 subjects.
  The transitive verb prefers " the" over "." in 99% of items; the
  intransitive prefers "." over " the" in 84% (train) and 65% (dev).
- Passive eval: 11,962 reliable / 3,371 negative-*by* / 375 n.s.-positive
  items, with good-fit (p1, p2) and bad-patient fit per item.
- Tail and XTail active sets are included for later transfer.

## Flags

- **Nothing here contradicts** "noun frequency has no effect" or "participle
  deficit ~ -1.0 to -1.1 at matched fit". Neither was re-estimated. The LP
  refit in task 4 reproduces the committed `good+bad_patient` XTail
  estimates (accuracy -4.38 / -9.34 pp).
- **"." is a weak intransitive target.** Intransitive actives mostly continue
  with a preposition (" to", " in", " with"): log P(".") is about -4.3 for Head.
  Only 65% of Head-dev intransitive items prefer "." over " the". Options:
  filter to items that pass, or use " the" vs the preposition mass.
- **Token-count confound for transfer.** All Head verbs are single tokens,
  while 98% of Tail and 100% of XTail items have a multi-token verb. An
  intervention at the verb-final subtoken learned on Head will be tested
  under a different tokenization for Tail/XTail.
- **bet/appear tokenization artifact.** After "was bet" the model predicts
  word-pieces ("rot", "tered": *betrothed*, *bettered*). Its *by* margin
  (-2.68) is mostly this, not argument structure. Consider dropping the pair.
- **Own-context advantage under the primary control.** With bad-patient fit
  included, the residual own-context accuracy advantage is no longer
  significant (CI includes 0). The write-up quotes the good-only control.
