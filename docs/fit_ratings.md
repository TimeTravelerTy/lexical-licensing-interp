# Plausibility (fit) ratings for the curated passive cross

## Question

In the verb band x context band cross, noun frequency had no effect, but
each verb pair's *own* curated context lifted accuracy to ceiling (+17 to
+26 pp over other verbs' curated contexts). This step measures how well each
context fits each verb, and asks two things. How much of the own-context
advantage is plausibility? Does the rare-verb deficit survive at matched fit?

## Items and raters

`scripts/build_fit_rating_items.py` collects the 28,854 unique good
sentences of the curated cross, plus 117 role reversals of own `passive_1`
contexts ("The gentleman was doffed by the hat"). `scripts/rate_fit.py`
reads each rater's next-token distribution over the digits 1-7 after a chat
prompt and records the expected value. There is no sampling. Two prompt
paraphrases are averaged. Both prompts ask for event plausibility and tell
the rater not to penalize rare words.

| Rater | TSUBAME job | Node | Output SHA-256 |
|---|---|---|---|
| `google/gemma-4-31B-it` (primary) | 8812117 | `gpu_1` | `d55e6135…7fb0` |
| `Qwen/Qwen2.5-7B-Instruct` (second) | 8812118 | `gpu_h` | `8205b693…eff` |

Both ran at commit `3083477` from the HDD model cache
`/gs/bs/tga-sip_arase/tyrone/hf_cache`.

## Validation (Gemma-4-31B-it)

- Format: digit probability mass median 1.000. Prompt paraphrases agree
  (Spearman 0.956).
- Direction: own > mean of other contexts for 99.6% of verb pairs, and own >
  role reversal for 92.9%.
- Rare verbs: own-context ratings are flat across bands (6.9-7.0) and role
  reversals are flat (2.5-2.7). A rater that did not know XTail verbs would
  compress both toward the middle. The 7B rater does this (own 5.7 -> 5.2,
  XTail reversal 3.0), so it is only a secondary check.
- Other contexts rate lower for rarer verbs (4.1 / 3.1 / 2.9, also within
  the same context). With own and reversal flat, this most likely reflects
  the narrower selectional range of rare verbs (*suckle*, *stopper*,
  *shoplift*), not rater ignorance.
- Rater agreement with Qwen2.5-7B: Spearman 0.726.
- Human agreement (one author, 132 blind stratified items in
  `data/fit_ratings/human_sheet.csv`; four mis-keys corrected by the rater):
  Gemma Spearman 0.853 [0.801, 0.892]; Qwen-7B 0.697. Within other contexts
  only (n = 62), Gemma agrees at 0.698. Condition means (human vs Gemma):
  own 6.97 / 6.92, reversal 2.83 / 2.88, other 4.24 / 3.50. Gemma is harsher
  on unselected contexts but ranks them similarly. The human also rates
  Head verbs highest in other contexts (5.05 vs 3.62 Tail, 4.10 XTail;
  about 20 items each). This supports the selectional-range reading.

## Findings

Full tables: `reports/fit_ratings/report.md` (Gemma) and
`reports/fit_ratings/qwen25_7b_it/report.md` (Qwen, as a robustness check).

1. **Fit predicts Pythia's preference.** Each rating point adds 0.41-0.55 to
   the full-sentence margin and about 5 pp accuracy, with no fit x verb band
   interaction.
2. **Fit explains most of the own-context advantage.** After controlling
   fit, it falls from 18.6 to 5.0 pp in `passive_1` and from 17.2 to 7.1 pp
   in `passive_2`. With Qwen ratings it falls to about 1.5-1.8 pp (n.s.). The
   remainder may be rating ceiling or collocational typicality
   (*hat* + *doff*) beyond plausibility.
3. **Part of the rare-verb deficit is a fit effect.** Rare verbs fit
   arbitrary contexts worse. At matched fit, the XTail full-sentence effect
   is -0.32 [-1.19, 0.58] / -0.55 [-1.51, 0.34] and the accuracy effect is
   -4 / -8 pp. All CIs include 0.
4. **A rare-verb deficit remains at the participle.** The XTail
   participle-token margin at matched fit is -1.05 [-1.79, -0.39]
   (`passive_1`) and -0.95 [-1.63, -0.34] (`passive_2`). It is the same with
   Qwen ratings (-1.04 / -1.08) and after controlling the good-minus-bad
   token count (-0.96 / -0.85). In the highest fit bin, XTail accuracy still
   trails: 85.8 vs 90.9 (Head) in `passive_1` and 81.3 vs 90.3 in `passive_2`.

## Limits

Ratings come from one LM, validated against one human rater on 132 items. Own contexts sit at
the rating ceiling, so the scale cannot tell "plausible" from "tailored".
Fit bins pool contexts and have no intervals. The participle-token residual
compares single-token Head verbs with multi-token XTail verbs; good and bad
token counts are balanced within each band.
