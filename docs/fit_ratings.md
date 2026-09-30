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

All estimates below use one shared fit slope across verb bands (per-band
slopes do not differ) and come from one two-way bootstrap:
`reports/fit_ratings/bad_fit_models.csv`, produced by
`scripts/analyze_bad_fit.py`. Bad-side fit is the rating of the intransitive
active with the patient as subject ("The hat salivated."); see
`reports/fit_ratings/bad_fit_report.md`.

1. **Fit predicts Pythia's preference.** Each good-fit rating point adds 0.55
   (`passive_1`) / 0.40 (`passive_2`) to the full-sentence margin and about
   4 pp accuracy.
2. **Fit explains most of the own-context advantage.** Pooled over bands, the
   own-context accuracy advantage falls from 21.6 [16.0, 26.2] to 5.0
   [0.3, 9.1] pp in `passive_1` and from 20.5 [13.7, 25.9] to 7.1 [0.3, 13.1]
   pp in `passive_2`. The remainder may be rating ceiling or collocational
   typicality (*hat* + *doff*) beyond plausibility.
3. **Part of the rare-verb deficit is a fit effect.** XTail - Head:

   | Measure | Fit control | `passive_1` | `passive_2` |
   |---|---|---:|---:|
   | Accuracy (pp) | None | -8.0 [-17.3, 1.7] | -12.9 [-22.8, -2.5] |
   | | Good | -3.4 [-12.2, 6.1] | -7.8 [-17.8, 2.1] |
   | | Good + bad patient | -4.4 [-12.9, 4.8] | -9.3 [-19.0, -0.2] |
   | Full-sentence margin | None | -0.76 [-1.64, 0.11] | -1.00 [-1.95, -0.18] |
   | | Good | -0.17 [-1.02, 0.70] | -0.46 [-1.38, 0.41] |
   | | Good + bad patient | -0.25 [-1.10, 0.63] | -0.60 [-1.51, 0.25] |
   | Participle margin | None | -1.39 [-2.07, -0.69] | -1.39 [-2.11, -0.74] |
   | | Good | -0.98 [-1.69, -0.34] | -0.87 [-1.53, -0.24] |
   | | Good + bad patient | -1.11 [-1.78, -0.46] | -1.00 [-1.66, -0.34] |

4. **A rare-verb deficit remains at the participle** under every fit
   control. Under the earlier per-band-slope spec it was similar with Qwen
   ratings (-1.04 / -1.08) and after controlling the good-minus-bad token
   count (-0.96 / -0.85). In the highest fit bin, XTail accuracy still trails:
   85.8 vs 90.9 (Head) in `passive_1` and 81.3 vs 90.3 in `passive_2`.
5. **The bad-side agent term is a confound.** In `passive_1`, agent fit
   "predicts" the participle and *by* margins, which precede the agent, and
   not the suffix that contains it. It mostly encodes whether the bad verb
   takes human subjects, so the primary control uses the patient only.

## Limits

Ratings come from one LM, validated against one human rater (132 good-side, 60 bad-side items). Own contexts sit at
the rating ceiling, so the scale cannot tell "plausible" from "tailored".
Fit bins pool contexts and have no intervals. The participle-token residual
compares single-token Head verbs with multi-token XTail verbs; good and bad
token counts are balanced within each band.
