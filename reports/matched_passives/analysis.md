# Matched-context passive pilot: interpretation

Run on 2026-09-23 with `EleutherAI/pythia-1.4b` on TSUBAME job `8762274`
(exit status 0). Code revision: `822e12382503d340f1d6480c4f738b3b01217190`.
Input SHA-256: `4632fc440347d5c6a9fb42253ed5e7760d6f4e10770bfdad3f9132e10094f15e`.
The score file contains exactly 608 distinct sentence pairs.

## Design checks

Each paradigm contains 38 unique good lemmas and 38 unique bad lemmas. Each
lemma occurs in one verb pair within that paradigm; the same 38 pairings are
reused across `passive_1` and `passive_2` intentionally. Each pairing appears
in eight identical human-patient frames. The frequencies below are absolute
good/bad gaps in English `wordfreq` Zipf units.

| Band | Verb pairs | Mean / max lemma gap | Mean / max participle gap |
| --- | ---: | ---: | ---: |
| Head | 11 | 0.115 / 0.25 | 0.193 / 0.34 |
| Tail | 15 | 0.102 / 0.24 | 0.079 / 0.18 |
| Xtail | 12 | 0.101 / 0.25 | 0.085 / 0.27 |

Good verbs were curated for plausible human patients. The shared contexts are
grammatical and broadly plausible, but some combinations remain semantically
unusual, especially for rare verbs. There was no independent human
acceptability check. Frequency matching also cannot match meaning or
tokenization: the mean good/bad participle token counts are 1.0/1.0 in head,
2.13/2.33 in tail, and 3.0/2.42 in xtail.

## Results

Accuracy is the proportion of sentences with positive full-sentence log
probability margin, averaged over verb pairs. The 95% intervals resample verb
pairs, not sentences. The [full report](report.md) lists **every verb pair's
average margin and accuracy separately for both paradigms**.

| Paradigm | Band | Accuracy | Mean margin | 95% margin interval | Pairs with positive average margin |
| --- | --- | ---: | ---: | ---: | ---: |
| `passive_1` | Head | 0.841 | 3.795 | [2.064, 5.672] | 10/11 |
| `passive_1` | Tail | 0.992 | 5.970 | [4.704, 7.310] | 15/15 |
| `passive_1` | Xtail | 0.875 | 4.640 | [2.868, 6.499] | 11/12 |
| `passive_2` | Head | 0.989 | 3.443 | [2.667, 4.232] | 11/11 |
| `passive_2` | Tail | 0.867 | 4.148 | [2.589, 5.621] | 13/15 |
| `passive_2` | Xtail | 0.865 | 4.269 | [2.721, 5.786] | 11/12 |

The head-minus-xtail mean-margin differences are **−0.845** for `passive_1`
(95% pair bootstrap interval [−3.375, 1.685]) and **−0.826** for `passive_2`
([−2.523, 0.873]). Removing one head or xtail verb pair at a time leaves
both margin differences negative. The full ranges are [−1.456, −0.322] and
[−1.276, −0.383]. Neither interval establishes a margin decline with rarity.

Accuracy tells a mixed story: head-minus-xtail is −0.034 for `passive_1`
(interval [−0.212, 0.152]) and +0.124 for `passive_2`
([0.008, 0.280]). One pair deletion reverses the `passive_1` accuracy
difference; no single deletion reverses the `passive_2` accuracy difference.
The `passive_2` accuracy drop deserves follow-up, but its mean margin does not
decline, and `passive_1` does not replicate it.

Most of the full-sentence margin comes at the participle. In `passive_1`, the
`by` token also favors the good verb by about 1.46–1.64 log-probability units
in all three bands. The suffix results therefore show some construction cue,
while the full-sentence comparison remains sensitive to lexical probability
and differing token counts.

## Conclusion

Pythia usually prefers the passivizable verb even in the xtail band of this
matched-context pilot. The pattern is **not a consistent passive-frequency
weakness across both paradigms and both readouts**. The small curated sample,
semantic variation, and tokenization differences make a flat or mixed result
inconclusive about frequency effects generally. A broader inventory and an
additional readout would be appropriate before using this specific passive
contrast as the target for a mechanistic intervention.

Files: `summary.csv` (regime statistics), `per_verb_pair.csv` (lexical unit),
`by_frame.csv` (context sensitivity), `head_xtail_comparison.csv` and
`leave_one_out.csv` (robustness), and the raw scores under
`results/matched_passives/pythia14b_scores.csv`.
