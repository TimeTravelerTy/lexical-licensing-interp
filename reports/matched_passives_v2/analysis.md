# Diverse-context passive pilot: Pythia 1.4B

Run on 2026-09-23 on TSUBAME, job `8763294` (`gpu_h=1`, exit status 0),
using `EleutherAI/pythia-1.4b` in bfloat16. Code revision:
`99a0f8dc32bb987286b141e3d5bd076b0203abe2`. Input SHA-256:
`72f0fa0116506b1d888fce989dde5b76fc5bccc40bd0664db286c5cba4ac6916`.
The score file has 504 distinct sentence pairs. The locally saved copy has
Unix line endings and SHA-256
`427d51921b43a50eb6491ceba5b9f7975bb1f50f79b1bca5596c1da163c44e55`.

## Design and checks

The released FreqBLiMP `passive_1` and `passive_2` files supplied the good
participle inventory and the tail/xtail bad inventory. The reviewed head bad
inventory came from the `codex/passive-head-coverage` branch of
`freqblimp-generation` (commit `50eabb4`). Each band uses one-to-one good/bad
verb matches; the 26 head, 50 tail, and 50 xtail lexical pairs each appear in
both paradigms and in two auxiliary templates (`was`, `had been`). The
surrounding words are identical within every good/bad sentence pair. Nouns
vary across lexical pairs so the grammatical sentence is plausible. The
[input README](../../data/matched_passives_v2/README.md) describes the
inventory and caveats.

The bands follow the Zipf frequency of the *realised participle*, as in the
released FreqBLiMP files. The matching minimizes total lemma-plus-participle
frequency gaps within 0.25 and 0.35 Zipf calipers, respectively. Each verb
appears in only one pair per paradigm. The mean signed good-minus-bad lemma
gaps are +0.006, -0.010, and -0.001 Zipf for head, tail, and xtail; the form
gaps are -0.019, approximately zero, and +0.003. Both lemmas fall inside the
named band's window for 25, 21, and 15 pairs, respectively.

The model scores each complete sentence without demonstrations or
interventions. A margin is `log P(good sentence) - log P(bad sentence)`;
positive values favor the passivizable verb. Accuracy is the proportion of
positive margins. Each verb pair's two templates are averaged before regime
statistics or bootstrap intervals are computed.

## Main results

| Paradigm | Band | Verb pairs | Accuracy | Mean margin | 95% pair-bootstrap interval |
| --- | --- | ---: | ---: | ---: | ---: |
| `passive_1` (with *by*) | Head | 26 | 1.000 | 9.330 | [8.285, 10.353] |
| `passive_1` | Tail | 50 | 0.990 | 9.969 | [8.815, 11.073] |
| `passive_1` | Xtail | 50 | 0.990 | 7.178 | [6.234, 8.120] |
| `passive_2` (without *by*) | Head | 26 | 1.000 | 6.890 | [5.720, 8.063] |
| `passive_2` | Tail | 50 | 0.990 | 7.888 | [7.030, 8.715] |
| `passive_2` | Xtail | 50 | 0.960 | 5.142 | [4.365, 5.908] |

The head-minus-xtail mean-margin difference is +2.152 for `passive_1`
(95% pair-bootstrap interval [0.809, 3.526]) and +1.748 for `passive_2`
([0.323, 3.164]). Removing any one head or xtail verb pair leaves these
differences positive: ranges [1.955, 2.417] and [1.509, 1.986]. The tail
mean exceeds the head mean in both paradigms, so the three bands do **not**
form a monotonic frequency gradient. Accuracy has little room to fall:
26/26 head pairs have positive average margins in each paradigm; the
corresponding counts are 49/50 and 50/50 in tail, and 50/50 and 48/50 in
xtail. The [full report](report.md) shows **each verb pair's average margin
and accuracy, separately for each paradigm**.

## Sensitivity and where the margin changes

Restricting to pairs where *both lemmas* also fall in the named band leaves
25 head, 21 tail, and 15 xtail pairs. In this subset the head-minus-xtail
margin difference shrinks to +1.088 for `passive_1` (95% pair-bootstrap
interval [-0.925, 3.165]) and +0.595 for `passive_2` ([-1.057, 2.348]).
This subset is small and no longer establishes a reliable head–xtail margin
difference. It should not be treated as an independent replication because
it is selected from the same pairs.

The head-minus-xtail full-sentence difference is concentrated at the
participle: the pair-mean verb contribution differs by +2.333 log units
(interval [1.169, 3.548]), while the total post-verb suffix contribution
differs by -0.181 in `passive_1` and -0.585 in `passive_2` (both intervals
include zero). In `passive_1`, the immediate *by* token favors the good verb
by 1.539, 1.084, and 0.407 units in head, tail, and xtail. Its head–xtail
difference is +1.132 (interval [0.196, 2.089]); later suffix tokens offset
this local decline. This is suggestive of weaker construction-specific
continuation after rare verbs, but the whole-sentence effect is mostly a
participle likelihood effect.

Both auxiliary frames give the same ordering of regime means. In head all
26 pairs are positive in both frames and paradigms. The reviewed head bad
pool has 16 simple intransitives and 10 verbs with clausal or prepositional
active frames; all head pairs are positive, including the simple-intransitive
subset. See `by_frame.csv`, `head_active_classes.csv`, and
`frequency_sensitivity.csv` for the detailed checks.

## Interpretation

This wider pilot finds a **robust head–xtail margin decline in the
form-defined primary set**, but only a small accuracy decline and no
monotonic head-to-tail-to-xtail pattern. The strict lemma-band subset is
inconclusive. Pythia still reliably prefers the grammatical sentence in
nearly every pair, so this test has not reproduced a strong passive
*accuracy* failure. Compared with the earlier 38-pair shared-human-context
pilot, the larger and more varied inventory changes the margin pattern;
the designs are not directly interchangeable.

Frequency matching balances good and bad verbs *within* pairs but does not
equate meaning, tokenization, or context *across* bands. Mean good/bad
participle token counts are 1.00/1.00 in head, 2.06/1.98 in tail, and
2.54/2.46 in xtail. The model's response therefore supports a
verb-frequency **association under these controlled lexical contrasts**,
not a causal effect of frequency or a clean grammaticality signal. Some
low-frequency verbs have uncommon alternate transitive senses; selected
patients were manually chosen to avoid those readings, without an
independent human acceptability study.

For the thesis gate, the passive contrast is usable as a **margin-sensitive
behavioral task**, especially the immediate *by* continuation, but the
current full-sentence accuracy readout is near ceiling. A mechanistic
intervention should be interpreted against the pair-level natural margins
and checked for an effect at the passive continuation, rather than claiming
that these results alone establish a broad rare-verb passive deficit.

Files: `report.md` and `per_verb_pair.csv` (every lexical pair),
`summary.csv` (regime means), `head_xtail_comparison.csv` and
`leave_one_out.csv` (robustness), `frequency_sensitivity.csv` and
`head_active_classes.csv` and `frequency_sensitivity_comparison.csv`
(subsets), and
`../../results/matched_passives_v2/pythia14b_scores.csv` (raw scores).
