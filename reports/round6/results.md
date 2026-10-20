# Round 6: results (paper 1: frequency and entrenchment)

Plan, predictions and decision rules: `plan.md`, each part committed before
its runs (part 0: `01f6f23`; A1–A2: `0b1d505`, code fixes after a second
review `d875bc7`, A2 run clarification in `02461c1`; B3, C4, C5: `02461c1`;
A6–A7: `37e842c`). An outside reviewer (Codex) reviewed every part's draft
and code before its runs, twice per part. Generated reports, one per
experiment:
- 0: `../results_index.md` (paper tags), `human_passives.md`;
- A1: `a1_exposure.md`; A2: `a2_suite_d11.md`;
- B3: `b3_alternations.md`; C4: `c4_classify.md`; C5: `c5_preemption.md`;
- A6: `a6_checkpoints.md`; A7: `a7_fit_controls.md`.

TSUBAME jobs: 9069457 (A2, seven models; 160M stopped at a check), 9069909
(A2, 160M rerun under the declared clarification), 9069908 (B3, C4, C5),
9070046 (A6 checkpoints and A7 released-context scores), 9070047 (A7 fit
ratings). Counts (A1) and training-stream slices (A6) were fetched locally
from the infini-gram API and the Hugging Face CDN.

Disclosures, also in `plan.md`: (1) the C4 classification of round-5 A4's
32 rare verbs was seen in a code smoke test before C4 was declared; (2)
mock analyses used to test the A1, A2 and A7 code mixed real round-5 scores
of the 433 D11 verbs that round 5 had already scored with random numbers
for the rest, and so printed partly real band contrasts before A2's
analysis; they were not used and the mocks were replaced.

## Summary

- **Part 0.** Round 5 was already in the results index; every row now
  carries a paper tag (P1 frequency / P2 mechanism / both) and the parked
  paper-2 items are listed. The FreqBLiMP human study is forced choice, so
  only accuracy by band is reported: annotators also find rare passives
  harder (`passive_1` correct among decided 94.6 → 85.7%, can't-decide 18
  → 38% from Head to XTail), on 15 items per cell (descriptive).
- **A2, the D11 frame contrast across the Pythia suite.** The rare-verb
  deficit replicates on the 281-pair inventory and survives token-count
  standardization.
  - At equal token counts, Tail pairs have a larger was–has contrast L
    than XTail pairs at every size from 160M to 12B (ΔL_tok +0.89 to
    +1.66 nats; 1.4B +0.99 [+0.43, +1.53]), so the deficit is not only a
    tokenization artifact of the Head − XTail comparison.
  - It is intransitive-side at every size: rare intransitive participles
    are rejected less after "was" (1.4B: −ΔΔ_pass(I) +1.40), while rare
    transitives are, if anything, accepted more (ΔΔ_pass(T) −0.41).
  - The held-out (DAS-unexposed) pairs point the same way at every size
    but are too few to resolve (1.4B three-cell +0.53 [−0.66, +1.63]).

## Part 0: housekeeping

- `../results_index.md`: a Paper column (P1 / P2 / P1+P2) on all 62 rows of
  rounds 1–5 and a note listing the parked paper-2 items (LEACE necessity,
  gap-neuron mediation, human annotation of B7, 6.9B DAS). Round-6 rows are
  added at the end of the round.
- Human data (`human_passives.md`): the study is forced choice (A / B /
  can't decide, then a naturalness rating of the good sentence only), so
  bad-item acceptance cannot be computed. Per the rule, accuracy by band,
  descriptive (15 pairs × 3 annotators per cell):

| paradigm | band | correct among decided (%) | can't decide (%) | 2-of-3 support | Pythia-1.4B LP, same items (%) |
|---|---|---|---|---:|---|
| `passive_1` | Original | 97.6 | 8.9 | 13 / 15 | — |
| | Head | 94.6 | 17.8 | 13 / 15 | 80.0 |
| | Tail | 88.2 | 24.4 | 10 / 15 | 66.7 |
| | XTail | 85.7 | 37.8 | 7 / 15 | 73.3 |
| `passive_2` | Original | 100.0 | 4.4 | 15 / 15 | — |
| | Head | 97.3 | 17.8 | 12 / 15 | 80.0 |
| | Tail | 83.8 | 17.8 | 11 / 15 | 80.0 |
| | XTail | 91.2 | 24.4 | 10 / 15 | 80.0 |

  The rarity decline in humans shows mostly as more "can't decide"
  responses (XTail `passive_1`: of 17 non-decisions, 14 give "both
  acceptable" and 7 "unknown words" as a reason; several reasons allowed). CIs are in the report; with 15 items per
  cell none of the band differences is resolved.

## A2. The D11 frame contrast across the Pythia suite

Spec: `plan.md`, A2; report `a2_suite_d11.md`. All checks passed (1.4B
reconstruction 3.5e-5, identity 4.1e-5; 160M under the declared 1e-2
tolerances: 1.6e-3, 2.9e-3, h deviation 9.2e-4); every verb that round 5
had scored reproduces round 5's log-probs (1.4B: exactly; 160M: 2.7e-3).

**Primary, token-standardized Tail − XTail (freq prefixes):**

| model | ΔL_tok | reading | ΔΔ_pass(T) / −ΔΔ_pass(I) | side | held-out three-cell | ΔL Head − XTail | Head AUC | singleton D_L |
|---|---|---|---|---|---|---|---|---|
| 160M | +0.97 [+0.40, +1.54] | deficit | −0.22 / +1.19 | intransitive | +1.12 [−0.39, +2.64] | +1.52 [+0.92, +2.16] | 0.94 | +1.28 [+0.51, +2.01] |
| 410M | +1.26 [+0.66, +1.86] | deficit | −0.22 / +1.47 | intransitive | +0.64 [−0.72, +2.02] | +1.65 [+0.97, +2.31] | 0.95 | +1.39 [+0.47, +2.31] |
| 1B | +1.66 [+1.08, +2.22] | deficit | +0.04 / +1.62 | intransitive | +1.40 [−0.00, +2.90] | +1.64 [+0.94, +2.34] | 0.94 | +1.51 [+0.45, +2.45] |
| **1.4B** | **+0.99 [+0.43, +1.53]** | **deficit** | −0.41 / +1.40 | intransitive | +0.53 [−0.66, +1.63] | +1.41 [+0.71, +2.15] | 0.93 | +1.56 [+0.58, +2.47] |
| 2.8B | +0.89 [+0.23, +1.52] | deficit | −0.41 / +1.30 | intransitive | +0.46 [−1.06, +1.98] | +1.20 [+0.50, +1.93] | 0.93 | +1.45 [+0.47, +2.39] |
| 6.9B | +1.04 [+0.47, +1.57] | deficit | −0.48 / +1.51 | intransitive | +1.05 [−0.08, +2.19] | +1.64 [+0.93, +2.33] | 0.94 | +1.40 [+0.44, +2.29] |
| 12B | +0.94 [+0.38, +1.48] | deficit | −0.48 / +1.42 | intransitive | +0.07 [−0.99, +1.10] | +1.39 [+0.68, +2.09] | 0.94 | +1.50 [+0.52, +2.43] |

Held-out readings are "unresolved" at every size. 1.4B details:
- Band means of L: Head 3.37, Tail 2.56, XTail 1.96; Δ_pass(I) −2.89 /
  −2.23 / −1.12; Δ_pass(T) +0.48 / +0.33 / +0.84. Within every token cell,
  Tail intransitives are rejected more after "was" than XTail ones (2×2:
  −2.27 vs −0.90).
- Sensitivities: OLS adjustment for token counts +0.82 [+0.28, +1.36]; the
  2×2 cell alone +0.67 [−0.02, +1.36]. Every prefix set reads a deficit,
  intransitive-side (been +1.35, cur +1.12, they +0.89).
- Head − XTail (token-confounded) +1.41 [+0.71, +2.15], smaller than on
  the curated 126 (+2.12). Head − Tail +0.81. ΔM_has (token-standardized)
  −0.56: a surplus, so the reading of the deficit as was–has-specific is
  unresolved, as in round 5.
- Single-token verbs: Head AUC 0.93; the singleton contrast D_L +1.56
  [+0.58, +2.47] comes from the intransitive rows (D_I −1.25; D_T +0.31).
- Three-cell contrast: all pairs +0.82 [+0.22, +1.38] and trainable pairs
  +0.97 [+0.31, +1.63] are deficits; held-out pairs +0.53 [−0.66, +1.63].

**Against the predictions.** 1.4B: deficit in the predicted range (+0.5 to
+2.0) and intransitive-side — **held**. Every size ≥ 410M: deficit and
intransitive-side — **held**; size within ±50% of 1.4B's — held except 1B
(+1.66, 68% above) — **mostly held**. 160M: deficit — held. Held-out at
1.4B: positive, deficit or unresolved — **held** (unresolved). Head AUC ≥
0.9 and singleton D_L above 0 — **held**.
