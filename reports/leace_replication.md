# LEACE necessity and cross-frame LEACE in three models (results, Nov 23)

Rules: `reports/leace_replication_rules.md` (declared before running). TSUBAME jobs 9532387–9532405 (chains), 9534460 (N1–N3, `scripts/analyze_leace_replication.py`). Per-model reports: `reports/{gemma,olmo}/leace/` (`leace_necessity.md`, `leace_permutation.md`, `leace_crossframe.md`, `leace_pcf_recentered.md`, `replication_N.json`). The generated reports' titles and boilerplate say "Pythia" in places; the numbers are each model's.

**All runs' checks pass** (fit diagnostics, identity hook, outside-anchor, offline = in-model, scrub replay; the permutation run's true labels reproduce the in-sample run to 1.8e-15; the cross-frame references match by prompts, folds and natural readout, max |Δ| 0). The N1–N3 script reproduces Pythia's Settled 1 numbers exactly.

## Necessity (primary range: half the depth)

| | Pythia (scrub 0–12) | Gemma (scrub 0–13) | OLMo (scrub 0–8) |
|---|---|---|---|
| natural *by* gap (nats) | 0.77 [0.51, 1.02] | 0.41 [0.17, 0.64] | 0.60 [0.38, 0.83] |
| cross-fitted share remaining | 0.59 [0.35, 0.70] | 0.24 [−0.73, 0.51] | 0.56 [0.31, 0.70] |
| **removed by erasers fit on other verbs** | **0.41 [0.30, 0.65]** | **0.76 [0.49, 1.73]** | **0.44 [0.30, 0.69]** |
| in-sample share remaining | 0.04 [−0.46, 0.28] | −0.26 [−1.77, 0.18] | 0.07 [−0.48, 0.33] |
| in-sample − cross-fitted (paired) | −0.55 [−0.83, −0.41] | −0.49 [−1.14, −0.30] | −0.49 [−0.80, −0.34] |
| permutation 5th percentile (20 perms) | 0.97 | 0.93 | 0.95 |
| N1 / N2 / N3 | yes / yes / yes | yes / yes / yes | yes / yes / yes |

**Declared reading: "verb-general + additional in-sample removal" replicates in Gemma and OLMo.** OLMo's numbers are close to Pythia's. Gemma's cross-fitted erasers remove more (0.76), but its natural gap is about half Pythia's, so its shares have wide CIs (the upper bound 1.73 means erasure can overshoot and reverse the gap in some draws).

## Cross-frame (erasers fit on actives, applied to passives; matched centering; primary range)

| | Pythia | Gemma | OLMo |
|---|---|---|---|
| A-cf removes (share of gap) | 0.34 [0.24, 0.53] | 0.60 (share remaining 0.40 [−0.37, 0.62]) | 0.39 (0.61 [0.37, 0.73]) |
| ρ(cf), matched centering | 0.82 [0.74, 0.91] | 0.82 [0.72, 0.93] | 0.94 [0.85, 1.04] |
| ρ(in) | 0.87 [0.84, 0.90] | 0.86 [0.82, 0.90] | 0.91 [0.88, 0.94] |
| reading (both pairings, variants c and u) | majority transfer | majority transfer | majority transfer |
| shift-only (translation) | removes 0.03 [0.01, 0.05] | leaves 1.03 [1.01, 1.08] (cf) / 1.11 (in) | leaves 1.03 [1.02, 1.05] / 1.06 |

**Declared reading: majority transfer in all three models.** Every reading is below the permutation minimum. The translation flag fires in Gemma and OLMo too (shift-only CI excludes 0), but in the opposite direction from Pythia: the constant shift alone slightly *widens* the gap, so the active-fit erasers' removal is not carried by the shift.

## What this changes

- The necessity row of the status table moves from "not run" to "done" for Gemma and OLMo; "verb-general + additional in-sample removal" is now a three-model claim, and so is the cross-frame result.
- The abstract's "one number" was Pythia-only (0.41). With item 5 (expanded verbs, Pythia), the Pythia number rises to 0.53; With the same expansion, Gemma rises to 0.83 and OLMo to 0.57 (`reports/expanded_eval_rules.md`, Result for Gemma and OLMo).
