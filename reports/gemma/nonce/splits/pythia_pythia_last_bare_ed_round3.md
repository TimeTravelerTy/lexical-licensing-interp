> **Pythia-1.4B, EXPLORATORY sensitivity split** (`pythia_pythia_last_bare_ed_round3`; `../../exploratory.md`, section 9): the unchanged analysis rerun on the stored Pythia nonce runs, restricted to the split's lemmas. Not part of the stored Pythia results. Comparison: `../summary.md`.

# C8. Nonce verbs: context-sensitive separation along d in passives

Spec: `round3_plan.md`, C8. Run: `run_nonce_passive.py`; analysis: `analyze_nonce_passive.py`. 80 nonce lemmas × 4 slots; per-lemma means; 95% CIs: lemma bootstrap (2,000 draws, seed 17). z: 0 = held-out active intransitive, 1 = transitive level of each site's d (a fixed ruler).

## Passive probe ("The house was dakked")

| Site | matched T − I | mismatched1 | mismatched2 | balanced AB − BA | matched − mismatched | real good − bad (neutral ctx) | active probe: matched |
|---:|---|---|---|---|---|---|---|
| 4 | +0.04 [+0.04, +0.05] | +0.02 [+0.02, +0.03] | +0.03 [+0.02, +0.03] | +0.02 [+0.02, +0.03] | +0.02 [+0.02, +0.02] | +0.63 [+0.57, +0.68] | +0.06 [+0.06, +0.07] |
| 6 | +0.14 [+0.13, +0.15] | +0.04 [+0.03, +0.05] | +0.05 [+0.04, +0.06] | +0.13 [+0.12, +0.13] | +0.10 [+0.08, +0.11] | +0.48 [+0.44, +0.53] | +0.19 [+0.18, +0.19] |
| 8 | +0.14 [+0.12, +0.15] | +0.04 [+0.03, +0.05] | +0.05 [+0.03, +0.07] | +0.10 [+0.09, +0.11] | +0.09 [+0.08, +0.11] | +0.37 [+0.33, +0.41] | +0.24 [+0.23, +0.26] |
| 10 | +0.15 [+0.12, +0.17] | +0.02 [+0.01, +0.04] | +0.04 [+0.02, +0.06] | +0.15 [+0.13, +0.16] | +0.11 [+0.10, +0.13] | +0.40 [+0.36, +0.43] | +0.31 [+0.29, +0.33] |
| 12 | +0.26 [+0.24, +0.28] | +0.09 [+0.08, +0.11] | +0.12 [+0.09, +0.15] | +0.21 [+0.19, +0.23] | +0.15 [+0.13, +0.18] | +0.36 [+0.32, +0.39] | +0.50 [+0.47, +0.52] |
| 14 | +0.24 [+0.22, +0.26] | +0.07 [+0.05, +0.09] | +0.10 [+0.06, +0.13] | +0.20 [+0.18, +0.23] | +0.16 [+0.13, +0.18] | +0.28 [+0.25, +0.31] | +0.56 [+0.53, +0.58] |
| 16 | +0.25 [+0.22, +0.27] | +0.07 [+0.05, +0.09] | +0.10 [+0.06, +0.13] | +0.20 [+0.18, +0.23] | +0.16 [+0.13, +0.19] | +0.22 [+0.19, +0.25] | +0.63 [+0.60, +0.66] |
| 17 | +0.23 [+0.21, +0.26] | +0.07 [+0.05, +0.09] | +0.10 [+0.07, +0.13] | +0.18 [+0.16, +0.20] | +0.15 [+0.12, +0.18] | +0.18 [+0.15, +0.20] | +0.61 [+0.58, +0.64] |

Log-probabilities at the passive probe (nats):

| Contrast | log P(" by") | log P(PREP without by) | log P(O) | log P(".") |
|---|---|---|---|---|
| matched T - I | +1.42 [+1.22, +1.63] | +0.88 [+0.76, +1.01] | +2.02 [+1.82, +2.24] | -1.71 [-1.91, -1.53] |
| mismatched1 T - I | +0.66 [+0.48, +0.87] | +0.56 [+0.45, +0.68] | +0.96 [+0.78, +1.14] | -0.80 [-0.92, -0.68] |
| mismatched2 T - I | +0.67 [+0.49, +0.88] | +0.59 [+0.48, +0.72] | +1.16 [+0.88, +1.43] | -0.90 [-1.09, -0.73] |
| balanced AB - BA | +0.55 [+0.43, +0.67] | +0.14 [+0.09, +0.19] | +1.27 [+1.07, +1.48] | -0.69 [-0.83, -0.57] |
| matched - mismatched | +0.75 [+0.58, +0.93] | +0.30 [+0.18, +0.42] | +0.96 [+0.71, +1.21] | -0.86 [-1.02, -0.71] |
| real good − bad (neutral ctx) | +0.42 [+0.14, +0.68] | -0.06 [-0.22, +0.09] | -0.30 [-0.50, -0.08] | -0.19 [-0.34, -0.04] |

Causal check (d_s coordinate of the matched-T probe into the matched-I probe; Δ from unpatched):

| Site | base | Δ log P(" by") | Δ log P(PREP\by) | Δ log P(O) |
|---:|---|---|---|---|
| 6 | matched_I | +0.02 [+0.01, +0.04] | -0.01 [-0.02, -0.01] | +0.02 [+0.01, +0.03] |
| 6 | matched_T | +0.01 [+0.00, +0.03] | +0.02 [+0.01, +0.03] | -0.07 [-0.08, -0.06] |
| 8 | matched_I | +0.05 [+0.03, +0.06] | -0.01 [-0.02, -0.01] | +0.01 [+0.00, +0.03] |
| 8 | matched_T | +0.01 [+0.00, +0.03] | +0.02 [+0.01, +0.03] | -0.10 [-0.12, -0.08] |

## Declared decisions (passive probe)

- Site 6: context-sensitive separation along d: **yes** (Δ_matched +0.138); verb-specific: **yes** (Δ_balanced +0.126); raises *by* verb-specifically: **yes** (Δ_balanced by +0.55 vs PREP\by +0.14). Scale: Δ_matched / real-verb gap = 0.29; passive / active Δ_matched = 0.74.
- Site 8: context-sensitive separation along d: **yes** (Δ_matched +0.137); verb-specific: **yes** (Δ_balanced +0.100); raises *by* verb-specifically: **yes** (Δ_balanced by +0.55 vs PREP\by +0.14). Scale: Δ_matched / real-verb gap = 0.37; passive / active Δ_matched = 0.56.
