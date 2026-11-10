> **Pythia-1.4B, EXPLORATORY sensitivity split** (`pythia_same_ed_count_round3`; `../../exploratory.md`, section 9): the unchanged analysis rerun on the stored Pythia nonce runs, restricted to the split's lemmas. Not part of the stored Pythia results. Comparison: `../summary.md`.

# C8. Nonce verbs: context-sensitive separation along d in passives

Spec: `round3_plan.md`, C8. Run: `run_nonce_passive.py`; analysis: `analyze_nonce_passive.py`. 80 nonce lemmas × 4 slots; per-lemma means; 95% CIs: lemma bootstrap (2,000 draws, seed 17). z: 0 = held-out active intransitive, 1 = transitive level of each site's d (a fixed ruler).

## Passive probe ("The house was dakked")

| Site | matched T − I | mismatched1 | mismatched2 | balanced AB − BA | matched − mismatched | real good − bad (neutral ctx) | active probe: matched |
|---:|---|---|---|---|---|---|---|
| 4 | +0.04 [+0.04, +0.05] | +0.02 [+0.01, +0.02] | +0.02 [+0.01, +0.02] | +0.02 [+0.02, +0.02] | +0.03 [+0.03, +0.03] | +0.63 [+0.57, +0.68] | +0.06 [+0.05, +0.06] |
| 6 | +0.13 [+0.12, +0.13] | +0.03 [+0.03, +0.04] | +0.03 [+0.02, +0.04] | +0.12 [+0.12, +0.13] | +0.10 [+0.09, +0.10] | +0.48 [+0.44, +0.53] | +0.18 [+0.17, +0.18] |
| 8 | +0.15 [+0.14, +0.16] | +0.05 [+0.04, +0.05] | +0.04 [+0.04, +0.05] | +0.11 [+0.10, +0.11] | +0.11 [+0.10, +0.12] | +0.37 [+0.33, +0.41] | +0.24 [+0.23, +0.25] |
| 10 | +0.15 [+0.14, +0.16] | +0.03 [+0.02, +0.03] | +0.02 [+0.01, +0.03] | +0.14 [+0.13, +0.15] | +0.13 [+0.12, +0.14] | +0.40 [+0.36, +0.43] | +0.30 [+0.29, +0.31] |
| 12 | +0.27 [+0.26, +0.28] | +0.10 [+0.09, +0.11] | +0.09 [+0.08, +0.11] | +0.21 [+0.20, +0.22] | +0.17 [+0.16, +0.19] | +0.36 [+0.32, +0.39] | +0.49 [+0.48, +0.50] |
| 14 | +0.25 [+0.24, +0.26] | +0.08 [+0.07, +0.09] | +0.07 [+0.06, +0.09] | +0.20 [+0.19, +0.21] | +0.18 [+0.16, +0.19] | +0.28 [+0.25, +0.31] | +0.55 [+0.54, +0.57] |
| 16 | +0.26 [+0.25, +0.27] | +0.08 [+0.07, +0.10] | +0.07 [+0.06, +0.09] | +0.20 [+0.18, +0.21] | +0.18 [+0.17, +0.20] | +0.22 [+0.19, +0.25] | +0.63 [+0.61, +0.65] |
| 17 | +0.24 [+0.23, +0.26] | +0.08 [+0.07, +0.10] | +0.07 [+0.06, +0.09] | +0.18 [+0.16, +0.19] | +0.17 [+0.15, +0.18] | +0.18 [+0.15, +0.20] | +0.60 [+0.59, +0.62] |

Log-probabilities at the passive probe (nats):

| Contrast | log P(" by") | log P(PREP without by) | log P(O) | log P(".") |
|---|---|---|---|---|
| matched T - I | +1.46 [+1.37, +1.55] | +0.90 [+0.83, +0.97] | +2.14 [+2.00, +2.28] | -1.66 [-1.77, -1.55] |
| mismatched1 T - I | +0.65 [+0.56, +0.75] | +0.56 [+0.49, +0.62] | +1.05 [+0.91, +1.20] | -0.75 [-0.84, -0.68] |
| mismatched2 T - I | +0.60 [+0.51, +0.69] | +0.54 [+0.49, +0.59] | +1.02 [+0.89, +1.16] | -0.76 [-0.86, -0.67] |
| balanced AB - BA | +0.55 [+0.48, +0.62] | +0.20 [+0.15, +0.24] | +1.32 [+1.18, +1.48] | -0.71 [-0.80, -0.63] |
| matched - mismatched | +0.83 [+0.74, +0.93] | +0.35 [+0.28, +0.43] | +1.11 [+0.94, +1.28] | -0.90 [-1.02, -0.79] |
| real good − bad (neutral ctx) | +0.42 [+0.13, +0.70] | -0.06 [-0.23, +0.10] | -0.30 [-0.51, -0.10] | -0.19 [-0.33, -0.04] |

Causal check (d_s coordinate of the matched-T probe into the matched-I probe; Δ from unpatched):

| Site | base | Δ log P(" by") | Δ log P(PREP\by) | Δ log P(O) |
|---:|---|---|---|---|
| 6 | matched_I | +0.03 [+0.02, +0.03] | -0.01 [-0.02, -0.01] | +0.01 [+0.00, +0.02] |
| 6 | matched_T | +0.01 [+0.00, +0.01] | +0.02 [+0.01, +0.02] | -0.06 [-0.07, -0.06] |
| 8 | matched_I | +0.05 [+0.04, +0.05] | -0.01 [-0.02, -0.01] | +0.01 [+0.00, +0.02] |
| 8 | matched_T | +0.01 [-0.00, +0.02] | +0.02 [+0.01, +0.03] | -0.11 [-0.12, -0.10] |

## Declared decisions (passive probe)

- Site 6: context-sensitive separation along d: **yes** (Δ_matched +0.129); verb-specific: **yes** (Δ_balanced +0.122); raises *by* verb-specifically: **yes** (Δ_balanced by +0.55 vs PREP\by +0.20). Scale: Δ_matched / real-verb gap = 0.27; passive / active Δ_matched = 0.72.
- Site 8: context-sensitive separation along d: **yes** (Δ_matched +0.150); verb-specific: **yes** (Δ_balanced +0.107); raises *by* verb-specifically: **yes** (Δ_balanced by +0.55 vs PREP\by +0.20). Scale: Δ_matched / real-verb gap = 0.40; passive / active Δ_matched = 0.63.
