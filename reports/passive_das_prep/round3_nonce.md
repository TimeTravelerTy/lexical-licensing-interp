# C8. Nonce verbs: context-sensitive separation along d in passives

Spec: `round3_plan.md`, C8. Run: `run_nonce_passive.py`; analysis: `analyze_nonce_passive.py`. 80 nonce lemmas × 4 slots; per-lemma means; 95% CIs: lemma bootstrap (2,000 draws, seed 17). z: 0 = held-out active intransitive, 1 = transitive level of each site's d (a fixed ruler).

## Passive probe ("The house was dakked")

| Site | matched T − I | mismatched1 | mismatched2 | balanced AB − BA | matched − mismatched | real good − bad (neutral ctx) | active probe: matched |
|---:|---|---|---|---|---|---|---|
| 4 | +0.05 [+0.04, +0.05] | +0.02 [+0.01, +0.02] | +0.02 [+0.01, +0.02] | +0.02 [+0.02, +0.03] | +0.03 [+0.03, +0.03] | +0.63 [+0.57, +0.68] | +0.06 [+0.06, +0.06] |
| 6 | +0.13 [+0.12, +0.13] | +0.03 [+0.03, +0.04] | +0.03 [+0.03, +0.04] | +0.12 [+0.12, +0.13] | +0.10 [+0.09, +0.10] | +0.48 [+0.44, +0.52] | +0.18 [+0.17, +0.18] |
| 8 | +0.15 [+0.14, +0.16] | +0.05 [+0.04, +0.05] | +0.05 [+0.04, +0.05] | +0.11 [+0.10, +0.11] | +0.11 [+0.10, +0.11] | +0.37 [+0.34, +0.41] | +0.24 [+0.23, +0.25] |
| 10 | +0.16 [+0.15, +0.17] | +0.03 [+0.02, +0.03] | +0.03 [+0.02, +0.04] | +0.14 [+0.14, +0.15] | +0.13 [+0.12, +0.14] | +0.40 [+0.36, +0.43] | +0.29 [+0.28, +0.31] |
| 12 | +0.27 [+0.26, +0.28] | +0.10 [+0.09, +0.11] | +0.10 [+0.09, +0.12] | +0.21 [+0.20, +0.22] | +0.17 [+0.16, +0.19] | +0.36 [+0.32, +0.39] | +0.49 [+0.48, +0.50] |
| 14 | +0.25 [+0.24, +0.26] | +0.08 [+0.07, +0.09] | +0.08 [+0.06, +0.10] | +0.20 [+0.19, +0.21] | +0.17 [+0.16, +0.19] | +0.28 [+0.25, +0.31] | +0.55 [+0.53, +0.56] |
| 16 | +0.26 [+0.25, +0.27] | +0.08 [+0.07, +0.09] | +0.08 [+0.06, +0.09] | +0.20 [+0.19, +0.21] | +0.18 [+0.17, +0.20] | +0.22 [+0.19, +0.25] | +0.62 [+0.61, +0.64] |
| 17 | +0.24 [+0.23, +0.26] | +0.08 [+0.06, +0.09] | +0.08 [+0.06, +0.09] | +0.18 [+0.16, +0.19] | +0.16 [+0.15, +0.18] | +0.18 [+0.15, +0.20] | +0.60 [+0.58, +0.62] |

Log-probabilities at the passive probe (nats):

| Contrast | log P(" by") | log P(PREP without by) | log P(O) | log P(".") |
|---|---|---|---|---|
| matched T - I | +1.44 [+1.35, +1.53] | +0.88 [+0.82, +0.94] | +2.14 [+2.01, +2.27] | -1.66 [-1.75, -1.57] |
| mismatched1 T - I | +0.66 [+0.57, +0.75] | +0.55 [+0.49, +0.61] | +1.00 [+0.87, +1.13] | -0.77 [-0.84, -0.69] |
| mismatched2 T - I | +0.64 [+0.54, +0.73] | +0.55 [+0.50, +0.61] | +1.06 [+0.93, +1.19] | -0.80 [-0.89, -0.71] |
| balanced AB - BA | +0.56 [+0.49, +0.62] | +0.18 [+0.14, +0.22] | +1.34 [+1.21, +1.46] | -0.71 [-0.79, -0.63] |
| matched - mismatched | +0.80 [+0.71, +0.89] | +0.33 [+0.26, +0.39] | +1.10 [+0.95, +1.26] | -0.88 [-0.98, -0.78] |
| real good − bad (neutral ctx) | +0.42 [+0.14, +0.69] | -0.06 [-0.22, +0.09] | -0.30 [-0.52, -0.09] | -0.19 [-0.34, -0.05] |

Causal check (d_s coordinate of the matched-T probe into the matched-I probe; Δ from unpatched):

| Site | base | Δ log P(" by") | Δ log P(PREP\by) | Δ log P(O) |
|---:|---|---|---|---|
| 6 | matched_I | +0.03 [+0.02, +0.04] | -0.01 [-0.02, -0.01] | +0.01 [+0.01, +0.02] |
| 6 | matched_T | +0.01 [-0.00, +0.01] | +0.02 [+0.01, +0.02] | -0.06 [-0.07, -0.06] |
| 8 | matched_I | +0.05 [+0.04, +0.06] | -0.01 [-0.02, -0.01] | +0.01 [+0.00, +0.02] |
| 8 | matched_T | +0.01 [-0.00, +0.01] | +0.02 [+0.01, +0.03] | -0.11 [-0.12, -0.10] |

## Declared decisions (passive probe)

- Site 6: context-sensitive separation along d: **yes** (Δ_matched +0.129); verb-specific: **yes** (Δ_balanced +0.122); raises *by* verb-specifically: **yes** (Δ_balanced by +0.56 vs PREP\by +0.18). Scale: Δ_matched / real-verb gap = 0.27; passive / active Δ_matched = 0.72.
- Site 8: context-sensitive separation along d: **yes** (Δ_matched +0.151); verb-specific: **yes** (Δ_balanced +0.107); raises *by* verb-specifically: **yes** (Δ_balanced by +0.56 vs PREP\by +0.18). Scale: Δ_matched / real-verb gap = 0.41; passive / active Δ_matched = 0.63.
