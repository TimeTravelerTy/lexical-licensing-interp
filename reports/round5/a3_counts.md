# Round 5, A3: training-data predictor (Pile counts)

Spec: `plan.md`, A3. Counts: infini-gram `v4_piletrain_llama` (`fetch_pile_counts.py`); analysis: `analyze_pile_counts.py`. Outcome: Δ_pass(V) (A1, freq frames). Decision: 10-fold CV error (folds grouped by pair) of ~ log c + log c_be vs ~ log c; verbs within band × contexts bootstrap.

Audit: 0 verbs fail the token-suffix check and are dropped: none.

**Hand audit of "was V" contexts** (up to 10 per verb, evenly spaced occurrences; labels by hand):

| class | band | verb | "was V" count | eventive passive | other |
|---|---|---|---:|---:|---:|
| I | head | apologized | 160 | 8 | 2 |
| I | head | happened | 3,530 | 0 | 10 |
| I | tail | flirted | 123 | 10 | 0 |
| I | tail | glared | 138 | 9 | 1 |
| T | head | rated | 154,759 | 10 | 0 |
| T | head | sent | 2,315,297 | 7 | 0 |
| T | tail | commemorated | 9,789 | 10 | 0 |
| T | tail | ingested | 7,998 | 10 | 0 |
| T | xtail | resupplied | 588 | 10 | 0 |

Shares: I: eventive passive 0.68, other 0.33; T: eventive passive 1.00, other 0.00

## curated verbs

| class | n | r(Zipf) | r(log c) | r(log c_be) | partial r(ρ_be \| log c) | in-sample R²: log c / log c_be / both / Zipf | CV MSE: ~log c | ~log c_be | ~log c + log c_be | difference (CI) | ~Zipf | decision |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| T | 126 | -0.14 [-0.31, +0.02] | -0.14 [-0.31, +0.01] | -0.06 [-0.22, +0.10] | +0.16 [+0.01, +0.31] | 0.02/0.00/0.05/0.02 | 2.070 | 2.095 | 2.026 | +0.045 [-0.091, +0.198] | 2.077 | **unresolved** |
| T | sensitivity | smoothing 0.1 / 1: unresolved / unresolved; capitalized auxiliaries: unresolved; was + been only: unresolved | | | | | | | | | | | |
| I | 126 | -0.58 [-0.68, -0.45] | -0.50 [-0.61, -0.36] | -0.26 [-0.40, -0.10] | +0.19 [+0.00, +0.37] | 0.25/0.07/0.28/0.34 | 2.649 | 3.350 | 2.606 | +0.043 [-0.212, +0.315] | 2.353 | **unresolved** |
| I | sensitivity | smoothing 0.1 / 1: unresolved / unresolved; capitalized auxiliaries: unresolved; was + been only: unresolved | | | | | | | | | | | |

Share with c = 0: I head 0.00, tail 0.00, xtail 0.00; T head 0.00, tail 0.00, xtail 0.00
Share with c_be = 0: I head 0.00, tail 0.00, xtail 0.00; T head 0.00, tail 0.00, xtail 0.00
Share with c_perf = 0: I head 0.00, tail 0.00, xtail 0.00; T head 0.00, tail 0.00, xtail 0.00

## pool verbs

| class | n | r(Zipf) | r(log c) | r(log c_be) | partial r(ρ_be \| log c) | in-sample R²: log c / log c_be / both / Zipf | CV MSE: ~log c | ~log c_be | ~log c + log c_be | difference (CI) | ~Zipf | decision |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| T | 627 | -0.01 [-0.09, +0.08] | -0.05 [-0.13, +0.04] | +0.14 [+0.05, +0.22] | +0.45 [+0.37, +0.52] | 0.00/0.02/0.20/0.00 | 2.049 | 2.016 | 1.636 | +0.413 [+0.247, +0.607] | 2.053 | **slot frequency adds** |
| T | sensitivity | smoothing 0.1 / 1: slot frequency adds / slot frequency adds; capitalized auxiliaries: not computed; was + been only: not computed | | | | | | | | | | | |
| I | 83 | -0.31 [-0.51, -0.09] | -0.32 [-0.51, -0.10] | +0.00 [-0.25, +0.23] | +0.42 [+0.18, +0.59] | 0.10/0.00/0.26/0.09 | 2.450 | 2.799 | 2.110 | +0.341 [-0.087, +0.997] | 2.450 | **unresolved** |
| I | sensitivity | smoothing 0.1 / 1: unresolved / unresolved; capitalized auxiliaries: not computed; was + been only: not computed | | | | | | | | | | | |

Share with c = 0: I head 0.00, tail 0.00, xtail 0.00; T head 0.00, tail 0.00, xtail 0.00
Share with c_be = 0: I head 0.00, tail 0.00, xtail 0.00; T head 0.00, tail 0.00, xtail 0.00
Share with c_perf = 0: I head 0.00, tail 0.00, xtail 0.00; T head 0.00, tail 0.00, xtail 0.00

## Pairs: L ~ T and I predictors (curated)

| smoothing / variant | CV MSE ~log c_T + log c_I | ~log c_be,T + log c_be,I | ~both | difference (CI) | ~Zipf_T + Zipf_I | decision |
|---|---|---|---|---|---|---|
| s=0.5 | 4.676 | 4.398 | 4.495 | +0.180 [-0.433, +0.764] | 4.403 | **unresolved** |
| s=0.1 | 4.676 | 4.399 | 4.496 | +0.180 [-0.432, +0.764] | 4.403 | unresolved |
| s=1.0 | 4.676 | 4.398 | 4.495 | +0.181 [-0.434, +0.763] | 4.403 | unresolved |
| capitalized | 4.676 | 4.397 | 4.494 | +0.181 [-0.432, +0.764] | 4.403 | unresolved |
| was + been | 4.676 | 3.929 | 4.078 | +0.598 [-0.290, +1.518] | 4.403 | unresolved |

Descriptive: CV MSE with wordfreq Zipf vs Pile log c is in the class tables (columns ~Zipf, ~log c).

