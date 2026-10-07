# Site-8 mechanism: decomposition and dose-response

Spec and predictions: `followup_plan.md` (steps 1–2), committed before the run. Run: `run_site8_mechanism.py` (fp32, LN scale frozen at the patched run); analysis: `analyze_site8_mechanism.py`. 278631 decomposed patches.

## Step 1: signed direct-logit decomposition (T-donor patch)

Centered logits through the final LayerNorm, with the LN scale frozen at the **patched** run (fp32). Terms: the carry (the site-8 displacement itself), 256 heads and 16 MLPs in layers 8–23. Pairs: passive 64 (primary bad passive bases), active 29 (held-out intransitive actives). Mean over pairs; 95% CI over pairs.

| Frame | Readout | P (positive terms) | N (negative terms) | Net | LN-scale term | Residual | Actual Δ |
|---|---|---|---|---|---|---|---|
| active | " by" | +1.43 [+1.31, +1.60] | -3.27 [-3.64, -2.94] | -1.83 [-2.16, -1.51] | +0.04 [-0.13, +0.23] | -0.000 [-0.000, +0.000] | -1.79 [-2.08, -1.51] |
| active | "." | +1.74 [+1.54, +1.99] | -4.34 [-4.71, -4.02] | -2.60 [-2.89, -2.31] | +0.03 [-0.16, +0.23] | +0.000 [-0.000, +0.000] | -2.57 [-2.87, -2.30] |
| active | " the" | +2.68 [+2.39, +3.04] | -1.35 [-1.54, -1.25] | +1.33 [+1.03, +1.59] | +0.02 [-0.16, +0.21] | +0.000 [-0.000, +0.000] | +1.35 [+1.09, +1.59] |
| active | " him" | +4.12 [+3.73, +4.57] | -1.33 [-1.55, -1.15] | +2.79 [+2.46, +3.13] | +0.01 [-0.11, +0.15] | -0.000 [-0.000, +0.000] | +2.80 [+2.46, +3.15] |
| active | Ō (object starts) | +2.19 [+2.02, +2.45] | -0.80 [-0.94, -0.71] | +1.39 [+1.16, +1.61] | +0.02 [-0.11, +0.17] | -0.000 [-0.000, +0.000] | +1.41 [+1.22, +1.60] |
| passive | " by" | +1.87 [+1.74, +2.01] | -1.42 [-1.51, -1.33] | +0.45 [+0.34, +0.55] | +0.05 [-0.02, +0.10] | -0.000 [-0.000, +0.000] | +0.50 [+0.42, +0.58] |
| passive | "." | +1.13 [+1.05, +1.21] | -1.35 [-1.42, -1.28] | -0.22 [-0.29, -0.15] | +0.05 [-0.01, +0.10] | -0.000 [-0.000, +0.000] | -0.17 [-0.22, -0.13] |
| passive | " the" | +1.28 [+1.22, +1.37] | -0.92 [-0.99, -0.87] | +0.37 [+0.30, +0.43] | +0.04 [-0.01, +0.08] | +0.000 [+0.000, +0.000] | +0.40 [+0.36, +0.45] |
| passive | " him" | +1.72 [+1.63, +1.83] | -0.57 [-0.62, -0.53] | +1.15 [+1.06, +1.25] | +0.02 [-0.01, +0.05] | -0.000 [-0.000, +0.000] | +1.17 [+1.07, +1.28] |
| passive | Ō (object starts) | +1.00 [+0.95, +1.06] | -0.50 [-0.53, -0.47] | +0.50 [+0.46, +0.55] | +0.03 [-0.01, +0.05] | +0.000 [-0.000, +0.000] | +0.53 [+0.49, +0.57] |

**Decision rule on Ō:** survival s = P_passive / P_active = +0.46 [+0.40, +0.50]; cancellation c = |N|/P: active 0.37, passive 0.50 (c_p − c_a = +0.13 [+0.06, +0.18]). **Outcome: story 1: nothing downstream pushes toward objects in passives.**

### Object-relevant components (|mean| ≥ 0.05 in either frame)

| Readout | Relevant | Sign change | Vanishes | Appears | Same sign, kept |
|---|---:|---:|---:|---:|---:|
| Ō (object starts) | 12 | 0 | 6 | 0 | 6 |
| " the" | 19 | 5 | 5 | 2 | 7 |
| " him" | 19 | 1 | 7 | 1 | 10 |

Flagged and top components on Ō (sorted by |active|; full list in `decomp_components_site8.csv`):

| Component | Active | Passive | Flag |
|---|---:|---:|---|
| carry | +0.493 | +0.426 |  |
| MLP18 | +0.451 | +0.061 | vanishes |
| MLP23 | -0.366 | +0.032 | vanishes |
| MLP21 | +0.276 | -0.036 | vanishes |
| MLP15 | +0.256 | -0.016 | vanishes |
| MLP22 | +0.184 | -0.021 | vanishes |
| L19H1 | +0.079 | +0.011 | vanishes |
| MLP11 | +0.052 | +0.044 |  |
| MLP20 | -0.051 | -0.075 |  |
| MLP8 | +0.047 | +0.055 |  |
| MLP16 | -0.041 | +0.052 |  |
| MLP12 | -0.041 | -0.113 |  |

Top components on " by" (by max |mean| over frames):

| Component | Active | Passive |
|---|---:|---:|
| MLP23 | -0.031 | -0.492 |
| MLP16 | -0.470 | +0.150 |
| MLP14 | -0.451 | +0.300 |
| MLP22 | +0.419 | -0.061 |
| MLP17 | -0.396 | +0.147 |
| MLP11 | -0.249 | +0.173 |
| MLP12 | -0.246 | -0.044 |
| MLP18 | -0.231 | -0.072 |
| MLP8 | +0.154 | +0.212 |
| MLP19 | -0.211 | -0.137 |
| MLP15 | +0.194 | -0.011 |
| MLP20 | -0.192 | -0.186 |
| MLP10 | +0.062 | +0.190 |
| MLP13 | -0.137 | +0.157 |
| L23H0 | +0.152 | +0.108 |

Net contribution by layer (heads summed, MLP separate), T donors, mean over pairs:

| Layer | Heads → Ō active | passive | MLP → Ō active | passive | Heads → " by" active | passive | MLP → " by" active | passive |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | -0.009 | -0.017 | +0.047 | +0.055 | -0.025 | -0.023 | +0.154 | +0.212 |
| 9 | -0.004 | +0.008 | -0.009 | +0.028 | -0.061 | -0.000 | -0.064 | +0.061 |
| 10 | +0.015 | +0.011 | -0.005 | -0.027 | +0.010 | +0.013 | +0.062 | +0.190 |
| 11 | -0.009 | -0.008 | +0.052 | +0.044 | +0.002 | +0.010 | -0.249 | +0.173 |
| 12 | +0.033 | +0.029 | -0.041 | -0.113 | +0.046 | +0.045 | -0.246 | -0.044 |
| 13 | +0.001 | -0.010 | +0.019 | +0.032 | +0.010 | -0.000 | -0.137 | +0.157 |
| 14 | +0.047 | +0.022 | +0.002 | +0.049 | -0.071 | -0.013 | -0.451 | +0.300 |
| 15 | -0.021 | -0.006 | +0.256 | -0.016 | -0.053 | -0.017 | +0.194 | -0.011 |
| 16 | -0.008 | -0.005 | -0.041 | +0.052 | -0.005 | -0.001 | -0.470 | +0.150 |
| 17 | +0.001 | -0.010 | -0.010 | +0.034 | +0.017 | -0.015 | -0.396 | +0.147 |
| 18 | -0.014 | -0.003 | +0.451 | +0.061 | +0.004 | +0.018 | -0.231 | -0.072 |
| 19 | +0.081 | -0.003 | +0.006 | -0.048 | +0.046 | +0.010 | -0.211 | -0.137 |
| 20 | +0.010 | +0.006 | -0.051 | -0.075 | +0.015 | +0.022 | -0.192 | -0.186 |
| 21 | +0.012 | -0.005 | +0.276 | -0.036 | +0.013 | +0.002 | +0.021 | -0.105 |
| 22 | +0.009 | +0.010 | +0.184 | -0.021 | +0.008 | +0.029 | +0.419 | -0.061 |
| 23 | -0.017 | +0.007 | -0.366 | +0.032 | +0.058 | +0.104 | -0.031 | -0.492 |
| carry | +0.493 (Ō) | +0.426 (Ō) | | | -0.017 (by) | -0.014 (by) | | |

### Downstream d-coordinates under the site-8 patch

z units of each site's basis for the same split and fold (0 = held-out active intransitive, 1 = transitive).

| Frame | Donor | z8 before → after | z12 before → after | z17 before → after | Δz12/Δz8 | Δz17/Δz8 |
|---|---|---|---|---|---|---|
| active | T | -0.00 → 1.00 | -0.00 → 0.88 | -0.00 → 0.74 | +0.88 [+0.86, +0.90] | +0.74 [+0.70, +0.78] |
| active | I | -0.00 → 0.00 | -0.00 → 0.00 | -0.00 → 0.00 | — | — |
| passive | T | 0.09 → 1.00 | 0.16 → 0.71 | 0.01 → 0.23 | +0.60 [+0.59, +0.61] | +0.24 [+0.23, +0.25] |
| passive | I | 0.09 → 0.00 | 0.16 → 0.11 | 0.01 → -0.00 | +0.62 [+0.60, +0.64] | +0.21 [+0.19, +0.22] |

Passive/active ratio of Δz17/Δz8 (T donors): 0.32 (predicted ≤ 0.4); passive Δz17 = +0.22 [+0.20, +0.23] (predicted ≈ 0.1).

fp32 check (passive, T donors, Δ log P, mean over pairs): O +0.367 (bf16 main test +0.355), by +0.650 (bf16 main test +0.649), dot -0.021 (bf16 main test -0.013).

### Supplement: D decomposition (T − I donors), P / N / net

| Frame | Readout | P | N | Net |
|---|---|---:|---:|---:|
| active | " by" | +1.44 | -3.24 | -1.79 |
| active | "." | +1.77 | -4.32 | -2.55 |
| active | " the" | +2.67 | -1.35 | +1.32 |
| active | " him" | +4.04 | -1.32 | +2.72 |
| active | Ō (object starts) | +2.18 | -0.79 | +1.39 |
| passive | " by" | +2.13 | -1.57 | +0.56 |
| passive | "." | +1.28 | -1.46 | -0.18 |
| passive | " the" | +1.42 | -1.01 | +0.41 |
| passive | " him" | +1.88 | -0.64 | +1.24 |
| passive | Ō (object starts) | +1.11 | -0.56 | +0.55 |

## Step 2: dose-response at site 8

The site-8 coordinate is set to z (split-0 basis, sign-aligned; 0 = active intransitive level, 1 = active transitive level). Curves: mean over pairs; 95% CIs over pairs and contexts (subjects for actives). R_p = [0.10, 0.45] (passive range), R_a = [0.45, 1.50].

| Frame | S_by(R_p) | S_O(R_p) | S_O(R_a) | S_by(R_a) | S_O(R_p)/S_O(R_a) | R² by on R_p | Outcome |
|---|---|---|---|---|---|---:|---|
| bad_passive | +0.92 [+0.81, +1.04] | +0.27 [+0.20, +0.33] | +0.61 [+0.54, +0.69] | +0.48 [+0.42, +0.54] | +0.43 [+0.34, +0.51] | 0.996 | intermediate |
| good_passive | +0.33 [+0.27, +0.40] | +0.50 [+0.41, +0.58] | +0.89 [+0.79, +0.99] | +0.21 [+0.16, +0.27] | +0.56 [+0.49, +0.63] | 0.995 | (reference) |
| intrans_active | -0.03 [-0.31, +0.23] | +3.14 [+2.55, +3.71] | +1.39 [+1.09, +1.70] | -1.12 [-1.42, -0.84] | +2.26 [+1.68, +3.06] | 0.306 | (reference) |
| trans_active | -1.46 [-1.77, -1.17] | +1.89 [+1.46, +2.41] | +0.26 [+0.20, +0.32] | -0.84 [-1.11, -0.62] | +7.32 [+5.64, +9.57] | 0.999 | (reference) |

Declared outcome applies to bad passives. Prediction: coupled; by ≈ +0.8/z and O ≈ +0.4/z; R² ≥ 0.95.

Selected points (bad passives; log P, mean over pairs):

| z | log P(" by") | log P(O) | log P(".") | log P(pron) | z12 | z17 |
|---:|---:|---:|---:|---:|---:|---:|
| -0.50 | -3.88 | -4.56 | -3.79 | -8.48 | -0.21 | -0.11 |
| -0.00 | -3.23 | -4.51 | -3.71 | -8.38 | 0.11 | 0.00 |
| 0.10 | -3.11 | -4.50 | -3.70 | -8.34 | 0.18 | 0.03 |
| 0.25 | -2.95 | -4.46 | -3.69 | -8.26 | 0.27 | 0.06 |
| 0.45 | -2.79 | -4.40 | -3.69 | -8.10 | 0.40 | 0.11 |
| 0.60 | -2.69 | -4.34 | -3.70 | -7.96 | 0.49 | 0.14 |
| 0.90 | -2.54 | -4.19 | -3.72 | -7.63 | 0.66 | 0.21 |
| 1.20 | -2.40 | -4.00 | -3.72 | -7.21 | 0.83 | 0.29 |
| 1.50 | -2.28 | -3.76 | -3.72 | -6.74 | 0.99 | 0.36 |
| 2.00 | -2.11 | -3.21 | -3.72 | -5.82 | 1.26 | 0.49 |

