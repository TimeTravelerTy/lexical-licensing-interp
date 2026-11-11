# OLMo replication, behavioural check: OLMo-2-1B vs Pythia-1.4B and Gemma-3-1B-pt

Rules (written before the runs): `rules.md`. Full tables, split by Gemma token bin and by OLMo token bin:
`behavior_tables.md`. CSVs: `../../results/olmo/behavior/`. Script: `scripts/analyze_olmo_behavior.py`.

## Setup

- **Model:** `allenai/OLMo-2-0425-1B`, `main` = a1847dff.
- **Precision:** bf16 weights, logits in fp32 from the final-norm output.
- **Verbs:** the same verbs, bands and definitions as Gemma step 2. All D11 (281 pairs) and curated (126 pairs)
  participles, no token-count filter.
- **BOS:**
  - `olmo-2-1b`: no BOS (primary).
  - `olmo-2-1b+eos`: `<|endoftext|>` prepended (sensitivity).
- **Pythia and Gemma rows:** the stored step-2 scores. Their CIs reproduce the stored step-2 tables to 9e-16
  (CSV round trip).
- **CIs:** 95% bootstrap over verbs within cell, 2,000 draws, seed 17.
- **OLMo token counts of " V"** (608 participles):

  | band | 1 token | 2 tokens | 3+ tokens |
  |---|---:|---:|---:|
  | Head | 111 | 1 | 0 |
  | Tail | 21 | 234 | 28 |
  | XTail | 0 | 115 | 98 |

  OLMo's 100k vocabulary splits more Tail participles than Gemma's 262k does. Head − XTail is confounded with
  token count, as for Gemma.

## Checks

- **Frame inputs** (`build_gemma_frame_inputs.py --model allenai/OLMo-2-0425-1B`):
  - 29,184 joined-tokenization checks passed ("prefix V" = prefix ids + participle ids).
  - `pairs.csv` is byte-identical to Gemma's.
- **Gate, OLMo fp32 batched vs one-at-a-time** (TSUBAME job 9352847; tolerance 1e-3; passed for both BOS
  settings):

  | items | no BOS max \|Δ\| | `<|endoftext|>` max \|Δ\| |
  |---|---:|---:|
  | 40 frame rows (Gemma's 10 check verbs × 2 nouns × was/has) | 1.4e-5 | 2.5e-5 |
  | 20 readout prompts | 3.0e-5 | 3.0e-5 |
  | 10 FreqBLiMP pairs | 3.3e-5 | 2.7e-5 |

- **bf16 batched vs one-at-a-time** (reported, not gated): max |Δ| 0.06–0.12. This is bf16 numerical noise,
  the same size as Gemma's 0.14–0.20.
- **Full frame runs:**
  - h deviation across batch rows 0.
  - First-subtoken reconstruction error ≤ 3.1e-5.
  - Max |logit| 29.8 (no BOS) and 31.3 (`<|endoftext|>`).

## Answers

### *by* gap: holds under the declared rule, in both BOS settings

The rule: the pooled T − I *by* gap has a CI above 0 in both inventories.

| T − I *by* gap | Head | Tail | XTail | all bands |
|---|---|---|---|---|
| Pythia, curated | +0.94 [+0.25, +1.61] | +0.96 [+0.61, +1.33] | +0.49 [+0.11, +0.85] | +0.77 [+0.51, +1.04] |
| Gemma, curated | +0.71 [+0.01, +1.35] | +0.47 [+0.13, +0.79] | +0.19 [−0.16, +0.50] | +0.41 [+0.17, +0.64] |
| **OLMo, curated** | +0.71 [+0.08, +1.31] | +0.62 [+0.27, +0.96] | +0.53 [+0.19, +0.88] | **+0.60 [+0.37, +0.83]** |
| OLMo +eos, curated | +0.68 [+0.15, +1.27] | +0.57 [+0.25, +0.89] | +0.37 [+0.09, +0.65] | +0.51 [+0.30, +0.72] |
| Pythia, D11 | +1.13 [+0.62, +1.61] | +1.09 [+0.85, +1.33] | +0.64 [+0.38, +0.91] | +0.93 [+0.76, +1.11] |
| Gemma, D11 | +0.93 [+0.49, +1.42] | +0.66 [+0.45, +0.86] | +0.33 [+0.09, +0.57] | +0.59 [+0.42, +0.74] |
| **OLMo, D11** | +0.98 [+0.53, +1.42] | +0.82 [+0.59, +1.05] | +0.55 [+0.31, +0.78] | **+0.75 [+0.59, +0.91]** |
| OLMo +eos, D11 | +0.92 [+0.51, +1.34] | +0.73 [+0.52, +0.94] | +0.41 [+0.20, +0.63] | +0.64 [+0.50, +0.79] |

- **Size:** OLMo sits between Gemma and Pythia. Its pooled gap is 0.78× (curated) and 0.80× (D11) Pythia's.
- **Band cells:** all 12 OLMo band cells have CIs above 0, both settings. That includes curated XTail, where
  Gemma's CI includes 0.
- **Shape:** the gap shrinks from Head to XTail, as in the other two models.
- **BOS sensitivity:** `<|endoftext|>` lowers the gap by about 0.1 nats everywhere. The verdict is the same, so
  the rule's stop condition does not apply.
- **By token bin:** CIs include 0 only in curated 3+-token cells (n_I = 10 by Gemma bin, 20 by OLMo bin). Point
  estimates there are +0.15 to +0.38, all positive. Pythia and Gemma's curated 3+ cells include 0 too. Every
  D11 bin excludes 0 in the primary run.

### Δ_pass(I) weakens from Head to XTail; less steep than Gemma, close to Pythia

| Δ_pass(I) | Head | Tail | XTail | Head − XTail | Tail − XTail, Gemma bin 2 |
|---|---|---|---|---|---|
| Pythia, D11 | −2.89 | −2.23 | −1.12 | −1.77 [−2.31, −1.22] | −1.26 [−1.71, −0.78] |
| Gemma, D11 | −3.97 | −2.64 | −0.84 | −3.13 [−3.91, −2.35] | −1.16 [−1.78, −0.58] |
| OLMo, D11 | −2.44 | −1.84 | −0.87 | −1.57 [−2.08, −1.03] | −1.05 [−1.49, −0.62] |
| OLMo +eos, D11 | −2.97 | −2.06 | −1.00 | −1.98 [−2.49, −1.44] | −1.15 [−1.61, −0.68] |
| Pythia, curated | −3.45 | −2.73 | −0.85 | −2.60 [−3.32, −1.89] | −2.28 [−3.00, −1.60] |
| Gemma, curated | −4.60 | −3.48 | −0.57 | −4.03 [−5.03, −3.06] | −2.28 [−3.25, −1.38] |
| OLMo, curated | −2.76 | −2.28 | −0.68 | −2.08 [−2.85, −1.32] | −1.77 [−2.40, −1.14] |
| OLMo +eos, curated | −3.27 | −2.52 | −0.84 | −2.43 [−3.15, −1.71] | −1.88 [−2.52, −1.25] |

At matched Gemma token count (bin 2), the Tail → XTail weakening has the same sign and similar size in all
three models.

### Transitive side roughly flat (XTail T slightly higher), as in Pythia and Gemma

| Δ_pass(T) | Head | Tail | XTail | Head − XTail | Tail − XTail |
|---|---|---|---|---|---|
| OLMo, D11 | +0.82 | +0.62 | +1.07 | −0.26 [−0.79, +0.26] | −0.45 [−0.80, −0.08] |
| OLMo +eos, D11 | +0.35 | +0.36 | +0.89 | −0.54 [−1.05, −0.01] | −0.53 [−0.90, −0.15] |
| OLMo, curated | +0.74 | +0.46 | +1.00 | −0.25 [−0.96, +0.41] | −0.54 [−1.13, +0.03] |
| OLMo +eos, curated | +0.22 | +0.13 | +0.81 | −0.59 [−1.34, +0.08] | −0.68 [−1.25, −0.14] |

- Pythia and Gemma rows: `behavior_tables.md`.
- The T-side contrasts are a fraction of the I-side ones. With no BOS, Head − XTail includes 0 in both
  inventories, as in Pythia and Gemma. With `<|endoftext|>`, the D11 upper bound is −0.01.

### FreqBLiMP released passives (accuracy, whole-sentence LP)

| Model | Paradigm | Head | Tail | XTail |
|---|---|---|---|---|
| Pythia-1.4B | passive_1 | 0.787 [0.762, 0.813] | 0.795 [0.769, 0.820] | 0.700 [0.672, 0.726] |
| Pythia-1.4B | passive_2 | 0.816 [0.793, 0.839] | 0.799 [0.774, 0.823] | 0.727 [0.699, 0.755] |
| Gemma-3-1B-pt | passive_1 | 0.818 [0.793, 0.842] | 0.733 [0.704, 0.759] | 0.683 [0.654, 0.711] |
| Gemma-3-1B-pt | passive_2 | 0.808 [0.782, 0.833] | 0.787 [0.763, 0.812] | 0.701 [0.673, 0.730] |
| OLMo-2-1B | passive_1 | 0.805 [0.780, 0.830] | 0.762 [0.735, 0.787] | 0.682 [0.653, 0.712] |
| OLMo-2-1B | passive_2 | 0.825 [0.802, 0.848] | 0.790 [0.763, 0.815] | 0.745 [0.718, 0.772] |
| OLMo-2-1B +eos | passive_1 | 0.790 [0.766, 0.815] | 0.746 [0.719, 0.773] | 0.679 [0.650, 0.705] |
| OLMo-2-1B +eos | passive_2 | 0.822 [0.798, 0.845] | 0.764 [0.739, 0.790] | 0.716 [0.688, 0.743] |

## Runs

TSUBAME `gpu_h`, branch `olmo-replication`, commit 5809066, job 9352847. Log: `../../results/olmo/logs/`.
transformers 4.57.6.

Output sha256 (first 16 hex digits):

| output | no BOS | `<|endoftext|>` |
|---|---|---|
| frame | `725a82f73b6cc613` | `a29a6954d1a74319` |
| readout (not committed, 15 MB each, regenerable) | `8195b9101b2cbdfb` | `bbd8cd16f239602b` |
| FreqBLiMP | `8b46e550fd85910c` | `e15b5e07bd753175` |
