# Gemma replication, step 2: behavioural check, Gemma-3-1B-pt vs Pythia-1.4B

Full tables (every measure × inventory, overall and by Gemma token count 1 / 2 / 3+):
`step2_tables.md`. CSVs: `../../results/gemma/behavior/{verb_scores,cells,gaps,freqblimp}.csv`, `meta.json`.
Script: `scripts/analyze_gemma_behavior.py` (definitions in its docstring). Step 1: `step1_refilter.md`.

## Setup

- **Verbs.** No token-count filter (step-1 decision): all D11 (281 pairs) and curated (126 pairs) participles,
  same verbs and bands for both models, full participle log-prob summed over subtokens. Gemma token counts:
  Head participles are all 1 token; XTail participles are 2 or 3+ tokens (except 2 one-token D11 T verbs). So
  Head − XTail is confounded with Gemma token count (fully for I; for T only 2 bin-1 XTail verbs, too few for a
  CI). Bin 2 is the only bin with a within-bin band contrast that has CIs (Tail − XTail), reported below.
- **BOS asymmetry.** Every Gemma prompt starts with `<bos>`. Pythia prompts have no BOS, as in every existing
  Pythia run, so the Pythia numbers here are the existing ones. Padding is right-sided in both.
- **Precision.** Gemma: bf16 weights, logits computed in fp32 from the final-norm output (bf16 logits of size ~30
  are quantized in 0.125–0.25-nat steps). Pythia: stored Δ_pass (fp32), stored curated *by* readout (bf16), new
  D11 *by* readout (fp32).
- **Measures.** Δ_pass(V) = log P(V | "The N was") − log P(V | "The N has"), averaged over the 24 freq nouns
  (existing definition). *by* gap = mean log P(" by") after T passives − after I passives. Curated: the task-1
  readout (curated `was` prefixes, other verbs' contexts); D11: after "The N was V", freq nouns (new for both
  models). 95% CIs: bootstrap over verbs within class × band (× token bin) cell, unpaired, 2,000 draws, seed 17.
  Cells with n < 5 get no CI.

## Checks

- Pythia through the modified scorers (default flags), on TSUBAME (job 9269789): 104 frame rows (26 verbs × 2
  nouns × was/has), 50 readout prompts, 25 FreqBLiMP pairs. Numerically identical (max |Δ| 0 on every scored
  column) to the pre-change scorers (commit 79ab4bf) for all three scorers (frame, next-token readout,
  FreqBLiMP). Equal to the stored scores exactly for the frame and FreqBLiMP scorers (max |Δ| 0). The stored
  bf16 readout comes from a different batch composition: mean |Δ lp(" by")| 0.03, mean signed +0.01, max 0.19
  (computed from the check outputs, `tmp_gemma_checks/out` on TSUBAME).
  The full new Pythia frame run (D11 *by* readout) also reproduces all 29,184 stored rows exactly.
- The curated *by*-gap point estimates reproduce task 1 (+0.94 / +0.96 / +0.49).
- Gemma batched vs one-at-a-time: 40 frame rows (10 verbs × 2 nouns × was/has), 20 readout prompts, 10
  FreqBLiMP pairs:
  - **fp32:** max |Δ| 4.7e-5 (frame), 4.0e-5 (readout), 3.5e-5 (FreqBLiMP). So padding and position indexing
    are correct.
  - **bf16:** max |Δ| 0.15 (frame), 0.20 (readout), 0.14 (FreqBLiMP); frame mean |Δ lp| 0.05, with no dependence
    on token count (from the check outputs). Numerical only.
  - Gemma *by* readout position = after the participle's last subtoken.
  - Gemma FreqBLiMP verb-token counts (offset spans) match the Gemma tokenization of " V" for all 6,000 good verbs
    (checked locally against the score file).

## Answers

### (i) T > I *by* gap: holds in every band, smaller than Pythia's; curated XTail CI includes 0

| T − I *by* gap | Head | Tail | XTail | all bands |
|---|---|---|---|---|
| Pythia, curated (task 1) | +0.94 [+0.25, +1.61] | +0.96 [+0.61, +1.33] | +0.49 [+0.11, +0.85] | +0.77 [+0.51, +1.04] |
| Gemma, curated (task 1) | +0.71 [+0.01, +1.35] | +0.47 [+0.13, +0.79] | +0.19 [−0.16, +0.50] | +0.41 [+0.17, +0.64] |
| Pythia, D11 (freq nouns) | +1.13 [+0.62, +1.61] | +1.09 [+0.85, +1.33] | +0.64 [+0.38, +0.91] | +0.93 [+0.76, +1.11] |
| Gemma, D11 (freq nouns) | +0.93 [+0.49, +1.42] | +0.66 [+0.45, +0.86] | +0.33 [+0.09, +0.57] | +0.59 [+0.42, +0.74] |

- The gap is positive in all 12 cells. Its CI is above 0 pooled in both inventories, and in 5 of 6 Gemma band
  cells.
- The exception is curated XTail. Curated Head's lower bound is +0.01.
- Gemma's pooled gap is 0.53× (curated) and 0.63× (D11) Pythia's.
- By Gemma token bin, the CI is above 0 in every D11 cell of bins 1–2. CIs that include 0:
  - D11 XTail, bin 3+: +0.32 [−0.10, +0.76]
  - curated Tail, bin 1: +0.34 [−0.08, +0.76]
  - curated XTail, bin 2: +0.28 [−0.13, +0.67]
  - curated XTail, bin 3+: −0.04 [−0.63, +0.54] (n_I = 10; the only negative band cell; the pooled curated bin-3+
    gap, which is the same verbs plus 2 Tail T verbs with no Tail I, is also negative, −0.08 [−0.68, +0.53])

### (ii) Δ_pass(I) weakens from Head to XTail: yes, in both inventories and more steeply than Pythia

| Δ_pass(I) | Head | Tail | XTail | Head − XTail | Tail − XTail, Gemma bin 2 only |
|---|---|---|---|---|---|
| Pythia, D11 | −2.89 | −2.23 | −1.12 | −1.77 [−2.31, −1.22] | −1.26 [−1.71, −0.78] |
| Gemma, D11 | −3.97 | −2.64 | −0.84 | −3.13 [−3.91, −2.35] | −1.16 [−1.78, −0.58] |
| Pythia, curated | −3.45 | −2.73 | −0.85 | −2.60 [−3.32, −1.89] | −2.28 [−3.00, −1.60] |
| Gemma, curated | −4.60 | −3.48 | −0.57 | −4.03 [−5.03, −3.06] | −2.28 [−3.25, −1.38] |

At matched Gemma token count (bin 2) the Tail → XTail weakening is about the same size in the two models
(D11 −1.16 vs −1.26; curated −2.28 vs −2.28).

### (iii) Transitive side roughly flat: yes; same shape as Pythia (XTail T slightly higher)

| Δ_pass(T) | Head | Tail | XTail | Head − XTail | Tail − XTail |
|---|---|---|---|---|---|
| Pythia, D11 | +0.48 | +0.33 | +0.84 | −0.36 [−0.83, +0.08] | −0.51 [−0.83, −0.19] |
| Gemma, D11 | +0.70 | +0.66 | +1.05 | −0.35 [−1.03, +0.31] | −0.39 [−0.88, +0.08] |
| Pythia, curated | +0.39 | +0.10 | +0.87 | −0.48 [−1.13, +0.08] | −0.77 [−1.31, −0.22] |
| Gemma, curated | +0.64 | +0.16 | +1.16 | −0.51 [−1.45, +0.38] | −1.00 [−1.76, −0.18] |

- Head − XTail on the T side includes 0 in both models and inventories. It is about 1/9 (Gemma, D11) to 1/8
  (Gemma, curated) of the I-side contrast.
- Curated Tail − XTail excludes 0 in both models, in the direction XTail T > Tail T.

### FreqBLiMP released passives (accuracy, whole-sentence LP)

| Model | Paradigm | Head | Tail | XTail |
|---|---|---|---|---|
| Pythia-1.4B | passive_1 | 0.787 [0.762, 0.813] | 0.795 [0.769, 0.820] | 0.700 [0.672, 0.726] |
| Pythia-1.4B | passive_2 | 0.816 [0.793, 0.839] | 0.799 [0.774, 0.823] | 0.727 [0.699, 0.755] |
| Gemma-3-1B-pt | passive_1 | 0.818 [0.793, 0.842] | 0.733 [0.704, 0.759] | 0.683 [0.654, 0.711] |
| Gemma-3-1B-pt | passive_2 | 0.808 [0.782, 0.833] | 0.787 [0.763, 0.812] | 0.701 [0.673, 0.730] |

Items resampled within paradigm × band; 1,000 items per cell. The good/bad first token is shared, so the
BOS-conditioned first-token log-prob cancels in the margin.

## Runs

TSUBAME `gpu_h`, branch `gemma-replication`, commit f046899. Check job 9269789; full job 9269848 (the fp32
batched-vs-single gate passed before the full runs). The logs are in `results/gemma/logs/`. Gemma-3-1B-pt resolved
commit fcf18a2a879aab110ca39f8bffbccd5d49d8eb29; transformers 4.57.6. Output sha256 (first 16 hex digits):
- Gemma frame `a1bf8c5e5fa06e1d`
- Pythia frame `6d849bc7952fb33d`
- Gemma readout `a241a85b7beb9448` (not committed, 15 MB, regenerable)
- Gemma FreqBLiMP `af1b6dbf606e7f6a`
