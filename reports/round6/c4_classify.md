# Round 6, C4: classify-from-z restore

Spec: `plan.md`, C4. Run: `run_restore.py --classify-manifest` (A4's setup; conditions unedited, oracle, classify; LM loss included); analysis: `analyze_classify.py`. The classify condition moves each rare single-token participle's row to the Head mean affinity of the class read from its site-8 z (T if z ≥ θ; θ maximizes balanced accuracy over A4's Head single-token verbs).

Checks: {"affinity_reproduces_max": 8.076507294063617e-07, "z_reproduces_max": 2.220446049250313e-16, "edit_target_err_oracle": 5.963399661440549e-07, "unedited_rows_max_oracle": 0.0, "edit_target_err_classify": 5.963399661440549e-07, "unedited_rows_max_classify": 0.0, "unedited_reproduces_fp32_max": 0.0004418492317199152}.

## Classification by z

- θ = 0.514 (Head: 282 T, 30 I); Head balanced accuracy in sample 0.988, leave-one-out 0.971.
- **A4's rare singletons** (23 T, 9 I): accuracy T +1.00 [+1.00, +1.00], I +0.89 [+0.67, +1.00]; balanced accuracy +0.944 [+0.833, +1.000] → **above chance** (θ refit on resampled Head verbs in every draw).
- D11 Tail singletons (24 T, 20 I; secondary): accuracy T +1.00 [+1.00, +1.00], I +1.00 [+0.90, +1.00]; balanced accuracy +1.000 [+0.950, +1.000].
- Sensitivity θ' = 0.514 (Head of A4 + 17 D11 Head singletons): balanced accuracy rare 0.944, D11 Tail 1.000.

z by verb (rare singletons; assigned class):

| verb | class | z | assigned |
|---|---|---|---|
| grinned | I | +0.02 | I |
| crept | I | +0.07 | I |
| chuckled | I | +0.10 | I |
| glared | I | +0.17 | I |
| branched | I | +0.30 | I |
| adhered | I | +0.31 | I |
| corresponded | I | +0.32 | I |
| inquired | I | +0.50 | I |
| mutated | I | +0.56 | T ✗ |
| exerted | T | +0.65 | T |
| clenched | T | +0.67 | T |
| doped | T | +0.68 | T |
| gripped | T | +0.70 | T |
| uttered | T | +0.74 | T |
| signaled | T | +0.75 | T |
| diced | T | +0.75 | T |
| perturbed | T | +0.80 | T |
| inhibited | T | +0.87 | T |
| disseminated | T | +0.89 | T |
| approximated | T | +0.90 | T |
| elicited | T | +0.94 | T |
| subtracted | T | +0.97 | T |
| precluded | T | +0.98 | T |
| deprecated | T | +1.01 | T |
| formatted | T | +1.03 | T |
| abbreviated | T | +1.03 | T |
| purified | T | +1.06 | T |
| accorded | T | +1.07 | T |
| superimposed | T | +1.08 | T |
| excised | T | +1.09 | T |
| calibrated | T | +1.16 | T |
| transcribed | T | +1.23 | T |

## Restoration (curated passives)

CIs condition on the frozen θ, assignments and Head targets (the classification CIs above do not).

### `passive_1` (affected pairs: 12)

| Condition | affected: Δ margin | affected: Δ accuracy (pp) | Head pairs: Δ margin | share of deficit closed |
|---|---|---|---|---|
| oracle | +0.528 [+0.240, +0.831] | +6.5 [+2.2, +11.4] | +0.000 [-0.000, +0.000] | +0.49 [-2.07, +2.94] |
| classify | +0.528 [+0.240, +0.831] | +6.5 [+2.2, +11.4] | +0.000 [-0.000, +0.000] | +0.49 [-2.07, +2.94] |

classify − oracle: +0.000 [-0.000, +0.000] (90% CI [-0.000, +0.000]); classify − 0.5 × oracle: +0.264 [+0.120, +0.415].

### `passive_2` (affected pairs: 12)

| Condition | affected: Δ margin | affected: Δ accuracy (pp) | Head pairs: Δ margin | share of deficit closed |
|---|---|---|---|---|
| oracle | +0.528 [+0.247, +0.833] | +5.1 [+1.5, +9.3] | -0.000 [-0.000, +0.000] | +1.36 [-10.67, +8.38] |
| classify | +0.528 [+0.247, +0.833] | +5.1 [+1.5, +9.3] | -0.000 [-0.000, +0.000] | +1.36 [-10.67, +8.38] |

classify − oracle: -0.000 [-0.000, +0.000] (90% CI [-0.000, +0.000]); classify − 0.5 × oracle: +0.264 [+0.123, +0.417].

## Released FreqBLiMP

| paradigm | band | items (affected) | unedited acc (%) | Δ acc oracle (pp) | Δ acc classify (pp) | affected items: acc unedited → oracle / classify |
|---|---|---|---|---|---|---|
| passive_1 | head | 1000 (0) | +79.1 [+76.6, +81.7] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | — |
| passive_1 | tail | 1000 (155) | +79.2 [+76.7, +81.6] | +0.4 [+0.1, +0.8] | +0.0 [-0.4, +0.5] | 73% → 75% / 73% |
| passive_1 | xtail | 1000 (0) | +69.4 [+66.5, +72.2] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | — |
| passive_2 | head | 1000 (0) | +81.5 [+79.0, +83.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | — |
| passive_2 | tail | 1000 (164) | +79.3 [+76.8, +81.7] | +0.7 [+0.2, +1.3] | +0.6 [+0.1, +1.2] | 77% → 82% / 81% |
| passive_2 | xtail | 1000 (0) | +73.2 [+70.4, +75.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | — |

Unaffected items pooled, Δ accuracy (oracle): +0.00 [+0.00, +0.00] pp.

Unaffected items pooled, Δ accuracy (classify): +0.00 [+0.00, +0.00] pp.

## LM loss

| Condition | Pile-10k: Δ NLL / token | edited tokens (Pile-val): Δ NLL, per-verb macro |
|---|---|---|
| oracle | +0.0000 [-0.0000, +0.0000] | +0.058 [+0.011, +0.113] |
| classify | +0.0000 [-0.0000, +0.0000] | +0.080 [+0.029, +0.137] |

## Declared decisions

- z classifies rare verbs above chance (balanced accuracy CI above 0.5): **True**.
- Classify-restore improves rare passives (affected curated passive_1, CI above 0 and ≥ 0.25): **True** (oracle: True).
- Achieves half the oracle: **True**; matches the oracle (90% CI within ±0.1): **True**.
- Specific (Head ±0.05 nats; unaffected ±0.5 pp; Pile-10k < +0.002 nats / token): classify True, oracle True.

