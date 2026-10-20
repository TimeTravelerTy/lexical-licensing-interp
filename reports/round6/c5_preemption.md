# Round 6, C5: in-context preemption

Spec: `plan.md`, C5. Contexts: `build_preemption.py`; run: `run_preemption.py` (Pythia-1.4B, fp32); analysis: `analyze_preemption.py`. Δ_pass = log P(V | ctx + "The N was") − log P(V | ctx + "The N has"), mean over A1's 24 nouns. k = 0 reproduces A1's D11 scores (max |Δ lp| 0.0e+00). E_arm = Δ_pass(arm) − Δ_pass(filler) at k = 8. 95% CIs: verbs within band × nouns within animacy, 2,000 draws. Rule (bound 0.25 nats): lowers = 95% CI below 0 and estimate ≤ −0.25; raises = mirror; negligible = 90% CI within ±0.25.

## Δ_pass by arm and k (means; rare I = Tail + XTail intransitives)

| group | arm | k=0 | k=1 | k=2 | k=4 | k=8 |
Primary population: 233 rare dynamic intransitives (of 234); Head reference 40; T control 80.
|---|---|---|---|---|---|---|
| rare I | pre | -1.74 | -2.63 | -2.18 | -2.47 | -2.39 |
| rare I | exp | -1.74 | -3.03 | -3.28 | -3.46 | -3.47 |
| rare I | mention | -1.74 | -1.38 | -1.54 | -1.41 | -1.26 |
| rare I | fill | -1.74 | -2.73 | -3.21 | -3.49 | -3.56 |
| rare I | other | -1.74 | -2.69 | -2.74 | -2.85 | -2.82 |
| rare I | vs | -1.74 | -2.85 | -2.70 | -3.01 | -2.98 |
| rare I | ing | -1.74 | -2.41 | -2.31 | -3.11 | -3.54 |
| Head I | pre | -2.81 | -3.56 | -3.67 | -4.14 | -4.18 |
| Head I | exp | -2.81 | -3.97 | -4.04 | -4.33 | -4.18 |
| Head I | mention | -2.81 | -3.73 | -4.05 | -4.32 | -4.59 |
| Head I | fill | -2.81 | -3.70 | -4.15 | -4.36 | -4.44 |
| Head I | other | -2.81 | -3.64 | -3.60 | -3.52 | -3.34 |
| Head I | vs | -2.81 | -3.71 | -3.86 | -4.23 | -4.23 |
| Head I | ing | -2.81 | -3.29 | -2.73 | -3.04 | -3.40 |
| rare T | trans | +0.66 | +0.02 | +0.24 | -0.47 | -0.62 |
| rare T | fill_t | +0.66 | +0.15 | -0.13 | -0.42 | -0.54 |

## Contrasts at k = 8, was / has (primary)

- **E_pre (rare I) +1.17 [+0.89, +1.43] → raises**; E_exp +0.09 [-0.23, +0.42]; E_mention +2.30 [+1.94, +2.65]; E_other +0.74 [+0.54, +0.93]; E_vs (present) +0.58 [+0.28, +0.87]; E_ing (is V-ing) +0.01 [-0.30, +0.32].
- E_pre − E_exp +1.08 [+0.78, +1.37] → beyond exposure: **False**; E_pre − E_mention -1.13 [-1.40, -0.86] → beyond exact-form mention: **True**; E_pre − E_other +0.43 [+0.21, +0.65] → verb-specific: **False**; E_pre − E_vs +0.59 [+0.41, +0.76]; E_pre − E_ing +1.15 [+0.81, +1.47].
- Head level (Head I under filler, k = 8) -4.44 [-4.93, -3.89]; rare − Head gap G +0.88 [+0.31, +1.41]; distance reduction -1.17 [-1.43, -0.89] → moves toward Head: **False**; share closed -1.33 [-3.93, -0.72]. Head I: E_pre +0.26 [-0.06, +0.58].
- Sensitivities: E_pre all rare I (incl. stative) +1.16 [+0.88, +1.42]; E_pre - E_mention all rare I (incl. stative) -1.14 [-1.41, -0.87]; E_pre theme subjects (rare, dynamic) +1.28 [+0.64, +1.90]; E_pre - E_mention theme subjects (rare, dynamic) -1.90 [-2.39, -1.40].
- Dose: slope of Δ_pass(pre) − Δ_pass(filler) on log2 k +0.32 [+0.23, +0.41] per doubling.
- **T control** E_T -0.08 [-0.39, +0.26] → does not fall: **False**.

Raw log-probs at k = 8 (arm − filler):

| group | arm | Δ lp(V \| was) | Δ lp(V \| has) | Δ lp(V \| has been) |
|---|---|---|---|---|
| rare I | pre | +12.41 [+11.95, +12.84] | +11.24 [+10.90, +11.56] | +11.84 [+11.45, +12.21] |
| rare I | exp | +7.03 [+6.64, +7.41] | +6.94 [+6.49, +7.37] | +8.21 [+7.77, +8.65] |
| rare I | mention | +13.14 [+12.50, +13.71] | +10.84 [+10.28, +11.38] | +12.18 [+11.56, +12.76] |
| rare I | other | +2.33 [+2.08, +2.59] | +1.59 [+1.30, +1.89] | +1.74 [+1.49, +1.97] |
| rare I | vs | +10.22 [+9.73, +10.66] | +9.64 [+9.25, +10.01] | +9.97 [+9.53, +10.39] |
| rare I | ing | +10.91 [+10.55, +11.27] | +10.89 [+10.51, +11.26] | +10.85 [+10.52, +11.18] |
| Head I | pre | +8.15 [+7.46, +8.85] | +7.89 [+7.25, +8.52] | +7.18 [+6.51, +7.90] |
| Head I | exp | +3.98 [+3.59, +4.38] | +3.72 [+3.30, +4.18] | +5.20 [+4.66, +5.77] |
| Head I | mention | +7.19 [+6.24, +8.12] | +7.35 [+6.55, +8.17] | +7.02 [+6.19, +7.87] |
| Head I | other | +2.62 [+2.17, +3.07] | +1.52 [+1.05, +2.01] | +1.81 [+1.30, +2.34] |
| Head I | vs | +6.84 [+6.21, +7.46] | +6.63 [+6.04, +7.20] | +6.13 [+5.51, +6.73] |
| Head I | ing | +9.21 [+8.50, +9.88] | +8.17 [+7.43, +8.91] | +8.09 [+7.34, +8.84] |
| rare T | trans | +13.68 [+13.04, +14.27] | +13.76 [+13.20, +14.29] | +12.81 [+12.30, +13.29] |

Raw levels by dose (rare I, mean lp after was / has):

| arm | k=0 | k=1 | k=2 | k=4 | k=8 |
|---|---|---|---|---|---|
| pre | -16.32 / -14.58 | -12.69 / -10.06 | -8.59 / -6.41 | -6.57 / -4.10 | -5.77 / -3.38 |
| exp | -16.32 / -14.58 | -12.38 / -9.35 | -11.44 / -8.16 | -11.07 / -7.60 | -11.15 / -7.68 |
| mention | -16.32 / -14.58 | -8.40 / -7.01 | -6.43 / -4.88 | -5.38 / -3.97 | -5.04 / -3.78 |
| fill | -16.32 / -14.58 | -16.90 / -14.17 | -17.37 / -14.16 | -17.83 / -14.33 | -18.18 / -14.62 |
| other | -16.32 / -14.58 | -16.28 / -13.59 | -15.82 / -13.08 | -15.89 / -13.03 | -15.85 / -13.03 |
| vs | -16.32 / -14.58 | -13.23 / -10.38 | -10.20 / -7.50 | -8.59 / -5.58 | -7.96 / -4.98 |
| ing | -16.32 / -14.58 | -12.31 / -9.89 | -8.88 / -6.57 | -7.65 / -4.54 | -7.27 / -3.73 |

Breakdowns (E_pre, was / has):

- Tail (131 verbs): E_pre +1.09 [+0.76, +1.43], E_exp +0.31 [-0.03, +0.67].
- XTail (102 verbs): E_pre +1.27 [+0.92, +1.62], E_exp -0.19 [-0.59, +0.22].
- single-token (20 verbs): E_pre -0.43 [-0.98, +0.05], E_exp -0.22 [-0.77, +0.23].
- multi-token (213 verbs): E_pre +1.32 [+1.03, +1.59], E_exp +0.12 [-0.21, +0.47].
- no complement (205 verbs): E_pre +1.20 [+0.90, +1.48], E_exp +0.10 [-0.23, +0.45].
- with complement (28 verbs): E_pre +0.94 [+0.37, +1.48], E_exp +0.00 [-0.45, +0.46].

## Contrasts at k = 8, has been / has

- **E_pre (rare I) +0.60 [+0.33, +0.86] → raises**; E_exp +1.27 [+1.03, +1.50]; E_mention +1.34 [+0.86, +1.74]; E_other +0.14 [-0.07, +0.35]; E_vs (present) +0.34 [+0.08, +0.59]; E_ing (is V-ing) -0.05 [-0.31, +0.21].
- E_pre − E_exp -0.67 [-0.88, -0.46] → beyond exposure: **True**; E_pre − E_mention -0.74 [-1.03, -0.40] → beyond exact-form mention: **True**; E_pre − E_other +0.45 [+0.26, +0.65] → verb-specific: **False**; E_pre − E_vs +0.26 [+0.16, +0.36]; E_pre − E_ing +0.64 [+0.44, +0.85].
- Head level (Head I under filler, k = 8) -2.94 [-3.48, -2.38]; rare − Head gap G +0.51 [-0.07, +1.11]; distance reduction -0.60 [-0.86, -0.33] → moves toward Head: **False**; share closed not interpreted (G not above 0). Head I: E_pre -0.71 [-1.08, -0.32].
- Sensitivities: E_pre all rare I (incl. stative) +0.59 [+0.33, +0.86]; E_pre - E_mention all rare I (incl. stative) -0.74 [-1.03, -0.40]; E_pre theme subjects (rare, dynamic) +1.01 [+0.45, +1.55]; E_pre - E_mention theme subjects (rare, dynamic) -1.78 [-2.29, -1.26].
- Dose: slope of Δ_pass(pre) − Δ_pass(filler) on log2 k +0.08 [+0.00, +0.17] per doubling.
- **T control** E_T -0.95 [-1.24, -0.67] → does not fall: **False**.

## Declared decisions (was / has)

- Preemption lowers Δ_pass(I): **raises**; beyond exposure: **False**; beyond exact-form mention: **True**; verb-specific: **False**; T control does not fall: **False** → in-context preemption claimed (all five): **False**.
- Moves toward the Head level (secondary): **False**.

