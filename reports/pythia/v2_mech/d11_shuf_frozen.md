# Round 4, D11. Conjunction neurons in MLPs 11 and 14

Spec: `plan.md`, D11. Run: `run_conjunction.py` (fp32); analysis: `analyze_conjunction.py`. B7 items (32 contexts per pair) with T and I donors, was and has frames, site-8 interchange. I_n = (was T − I) − (has T − I) of the neuron's post-activation at the participle's last token, per pair. Pairs split in halves within band (seed 17): selection on half A (43 pairs; t-test, BH-FDR q < 0.05), tests on half B (46 pairs). Direct effects on " by" through one reference final-LN scale per pair.

**Selected on half A:** 10082 neurons (MLP11: 5040, MLP14: 5042); categories: active-conjunction 4228, graded (same sign in both frames) 1633, opposite signs 1181, passive-conjunction 3040.

**Frozen set:** the selection above is this run's own and is not used. Every half-B test below uses the 2986 passive-conjunction neurons and half-A signs of `results/pythia/das_v2/mech/conj_v2/conjunction_selection.npz`.
- Passive-conjunction set on half B: sign agreement 0.97; aligned I_n +0.020 [+0.019, +0.022]; was T − I +0.023 [+0.021, +0.024]; has T − I +0.002 [+0.002, +0.002].
- **Conjunction neurons exist: yes** (≥ 10 passive-conjunction neurons; half B: ≥ 80% sign agreement, aligned I_n and was effect CIs above 0, |has| < 0.5 × was).
- Share of the MLP 11 + 14 " by" switch (direct effect of I_n, half B): +0.650 [+0.605, +0.698] (+0.413 of +0.635); share of the passive T − I effect: +1.354 [+1.251, +1.478] → **they carry the by switch: yes** (switch share ≥ 0.25, CI above 0).

Generalization (sign-aligned mean activation difference of the passive-conjunction set):

- natural good − bad passives: Head +0.028 [+0.020, +0.037] (12 pairs), Tail +0.025 [+0.019, +0.030], XTail +0.029 [+0.025, +0.033]; XTail / Head +1.043 [+0.746, +1.512]
- nonce probes: matched T − I +0.011 [+0.010, +0.012]; balanced AB − BA +0.004 [+0.003, +0.005]

Top passive-conjunction neurons (|t| on half A):

| neuron | t (A) | I (A) | I (B) | was: T − I | has: T − I | w_by |
|---|---:|---:|---:|---:|---:|---:|
| MLP11.n1599 | +21.4 | +0.141 | +0.127 | +0.129 | -0.005 | -0.0537 |
| MLP14.n7081 | +20.7 | +0.237 | +0.240 | +0.231 | -0.008 | -0.0926 |
| MLP11.n3462 | +19.5 | +0.546 | +0.445 | +0.446 | -0.048 | +0.7518 |
| MLP11.n7480 | +19.4 | +0.087 | +0.075 | +0.112 | +0.032 | -0.0942 |
| MLP11.n3120 | -18.8 | -0.009 | -0.011 | -0.009 | +0.001 | -0.1743 |
| MLP11.n1686 | +17.8 | +0.181 | +0.167 | +0.267 | +0.093 | +0.3609 |
| MLP14.n4162 | +17.6 | +0.154 | +0.134 | +0.120 | -0.024 | +0.0093 |
| MLP11.n4817 | +17.0 | +0.082 | +0.093 | +0.102 | +0.014 | -0.0259 |
| MLP11.n7710 | +16.9 | +0.015 | +0.013 | +0.015 | +0.001 | -0.1635 |
| MLP11.n6821 | +16.7 | +0.010 | +0.011 | +0.011 | +0.001 | +0.0010 |
| MLP11.n4274 | +16.5 | +0.107 | +0.119 | +0.127 | +0.014 | -0.0553 |
| MLP11.n4129 | -16.5 | -0.008 | -0.010 | -0.008 | +0.001 | -0.0918 |
| MLP14.n5407 | +16.4 | +0.311 | +0.274 | +0.292 | +0.001 | +1.0670 |
| MLP11.n3907 | +16.4 | +0.080 | +0.071 | +0.071 | -0.004 | -0.0599 |
| MLP11.n6589 | +16.1 | +0.073 | +0.071 | +0.095 | +0.023 | -0.0855 |
| MLP14.n6147 | +16.0 | +0.030 | +0.037 | +0.045 | +0.011 | +0.0061 |
| MLP14.n108 | +15.9 | +0.192 | +0.190 | +0.166 | -0.025 | +0.0465 |
| MLP11.n453 | +15.8 | +0.064 | +0.056 | +0.107 | +0.047 | -0.0105 |
| MLP11.n7847 | +15.7 | +0.070 | +0.059 | +0.144 | +0.080 | -0.0763 |
| MLP14.n3309 | +15.7 | +0.069 | +0.074 | +0.110 | +0.039 | +0.0401 |
| MLP11.n3963 | +15.6 | +0.008 | +0.009 | +0.005 | -0.003 | +0.0038 |
| MLP14.n1044 | +15.6 | +0.169 | +0.157 | +0.171 | +0.008 | -0.0603 |
| MLP11.n3040 | +15.6 | +0.106 | +0.121 | +0.120 | +0.006 | +0.0171 |
| MLP14.n1343 | +15.1 | +0.159 | +0.131 | +0.108 | -0.037 | -0.1146 |
| MLP14.n4774 | +15.1 | +0.295 | +0.263 | +0.275 | -0.004 | -0.1374 |

## Exploratory (not declared)

Passive-conjunction neurons ranked by their half-A direct effect on the switch; shares on half B (of the whole MLP 11 + 14 switch, +0.635): top 10 0.67, top 50 0.91, top 100 1.00, top 2986 0.65. MLP11 / MLP14 share of the switch: 0.38 / 0.62. Largest: MLP11.n3462 (0.16), MLP14.n5407 (0.14), MLP14.n2114 (0.12), MLP11.n4925 (0.05), MLP14.n4241 (0.05).

Natural good − bad activation difference × " by" output weight (unscaled), summed over the set, per band (pair bootstrap); patched was T − I in the same units:

| set | Head | Tail | XTail | XTail / Head | patched was T − I (Head / Tail / XTail) |
|---|---|---|---|---|---|
| top 10 | +1.368 [+0.570, +2.371] | +1.007 [+0.475, +1.580] | +0.688 [+0.123, +1.241] | +0.503 [+0.092, +1.428] | +1.29 / +1.37 / +1.50 |
| top 50 | +2.054 [+1.155, +3.077] | +1.535 [+0.984, +2.131] | +1.204 [+0.645, +1.794] | +0.586 [+0.290, +1.177] | +1.63 / +1.87 / +1.98 |
| top 100 | +2.348 [+1.482, +3.417] | +1.774 [+1.177, +2.370] | +1.482 [+0.892, +2.107] | +0.631 [+0.345, +1.166] | +1.79 / +2.07 / +2.19 |

All MLP 11 + 14 neurons, same quantity: Head +1.40, Tail +1.28, XTail +2.51. Share of neurons with negative mean post-activation (has / was frame): passive-conjunction 0.89 / 0.79, all 0.85 / 0.85.

