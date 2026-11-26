# Round 4, D11. Conjunction neurons in MLPs 11 and 14

Spec: `plan.md`, D11. Run: `run_conjunction.py` (fp32); analysis: `analyze_conjunction.py`. B7 items (32 contexts per pair) with T and I donors, was and has frames, site-8 interchange. I_n = (was T − I) − (has T − I) of the neuron's post-activation at the participle's last token, per pair. Pairs split in halves within band (seed 17): selection on half A (43 pairs; t-test, BH-FDR q < 0.05), tests on half B (46 pairs). Direct effects on " by" through one reference final-LN scale per pair.

**Selected on half A:** 9758 neurons (MLP11: 4956, MLP14: 4802); categories: active-conjunction 4042, graded (same sign in both frames) 1611, opposite signs 1093, passive-conjunction 3012.

**Frozen set:** the selection above is this run's own and is not used. Every half-B test below uses the 2986 passive-conjunction neurons and half-A signs of `results/pythia/das_v2/mech/conj_v2/conjunction_selection.npz`.
- Passive-conjunction set on half B: sign agreement 0.92; aligned I_n +0.027 [+0.025, +0.029]; was T − I +0.030 [+0.028, +0.032]; has T − I +0.002 [+0.002, +0.003].
- **Conjunction neurons exist: yes** (≥ 10 passive-conjunction neurons; half B: ≥ 80% sign agreement, aligned I_n and was effect CIs above 0, |has| < 0.5 × was).
- Share of the MLP 11 + 14 " by" switch (direct effect of I_n, half B): +0.699 [+0.653, +0.748] (+0.560 of +0.802); share of the passive T − I effect: +1.226 [+1.125, +1.355] → **they carry the by switch: yes** (switch share ≥ 0.25, CI above 0).

Generalization (sign-aligned mean activation difference of the passive-conjunction set):

- natural good − bad passives: Head +0.028 [+0.020, +0.037] (12 pairs), Tail +0.025 [+0.019, +0.030], XTail +0.029 [+0.025, +0.033]; XTail / Head +1.043 [+0.746, +1.512]
- nonce probes: matched T − I +0.011 [+0.010, +0.012]; balanced AB − BA +0.004 [+0.003, +0.005]

Top passive-conjunction neurons (|t| on half A):

| neuron | t (A) | I (A) | I (B) | was: T − I | has: T − I | w_by |
|---|---:|---:|---:|---:|---:|---:|
| MLP11.n1599 | +21.4 | +0.137 | +0.141 | +0.112 | -0.027 | -0.0537 |
| MLP14.n7081 | +20.7 | +0.338 | +0.335 | +0.333 | -0.003 | -0.0926 |
| MLP11.n3462 | +19.5 | +0.769 | +0.681 | +0.686 | -0.037 | +0.7518 |
| MLP11.n7480 | +19.4 | +0.153 | +0.139 | +0.208 | +0.062 | -0.0942 |
| MLP11.n3120 | -18.8 | -0.017 | -0.018 | -0.016 | +0.002 | -0.1743 |
| MLP11.n1686 | +17.8 | +0.204 | +0.176 | +0.252 | +0.063 | +0.3609 |
| MLP14.n4162 | +17.6 | +0.127 | +0.105 | +0.067 | -0.049 | +0.0093 |
| MLP11.n4817 | +17.0 | +0.177 | +0.189 | +0.213 | +0.029 | -0.0259 |
| MLP11.n7710 | +16.9 | +0.030 | +0.027 | +0.018 | -0.011 | -0.1635 |
| MLP11.n6821 | +16.7 | +0.018 | +0.020 | +0.016 | -0.003 | +0.0010 |
| MLP11.n4274 | +16.5 | +0.135 | +0.140 | +0.149 | +0.011 | -0.0553 |
| MLP11.n4129 | -16.5 | -0.009 | -0.011 | -0.016 | -0.006 | -0.0918 |
| MLP14.n5407 | +16.4 | +0.266 | +0.208 | +0.240 | +0.004 | +1.0670 |
| MLP11.n3907 | +16.4 | +0.184 | +0.173 | +0.186 | +0.008 | -0.0599 |
| MLP11.n6589 | +16.1 | +0.124 | +0.118 | +0.167 | +0.046 | -0.0855 |
| MLP14.n6147 | +16.0 | +0.042 | +0.045 | +0.062 | +0.019 | +0.0061 |
| MLP14.n108 | +15.9 | +0.158 | +0.169 | +0.092 | -0.071 | +0.0465 |
| MLP11.n453 | +15.8 | +0.087 | +0.068 | +0.090 | +0.013 | -0.0105 |
| MLP11.n7847 | +15.7 | +0.204 | +0.177 | +0.216 | +0.026 | -0.0763 |
| MLP14.n3309 | +15.7 | +0.055 | +0.051 | +0.069 | +0.016 | +0.0401 |
| MLP11.n3963 | +15.6 | +0.013 | +0.016 | +0.007 | -0.008 | +0.0038 |
| MLP14.n1044 | +15.6 | +0.208 | +0.190 | +0.208 | +0.009 | -0.0603 |
| MLP11.n3040 | +15.6 | +0.099 | +0.109 | +0.035 | -0.069 | +0.0171 |
| MLP14.n1343 | +15.1 | +0.221 | +0.171 | +0.234 | +0.038 | -0.1146 |
| MLP14.n4774 | +15.1 | +0.359 | +0.309 | +0.325 | -0.008 | -0.1374 |

## Exploratory (not declared)

Passive-conjunction neurons ranked by their half-A direct effect on the switch; shares on half B (of the whole MLP 11 + 14 switch, +0.802): top 10 0.67, top 50 0.92, top 100 1.03, top 2986 0.70. MLP11 / MLP14 share of the switch: 0.45 / 0.55. Largest: MLP11.n3462 (0.20), MLP14.n2114 (0.16), MLP14.n5407 (0.09), MLP11.n4925 (0.05), MLP14.n4241 (0.04).

Natural good − bad activation difference × " by" output weight (unscaled), summed over the set, per band (pair bootstrap); patched was T − I in the same units:

| set | Head | Tail | XTail | XTail / Head | patched was T − I (Head / Tail / XTail) |
|---|---|---|---|---|---|
| top 10 | +1.354 [+0.564, +2.311] | +1.013 [+0.468, +1.604] | +0.732 [+0.159, +1.300] | +0.541 [+0.121, +1.473] | +1.83 / +1.78 / +1.83 |
| top 50 | +2.082 [+1.187, +3.117] | +1.517 [+0.963, +2.111] | +1.264 [+0.700, +1.851] | +0.607 [+0.305, +1.202] | +2.29 / +2.41 / +2.46 |
| top 100 | +2.329 [+1.377, +3.460] | +1.799 [+1.199, +2.392] | +1.573 [+0.998, +2.185] | +0.675 [+0.375, +1.299] | +2.56 / +2.73 / +2.75 |

All MLP 11 + 14 neurons, same quantity: Head +1.40, Tail +1.28, XTail +2.51. Share of neurons with negative mean post-activation (has / was frame): passive-conjunction 0.89 / 0.79, all 0.86 / 0.85.

