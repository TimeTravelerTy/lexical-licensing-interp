# Round 4, D11. Conjunction neurons in MLPs 11 and 14

Spec: `plan.md`, D11. Run: `run_conjunction.py` (fp32); analysis: `analyze_conjunction.py`. B7 items (32 contexts per pair) with T and I donors, was and has frames, site-8 interchange. I_n = (was T − I) − (has T − I) of the neuron's post-activation at the participle's last token, per pair. Pairs split in halves within band (seed 17): selection on half A (31 pairs; t-test, BH-FDR q < 0.05), tests on half B (33 pairs). Direct effects on " by" through one reference final-LN scale per pair.

**Selected on half A:** 9344 neurons (MLP11: 4742, MLP14: 4602); categories: active-conjunction 4083, graded (same sign in both frames) 1362, opposite signs 1117, passive-conjunction 2782.
- Passive-conjunction set on half B: sign agreement 0.99; aligned I_n +0.036 [+0.034, +0.039]; was T − I +0.041 [+0.039, +0.044]; has T − I +0.005 [+0.004, +0.006].
- **Conjunction neurons exist: yes** (≥ 10 passive-conjunction neurons; half B: ≥ 80% sign agreement, aligned I_n and was effect CIs above 0, |has| < 0.5 × was).
- Share of the MLP 11 + 14 " by" switch (direct effect of I_n, half B): +0.649 [+0.602, +0.697] (+0.713 of +1.099); share of the passive T − I effect: +1.220 [+1.133, +1.318] → **they carry the by switch: yes** (switch share ≥ 0.25, CI above 0).

Generalization (sign-aligned mean activation difference of the passive-conjunction set):

- natural good − bad passives: Head +0.033 [+0.024, +0.041] (9 pairs), Tail +0.028 [+0.023, +0.034], XTail +0.028 [+0.024, +0.032]; XTail / Head +0.839 [+0.636, +1.211]
- nonce probes: matched T − I +0.012 [+0.011, +0.013]; balanced AB − BA +0.005 [+0.004, +0.006]

Top passive-conjunction neurons (|t| on half A):

| neuron | t (A) | I (A) | I (B) | was: T − I | has: T − I | w_by |
|---|---:|---:|---:|---:|---:|---:|
| MLP11.n1686 | +18.6 | +0.273 | +0.289 | +0.423 | +0.142 | +0.3609 |
| MLP11.n3120 | -18.1 | -0.019 | -0.018 | -0.014 | +0.004 | -0.1743 |
| MLP11.n3462 | +17.6 | +0.850 | +0.912 | +0.827 | -0.055 | +0.7518 |
| MLP11.n1599 | +17.0 | +0.244 | +0.206 | +0.203 | -0.022 | -0.0537 |
| MLP14.n7081 | +16.4 | +0.371 | +0.377 | +0.365 | -0.009 | -0.0926 |
| MLP11.n4817 | +15.9 | +0.155 | +0.176 | +0.184 | +0.019 | -0.0259 |
| MLP14.n5776 | -15.8 | -0.021 | -0.019 | -0.014 | +0.006 | +0.0648 |
| MLP11.n7710 | +15.1 | +0.029 | +0.026 | +0.023 | -0.004 | -0.1635 |
| MLP11.n5746 | +14.5 | +0.044 | +0.043 | +0.042 | -0.002 | -0.0082 |
| MLP11.n7847 | +14.2 | +0.202 | +0.172 | +0.276 | +0.090 | -0.0763 |
| MLP11.n7480 | +14.2 | +0.155 | +0.137 | +0.192 | +0.046 | -0.0942 |
| MLP11.n2600 | -14.0 | -0.187 | -0.175 | -0.125 | +0.056 | -0.0481 |
| MLP11.n453 | +13.9 | +0.099 | +0.102 | +0.177 | +0.076 | -0.0105 |
| MLP11.n6821 | +13.9 | +0.020 | +0.018 | +0.020 | +0.001 | +0.0010 |
| MLP14.n3517 | +13.7 | +0.554 | +0.595 | +0.539 | -0.036 | +0.1702 |
| MLP14.n4036 | +13.5 | +0.155 | +0.138 | +0.129 | -0.018 | -0.0110 |
| MLP11.n7604 | +13.5 | +0.218 | +0.226 | +0.274 | +0.053 | +0.0689 |
| MLP11.n3907 | +13.2 | +0.162 | +0.178 | +0.170 | -0.000 | -0.0599 |
| MLP11.n7177 | +13.2 | +0.180 | +0.201 | +0.290 | +0.099 | +0.1333 |
| MLP14.n4327 | +13.1 | +0.199 | +0.226 | +0.161 | -0.052 | +0.0147 |
| MLP14.n4145 | +13.0 | +0.167 | +0.176 | +0.240 | +0.068 | -0.0347 |
| MLP11.n2900 | +13.0 | +0.017 | +0.010 | +0.012 | -0.001 | -0.0438 |
| MLP11.n2494 | +12.9 | +0.087 | +0.090 | +0.141 | +0.052 | +0.0339 |
| MLP11.n3468 | +12.9 | +0.102 | +0.091 | +0.067 | -0.029 | +0.0068 |
| MLP11.n6589 | +12.9 | +0.147 | +0.145 | +0.187 | +0.041 | -0.0855 |

## Exploratory (not declared)

Passive-conjunction neurons ranked by their half-A direct effect on the switch; shares on half B (of the whole MLP 11 + 14 switch, +1.099): top 10 0.66, top 50 0.86, top 100 0.95, top 2782 0.65. MLP11 / MLP14 share of the switch: 0.41 / 0.59. Largest: MLP11.n3462 (0.20), MLP14.n2114 (0.14), MLP14.n5407 (0.11), MLP11.n4925 (0.05), MLP14.n4241 (0.05).

Natural good − bad activation difference × " by" output weight (unscaled), summed over the set, per band (pair bootstrap); patched was T − I in the same units:

| set | Head | Tail | XTail | XTail / Head | patched was T − I (Head / Tail / XTail) |
|---|---|---|---|---|---|
| top 10 | +1.331 [+0.385, +2.659] | +1.229 [+0.676, +1.815] | +0.347 [-0.187, +0.860] | +0.261 [-0.141, +1.421] | +2.08 / +2.24 / +2.35 |
| top 50 | +1.949 [+0.927, +3.260] | +1.697 [+1.107, +2.311] | +0.847 [+0.292, +1.372] | +0.435 [+0.138, +1.032] | +2.57 / +3.01 / +3.00 |
| top 100 | +2.335 [+1.333, +3.737] | +2.006 [+1.333, +2.649] | +1.174 [+0.603, +1.771] | +0.503 [+0.216, +1.024] | +2.87 / +3.36 / +3.36 |

All MLP 11 + 14 neurons, same quantity: Head +0.68, Tail +1.51, XTail +2.15. Share of neurons with negative mean post-activation (has / was frame): passive-conjunction 0.89 / 0.77, all 0.85 / 0.85.

