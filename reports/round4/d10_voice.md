# Round 4, D10. Clean voice contrast ("has been V" vs "has V")

Spec: `plan.md`, D10. Run: `run_path_patch_frames.py been` (fp32); analysis: `analyze_voice.py`. B7 items and rows (site-8 T-donor interchange in every frame). Base = "The N has been V", counterfactual = "The N has V", sensitivity = "The N was V". Units as B7; pair means, then the mean over pairs; 95% CI over pairs.

Exactness (all components replaced vs S, max |error|): MLP11 2.6e-06, MLP13 2.5e-06, MLP14 3.0e-06, MLP16 3.2e-06, MLP17 3.5e-06, MLP15 1.6e-06, MLP18 2.0e-06, MLP21 1.7e-06, MLP22 2.1e-06

| MLP | readout | step-1 active | Δ has been | Δ has | S = has − has been (T donors) | B7 S (has − was) | S_D (T − I donors) | S_sens = was − has been | PE heads | PE MLPs | half B: fixed top 5 / all heads |
|---|---|---:|---:|---:|---|---:|---|---:|---|---|---:|
| 11 | by | -0.249 | +0.173 | -0.236 | -0.410 [-0.439, -0.379] | -0.414 | -0.416 [-0.444, -0.389] | +0.005 | -0.387 [-0.412, -0.362] | +0.113 [+0.091, +0.138] | +0.310 / +0.386 |
| 13 | by | -0.137 | +0.148 | -0.111 | -0.259 [-0.298, -0.224] | -0.268 | -0.279 [-0.314, -0.245] | +0.012 | -0.270 [-0.294, -0.246] | +0.036 [+0.001, +0.068] | +0.185 / +0.268 |
| 14 | by | -0.451 | +0.310 | -0.313 | -0.623 [-0.680, -0.570] | -0.615 | -0.648 [-0.698, -0.600] | -0.007 | -0.455 [-0.490, -0.422] | -0.002 [-0.047, +0.042] | +0.192 / +0.454 |
| 16 | by | -0.470 | +0.141 | -0.321 | -0.462 [-0.502, -0.419] | -0.471 | -0.483 [-0.524, -0.446] | +0.011 | -0.277 [-0.302, -0.253] | -0.068 [-0.109, -0.027] | +0.124 / +0.280 |
| 17 | by | -0.396 | +0.143 | -0.293 | -0.435 [-0.485, -0.387] | -0.440 | -0.454 [-0.505, -0.407] | +0.008 | -0.351 [-0.382, -0.322] | -0.015 [-0.057, +0.025] | +0.139 / +0.346 |
| 15 | Obar | +0.256 | -0.002 | +0.242 | +0.244 [+0.226, +0.263] | +0.255 | +0.250 [+0.231, +0.269] | -0.012 | -0.020 [-0.031, -0.009] | +0.270 [+0.251, +0.290] | -0.010 / -0.017 |
| 18 | Obar | +0.451 | +0.091 | +0.338 | +0.246 [+0.220, +0.271] | +0.275 | +0.244 [+0.221, +0.266] | -0.030 | +0.029 [+0.020, +0.039] | +0.213 [+0.190, +0.237] | +0.014 / +0.030 |
| 21 | Obar | +0.276 | -0.019 | +0.155 | +0.174 [+0.156, +0.192] | +0.189 | +0.180 [+0.161, +0.197] | -0.015 | -0.012 [-0.017, -0.007] | +0.200 [+0.181, +0.219] | +0.003 / -0.012 |
| 22 | Obar | +0.184 | -0.034 | +0.137 | +0.172 [+0.147, +0.198] | +0.159 | +0.182 [+0.158, +0.207] | +0.012 | -0.028 [-0.035, -0.020] | +0.220 [+0.196, +0.246] | +0.005 / -0.030 |

## Declared decisions

- **Gate:** 9 of 9 → **passes**.
- **Voice switch** (B7 sign, ≥ 50% of B7's |S|, and S_D of the same sign): [11, 13, 14, 16, 17, 15, 18, 21, 22] → **holds**.
- **Not an auxiliary-token effect** (|S_sens| < 0.5 |S|): [11, 13, 14, 16, 17, 15, 18, 21, 22] → **holds**.
- **Two-stage routing:** heads ≥ 0.5|S| for by-MLPs [11, 13, 14, 16, 17]; earlier MLPs ≥ 0.5|S| for object MLPs [15, 18, 21, 22] → **replicates**.
- **B7 heads retain their role:** **holds** (top 10: L9H7, L3H3, L10H2, L8H4, L3H5, L1H6, L4H15, L6H2, L12H12, L3H9; overlap ['L9H7', 'L10H2', 'L3H3', 'L1H6', 'L8H4']).
- New top 5 (half A): L9H7, L3H3, L10H2, L8H4, L3H5.
- Base-run attention from the participle's last token, fixed top 5: to "has" L9H7 0.05, L10H2 0.17, L3H3 0.02, L1H6 0.21, L8H4 0.08; to "been" L9H7 0.28, L10H2 0.17, L3H3 0.76, L1H6 0.44, L8H4 0.37.

