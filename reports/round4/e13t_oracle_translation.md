# Round 4, E13-T. Oracle at the translation stage

Spec: `plan.md`, E13-T. Run: `run_oracle_translation.py` (fp32); analysis: `analyze_oracle_translation.py`. Curated `passive_1` and `passive_2` sentences of all 126 band-cross pairs. At the participle's last token, set to the Head mean of the true class (same context, own pair excluded): `mlp` = MLP 11–17 outputs; `neurons` = D11's top-50 switch neurons; `random` = mean over 5 random sets of 50 (same layer split); `wmatched` = 50 neurons D11 did not select, matched one-to-one on |w_by|. Mean |w_by|: neurons 0.234, random1 0.049, random2 0.045, random3 0.047, random4 0.052, random5 0.047, wmatched 0.176. Margins good − bad; Δ = patched − natural; first = the first suffix token (" by" in passive_1, "." in passive_2). 95% CIs: pairs within band × contexts, the same draws for every intervention.

## All 126 pairs

### passive_1: Δ first (" by") margin

| Group | natural | mlp | neurons | random | wmatched |
|---|---|---|---|---|---|
| xtail@head | +0.488 [+0.129, +0.858] | +0.034 [-0.245, +0.325] | +0.165 [+0.103, +0.229] | -0.004 [-0.006, -0.002] | -0.025 [-0.036, -0.016] |
| tail@head | +0.967 [+0.604, +1.329] | -0.104 [-0.382, +0.173] | +0.054 [-0.011, +0.114] | -0.002 [-0.004, +0.001] | -0.008 [-0.017, +0.001] |
| head@head | +0.947 [+0.189, +1.677] | -0.066 [-0.691, +0.540] | -0.094 [-0.208, +0.019] | +0.001 [-0.003, +0.005] | -0.002 [-0.013, +0.010] |
| head@xtail | +0.947 [+0.189, +1.677] | -0.295 [-0.899, +0.291] | -0.323 [-0.469, -0.181] | +0.004 [-0.000, +0.009] | +0.020 [+0.009, +0.031] |

### passive_1: Δ whole margin

| Group | natural | mlp | neurons | random | wmatched |
|---|---|---|---|---|---|
| xtail@head | +2.135 [+1.613, +2.689] | +0.048 [-0.253, +0.361] | +0.159 [+0.096, +0.223] | -0.005 [-0.008, -0.003] | -0.028 [-0.040, -0.017] |
| tail@head | +2.758 [+2.159, +3.366] | -0.129 [-0.413, +0.156] | +0.050 [-0.016, +0.110] | -0.003 [-0.005, -0.000] | -0.009 [-0.018, -0.000] |
| head@head | +2.956 [+2.206, +3.658] | +0.034 [-0.622, +0.671] | -0.097 [-0.212, +0.018] | +0.000 [-0.004, +0.005] | -0.003 [-0.014, +0.009] |
| head@xtail | +2.956 [+2.206, +3.658] | -0.224 [-0.849, +0.395] | -0.324 [-0.475, -0.177] | +0.004 [-0.000, +0.010] | +0.022 [+0.012, +0.033] |

- mlp: share of the Head − XTail " by" deficit (+0.459) closed +0.074 [-2.632, +2.480]; patched gap +0.425 [-0.339, +1.163]
- neurons: share of the Head − XTail " by" deficit (+0.459) closed +0.359 [-3.226, +3.554]; patched gap +0.294 [-0.497, +1.097]
- random: share of the Head − XTail " by" deficit (+0.459) closed -0.009 [-0.083, +0.094]; patched gap +0.463 [-0.347, +1.260]
- wmatched: share of the Head − XTail " by" deficit (+0.459) closed -0.055 [-0.554, +0.604]; patched gap +0.485 [-0.327, +1.279]
- XTail Δ " by", neurons − random: +0.169 [+0.107, +0.234]
- XTail Δ " by", neurons − wmatched: +0.190 [+0.124, +0.258]

### passive_2: Δ first (".") margin

| Group | natural | mlp | neurons | random | wmatched |
|---|---|---|---|---|---|
| xtail@head | +0.107 [-0.248, +0.465] | -0.291 [-0.520, -0.061] | +0.008 [-0.022, +0.039] | -0.003 [-0.005, -0.001] | -0.006 [-0.017, +0.003] |
| tail@head | +0.629 [+0.226, +1.039] | -0.587 [-0.851, -0.337] | -0.084 [-0.139, -0.032] | -0.005 [-0.007, -0.004] | -0.006 [-0.015, +0.003] |
| head@head | -0.281 [-0.956, +0.365] | -0.023 [-0.480, +0.445] | -0.118 [-0.208, -0.033] | -0.002 [-0.007, +0.002] | -0.009 [-0.022, +0.002] |
| head@xtail | -0.281 [-0.956, +0.365] | +0.269 [-0.217, +0.758] | -0.164 [-0.294, -0.043] | +0.000 [-0.004, +0.005] | -0.011 [-0.023, -0.000] |

### passive_2: Δ whole margin

| Group | natural | mlp | neurons | random | wmatched |
|---|---|---|---|---|---|
| xtail@head | +1.515 [+1.005, +2.009] | -0.291 [-0.520, -0.061] | +0.008 [-0.022, +0.039] | -0.003 [-0.005, -0.001] | -0.006 [-0.017, +0.003] |
| tail@head | +2.604 [+2.081, +3.146] | -0.587 [-0.851, -0.337] | -0.084 [-0.139, -0.032] | -0.005 [-0.007, -0.004] | -0.006 [-0.015, +0.003] |
| head@head | +2.565 [+1.888, +3.289] | -0.023 [-0.480, +0.445] | -0.118 [-0.208, -0.033] | -0.002 [-0.007, +0.002] | -0.009 [-0.022, +0.002] |
| head@xtail | +2.565 [+1.888, +3.289] | +0.269 [-0.217, +0.758] | -0.164 [-0.294, -0.043] | +0.000 [-0.004, +0.005] | -0.011 [-0.023, -0.000] |

- mlp: share of the Head − XTail "." deficit (-0.389) closed +0.747 [-5.646, +6.309]; patched gap -0.098 [-0.785, +0.577]
- neurons: share of the Head − XTail "." deficit (-0.389) closed -0.019 [-0.385, +0.291]; patched gap -0.396 [-1.143, +0.324]
- random: share of the Head − XTail "." deficit (-0.389) closed +0.007 [-0.065, +0.072]; patched gap -0.386 [-1.141, +0.337]
- wmatched: share of the Head − XTail "." deficit (-0.389) closed +0.016 [-0.130, +0.139]; patched gap -0.383 [-1.137, +0.340]
- XTail Δ ".", neurons − random: +0.010 [-0.019, +0.042]
- XTail Δ ".", neurons − wmatched: +0.014 [-0.013, +0.042]

## Primary pairs (as E13)

### passive_1: Δ first (" by") margin

| Group | natural | mlp | neurons | random | wmatched |
|---|---|---|---|---|---|
| xtail@head | +0.414 [-0.080, +0.910] | +0.056 [-0.334, +0.454] | +0.183 [+0.100, +0.263] | -0.004 [-0.007, -0.001] | -0.028 [-0.039, -0.018] |
| tail@head | +0.822 [+0.346, +1.268] | +0.127 [-0.246, +0.523] | +0.094 [+0.021, +0.174] | -0.002 [-0.005, +0.001] | -0.020 [-0.029, -0.012] |
| head@head | +0.958 [-0.337, +2.130] | +0.142 [-0.810, +1.150] | -0.023 [-0.185, +0.145] | +0.005 [-0.003, +0.013] | -0.013 [-0.030, +0.006] |
| head@xtail | +0.958 [-0.337, +2.130] | -0.088 [-1.008, +0.878] | -0.194 [-0.381, +0.007] | +0.010 [+0.001, +0.019] | +0.010 [-0.008, +0.027] |

### passive_1: Δ whole margin

| Group | natural | mlp | neurons | random | wmatched |
|---|---|---|---|---|---|
| xtail@head | +2.134 [+1.473, +2.748] | +0.077 [-0.341, +0.490] | +0.179 [+0.100, +0.257] | -0.005 [-0.009, -0.002] | -0.031 [-0.044, -0.018] |
| tail@head | +3.353 [+2.390, +4.261] | +0.084 [-0.322, +0.504] | +0.090 [+0.015, +0.167] | -0.002 [-0.005, +0.000] | -0.020 [-0.029, -0.012] |
| head@head | +3.433 [+1.969, +4.622] | +0.231 [-0.741, +1.226] | -0.026 [-0.189, +0.140] | +0.006 [-0.003, +0.015] | -0.011 [-0.027, +0.008] |
| head@xtail | +3.433 [+1.969, +4.622] | -0.040 [-0.980, +0.913] | -0.194 [-0.380, +0.006] | +0.011 [+0.002, +0.022] | +0.015 [-0.002, +0.031] |

- mlp: share of the Head − XTail " by" deficit (+0.544) closed +0.103 [-1.803, +2.241]; patched gap +0.488 [-0.850, +1.684]
- neurons: share of the Head − XTail " by" deficit (+0.544) closed +0.335 [-2.533, +2.202]; patched gap +0.362 [-1.004, +1.638]
- random: share of the Head − XTail " by" deficit (+0.544) closed -0.008 [-0.055, +0.061]; patched gap +0.549 [-0.833, +1.825]
- wmatched: share of the Head − XTail " by" deficit (+0.544) closed -0.052 [-0.371, +0.412]; patched gap +0.573 [-0.810, +1.845]
- XTail Δ " by", neurons − random: +0.187 [+0.106, +0.268]
- XTail Δ " by", neurons − wmatched: +0.211 [+0.128, +0.293]
- E13 site 8, XTail Δ " by" (same sentences and draws): +0.045 [+0.001, +0.087]; mlp − E13 +0.011 [-0.376, +0.407]; neurons − E13 +0.138 [+0.059, +0.215]

### passive_2: Δ first (".") margin

| Group | natural | mlp | neurons | random | wmatched |
|---|---|---|---|---|---|
| xtail@head | +0.227 [-0.251, +0.684] | -0.389 [-0.677, -0.100] | +0.027 [-0.005, +0.060] | -0.001 [-0.004, +0.002] | +0.002 [-0.004, +0.009] |
| tail@head | +0.234 [-0.209, +0.650] | -0.379 [-0.664, -0.085] | +0.016 [-0.013, +0.045] | -0.005 [-0.008, -0.002] | +0.006 [-0.006, +0.017] |
| head@head | -1.460 [-2.335, -0.724] | +0.655 [+0.148, +1.266] | +0.051 [-0.010, +0.141] | +0.004 [-0.001, +0.008] | +0.005 [-0.013, +0.024] |
| head@xtail | -1.460 [-2.335, -0.724] | +0.974 [+0.457, +1.605] | +0.076 [+0.011, +0.167] | +0.007 [+0.002, +0.011] | +0.001 [-0.014, +0.018] |

### passive_2: Δ whole margin

| Group | natural | mlp | neurons | random | wmatched |
|---|---|---|---|---|---|
| xtail@head | +1.532 [+0.898, +2.196] | -0.389 [-0.677, -0.100] | +0.027 [-0.005, +0.060] | -0.001 [-0.004, +0.002] | +0.002 [-0.004, +0.009] |
| tail@head | +2.532 [+1.661, +3.385] | -0.379 [-0.664, -0.085] | +0.016 [-0.013, +0.045] | -0.005 [-0.008, -0.002] | +0.006 [-0.006, +0.017] |
| head@head | +1.846 [+0.855, +2.767] | +0.655 [+0.148, +1.266] | +0.051 [-0.010, +0.141] | +0.004 [-0.001, +0.008] | +0.005 [-0.013, +0.024] |
| head@xtail | +1.846 [+0.855, +2.767] | +0.974 [+0.457, +1.605] | +0.076 [+0.011, +0.167] | +0.007 [+0.002, +0.011] | +0.001 [-0.014, +0.018] |

- mlp: share of the Head − XTail "." deficit (-1.686) closed +0.231 [+0.071, +0.428]; patched gap -1.298 [-2.203, -0.536]
- neurons: share of the Head − XTail "." deficit (-1.686) closed -0.016 [-0.043, +0.003]; patched gap -1.714 [-2.679, -0.842]
- random: share of the Head − XTail "." deficit (-1.686) closed +0.001 [-0.001, +0.003]; patched gap -1.685 [-2.647, -0.825]
- wmatched: share of the Head − XTail "." deficit (-1.686) closed -0.001 [-0.008, +0.002]; patched gap -1.689 [-2.650, -0.828]
- XTail Δ ".", neurons − random: +0.029 [-0.005, +0.062]
- XTail Δ ".", neurons − wmatched: +0.025 [-0.010, +0.058]
- E13 site 8, XTail Δ "." (same sentences and draws): -0.012 [-0.027, +0.002]; mlp − E13 -0.376 [-0.667, -0.082]; neurons − E13 +0.040 [+0.006, +0.072]

## Pairs outside D11

### passive_1: Δ first (" by") margin

| Group | natural | mlp | neurons | random | wmatched |
|---|---|---|---|---|---|
| xtail@head | +0.562 [+0.018, +1.046] | +0.012 [-0.369, +0.432] | +0.147 [+0.049, +0.248] | -0.004 [-0.007, -0.000] | -0.022 [-0.041, -0.006] |
| tail@head | +1.111 [+0.531, +1.644] | -0.335 [-0.715, +0.063] | +0.015 [-0.099, +0.111] | -0.001 [-0.005, +0.002] | +0.004 [-0.010, +0.019] |
| head@head | +0.941 [+0.026, +1.823] | -0.177 [-0.941, +0.597] | -0.132 [-0.279, +0.002] | -0.001 [-0.006, +0.003] | +0.004 [-0.009, +0.017] |
| head@xtail | +0.941 [+0.026, +1.823] | -0.404 [-1.109, +0.347] | -0.392 [-0.575, -0.219] | +0.001 [-0.003, +0.005] | +0.026 [+0.013, +0.039] |

### passive_1: Δ whole margin

| Group | natural | mlp | neurons | random | wmatched |
|---|---|---|---|---|---|
| xtail@head | +2.136 [+1.307, +3.008] | +0.020 [-0.400, +0.496] | +0.139 [+0.037, +0.243] | -0.005 [-0.008, -0.001] | -0.025 [-0.045, -0.007] |
| tail@head | +2.163 [+1.372, +2.946] | -0.343 [-0.711, +0.047] | +0.011 [-0.104, +0.110] | -0.003 [-0.007, +0.001] | +0.002 [-0.013, +0.016] |
| head@head | +2.704 [+1.859, +3.477] | -0.070 [-0.884, +0.740] | -0.134 [-0.288, +0.005] | -0.003 [-0.007, +0.002] | +0.002 [-0.012, +0.015] |
| head@xtail | +2.704 [+1.859, +3.477] | -0.321 [-1.095, +0.463] | -0.394 [-0.588, -0.209] | +0.000 [-0.004, +0.005] | +0.026 [+0.013, +0.039] |

- mlp: share of the Head − XTail " by" deficit (+0.379) closed +0.032 [-2.884, +3.369]; patched gap +0.367 [-0.553, +1.239]
- neurons: share of the Head − XTail " by" deficit (+0.379) closed +0.389 [-3.041, +3.225]; patched gap +0.232 [-0.736, +1.242]
- random: share of the Head − XTail " by" deficit (+0.379) closed -0.010 [-0.086, +0.086]; patched gap +0.383 [-0.588, +1.410]
- wmatched: share of the Head − XTail " by" deficit (+0.379) closed -0.059 [-0.450, +0.484]; patched gap +0.402 [-0.579, +1.427]
- XTail Δ " by", neurons − random: +0.151 [+0.052, +0.253]
- XTail Δ " by", neurons − wmatched: +0.170 [+0.063, +0.282]

### passive_2: Δ first (".") margin

| Group | natural | mlp | neurons | random | wmatched |
|---|---|---|---|---|---|
| xtail@head | -0.012 [-0.557, +0.535] | -0.192 [-0.545, +0.165] | -0.012 [-0.064, +0.039] | -0.004 [-0.007, -0.001] | -0.015 [-0.037, +0.002] |
| tail@head | +1.024 [+0.437, +1.616] | -0.795 [-1.157, -0.425] | -0.184 [-0.270, -0.104] | -0.006 [-0.008, -0.003] | -0.018 [-0.029, -0.006] |
| head@head | +0.342 [-0.393, +1.170] | -0.382 [-1.003, +0.193] | -0.208 [-0.318, -0.101] | -0.005 [-0.011, +0.000] | -0.017 [-0.032, -0.003] |
| head@xtail | +0.342 [-0.393, +1.170] | -0.105 [-0.753, +0.497] | -0.291 [-0.453, -0.133] | -0.003 [-0.008, +0.003] | -0.017 [-0.032, -0.004] |

### passive_2: Δ whole margin

| Group | natural | mlp | neurons | random | wmatched |
|---|---|---|---|---|---|
| xtail@head | +1.497 [+0.704, +2.245] | -0.192 [-0.545, +0.165] | -0.012 [-0.064, +0.039] | -0.004 [-0.007, -0.001] | -0.015 [-0.037, +0.002] |
| tail@head | +2.676 [+2.090, +3.260] | -0.795 [-1.157, -0.425] | -0.184 [-0.270, -0.104] | -0.006 [-0.008, -0.003] | -0.018 [-0.029, -0.006] |
| head@head | +2.945 [+2.046, +3.915] | -0.382 [-1.003, +0.193] | -0.208 [-0.318, -0.101] | -0.005 [-0.011, +0.000] | -0.017 [-0.032, -0.003] |
| head@xtail | +2.945 [+2.046, +3.915] | -0.105 [-0.753, +0.497] | -0.291 [-0.453, -0.133] | -0.003 [-0.008, +0.003] | -0.017 [-0.032, -0.004] |

- mlp: share of the Head − XTail "." deficit (+0.354) closed -0.543 [-7.406, +6.278]; patched gap +0.547 [-0.239, +1.380]
- neurons: share of the Head − XTail "." deficit (+0.354) closed -0.034 [-0.561, +0.760]; patched gap +0.366 [-0.551, +1.348]
- random: share of the Head − XTail "." deficit (+0.354) closed -0.012 [-0.109, +0.115]; patched gap +0.358 [-0.564, +1.343]
- wmatched: share of the Head − XTail "." deficit (+0.354) closed -0.041 [-0.400, +0.582]; patched gap +0.369 [-0.543, +1.344]
- XTail Δ ".", neurons − random: -0.008 [-0.060, +0.043]
- XTail Δ ".", neurons − wmatched: +0.002 [-0.040, +0.046]

## Declared decisions (all 126 pairs, passive_1, " by")

- **mlp**: XTail Δ by +0.034 [-0.245, +0.325] (threshold 0.115); Head control -0.066 [-0.691, +0.540] → raises rare verbs' by margin: **no**. Closes a real share (E13 rule; by deficit +0.459 [-0.350, +1.257], share +0.074 [-2.632, +2.480]): **no**. Patched Head − XTail gap +0.425 [-0.339, +1.163].
- **neurons**: XTail Δ by +0.165 [+0.103, +0.229] (threshold 0.115); Head control -0.094 [-0.208, +0.019]; neurons − random +0.169 [+0.107, +0.234]; neurons − wmatched +0.190 [+0.124, +0.258] → raises rare verbs' by margin: **no**. Closes a real share (E13 rule; by deficit +0.459 [-0.350, +1.257], share +0.359 [-3.226, +3.554]): **no**. Patched Head − XTail gap +0.294 [-0.497, +1.097].
- **random** (control; reported): XTail Δ by -0.004 [-0.006, -0.002] (threshold 0.115); Head control +0.001 [-0.003, +0.005] → raises rare verbs' by margin: **no**. Closes a real share (E13 rule; by deficit +0.459 [-0.350, +1.257], share -0.009 [-0.083, +0.094]): **no**. Patched Head − XTail gap +0.463 [-0.347, +1.260].
- **wmatched** (control; reported): XTail Δ by -0.025 [-0.036, -0.016] (threshold 0.115); Head control -0.002 [-0.013, +0.010] → raises rare verbs' by margin: **no**. Closes a real share (E13 rule; by deficit +0.459 [-0.350, +1.257], share -0.055 [-0.554, +0.604]): **no**. Patched Head − XTail gap +0.485 [-0.327, +1.279].

