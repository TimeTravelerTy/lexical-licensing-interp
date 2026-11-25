# Split-edit test, frame invariance and side splits (OLMo-2-1B)

Rules, declared and committed before running: `split_edit_replication_rules.md`. Run: `run_split_edit.py` (v2 expanded, job 9562784); tables: `analyze_split_edit.py`; outputs `results/olmo/das_v2/expanded_eval/split_edit/`.

## Checks

| Site | full: active IIA vs stored (max abs diff) | full: passive D(O) / D(by) vs stored | write = read max abs dM | a·u + w = v max err | max abs w·u | u vs direction check max abs |
|---:|---|---|---:|---:|---:|---:|
| 9 | 0.0000 (pass) | n/a | 0 | 6.9e-18 | 2.8e-16 | nan |
| 11 | 0.0000 (pass) | n/a | 0 | 6.9e-18 | 2.5e-16 | nan |
| 12 | 0.0000 (pass) | n/a | 0 | 6.9e-18 | 2.3e-16 | nan |
| 13 | 0.0000 (pass) | n/a | 0 | 6.9e-18 | 2.3e-16 | nan |

B0: no LEACE run for this model. LEACE items: {'prompts': 31500, 'pairs': 126, 'contexts': 126, 'prompts_per_verb': [125]}. C fits: {'0': {'pairs_kept': 108, 'pairs_excluded': 18}, '1': {'pairs_kept': 109, 'pairs_excluded': 17}, '2': {'pairs_kept': 108, 'pairs_excluded': 18}, '3': {'pairs_kept': 108, 'pairs_excluded': 18}, '4': {'pairs_kept': 106, 'pairs_excluded': 20}, 'active': {'pairs_kept': 118}}.

## A. Split edit

### Passives (primary population, bad bases, D = T donor − I donor)

95% CI: three-way cluster bootstrap (2,000 draws, seed 17); shares per draw on shared draws. Class: the declared passive-test rule with the site's δ_s.

| Site | Readout | D full | class | D on | class | D off | class | on / full | off / full | full − on − off | D u_md | class | u_md / full |
|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9 | M | 1.712 [1.576, 1.855] |  | 0.646 [0.578, 0.717] |  | 0.878 [0.796, 0.963] |  | 0.38 [0.36, 0.39] | 0.51 [0.50, 0.53] | 0.188 [0.147, 0.229] | 1.134 [1.016, 1.262] |  | 0.66 [0.62, 0.71] |
| 9 | log P(O) | 1.557 [1.432, 1.684] | RISE | 0.597 [0.535, 0.663] | RISE | 0.793 [0.716, 0.871] | RISE | 0.38 [0.37, 0.40] | 0.51 [0.49, 0.53] | 0.167 [0.126, 0.207] | 1.048 [0.940, 1.166] | RISE | 0.67 [0.63, 0.72] |
| 9 | log P(" by") | 0.474 [0.420, 0.527] | NO RISE | 0.280 [0.246, 0.313] | NO RISE | 0.266 [0.235, 0.297] | NO RISE | 0.59 [0.56, 0.62] | 0.56 [0.53, 0.59] | -0.071 [-0.087, -0.057] | 0.406 [0.352, 0.459] | NO RISE | 0.85 [0.79, 0.92] |
| 9 | log P(".") | -0.329 [-0.382, -0.274] | NO RISE | -0.142 [-0.175, -0.111] | NO RISE | -0.060 [-0.092, -0.028] | NO RISE | 0.43 [0.38, 0.48] | 0.18 [0.10, 0.25] | -0.126 [-0.148, -0.106] | -0.235 [-0.284, -0.189] | NO RISE | 0.72 [0.64, 0.80] |
| 11 | M | 2.619 [2.433, 2.801] |  | 0.786 [0.716, 0.861] |  | 1.651 [1.509, 1.794] |  | 0.30 [0.28, 0.32] | 0.63 [0.61, 0.65] | 0.183 [0.110, 0.252] | 1.327 [1.213, 1.449] |  | 0.51 [0.48, 0.54] |
| 11 | log P(O) | 2.346 [2.185, 2.504] | RISE | 0.720 [0.655, 0.791] | RISE | 1.530 [1.399, 1.661] | RISE | 0.31 [0.29, 0.32] | 0.65 [0.63, 0.67] | 0.096 [0.025, 0.163] | 1.209 [1.105, 1.317] | RISE | 0.52 [0.48, 0.55] |
| 11 | log P(" by") | 0.247 [0.195, 0.297] | NO RISE | 0.289 [0.253, 0.324] | NO RISE | 0.089 [0.058, 0.119] | NO RISE | 1.17 [1.03, 1.37] | 0.36 [0.27, 0.44] | -0.131 [-0.155, -0.110] | 0.424 [0.369, 0.477] | NO RISE | 1.72 [1.51, 2.04] |
| 11 | log P(".") | -0.717 [-0.803, -0.634] | FALL | -0.285 [-0.331, -0.242] | NO RISE | -0.267 [-0.300, -0.234] | NO RISE | 0.40 [0.37, 0.43] | 0.37 [0.34, 0.40] | -0.165 [-0.193, -0.139] | -0.446 [-0.519, -0.378] | NO RISE | 0.62 [0.57, 0.68] |
| 12 | M | 4.549 [4.269, 4.821] |  | 1.133 [1.041, 1.226] |  | 2.804 [2.593, 3.006] |  | 0.25 [0.24, 0.26] | 0.62 [0.60, 0.63] | 0.612 [0.517, 0.709] | 2.177 [1.998, 2.359] |  | 0.48 [0.45, 0.50] |
| 12 | log P(O) | 3.616 [3.425, 3.788] | RISE | 1.056 [0.969, 1.145] | RISE | 2.534 [2.351, 2.708] | RISE | 0.29 [0.28, 0.31] | 0.70 [0.68, 0.72] | 0.026 [-0.094, 0.138] | 1.972 [1.810, 2.131] | RISE | 0.55 [0.51, 0.58] |
| 12 | log P(" by") | -0.783 [-0.919, -0.648] | FALL | 0.144 [0.108, 0.179] | NO RISE | -0.241 [-0.291, -0.195] | NO RISE | -0.18 [-0.25, -0.13] | 0.31 [0.28, 0.34] | -0.686 [-0.778, -0.594] | 0.140 [0.075, 0.204] | NO RISE | -0.18 [-0.29, -0.09] |
| 12 | log P(".") | -1.579 [-1.727, -1.433] | FALL | -0.390 [-0.436, -0.348] | NO RISE | -0.587 [-0.639, -0.535] | unresolved | 0.25 [0.23, 0.27] | 0.37 [0.36, 0.39] | -0.602 [-0.686, -0.521] | -0.757 [-0.854, -0.667] | FALL | 0.48 [0.44, 0.52] |
| 13 | M | 6.008 [5.624, 6.363] |  | 1.588 [1.473, 1.701] |  | 4.714 [4.344, 5.063] |  | 0.26 [0.25, 0.27] | 0.78 [0.77, 0.80] | -0.293 [-0.430, -0.165] | 3.222 [3.007, 3.441] |  | 0.54 [0.51, 0.56] |
| 13 | log P(O) | 4.077 [3.875, 4.262] | RISE | 1.446 [1.342, 1.548] | RISE | 3.699 [3.471, 3.905] | RISE | 0.35 [0.34, 0.37] | 0.91 [0.88, 0.93] | -1.068 [-1.229, -0.904] | 2.738 [2.564, 2.913] | RISE | 0.67 [0.64, 0.70] |
| 13 | log P(" by") | -1.843 [-2.059, -1.633] | FALL | 0.100 [0.071, 0.126] | NO RISE | -1.186 [-1.363, -1.021] | FALL | -0.05 [-0.07, -0.04] | 0.64 [0.62, 0.67] | -0.757 [-0.825, -0.689] | -0.106 [-0.175, -0.038] | NO RISE | 0.06 [0.02, 0.09] |
| 13 | log P(".") | -2.564 [-2.785, -2.331] | FALL | -0.662 [-0.730, -0.598] | unresolved | -1.160 [-1.309, -1.012] | FALL | 0.26 [0.24, 0.28] | 0.45 [0.43, 0.48] | -0.742 [-0.801, -0.683] | -1.517 [-1.679, -1.360] | FALL | 0.59 [0.55, 0.63] |

### Passive outcome per site and condition

| Site | full | on | umd | u_pas |
|---:|---|---|---|---|
| 9 | surface (object next) | surface (object next) | surface (object next) | nothing moves (active-specific) |
| 11 | surface (object next) | surface (object next) | surface (object next) | nothing moves (active-specific) |
| 12 | surface (object next) | surface (object next) | surface (object next) | nothing moves (active-specific) |
| 13 | surface (object next) | surface (object next) | surface (object next) | nothing moves (active-specific) |

### Held-out actives

IIA: cross-class held-out IIA, mean [min, max] over the 15 bases. Δ log P(O): I base ← T source, mean over pairs, 95% CI over pairs (2,000 draws, seed 0); share = condition / full per draw.

| Site | Condition | IIA | same-class IIA | Δ log P(O) | share of full |
|---:|---|---|---:|---|---|
| 9 | full | 0.836 [0.696, 0.923] | 0.991 | 2.706 [2.337, 3.073] |  |
| 9 | on | 0.088 [0.000, 0.278] | 0.994 | 1.317 [1.100, 1.546] | 0.49 [0.45, 0.52] |
| 9 | off | 0.241 [0.062, 0.421] | 0.993 | 1.691 [1.432, 1.951] | 0.62 [0.59, 0.66] |
| 9 | umd | 0.382 [0.266, 0.474] | 0.992 | 1.944 [1.655, 2.234] | 0.72 [0.68, 0.76] |
| 9 | u_pas | 0.023 [0.000, 0.083] | 0.983 | 0.144 [0.086, 0.203] |  |
| 9 | full − on − off | | | -0.301 [-0.508, -0.115] | |
| 11 | full | 0.945 [0.867, 1.000] | 0.989 | 2.928 [2.540, 3.304] |  |
| 11 | on | 0.143 [0.026, 0.343] | 0.995 | 1.494 [1.259, 1.736] | 0.51 [0.48, 0.54] |
| 11 | off | 0.423 [0.206, 0.599] | 0.993 | 2.013 [1.718, 2.303] | 0.69 [0.65, 0.72] |
| 11 | umd | 0.539 [0.374, 0.635] | 0.993 | 2.172 [1.869, 2.480] | 0.74 [0.71, 0.77] |
| 11 | u_pas | 0.023 [0.000, 0.081] | 0.985 | 0.068 [0.026, 0.110] |  |
| 11 | full − on − off | | | -0.579 [-0.797, -0.381] | |
| 12 | full | 0.976 [0.946, 1.000] | 0.987 | 3.126 [2.736, 3.517] |  |
| 12 | on | 0.181 [0.067, 0.397] | 0.995 | 1.477 [1.249, 1.712] | 0.47 [0.44, 0.50] |
| 12 | off | 0.696 [0.566, 0.806] | 0.995 | 2.486 [2.134, 2.831] | 0.80 [0.76, 0.83] |
| 12 | umd | 0.661 [0.509, 0.752] | 0.993 | 2.314 [1.996, 2.630] | 0.74 [0.71, 0.77] |
| 12 | u_pas | 0.020 [0.000, 0.081] | 0.986 | 0.045 [0.009, 0.082] |  |
| 12 | full − on − off | | | -0.837 [-1.072, -0.621] | |
| 13 | full | 0.990 [0.954, 1.000] | 0.984 | 3.348 [2.962, 3.725] |  |
| 13 | on | 0.226 [0.081, 0.421] | 0.995 | 1.444 [1.231, 1.667] | 0.43 [0.40, 0.46] |
| 13 | off | 0.889 [0.760, 0.980] | 0.994 | 3.190 [2.803, 3.561] | 0.95 [0.94, 0.96] |
| 13 | umd | 0.772 [0.671, 0.891] | 0.992 | 2.419 [2.097, 2.745] | 0.72 [0.69, 0.75] |
| 13 | u_pas | 0.015 [0.000, 0.056] | 0.987 | 0.026 [-0.001, 0.055] |  |
| 13 | full − on − off | | | -1.287 [-1.513, -1.075] | |

### Push size and class separation (mean over the 15 bases)

a = v·u, b = ‖w‖. Push: median over rows of |δ·a| / sd(h·u) (on) and |δ·b| / sd(h·ŵ) (off); actives: cross-class I ← T held-out swaps, sd over the 406 items; passives: plan rows, sd over primary good and bad prompts. Separation: standardized mean difference T − I (actives) / good − bad (passives) along u and ŵ.

| Site | a | b | act on | act off | act off > 2 sd | pas on | pas off | pas off > 2 sd | sep act u | sep act ŵ | sep pas u | sep pas ŵ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | 0.751 | 0.660 | 1.07 | 4.02 | 0.98 | 1.67 | 2.25 | 0.50 | 6.41 | 0.31 | 1.86 | 0.29 |
| 11 | 0.753 | 0.658 | 1.09 | 4.58 | 0.99 | 1.89 | 2.81 | 0.52 | 7.36 | 0.27 | 2.00 | -0.10 |
| 12 | 0.725 | 0.689 | 1.02 | 5.66 | 1.00 | 1.95 | 3.55 | 0.53 | 7.82 | 0.27 | 1.76 | -0.19 |
| 13 | 0.705 | 0.709 | 0.96 | 6.26 | 1.00 | 2.04 | 4.16 | 0.54 | 7.87 | 0.24 | 1.28 | -0.19 |

## B. Frame invariance of the class axis

u_act: T − I over the 406 active items; u_pas: T − I over the LEACE items (126 pairs × 125 contexts); both from per-verb means, same bf16 runner. Split-half: 200 random halves of the pairs, mean [2.5, 97.5 pct]. Sign-flip reference: 1,000 draws, |cos(u_act, u_pas,flip)| mean / 97.5th pct. FI = cos / sqrt(r_act r_pas) (approximate; an upper bound).

| Site | cos(u_act, u_pas) | split-half act | split-half pas | sign-flip ref | FI | |cos(u_pas, LEACE edit)| | |cos(v, u_pas)| | |cos(u_md fold, u_pas)| |
|---:|---:|---|---|---|---:|---:|---|---:|
| 0 | 0.373 | 0.276 [0.210, 0.325] | 0.220 [0.135, 0.294] | 0.085 / 0.230 | 1.51 | nan |  |  |
| 1 | 0.474 | 0.373 [0.323, 0.410] | 0.494 [0.449, 0.530] | 0.077 / 0.209 | 1.11 | nan |  |  |
| 2 | 0.435 | 0.417 [0.366, 0.461] | 0.602 [0.569, 0.632] | 0.079 / 0.208 | 0.87 | nan |  |  |
| 3 | 0.473 | 0.471 [0.417, 0.513] | 0.656 [0.616, 0.691] | 0.090 / 0.229 | 0.85 | nan |  |  |
| 4 | 0.399 | 0.535 [0.475, 0.583] | 0.733 [0.699, 0.763] | 0.087 / 0.223 | 0.64 | nan |  |  |
| 5 | 0.382 | 0.588 [0.532, 0.637] | 0.764 [0.727, 0.797] | 0.086 / 0.215 | 0.57 | nan |  |  |
| 6 | 0.330 | 0.617 [0.557, 0.667] | 0.783 [0.741, 0.817] | 0.080 / 0.204 | 0.48 | nan |  |  |
| 7 | 0.312 | 0.622 [0.560, 0.680] | 0.804 [0.754, 0.839] | 0.082 / 0.214 | 0.44 | nan |  |  |
| 8 | 0.254 | 0.660 [0.601, 0.716] | 0.813 [0.761, 0.850] | 0.073 / 0.189 | 0.35 | nan |  |  |
| 9 | 0.252 | 0.679 [0.620, 0.736] | 0.806 [0.749, 0.846] | 0.073 / 0.188 | 0.34 | nan | 0.247 [0.242, 0.256] | 0.249 |
| 10 | 0.265 | 0.712 [0.651, 0.767] | 0.810 [0.752, 0.851] | 0.078 / 0.200 | 0.35 | nan |  |  |
| 11 | 0.257 | 0.743 [0.683, 0.797] | 0.835 [0.780, 0.874] | 0.083 / 0.211 | 0.33 | nan | 0.218 [0.209, 0.232] | 0.256 |
| 12 | 0.240 | 0.769 [0.713, 0.818] | 0.856 [0.811, 0.890] | 0.084 / 0.210 | 0.30 | nan | 0.194 [0.181, 0.205] | 0.239 |
| 13 | 0.182 | 0.797 [0.746, 0.836] | 0.848 [0.800, 0.884] | 0.077 / 0.190 | 0.22 | nan | 0.152 [0.140, 0.181] | 0.183 |
| 14 | 0.150 | 0.819 [0.773, 0.855] | 0.824 [0.769, 0.867] | 0.076 / 0.195 | 0.18 | nan |  |  |
| 15 | 0.085 | 0.863 [0.827, 0.894] | 0.821 [0.769, 0.864] | 0.094 / 0.248 | 0.10 | nan |  |  |

## C. Patching along u_pas (cross-fitted; exploratory)

Passive D for u_pas is in the A passive CSV (`summary_passive.csv`, columns `u_pas_*`, `share_u_pas_*` = u_pas / full v); actives in `summary_active.csv` (condition u_pas). Stored v and u_md (fold) rows: `../das_direction_check/summary_passive_patching.csv`.

| Site | Readout | D full v | D u_pas | class | u_pas / full v |
|---:|---|---|---|---|---|
| 9 | M | 1.712 [1.576, 1.855] | 0.002 [-0.014, 0.020] |  | 0.00 [-0.01, 0.01] |
| 9 | log P(O) | 1.557 [1.432, 1.684] | 0.006 [-0.010, 0.022] | NO RISE | 0.00 [-0.01, 0.01] |
| 9 | log P(" by") | 0.474 [0.420, 0.527] | 0.125 [0.086, 0.164] | NO RISE | 0.26 [0.18, 0.36] |
| 9 | log P(".") | -0.329 [-0.382, -0.274] | 0.066 [0.042, 0.092] | NO RISE | -0.20 [-0.30, -0.12] |
| 11 | M | 2.619 [2.433, 2.801] | 0.001 [-0.017, 0.020] |  | 0.00 [-0.01, 0.01] |
| 11 | log P(O) | 2.346 [2.185, 2.504] | 0.005 [-0.012, 0.024] | NO RISE | 0.00 [-0.00, 0.01] |
| 11 | log P(" by") | 0.247 [0.195, 0.297] | 0.142 [0.100, 0.184] | NO RISE | 0.57 [0.38, 0.82] |
| 11 | log P(".") | -0.717 [-0.803, -0.634] | 0.087 [0.058, 0.117] | NO RISE | -0.12 [-0.17, -0.08] |
| 12 | M | 4.549 [4.269, 4.821] | -0.014 [-0.029, -0.001] |  | -0.00 [-0.01, -0.00] |
| 12 | log P(O) | 3.616 [3.425, 3.788] | -0.009 [-0.023, 0.004] | NO RISE | -0.00 [-0.01, 0.00] |
| 12 | log P(" by") | -0.783 [-0.919, -0.648] | 0.132 [0.092, 0.172] | NO RISE | -0.17 [-0.22, -0.12] |
| 12 | log P(".") | -1.579 [-1.727, -1.433] | 0.084 [0.057, 0.113] | NO RISE | -0.05 [-0.07, -0.04] |
| 13 | M | 6.008 [5.624, 6.363] | 0.003 [-0.006, 0.014] |  | 0.00 [-0.00, 0.00] |
| 13 | log P(O) | 4.077 [3.875, 4.262] | 0.004 [-0.006, 0.016] | NO RISE | 0.00 [-0.00, 0.00] |
| 13 | log P(" by") | -1.843 [-2.059, -1.633] | 0.148 [0.086, 0.208] | NO RISE | -0.08 [-0.11, -0.05] |
| 13 | log P(".") | -2.564 [-2.785, -2.331] | 0.101 [0.058, 0.144] | NO RISE | -0.04 [-0.06, -0.02] |

## Declared readings

```
{
  "outcomes": {
    "full": {
      "9": "surface (object next)",
      "11": "surface (object next)",
      "12": "surface (object next)",
      "13": "surface (object next)"
    },
    "on": {
      "9": "surface (object next)",
      "11": "surface (object next)",
      "12": "surface (object next)",
      "13": "surface (object next)"
    },
    "umd": {
      "9": "surface (object next)",
      "11": "surface (object next)",
      "12": "surface (object next)",
      "13": "surface (object next)"
    },
    "u_pas": {
      "9": "nothing moves (active-specific)",
      "11": "nothing moves (active-specific)",
      "12": "nothing moves (active-specific)",
      "13": "nothing moves (active-specific)"
    }
  },
  "R-A1": {
    "sites": {
      "9": {
        "off_share_O": 0.5092626214027405,
        "class_on_O": "RISE",
        "held": false
      },
      "11": {
        "off_share_O": 0.6520841717720032,
        "class_on_O": "RISE",
        "held": false
      },
      "12": {
        "off_share_O": 0.7007738351821899,
        "class_on_O": "RISE",
        "held": false
      },
      "13": {
        "off_share_O": 0.9071974754333496,
        "class_on_O": "RISE",
        "held": false
      }
    },
    "held_at_rule_sites": false,
    "rule_sites_applicable": [
      9,
      11
    ],
    "held_all_applicable_sites": false,
    "held_n": "0 / 4"
  },
  "R-A2": {
    "sites": {},
    "held_at_rule_sites": "n/a",
    "rule_sites_applicable": [],
    "held_all_applicable_sites": false,
    "held_n": "0 / 0"
  },
  "S1": {
    "full": {
      "abstract": [],
      "surface": [
        9,
        11,
        12,
        13
      ],
      "pass": false
    },
    "on": {
      "abstract": [],
      "surface": [
        9,
        11,
        12,
        13
      ],
      "pass": false
    },
    "umd": {
      "abstract": [],
      "surface": [
        9,
        11,
        12,
        13
      ],
      "pass": false
    },
    "u_pas": {
      "abstract": [],
      "surface": [],
      "pass": false
    }
  },
  "T1": {
    "full": {
      "by_class": {
        "9": "NO RISE",
        "11": "NO RISE"
      },
      "pass": false
    },
    "on": {
      "by_class": {
        "9": "NO RISE",
        "11": "NO RISE"
      },
      "pass": false
    },
    "umd": {
      "by_class": {
        "9": "NO RISE",
        "11": "NO RISE"
      },
      "pass": false
    },
    "u_pas": {
      "by_class": {
        "9": "NO RISE",
        "11": "NO RISE"
      },
      "pass": false
    }
  },
  "B1": {
    "held": false,
    "sites_failing": [
      13,
      14,
      15
    ]
  },
  "B2": {
    "9": {
      "cos": 0.251973114817709,
      "split_half_pas": 0.8055065703076665,
      "below": true
    },
    "11": {
      "cos": 0.2570038579142273,
      "split_half_pas": 0.8348551506569958,
      "below": true
    }
  },
  "FI_all": {
    "0": 1.514,
    "1": 1.105,
    "2": 0.867,
    "3": 0.851,
    "4": 0.637,
    "5": 0.57,
    "6": 0.475,
    "7": 0.442,
    "8": 0.348,
    "9": 0.341,
    "10": 0.349,
    "11": 0.326,
    "12": 0.295,
    "13": 0.221,
    "14": 0.183,
    "15": 0.101
  },
  "FI": {
    "values": {
      "9": 0.3406791219907939,
      "11": 0.3263853010036358
    },
    "decision": "low: 'one direction every construction reads'"
  }
}
```
