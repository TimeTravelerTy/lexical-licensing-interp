# A1. Split-half reliability of per-pair measures

Spec: `round3_plan.md`, A1. Script: `analyze_reliability.py`. 64 primary pairs. Halves stratified by context band; R = Spearman-Brown of the half correlation; median over 1000 random splits; 95% CI from a pair bootstrap (2000 draws, fresh split per draw). R measures stability across sampled contexts.

## Reliability

| Measure | Variant | Pairs | Units per pair | R (Pearson) | range over splits | 95% CI (pairs) | R (Spearman) |
|---|---|---:|---:|---:|---|---|---:|
| passive_gap_s4 | raw | 64 | 125 | 1.000 | [1.000, 1.000] | [1.000, 1.000] | 1.000 |
| passive_gap_s4 | zipf_band_residual | 64 | 125 | 1.000 | [1.000, 1.000] | [0.999, 1.000] | 0.999 |
| active_gap_s4 | raw | 64 | 7 | 0.996 | [0.995, 0.997] | [0.993, 0.998] | 0.993 |
| passive_gap_s6 | raw | 64 | 125 | 1.000 | [0.999, 1.000] | [0.999, 1.000] | 0.999 |
| passive_gap_s6 | zipf_band_residual | 64 | 125 | 0.999 | [0.999, 1.000] | [0.999, 1.000] | 0.999 |
| active_gap_s6 | raw | 64 | 7 | 0.991 | [0.987, 0.994] | [0.983, 0.995] | 0.988 |
| passive_gap_s8 | raw | 64 | 125 | 0.999 | [0.999, 0.999] | [0.998, 1.000] | 0.999 |
| passive_gap_s8 | zipf_band_residual | 64 | 125 | 0.999 | [0.998, 0.999] | [0.998, 0.999] | 0.998 |
| active_gap_s8 | raw | 64 | 7 | 0.988 | [0.982, 0.992] | [0.977, 0.994] | 0.986 |
| passive_gap_s10 | raw | 64 | 125 | 0.999 | [0.999, 0.999] | [0.998, 0.999] | 0.999 |
| passive_gap_s10 | zipf_band_residual | 64 | 125 | 0.999 | [0.998, 0.999] | [0.998, 0.999] | 0.998 |
| active_gap_s10 | raw | 64 | 7 | 0.988 | [0.981, 0.992] | [0.975, 0.994] | 0.985 |
| passive_gap_s12 | raw | 64 | 125 | 0.999 | [0.998, 0.999] | [0.998, 0.999] | 0.997 |
| passive_gap_s12 | zipf_band_residual | 64 | 125 | 0.999 | [0.998, 0.999] | [0.997, 0.999] | 0.997 |
| active_gap_s12 | raw | 64 | 7 | 0.986 | [0.979, 0.992] | [0.972, 0.994] | 0.985 |
| passive_gap_s14 | raw | 64 | 125 | 0.999 | [0.998, 0.999] | [0.997, 0.999] | 0.998 |
| passive_gap_s14 | zipf_band_residual | 64 | 125 | 0.998 | [0.997, 0.999] | [0.997, 0.999] | 0.997 |
| active_gap_s14 | raw | 64 | 7 | 0.986 | [0.980, 0.992] | [0.974, 0.993] | 0.987 |
| passive_gap_s16 | raw | 64 | 125 | 0.998 | [0.997, 0.999] | [0.997, 0.999] | 0.997 |
| passive_gap_s16 | zipf_band_residual | 64 | 125 | 0.998 | [0.997, 0.998] | [0.996, 0.999] | 0.996 |
| active_gap_s16 | raw | 64 | 7 | 0.986 | [0.979, 0.991] | [0.972, 0.993] | 0.983 |
| passive_gap_s17 | raw | 64 | 125 | 0.997 | [0.996, 0.998] | [0.995, 0.999] | 0.996 |
| passive_gap_s17 | zipf_band_residual | 64 | 125 | 0.997 | [0.995, 0.998] | [0.994, 0.999] | 0.996 |
| active_gap_s17 | raw | 64 | 7 | 0.985 | [0.978, 0.990] | [0.970, 0.993] | 0.983 |
| by_pref | raw | 64 | 125 | 0.997 | [0.996, 0.998] | [0.995, 0.999] | 0.996 |
| by_margin_released | raw | 59 | 300 | 0.999 | [0.998, 0.999] | [0.998, 0.999] | 0.997 |
| lp_margin | raw | 59 | 125 | 0.989 | [0.986, 0.993] | [0.980, 0.995] | 0.987 |

`zipf_band_residual`: both halves residualized on participle Zipf and band across pairs before correlating (reliability of the part of the gap that frequency does not explain).

## Consistency of the passive gap across basis splits

Per-pair passive gap computed with one split seed's bases only (cross-fitted within the split).

| Measure | r(0,1) | r(0,2) | r(1,2) | mean r | Spearman-Brown for 3 |
|---|---:|---:|---:|---:|---:|
| passive_gap_s4 | 0.991 | 0.996 | 0.989 | 0.992 | 0.997 |
| passive_gap_s6 | 0.994 | 0.993 | 0.993 | 0.993 | 0.998 |
| passive_gap_s8 | 0.993 | 0.995 | 0.992 | 0.993 | 0.998 |
| passive_gap_s10 | 0.993 | 0.993 | 0.993 | 0.993 | 0.998 |
| passive_gap_s12 | 0.992 | 0.994 | 0.991 | 0.992 | 0.997 |
| passive_gap_s14 | 0.990 | 0.990 | 0.992 | 0.991 | 0.997 |
| passive_gap_s16 | 0.992 | 0.988 | 0.992 | 0.991 | 0.997 |
| passive_gap_s17 | 0.979 | 0.982 | 0.986 | 0.983 | 0.994 |

## Passive gap vs behaviour, disattenuated (Pearson primary)

Reliabilities on each test's exact pair population. Declared rule: measurement-limited if either R < 0.6; significance from the CI of r observed; no moderate link if the whole disattenuated CI lies in [-0.3, 0.3]; otherwise no detectable link, moderate not excluded. The single-prompt " by" preference shares contexts with the gap, so its errors may correlate with the gap's.

| Site | Behaviour | Pairs | r observed [95% CI] | R gap | R beh. | ceiling | r disattenuated [95% CI] | Verdict | Spearman disatt. [CI] |
|---:|---|---:|---|---:|---:|---:|---|---|---|
| 4 | single-prompt " by" preference | 64 | +0.20 [-0.02, +0.42] | 1.00 | 1.00 | 1.00 | +0.20 [-0.02, +0.42] | no detectable link; moderate not excluded | +0.20 [-0.05, +0.43] |
| 4 | released by margin (task 0) | 59 | +0.17 [-0.05, +0.40] | 1.00 | 1.00 | 1.00 | +0.17 [-0.05, +0.40] | no detectable link; moderate not excluded | +0.24 [-0.00, +0.45] |
| 4 | curated passive_2 LP margin | 59 | +0.22 [+0.00, +0.43] | 1.00 | 0.99 | 0.99 | +0.22 [+0.00, +0.44] | significant positive link | +0.24 [-0.01, +0.46] |
| 6 | single-prompt " by" preference | 64 | +0.06 [-0.16, +0.30] | 1.00 | 1.00 | 1.00 | +0.06 [-0.17, +0.30] | no detectable link; moderate not excluded | +0.05 [-0.18, +0.29] |
| 6 | released by margin (task 0) | 59 | +0.01 [-0.24, +0.28] | 1.00 | 1.00 | 1.00 | +0.01 [-0.24, +0.28] | no moderate link | +0.05 [-0.22, +0.29] |
| 6 | curated passive_2 LP margin | 59 | +0.13 [-0.12, +0.39] | 1.00 | 0.99 | 0.99 | +0.13 [-0.12, +0.39] | no detectable link; moderate not excluded | +0.11 [-0.15, +0.37] |
| 8 | single-prompt " by" preference | 64 | +0.10 [-0.15, +0.35] | 1.00 | 1.00 | 1.00 | +0.10 [-0.15, +0.35] | no detectable link; moderate not excluded | +0.11 [-0.14, +0.36] |
| 8 | released by margin (task 0) | 59 | +0.02 [-0.27, +0.30] | 1.00 | 1.00 | 1.00 | +0.02 [-0.27, +0.30] | no detectable link; moderate not excluded | +0.08 [-0.20, +0.34] |
| 8 | curated passive_2 LP margin | 59 | +0.04 [-0.19, +0.28] | 1.00 | 0.99 | 0.99 | +0.04 [-0.19, +0.28] | no moderate link | +0.05 [-0.20, +0.31] |
| 10 | single-prompt " by" preference | 64 | +0.11 [-0.13, +0.34] | 1.00 | 1.00 | 1.00 | +0.11 [-0.13, +0.34] | no detectable link; moderate not excluded | +0.11 [-0.13, +0.34] |
| 10 | released by margin (task 0) | 59 | +0.04 [-0.24, +0.31] | 1.00 | 1.00 | 1.00 | +0.04 [-0.24, +0.31] | no detectable link; moderate not excluded | +0.08 [-0.19, +0.33] |
| 10 | curated passive_2 LP margin | 59 | +0.05 [-0.16, +0.28] | 1.00 | 0.99 | 0.99 | +0.05 [-0.16, +0.28] | no moderate link | +0.06 [-0.19, +0.30] |
| 12 | single-prompt " by" preference | 64 | +0.11 [-0.13, +0.34] | 1.00 | 1.00 | 1.00 | +0.11 [-0.13, +0.34] | no detectable link; moderate not excluded | +0.13 [-0.11, +0.36] |
| 12 | released by margin (task 0) | 59 | +0.05 [-0.22, +0.32] | 1.00 | 1.00 | 1.00 | +0.05 [-0.22, +0.32] | no detectable link; moderate not excluded | +0.11 [-0.15, +0.35] |
| 12 | curated passive_2 LP margin | 59 | +0.01 [-0.21, +0.24] | 1.00 | 0.99 | 0.99 | +0.01 [-0.21, +0.24] | no moderate link | +0.02 [-0.22, +0.27] |
| 14 | single-prompt " by" preference | 64 | +0.13 [-0.11, +0.37] | 1.00 | 1.00 | 1.00 | +0.13 [-0.11, +0.37] | no detectable link; moderate not excluded | +0.18 [-0.06, +0.40] |
| 14 | released by margin (task 0) | 59 | +0.07 [-0.20, +0.33] | 1.00 | 1.00 | 1.00 | +0.07 [-0.20, +0.33] | no detectable link; moderate not excluded | +0.14 [-0.13, +0.38] |
| 14 | curated passive_2 LP margin | 59 | +0.03 [-0.19, +0.27] | 1.00 | 0.99 | 0.99 | +0.03 [-0.19, +0.27] | no moderate link | +0.03 [-0.21, +0.28] |
| 16 | single-prompt " by" preference | 64 | +0.13 [-0.12, +0.36] | 1.00 | 1.00 | 1.00 | +0.13 [-0.12, +0.36] | no detectable link; moderate not excluded | +0.20 [-0.03, +0.41] |
| 16 | released by margin (task 0) | 59 | +0.09 [-0.17, +0.36] | 1.00 | 1.00 | 1.00 | +0.09 [-0.17, +0.36] | no detectable link; moderate not excluded | +0.18 [-0.07, +0.41] |
| 16 | curated passive_2 LP margin | 59 | +0.00 [-0.22, +0.24] | 1.00 | 0.99 | 0.99 | +0.00 [-0.22, +0.24] | no moderate link | +0.06 [-0.18, +0.31] |
| 17 | single-prompt " by" preference | 64 | +0.10 [-0.18, +0.37] | 1.00 | 1.00 | 1.00 | +0.10 [-0.18, +0.37] | no detectable link; moderate not excluded | +0.18 [-0.07, +0.42] |
| 17 | released by margin (task 0) | 59 | +0.03 [-0.24, +0.32] | 1.00 | 1.00 | 1.00 | +0.03 [-0.24, +0.32] | no detectable link; moderate not excluded | +0.14 [-0.14, +0.40] |
| 17 | curated passive_2 LP margin | 59 | -0.06 [-0.29, +0.16] | 1.00 | 0.99 | 0.99 | -0.06 [-0.29, +0.16] | no moderate link | -0.03 [-0.27, +0.22] |

Verdict counts (Pearson): no detectable link; moderate not excluded: 16, no moderate link: 7, significant positive link: 1
