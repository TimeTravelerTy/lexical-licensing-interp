# V2 Head-to-Low Results

## Baseline Sanity

| regime | subtask | side | n | mean_expected_logodds | expected_preferred_rate |
| --- | --- | --- | --- | --- | --- |
| head | causative | bad | 440 | 1.2166 | 0.7318 |
| head | causative | good | 620 | 3.8668 | 0.8661 |
| head | inchoative | bad | 320 | 5.6861 | 1.0000 |
| head | inchoative | good | 620 | -1.5769 | 0.2145 |
| low | causative | bad | 460 | 1.0096 | 0.7022 |
| low | causative | good | 280 | 4.3433 | 0.9464 |
| low | inchoative | bad | 540 | 4.2734 | 0.9981 |
| low | inchoative | good | 280 | -1.2810 | 0.2286 |

## Red/Blue Baseline

Prompt text is unchanged; this scores `log p(" red") - log p(" blue")` and ignores the original targets.

| regime | subtask | side | n | mean_expected_logodds | expected_preferred_rate |
| --- | --- | --- | --- | --- | --- |
| head | causative | bad | 440 | 0.4727 | 0.6636 |
| head | causative | good | 620 | 0.8318 | 0.7565 |
| head | inchoative | bad | 320 | 0.7198 | 0.8156 |
| head | inchoative | good | 620 | 0.7900 | 0.7839 |
| low | causative | bad | 460 | 0.8616 | 0.8174 |
| low | causative | good | 280 | 0.9863 | 0.8821 |
| low | inchoative | bad | 540 | 0.7662 | 0.8537 |
| low | inchoative | good | 280 | 0.6777 | 0.8286 |

## Attribution Patching Top Sites

| regime | subtask | direction | site | n | mean_attribution | mean_abs_attribution |
| --- | --- | --- | --- | --- | --- | --- |
| head | causative | bad_to_good | resid_post_layer_18 | 440 | 5.1649 | 5.2821 |
| head | causative | bad_to_good | resid_post_layer_23 | 440 | 5.1021 | 5.2212 |
| head | causative | bad_to_good | resid_post_layer_22 | 440 | 4.8655 | 4.9363 |
| head | causative | good_to_bad | resid_post_layer_19 | 440 | 5.2060 | 5.4788 |
| head | causative | good_to_bad | resid_post_layer_23 | 440 | 5.1021 | 5.2212 |
| head | causative | good_to_bad | resid_post_layer_20 | 440 | 4.1248 | 4.5243 |
| low | causative | bad_to_good | resid_post_layer_23 | 280 | 5.1975 | 5.3774 |
| low | causative | bad_to_good | resid_post_layer_16 | 280 | 5.0254 | 5.2807 |
| low | causative | bad_to_good | resid_post_layer_18 | 280 | 5.0992 | 5.2406 |
| low | causative | good_to_bad | resid_post_layer_20 | 280 | 5.7629 | 5.9535 |
| low | causative | good_to_bad | resid_post_layer_19 | 280 | 5.4744 | 5.8179 |
| low | causative | good_to_bad | resid_post_layer_21 | 280 | 5.4332 | 5.7213 |
| head | inchoative | bad_to_good | resid_post_layer_21 | 320 | 4.5578 | 4.7628 |
| head | inchoative | bad_to_good | resid_post_layer_20 | 320 | 4.1572 | 4.3416 |
| head | inchoative | bad_to_good | resid_post_layer_23 | 320 | 4.1311 | 4.1550 |
| head | inchoative | good_to_bad | resid_post_layer_18 | 320 | 4.4452 | 4.4578 |
| head | inchoative | good_to_bad | resid_post_layer_16 | 320 | 4.3324 | 4.3371 |
| head | inchoative | good_to_bad | resid_post_layer_14 | 320 | 4.2373 | 4.2578 |
| low | inchoative | bad_to_good | resid_post_layer_21 | 280 | 3.4757 | 3.6707 |
| low | inchoative | bad_to_good | resid_post_layer_15 | 280 | 3.3065 | 3.5747 |
| low | inchoative | bad_to_good | resid_post_layer_20 | 280 | 2.9300 | 3.2659 |
| low | inchoative | good_to_bad | resid_post_layer_18 | 280 | 3.5349 | 3.6281 |
| low | inchoative | good_to_bad | resid_post_layer_22 | 280 | 3.1240 | 3.2655 |
| low | inchoative | good_to_bad | resid_post_layer_19 | 280 | 3.1161 | 3.2627 |

## Exact Patching Top Sites

| regime | subtask | direction | site | n | mean_exact_effect | mean_normalized_effect |
| --- | --- | --- | --- | --- | --- | --- |
| head | causative | bad_to_good | resid_post_layer_16 | 440 | 5.0999 | 1.0000 |
| head | causative | bad_to_good | resid_post_layer_17 | 440 | 5.0999 | 1.0000 |
| head | causative | bad_to_good | resid_post_layer_18 | 440 | 5.0999 | 1.0000 |
| head | causative | good_to_bad | resid_post_layer_16 | 440 | 5.0999 | 1.0000 |
| head | causative | good_to_bad | resid_post_layer_17 | 440 | 5.0999 | 1.0000 |
| head | causative | good_to_bad | resid_post_layer_18 | 440 | 5.0999 | 1.0000 |
| low | causative | bad_to_good | resid_post_layer_23 | 280 | 5.1960 | 1.0000 |
| low | causative | bad_to_good | resid_post_layer_22 | 280 | 5.1940 | 0.9999 |
| low | causative | bad_to_good | resid_post_layer_20 | 280 | 5.1858 | 1.0077 |
| low | causative | good_to_bad | resid_post_layer_16 | 280 | 5.2195 | 1.0059 |
| low | causative | good_to_bad | resid_post_layer_17 | 280 | 5.2138 | 1.0072 |
| low | causative | good_to_bad | resid_post_layer_19 | 280 | 5.2106 | 1.0049 |
| head | inchoative | bad_to_good | resid_post_layer_16 | 320 | 4.1310 | 1.0000 |
| head | inchoative | bad_to_good | resid_post_layer_17 | 320 | 4.1310 | 1.0000 |
| head | inchoative | bad_to_good | resid_post_layer_18 | 320 | 4.1310 | 1.0000 |
| head | inchoative | good_to_bad | resid_post_layer_16 | 320 | 4.1310 | 1.0000 |
| head | inchoative | good_to_bad | resid_post_layer_17 | 320 | 4.1310 | 1.0000 |
| head | inchoative | good_to_bad | resid_post_layer_18 | 320 | 4.1310 | 1.0000 |
| low | inchoative | bad_to_good | resid_post_layer_23 | 280 | 3.0412 | 1.0000 |
| low | inchoative | bad_to_good | resid_post_layer_22 | 280 | 3.0421 | 0.9939 |
| low | inchoative | bad_to_good | resid_post_layer_21 | 280 | 3.0401 | 0.9918 |
| low | inchoative | good_to_bad | resid_post_layer_22 | 280 | 3.0425 | 0.9948 |
| low | inchoative | good_to_bad | resid_post_layer_23 | 280 | 3.0412 | 1.0000 |
| low | inchoative | good_to_bad | resid_post_layer_21 | 280 | 3.0414 | 0.9998 |

## DAS Summary

Aggregate control comparison across subtasks and directions:

| control | seed | regime | n | mean_effect | mean_normalized_effect | patched_success_rate |
| --- | --- | --- | --- | --- | --- | --- |
| dummy_pair | 17 | head | 406 | 0.0000 | nan | 0.5000 |
| dummy_pair | 17 | low | 1120 | 0.0000 | nan | 0.5000 |
| none | 17 | head | 312 | 12.4769 | 3.4605 | 0.9936 |
| none | 17 | low | 1120 | 11.2010 | 2.7103 | 0.9563 |
| random_direction | 17 | head | 312 | 0.0027 | -0.0046 | 0.2628 |
| random_direction | 17 | low | 1120 | 0.0027 | 0.0014 | 0.2795 |
| red_blue | 17 | head | 312 | -0.0016 | nan | 0.6987 |
| red_blue | 17 | low | 1120 | 0.0051 | nan | 0.8295 |
| shuffled_label | 17 | head | 312 | -0.0126 | 0.5203 | 0.5192 |
| shuffled_label | 17 | low | 1120 | -0.0092 | 0.5912 | 0.5241 |

Per-subtask and per-direction detail:

| control | seed | regime | subtask | direction | n | mean_effect | effect_ci95_lo | effect_ci95_hi | mean_normalized_effect | patched_success_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dummy_pair | 17 | head | causative | bad_to_good | 118 | 0.0000 | 0.0000 | 0.0000 | nan | 0.0000 |
| dummy_pair | 17 | head | causative | good_to_bad | 118 | 0.0000 | 0.0000 | 0.0000 | nan | 1.0000 |
| dummy_pair | 17 | head | inchoative | bad_to_good | 85 | 0.0000 | 0.0000 | 0.0000 | nan | 1.0000 |
| dummy_pair | 17 | head | inchoative | good_to_bad | 85 | 0.0000 | 0.0000 | 0.0000 | nan | 0.0000 |
| dummy_pair | 17 | low | causative | bad_to_good | 280 | 0.0000 | 0.0000 | 0.0000 | nan | 0.0000 |
| dummy_pair | 17 | low | causative | good_to_bad | 280 | 0.0000 | 0.0000 | 0.0000 | nan | 1.0000 |
| dummy_pair | 17 | low | inchoative | bad_to_good | 280 | 0.0000 | 0.0000 | 0.0000 | nan | 1.0000 |
| dummy_pair | 17 | low | inchoative | good_to_bad | 280 | 0.0000 | 0.0000 | 0.0000 | nan | 0.0000 |
| none | 17 | head | causative | bad_to_good | 98 | 12.9464 | 11.9962 | 13.9180 | 3.2060 | 1.0000 |
| none | 17 | head | causative | good_to_bad | 98 | 12.4049 | 11.4833 | 13.3542 | 3.0691 | 0.9898 |
| none | 17 | head | inchoative | bad_to_good | 58 | 12.7059 | 11.8637 | 13.5659 | 4.1977 | 1.0000 |
| none | 17 | head | inchoative | good_to_bad | 58 | 11.5765 | 10.7732 | 12.4068 | 3.8146 | 0.9828 |
| none | 17 | low | causative | bad_to_good | 280 | 13.6875 | 13.0446 | 14.2258 | 2.8525 | 0.9714 |
| none | 17 | low | causative | good_to_bad | 280 | 13.5741 | 12.8996 | 14.1915 | 2.7850 | 0.9857 |
| none | 17 | low | inchoative | bad_to_good | 280 | 9.2177 | 8.6417 | 9.7424 | 2.7478 | 0.9893 |
| none | 17 | low | inchoative | good_to_bad | 280 | 8.3246 | 7.7978 | 8.8050 | 2.4560 | 0.8786 |
| random_direction | 17 | head | causative | bad_to_good | 98 | -0.0086 | -0.0204 | 0.0026 | -0.0099 | 0.1020 |
| random_direction | 17 | head | causative | good_to_bad | 98 | -0.0038 | -0.0153 | 0.0083 | -0.0082 | 0.2245 |
| random_direction | 17 | head | inchoative | bad_to_good | 58 | 0.0221 | 0.0129 | 0.0323 | 0.0053 | 0.8621 |
| random_direction | 17 | head | inchoative | good_to_bad | 58 | 0.0135 | 0.0027 | 0.0248 | 0.0007 | 0.0000 |
| random_direction | 17 | low | causative | bad_to_good | 280 | -0.0009 | -0.0071 | 0.0051 | 0.0016 | 0.0536 |
| random_direction | 17 | low | causative | good_to_bad | 280 | -0.0011 | -0.0069 | 0.0049 | -0.0020 | 0.3036 |
| random_direction | 17 | low | inchoative | bad_to_good | 280 | 0.0074 | 0.0025 | 0.0118 | 0.0055 | 0.7607 |
| random_direction | 17 | low | inchoative | good_to_bad | 280 | 0.0055 | 0.0013 | 0.0099 | 0.0005 | 0.0000 |
| red_blue | 17 | head | causative | bad_to_good | 98 | -0.2186 | -0.2761 | -0.1582 | nan | 0.6735 |
| red_blue | 17 | head | causative | good_to_bad | 98 | 0.2047 | 0.1463 | 0.2601 | nan | 0.7041 |
| red_blue | 17 | head | inchoative | bad_to_good | 58 | 0.1352 | 0.0776 | 0.1953 | nan | 0.7241 |
| red_blue | 17 | head | inchoative | good_to_bad | 58 | -0.1202 | -0.1724 | -0.0673 | nan | 0.7069 |
| red_blue | 17 | low | causative | bad_to_good | 280 | -0.1727 | -0.1912 | -0.1527 | nan | 0.8143 |
| red_blue | 17 | low | causative | good_to_bad | 280 | 0.1708 | 0.1515 | 0.1893 | nan | 0.8393 |
| red_blue | 17 | low | inchoative | bad_to_good | 280 | 0.2080 | 0.1895 | 0.2274 | nan | 0.8714 |
| red_blue | 17 | low | inchoative | good_to_bad | 280 | -0.1856 | -0.2022 | -0.1693 | nan | 0.7929 |
| shuffled_label | 17 | head | causative | bad_to_good | 98 | -0.0210 | -0.5941 | 0.6272 | 0.4842 | 0.4694 |
| shuffled_label | 17 | head | causative | good_to_bad | 98 | -0.0561 | -0.6492 | 0.5045 | 0.4483 | 0.5714 |
| shuffled_label | 17 | head | inchoative | bad_to_good | 58 | 0.0851 | -0.5981 | 0.8287 | 0.6513 | 0.4655 |
| shuffled_label | 17 | head | inchoative | good_to_bad | 58 | -0.0226 | -0.6202 | 0.5841 | 0.5719 | 0.5690 |
| shuffled_label | 17 | low | causative | bad_to_good | 280 | -0.4174 | -0.7625 | -0.0345 | 0.5674 | 0.5357 |
| shuffled_label | 17 | low | causative | good_to_bad | 280 | 0.3645 | -0.0134 | 0.7258 | 0.5529 | 0.5250 |
| shuffled_label | 17 | low | inchoative | bad_to_good | 280 | 0.1227 | -0.1506 | 0.4041 | 0.6487 | 0.5393 |
| shuffled_label | 17 | low | inchoative | good_to_bad | 280 | -0.1066 | -0.3721 | 0.1408 | 0.5960 | 0.4964 |

## Output Files

- `reports/v2_sanity_summary.csv`
- `reports/v2_sanity_source_summary.csv`
- `reports/v2_red_blue_sanity_summary.csv`
- `reports/v2_red_blue_sanity_source_summary.csv`
- `reports/v2_ap_top_sites.csv`
- `reports/v2_exact_top_sites.csv`
- `reports/v2_das_eval_summary.csv`
- `reports/v2_das_control_comparison.csv`
- `reports/v2_das_subject_summary.csv`
- `reports/v2_das_source_summary.csv`
