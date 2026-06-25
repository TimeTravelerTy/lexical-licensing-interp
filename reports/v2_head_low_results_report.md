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
| none@l18 | 17 | head | 312 | 6.9981 | 1.8763 | 0.9359 |
| none@l18 | 17 | low | 1120 | 6.5376 | 1.6218 | 0.8920 |
| none@l18@verb_final_subtoken_to_subject_final_subtoken | 17 | head | 312 | 0.0748 | 0.0204 | 0.2660 |
| none@l18@verb_final_subtoken_to_subject_final_subtoken | 17 | low | 1120 | 0.0601 | 0.0174 | 0.2875 |
| none@l20 | 17 | head | 312 | 7.5834 | 2.0373 | 0.9391 |
| none@l20 | 17 | low | 1120 | 7.0665 | 1.7431 | 0.9009 |
| none@l20@verb_final_subtoken_to_subject_final_subtoken | 17 | head | 312 | 0.0502 | 0.0232 | 0.2821 |
| none@l20@verb_final_subtoken_to_subject_final_subtoken | 17 | low | 1120 | -0.0150 | 0.0114 | 0.2920 |
| random_direction | 17 | head | 312 | 0.0027 | -0.0046 | 0.2628 |
| random_direction | 17 | low | 1120 | 0.0027 | 0.0014 | 0.2795 |
| red_blue | 17 | head | 312 | -0.0016 | nan | 0.6987 |
| red_blue | 17 | low | 1120 | 0.0051 | nan | 0.8295 |
| shuffled_label | 17 | head | 312 | -0.0126 | 0.5203 | 0.5192 |
| shuffled_label | 17 | low | 1120 | -0.0092 | 0.5912 | 0.5241 |

Per-subtask and per-direction detail:

| control | seed | regime | subtask | direction | n | mean_effect | effect_ci95_lo | effect_ci95_hi | mean_normalized_effect | patched_success_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none@l18@verb_final_subtoken_to_subject_final_subtoken | 17 | head | causative | bad_to_good | 98 | 0.0430 | 0.0233 | 0.0641 | 0.0055 | 0.1020 |
| none@l18@verb_final_subtoken_to_subject_final_subtoken | 17 | head | causative | good_to_bad | 98 | 0.0759 | 0.0453 | 0.1135 | 0.0176 | 0.2245 |
| none@l18@verb_final_subtoken_to_subject_final_subtoken | 17 | head | inchoative | bad_to_good | 58 | 0.1433 | 0.1088 | 0.1832 | 0.0477 | 0.8793 |
| none@l18@verb_final_subtoken_to_subject_final_subtoken | 17 | head | inchoative | good_to_bad | 58 | 0.0582 | 0.0221 | 0.0964 | 0.0229 | 0.0000 |
| none@l18@verb_final_subtoken_to_subject_final_subtoken | 17 | low | causative | bad_to_good | 280 | 0.0602 | 0.0421 | 0.0769 | 0.0158 | 0.0536 |
| none@l18@verb_final_subtoken_to_subject_final_subtoken | 17 | low | causative | good_to_bad | 280 | 0.0487 | 0.0355 | 0.0616 | 0.0157 | 0.3107 |
| none@l18@verb_final_subtoken_to_subject_final_subtoken | 17 | low | inchoative | bad_to_good | 280 | 0.0646 | 0.0518 | 0.0761 | 0.0174 | 0.7857 |
| none@l18@verb_final_subtoken_to_subject_final_subtoken | 17 | low | inchoative | good_to_bad | 280 | 0.0671 | 0.0523 | 0.0835 | 0.0206 | 0.0000 |
| none@l18 | 17 | head | causative | bad_to_good | 98 | 6.7066 | 6.1177 | 7.2803 | 1.5311 | 0.9796 |
| none@l18 | 17 | head | causative | good_to_bad | 98 | 7.0994 | 6.4943 | 7.7034 | 1.6393 | 0.9388 |
| none@l18 | 17 | head | inchoative | bad_to_good | 58 | 7.4604 | 6.9682 | 7.9617 | 2.4946 | 1.0000 |
| none@l18 | 17 | head | inchoative | good_to_bad | 58 | 6.8572 | 6.4133 | 7.3109 | 2.2416 | 0.7931 |
| none@l18 | 17 | low | causative | bad_to_good | 280 | 6.8074 | 6.4373 | 7.1462 | 1.3936 | 0.9071 |
| none@l18 | 17 | low | causative | good_to_bad | 280 | 8.1964 | 7.7791 | 8.5842 | 1.7551 | 0.9679 |
| none@l18 | 17 | low | inchoative | bad_to_good | 280 | 5.9195 | 5.5838 | 6.2290 | 1.7643 | 0.9964 |
| none@l18 | 17 | low | inchoative | good_to_bad | 280 | 5.2271 | 4.9468 | 5.4853 | 1.5743 | 0.6964 |
| none@l20@verb_final_subtoken_to_subject_final_subtoken | 17 | head | causative | bad_to_good | 98 | -0.0686 | -0.1559 | 0.0140 | 0.0104 | 0.1020 |
| none@l20@verb_final_subtoken_to_subject_final_subtoken | 17 | head | causative | good_to_bad | 98 | 0.1301 | 0.0721 | 0.2015 | 0.0189 | 0.2551 |
| none@l20@verb_final_subtoken_to_subject_final_subtoken | 17 | head | inchoative | bad_to_good | 58 | 0.3060 | 0.2004 | 0.4224 | 0.0858 | 0.9138 |
| none@l20@verb_final_subtoken_to_subject_final_subtoken | 17 | head | inchoative | good_to_bad | 58 | -0.1401 | -0.2597 | -0.0145 | -0.0105 | 0.0000 |
| none@l20@verb_final_subtoken_to_subject_final_subtoken | 17 | low | causative | bad_to_good | 280 | -0.2564 | -0.3027 | -0.2077 | -0.0328 | 0.0536 |
| none@l20@verb_final_subtoken_to_subject_final_subtoken | 17 | low | causative | good_to_bad | 280 | 0.1231 | 0.0924 | 0.1549 | 0.0331 | 0.3214 |
| none@l20@verb_final_subtoken_to_subject_final_subtoken | 17 | low | inchoative | bad_to_good | 280 | 0.1277 | 0.1011 | 0.1540 | 0.0472 | 0.7893 |
| none@l20@verb_final_subtoken_to_subject_final_subtoken | 17 | low | inchoative | good_to_bad | 280 | -0.0545 | -0.0971 | -0.0115 | -0.0017 | 0.0036 |
| none@l20 | 17 | head | causative | bad_to_good | 98 | 7.1288 | 6.4987 | 7.7417 | 1.6547 | 0.9796 |
| none@l20 | 17 | head | causative | good_to_bad | 98 | 7.9148 | 7.1814 | 8.6455 | 1.8399 | 0.9388 |
| none@l20 | 17 | head | inchoative | bad_to_good | 58 | 8.2085 | 7.6398 | 8.8004 | 2.7254 | 1.0000 |
| none@l20 | 17 | head | inchoative | good_to_bad | 58 | 7.1665 | 6.6929 | 7.6369 | 2.3290 | 0.8103 |
| none@l20 | 17 | low | causative | bad_to_good | 280 | 7.2699 | 6.8821 | 7.6214 | 1.4823 | 0.9286 |
| none@l20 | 17 | low | causative | good_to_bad | 280 | 9.5421 | 9.0446 | 10.0067 | 2.0773 | 0.9679 |
| none@l20 | 17 | low | inchoative | bad_to_good | 280 | 6.0545 | 5.6991 | 6.4073 | 1.7936 | 0.9964 |
| none@l20 | 17 | low | inchoative | good_to_bad | 280 | 5.3997 | 5.1023 | 5.6794 | 1.6193 | 0.7107 |
| dummy_pair | 17 | head | causative | bad_to_good | 118 | 0.0000 | 0.0000 | 0.0000 | nan | 0.0000 |
| dummy_pair | 17 | head | causative | good_to_bad | 118 | 0.0000 | 0.0000 | 0.0000 | nan | 1.0000 |
| dummy_pair | 17 | head | inchoative | bad_to_good | 85 | 0.0000 | 0.0000 | 0.0000 | nan | 1.0000 |
| dummy_pair | 17 | head | inchoative | good_to_bad | 85 | 0.0000 | 0.0000 | 0.0000 | nan | 0.0000 |
| dummy_pair | 17 | low | causative | bad_to_good | 280 | 0.0000 | 0.0000 | 0.0000 | nan | 0.0000 |
| dummy_pair | 17 | low | causative | good_to_bad | 280 | 0.0000 | 0.0000 | 0.0000 | nan | 1.0000 |
| dummy_pair | 17 | low | inchoative | bad_to_good | 280 | 0.0000 | 0.0000 | 0.0000 | nan | 1.0000 |
| dummy_pair | 17 | low | inchoative | good_to_bad | 280 | 0.0000 | 0.0000 | 0.0000 | nan | 0.0000 |

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
