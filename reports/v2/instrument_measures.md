# v2 instrument measures (Nov 24; rules: reports/v2_retrain_rules.md)

Analysis: `scripts/analyze_das_v2.py` (TSUBAME, after `jobs/v2_post.sh`); tables in `results/v2_summary/`.

## Finals and checks

49 v2 finals (Pythia 17, Gemma 16, OLMo 16). Every site passed checks 1–5: pairs sha, stored items and folds, recompute (held-out swaps at the sites that kept them, the stored `summary.csv` elsewhere, as amended), zero disjointness hits on each model's evaluation files, and layout. New pairs kept after the behavior filter: Pythia 43 (dropped tweeze/teethe), Gemma 40 (bewail/mew, counsel/reverberate, outmatch/cohere, outrace/gab), OLMo 42 (counsel/reverberate, sod/patter). v2's own δ is recomputed at every site (`frozen_config*.json`); the stored δ_v blocks are kept in `delta_v_stored.json`.

## 1. Gate: held-out cross IIA, v2 − v (paired over stored pairs, 95% bootstrap CI)

| model | site | v | v2 | v2 − v [95% CI] | pairs v2 higher / lower | v2 replaces v? |
|---|---|---|---|---|---|---|
| Pythia | 8 | 0.782 | 0.795 | +0.012 [−0.021, +0.044] | 18 / 11 | yes |
| Pythia | 12 | 0.900 | 0.913 | +0.013 [−0.012, +0.041] | 15 / 13 | yes |
| Gemma | 5 | 0.862 | 0.929 | +0.068 [+0.024, +0.119] | 19 / 6 | yes |
| Gemma | 8 | 0.881 | 0.926 | +0.046 [+0.019, +0.075] | 19 / 7 | yes |
| Gemma | 12 | 0.947 | 0.961 | +0.013 [+0.003, +0.026] | 14 / 6 | yes |
| OLMo | 9 | 0.808 | 0.839 | +0.030 [−0.007, +0.071] | 14 / 14 | yes |
| OLMo | 11 | 0.926 | 0.945 | +0.019 [+0.000, +0.041] | 15 / 9 | yes |
| OLMo | 12 | 0.970 | 0.976 | +0.006 [−0.007, +0.023] | 12 / 7 | yes |

No CI lies below −0.02, so **v2 replaces v in all three models**. The point estimate is positive at every rule site, and the CI excludes zero at Gemma 5, 8, 12 (and touches it at OLMo 11). Median patched fraction (passive-class bases) also rises at every rule site (for example Pythia 8: 0.77 → 0.78; Gemma 8: 0.91 → 0.99).

Descriptive, all sites (fold means from `summary.csv`, unpaired): v2 is at or above v from the middle layers on (Pythia ≥ 6, Gemma ≥ 4, OLMo ≥ 4). At the earliest sites v2 is lower: by up to 0.025 in Pythia 0–5, 0.059 / 0.067 in Gemma 1 / 2, and 0.037 in OLMo 3. These are sites where IIA is low for both (Pythia 0–3 < 0.28), outside every declared reading.

|cos(v2, v)| per fold (min / median / max of 15): Pythia 8 0.91 / 0.92 / 0.93, Pythia 12 0.92 / 0.93 / 0.93; Gemma 5 0.61 / 0.71 / 0.80, Gemma 8 0.89 / 0.90 / 0.92; OLMo 9 0.91 / 0.93 / 0.93, OLMo 11 0.91 / 0.93 / 0.93. v2 is 0.97–0.98 from P (which also had the 18 excluded pairs), except Gemma 5 (0.80–0.94).

## 2. Resolution (directions trained on disjoint verb samples; split 0, folds 0–2, 3 draws each)

Median over 9 draws; |cos(A, B)| range in brackets. Held-out IIA is on the stored fold.

| model | site | n = 10 | n = 20 | n = 30 | IIA at n = 10 / 20 / 30 |
|---|---|---|---|---|---|
| Pythia | 8 | 0.69 [0.67, 0.75] | 0.80 [0.78, 0.83] | 0.85 [0.83, 0.86] | 0.51 / 0.66 / 0.71 |
| Pythia | 12 | 0.71 [0.67, 0.73] | 0.83 [0.80, 0.85] | 0.87 [0.85, 0.89] | 0.67 / 0.84 / 0.88 |
| Gemma | 5 | 0.48 [0.28, 0.56] | 0.60 [0.58, 0.71] | 0.71 [0.61, 0.78] | 0.42 / 0.81 / 0.84 |
| Gemma | 8 | 0.69 [0.63, 0.76] | 0.83 [0.78, 0.85] | 0.87 [0.86, 0.88] | 0.64 / 0.87 / 0.89 |
| OLMo | 9 | 0.76 [0.73, 0.79] | 0.85 [0.85, 0.87] | 0.88 [0.87, 0.90] | 0.60 / 0.75 / 0.80 |
| OLMo | 11 | 0.77 [0.75, 0.80] | 0.86 [0.85, 0.87] | 0.89 [0.88, 0.90] | 0.78 / 0.88 / 0.93 |

Agreement between directions from different verbs rises with n at every site, and so does held-out IIA. At n = 20 (about v's per-fold training size), two directions from disjoint verbs agree at 0.80–0.86, and at 0.60 at Gemma 5. So a v-sized direction depends noticeably on which verbs trained it. Gemma 5 is the least determined site, which matches its lower seed range (Settled 19) and its lower |cos(v2, v)|.

Not a fair comparison, so not claimed: the half-sample directions are closer to v2 (0.92–0.94 at n = 20) than to v (0.86–0.88), but their pairs come from v2's own pool.

## 3. Necessity

Pending: Pythia direction check and mean ablation along v2 (`v2c_pythia_dirchk`).

## Downstream chain

Submitted Nov 24 (`jobs/v2_chain.sh`): passive (rule sites + depth), constructions + in-range, filler-gap, nonce for each model; Pythia direction check; split edit after passive (and the direction check in Pythia). Readings are reported three ways (own δ, common δ_v, raw / natural gap) in `reports/v2_replication_criteria.md` once they are in.
