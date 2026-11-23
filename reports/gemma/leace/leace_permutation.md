# Permutation control for in-sample LEACE (Pythia-1.4B)

Scripts: `scripts/run_leace_permutation.py` (GPU), `scripts/analyze_leace_permutation.py` (this report). 20 permutations in 4 jobs (107 GPU-min).

## Setup

- Same items (126 pairs x 125 contexts, T and I prompt each), position (verb's last token = readout position), sites, fitting and rank (1) as the in-sample conditions of `reports/pythia/leace_necessity.md`: one eraser set fit on all prompts and applied to the same prompts.
- Concept: instead of verb class, a permuted verb label. Per permutation, exactly half of the 126 T verbs and half of the 126 I verbs are flipped at random; every prompt takes its verb's label. Labels are balanced; correlation with the true class over prompts: max |r| = 0 (< 0.05 required).
- Conditions: sequential scrubbing over sites 0–8, 0–12 and 0–23 (prefixes of one sequential fit per permutation) and single-site LEACE at site 8, each fit with the permuted labels.
- Measured on the true items: share of the natural *by* gap (T − I log P(by)) remaining, and mean KL (natural ‖ erased) at the verb position over all prompts. True-label rows: the in-sample run, with 95% cluster-bootstrap CIs.
- Permutation summary: mean, 5th percentile (linear interpolation; with 20 permutations it lies between the lowest and second-lowest), range. "Well below" = the true-label CI upper bound is below the permutation 5th percentile.

## Checks

- True labels run through the permutation code (part0) vs the in-sample run, per-prompt log P(by): max |Δ| scrub_0_8 1.8e-15, scrub_0_13 1.8e-15, scrub_all 1.8e-15, single_8 8.9e-16.
- Natural log P(by) vs the in-sample run: max |Δ| 8.9e-16 over jobs.
- Permuted-label erasers: rank [1]; top singular value of W Σ_xz 0.263–0.496 (true labels 0.409–0.497; svd_tol 0.01: 0 fits below it); training cross-covariance residual max 5.1e-14.

## Flags

- None: for every condition the true-label CI upper bound is below the permutation 5th percentile.

## Share of the natural *by* gap remaining

Natural gap +0.41 nats.

| Condition | true labels (95% CI) | perm mean | perm 5th pct | perm range | perms ≤ true | KL true | KL perm mean (range) | true well below perm 5th pct |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| scrub, sites 0–8 | -0.10 [-1.40, 0.28] | 0.98 | 0.93 | 0.92–1.04 | 0/20 | 0.053 | 0.005 (0.003–0.011) | yes |
| scrub, sites 0–13 | -0.26 [-1.77, 0.18] | 0.98 | 0.93 | 0.92–1.05 | 0/20 | 0.061 | 0.006 (0.004–0.014) | yes |
| scrub, all sites (0–L−1) | -1.03 [-3.52, -0.33] | 0.97 | 0.93 | 0.92–1.04 | 0/20 | 0.095 | 0.009 (0.005–0.024) | yes |
| single site 8 | 0.32 [-0.55, 0.55] | 0.99 | 0.98 | 0.97–1.01 | 0/20 | 0.021 | 0.001 (0.000–0.002) | yes |

## Files

- `reports/pythia/leace_permutation_summary.csv`: the table above.
- `reports/pythia/leace_permutation_per_perm.csv`: gap, share remaining and KL per permutation and condition (and the true labels rerun through this code).
- `results/gemma/leace_permutation/part*/`: labels per verb, fit diagnostics, checks, metadata; per-prompt `items.csv.gz` not in git (sha256 part0 b7f662e81894749b…, part1 5a3f1f172bc2b14f…, part2 5754d49d40006c88…, part3 86c21b50cd13effe…; on TSUBAME under `lli_leace/results/gemma/leace_permutation/`).

