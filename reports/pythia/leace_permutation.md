# Permutation control for in-sample LEACE (Pythia-1.4B)

Scripts: `scripts/run_leace_permutation.py` (GPU), `scripts/analyze_leace_permutation.py` (this report). 20 permutations in 4 jobs (157 GPU-min).

## Setup

- Same items (126 pairs x 125 contexts, T and I prompt each), position (verb's last token = readout position), sites, fitting and rank (1) as the in-sample conditions of `reports/pythia/leace_necessity.md`: one eraser set fit on all prompts and applied to the same prompts.
- Concept: instead of verb class, a permuted verb label. Per permutation, exactly half of the 126 T verbs and half of the 126 I verbs are flipped at random; every prompt takes its verb's label. Labels are balanced; correlation with the true class over prompts: max |r| = 0 (< 0.05 required).
- Conditions: sequential scrubbing over sites 0–8, 0–12 and 0–23 (prefixes of one sequential fit per permutation) and single-site LEACE at site 8, each fit with the permuted labels.
- Measured on the true items: share of the natural *by* gap (T − I log P(by)) remaining, and mean KL (natural ‖ erased) at the verb position over all prompts. True-label rows: the in-sample run, with 95% cluster-bootstrap CIs.
- Permutation summary: mean, 5th percentile (linear interpolation; with 20 permutations it lies between the lowest and second-lowest), range. "Well below" = the true-label CI upper bound is below the permutation 5th percentile.

## Checks

- True labels run through the permutation code (part0) vs the in-sample run, per-prompt log P(by): max |Δ| scrub_0_8 1.8e-15, scrub_0_12 1.8e-15, scrub_all 8.9e-16, single_8 1.8e-15.
- Natural log P(by) vs the in-sample run: max |Δ| 8.9e-16 over jobs.
- Permuted-label erasers: rank [1]; top singular value of W Σ_xz 0.325–0.500 (true labels 0.402–0.500; svd_tol 0.01: 0 fits below it); training cross-covariance residual max 4.5e-14.

## Flags

- None: for every condition the true-label CI upper bound is below the permutation 5th percentile.

## Share of the natural *by* gap remaining

Natural gap +0.77 nats.

| Condition | true labels (95% CI) | perm mean | perm 5th pct | perm range | perms ≤ true | KL true | KL perm mean (range) | true well below perm 5th pct |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| scrub, sites 0–8 | 0.16 [-0.27, 0.37] | 0.98 | 0.97 | 0.96–1.00 | 0/20 | 0.070 | 0.006 (0.004–0.011) | yes |
| scrub, sites 0–12 | 0.04 [-0.46, 0.28] | 0.98 | 0.97 | 0.96–1.00 | 0/20 | 0.083 | 0.007 (0.005–0.012) | yes |
| scrub, sites 0–23 | -0.04 [-0.58, 0.22] | 0.98 | 0.97 | 0.96–1.00 | 0/20 | 0.102 | 0.010 (0.006–0.017) | yes |
| single site 8 | 0.34 [-0.01, 0.51] | 1.00 | 0.99 | 0.99–1.01 | 0/20 | 0.034 | 0.001 (0.001–0.002) | yes |

## Files

- `reports/pythia/leace_permutation_summary.csv`: the table above.
- `reports/pythia/leace_permutation_per_perm.csv`: gap, share remaining and KL per permutation and condition (and the true labels rerun through this code).
- `results/pythia/leace_permutation/part*/`: labels per verb, fit diagnostics, checks, metadata; per-prompt `items.csv.gz` not in git (sha256 part0 9c0bc7409f586284…, part1 693afbfa2a234cf7…, part2 d93b0fe6d947e08d…, part3 95825b2385095257…; on TSUBAME under `lli_leace/results/pythia/leace_permutation/`).

