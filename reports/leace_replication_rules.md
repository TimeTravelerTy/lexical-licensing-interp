# LEACE necessity and cross-frame LEACE on Gemma-3-1B and OLMo-2-1B: rules (paper plan, work item 7)

Declared Nov 23, 2026, before any run. Replicates the Pythia chain with the same code, items and readings:
`reports/pythia/leace_necessity.md` (Settled 1), `leace_permutation.md`, `pythia/leace_crossframe_rules.md` (as amended) and `pythia/leace_pcf_recentered.md` (Settled 13). The plan listed the necessity criteria as "still to declare"; they are declared here.

## Items, position, readout, fitter

Exactly the Pythia runs': 126 pairs × 125 curated `was` contexts (other pairs' contexts), log P(" by") at the participle's last token (= erasure position), 5 verb-level folds (seed 0), LeaceFitter defaults in float64, the model in fp32 (TF32 off). Each model's architecture defaults (`model_arch`: Gemma BOS and fp32 unembedding; OLMo fp32 unembedding), as its DAS runs.

## Depth

Sites 0..L−1 (L = 26 for Gemma, 16 for OLMo; `LEACE_N_SITES`). The Pythia primary range, scrub 0–12 of 24 sites, is half the depth. The declared primary range per model is the same relative depth (`LEACE_NEC_TOP`): **Gemma scrub 0–13, OLMo scrub 0–8.** Every scrub prefix 0..k and every single site are reported as curves (`--cumulative`), with each model's DAS rule sites marked (Gemma 5 / 8, OLMo 9 / 11). The fixed single site stays 8 (it exists in all three models; Gemma 0.31 of depth, OLMo 0.50).

## Runs per model

1. Cross-fitted necessity (`run_leace_necessity.py --cumulative`).
2. In-sample (`--insample --cumulative`).
3. Permutation control: 20 label permutations, in-sample (`run_leace_permutation.py --with-true`, reproducing run 2).
4. Cross-frame (`run_leace_crossframe.py`, actives "The AGENT VERB-ed"; erasers fit on actives, applied to the passives; variants c (primary), u, t; 20 permutations), with run 1's folds.
5. P-cf re-centered (`run_leace_pcf_recentered.py`), so that the verb-held-out cross-frame ratio is matched-centering from the start.

## Readings

Primary range as above; share remaining = erased gap / natural gap, the task-1 two-way cluster bootstrap (1000 draws, seed 17).

- **N1, verb-general removal:** the cross-fitted share remaining has its 95% CI upper bound below 1. Reported as removed = 1 − share, with CI (the paper's "one number", per model).
- **N2, additional in-sample removal:** in-sample share remaining below the cross-fitted share on the same draws (the paired difference's CI below 0).
- **N3, permutation control:** the true in-sample CI upper bound lies below the 5th percentile of the 20 permutations' share remaining (as Settled 1).
- **X, cross-frame:** the amended Pythia readings for A-cf vs P-cf (matched centering, variant c) and A-in vs P-in: invalid / majority transfer (ρ lower bound ≥ 0.5) / minority transfer (ρ upper bound < 0.5) / no meaningful transfer (ρ upper bound < 0.2) / inconclusive; the translation flag (shift-only CI excludes 0) as in Pythia.

**Replication** of "verb-general + additional in-sample removal" in a model: N1, N2 and N3 all hold. If N1 holds but N2 or N3 fails, the model replicates only the verb-general part, and that is what the paper says for it. A model where N1 fails is reported as not replicating the verb-general removal at the declared range, with the curves.

**Checks** (each run's own, enforced as in Pythia): fitting n, cross-covariance residual, identity-hook self-patch, outside-anchor untouched, offline = in-model erasure, scrub replay; permutation balance and |r| < 0.05; the permutation run's true labels reproduce the in-sample run; cross-frame and P-cf checks as in their Pythia rules. A failed check makes that run's reading invalid.

## Use

The three-model table (removed share cross-fitted, in-sample, permutation gate, ρ) goes in the necessity section with the Pythia numbers. Model differences are findings. The Pythia numbers are unchanged.
