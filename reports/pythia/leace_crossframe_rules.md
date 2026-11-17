# Cross-frame LEACE (Pythia-1.4B): rules, declared before running (2026-11-17)

Paper plan: Settled 5 (the LEACE edit is the passive class mean difference u_pas; cos(u_act, u_pas) 0.24–0.51) and
the wording of the necessity claim. Nothing below has been run. Branch `paper`.

## Question

The necessity run erases verb class from passive states with erasers fit on passives. Its edit is the passive class
mean difference, which only partly aligns with the active class axis, and the transferred direction (DAS v, u_md)
is defined on actives. So "the feature is necessary" does not follow. Here the erasers are fit on **actives** and
applied to the same passive items. If erasing the class information that actives carry removes the passive *by*
preference, the necessary information is shared across frames.

## Setup (all as in `reports/pythia/leace_necessity.md` unless stated)

- **Passive items, readout, position, sites, fitter:** unchanged. 126 pairs × 125 contexts, T and I prompts,
  log P(" by") at the participle's last token, sites 0–23 (DAS convention), LeaceFitter defaults, float64
  statistics, fp32 with TF32 off. Same fold assignment (`run_das_round2.folds`, 5 folds, seed 0; checked equal to
  the stored `pairs_folds.csv`).
- **Active fitting items:** for every passive item, the active "The AGENT VERB-ed" with the agent of the same curated
  context (`passive_1` context with the same lemma) and the verb's past form (`build_das_prep_prompts.PAST`, else the
  participle form). This is task 1's active set. Position: the verb's last token (last prompt token). Class T = 1,
  I = 0. 31,500 active prompts.
- **Fittings:**
  - **A-in** (cross-frame, all verbs): one eraser set fit on all active prompts, applied to every passive item. The
    passive reference is the in-sample run (P-in), which also saw every verb.
  - **A-cf** (cross-frame, verb-held-out): erasers for fold f fit on the actives of the other four folds, applied to
    fold f's passives. The passive reference is the cross-fitted run (P-cf).
- **Conditions:** single-site LEACE at each site 0–23, fit on natural active states; sequential scrubbing fit on
  actives (eraser L fit on active site-L states with erasers 0..L−1 applied), evaluated for every prefix 0..k,
  k = 0–23.
- **Permutation control:** A-in with permuted verb labels (as `leace_permutation.md`: half the T and half the I verbs
  flipped, |r| with true class < 0.05), 10 permutations, scrub 0–12 and single site 8.
- **Active-side check:** the A-in and A-cf erasers (scrub 0–12 and single 8; A-cf on each fold's held-out actives)
  applied to the actives, readout log P(object start) (logsumexp over " the a an his her their its this that some
  him them it"), share of the natural active T − I gap remaining. This shows the eraser removes class information
  that matters in its own frame.
- **Geometry:** per site, |cos| between the A-in and P-in single-site edit vectors (and A-cf vs P-cf, per fold),
  from the stored `erasers.npz`. The passive class-mean difference after / before erasure (norm ratio) on the
  passive items, per condition.

## Quantities

Natural gap, share remaining and KL as in the necessity report: the task-1 two-way cluster bootstrap (1000 draws,
seed 17), the same draws for every condition, and the passive references recomputed from their stored per-prompt
files (`lli_leace/results/pythia/leace_{necessity,insample}/items.csv.gz`, sha256 recorded). Their natural readout
must equal this run's (max |Δ| < 1e-3).

Removed = 1 − share remaining. **Transfer ratio ρ = removed(A) / removed(P)** for the matched pairing (A-in vs P-in,
A-cf vs P-cf), paired bootstrap on the same draws.

## Confirmatory readings

Primary condition: scrub 0–12 (the necessity-relevant range, as in the paper's headline). For each pairing:

- **Shared:** A's share remaining is well below the permutation control (95% upper bound below the permutation 5th
  percentile; A-in pairing only, the control is fit in-sample), its removed CI excludes 0, and ρ's lower 95% bound
  is ≥ 0.5.
- **Partly shared:** removed CI excludes 0 (and, for A-in, well below the permutation control), but ρ's lower bound
  is < 0.5.
- **Frame-specific:** removed CI includes 0, or (A-in) not well below the permutation control.

Secondary, reported as curves with CIs, no threshold: ρ for single-site erasure at every site 0–12, and for scrub
prefixes 0..k. Sites after ~12 sit next to the readout (as in the necessity report) and are shown, not read.

## Expected (stated to be checked, not criteria)

- From the frame-invariance results (corrected cosine ≈ 1 at 0–4, ≈ 0.5 at 8–12): single-site ρ high at 0–4 and
  falling with depth.
- A-cf (verb-general class information) is the more frame-shared part: ρ(A-cf) ≥ ρ(A-in) at scrub 0–12. The extra
  removal that in-sample erasure adds over cross-fitting (verb-specific information) may be partly frame-specific.
- The active-side check removes most of the active object gap at scrub 0–12.

## Checks

- Passive items and folds equal the stored necessity run's (prompts, item ids, fold per pair).
- Identity hook, non-anchor positions untouched, in-model vs offline single-site erasure: max |Δ| = 0 as before.
- Scrub replay on the active fitting items: pre-erasure mean / cross-covariance equal to those each eraser was fit on.
- Erasers rank 1, top singular value above svd_tol.
- Natural passive log P(by) equals the stored runs' (max |Δ| < 1e-3).

## How this enters the paper

The result decides the wording of the necessity claim. Shared: "erasing the class information carried in actives
removes X of the passive *by* gap". Partly shared: "class information at the verb is needed; the part shared with
actives accounts for X". Frame-specific: the necessity claim stays about passive class information, and "the
feature" is not used for it.

## Outputs

`results/pythia/leace_crossframe/` (per-prompt files gitignored, sha256 listed), `reports/pythia/leace_crossframe.md`.
Scripts: `scripts/run_leace_crossframe.py`, `scripts/analyze_leace_crossframe.py`. Gemma and OLMo get the same test
inside their LEACE runs (paper plan, item 7).
