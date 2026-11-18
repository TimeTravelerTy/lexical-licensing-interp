# Cross-frame LEACE (Pythia-1.4B): rules, declared before running (2026-11-17; amended 2026-11-18, see the end)

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

## Amendment (2026-11-18, after a Codex review and two smoke runs on 9 pairs; no full-run result exists)

Where this section and the text above differ, this section holds.

**Translation confound and variants.** An active-fit eraser is x' = x − e·r(x − μ_A). On passives it also adds the
constant −e·r(μ_P − μ_A) to every state, which can change log P(by) downstream without touching the class contrast.
So each eraser is evaluated in three variants:
- **c (primary): centered.** The bias is replaced by the label-free mean of the passive states it is applied to: all
  passives for A-in, fold f's passives for A-cf. For scrubbing, it is the mean at site L under the centered erasers
  0..L−1, computed sequentially on the passives. The edit direction and the read direction are the active fit's.
- **u: as fit** (the active mean). Reported as secondary.
- **t: translation only.** The constant shift of u alone (x + E_u(μ_P) − μ_P; for scrubbing, μ_P is taken from the
  c sequence). This is a control.
The passive-fit references are centered on passive means by construction, so c is the matched comparison.

**Readings (scrub 0–12, variant c; also reported for u).** For each pairing (A-cf vs P-cf, A-in vs P-in):
- **invalid:** any check below fails, the passive reference's removed CI does not exclude 0 (lower bound ≤ 0), or ρ
  is not finite;
- **gap enhancement:** A's removed CI is entirely below 0 (erasure widens the gap);
- if A's removed CI is above 0 and its share remaining (upper 95%) is below the **minimum** of the permutation
  shares:
  - **majority transfer** if ρ's lower 95% bound ≥ 0.5;
  - **minority transfer** if ρ's upper bound < 0.5;
  - **positive transfer, share unresolved** otherwise;
- **no meaningful transfer:** ρ's upper bound < 0.2 (an equivalence bound; a CI that merely includes 0 is not
  evidence of frame specificity);
- **inconclusive:** otherwise.

Flags (reported with the reading; they do not change the label):
- **eraser inert in its own frame:** the active-side removal CI includes 0;
- **translation alone removes part of the gap:** the t variant's removed CI excludes 0.

The paper does not use the "shared" wording if either flag is raised.

**Which result governs the paper.**
- **A-cf vs P-cf governs the shared-feature claim.** It is verb-general class information, and the eraser never saw
  the evaluated verbs, so token-identity memorization is excluded.
- A-in vs P-in is secondary: class information for the same verb tokens, shared across frames.
- If they disagree, the paper reports both and states the claim at the level A-cf supports.
- "X" in the paper sentence is the absolute removal (share of the natural gap, with CI); ρ is reported next to it,
  as a ratio to the matched passive intervention, not as a fraction of shared information. The expected ρ(A-cf) ≥
  ρ(A-in) is replaced by a comparison of absolute removals: ρ has different denominators (P-cf removes ≈ 0.41, P-in
  ≈ 0.96).
- A positive result supports "class information that actives carry is needed for the passive *by* preference". It
  does not identify DAS v. The direction-specific evidence remains the v / u_md ablation (Settled 3). Geometry versus
  u_md is reported (|cos| of the A-in edit with each u_md basis at 4, 6, 8, 12, 17).

**Permutation control.** 20 permutations (was 10), A-in fitting, variants c and u, scrub 0–12 and single 8. The gate
is the permutation minimum, not an interpolated 5th percentile. It applies to both pairings: the permuted erasers are
fit on all actives, which is at least as strong an erasure as A-cf. Permutation KL and edit norms are reported next
to the true-label ones. This is a label-specificity control (class-orthogonal labels), not a calibrated test.

**Active-side readout.** log P(object start) over the declared 13 tokens (`analyze_das_prep.OBJ`; the first draft's
code used the DAS runner's 27-token set). Token ids are stored in `meta.json`.

**Checks are enforced.** A full run requires the stored fold file and both references. The reading is "invalid" if:
- the folds, prompts or in-sample / cross-fit roles of the references differ;
- the natural log P(by) differs by ≥ 1e-3;
- the recomputed stored necessity shares differ by > 1e-6;
- identity hook, non-anchor positions or active repeat exceed 1e-6 / 1e-6 / 1e-5;
- in-model vs offline erasure exceeds 1e-5;
- scrub replay mean, cross-covariance or post-erasure residual exceed 1e-6 / 1e-6 / 1e-5;
- an eraser is not rank 1, or a top singular value is ≤ 0.01;
- the permutation balance is |r| ≥ 0.05.

The sha256 of the new per-prompt files and erasers is recorded.

**Also reported:**
- the absolute gap change in nats;
- mean edit norm per erased site;
- the passive class-mean difference per fold (A-cf) and pooled (scrub rows compare with the natural difference at
  the site, not before vs after the local eraser).

## Result (2026-11-18, job 9438425, commit 665dabf)

Report: `reports/pythia/leace_crossframe.md`. All checks pass. Both pairings and both variants read **majority transfer**, with the flag "translation alone removes part of the gap" (shift-only removes 0.02–0.03). A-cf vs P-cf (c): share remaining 0.66 [0.47, 0.76] vs 0.59, ρ 0.81 [0.73, 0.90]. The paper wording and the known centering mismatch of the stored P-cf reference are in `docs/paper_plan.md`, Settled 13.

## Follow-up declared after the full run (2026-11-18): P-cf re-centered

The Codex review of the result found a centering mismatch in the verb-held-out pairing. A-cf (c) is centered on the evaluated fold's passive mean, but the stored P-cf erasers keep the mean of the passives they were fit on (the training folds). So ρ(cf) compares slightly differently centered interventions. This follow-up was declared after seeing the result above. Nothing in it has been run.

- **Erasers.** The stored P-cf erasers (`lli_leace/results/pythia/leace_necessity/erasers.npz`, rank-1 edit / read / bias per fold and site, single and scrub), rebuilt as LeaceEraser(edit, read, bias).
- **Variants on the same passive items, folds and readout:**
  - **u (as stored):** a reproduction check. It must reproduce the stored per-prompt log P(by) for every single site and scrub prefix (max |Δ| < 1e-4 nats).
  - **c (re-centered):** the bias is replaced by fold f's label-free passive mean. For scrub, this is the mean at site L under the re-centered erasers 0..L−1, computed sequentially, exactly as A-cf (c).
- **Conditions:** single sites 0–23 and scrub prefixes 0..k, k = 0–23.
- **Quantities:** the same bootstrap draws (1000, seed 17, task-1 item table). Share remaining for P-cf (c), and ρ(cf, matched) = removed(A-cf c) / removed(P-cf c).
- **Reading:** the amended rules' A-cf vs P-cf (c) reading recomputed with P-cf (c) as the reference. The permutation gate, flags and A-cf numbers are unchanged.
  - If the label is the same, the paper keeps it and quotes the matched ρ.
  - If it changes, the paper uses the matched reading and states that the stored reference was centered differently.
  - The absolute A-cf removal (0.34) does not depend on this.
- **Outputs:** `results/pythia/leace_pcf_recentered/`, `reports/pythia/leace_pcf_recentered.md`. Scripts: `scripts/run_leace_pcf_recentered.py`, `scripts/analyze_leace_pcf_recentered.py`.

### Follow-up result (2026-11-18, job 9445093)

Report: `reports/pythia/leace_pcf_recentered.md`. The as-stored erasers reproduce the stored run (max |Δ| 1.8e-15). P-cf re-centered leaves 0.59 [0.36, 0.71]; matched ρ is 0.82 [0.74, 0.91] (stored reference 0.81 [0.73, 0.90]). The label is the same (majority transfer, translation flag), so the paper keeps it and quotes the matched ρ.
