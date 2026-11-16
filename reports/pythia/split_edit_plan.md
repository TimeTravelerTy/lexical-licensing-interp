# Split-edit test, frame invariance and side splits (Pythia-1.4B): rules, declared before running (2026-11-16)

Paper plan tasks 1, 3 and 8 (`docs/paper_plan.md`). Nothing below has been run. The DAS bases, the passive-test plan
and the LEACE erasers are the stored ones; no existing output is modified. New outputs go to
`results/pythia/split_edit/` and `reports/pythia/split_edit.md`. Branch `paper`. The report has tables, checks and
the declared readings, nothing more.

Notation as in `das_direction_check_plan.md`. Sites: 4, 6, 8, 12, 17 (site s = residual stream entering layer s).
There are 15 final-run rank-1 bases per site (3 splits × 5 folds).
- v = the unit DAS vector.
- u = u_md: the unit class mean difference of the basis's training items. Natural states, bf16 model, as in the
  direction check.

## A. Split edit

### Decomposition (per basis)

v = a·u + w, where a = v·u and w = v − a·u (so w ⊥ u, b = ‖w‖, ŵ = w / b).

For an interchange with base state h_b and source state h_s at site s, the stored v patch adds δ·v, with
δ = v·(h_s − h_b) taken in-batch, exactly as `Runner.patched`. It is split into two pieces that sum to it exactly:

| Condition | State edit | What moves |
|---|---|---|
| full | h ← h + δ·v | the stored v patch |
| on | h ← h + δ·a·u | v's push, on-axis part only |
| off | h ← h + δ·w | v's push, off-axis part only |

The read direction is v in all three, so the size of the push is v's. Only the written direction changes. This
differs from the direction check's u_md patch, which reads and writes along u.

### Rows

- **Passives:** the same plan rows as the direction check (primary population, bad bases, T and I donors, all 3
  splits, the basis the plan assigns), giving D(R) = T donor − I donor for R ∈ {M, log P(O), log P(" by"),
  log P(".")}.
  - Three-way cluster bootstrap of `analyze_passive_test.py` (2,000 draws, seed 17).
  - D class by the declared passive-test rule with the site's δ_s.
- **Held-out actives:** the same swaps as the final runs (`run_das_round2.evaluate`), with the written direction
  replaced. Reported:
  - cross-class IIA;
  - Δ log P(O) for I base ← T source (`per_pair`, pair bootstrap, 2,000 draws, seed 0).

### Measures

- Shares: D_on / D_full and D_off / D_full per readout, ratio per bootstrap draw.
- Additivity residual: D_full − D_on − D_off. It is not 0 in general, because the model is nonlinear.
- **Push size against natural spread** (per site, pooled over bases):
  - the median over rows of |δ·a| / sd(h·u) and of |δ·b| / sd(h·ŵ);
  - the share of rows with |δ·b| > 2 sd(h·ŵ).
  - sd is over the natural states of (i) all primary passive prompts (good and bad) and (ii) all 406 active
    items.
  - δ here is computed from the stored natural states (offline).
- **Class separation along each axis** (standardized mean difference, mean over bases): T − I on actives, good −
  bad on primary passives, along u and along ŵ.

### Predictions

- **A1.** At site 12, the off-axis piece carries most of v's passive O rise: D_off(O) / D_full(O) ≥ 0.5, and D_on(O)
  is not classed RISE.
- **A2.** At sites 4, 6, 8 and 12, the on-axis piece carries most of v's passive *by* effect: D_on(by) / D_full(by) ≥
  0.5 at each site.
- No prediction for site 17 or for the push-size and separation tables (descriptive).

### Decision (paper plan rule)

- If A1 holds: Figure 1's staging curve is drawn with the on-axis piece and with u_md in every model, and v is
  reported as an upper bound.
- If D_on(O) / D_full(O) ≥ 0.5 at site 12 instead: staging stays drawn with v, with u_md's later boundary noted.
- Anything between: report both curves, and the figure choice is left to the write-up.

## B. Frame invariance of the class axis

### Passive class axis

u_pas(s) = the unit class mean difference (T − I) of the verb-position states of the LEACE items. Those are
`run_leace_necessity.load_items`: curated `was` prefixes + participle, other verbs' contexts only, 126 pairs × 125
contexts, both prompts, at the participle's last token. It is computed from per-verb mean states (every verb has 125
contexts, so this equals the prompt mean).

### Active class axis

u_act(s) = the same over the 406 active DAS items (per-verb means; every verb has 7 items). The fold-specific u_md
of A is reported alongside.

### Measures

All sites 0–23, same bf16 runner.
- **Check B0:** |cos(u_pas, LEACE in-sample `single_edit`)| at every site. For a binary label, the rank-1 LEACE
  edit vector is proportional to the class cross-covariance, which is the class mean difference on its fit items.
  - **Pass:** ≥ 0.98 at every site (bf16 states vs the eraser's fp32 fit).
  - If it fails, the LEACE/u_md cosine of the direction check is not just a frame difference, and that is
    reported.
- **cos(u_act, u_pas)**, signed.
- **Reliability ceilings:** split-half cosines over 200 random halves of the pairs (numpy seed 0).
  - Passive: 63 vs 63 of the 126 pairs.
  - Active: 14 vs 15 of the 29 pairs.
  - Mean and 2.5–97.5 percentiles.
- **Sign-flip reference:** each passive pair's T/I labels are swapped with probability ½ (1,000 draws, seed 1), and
  cos(u_act, u_pas,flip) is recomputed. Mean and 97.5th percentile of |cos|.
- **Frame-invariance index:** FI = cos(u_act, u_pas) / sqrt(r_act · r_pas), with r the mean split-half cosine.
  - This is an approximate disattenuation. Half-sample reliabilities understate full-sample ones, so FI is an
    upper bound.
- |cos(v, u_pas)| per basis at sites 4, 6, 8, 12, 17.

### Predictions

- **B1.** cos(u_act, u_pas) is above the sign-flip 97.5th percentile at every site 0–23.
- **B2.** At sites 8 and 12, cos(u_act, u_pas) is below the passive split-half ceiling (the axis is partly
  frame-specific). Expected range at site 8: 0.3–0.5, which B0 plus the direction check (|cos(u_md, edit)| 0.36 at
  8) already imply.

### Decision (paper plan rule)

- "Low" means FI < 0.5 at both sites 8 and 12. Then the paper says "one direction that every construction reads"
  rather than "one feature with one class axis", and keeps transfer as the evidence for sharing.
- FI ≥ 0.5 at both sites: "one feature" stands.
- Mixed: report and decide in the write-up.

## C. Patching along the passive class axis (exploratory)

Interchange along u_pas (read and write u_pas, `Runner.patched`), cross-fitted so that no direction saw the verbs it
is tested on.

- **Passives** (the A plan rows):
  - The 64 primary pairs are dealt into 5 groups (sorted pair ids, numpy seed 2 permutation, round-robin).
  - For rows whose base pair is in group g, u_pas is fit on the LEACE items excluding:
    - every verb (participle) of group g's pairs;
    - every participle of the 29 DAS pairs (the donors' verbs).
- **Held-out actives:** u_pas fit excluding every participle of the 29 DAS pairs (one direction per site).
- The same measures as the direction check's u_md rows: D(R) with class; IIA; Δ log P(O).
- Reported next to the stored v and u_md (fold) rows, from `summary_passive_patching.csv` and
  `summary_active_patching.csv`.
- No prediction.

## D. Side splits of the gap-closing results (no GPU)

Gap G = good − bad log P(" by"). For a manipulation, ΔG = G_after − G_natural. The bad-side share is

(bad_after − bad_natural) / (−ΔG),

the part of the closure that comes from bad passives gaining *by*. The good-side share is 1 minus that. Computed from
pair means, with the ratio of means over a pair bootstrap (2,000 draws, seed 17).

- **D-abl:** the direction check's passive mean ablation (`ablation_passive_site{s}.parquet`), for v and u_md.
  - Natural = the same-batch "none" rows. Each item is averaged over its 3 splits. 64 primary pairs.
  - The v rows at 4–12 were seen once, uncommitted ("about three-quarters from bad"); they are committed here and
    are not a prediction.
- **D-leace:** the LEACE per-prompt dumps (`leace_insample/items.csv.gz`, `leace_necessity/items.csv.gz`), for
  single 8, scrub 0–8, scrub 0–12 and scrub 0–23.
  - Natural = `lp_by__natural` (same batch). 126 pairs.
- **Prediction D1:** the bad-side share is ≥ 0.5 for in-sample and cross-fitted scrub 0–12 (the suppression
  framing: class information mainly suppresses *by* after intransitives).
  - If D1 holds in both, the suppression framing goes into the paper as a reading of the necessity result.
  - If not, it is dropped.

## Checks (run before any table is read)

1. Fold assignment equals the frozen one; the plan regenerates with the stored SHA-256 (as the direction check).
2. **Write-equals-read reproduction:** the split-edit function with write = v reproduces `Runner.patched` with basis
   v bitwise on a batch per site (max |ΔM| = 0).
3. **Direction-check reproduction:**
   - the full condition's held-out active IIA per basis within 0.01 of the stored final runs;
   - passive D(O) and D(by) for the full condition within 0.02 nats of the stored values;
   - u per basis within 1e-4 (max abs) of the direction check's stored `u_md`, if those files are given.
4. a·u + w = v to 1e-6; w·u < 1e-6.
5. B0 above.
6. C: the fit sets exclude the listed verbs (asserted); at least 40 pairs per fit.
7. LEACE items: 126 pairs, 125 contexts per verb (its own context left out; 126 contexts in all), 31,500 prompts. Natural log P(by) in the runner (bf16) is not compared;
   only states are used.

If a check fails, the affected rows are reported with the failure next to them. They are not dropped.

## Outputs

`results/pythia/split_edit/`:
- `split_patches_site{s}.parquet`: plan row, condition, readouts, δ;
- `axis_patches_site{s}.parquet`: C rows;
- `active_summary.csv`, `active_swaps.csv.gz`: conditions full / on / off / u_pas;
- `directions_site{s}.npz`: v, u, a, w per basis;
- `class_states.npz`: per-verb mean states, sites 0–23, both frames;
- `push.csv`, `checks.json`, `meta.json`;
- summary CSVs from the analysis.

Report: `reports/pythia/split_edit.md`.
