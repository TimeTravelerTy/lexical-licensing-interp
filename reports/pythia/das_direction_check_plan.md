# DAS direction check (Pythia-1.4B): rules, declared before running (2026-11-12)

Exploratory check of the rank-1 DAS directions v against the subspace-patching
illusion (Makelov et al. 2024): an interchange along v can move the readout
through a direction that is dormant in natural runs, by pulling in variation
from a nearby causally used direction. Three checks: does v line up with the
natural class axis, does the class axis alone patch as well as v, and does
removing the v coordinate in natural runs remove the behavioural gaps.

Nothing below has been run. No existing DAS output is modified; everything is
written to new files (`results/pythia/das_direction_check/`,
`reports/pythia/das_direction_check.md`). Branch `das-direction-check`, off
`leace-necessity`. Numbers only: the report has tables and the checks, no
reading.

## Directions

Sites (DAS convention, site s = residual stream entering layer s): **4, 6, 8,
12, 17**. Existing final-run bases, rank 1, all 3 split seeds × 5 folds = 15
bases per site:

| Site | Run | Epochs |
|---:|---|---:|
| 4 | `results/das_round2/final_strict_site4` | 1 |
| 6 | `results/das_round2/final_strict_site6` | 2 |
| 8 | `results/das_round2/final_strict_site8` | 3 |
| 12 | `results/das_round2/final_strict_site12` | 2 |
| 17 | `results/das_round2/final_strict` | 1 |

Per site s and basis (split k, fold f):

- **v**: the unit DAS vector `r1_s{k}_f{f}`.
- **u_md**: the class mean-difference direction, normalised. Mean
  verb-position state (last prompt token, natural run, site s) of the
  basis's DAS training items with class T minus the same for class I.
  - Training items are exactly those v was trained on: `run_das_round2`
    items (29 strict pairs, "<Subj> has/have <participle>"), pairs outside
    fold f of split k, the 6 training subjects (`fold_indices`).
  - States come from `run_das_round2.setup` (bf16 model, fp32 states), as
    in training.
- **u_leace**: the saved single-site LEACE eraser at site s.
  - Primary: the in-sample eraser, fit on all 126 passive verb pairs
    (`results/pythia/leace_insample/erasers.npz`, `single_edit` and
    `single_read`, each normalised; both reported).
  - Secondary: the 5 cross-fitted erasers (`leace_necessity/erasers.npz`),
    mean |cos| over folds.
  - Note: these erasers were fit on passive `was` items (verb class at the
    participle), not on the active DAS items.

## 1. Natural class axis

Per site, over the 15 bases: mean, min and max of
- |cos(v, u_md)|;
- |cos(v, u_leace)| (edit and read vectors);
- |cos(u_md, u_leace)| (edit and read), for reference.

**References** (same table):
- **Random isotropic:** |cos(v, r)| for 10,000 random unit vectors r per
  basis (numpy seed 0): mean and 95th percentile, pooled over bases.
- **Shuffled-label mean difference:** per basis, 100 permutations of the
  class labels over the training verbs (verb-level, all of a verb's items
  move together; `np.random.default_rng(1000 * (5k + f) + p)`). u_md is
  recomputed under each permutation. Reported: mean and 95th percentile of
  |cos(v, u_md_perm)|.
  - This reference keeps the anisotropy of the residual stream, which the
    random isotropic one does not.

**Extra column:** variance of the item projections onto v, as a multiple of
the mean per-dimension variance (trace / d). Computed over all 406 active
items. The same is reported for u_md.

## 2. Component patching: u_md vs v

Same interchange as training and as the passive test
(`Runner.patched`, last token):
h ← h + ((h_src − h)·u) u,
with u = v or u = u_md of the same basis.
- bf16 model, fp32 patch arithmetic.

### Actives (held-out)

Same items, folds and swaps as the final runs (`run_das_round2.evaluate`
on the held-out pairs of each basis). Per direction:
- held-out cross-class IIA, mean over the 15 bases;
- median positive-control fraction (mean over bases);
- active effect: Δ log P(O) for intransitive base ← transitive source.
  - Computed by `analyze_das_round2_results.per_pair`: mean within pair and
    split, then over splits.
  - Then the mean over the 29 pairs, with a bootstrap CI over pairs
    (2,000 draws, seed 0).
  - This is the quantity behind δ_s.

### Passives

Same rows as the stored passive test (`build_passive_test.py`; the plan
regenerates with the stored SHA-256 `fa17…6ff5`):
- primary population (64 pairs with a plain bad verb);
- bad passive bases;
- conditions T and I;
- all 3 splits;
- the basis assigned per (item, split).

D(R) = mean ΔR after T donors − mean ΔR after I donors. It is computed for
R ∈ {log P(O), log P(" by"), log P("."), M}.

Inference: the three-way cluster bootstrap of `analyze_passive_test.py`
(base pairs within band, contexts, donor pairs; 2,000 draws, seed 17).

### Ratios

- u_md effect / v effect, for:
  - active IIA;
  - active Δ log P(O);
  - passive D(O);
  - passive D(" by").
- Passive ratios get a CI from the shared bootstrap draws: the ratio is
  taken per draw.
- Active Δ log P(O) ratio: the ratio of the pair means, with a paired
  bootstrap over pairs.
- The declared class of D (RISE / FALL / NO RISE / unresolved, with the
  site's δ_s from `frozen_config_site*.json` or `deltas.json`) is listed
  for both directions. The rule is unchanged.

## 3. Necessity of v: mean ablation in natural runs

Ablation at site s, last token:
h ← h + (μ − h·u) u.
- μ is the grand mean of h·u over all items of the frame being evaluated,
  per basis:
  - actives: all 406 DAS items, both classes, all 7 subjects;
  - passives: every good and bad prompt of the primary population.
- Applied to every prompt of the frame. No donors.

**Cross-fitting:**
- Actives: each item is ablated with the 3 bases (one per split) whose
  fold holds its pair out. Readouts are averaged over the 3.
- Passives: each item (good and bad prompt) is ablated with the basis the
  plan assigns to (item, split), and readouts are averaged over the 3
  splits.

**Directions ablated:**
- v (the test);
- u_md and 10 random isotropic unit directions per basis (torch CPU
  generator, seed 11), as references.

### Measures

- **Actives.** Object gap G_O = mean over the 29 pairs of
  (mean log P(O) over the T verb's items − the same for the I verb).
  - Natural and ablated.
  - Share removed = 1 − G_O(ablated) / G_O(natural).
  - Same for M.
  - CI: bootstrap over pairs (2,000 draws, seed 17); the share is taken
    per draw.
- **Passives.** By gap G_by = good − bad log P(" by").
  - Mean within pair over contexts, then over the 64 primary pairs.
  - Natural and ablated.
  - Share removed = 1 − G_by(ablated) / G_by(natural).
  - CI: two-way bootstrap (pairs within band, contexts; 2,000 draws,
    seed 17), the `analyze_passive_test.py` machinery.
  - log P(O) gap and M are also written to the CSV.
- **Random directions:** mean and min–max of the share removed over the
  10 draws (point estimates).

## Checks (run before any table is read)

1. Bases: the fold assignment in each site's `pairs_folds.csv` equals the
   frozen one (as in `run_passive_test.site_bases`). The plan regenerates
   with the stored SHA-256.
2. **v reproduction, actives:** held-out IIA per basis within 0.01 of the
   stored `summary.csv` (`control == das`, rank 1). The active Δ log P(O)
   for v is within 0.05 of δ_s / 0.2.
3. **v reproduction, passives:** D(O) and D(" by") point estimates for v
   within 0.02 nats of the stored `transfer_summary.csv` (D_bad, primary,
   all bands): `passive_test` for sites 8, 12, 17 and `passive_test_fill`
   for sites 4, 6.
4. **Self-patch:** an interchange with the base's own state along u_md
   leaves the readout unchanged (max |ΔM| = 0).
5. **Identity ablation:** ablating with μ replaced by each item's own
   projection leaves the readout unchanged (max |ΔM| = 0 up to fp32 → bf16
   rounding; the value is reported).
6. u_md and v are unit norm (|‖u‖ − 1| < 1e-5).

**If a check fails,** that site's rows are reported with the failure next
to them. They are not dropped.

## Outputs

`results/pythia/das_direction_check/`:
- `cosines.csv`: per site × basis;
- `active_summary.csv`: per site × direction × basis;
- `active_swaps.csv.gz`: held-out swaps, both directions;
- `passive_patches_site{s}.parquet`: plan row, direction, readouts;
- `ablation_active.csv.gz`, `ablation_passive_site{s}.parquet`;
- `directions_site{s}.npz`: u_md per basis;
- `checks.json`, `meta.json`.

Report: `reports/pythia/das_direction_check.md`, tables for items 1-3 and
the checks.

## Addendum (2026-11-12, after a smoke run at site 8 on a 256-item passive subset, before the full run)

The rules above are unchanged. Two implementation details are fixed now.
- **Checks 4 and 5:** the reference is an unpatched forward of the same
  batch (zero basis), not the separate natural pass. Against the natural
  pass, the smoke run showed max |ΔM| 0.42 (self-patch) and 0.06 (identity
  ablation). That is bf16 batch-composition noise, the same size as the
  stored passive test's held-out recompute (max 0.41). The difference from
  the natural pass is reported as its own column.
- **Natural gaps in item 3** come from unpatched readouts run in the same
  batches as the ablations (a zero-basis "none" condition). The gap from the
  separate natural pass is written alongside. Bootstrap percentiles drop
  NaN draws and count them.
