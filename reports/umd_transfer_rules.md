# u_md transfer (OR, TC, in-range, filler-gap) in all three models: rules, declared before running (2026-11-17)

Paper plan: Status table, "u_md transfer next to v in all three"; Kumon & Yanaka defense (u_md is training-free).
Passive *by* with u_md is already done (split edit, Settled 10; Pythia direction check). Nothing below has been run.
Branch `paper`.

## How the results are read

As in `split_edit_replication_rules.md`: the readings below are the confirmatory floor. Every site and instrument
(v and u_md) is reported as curves with CIs. Model differences are findings. Anything chosen after seeing the data
is labelled exploratory.

## Directions

u_md for each (split, fold) basis is the unit class mean difference of that basis's own training items, as stored
by the split edit (`results/{model}/split_edit/directions_site{s}.npz`). `scripts/export_umd_bases.py` writes it in
the format of `bases_rank1.npz` and checks that each stored v equals the transfer run's basis with the same key
(max deviation 0.0 in all three models, checked 2026-11-17). An interchange patch along u is invariant to u's sign.

## Runs

The same runners, items, plans, folds and patch procedure as the v transfer runs, with `--bases bases_umd.npz`.
Outputs go to `{root}/umd/{constructions,inrange,fillergap}`. The natural passes are recomputed. For filler-gap this
is needed, not just a check: the in-range targets come from each verb's projection onto the direction, stored by the
natural stage. The filler-gap gates are recomputed with `analyze_fillergap_r5.py gates`: the natural gate and kept
verbs depend on readouts only and must equal v's (`gates.json` compared byte for byte); the B7 base-form gate is
along the direction, so it is read along u. (Procedure corrected 2026-11-17 after the first smoke run, which tried
to reuse v's projections and stopped at the runner's bases-hash check; no result was read.)

| | Pythia-1.4B | Gemma-3-1B | OLMo-2-1B |
|---|---|---|---|
| Sites | 4, 6, 8, 12, 17 (where u was stored) | 0, 1, 2, 3, 4, 5, 8, 12, 15 | 9, 11, 12, 13 |
| Rule sites | 8 / 12 | 5 / 8 | 9 / 11 |
| Root (δ, items, folds) | `results/das_round2` | `results/gemma/das/transfer` | `results/olmo/das/transfer` |
| Filler-gap out (v) | `results/round5/fillergap` | `{root}/fillergap` | `{root}/fillergap` |

Pythia's v transfer also has sites 10, 14 and 16. u was not stored there, so they are not run.

## Margins

- **Declared margin (primary):** the analysis scripts' own, unchanged. O is classed against δ_s = 0.2 × v's active O
  effect at the site; L against δ_L = 0.2 × the natural good − bad gap. This is the same bar v met.
- **Own-effect margin (secondary, declared now):** O classed against δ_u = 0.2 × u_md's held-out active D(O) at the
  site, read from `results/{gemma,olmo}/split_edit/summary_active.csv` (condition umd) and
  `results/pythia/das_direction_check/summary_active_patching.csv` (u_md). Same classifier, stored CIs. L margins are
  natural-gap based and do not change. Reason: u_md pushes less than v on actives (0.35–0.76 of v's D(O)), so the
  primary margin holds it to v's effect size. This reading is now declared, not post hoc.
- **Amendment (2026-11-17, after the Codex review of the split edit, before any full run or reading of smoke
  numbers):** the own-effect margin treats u_md's active effect as exact, which made one split-edit reading hinge on
  0.0004 nats. So an own-effect RISE counts only if it holds with δ_u taken from the upper 95% bound of u_md's active
  D(O) (conservative). Classes at the point estimate and at multipliers 0.15 and 0.25 are reported as sensitivity.
  The primary margin is unchanged.

## Confirmatory readings

- **T2-umd.** OR and TC D(O) (T − I donors, bad bases) classed RISE at both rule sites, primary margin.
  Also read with the own-effect margin.
- **T3-umd.** The natural gate passes (it equals v's), the base-form gate is read along u (as for v, a matrix-wh reading at a site that fails it is "inconclusive"), and the in-range filler-gap patch along u_md is classed
  licensing-like at one or more of the model's declared sites (Pythia 8, 12, 17; Gemma 5, 8, 12, 15; OLMo 9, 11,
  12, 13).
- **Ratio.** u_md / v for D(O) per construction and site, with a paired bootstrap CI where both are on the same
  rows. No threshold is declared; reported next to the active ratio.

## Expected direction (stated to be checked, not criteria)

- u_md transfers to OR and TC wherever v does, at about the active ratio (0.35–0.76).
- In OLMo, u_md's O effect on OR / TC is RISE at 9 and 11, as on passives (u_md passive O RISE from 7).

## Checks

- The u_md bases file passes the mapping and unit-norm checks (above). Its SHA-256 is recorded.
- The natural passes reproduce v's runs exactly (same model, items, dtype): max abs difference of every natural
  readout ≤ 1e-6 (constructions, in-range, filler-gap).
- The filler-gap natural readouts reproduce v's (≤ 1e-6) and `gates.json` equals v's byte for byte.

## Outputs

`{root}/umd/...`, large patch dumps gitignored and SHA-256 listed as in v's runs. Report:
`reports/umd_transfer.md`, with the readings added to `reports/replication_criteria.md`.
