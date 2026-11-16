# Split edit, u_md patching and frame invariance on Gemma-3-1B and OLMo-2-1B: rules, declared before running (2026-11-16)

Paper plan tasks 1 and 3 (`docs/paper_plan.md`). Same procedure as Pythia's `pythia/split_edit_plan.md` parts A–C,
with one added condition (u_md). Part D (side splits) needs LEACE runs and is left for those. Nothing below has
been run. Branch `paper`.

## How the results are read

The criteria below are the confirmatory floor: what was stated before the run. They do not decide what gets
reported. Every site and every instrument is reported as a curve with CIs. The paper describes each model's
mechanism from the full curves, and anything picked after seeing the data is labelled exploratory. A declared
criterion that fails is reported as failed, next to what the curves show.

Why the full depth range: the declared rule sites come from active IIA thresholds. These land at different
relative depths per model (OLMo 9 / 11 are at 0.56 / 0.69 of depth; Pythia 8 / 12 at 0.33 / 0.50). So a claim is
read over the whole curve, with the rule sites marked.

## Runs

`run_split_edit.py --umd` with each model's stored final runs, passive-test plan and items:

| | Gemma-3-1B | OLMo-2-1B |
|---|---|---|
| Sites | 0–15 | 0–15 |
| Primary (`final_strict/`) | 15 | 13 |
| Declared rule sites (0.8 / 0.9) | 5 / 8 | 9 / 11 |
| Root | `results/gemma/das/transfer` | `results/olmo/das/transfer` |
| Plan | `data/gemma/transfer/passive_test` | `data/olmo/transfer/passive_test` |
| Out | `results/gemma/split_edit` | `results/olmo/split_edit` |

- The bases are rank 1 from each site's final run (3 splits × 5 folds).
- Frame invariance (B) runs at every site 0 to n_layers − 1 (Gemma 0–25, OLMo 0–15). It uses the same curated
  LEACE item set as Pythia (126 pairs × 125 contexts). That set is defined by the item list, not by any model's
  behaviour.
- LEACE: none yet, so check B0 is n/a.

## Conditions

For each passive plan row (primary population, bad bases, T and I donors, all splits) and each held-out active swap:

| Condition | Read | Write | What it is |
|---|---|---|---|
| full | v | v | the stored v patch |
| on | v | a·u | v's push, on-axis part |
| off | v | w | v's push, off-axis part |
| u_md | u | u | the class mean-difference interchange (the direction check's u_md patch) |
| u_pas | u_pas | u_pas | patching along the passive class axis, cross-fitted (exploratory, as Pythia's C) |

u is the unit class mean difference of the basis's training items (per split and fold); v = a·u + w.

## Confirmatory readings

**R-A1 (off-axis carries the object push).** At each site where v's D(O) is RISE: the off-axis share of D(O) is ≥ 0.5,
and the on-axis D(O) is not RISE.
- Held if this is true at both rule sites (Gemma 5 and 8; OLMo 9 and 11).
- The count over all RISE sites is reported.

**R-A2 (on-axis carries *by*).** At each site where v's D(by) is RISE: the on-axis share of D(by) is ≥ 0.5.
- Held if this is true at every such site among the rule sites, and in the count over all such sites.
- OLMo's v is NO RISE at 9 and 11, so there R-A2 is read from the count over all sites only (0 and 4–6 are RISE).

**S1 with the on-axis piece and with u_md** (as declared in `olmo/finals_rules.md` §3). One or more abstract
sites, and every surface site deeper than every abstract site. Read over sites 0–15.

**T1 with the on-axis piece and with u_md.** D(by) classed RISE at both rule sites.
- These are secondary instruments. T1 with v stays as declared (`replication_criteria.md`).
- If T1 with on-axis or u_md passes where T1 with v fails, the paper says so in these terms: "passive *by* transfer
  holds along the class axis, while the full v patch is dominated by an off-axis object push".
- The same readings are applied to Pythia from the stored split edit (on) and the direction check (u_md, sites 4,
  6, 8, 12, 17).

**B1 (frame invariance).** cos(u_act, u_pas) is above the sign-flip 97.5th percentile at every site. FI is reported
at every site. The FI decision rule of the Pythia plan is applied at the rule sites (Gemma 5 / 8; OLMo 9 / 11) in
place of Pythia's 8 / 12.

No prediction is made for u_pas (C) or for the push and separation tables.

Expected direction, stated to be checked, not a criterion: if OLMo's early object push rides on w as in Pythia, then
the on-axis and u_md curves move OLMo's abstract range deeper than v's (v is abstract at sites 0 and 4, surface from 7).

## Checks (before any table is read)

As in `pythia/split_edit_plan.md`, checks 1–4, 6 and 7:
- the folds equal the frozen ones;
- write = read reproduces `Runner.patched` bitwise;
- the full condition reproduces the stored held-out active IIA per basis (within 0.01) and the stored passive D(O)
  and D(by) (within 0.02 nats);
- the decomposition is exact (a·u + w = v);
- the C fit sets exclude the listed verbs;
- the LEACE item counts are as stated.

A check that fails is reported next to the affected rows.

## Outputs

`results/{gemma,olmo}/split_edit/`, as Pythia's. Large dumps (patch parquets, active swaps, class states) stay on
TSUBAME with their SHA-256 listed. Reports go to `reports/gemma/split_edit.md` and `reports/olmo/split_edit.md`, and
the readings are added to `reports/replication_criteria.md`.
