# Round 3: results

Plan, predictions and decision rules: `round3_plan.md`, each part committed
before its runs (Part A: `a163716`). Generated reports:
- A1: `round3_reliability.md`;
- A2: `round3_late_projection.md`;
- A3: `round3_dose_decomp.md`;
- A4: `round3_token_rank.md`.

TSUBAME jobs (commit `a163716`): 8938121 (A2), 8938122 (A3), 8938123–8938124
(A4 ranks). A1 and the A4 token split ran locally on committed outputs.

# Part A: cheap checks

## Summary

- **A1.** Every per-pair measure is stable across contexts (R ≥ 0.985), so
  the frequency-vs-behaviour null is **not a measurement failure**. It is
  "no moderate link" for 7 of 24 tests and "no detectable link, moderate
  not excluded" for 16; site 4 vs the LP margin is significant (+0.22).
- **A2.** Along d_8, the transitive/intransitive separation is **present at
  every site through 23 in both voices**. Removing the d_17 component shows
  the voices diverge: the passive separation largely stays (0.79× at site
  17), the active one is **consumed** (0.34× at 17, ≈ 0 by 22). The declared
  rule reads "unresolved" at 14–17, because the ratio rises instead of
  staying flat.
- **A3.** The object acceleration above the passive range is real in the
  logits, but its net source is mostly the **final-LN scale term**. The late
  MLPs (15, 18, 21, 22) do write more objects there, but MLPs 11 and 13
  write fewer by about as much: a **handover**, not a net gain. " by"
  flattens because MLPs 8–17 stop raising it.
- **A4.** Low early IIA is mostly **not a rank limit** (rank 4: +0.07 IIA at
  site 4, +0.04 at site 6; the rule keeps rank 1) and **not a token-count
  effect** (within expansion pairs, multi-token verbs are not worse).

## A1. Split-half reliability

| Measure | Units per pair | R (Pearson, median over 1,000 splits) |
|---|---:|---|
| passive gap, sites 4–17 | 125 | 0.997–1.000 |
| passive gap, Zipf/band-residualized | 125 | 0.997–1.000 |
| active gap, sites 4–17 | 7 subjects | 0.985–0.996 |
| single-prompt " by" preference | 125 | 0.997 |
| released *by* margin | 300 | 0.999 |
| curated LP margin | 125 | 0.989 |

- The per-pair passive gap is also consistent across the three basis
  splits (mean r 0.98–0.99).
- So the disattenuated correlations equal the observed ones to two
  decimals.

| Behaviour | Sites 4–17, r observed | Verdicts |
|---|---|---|
| curated LP margin | −0.06 to +0.22 | no moderate link at 8–17 (6 sites); not excluded at 6; **significant at 4** (+0.22 [+0.00, +0.43]) |
| released *by* margin | +0.01 to +0.17 | no moderate link at 6 (upper bound 0.28); not excluded elsewhere (upper bounds 0.30–0.40) |
| single-prompt " by" preference | +0.06 to +0.20 | not excluded at every site (upper bounds 0.30–0.42) |

**Against the predictions.**
- Reliability ≥ 0.9 for context-averaged measures, ≥ 0.7 for the active gap,
  ≥ 0.8 after residualizing, ≥ 0.9 across bases: **held** (all ≥ 0.98).
- Most tests land in "no detectable link": **held** (16 of 24). Site 4
  significant: **held** (LP margin only).

**Reading.** The null is reportable as "no link of moderate size" for the
LP margin, and as "at most a weak link (r ≲ 0.4)" for the *by* measures.
It is not a "couldn't measure" result. Two limits remain: context-split
reliability shows stability, not validity; and the single-prompt " by"
preference shares contexts with the gap, so their errors may correlate.

## A2. Is the site-8 separation still present late?

Raw projection units, 64 primary pairs (full tables:
`round3_late_projection.md`).

| Site | d_8: passive gap | d_8: active gap | d_8 ratio | d_8⊥17: passive gap | d_8⊥17: active gap | d_8⊥17 ratio | own-site ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 3.67 | 8.76 | 0.42 | 3.31 | 6.85 | 0.48 | 0.42 |
| 12 | 4.03 | 9.81 | 0.41 | 3.13 | 6.06 | 0.52 | 0.38 |
| 14 | 3.94 | 10.05 | 0.39 | 2.82 | 4.97 | 0.57 | 0.32 |
| 16 | 4.09 | 11.42 | 0.36 | 2.79 | 3.50 | 0.80 | 0.25 |
| 17 | 4.20 | 12.26 | 0.34 | 2.63 | 2.30 | 1.15 | 0.20 |
| 20 | 4.07 | 14.16 | 0.29 | 2.45 | 1.32 | 1.87 | — |
| 23 | 3.60 | 14.31 | 0.25 | 1.96 | 0.09 | — | — |

- **d_8.** The passive gap is flat from site 8 to 23 (retention 0.98–1.16);
  the active gap grows to 1.6×. Passive AUC over verb means stays ≥ 0.96.
- **d_8⊥17.** Passives keep 0.79× of the site-8 gap at 17 (AUC 0.96) and
  0.59× at 23 (AUC 0.85). Actives fall to 0.34× at 17 and to ≈ 0 at 22–23
  (AUC 0.51): the early-only component is gone in actives.
- **Declared rule** (ratio within ±0.1 of site 8 while own-site falls):
  "retained alongside the conversion" at 12; **unresolved** at 14, 16 and
  17. There the d_8⊥17 ratio *rises* (+0.08, +0.31, +0.66), because the
  active gap vanishes. The rule did not anticipate this.

**Against the predictions.**
- Present in both voices through 17, and at 20 and 23: **held** on d_8. On
  d_8⊥17, actives fail at 20–23.
- Active d_8 gap retained or growing: **held** (1.40× at 17).
- The d_8 ratio declines but stays above the own-site ratio: **held**
  (0.34 vs 0.20 at 17).
- The d_8⊥17 ratio stays near its site-8 value: **held at 12, missed at
  14–17.**

**Reading.** In actives, the part of the early separation that is not
shared with d_17 is consumed between sites 12 and 20. It is either
rotated into the late "object next" direction or erased. In passives the
same component stays largely in place. This fits the site-8 mechanism
result: the early verb-class value is converted in actives and mostly left
unconverted in passives. The caveat is that this is retained *linear*
separation along one fixed read-out; it is not shown to be the same
functional variable.

## A3. Decomposing the gradual push

Bad passives, coordinate-setting patches at site 8, per z unit (full
tables: `round3_dose_decomp.md`). Both LN versions (per-interval and fixed
σ) agree to two decimals.

| Interval | Ō actual | carry | downstream net (P / N) | LN-scale term | Δ log P(O) | Δ log P(" by") |
|---|---:|---:|---|---:|---:|---:|
| R_p [0.10, 0.45] | +0.50 | +0.47 | +0.09 (+0.77 / −0.68) | −0.06 | +0.27 | +0.92 |
| [0.45, 0.90] | +0.61 | +0.47 | +0.08 (+0.57 / −0.48) | +0.06 | +0.46 | +0.56 |
| [0.90, 1.50] | +0.76 | +0.47 | +0.12 (+0.60 / −0.48) | +0.17 | +0.73 | +0.43 |

**Declared decisions.**
- **Acceleration in the logits: yes.** A = +0.26 [+0.21, +0.32] per z,
  ratio 1.52.
- **Creep = carry: no.** The carry is 94% [86%, 104%] of the net Ō change
  in R_p. But downstream components write in both directions, P 0.77 and N
  0.68 per z, each larger than the carry (0.47), and the rule requires
  smaller.
- **Late-MLP contribution: yes.** MLPs 15, 18, 21 and 22 raise their
  object rate by +0.24 [+0.22, +0.27], 0.93 of A.

**What the rule did not capture.** A splits into carry +0.01, downstream
net +0.03 [−0.03, +0.09] and **LN-scale term +0.23 [+0.19, +0.27]**.
- The late-MLP increase is offset by decreases in MLP11 (−0.14), MLP13
  (−0.09), MLP8 and MLP16. Downstream components hand the object push from
  mid to late MLPs as z rises; they add almost nothing net.
- The net acceleration of Ō is the final LayerNorm scale shrinking as z
  rises, which sharpens the whole output distribution (object tokens have
  high centered logits).
- In log P(O), the acceleration (0.27 → 0.73) is split between the object
  set (LSE_O: 0.18 → 0.39) and the partition function (LSE_all: −0.09 →
  −0.34).

**" by" flattening** (0.83 → 0.09 per z in Ō units; log P 0.92 → 0.43).
- Downstream net falls from +0.96 to −0.20, mainly because the MLPs that
  raise " by" in R_p (8, 9, 10, 12, 13, 14, 16, 17) lower their rates.
  Together, MLPs 11, 13, 14, 16 and 17 lose −0.73 per z.
- MLP23 still lowers " by" above the range, but by less than inside it
  (+0.22 rate change). It is not what flattens " by".

**Against the predictions.**
- Ō rate rises (ratio ≥ 1.25): **held.**
- Creep mostly carry (≥ 60%): **held on net share (94%)**, but the declared
  rule says no because of large two-way downstream writes.
- MLPs 15, 18, 21, 22 give ≥ 50% of A: **held by the rule**, but
  **misleading**. They are offset by MLPs 11 and 13, and the net
  acceleration is the LN term.
- " by" flattens because MLPs 11–17 stop growing (**held**) and MLP23 keeps
  lowering it (**missed**: MLP23's rate falls).

## A4. Robustness of low early IIA

**Rank** (paired by fold × split, 15 runs per site):

| Site | Rank 1 IIA | Rank 2 gain | Rank 4 gain | Rule |
|---:|---:|---|---|---|
| 4 | 0.45 | +0.02 [+0.01, +0.04] | +0.07 [+0.05, +0.09] | rank 1 (gap fraction +0.06 < 0.1) |
| 6 | 0.66 | +0.02 [+0.00, +0.03] | +0.04 [+0.02, +0.07] | rank 1 (+0.05) |

**Token count, expansion pairs only** (6 single vs 15 multi), held-out
cross-class IIA:

| Site | I←T: single / multi | T←I: single / multi |
|---:|---|---|
| 4 | 0.52 / 0.54 | 0.40 / 0.37 |
| 6 | 0.65 / 0.67 | 0.62 / 0.73 |
| 12 | 0.83 / 0.91 | 0.94 / 0.96 |

The multi − single differences are −0.03 to +0.10, with CIs spanning about
±0.3 at sites 4–6.

**Against the predictions.**
- The rank rule keeps rank 1: **held.**
- Rank-4 IIA gain < 0.05: **missed at site 4** (+0.07), held at 6.
- Multi-token IIA ≥ 0.1 below single-token at 4 and 6: **missed.** The
  difference has the opposite sign or is near zero.

**Reading.** At the rank-1 training budget, low early IIA is neither a rank
limit (rank 4 recovers only a small part) nor a visible detokenization
effect. With 6 single-token expansion pairs, the token comparison can
exclude only large differences. The early sites simply carry a weaker,
less complete version of the class variable.
