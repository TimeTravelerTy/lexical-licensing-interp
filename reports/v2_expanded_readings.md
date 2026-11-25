# v2 on the expanded evaluation set, read by the audited criteria (Nov 25)

The paper's evaluation set: 224 pairs held out from both instruments' fitting (`data/das_v2/expanded_eval_pairs.csv`).
The primary population is the **89 plain pairs**, 59 original and 30 new; 23 of the new pairs are lemma-fresh. The 128
prepositional pairs are reported as their own group. v2 is the paper's instrument; v is shown beside it on the same
pairs and draws. Criteria are those of `reports/criteria_audit.md`, including the OR/TC amendment and the classifier
erratum below. Numbers are nats with 95% CIs. G is the natural good − bad gap on the same items and draws.

Outputs:
- `results/{m}/das_v2/expanded_eval/`: `transfer.csv`, `outcomes.csv`, `staging.json`, `paired.csv`, `retention.csv`
  and `checks.json`, plus `constructions_{v2,v}_{eval,eval_new}/` and `split_edit/`.
- Split-edit reports: `reports/{m}/v2_expanded/`.

## Checks (all pass)

- **Natural readouts.** Every v2 passive chunk's unpatched readouts equal the stored expanded v run's exactly: max |Δ| 0 in all 12 chunks; the construction runs also give 0.
- **v's stored expanded transfer** is reproduced by this analysis to 4e-16.
- **Paired draws.** The paired v2 − v draws give the same D as `transfer.csv`: max |Δ| 0.
- **Split edit.** The split edit's full-edit D equals the passive run's at every site. For example, Pythia site 8 *by* is 0.80 in both, Gemma site 5 0.49 and OLMo site 9 0.47.
- **TC base-form gate.** It passes at every site, for both instruments, in all three models.

## Erratum before the readings

As first committed, the audited classifier checked "positive, magnitude unresolved" before NO RISE. A significantly
positive shift whose 90% CI lay inside the margin was then never NO RISE. That contradicts the audit's own statement
that only the margin's unit and CI testing changed from the old classifier, and the class's name. Equivalence is now tested
first (`criteria_audit.md`, erratum). Only O classes change, and with them some staging outcomes ("unresolved" → "abstract"
at shallow sites). I found this while reading the output, so §2 shows S1 under both orders.

Codex's objection, which is fair: the frozen text and code agreed with each other. The old classifier had no
"positive" class, so it sets no precedent for the overlap. On that view equivalence-first is a post-output amendment, and the
frozen order should stay primary with equivalence-first as a sensitivity row. It matters for two S1 verdicts: OLMo v2 and Pythia v
(see §2). **Decision (Nov 25, author): equivalence first is primary.** The frozen order is reported as a sensitivity
column in §2, and the paper discloses the amendment and both S1 readings. The S1 verdicts for OLMo v2 and Pythia v hold only
under the amended order and should be stated that way.

## 1. Passive *by* transfer (claim 1): v2, eval plain (89)

| Model | Site | G | D(*by*) | D / G | T-arm recovery | *by* | D(O) [δ_O] | O |
|---|---|---|---|---|---|---|---|---|
| Pythia | 8 | 0.76 [0.48, 1.03] | 0.80 [0.70, 0.89] | 1.05 [0.77, 1.65] | 0.89 | RISE | 0.29 [0.23, 0.36] [0.50] | NO RISE |
| Pythia | 12 | | 0.84 [0.75, 0.93] | 1.11 [0.82, 1.73] | 0.88 | RISE | 1.00 [0.86, 1.15] [0.54] | RISE |
| Pythia | 17 | | −1.14 [−1.35, −0.94] | −1.51 | −1.51 | FALL | 3.76 [0.59] | RISE |
| Gemma | 5 | 0.39 [0.15, 0.62] | 0.49 [0.38, 0.61] | 1.27 [0.77, 3.08] | 1.26 | RISE | 0.87 [0.71, 1.03] [0.57] | RISE |
| Gemma | 8 | | 0.68 [0.59, 0.78] | 1.77 [1.12, 4.32] | 1.62 | RISE | 1.16 [0.55] | RISE |
| Gemma | 12 | | 0.95 [0.84, 1.06] | 2.45 [1.55, 6.00] | 1.99 | RISE | 1.73 [0.57] | RISE |
| Gemma | 15 | | 0.90 [0.82, 1.00] | 2.34 [1.46, 5.83] | 1.98 | RISE | 2.94 [0.58] | RISE |
| OLMo | 9 | 0.53 [0.28, 0.79] | 0.47 [0.42, 0.53] | 0.89 [0.61, 1.70] | 0.80 | RISE | 1.56 [1.43, 1.69] [0.54] | RISE |
| OLMo | 11 | | 0.25 [0.19, 0.30] | 0.46 [0.30, 0.91] | 0.36 | RISE | 2.35 [0.59] | RISE |
| OLMo | 12 | | −0.78 [−0.92, −0.65] | −1.47 | −1.50 | FALL | 3.62 [0.63] | RISE |
| OLMo | 13 | | −1.84 [−2.07, −1.63] | −3.46 | −3.46 | FALL | 4.08 [0.67] | RISE |

T-arm recovery is (T-patched − unpatched) / G on bad bases, the quantity behind any "reproduces" wording.

- **The *by* edit transfers at every first and second rule site in all three models** (RISE on the gap scale).
  - Pythia 8 / 12 and OLMo 9: the donor contrast D is about the whole natural gap (1.05, 1.11, 0.89), and the T arm alone recovers 0.89, 0.88 and 0.80 of it.
  - The gap's CI is wide (OLMo 9 D / G 0.61–1.70), so "about the size of the natural gap" is the supported wording, not "reproduces".
  - OLMo 11: D / G 0.46, T arm 0.36.
  - **OLMo passes on the natural-gap unit.** Under the old δ it failed, because δ (0.54) exceeded its whole natural gap.
- **Gemma overshoots.** From site 8 on, D is 1.8 to 2.5 times the natural gap, and the T arm alone gives 1.6 to 2.0. Gemma's
  natural *by* gap on this set is small (0.39 [0.15, 0.62]), and the edit moves *by* by about the same number of nats as
  in the other models. Exceeding the gap is supported at 8, 12 and 15; at 5 both ratio CIs contain 1.
  - **Not a uniform *by* boost.** At site 5 the T arm raises *by* by 0.48 on bad bases and 0.05 on good bases. Same-verb
    writes move it by 0.00 and 0.01.
  - Deeper, the selectivity weakens: at 15 the T arm gives 0.76 on bad bases and 0.39 on good bases (same-verb good 0.34).
    So part of Gemma's deep-site effect is a general *by* push.
  - The arms are point estimates; a base × donor interaction CI needs the paired draws.
- **Deepest sites (Pythia 17, OLMo 12–13).** *by* falls below its unpatched level while O rises by 3.6 to 4.1 nats. This is the object-next regime.
- **Prepositional pairs** (128): *by* RISE at every first and second rule site too. D / G: Pythia 1.32 / 1.04, Gemma
  1.38 / 1.80, OLMo 1.20 / 0.73.

## 2. Staging (S1): depth profile, v2, eval plain

Every frozen site of the run is included. A = abstract (*by* RISE, O NO RISE), M = mixed (both RISE), S− = surface with
*by* suppressed, ? = neither.

| Model | Sites (equivalence first, primary) | S1 | S1, frozen order (sensitivity) |
|---|---|---|---|
| Pythia v2 | 0 ?, **1–9 A**, 10 ?, 11–14 M, **16–17 S−** | holds | holds (A = 1–3) |
| Gemma v2 | **0–4 A**, 5–15 M (no surface site within the frozen range) | fails | fails (A = 2) |
| OLMo v2 | 0 ?, **1–4 A**, 5 ?, 6–11 M, **12–15 S−** | holds | **fails** (no A) |

- **Pythia and OLMo order the regimes the same way:** *by* rise with small O first, then both, then O with *by* suppressed.
- **Gemma shows the first two regimes only.** Its frozen sites stop before any site where *by* stops rising.
- S1 tests only abstract before surface; where the mixed sites fall is descriptive.
- **"Abstract" means a *by* rise with O inside the small-effect margin, not O = 0.** At Pythia 8, O rises 0.29 [0.23, 0.36] nats,
  inside δ_O = 0.50, which is 12% of the edit's active O effect. The main text should use that wording.
- **v (rule sites only):**
  - Pythia: 8 A, 12 M, 17 S−; S1 holds (fails under the frozen order: 8 is "?").
  - Gemma: M at every rule site.
  - OLMo: 9 M, then surface.
- The lemma-fresh new verbs give the same broad curve: Pythia A 1–8 (9 becomes "?"), OLMo A 1–4, Gemma A 0–4.

## 3. OR / TC (claim 2): does the edit reproduce the construction's natural profile?

- **L** (closure vs PREP) is RISE in every cell except three deep v cells: Pythia TC 17, and OLMo OR 12 and TC 12.
  - At the first rule sites the closure numerator itself rises: OR 'MAIN' +1.60 Pythia, +0.46 Gemma, +1.40 OLMo; TC END +0.87, +0.25, +0.65.
  - At Pythia 17 and OLMo 12–13 the numerator falls (Pythia OR −1.33, TC −1.53), and L rises only because PREP falls more.
    There, an L RISE is not "closure licensed".
- **O against the construction's own natural O gap** decides the reading:

| Model | Construction | First rule site, v2 | Deeper sites |
|---|---|---|---|
| Pythia | OR | **licensing-like** (L 0.89 G_L; O 0.98 G_O, D_O − G_O [−0.27, +0.21]) | L RISE plus O overshoot (O 1.95 to 2.64 G_O) |
| Pythia | TC | L RISE plus O overshoot (O 2.1 G_O; TC's natural O gap is 0.42) | same, growing |
| Gemma | OR | L RISE plus O overshoot from site 5 (O 2.1 G_O) | same |
| Gemma | TC | L RISE plus O overshoot | same |
| OLMo | OR | **licensing-like** (L 0.78 G_L; O 1.15 G_O, D_O − G_O [−0.02, +0.50]) | L RISE plus O overshoot |
| OLMo | TC | L RISE plus O overshoot (natural O gap 0.04 [−0.22, …]) | same |

v gives the same readings at every first rule site.

- **At the first rule site of Pythia and OLMo, L rises and an O overshoot is not detected.** This is not equivalence: OLMo's
  interval allows up to 0.5 nats beyond natural.
- **Everywhere else, closure is licensed but the object push exceeds what natural transitives show.**
- **New verbs only** (`constructions_*_eval_new`): L is RISE except at five deep cells (v2 OLMo OR 12; v OLMo OR 12, TC 12, TC 13 and Pythia TC 17 read surface). At the first OR sites, O moves just past natural: lo95(D_O − G_O) is 0.06 / 0.17 for v2 in Pythia / OLMo, and 0.12 / 0.05 for v. Those cells read "plus overshoot".

## 4. Split-edit dissociation (claim 5): v2, eval plain

- Δ_by = share_on(*by*) − share_off(*by*), applied where the full *by* is RISE.
- Δ_O = share_off(O) − share_on(O), applied where the full O is RISE.
- Interaction = full − on − off.

| Model | Site | Δ_by | Δ_O | Interaction (*by*) | Holds |
|---|---|---|---|---|---|
| Pythia | 4 | −0.18 [−0.23, −0.13] | n/a (O NO RISE) | −0.11 | no (off > on) |
| Pythia | 6 | +0.02 [−0.03, +0.06] | n/a | −0.16 | no (unresolved) |
| Pythia | 8 | −0.07 [−0.11, −0.03] | n/a | −0.19 | no (off > on) |
| Pythia | 12 | **+0.34 [0.28, 0.40]** | **+0.61 [0.57, 0.66]** | −0.08 | **yes** |
| Pythia | 17 | n/a (*by* FALL) | +0.38 [0.37, 0.39] | | yes (O only) |
| Gemma | 5 / 8 / 12 / 15 | −0.54 / −0.23 / −0.29 / −0.41 | +0.70 / +0.45 / +0.55 / +0.64 | | no (off carries both) |
| OLMo | 9 | +0.03 [−0.02, +0.08] | +0.13 [0.11, 0.14] | −0.07 | no (Δ_by unresolved) |
| OLMo | 11 | **+0.81 [0.62, 1.06]** | **+0.35 [0.32, 0.37]** | −0.13 | **yes** |
| OLMo | 12 / 13 | n/a (*by* FALL) | +0.41 / +0.55 | | yes (O only) |

- **At Pythia 12 and OLMo 11 (each the second rule site), the on-axis write gives the larger standalone *by* increase and the
  off-axis write the larger standalone O increase.** The pieces overlap and interact, so this is a ranking of standalone effects, not a split of the full effect.
- At each model's first rule site, the off-axis piece alone recovers as much of *by* as the on-axis piece, or more.
- **In Gemma, the off-axis piece carries both readouts at every site.**
- The "O only" rows meet the rule as written, but they are not counted as dissociations in the text. They are reported as
  off-axis dominance of O alongside full-edit *by* suppression.

## 5. New verbs (claim 6)

Claim 1 on the 30 new plain pairs, v2:

| Model | Site | D(*by*) | G | D / G | *by* class |
|---|---|---|---|---|---|
| Pythia | 8 / 12 | 0.85 / 0.89 | 0.92 | 0.92 / 0.96 | RISE / RISE |
| Gemma | 5 / 8 | 0.42 / 0.62 | 0.42 | 1.01 / 1.47 | RISE / RISE |
| OLMo | 9 / 11 | 0.45 / 0.22 | 0.70 | 0.65 / 0.32 | RISE / positive, magnitude unresolved |

- **Lemma-fresh (23):** the same classes, except OLMo 11 is RISE (lo95(D − 0.2 G) just above 0).
- **Gemma's lemma-fresh natural gap** has lo95 below 0 (0.33 [−0.05, …]), so no ratio is shown there.
- **D(*by*) in nats is about the same for new and original verbs.** Pythia 8: 0.85 vs 0.77. OLMo 9: 0.45 vs 0.48. The natural
  gaps differ (OLMo new 0.70 vs original 0.45), so D / G differs.

**Retention** (`retention.csv`): D_new − 0.5·D_orig on one set of draws, v2, *by*.

| Model | Site | D_new / D_orig, new (30) | D_new − 0.5 D_orig, new | Same contrast, lemma-fresh (23) |
|---|---|---|---|---|
| Pythia | 8 | 1.10 [0.93, 1.30] | 0.46 [0.34, 0.60] | 0.43 [0.30, 0.57] |
| Pythia | 12 | 1.09 [0.93, 1.25] | 0.48 [0.36, 0.60] | 0.42 [0.30, 0.53] |
| Gemma | 5 | 0.81 [0.53, 1.23] | 0.16 [0.02, 0.34] | 0.15 [0.00, 0.31] |
| Gemma | 8 | 0.86 [0.67, 1.11] | 0.26 [0.13, 0.42] | 0.23 [0.12, 0.34] |
| OLMo | 9 | 0.94 [0.76, 1.13] | 0.21 [0.13, 0.29] | 0.21 [0.13, 0.30] |
| OLMo | 11 | 0.87 [0.57, 1.25] | 0.10 [0.02, 0.17] | 0.09 [0.02, 0.17] |

- **New verbs keep at least half of the original-verb effect at every first and second rule site**, including lemma-fresh.
  Every new / original ratio CI contains 1.
- **Exception:** Gemma 5 lemma-fresh's lower bound is 0.00, so that row is borderline.
- **v:** the same, except OLMo 11 is inconclusive (−0.03 to 0.12).

## 6. v2 against v (paired, eval plain, one set of draws)

| Model | Site | *by* v2 − v | O v2 / v |
|---|---|---|---|
| Pythia | 8 | +0.05 [0.01, 0.09] | 0.73 [0.67, 0.78] |
| Pythia | 12 | +0.02 [−0.01, 0.05] | 0.82 [0.78, 0.86] |
| Gemma | 5 | −0.04 [−0.11, 0.02] | 0.59 [0.51, 0.67] |
| Gemma | 8 | +0.08 [0.04, 0.11] | 0.84 [0.81, 0.88] |
| OLMo | 9 | +0.05 [0.03, 0.07] | 0.90 [0.87, 0.93] |
| OLMo | 11 | +0.11 [0.09, 0.13] | 0.98 [0.96, 1.01] |

- **At the rule sites, v2 raises *by* more than v or by an amount that cannot be told apart** (Gemma 5: −0.04 [−0.11, +0.02]).
  **It pushes the object less or equally** (OLMo 11: 0.98 [0.96, 1.01]). The largest gain is in Gemma 5, where the O push is cut by 41%.
- At the deepest sites, v2 suppresses *by* more: Pythia 17 −0.16, OLMo 13 −0.75.

## What this means for the paper's claims

1. **C1 (an edit in the transitivity subspace licenses the passive *by* phrase).** Holds at both rule sites in all three
   models on the held-out expanded set, in the gap unit.
   - "About the size of the natural gap": Pythia 8 / 12 and OLMo 9.
   - "Exceeds it": Gemma 8–15. Partly a general *by* push at deep sites.
   - "About half of the donor contrast": OLMo 11 (T arm 0.36).
2. **OR / TC.** At the first rule sites, the edit raises closure and lowers PREP in every model.
   - Only OR in Pythia and OLMo shows no detectable object overshoot. On the full eval set; new verbs alone overshoot slightly.
   - Elsewhere the object push exceeds the natural gap, mostly more so with depth (Gemma OR dips from 5 to 8).
   - At the deepest sites, L rises only because PREP falls faster than closure.
3. **Staging.** *by* with small O, then both, then O with *by* suppressed, in depth order, in Pythia and OLMo. OLMo's S1 depends on the
   classifier order. Gemma shows the first two regimes only.
4. **Split edit.** The dissociation holds at the second rule site (Pythia 12, OLMo 11), not at the first, and not in Gemma.
   This is weaker than "the on-axis piece is the licensing piece".
5. **Forking paths** (as in the audit):
   - The criteria were fixed before this output was read, except the classifier erratum (equivalence first adopted as primary after the output was read; frozen order kept as sensitivity, see top).
   - 59 of the 89 primary pairs are originals whose 126-pair readings were seen. The new and lemma-fresh rows are the cleaner test, and they agree.
