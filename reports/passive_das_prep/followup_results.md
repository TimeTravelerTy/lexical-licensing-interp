# After the passive test: results (2026-10-07)

Plan, predictions and decision rules: `followup_plan.md`, committed before
anything ran. Generated reports:
- step 1–2: `site8_mechanism_results.md`;
- step 3: `passive_test_results_fill.md` and `layer_fillin_results.md`;
- random controls: `passive_controls_site{N}.md`;
- step 4: `frequency_results.md`.

## Summary

The site-8 variable separates transitive from intransitive verbs in both
voices. It is converted to "object next" in actives but to " by" in
passives, and the split happens in mid- and late-layer MLPs.
- **Depth.** Across eight DAS sites the passive test reads:
  - abstract at 4, 6 and 8;
  - mixed at 10 and 12;
  - surface at 14, 16 and 17.
- **Mechanism (step 1).** In passives the late MLPs that write object starts
  in actives (15, 18, 21, 22) go silent. Most of what object push remains is
  the patched direction's own direct read-out. This is story 1, at the
  boundary of the declared rule.
- **Dose-response (step 2).** On passives, " by" rises linearly through the
  passive range while objects rise slowly and then accelerate. That is
  between the "overshoot" and "coupled" readings; neither prediction held
  exactly.

## Step 1: signed decomposition (site 8, T-donor patch)

**The decomposition is exact.**
- Residual 0.000.
- The LN-scale term is ≤ 0.05.
- The fp32 effects match the bf16 main test:
  - O +0.37 vs +0.36;
  - " by" +0.65 vs +0.65.

| Ō (object starts) | P | N | Net | Actual |
|---|---:|---:|---:|---:|
| actives | +2.19 | −0.80 | +1.39 | +1.41 |
| passives | +1.00 | −0.50 | +0.50 | +0.53 |

**Decision rule on Ō: story 1, nothing downstream pushes toward objects in
passives.**
- Survival s = 0.46 [0.40, 0.50] (< 0.5).
- The cancellation ratio rises from 0.37 to 0.50, a change of +0.13
  [0.06, 0.18] (≤ 0.2).
- **This is at the boundary.** The upper CI of s reaches 0.50, and some
  extra cancellation exists. It is not a clean "no cancellation" result.

**Where the object push goes.**
- The **carry** (the patched site-8 direction read straight through the
  unembedding) gives +0.49 on actives and +0.43 on passives.
  - On passives that is 85% of the net object push.
  - Downstream components add only +0.07 on passives, against +0.90 on
    actives.
- **Six late MLP writers vanish on passives; none changes sign on Ō:**
  - MLP18: +0.45 → +0.06;
  - MLP21: +0.28 → −0.04;
  - MLP15: +0.26 → −0.02;
  - MLP22: +0.18 → −0.02;
  - L19H1: +0.08 → +0.01;
  - the negative MLP23: −0.37 → +0.03.
- " the" has 5 sign changes and " him" has 1. The full list is in
  `decomp_components_site8.csv`.

**" by" (not part of the declared rule).** Net −1.83 on actives and +0.45
on passives.
- MLPs 11, 13, 14, 16 and 17 flip sign. They lower " by" on actives (−0.14
  to −0.47) and raise it on passives (+0.15 to +0.30).
- MLP23 lowers " by" on passives (−0.49).
- The voice-dependent conversion of the site-8 value therefore runs through
  the same mid-layer MLPs in opposite directions.

**Downstream coordinates (T donors).** These are ratios of the downstream
coordinate change to the site-8 change.

| Frame | Δz12 / Δz8 | Δz17 / Δz8 |
|---|---|---|
| actives | 0.88 | 0.74 |
| passives | 0.60 | 0.24 |

- **Predicted:** the passive/active ratio for Δz17 ≤ 0.4. **Observed: 0.32
  (held).**
- **Predicted:** passive Δz17 ≈ 0.1. **Observed: 0.22 [0.20, 0.23]
  (missed; twice the prediction).**
  - The site-17 coordinate moves more on passives than linear scaling from
    D(O) implied.
  - Objects still rise less than a direct site-17 patch of that size would
    give.

## Step 2: dose-response at site 8

The bad-passive slopes, per z unit, are compared with the prediction:

| Quantity | Observed | Predicted |
|---|---|---|
| " by" slope over the passive range R_p = [0.10, 0.45] | +0.92 [0.81, 1.04], linear (R² 0.996) | ≈ +0.8, R² ≥ 0.95 ✓ |
| Object slope over R_p | +0.27 [0.20, 0.33] | ≈ +0.4 |
| Object slope above the range, R_a = [0.45, 1.50] | +0.61 [0.54, 0.69] | ≈ +0.4 (no change from R_p) |
| Ratio of the R_p to R_a object slopes | 0.43 [0.34, 0.51] | ≥ 0.5 for "coupled" |
| " by" slope above the range | +0.48 [0.42, 0.54] (flattens) | |

- **Outcome: intermediate.** It is neither overshoot (ratio ≤ 0.25) nor
  coupled (ratio ≥ 0.5).
- Objects rise slowly through the passive range and accelerate beyond it.
  " by" does the opposite.
- Selectivity inside the passive range is 3.4 (I predicted about 2).
- **Actives are the mirror image.** On intransitive actives, objects jump at
  low z (+3.1 per z in R_p, +1.4 above), and " by" falls.
- The coordinate is the same, but the response curves differ by frame.

## Step 3: layer fill-in

D on primary bad passives; full CIs in `layer_fillin_results.md`.

| Site | Active IIA (per fold) | D(O) | D(" by") | Outcome | Passive / active z gap (mean per pair) |
|---:|---:|---:|---:|---|---:|
| 4 | 0.45 | +0.10 | +0.58 | abstract | 0.86 |
| 6 | 0.66 | +0.23 | +0.69 | abstract | 0.54 |
| 8 | 0.78 | +0.36 | +0.72 | abstract | 0.42 |
| 10 | 0.82 | +0.69 | +0.77 | mixed | 0.43 |
| 12 | 0.90 | +1.17 | +0.81 | mixed | 0.39 |
| 14 | 0.95 | +2.40 | +0.46 | surface | 0.32 |
| 16 | 0.97 | +3.43 | −0.47 | surface | 0.25 |
| 17 | 0.98 | +3.58 | −0.93 | surface | 0.21 |

Against the predictions:
- D(O) rises monotonically with depth. **Held.**
- D(" by") turns negative between 14 and 17. **Held**: first negative at
  16. At 14 it is +0.46, below δ.
- The first site with O RISE is 10 or 12. **Held: 10.**
- Per site:
  - 4 and 6: predicted abstract or unresolved, observed **abstract**;
  - 10: predicted abstract or mixed, observed **mixed**;
  - 14: predicted mixed, observed **surface** (missed: " by" is +0.46, not
    a RISE);
  - 16: predicted surface, observed **surface**.
- The norm-matched random control (100 draws, split 0, per site): **DAS
  beats the null at all eight sites** on every readout classified RISE or
  FALL. The null's max |D| is 0.04–0.13, against DAS effects of 0.10–3.6
  nats (`passive_controls_site{N}.md`; the first fill-in submission collided
  with the plan regeneration and was rerun as jobs 8927007–8927011).

The passive/active gap ratio falls steadily with depth. Early on, passive
participles are separated along d almost as much as active ones (0.86 at
site 4). The passive separation then shrinks as the variable turns into
"object next".

## Step 4: frequency

Over 64 pairs, the passive z gap rises with participle Zipf at every site:
- +0.054 to +0.089 z per Zipf unit, every CI above 0;
- at site 8, +0.065 [0.037, 0.099], about 0.25 z across the 1.35–5.18
  Zipf range.

The passive/active ratio also rises with Zipf at sites 8–17 (+0.05 to
+0.07 per unit). At site 4 it falls (−0.07), because the active gap there
rises faster with frequency than the passive gap.

The per-pair passive gap does not reliably track Pythia's behavioural
passive margins:
- Spearman 0.02–0.24 across sites and measures;
- only site 4 has CIs that touch 0.00 from above, out of 24 tests.

## Caveats

- **Step 1 captures direct effects only.** The vanishing MLP writers could
  be silenced by something upstream. That is step 5 (path patching).
- **The story-1 call is at the boundary** of the declared survival
  threshold.
- **The step-1 "carry" is not a downstream mechanism.** It is d_8's own
  projection onto the object-start unembedding. On passives it is most of
  the remaining object push.
