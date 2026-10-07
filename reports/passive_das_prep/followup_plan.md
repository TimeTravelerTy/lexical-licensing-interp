# After the passive test: declared plan (2026-10-07)

Written and committed before any of steps 1–3 is run. It builds on
`passive_test_plan.md`, `passive_test_results.md` and
`passive_controls_site8.md`. Predictions are mine and are stated before
the data exist.

## 0. Quick checks (done with this commit)

**Why the primary population has only 9 Head pairs.** The binding
constraint is the class rule.
- 17 of the 26 original Head pairs have a `prep_object` bad verb, and so do
  all 6 new Head eval pairs.
- Nothing is lost to single-token/alignment constraints: every Head
  participle is one token, and the passive test drops no item for
  alignment.
- Nothing is lost to the training holdout: training pairs are kept and
  cross-fitted.
- Overlap with DAS training: 8 of the 9 Head plain pairs are the 8
  `orig_head` training pairs. The ninth, bet/appear, was kept out of
  training by hand for its tokenization artifact.
- So Head passive results rest on 8 cross-fitted training pairs plus
  bet/appear. Step 4 replaces the band comparison with a continuous Zipf
  regression for this reason.

**Shuffled-label cosine.**
- `scripts/export_bases.py` writes every rank-1 basis to
  `results/das_round2/bases_rank1.npz` (torch-free).
- `analyze_passive_test.py controls` now computes the shuffled-label vs d
  cosine from that file (mean 0.69, per fold 0.61–0.77).
- The write-ups describe shuffled-label DAS as a weaker copy of d, not as a
  null. The null is the random-direction control.

**δ_s for new sites (confirmed).** δ_s = 0.2 × the held-out Δ log P(O)
(intransitive base ← transitive source, mean over held-out pairs) from the
site's own final run. That gave 0.49 / 0.53 / 0.57 at sites 8 / 12 / 17.

## 1. Signed decomposition of the site-8 effect (first)

**Patches.** These are the site-8 T-donor patches of the passive test, with
the same rows, bases and donors.
- **Passive frame:** primary bad passive bases (64 pairs, all 3 splits).
- **Active frame:** held-out intransitive active bases ("She has
  vanished"), with transitive donors from the same held-out fold and the
  same subject (the held-out positive-control swaps), all 15 fold bases.
- I-donor patches are decomposed too. A D decomposition (T − I) is
  reported as a supplement.

**Terms.** All are direct effects on the last-position logits. Pythia's
attention and MLP run in parallel, so they are separate terms:
- the **carry**: the patch displacement at site 8, Δx_8 = (c_src − c_base) d;
- each **attention head** (l, h) for l = 8..23, h = 0..15: the change in
  z_h W_O^h (the attention output bias does not change);
- each **MLP** for l = 8..23.

Exactly, Δx_24 = carry + Σ heads + Σ MLPs, which gives 273 terms.

**Logits.**
- Centered unembedding (W_U minus its vocabulary mean), through the final
  LayerNorm.
- The **LN scale σ is frozen at the patched run's value.** Each term
  contributes r_t · Δterm / σ_patched, with r_t = center(W_U,c[t] ⊙ γ).
- The remainder, r_t · x_unpatched · (1/σ_patched − 1/σ_unpatched), is the
  LN-scale term. It is reported separately.
- **Residual** = actual Δ(centered logit) − Σ terms − LN-scale term. It
  should be numerical error only, and is a correctness check.
- Run in fp32. The fp32 total effect is compared against the bf16 main
  test on the same rows.

**Readouts:**
- centered logits of " by", "." , " the" and " him";
- Ō = the mean centered logit over the 27 object-start tokens;
- actual Δ log P alongside, for reference.

**Report, per frame and readout** (component contributions averaged within
pair, then over pairs):
- sum of positive terms (P), sum of negative terms (N), net, LN-scale term,
  residual and actual Δ;
- the top components.

**Flags, on Ō, " the" and " him".** A component is *relevant* if
|mean contribution| ≥ 0.05 in either frame.
- **Sign change:** relevant in both frames, with opposite signs.
- **Vanishes:** relevant on actives, and its passive magnitude is
  < 0.25 × its active magnitude.
- **Appears:** the reverse.

**Coordinates downstream.** Under the same patch, record the site-12 and
site-17 d-coordinates.
- They are measured in z units of those sites' bases for the same split and
  fold.
- Report Δz_12 and Δz_17 on active vs passive bases, and their ratio to the
  patch's own Δz_8.

**Decision rule, on Ō.**
- P_a, N_a are the sums on actives; P_p, N_p on passives.
- Survival s = P_p / P_a.
- Cancellation c = |N| / P in each frame.

| Result | Story |
|---|---|
| s < 0.5 and c_p ≤ c_a + 0.2 | **story 1:** nothing downstream pushes toward objects in passives |
| s ≥ 0.5 and c_p > c_a + 0.2 | **story 2:** object pushes survive but are cancelled |
| s < 0.5 and c_p > c_a + 0.2 | both |
| s ≥ 0.5 and c_p ≤ c_a + 0.2 | neither; would contradict the behavioural result, so check the LN term and residual |

This step captures direct effects only. Indirect suppression (a component
that silences another) is step 5.

**Prediction: story 1, with the gate at the conversion step.**
- The site-17 patch raises object mass by +3.58 nats on passives, so
  nothing late cancels an "object next" value.
- The site-8 patch should therefore barely reach the site-17 coordinate on
  passives. I predict:
  - Δz_17 / Δz_8 on passives ≤ 0.4 × the same ratio on actives;
  - passive Δz_17 ≈ 0.1. Linear scaling from D(O): 0.36 vs 3.58 at
    site 17.
- Δz_12 on passives will sit in between (site 12 was mixed).
- Few components will change sign. Most object-relevant components will
  "vanish" on passives rather than reverse.

## 2. Dose-response at site 8

**Intervention.** Set the site-8 coordinate, in z units of the assigned
basis (sign-aligned, held-out μ_I and μ_T), to each grid value:
- z = −0.50 to 2.00 in steps of 0.05 (51 points);
- h ← h + (c_target − h·d) d.

**Bases.**
- Primary bad passives: every item, using the item's split-0 basis.
- For reference: the good passives of the same items, and the intransitive
  and transitive active DAS items (held-out basis, split 0).

**Readouts:**
- log P(" by"), log P(O), log P(".");
- also determiners, pronouns, reflexives, " the", " him" and I.

**Curves:**
- f_X(z) = mean over pairs of the pair-mean log P(X).
- CIs: two-way bootstrap over pairs and contexts (subjects for actives),
  2,000 draws, seed 17.

**Ranges.**
- Passive range R_p = [0.10, 0.45]: the site-8 primary bad and good passive
  levels (0.085 and 0.448) rounded to the grid.
- Above range R_a = [0.45, 1.50].
- S_X(R) = the OLS slope of f_X over the grid points in R.

**Classification:**

| Result | Reading |
|---|---|
| S_by(R_p) > 0 (95% CI), S_O(R_a) > 0 (95% CI), and S_O(R_p) ≤ 0.25 × S_O(R_a) | **overshoot**: objects stay flat over the passive range and rise only beyond it |
| S_O(R_p) ≥ 0.5 × S_O(R_a) | **coupled**: objects rise inside the passive range at a similar rate as above it |
| anything else | intermediate |

Also reported:
- R² of a linear fit of f_by on R_p;
- S_by(R_a), to show saturation or turnover;
- selectivity S_by(R_p) / S_O(R_p);
- the active curves, which show where objects take off when the base is
  an active.

**Prediction: coupled (or intermediate), not overshoot.**
- While P(O) is small, log P(O) moves with its logit almost linearly. A
  linear readout of a linear push then gives a straight line. A threshold
  at the edge of the passive range would need a downstream nonlinearity.
- Numerically, from D (transitive donors sit at z ≈ 0.87, intransitive at
  ≈ 0) and assuming linearity, I predict:
  - " by" ≈ +0.8 per z unit and objects ≈ +0.4 per z unit, inside and
    above R_p;
  - " by" rising roughly linearly across R_p (R² ≥ 0.95);
  - selectivity in R_p about 2.
- If the curve instead shows overshoot, the object leakage in D is a
  side-effect of pushing passives to active-transitive levels, and the
  site-8 variable is cleaner than D suggests.

## 3. Layer fill-in

**New DAS sites.**
- **4 and 6** (below 8).
- **14 and 16** (between 12 and 17).
- **10**, added because the abstract → mixed change happens between 8 and
  12.

**Recipe:**
- the same as sites 8 and 12: the 29-pair strict set, rank 1, 5 folds × 3
  splits;
- epochs by the frozen epoch rule: 3, 2, 2, 2 and 2 for sites 4, 6, 10, 14
  and 16 (`frozen_config_site{s}.json`);
- the same active controls.

**δ_s:** set by the rule above from each site's final run, and committed
before the passive test runs at that site.

**Passive test:**
- the same plan, populations, contrast D, bootstrap and classification as
  `passive_test_plan.md`;
- projections included;
- run with `run_passive_test.py --sites 4,6,10,14,16` into
  `results/das_round2/passive_test_fill/`.

**Norm-matched random-direction control** (100 draws, split 0, as at site
8) at sites 4, 6, 10, 12, 14, 16 and 17. The criterion: DAS D lies beyond
the null's 95th percentile (5th for falls) on every readout that the site's
classification calls RISE or FALL.

**Predictions:**
- D(O) increases monotonically with depth across 4, 6, 8, 10, 12, 14, 16
  and 17.
- D(" by") is positive up to about 12 and turns negative between 14 and 17.
- Classifications:
  - 4 and 6: "abstract" or unresolved/nothing moves (weak d; active IIA
    0.45 and 0.66);
  - 10: abstract or mixed;
  - 14: mixed;
  - 16: surface.
- The first site where O is RISE is 10 or 12.
- Every DAS effect classified RISE or FALL beats its random null.

## 4. Frequency (replaces the Head/Tail/XTail comparison)

**Regressions**, over the 64 primary pairs. Verbs do not repeat across
pairs, so clustering over verbs is clustering over pairs.
- Outcomes:
  - the per-pair passive-frame z gap (good − bad);
  - the passive/active gap ratio.
- Predictor: the pair's participle Zipf (the mean of the two forms; pairs
  are matched within band). Lemma Zipf is a secondary predictor.
- Sites 8, 12 and 17, plus the fill-in sites once run.
- OLS slope per Zipf unit, with a 95% pair-bootstrap CI (2,000 draws,
  seed 17).

**Correlations** of the site-8 per-pair passive gap with Pythia's
behavioural passive margins (Spearman and Pearson, pair-bootstrap CIs):
- the released-context *by* margin (task 0; 59 original pairs);
- the curated `passive_2` whole-sentence LP margin, averaged over contexts
  (59 original pairs);
- the single-prompt " by" preference from the unpatched readout,
  log P(" by" | good) − log P(" by" | bad) (all 64 pairs).

No prediction is declared. This step replaces a comparison, not a test.

## 5–6. Later

- **5. Path patching** on the components that step 1 flags. Report the sum
  of individual path effects next to the total effect; they are not
  expected to add up exactly.
- **6. In-context nonce passives**, read at site 8 or wherever step 3
  places the conversion. The main measure is the natural projection (no
  injection).
