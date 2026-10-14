# Round 5: declared plan

Builds on rounds 2–4 (`../results_index.md` indexes every earlier result).
Goals: finish the mechanism tests, locate the rare-verb passive deficit, and
prepare the paper. As before, each part is written and committed before any
of its runs, with predictions stated before the data exist; where a
prediction leans on earlier outputs, that is said. An outside reviewer
(Codex) reviews each part's draft and code before the runs.

Run order: 0 (results index, no compute), A1–A2, B5–B6, C8–C9, A3–A5, B7,
D. Parts are declared in that order, each in its own commit.

Notation as in round 4 (`../round4/plan.md`): site s = residual output of
layer s − 1; d_s = the active-trained rank-1 DAS direction; T / I =
transitive (passivizable, "good") / intransitive ("bad") verb of a pair;
CIs are 95%, 2,000 draws, seed 17, unless stated otherwise. Layout: plan and
reports in `reports/round5/`, regenerable inputs in `data/round5/<part>/`,
outputs in `results/round5/<part>/`.


# Part A1–A2: where the LP deficit sits (declared 2026-10-14)

Revised after an outside review (Codex) of the first draft, before any run.

## What is already known

On the 126 curated band-cross pairs (26 Head, 50 Tail, 50 XTail), crossed
with all 126 curated `passive_1` contexts ("The N was V by the X."; existing
fp32 scores `results/passive_band_cross/pythia14b_scores.csv`):

| | Head | Tail | XTail | Head − XTail |
|---|---:|---:|---:|---:|
| whole-sentence margin | 2.95 | 2.78 | 2.20 | 0.76 [−0.12, +1.57] |
| participle (verb) margin | 2.85 | 2.01 | 1.46 | 1.39 [+0.69, +2.08] |
| suffix margin | 0.10 | 0.77 | 0.73 | −0.63 |

The participle carries more than the whole deficit; rare pairs gain some back
after the verb. The whole-sentence denominators are sign-unstable (Head −
XTail: 5% of draws ≤ 0; Head − Tail +0.17, 38% ≤ 0). At matched fit
(good-sentence and bad-patient ratings) the participle deficit is −1.11
[−1.78, −0.46] (XTail − Head, `passive_1`; `../fit_ratings/bad_fit_models.csv`,
model `good+bad_patient`). Released contexts: whole −1.30, participle −1.11
(`../passive_band_cross/report.md`).

Pairs are matched on frequency: the within-pair participle (form) Zipf gap
T − I averages −0.02 / 0.00 / 0.00 (Head / Tail / XTail; max |gap| 0.17), the
lemma gap 0.01 / −0.01 / 0.00. Pair-mean participle Zipf: Head 3.54–5.18,
Tail 2.43–3.16, XTail 1.31–2.18.

**Tokenization** (Pythia, leading space): every Head participle is one token;
Tail has 6 single-token T and 7 single-token I participles; XTail has none.
Only 27 pairs (26 Head, 1 Tail) have both participles as single tokens; no
pair shares its first subtoken. Band and token count are therefore
confounded, with no common support for a Head − XTail comparison at fixed
token count. Nothing below separates frequency from tokenization; results
are also reported by token-count cell.

## A1. Frame contrast

**Question.** Is the participle deficit specific to the was-frame, relative
to the has-frame, or does the rare transitive participle lose probability
against its matched intransitive after "has" as well?

**Measures.** For a participle V (leading space; full log-prob summed over
its subtokens) and a prefix c:
- Δ_pass(V, c) = log P(V | "The N was") − log P(V | "The N has");
- L(c) = Δ_pass(V_T, c) − Δ_pass(V_I, c), per pair;
- M_was(c) = log P(V_T | was-prefix) − log P(V_I | was-prefix) (the plain
  was-frame T − I margin); M_has(c) the same after "has". Identity:
  M_was = L + M_has.

L is a was–has interaction, not a pure voice contrast: after "was" the noun
is read as a patient (and "was" also allows adjectival participles), after
"has" as an active subject (and "has" also allows possession + NP). Fixed
nouns give every band the same contexts but not the same verb–context fit
(rare verbs have narrower selectional ranges, `../../docs/fit_ratings.md`).
Readings are therefore labelled **was–has-specific** / **has-frame-general**,
and a voice-matched sensitivity is added (below). Per-subtoken log-probs are
saved, so L = L₁ (first subtoken) + L_rest (later subtokens) exactly.

**Prefix sets** (the same prefixes for every pair):
- **freq** (primary): "The N was" / "The N has", N one of 24 frequent
  singular nouns, 12 animate (man, woman, child, boy, girl, student,
  teacher, doctor, officer, player, friend, father) and 12 inanimate
  (house, car, book, door, letter, report, plan, system, building, window,
  road, project). Checked before declaration (wordfreq 3.1.1, Pythia
  tokenizer, transformers 4.57.6): each is one token with a leading space and
  has Zipf ≥ 4.8 (window 4.80 is the lowest). Reserves, in order: animate
  mother, king, judge; inanimate table, phone, film; the builder fails if a
  reserve list runs out and writes the final list with Zipf values and
  library versions to `data/round5/frame/`.
- **been** (voice-matched sensitivity): "The N has been" / "The N has", same
  24 nouns; Δ_voice(V) and L_voice defined as above with "has been" in place
  of "was" (both perfect aspect; round 4 D10 used this contrast).
- **cur** (sensitivity): the 126 curated `passive_1` prefixes ("The house
  was"), each with " was" → " has"; every pair in every context (as the band
  cross). The 126 context IDs contain 109 distinct prefixes; contexts are
  weighted per context ID (as the band-cross analysis). **own**: each pair's
  own curated context (descriptive, paired).
- **they** (robustness): "They were" / "They have" (one prefix).

No BOS token, as `score_matched_passives.py`. The participle's ids are those
of " V" alone; the builder checks that tokenizing the joined string gives
prefix ids + participle ids.

**Pool verbs** (secondary, per-verb only). Every distinct participle of the
released FreqBLiMP `passive_1` and `passive_2` files
(`data/freqblimp_original_passives/pairs.jsonl`; good = T pool, bad = I
pool, band as released; a form in both pools or not a single alphabetic
word is excluded; curated verbs are deduplicated into their curated row),
scored in the freq frames. They enlarge the single-token population for
A2's per-verb analysis and the A4 gate. Zipf from wordfreq (participle
form).

**Analyses** (each prefix set separately; freq primary):
- Band means of L, Δ_pass(T), Δ_pass(I), M_was, M_has, L₁ and L_rest; band
  differences Head − Tail and Head − XTail.
- Slopes on Zipf (OLS across pairs; secondary consistency checks, since
  bands occupy separate Zipf ranges): pair quantities on the pair-mean
  participle Zipf, Δ_pass(T) on the T participle's Zipf, Δ_pass(I) on the I
  participle's.
- Side contributions to the Head − XTail L deficit: ΔΔ_pass(T) and
  −ΔΔ_pass(I) (signed, additive: ΔL = ΔΔ_pass(T) − ΔΔ_pass(I)).
- freq split by noun animacy (12 + 12).
- By token-count cell (T tokens × I tokens): band means of L, L₁, L_rest;
  equal-count Tail vs XTail comparisons where both bands have ≥ 5 pairs in
  a cell; band contrasts of L₁ and L_rest separately.
- Within-pair Zipf gaps by band (above) and the Spearman correlation of a
  pair's L with its gap (descriptive).
- **Accounting for the earlier deficits** (cur contexts, where they were
  measured), signed and additive, Head − XTail and Head − Tail:
  - whole-sentence: ΔW = ΔL + ΔM_has + ΔM_suffix (M_suffix from the
    existing scores; W = M_was + M_suffix);
  - fit-controlled: β(M_was) = β(L) + β(M_has), the XTail (Tail)
    coefficients of the `good+bad_patient` WLS model of `analyze_bad_fit.py`
    refit with each outcome on identical rows, design and draw weights.
    This is regression accounting under the earlier specification,
    conditional on the fixed ratings, not a mediated share.
  Each term is reported with its CI. Ratios (ΔL/ΔW, ΔL/ΔM_was,
  β(L)/β(M_was)) are reported only where the denominator's sign is stable
  (≤ 2.5% of draws with the wrong sign); otherwise only the additive terms
  are interpreted. The whole-sentence ratios are already known to be
  unstable. ΔL on the freq contexts is shown beside the cur terms
  (descriptive, different contexts).
- **Bootstrap:** pairs resampled within band; contexts resampled jointly
  for every band, frame and outcome of a set (one draw per set): freq and
  been, nouns within animacy strata; cur, contexts within context band; they,
  pairs only. 2,000 draws, seed 17. Intervals describe variation across the
  chosen lexical and context inventories, not a representative sample.

**Check (fail-fast).** M_was on the cur contexts reproduces the existing
participle margins (`verb_margin` of the band-cross scores) to ≤ 1e-3 nats
for every pair × context.

**Decision rules** (freq; the same rules are reported for been, cur and
they). For a band contrast X = Head − XTail (also reported: Head − Tail),
with bound 0.25 nats (a quarter of the fit-controlled residual):
- **deficit:** 95% CI above 0 and estimate ≥ +0.25;
- **surplus:** 95% CI below 0 and estimate ≤ −0.25;
- **negligible:** 90% CI within ±0.25;
- otherwise **unresolved**.
These are mutually exclusive. Applied to ΔL and ΔM_has:
- **was–has-specific:** deficit in L, M_has negligible;
- **has-frame-general:** L negligible, deficit in M_has;
- **both:** deficits in both; **neither:** both negligible;
- otherwise unresolved (signs and CIs reported).

**Predictions.**
- L > 0 in every band (95% CI above 0).
- ΔL (Head − XTail): **deficit**, estimate ≈ +1.0 (0.5–1.5) nats; Zipf
  slope positive.
- Both sides contribute: ΔΔ_pass(T) > 0 and −ΔΔ_pass(I) > 0, each 30–70%
  of ΔL.
- ΔM_has point estimate within ±0.25. Whether "negligible" is reached
  depends on precision, so the predicted reading is **was–has-specific**
  if it is, otherwise unresolved.
- been: the same reading as freq. Animate and inanimate nouns: a deficit in
  both.
- cur: ΔM_has within ±0.25. Given the known ΔM_was (1.39), this implies
  ΔL(cur) ≈ 1.14–1.64 (a derived constraint, not a separate prediction).
  Fit-controlled: β(L) ≈ β(M_was) (about −1.1), β(M_has) ≈ 0.
- they: the same reading as freq.

## A2. Exact decomposition

**Identity.** T and I share the prefix, so their first subtokens are
predicted from the same final-LN output h at the auxiliary. With
log P(t | c) = h_c · u_t − LSE_c (Pythia's unembedding has no bias), the
normalizers cancel:

L₁(c) = (h_was,c − h_has,c) · (u_T1 − u_I1),

where u_T1, u_I1 are the unembedding rows of the two first subtokens. For a
single-token pair L₁ = L. This and P = mean L₁ below are algebraic; the
numerical checks test the implementation, not a hypothesis. For single-token
verbs, the dependence of L on output rows is therefore inevitable; the
empirical question is whether rare single-token verbs have a smaller T − I
affinity contrast along the shared direction.

**Measures.**
- **Check:** the identity holds for every pair and prefix (all sets) to
  ≤ 1e-3 nats in fp32. If not, stop.
- Δh̄ = mean over the freq contexts of Δh_c = h_was,c − h_has,c (recomputed
  in every context draw). Reported: ‖Δh̄‖ and the per-context deviation share
  mean_c ‖Δh_c − Δh̄‖² / mean_c ‖Δh_c‖².
- **Per pair:** P = (u_T1 − u_I1) · Δh̄ (= the freq-mean L₁); by band. For
  the 27 single-token pairs it is the freq-mean L.
- **Per verb ("passive-slot affinity"):** a_V = u_V1 · Δh̄ = ‖u_V1‖ ‖Δh̄‖
  cos(u_V1, Δh̄); norm and cosine reported separately. Slopes on the verb's
  Zipf, T and I separately, kept apart:
  (i) **single-token verbs** (curated: 32 T, 33 I; curated + pool,
  deduplicated, as the secondary set), with their observed Zipf ranges;
  (ii) **first-subtoken fragments** of multi-token verbs (a fragment's a_V is
  a property of a sub-word shared with other words, not of the verb).
  Class separation among Head verbs: AUC of a_V, T vs I.
- **Multi-token pairs** (99 pairs with at least one multi-token participle):
  mean L₁, mean L_rest and their signed shares of mean L, by band and by
  token-count cell. An output-row edit cannot directly change later
  fragments' logits (it can change their normalizers), so a large L_rest
  share limits what A4 could reach.
- Bootstrap as A1: curated per-verb quantities resample pairs (keeping T and
  I together); pool verbs resample within role × band cells; Δh̄ and every
  affinity are recomputed in each context draw.

**Gate for A4** (declared now; A4 itself is declared with A3–A5). The
eligible population is the single-token participles of the curated and pool
inventories, deduplicated; from existing token counts about 282 Head T,
23 rare T, 30 Head I and 9 rare I (the builder's counts are used). **Every
rare single-token participle is Tail; none is XTail**, so A4 would be a
Tail-singleton intervention, and its transfer to XTail (all multi-token)
would remain untested. A4 runs only if all hold:
1. A1 reads "deficit" for ΔL (freq).
2. **Singleton contrast deficit:** D_L = (ā_T,Head − ā_I,Head) −
   (ā_T,rare − ā_I,rare) has a 95% CI above 0 (verbs resampled within
   role × band cells; Δh̄ re-drawn). Reported with it: D_T = ā_T,Head −
   ā_T,rare and D_I = ā_I,Head − ā_I,rare (which side A4 would edit), the
   same contrast on curated verbs only (26/6/26/7), and a version with Head
   verbs restricted to Zipf ≤ 4.0 (closer to the rare range).
3. At least 5 rare single-token participles in each class (expected: met).
If 1 holds but 2 fails, A4 is not run and the result is reported as "no
supported affinity gap in the eligible singleton population" (not as
evidence that none exists).

**Predictions.**
- Δh̄ is the larger part of Δh (deviation share < 0.5).
- Head T verbs have higher a_V than Head I verbs (AUC ≥ 0.8).
- D_L > 0 with CI above 0 (rare singletons have a compressed T − I affinity
  contrast): **gate 2 met** (low confidence).
- Multi-token pairs: the first subtoken carries 40–70% of L (low
  confidence).

## Pre-analysis clarification (A1–A2, 2026-10-14, after the run's checks, before any analysis)

The fail-fast reproduction check failed: M_was differs from the band-cross
`verb_margin` by up to 1.74 nats (median 0.08). The cause is the reference,
not the scorer: the band-cross scores were computed in **bfloat16** (job
8811408, `--dtype bfloat16`), while this run is fp32, so ≤ 1e-3 could not be
met. Fix: the 15,876 curated `passive_1` pairs are rescored with the same
scorer (`score_matched_passives.py`) in fp32; the check is run against those
scores, and the accounting uses their whole, verb and suffix margins (and
refits the fit-controlled model on them). The bf16-based deficits quoted
above are kept for reference. No A1–A2 result was looked at before this
change.

## Order (A1–A2)

1. Commit this part (with the results index, part 0).
2. Builder (tokenizer only, TSUBAME login node), then one TSUBAME job
   (fp32): A1 scores, h at the auxiliary, the checks.
3. Analysis locally; report in `results.md`.

# Part B5–B6: mechanism completion (declared 2026-10-14)

Shared setup: the round-4 D10 frames ("The N has been V" = passive base;
"The N has V" = counterfactual; "The N was V" = second passive), the site-8
rank-1 d (split-0 basis, each item's cross-fitted fold, as B7 / D10 / D11),
fp32, interventions and readouts at the participle's last token. Items: the
64 primary pairs. The D11 switch neurons are ranks 1–10 (4 in MLP11, 6 in
MLP14) and 1–50 (16 / 34) of `results/round4/translation/conjunction_switch_neurons.csv`.
Pair bootstrap within band (× contexts where stated), 2,000 draws, seed 17.

## B5. Necessity on good (transitive) passives

**Question.** Round 4 showed sufficiency: moving an intransitive verb's d
coordinate within the passive range gives 0.44 of the natural " by" gap (A2),
and setting the switch neurons helps rare verbs (E13-T). Are d and the
switch neurons also necessary: does setting a transitive verb's value to the
intransitive level remove its " by" preference?

**Items.** The B7 items (32 contexts per primary pair, seed 17; 2,048 items),
good prompt as base, in the has-been frame (primary) and the was frame
(secondary). A natural pass over every primary passive-test item (good and
bad prompt, both frames) supplies the targets.

**Targets** (computed once from the natural pass; inference conditions on
them). For an item in context c and frame f, eligible pairs = primary pairs
other than its own and outside the DAS training pairs of its basis:
- t_I(d) = mean raw projection on the item's basis of the eligible pairs'
  **bad** items in c and f; t_T(d) the same over their **good** items;
- for each switch neuron, its mean post-activation over the eligible bad
  items in c and f (I-mean) and over the eligible good items (T-mean).

**Interventions on the good item** (one at a time):
- **(a) d_I:** the site-8 coordinate set to t_I: h ← h + (t_I − h·d) d;
- **(b) n10_I / n50_I:** the top-10 / top-50 switch neurons set to their
  I-means (not zero);
- **(c) c10 / c50:** (a) and (b) together;
- controls: **d_T** and **n50_T** (set to the good-class means, leave-own-out:
  specificity; under a common linear response they move little);
  **r10_k / r50_k**, k = 1–5: five layer-matched random neuron sets (10 = 4 +
  6, drawn with seed 17 outside the top 50; 50 = the round-4 E13-T `random1`–
  `random5` sets, `data/round4/oracle_translation/neurons.csv`), set to their
  own I-means; reported as the mean over the five sets.

**Readouts:** log P(" by") (primary), log P(O), log P("."), log P(PREP
without " by") (the C9 sets). **Share removed** = mean(by_nat,good −
by_cond) / mean(by_nat,good − by_nat,bad) over the same items (ratio of
population means; pairs within band × contexts). Δ in nats is reported
with it.

**Checks (fail-fast).** Setting each item to its own natural value (d
coordinate, neuron activations) reproduces the natural readouts (≤ 1e-4);
patched entries equal their targets; nothing else in the hooked tensors
changes.

**Decision rules** (has-been frame; was reported alongside).
- **d is partly necessary:** share(a) has a 95% CI above 0 and is ≥ 0.25.
- **The neurons carry class information beyond d:** share(n50_I) −
  share(a) (paired) has a 95% CI above 0.
- **Neuron specificity:** share(n50_I) − share(r50) and share(n10_I) −
  share(r10) have 95% CIs above 0.
- **d adds beyond the neurons:** share(c50) − share(n50_I) has a 95% CI
  above 0 and is ≥ 0.1; **nothing beyond** if its 90% CI is within ±0.1;
  otherwise unresolved.
- **Controls:** d_T and n50_T change " by" by less than 0.1 nats (95% CI
  within ±0.1). If not, the shares are flagged as partly a generic response
  to class-mean replacement (as E13).

**Predictions.**
- share(a) ≈ 0.45 (0.3–0.6): d is partly necessary (the sufficiency value
  was 0.44).
- share(n50_I) ≈ 0.6 (0.4–0.8); share(n10_I) ≈ 0.4 (0.25–0.6). From D11:
  the top 50's natural good − bad direct effect is ≈ 1.4 w_by units ≈ 0.45
  logit after the final-LN scale, against a natural gap of ≈ 0.7–0.9 nats,
  plus indirect effects.
- The neurons carry class information beyond d: **yes**. Specificity:
  **yes** (random sets ≈ 0).
- c50 − n50_I ≈ +0.1–0.2 (d also acts through MLPs 13, 16, 17): "d adds
  beyond the neurons" borderline.
- Controls within ±0.1: d_T yes; n50_T may fail (E13-T's Head control was
  −0.09 with a CI to −0.21).

## B6. Serial chain into the late object MLPs

**Question.** D10 / B7: attention heads switch the " by" MLPs 11–17
between frames, and earlier MLPs switch the late object MLPs 15, 18, 21,
22 (heads ≈ 0 there). Is it one chain: heads → mid MLPs 11–17 → late
MLPs?

**Rows and quantities.** The D10 rows (B7 items, all split-0 T-donor rows,
site-8 interchange in every frame). As D10: for a target MLP t (readout
Ō), Δ_t = its readout under (patched − unpatched) in a frame; the switch
S_t = Δ_t(has) − Δ_t(has been); a path effect replaces some upstream
contributions in t's input at the participle's last token by their has-frame
values (patched and unpatched runs each with their own), with the
final-LN reference scale of the base patched run.
Known one-hop shares (D10): earlier MLPs jointly 1.11, 0.87, 1.15, 1.28 of
S for MLPs 15, 18, 21, 22; heads −0.16 to +0.12.

**Paths** (targets t ∈ {15, 18, 21, 22}; M_t = the MLPs in 11–17 below t:
11–14 for t = 15, 11–17 otherwise):
- **one-hop, mid:** the outputs of M_t replaced by their has-frame values
  (reference for (a));
- **(a) two-hop:** step 1, each m ∈ M_t recomputed with its input changed
  only by the has-frame minus base outputs of all heads below m; step 2,
  t's input changed only by the step-1 changes of M_t. **Parallel**
  (primary): in step 1 each m sees only the head changes. **Sequential**
  (sensitivity): m also sees the step-1 changes of the M_t MLPs below it.
  Also each m alone (summed vs joint).
- **(b) freeze:** full forward passes in the has frame (patched and
  unpatched) with the outputs of M_t at the participle's last token held at
  their has-been values from the matching base run; the remaining switch
  of t is Δ_t(has, frozen) − Δ_t(has been). Two freeze runs: 11–14 (for
  t = 15) and 11–17 (for 18, 21, 22). Other positions are not frozen.
- **(c) two-hop through the top-50 neurons only:** as (a), but step 2 uses
  only the change in the top-50 neurons' contributions (Δ activation × the
  neuron's output weight), all in MLPs 11 and 14; compared with (a)
  restricted to MLPs 11 and 14 (full outputs).

**Reported per target:** S_t; one-hop shares (all earlier MLPs, mid);
two-hop shares (parallel, sequential); the sum of the single-m two-hop
effects vs the joint two-hop and the unexplained remainder S_t − joint;
the remaining share after freezing; (c) / (a: 11+14). Shares are signed
fractions of S_t (sign(S_t) · PE / |S_t|), with pair-bootstrap CIs.
Exactness check: replacing every upstream component reproduces S_t
(≤ 1e-5), as D10.

**Decision rules** (eligible targets: |S_t| ≥ 0.05).
- **The chain carries the late switch:** parallel two-hop share ≥ 0.5,
  CI above 0, for ≥ 3 of the 4 targets.
- **Mid MLPs are necessary:** remaining share after freezing ≤ 0.5 (point)
  with its CI upper bound < 1, for ≥ 3 of 4.
- **The switch neurons carry the chain:** (c) ≥ 0.5 × (a: 11+14) for ≥ 3 of
  4.
- Joint vs summed and the remainder are descriptive.

**Predictions.**
- One-hop mid share 0.7–1.0 (most of the earlier-MLP path is mid MLPs:
  12, 16, 13, 17, 11, 14 led in B7).
- Parallel two-hop ≈ 0.5 (0.3–0.7) of S_t: the chain holds for 3 of 4
  (low confidence); sequential larger by ≈ 0.1.
- Freezing removes most of the switch: remaining ≤ 0.4 for 18, 21, 22;
  ≈ 0.5 for 15 (only 11–14 frozen). Mid MLPs necessary: **yes**.
- The top-50 neurons carry < 0.5 of the 11 + 14 two-hop (they were selected
  on " by", not on what feeds the object MLPs): **no**.
- Joint ≈ summed within 20%.

## Order (B5–B6)

1. Commit this part (after review).
2. One TSUBAME job each (fp32); B6 extends `run_path_patch_frames.py`.
3. Analysis; report in `results.md`.
