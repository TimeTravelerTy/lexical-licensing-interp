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

Revised after an outside review (Codex) of the first draft, before any run.

Shared setup: the round-4 D10 frames ("The N has been V" = passive base;
"The N has V" = counterfactual; "The N was V" = second passive), the site-8
rank-1 d (split-0 basis, each item's cross-fitted fold, as B7 / D10 / D11),
fp32, interventions and readouts at the participle's last token. Items: the
64 primary pairs. Switch neurons: ranks 1–10 (4 in MLP11, 6 in MLP14) and
1–50 (16 / 34) of `results/round4/translation/conjunction_switch_neurons.csv`.
They were selected in D11 on these primary pairs (half A), so B5–B6 are new
causal tests of an existing selected set, not independent replications;
results on D11's half-B pairs (33) are reported alongside all 64. Values are
averaged within pair (over contexts and donors) before pairs are averaged;
shares are ratios of population means; every ratio and paired comparison
uses the same bootstrap draws (pairs within band, × contexts where stated;
2,000 draws, seed 17).

## B5. Necessity on good (transitive) passives

**Question.** Round 4 showed sufficiency-type results: on bad items, moving
the d coordinate from the intransitive to the transitive class mean gives
0.44 of the natural " by" gap (A2), and setting the switch neurons to the
frequent-verb mean helps rare verbs (E13-T). B5 asks the reverse: does
setting a transitive verb's value to the intransitive level remove its
" by" preference? (A2's 0.44 is R_I(t_T) − R_I(t_I) on bad items; B5
measures R_T(natural) − R_T(t_I) on good items, so using 0.44 as a
prediction assumes a symmetric response.)

**Items.** The B7 items (32 contexts per primary pair, seed 17; 2,048
items), natural good prompt as the base (no site-8 donor patch), in the
has-been frame (primary) and the was frame (secondary). A natural pass over
every primary passive-test item (good and bad prompt, both frames) supplies
the targets.

**Targets** (from the natural pass; inference conditions on them). For an
item in context c and frame f, eligible pairs = primary pairs other than its
own and outside the DAS training pairs of its basis:
- t_I(d) / t_T(d) = mean raw projection on the item's basis of the eligible
  pairs' natural **bad** / **good** items in c and f;
- per neuron, its mean post-activation over the eligible natural bad items
  in c and f (I-mean) and over the good items (T-mean).

**Interventions on the good item** (one at a time):
- **(a) d_I:** the site-8 coordinate set to t_I: h ← h + (t_I − h·d) d;
- **(b) n10_I / n50_I:** the top-10 / top-50 switch neurons set to their
  I-means (not zero); no site-8 change;
- **(c) c10 / c50:** (a) and (b) together;
- controls: **d_T**, **n50_T** (good-class means, leave-own-out);
  **r10_k / r50_k**, k = 1–5, layer-matched random sets (10 = 4 + 6, drawn
  with seed 17 outside the top 50; 50 = the E13-T `random1`–`random5`) and
  **w50** (the E13-T `wmatched` set, matched on " by" output weight), each
  set to its own I-means; random sets reported as the mean over the five.
- **(d) equalized coordinate** (for "class information beyond d"): good
  and bad items both get the same site-8 coordinate, the context mean
  t_mid = (t_T + t_I) / 2; the top-50 neurons' good − bad activation
  difference (in w_by units: Σ Δa_n · w_by,n) and the " by" good − bad gap
  are recomputed and compared with their natural values. Remaining
  fraction = equalized / natural (ratio of population means).

**Readouts:** log P(" by") (primary), log P(O), log P("."), log P(PREP
without " by") (the C9 sets). **Share removed** = mean(by_nat,good −
by_cond) / mean(by_nat,good − by_nat,bad). Also reported: Δ in nats; the
target contrasts R(d_T) − R(d_I) and R(n50_T) − R(n50_I); the interaction
(c) − (a) − (b).

**Checks (fail-fast).** Setting each item to its own natural values
reproduces the natural readouts (≤ 1e-4); patched entries equal their
targets and nothing else in the hooked tensor changes at that hook.

**Gate.** The natural good − bad " by" gap in the frame must have a 95% CI
above 0 and be ≥ 0.2 nats; otherwise B5 is unresolved in that frame.

**Decision rules** (has-been frame; was reported alongside).
- **d is partly necessary:** share(a) has a 95% CI above 0 and is ≥ 0.25.
- **Neuron replacement removes more than coordinate replacement:**
  share(n50_I) − share(a) (paired) has a 95% CI above 0. This compares
  interventions; by itself it does not show that the neurons get class
  information from outside d.
- **Class information beyond the d coordinate:** under (d), the top-50
  neurons' good − bad difference keeps a 95% CI above 0 and ≥ 25% of its
  natural value. (Beyond this rank-1 coordinate only; other directions at
  site 8 are not equalized.)
- **Neuron specificity:** share(n50_I) − share(r50), share(n50_I) −
  share(w50) and share(n10_I) − share(r10) have 95% CIs above 0.
- **d adds beyond the clamped neurons:** share(c50) − share(n50_I) has a
  95% CI above 0 and is ≥ 0.1; **nothing beyond** if its 90% CI is within
  ±0.1; otherwise unresolved.
- **Controls:** d_T and n50_T should change " by" by less than 0.1 nats
  (95% CI within ±0.1). A failure qualifies the shares (it can reflect a
  generic response to class-mean replacement, nonlinearity or lost
  verb-specific content); it is reported, not treated as proof of either.

**Predictions** (motivated by A2 and D11, not determined by them).
- share(a) ≈ 0.45 (0.3–0.6): **d is partly necessary**.
- share(n50_I) ≈ 0.6 (0.4–0.8); share(n10_I) ≈ 0.4 (0.25–0.6). (D11: the
  top 50's natural good − bad direct effect ≈ 1.4 w_by units ≈ 0.45 logit
  after the final-LN scale, against a natural gap of ≈ 0.7–0.9 nats;
  indirect effects need not add.)
- Neuron replacement removes more than coordinate replacement: **yes**.
- Equalized coordinate: the neurons keep ≈ 50% of their good − bad
  difference: **class information beyond d: yes** (the verbs differ in
  more than the rank-1 coordinate).
- Specificity: **yes** against random; against w50 smaller but CI above 0.
- c50 − n50_I ≈ +0.1–0.2 (d also acts through MLPs 13, 16, 17): borderline.
- d_T within ±0.1: yes; n50_T may fail (E13-T's Head control was −0.09,
  CI to −0.21).

## B6. Serial chain into the late object MLPs

**Question.** D10 / B7: attention heads switch the " by" MLPs 11–17
between frames, and earlier MLPs switch the late object MLPs 15, 18, 21,
22 (heads ≈ 0 there). Is it one chain: heads → mid MLPs 11–17 → late
MLPs?

**Rows.** The D10 rows (B7 items, all split-0 T-donor rows, site-8
interchange in every frame). Known (D10): S_t and its eligibility; the
one-hop share of all earlier MLPs jointly (1.11, 0.87, 1.15, 1.28 of S for
MLPs 15, 18, 21, 22) and of heads (−0.16 to +0.12); the saved D10 singles
give summed mid-only single-MLP shares 0.79, 0.49, 0.76, 1.40 (sums of
singles, not joint effects).

**Definitions.** q ∈ {p, u} = patched / unpatched (site-8 T-donor
interchange or not); b = the has-been (base) frame, h = the has frame.
x^q_{b,m} = the residual input of layer m at the participle's last token in
run (b, q); G_m(x) = MLP m's output vector including its input LayerNorm;
r = the centered Ō readout; σ = the final-LN scale of the (b, p) run, used
for every quantity below (S, one-hop, two-hop, frozen). Pythia's attention
and MLP read the same layer input, so "heads below m" are the heads of
layers < m.
- **Switch:** S_t = Δ_t(h) − Δ_t(b), Δ_t(f) = r·[G_t(x^p_{f,t}) −
  G_t(x^u_{f,t})] / σ.
- **Path effect** of input changes (e^p, e^u) into t: PE_t(e) = r·[G_t(x^p_{b,t}
  + e^p) − G_t(x^u_{b,t} + e^u)] / σ − Δ_t(b).
- H^q_m = Σ over heads in layers < m of (head output in (h, q) − head
  output in (b, q)) at the anchor.
- M_t = the MLPs in 11–17 below t (11–14 for t = 15; 11–17 for 18, 21, 22).
- **One-hop, mid:** e^q = Σ_{m∈M_t} (mlp_m(h, q) − mlp_m(b, q)).
- **(a) two-hop, parallel (primary):** v^q_m = G_m(x^q_{b,m} + H^q_m) −
  G_m(x^q_{b,m}); e^q = Σ_{m∈M_t} v^q_m. **Sequential (sensitivity; longer
  paths, not strictly two-hop):** v^q_m = G_m(x^q_{b,m} + H^q_m + Σ_{k∈M_t,
  k<m} v^q_k) − G_m(x^q_{b,m}). **Singles:** each m alone, e^q = v^q_m from
  the same (joint) step 1; summed vs joint measures non-additivity at the
  target.
- **(c) neurons only (parallel):** e^q = Σ over the top-50 neurons n of
  [a_n(x^q_{b,m} + H^q_m) − a_n(x^q_{b,m})] W_out[:, n]; comparator: (a)
  restricted to m ∈ {11, 14}.
- **(b) anchor-output freeze:** full forward passes in the has frame (p and
  u) with the outputs of M_t at the anchor held at their (b, q) values;
  remaining switch = r·[G_t(x^p) − G_t(x^u)] / σ − Δ_t(b) in the frozen runs.
  Two freeze runs: 11–14 (for t = 15) and 11–17 (for 18, 21, 22). Other
  positions keep the has frame, and everything downstream recomputes, so
  the freeze tests necessity of the anchor outputs under this clamp; it
  need not equal 1 − the two-hop share.

**Checks.** Replacing every upstream component reproduces S_t (≤ 1e-5),
as D10; with zero source change (H = 0), both two-hop variants give 0; a
self-freeze (the has frame's own values) reproduces Δ_t(h).

**Reported per target:** S_t; shares (signed, sign(S_t) · PE / |S_t|,
unrestricted) of one-hop mid, two-hop parallel and sequential, singles
and their sum, the remainder S_t − joint two-hop (signed; omitted paths
plus interactions), (c) / S_t and (c) / (a: 11 + 14), and the remaining
share after freezing.

**Decision rules** (eligible targets: |S_t| ≥ 0.05; with fewer than 3
eligible, unresolved).
- **The chain carries the late switch:** parallel two-hop share ≥ 0.5 with
  CI above 0, for ≥ 3 of the 4 targets.
- **Anchor outputs of the mid MLPs are necessary:** remaining share after
  freezing ≤ 0.5 (point) with its CI upper bound < 1, for ≥ 3 of 4 (t = 15
  freezes 11–14, reported separately from the 11–17 freezes).
- **The switch neurons carry the chain:** for targets where (a: 11 + 14)
  has a share ≥ 0.25 with CI above 0, (c) ≥ 0.5 × (a: 11 + 14); needs ≥ 3
  such targets, otherwise unresolved.

**Predictions.**
- One-hop mid share 0.6–1.2 (the summed singles vary, 0.49–1.40).
- Parallel two-hop ≈ 0.5 (0.3–0.7) of S_t: **the chain holds for 3 of 4**
  (low confidence); sequential larger by ≈ 0.1.
- Freeze: remaining ≤ 0.4 for 18, 21, 22 and ≈ 0.5 for 15 (only 11–14
  frozen): **necessary: yes**.
- The top-50 neurons carry < 0.5 of the 11 + 14 two-hop (selected on " by",
  not on what feeds the object MLPs): **no**.
- Joint ≈ summed within 20%.

## Order (B5–B6)

1. Commit this part.
2. One TSUBAME job each (fp32); B6 extends `run_path_patch_frames.py`.
3. Analysis; report in `results.md`.


# Part C8–C9: generalization robustness (declared 2026-10-14)

Revised after an outside review (Codex) of the first draft, before any run.

Builds on round-4 Part B (`../round4/plan.md`, B6 / B7; results
`../round4/b_fillergap.md`): frames "I know that / what NAME V-ed"
(embedded, B6) and "AUX NAME V" / "What AUX NAME V" (matrix, B7); 24
contexts per experiment, halves A / B as round 4; the active-trained d_s at
sites 4–17 with each verb's cross-fitted fold; readouts O, PREP, "." (B6) /
"?" (B7), L = log P("." or "?") − log P(PREP) (PREP = summed probability
of the 26-token set).

## C8. Filler-gap with non-DAS verbs as the primary set

**Why.** In round 4 the licensing-like crossover at site 8 weakened to
"mixed" in the embedded experiment when only non-DAS verbs were bases; the
matrix experiment kept it at 8 and was mixed at 6. That sensitivity changed
only the bases: the in-range targets, the natural gaps behind the bounds and
the "O against the T level" reference still pooled DAS verbs.

**Verbs.** The round-4 lists (`data/round4/fillergap/verbs.csv`). DAS verbs
= those in any DAS training pair. Counts (fixed inputs): non-DAS 97 T, kept
I 28 (B6) / 34 (B7); DAS 9 T, kept I 11 / 11. The kept I lists stay round
4's (half-A, per-verb gate; removing DAS verbs does not require
re-selection); whether an fp32 I gate would keep the same verbs is reported.

**Primary (non-DAS):** bases = kept non-DAS I verbs; natural gate (half B),
in-range targets (t_T = mean projection of the non-DAS T verbs, t_I = mean
of the kept non-DAS I verbs other than the base; same experiment, frame and
context), the natural T − I gaps behind the bounds and the "O against the T
level" reference all use non-DAS verbs only.

**Secondary:**
- *DAS-base sensitivity with pooled targets* (historical comparison with
  round 4): DAS bases, targets / natural gaps / T reference from all verbs
  outside the basis's DAS training pairs, as round 4. (A DAS-only target
  pool would have only 1–3 eligible T verbs per fold, below round 4's
  minimum of 5.)
- *DAS bases with the non-DAS targets* (isolates the base population).
- Round 4's other sensitivities: all I verbs, multi-token verbs only.

**Frame strata**, reported separately at every site, readings at 6 and 8:
matrix verb (B6: know, remember, forgot, heard) and auxiliary (B7: Did,
Will, Can, Would). Primary within a stratum: its half-B contexts (3);
secondary: all 6. Bounds and the T reference come from the stratum and the
declared population.

**Same as round 4:** frames, contexts, halves, readouts, donors, bounds,
the RISE / FALL / NO RISE classes, the reading table, B7's per-site
base-form gate, and the declared sites (6 and 8; others descriptive).

**Runs** (fp32; round 4 ran in bf16, at a few seconds per site): (1) the
natural pass (projections, readouts; gates analysed locally and committed);
(2) only if a natural gate passes, both patch types for every kept I base at
all eight sites, in-range with the non-DAS and the pooled targets.

**Decision rules.** As round-4 B6 / B7. A failed natural gate → no patch
reading for that experiment.

**Predictions.**
- Natural gates pass in both experiments.
- In range: B7 **licensing-like** at 8; at 6 round 4's non-DAS-base run was
  mixed, and the non-DAS targets make it licensing-like (new, low
  confidence). B6 **mixed** at 6 and 8.
- T − I donors: as round 4 (B6 surface, B7 mixed at 6–8).
- Strata: heterogeneous as in round 4 ("Can" surface; "know" / "heard"
  not licensing-like).

## C9. Shared readers

**Question.** In passives, D11's switch neurons in MLPs 11 and 14 respond to
the verb class only after a passive auxiliary, and MLPs 11, 13, 14, 16, 17
write the " by" switch. Do the same neurons or MLP locations read the verb
class in the filler-gap frames? Two separate questions:
- **class-reader recruitment:** do the neurons separate T from I in the
  "what" frame?
- **filler-conditioned recruitment** (the analogue of D11's was − has
  interaction): do they separate them more with the filler than without?

**Items.** C8's primary verbs and contexts (both experiments, both frames);
half B primary, all 24 contexts secondary. Passive reference: the B7
passive items (64 primary pairs, "The N was V"). Readouts at the verb's last
token, fp32; inference conditions on the in-range targets.

**(a) Switch neurons** (top 50 primary, top 10 secondary; fixed ranks of
`conjunction_switch_neurons.csv`; sign s_n = sign(I_A)):
- A = mean over neurons of s_n × (T − I activation difference), averaged
  over verbs and contexts first. Natural: T − I verbs in each filler-gap
  frame; good − bad in passives.
- In-range: D_in(A) = A at t_T − A at t_I on the I bases (site 8; C8's
  non-DAS targets in filler-gap; round-4 A2 targets on the bad passives).
- Ratios: ρ_nat = A_fg,what(natural) / A_pass(natural), ρ_in =
  D_in(A)_fg,what / D_in(A)_pass (like with like); numerator and
  denominator re-drawn in every bootstrap draw. Denominator gate: the
  passive A has a 95% CI above 0 (else unresolved).
- Also: the coordinate steps (t_T − t_I, in z units) in both constructions;
  per-neuron sign agreement (the share of the 50 whose filler-gap what-frame
  difference has the sign of their passive difference); the cosine between
  the 50-dimensional passive and filler-gap difference vectors.
- Filler conditioning: the natural interaction A_what − A_that and the
  patched interaction D_in(A)_what − D_in(A)_that.

**(b) Writer locations.** For a construction, readout vector r and the
in-range patch (t_I = reference, t_T = patched): DE_m = r·(m_T − m_I) / σ_T
for every MLP and attention layer in 8–23, the site-8 carry, and the
LN-scale term r·x_I (1/σ_T − 1/σ_I), with the residual reported. Readouts:
passive, the centered " by" logit; filler-gap, the linear L̃ = z_q −
mean_PREP z_p. L̃ is not L (L uses logsumexp over PREP): the agreement of
D_in(L̃) with D_in(L) and the correction D[lse(z_PREP) − mean(z_PREP)] are
reported, and the writer comparison is made only if |correction| < 0.25 ×
|D_in(L̃)|.
- Profiles: p_m = max(DE_m, 0) / Σ_m max(DE_m, 0) over the MLPs 8–23 (positive
  contributions toward the construction's change); the cancelling negative
  total is reported. Overlap O = Σ_m min(p_m^pass, p_m^fg) ∈ [0, 1].
- Magnitude gate: in each construction, Σ_m max(DE_m, 0) ≥ 0.2 logits with
  a 95% CI above 0.

**Decision rules** (half B; per experiment, then combined).
- **(a) Class-reader recruitment:** **shared** if ρ_nat and ρ_in are both
  ≥ 0.5 with 95% CIs above 0 in both experiments; **small relative to
  passives** if both have 90% CIs within ±0.25; **reversed** if either has a
  95% CI below 0; otherwise **intermediate / imprecise** (reported with
  which ratio and experiment). Top 10 is a sensitivity; a conflict with top
  50 is reported, not combined.
- **Filler-conditioned recruitment:** the patched interaction D_in(A)_what −
  D_in(A)_that has a 95% CI above 0 in both experiments. A null
  interaction does not negate class-reader recruitment.
- **(b) Writer locations:** **shared dominant locations** if O ≥ 0.5 with
  a 95% CI lower bound ≥ 0.35 in both experiments; **different dominant
  locations** if O's 95% CI upper bound ≤ 0.35 in both; otherwise
  unresolved. Top-5 MLP overlap with bootstrap selection frequencies is
  descriptive. This is about MLP layers, not about the same neurons.
- These test recruitment and location, not mediation (no neuron
  intervention in filler-gap frames).

**Predictions.**
- Switch neurons: they separate T from I in the filler-gap frames, but less
  than in passives (ρ_nat and ρ_in ≈ 0.25–0.5): **intermediate**. Filler
  conditioning: **no** (D11's neurons are conditioned on the passive
  auxiliary; the filler-gap frames are active).
- Writers: passive writers are mid MLPs (11–17); filler-gap writers
  include late MLPs (18–23): O ≈ 0.3, **unresolved** or **different**.
- If so, the reading is an early shared value (d) read by largely
  construction-specific translation; the plan does not claim more than
  recruitment and location.

The optional DAS on the filler-gap contrast (tested on passives) is not
part of this declaration; if it is run, it is declared separately first.

## Order (C8–C9)

1. Commit this part.
2. C8 natural pass (fp32); gates analysed and committed.
3. C8 patches and C9 captures (fp32), only for experiments whose natural
   gate passes.
4. Analysis; report in `results.md`.


# Part A3–A5: training data, restoration, a non-LP readout (declared 2026-10-14)

Written after A1–A2, B5–B6 and C8 ran (`results.md`); revised after an
outside review (Codex) of the first draft, before any run.

## A3. Training-data predictor

**Question.** In A1–A2, the rare-verb deficit in L comes from intransitive
participles. Pythia rejects them after "was" in proportion to their overall
frequency, while transitive participles show no frequency trend. Does
passive-slot frequency in the training data predict Δ_pass and L better
than overall frequency?

**Counts.** infini-gram API, index `v4_piletrain_llama`. This is the Pile
training set; Pythia-1.4B saw just under one epoch of it, so the counts are
exposure proxies. The API returns exact counts of the token sequence its
Llama tokenizer assigns to the query, and every query, its token ids and
decoded tokens are saved (`data/round5/counts/`).
- overall c(V) = count("V") (lowercase, word-initial marker);
- passive slot c_be(V) = Σ count("X V"), X ∈ {was, were, been, is, are,
  be}, with per-auxiliary counts kept;
- perfect slot c_perf(V) = Σ count("X V"), X ∈ {has, have, had}
  (secondary: the corpus analogue of A1's was − has contrast).
- Sensitivity: capitalized auxiliaries ("Was V", …) added to both slots.

**Audit** (before analysis):
- each slot query's token ids must end with the overall query's ids;
  verbs failing this are dropped and listed;
- for 10 verbs (5 T, 5 I, spread over bands), up to 10 contexts of "was V"
  are retrieved (find + get_doc_by_rank) and classified by hand as
  eventive passive / adjectival / other, to show what the proxy counts;
- "be + participle" includes adjectival and other uses and misses passives
  with intervening words. It is the requested proxy, not a count of
  grammatical passives.

**Variables.** Smoothing +0.5 on every count (sensitivity: +0.1 and +1); the
share of zero counts is reported per class and band. Predictors:
- Zipf (wordfreq, as A1);
- log10(c + 0.5) (Pile overall);
- **log10(c_be + 0.5)** (primary slot predictor);
- ρ_be = log((c_be + 0.5) / (c + 0.5)) (secondary; with c_be ≈ 0 it reduces
  to inverse overall exposure, so it is read only conditional on c).

Outcome: Δ_pass(V) (A1, freq frames). For single-token verbs the affinity
a_V equals Δ_pass(V) up to a verb-independent constant, so it adds nothing
and is not analysed separately. Per pair, L, with T and I predictors
entered separately.

**Analyses** (curated verbs primary, pool verbs secondary; per class):
- correlations and in-sample R² (descriptive);
- **out-of-sample error:** 10-fold cross-validation with folds grouped by
  verb pair (pool verbs: by verb), for outcome ~ Zipf, ~ log c, ~ log c_be,
  ~ log c + log c_be;
- uncertainty: verbs (pairs) within band × the 24 noun contexts, 2,000
  draws, seed 17; in each draw the cross-validation is repeated on the
  resampled outcome.

**Decision rules** (per class, curated; L at pair level the same way):
- **Slot frequency adds to overall frequency** if the CV error of
  ~ log c + log c_be is lower than that of ~ log c, with the paired
  difference's 95% CI above 0.
- **Overall frequency suffices** if that difference's 90% CI lies within
  ±2% of the ~ log c error.
- Otherwise unresolved. Descriptive: whether Pile log c beats wordfreq Zipf.

**Predictions.** Statistical preemption: a frequent intransitive is seen
many times, almost never after a be-form, and its rejection grows with
overall frequency, not with its slot count. (A1 already gives r(Δ_pass(I),
Zipf) = −0.58; the new question is what slot counts add.)
- I verbs: **overall frequency suffices**.
- T verbs: Δ_pass(T) has no Zipf trend; slot counts add a little:
  **slot frequency adds** (low confidence).
- L: overall frequency suffices.

If the counts cannot be obtained (API failure), A3 is reported as not run,
with what it would need.

**Pre-analysis clarification (A3, 2026-10-14, while the counts were being
collected, before any analysis).** The infini-gram API throttles a client
after a few hundred requests, and OR (CNF) counts are approximate and take
about a minute each on the server. A test gave 52,084 for "was insured OR
were insured" against an exact sum of 52,085. The query set (16 per verb
for 962 verbs) was therefore cut to exact single-sequence counts only, with
no count looked at:
- the 252 curated verbs (primary) keep every auxiliary (be and perfect);
- the capitalization sensitivity is "Was V" and "Were V" for curated
  verbs;
- pool verbs (secondary) get overall, "was / were / been V" (their c_be)
  and "has V" (their c_perf);
- the per-auxiliary (was + been) sensitivity is curated-only.

## A4. Output-side restoration on single-token rare participles

A2's gate was met. Every rare single-token participle is Tail, and no XTail
participle is a single token. So A4 is a **Tail-singleton intervention**;
its transfer to XTail is not tested.

**Edit.** For an edited participle V with affinity a_V = u_V · Δh̄ (A2's Δh̄,
freq contexts), the unembedding row moves along Δh̄ only:

u_V ← u_V + (a* − a_V) Δh̄ / ‖Δh̄‖².

This is the minimum-norm change that sets the projection on the *mean*
direction to a*. It leaves the orthogonal part unchanged; projections on
individual contexts' Δh are not equalized. Pythia's input and output
embeddings are separate, so only V's output logits change, by
δℓ_V(h) = (a* − a_V) h·Δh̄ / ‖Δh̄‖². Every normalized probability can still
change through the normalizer.

**Conditions** (all rare single-token participles: 23 T, 9 I, curated +
pool; frozen manifest per paradigm in `data/round5/restore/`):
- **oracle:** a* = the mean affinity of the Head single-token verbs of V's
  class (282 T, 30 I);
- **oracle, T only** and **oracle, I only:** to show which side produces a
  gain;
- **label-free:** a* = α + β z_V.
  - z_V: V's site-8 coordinate on d at its (single) token, in "She has V",
    "He has V", "They have V" (mean over the three subjects).
  - Units: z per split-0 fold basis, each with its own sign and held-out
    active levels (as `analyze_nonce_passive.z_table`), then the mean over
    the five folds; DAS verbs use their held-out fold only.
  - (α, β): OLS of a_V on z_V over the Head single-token verbs (both
    classes, unweighted), applied unchanged to the rare verbs.
  - Diagnostics: 10-fold CV R² within Head; residuals by class and Zipf;
    comparison with an intercept-only predictor and with shuffled z.
  - "Label-free" means no class label of the edited verbs is used. d itself
    was trained with class labels.
- Known before the run: the oracle equalizes the eligible population's
  affinity gap by construction. Three verbs move the "wrong" way (diced,
  doped down; corresponded up), and exerted moves by +4.48. The stored
  hidden states imply a ≈ +0.53 nat mean first-token margin gain on the
  affected curated pairs.

**Measures** (unedited vs each condition; fp32):
1. **Passive LP by band:**
   - curated `passive_1` and `passive_2`, all pairs, with the affected
     pairs (12 Tail pairs, edited T and/or I; tail/utter/creep has both)
     reported separately;
   - released FreqBLiMP `passive_1` / `passive_2`, with affected items
     reported separately.
   Reported: accuracy and whole-sentence margin, and the share of the
   affected pairs' margin deficit relative to curated Head pairs that is
   closed.
2. **Specificity:**
   - curated Head pairs (no edited token);
   - all 67 FreqBLiMP paradigms × 3 bands under LP, split into items with
     and without an edited token (accuracy change per cell; the affected
     non-passive items are the direct collateral effects).
3. **LM loss:**
   - Pile *validation* documents containing each edited verb (infini-gram
     `v4_pileval_llama`, find + get_doc_by_rank). Up to 20 occurrences per
     verb at evenly spaced ranks; 512-token windows centred on the
     occurrence, deduplicated.
   - Reported: occurrences per verb; token-weighted and per-verb
     macro-averaged NLL change, over all tokens and at the edited verb's
     tokens; the latter split by the preceding word (be-form / perfect /
     other).
   - Also 500 random Pile-10k documents, truncated to 512 tokens: a
     general diagnostic only, since these documents come from training
     data.
Bootstrap: pairs × contexts (curated), items (released), documents / verbs
(LM loss); 2,000 draws, seed 17.

**Decision rules.**
- **Oracle improves rare passives:** on the affected curated `passive_1`
  pairs, the mean margin change has a 95% CI above 0 and is ≥ 0.25 nats.
  The share of the deficit closed is reported with it.
- **Specific:**
  - curated Head pairs' mean margin change has a 95% CI within ±0.05 nats;
  - over all unaffected FreqBLiMP items pooled, the accuracy change has a
    95% CI within ±0.5 pp (cells reported);
  - the mean NLL change on Pile-10k text has a 95% CI below +0.002 nats per
    token.
  The bounds are set to about a tenth of the smallest effects reported so
  far.
- **Label-free achieves half the oracle:** given that the oracle improves,
  the 95% CI of Δ_LF − 0.5 Δ_oracle (affected pairs, paired draws) lies
  above 0.
- Side-specific arms and NLL at the edited tokens: reported, not gated.

**Predictions.**
- Oracle: whole-sentence margins on affected curated `passive_1` pairs
  rise by ≈ +0.5 nats (close to the first-token +0.53): **improves**. Most
  of it comes from the I-only arm.
- Specific: **yes**, with Head-pair changes < 0.02 nats.
- NLL at the edited tokens changes by < 0.1 nats on average and rises for
  some intransitive participles after be-forms.
- Label-free: below half the oracle (Head CV R² of a_V on z_V is modest,
  and rare-verb z values lie outside the Head range): **not achieved**
  (low confidence).

## A5. Non-LP readout: a grammaticality probe

**Method.** Wang et al. 2026 (ACL), "Implicit Representations of
Grammaticality in Language Models"; code at
github.com/SusanWYS/grammaticality_probe, used as the reference for details.
Declared adaptations: corpus, sentence splitting, tokenization and solver.
- **Corpus.** Up to 25,000 sentences (6–40 words) from Project Gutenberg
  (`jondurbin/gutenberg-dpo-v0.1`, "chosen") and up to 25,000 from
  WikiText-103 raw (test + validation), seed 17; any shortfall is filled
  from Gutenberg, and realized counts are reported. WikiText stands in for
  their Penn Treebank half (LDC-licensed).
  - Splitting is rule-based (PySBD and spaCy are not installed here or
    locally): split on . ! ? followed by space and a capital, quotes kept
    together, WikiText headings and markup dropped.
  - A 200-sentence sample is audited for fragments.
- **Tokens and perturbations.** Regex tokens (words, contractions and
  punctuation marks are separate tokens, as spaCy would mostly give). One
  perturbation per sentence, chosen uniformly:
  - insertion of k ∈ {1..5} alphabetic tokens drawn from the corpus
    vocabulary at random token boundaries;
  - deletion of k ∈ {1..5} alphabetic tokens;
  - a shuffle of a contiguous 5-token window.
  Sentences that cannot take their perturbation are dropped. Pairs (and
  everything) are frozen before any model run.
- **Splits.** 80 / 20 train / dev **by document** (Gutenberg book,
  WikiText article), keeping each original with its perturbation;
  duplicates across splits are removed.
- **Probe.** For each layer (residual outputs 1–24), on the last token's
  hidden state (no BOS; right padding; the final-LN output is not used):
  - features z-scored with training statistics;
  - L2 logistic regression (scikit-learn lbfgs, 1,000 iterations; C from
    2^−5 … 2^2, the inverse of their α grid up to loss normalization),
    chosen by dev AUC;
  - the layer with the best dev AUC is used.
  Three probe seeds (different train/dev document splits) give a stability
  range.
- **Baselines** on dev: a logistic model on length and punctuation counts
  only, and the LM's own sentence log-probability.
- **Scoring.** A pair is correct if the probe *logit* is higher for the
  good sentence; continuous logit margins are reported as well.
  - Primary: released FreqBLiMP `passive_1`, by band.
  - Secondary: `passive_2`, the curated band-cross `passive_1` pairs, and
    all 67 paradigms × 3 bands (descriptive).
  - LP accuracy on the same items is the comparison.

**Probe quality.** The probe is informative if its dev AUC is ≥ 0.85,
exceeds the length/punctuation baseline by ≥ 0.1, and its released Head
`passive_1` accuracy has a 95% CI above 50%. Otherwise the results are
still reported, labelled "uninformative probe".

**Decision rules** (released `passive_1`). D_m = accuracy_Head −
accuracy_XTail for m ∈ {probe, LP}, paired on the same items; items are
resampled within band, with a good-verb-cluster resampling as a
sensitivity. Tail is reported throughout.
- **Detectable decline:** D_probe has a 95% CI above 0.
- **Declines like LP** (keeps at least half of LP's drop): D_probe −
  0.5 D_LP has a 95% CI above 0.
- **Smaller decline than LP:** D_LP − D_probe has a 95% CI above 0.
- **Practically no decline:** D_probe's 90% CI lies within ±4 pp.
Several can hold at once; all are reported.

**Predictions.**
- The probe is informative. Its Head `passive_1` accuracy is below LP's.
- Rarity: the probe's drop is smaller than LP's (≈ 0–4 pp vs ≈ 9 pp):
  **smaller decline than LP: yes; declines like LP: no**; "practically no
  decline" possible (low confidence). Rationale: the deficit sits in the
  participle's output probability (A1), which a probe on the final state
  need not inherit. A1–A2 do not decide this.

**Pre-run clarification (A5, 2026-10-14, after a code review, before any
model run).**
- **Corpus shortcut controls.** The first build gave perturbed sentences
  surface cues that originals never had: 5,348 lowercase starts, 964
  without terminal punctuation, glued words, and capitalized inserted
  words. The frozen corpus (`data/round5/probe/`) now:
  - inserts lowercase common word types only;
  - restores sentence-initial casing after a perturbation;
  - keeps insertions and shuffles before the terminal punctuation;
  - re-draws a perturbation until the sentence starts like an original;
  - renders originals and perturbed sentences with one spacing rule;
  - drops originals with unbalanced brackets or quotes; initials no longer
    end sentences.
  Audit: 0 bad starts and 0 missing terminal punctuation in either class;
  mid-sentence capitals 1.74 vs 1.68 per sentence.
- **Baseline.** The artifact baseline uses length, punctuation and casing
  counts (token, word, punctuation, mid-sentence-capital and comma counts).
- Splits 1–2 re-select C at split 0's layer; they measure stability of the
  probe at that layer, not of the layer choice.

## Order (A3–A5)

1. Commit this part.
2. A3: counts (API, locally), audit, analysis.
3. A4: z coordinates and edits; Pile-validation documents fetched locally
   via the API; one TSUBAME job.
4. A5: corpus and pairs built locally and frozen; one TSUBAME job.
5. Report in `results.md`.

# Part B7: switch neurons in the wild (declared 2026-10-14)

Written after C9 showed that D11's switch neurons separate the verb classes
in wh-gap frames as well as passive ones. Revised after an outside review
(Codex), before any run.

**Question.** What do D11's top switch neurons respond to in ordinary text?

**Neurons.** MLP11.n3462, MLP14.n2114, MLP14.n5407 (ranks 1–3, carrying 0.44
of the half-B switch), plus ranks 4–10 of `conjunction_switch_neurons.csv`.
s_n = sign(I_A) is the switch direction; it is negative for ranks 9 and 10.

**Text.** 2,000 documents drawn with seed 17 from NeelNanda/pile-10k (a Pile
training sample; input hash recorded), each truncated to 512 Pythia
tokens. Saved per token: token id, character offsets, the full word it
ends, the preceding words, the next token id, and the post-activation of
the ten neurons (input of `dense_4h_to_h`), fp32.

**Measures.**
- Per neuron, the activation distribution: mean, SD, share > 0, 99th and
  99.9th percentiles.
- **Top contexts.** The 100 tokens with the largest raw activation
  (primary; ties by document then position). Sensitivity: the signed
  ranking s_n · a, which for ranks 9 and 10 is the *lowest*-activation
  contexts and is labelled so. Each is reported with 24 tokens before and 4
  after, the number of unique documents and words, and a version
  deduplicated to one context per document.
- **Surface proxies** for the top 100, with the same proxies in two
  baselines (all tokens; tokens meeting the participle-candidate definition
  (a)):
  - (a) participle candidate: ends a word in -ed / -en, or the word is among
    A1's 962 participles;
  - (b) be/get-form among the 3 preceding words;
  - (c) wh-word or relativizer among the 8 preceding words, no be/get-form
    among the 3;
  - (d) the next token is " by" (exact token id).
  These are proxies, not grammatical classifications (e.g. (b) also catches
  copulas and progressives, (c) "that"-declaratives). Uncertainty comes
  from document-level resampling with the top list recomputed in each draw
  (1,000 draws).
- **Construction annotation** (the basis for any label). The top 40 raw
  contexts of each top-3 neuron are labelled independently by Claude and by
  Codex (context-only, blind to the hypothesis and to each other). Each
  context gets one of: verbal passive participle; adjectival / copular
  participle; active perfect or simple past; object gap (wh / relative)
  after the verb; other; uncertain. Both label sets are frozen before
  comparison; agreement is Cohen's κ. Each annotator also writes a one-line
  free description.

**Decision rules** (per top-3 neuron, from the annotation; both annotators'
rates reported):
- **passive-participle detector:** the joint "verbal passive participle"
  rate is ≥ 0.5 for both annotators;
- **gap-sensitive participle reader:** verbal passive + object-gap ≥ 0.5 for
  both, with object-gap ≥ 0.15;
- **other:** anything else, described.
Ranks 4–10: proxies and descriptions only.

**Predictions.**
- The top-3 neurons are **passive-participle detectors**: verbal passive in
  ≥ 0.5 of the top 40, and proxy (b) well above the participle-candidate
  baseline.
- Following C9, some top contexts are object gaps: gap-sensitive reading
  possible for one of the three.
- " by" follows in ≥ 20% of the top 100.

# Part D: scale-up prep (declared 2026-10-14)

Feasibility only. Revised after an outside review (Codex), before any run.

## D10. Feasibility on Pythia-6.9B and 12B

Separate processes per model (1.4B as the reference), fp32, TF32 off, on an
80 GB H100. Each measurement uses warm-up, CUDA synchronization and 5
measured repetitions, and reports peak allocated and reserved memory.
Feasibility updates are discarded; no scientific DAS result is produced.
- **Load memory and forward throughput** on the passive-test prompts.
- **DAS.** One round-2 training step: rank-1 rotation as
  `run_das_round2.py` (frozen model, Adam on the rotation only), the
  round-2 batch size, at the earliest, middle and latest planned site.
  Fallback if out of memory: halve the batch, keep fp32. A **DAS run** is
  one site × 15 fold-split fits:
  T_run = T_setup + N_steps · t_step + T_eval, with N_steps from round 2's
  swaps, epochs and batch. Controls (shuffled labels, random) are estimated
  as extra runs.
- **Path patching.** One complete representative batch of
  `run_path_patch_frames.py` (all captures, head / MLP replacements and the
  component-chunk evaluations) at short and long prompt lengths. Per run
  (D10's 11,657 rows): T = batches × t_batch, two-frame and three-frame
  versions.
- **Site plan** at matched relative depth, s' = ⌊s × L' / 24 + 0.5⌋:
  - 6.9B (32 layers): 5, 8, 11, 13, 16, 19, 21, 23;
  - 12B (36 layers): 6, 9, 12, 15, 18, 21, 24, 26;
  - ±1 adjacent sites as a later sensitivity.
  Matched depth is a search heuristic, not evidence of homologous
  mechanisms.
Output: `reports/round5/d10_scaleup.md` (table and plan).

## D10b. A1/A2 across the Pythia suite

A1's runner on Pythia 160M, 410M, 1B, 2.8B, 6.9B and 12B (1.4B as run).
- They share the tokenizer, so A1's inputs carry over; each model still
  runs its own numerical checks (tokenization, first-subtoken
  reconstruction, identity).
- A small block (one prefix set, multi-token verbs included) is timed
  first. The full runs go ahead under a cap of 4 GPU-hours in total.
- Per model, A1's tables and readings (L, Δ_pass(T), Δ_pass(I), M_has, by
  band) and A2's checks, Head AUC and singleton D_L.
- No curated-score accounting (it needs per-model curated scores).
- Descriptive across size.

**Prediction** (1.4B is the reference: ΔL 2.12, intransitive side +2.60).
The L deficit appears at every size ≥ 410M and stays on the intransitive
side. Its size is within ±50% of 1.4B's.

## D10c. FreqBLiMP passive_1 Yes/No on 6.9B and 12B

Task 4's prompt (`YES_NO_BASE_PROMPT`).
- Score = LSE(Yes logits) − LSE(No logits), with Yes = {"Yes", " Yes"} and
  No = {"No", " No"} (token ids verified).
- A pair is correct if the good sentence scores higher; exact ties get half
  credit.
- Items: released FreqBLiMP `passive_1`, 3 bands × 1,000. Models: 1.4B
  (reference), 6.9B, 12B; fp32.
- Also reported: continuous margins and the Yes + No probability mass.
- Item bootstrap, with a good-verb-cluster sensitivity and paired draws
  for comparisons between models.

**Decision rules** (per model and band; both reported):
- **above chance:** 95% CI above 50%;
- **practically at chance:** 90% CI within 45–55%;
- **below chance:** 95% CI below 50%;
- otherwise unresolved.

**Prediction.** Task 4 found 1.4B at 60.8% [50.5, 69.8] on curated Head
contexts. On released items: 1.4B Head about 55% (unresolved or above
chance), Tail and XTail practically at chance; 6.9B and 12B above chance on
Head (60–70%) and weaker on XTail.

## D11. Expanded verb list

**Goal.** An expanded inventory of at least 128 T/I pairs (twice the 64
primary pairs), at least 40 per band. Existing curated pairs may be
included (flagged), single-token participles are flagged, and a held-out
split is never used for DAS.

**Sources** (VerbNet is not available here without another download, so
the existing inventories are used):
- FreqBLiMP's released `passive_1` / `passive_2` good and bad participles
  (freq-blimp data as on TSUBAME, hashes recorded);
- the round-2 / 3 expansion inventory (`data/verb_expansion/verbs.csv`,
  with its earlier rater decisions);
- round 4's candidates (`data/round4/fillergap/candidates.csv`);
- the 126 curated pairs.

**Criteria**, applied independently by Claude and Codex (blind to each
other), with labels and reasons stored:
- T: has an ordinary direct-object use and a natural verbal passive in
  the relevant sense; obligatory vs optional transitivity flagged.
- I: dominantly intransitive, with no common passivizable transitive use;
  rare transitive senses flagged.
- Used only if both raters accept. This is a screening procedure, not
  lexical ground truth.

**Matching.** Within band (wordfreq 3.1.1 participle Zipf; bands as
FreqBLiMP). First maximize the number of one-to-one T–I pairs within
calipers (participle 0.35, lemma 0.25), then minimize the total Zipf gap;
ties broken by lemma.

**Split.**
- Pairs whose verbs appear in any historical DAS training inventory
  (round-2 `train_pairs.csv`, `verb_expansion/das_train_pairs.csv`) are
  **trainable** only.
- Of the rest, one third per band × token-count cell (seed 17) is
  **held-out**: never used for DAS training, for site / rank / epoch
  selection, or for donor or reference calibration.
- Splits are by lemma, and loaders check them.

**Rule.** Success means ≥ 128 pairs in total and ≥ 40 per band. Otherwise
the shortfall per band is reported. Head intransitives are the likely
bottleneck (only 26 clear Head pairs before). No model is run on the list
in this round.

## Order (B7, D)

1. Commit this part.
2. B7: one TSUBAME job; the annotation after it.
3. D10, D10b, D10c: TSUBAME jobs.
4. D11: built locally with Codex as the second rater.
5. Report in `results.md`.
