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

## Order (A1–A2)

1. Commit this part (with the results index, part 0).
2. Builder (tokenizer only, TSUBAME login node), then one TSUBAME job
   (fp32): A1 scores, h at the auxiliary, the checks.
3. Analysis locally; report in `results.md`.
