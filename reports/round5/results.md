# Round 5: results

Plan, predictions and decision rules: `plan.md`, each part committed before
its runs (part 0 and A1–A2: `d3b6630`, builder fix `c9b673a`, analysis fixes
after code review `32feb71`, bf16 clarification `21671df`; B5–B6:
`c6be4ba`). Generated reports, one per experiment:
- 0: `../results_index.md`;
- A1–A2: `a1_frame.md`, `a2_decomposition.md`;
- B5–B6: `b5_necessity.md`, `b6_serial.md`;
- C8–C9: `c8_fillergap_gates.md`, `c8_fillergap.md`, `c9_shared_readers.md`.

TSUBAME jobs: 9007119 (A1–A2), 9007211 (curated `passive_1` rescored in
fp32), 9007340 (B5; earlier attempts 9007281 and 9007322 stopped at the
self-patch check, see below), 9007282 (B6), 9007314 (C8 natural pass),
9007336 (C8 patches), 9008384 (C9).

## Summary

- **A1, where the participle deficit sits.** The passive slot itself.
  - After "was" vs "has", a pair's transitive − intransitive contrast L is
    large in every band (Head 3.84, Tail 2.82, XTail 1.72 nats) and is 2.1
    nats smaller for XTail than for Head.
  - Almost all of that comes from the **intransitive** side. After "was",
    frequent intransitive participles lose 3.45 nats against "has", while
    rare ones lose 0.85. Transitive participles gain about the same in
    every band.
  - The has-frame margin goes the other way and offsets part of it, so the
    plain was-frame deficit (1.41) is smaller than the L deficit.
  - The same holds with a voice-matched contrast ("has been" vs "has"),
    with animate and inanimate nouns, with the curated contexts and with
    "They were / have".
  - At matched fit, the L deficit (−1.83) exceeds the earlier ≈ 1 log unit
    participle residual (−1.16).
- **A2, the exact decomposition.**
  - Head participles are single tokens, and their L is the projection of the
    two output rows on one shared "was − has" direction. Along it,
    intransitive rows sit lower the more frequent the verb. Transitive rows
    show no frequency trend.
  - Rare multi-token participles get nothing from their first fragment
    (XTail: 3% of L). Their passive-slot preference appears only in the
    later subtokens.
  - The A4 gate is met: rare single-token participles (all Tail) have a
    compressed T − I contrast along the direction. A4 will run as a
    Tail-singleton intervention.
- **B5, necessity.** The early value is far from necessary.
  - Setting a transitive passive's d coordinate to the intransitive class
    mean removes only 15% of its " by" preference.
  - Clamping the 50 switch neurons to their intransitive means removes 9%,
    and both together 19%.
  - With good and bad items equalized on d, 59% of the " by" gap and 39%
    of the neurons' class contrast remain.
  - So the verb class reaches the translation stage through more than the
    rank-1 coordinate. The neurons respond to it specifically: random and
    weight-matched sets remove nothing.
- **B6, serial chain.** The late object MLPs depend on the mid MLPs.
  - Freezing the anchor outputs of MLPs 11–17 (11–14 for MLP15) removes
    74–86% of every late MLP's frame switch.
  - The strict two-hop path heads → mid MLPs → late MLP carries the whole
    switch for MLPs 21 and 22, but only 0.46 and 0.29 for 15 and 18. With
    relays among the mid MLPs it carries 0.66–0.70 there.
  - The top-50 switch neurons carry 42–49% of what MLPs 11 and 14 pass on.
- **C8, filler-gap without DAS verbs.** The licensing-like crossover
  survives in matrix questions (sites 4, 8, 10) but not in embedded ones,
  where objects overshoot the non-DAS transitive level. Round 4's embedded
  result needed the DAS-verb bases.
- **C9, shared readers.** D11's switch neurons separate transitive from
  intransitive verbs in wh questions, and only when the filler is present.
  They read the verb class whenever the object position is empty (passive
  or wh gap); they are not passive-specific.

# Part A1–A2

## A1. Frame contrast

Δ_pass(V) = log P(V | "The N was") − log P(V | "The N has"); L = Δ_pass(T) −
Δ_pass(I); M_was = L + M_has. 126 curated pairs, 24 frequent nouns (primary,
`freq`). Rule: deficit = 95% CI above 0 and estimate ≥ 0.25 nats; negligible
= 90% CI within ±0.25.

| freq | Head | Tail | XTail | Head − XTail |
|---|---|---|---|---|
| L | +3.84 | +2.82 | +1.72 | **+2.12 [+1.26, +2.94]**, deficit |
| Δ_pass(T) | +0.39 | +0.10 | +0.87 | −0.48 [−1.17, +0.14] |
| Δ_pass(I) | −3.45 | −2.73 | −0.85 | −2.60 [−3.28, −1.85] |
| M_was (plain was-frame margin) | +2.77 | +1.89 | +1.35 | +1.41 [+0.67, +2.17] |
| M_has | −1.07 | −0.94 | −0.37 | −0.71 [−1.45, +0.04], unresolved |

The Zipf slope of L is +0.97 [+0.63, +1.28] per Zipf unit. Of ΔL, the
intransitive side contributes +2.60 and the transitive side −0.48.

| Prefix set | ΔL (Head − XTail) | ΔM_has | Reading |
|---|---|---|---|
| freq (primary) | +2.12 [+1.26, +2.94] deficit | −0.71 unresolved | unresolved |
| freq, animate / inanimate nouns | +1.94 / +2.29, deficit | −0.89 / −0.52 | unresolved |
| been ("has been" / "has", voice-matched) | +1.82 [+0.98, +2.67] deficit | −0.71 unresolved | unresolved |
| cur (curated contexts) | +2.12 [+1.27, +2.95] deficit | −0.68 unresolved | unresolved |
| they ("They were / have") | +1.73 [+0.87, +2.57] deficit | −0.22 unresolved | unresolved |
| own context (descriptive) | +3.43 [+2.20, +4.66] | −0.60 | — |

Every set reads "unresolved" for one reason: the M_has difference is
neither negligible nor a declared surplus. Its point estimate is −0.2 to
−0.9, so rare pairs do relatively *better* after "has", and its 95% CI just
reaches 0. ΔL is a deficit everywhere.

**Tokenization.** At equal token counts, Tail still beats XTail. In the 2×2
cell (25 Tail, 15 XTail pairs), L is +1.78 [+0.69, +2.81] higher for Tail,
mostly in the first subtoken (+1.61). The 2×3 and 3×2 cells point the same
way, but their CIs include 0. Head has no multi-token pairs, so frequency
and tokenization stay confounded for Head vs XTail.

**Accounting for the earlier deficits** (curated contexts, fp32 scores):

| Head − XTail | ΔW (whole sentence) | ΔL | ΔM_has | ΔM_suffix | ΔM_was |
|---|---|---|---|---|---|
| | +0.82 [−0.08, +1.67] | +2.12 | −0.68 | −0.62 | +1.44 [+0.76, +2.12] |

- ΔL / ΔM_was = 1.47 [0.95, 2.50]. The whole-sentence ratio is not
  interpreted, because its denominator has the wrong sign in 4% of draws.
- **At matched fit** (the `good+bad_patient` model, refit per outcome):
  β(M_was) −1.16 [−1.81, −0.49] = β(L) −1.83 [−2.66, −0.97] + β(M_has)
  +0.66 [−0.07, +1.38] (XTail − Head).
- So the passive-slot interaction is larger than the whole fit-controlled
  residual, and the has-frame part offsets about a third of it.

**Against the predictions.**
- L > 0 in every band: **held**.
- ΔL deficit: **held**. Size ≈ 1.0 (0.5–1.5): **missed**; it is 2.12.
- Positive Zipf slope: **held**.
- Both sides contribute 30–70%: **missed**. The intransitive side carries
  all of it (1.23 of ΔL), and the transitive side slightly opposes it.
- ΔM_has within ±0.25 (was–has-specific): **missed** (−0.71). The reading
  is unresolved everywhere.
- been and they give the same reading as freq: **held** (unresolved, with
  an L deficit). Animate and inanimate both show a deficit: **held**.
- cur: ΔM_has within ±0.25: **missed** (−0.68). So the derived range
  ΔL(cur) 1.14–1.64 does not apply (2.12). β(L) ≈ β(M_was): **missed**;
  β(L) is larger (−1.83 vs −1.16).

**Reading.** The rare-verb participle deficit is in the passive slot, and
in the frame contrast it comes almost entirely from intransitive verbs.
Pythia strongly expects a frequent intransitive participle *not* to follow
"was" ("The house was vanished"): 3.45 nats less likely than after "has". A
rare intransitive participle is nearly as likely after "was" as after
"has" (0.85). Rare transitive participles are not under-predicted after
"was"; they gain as much as frequent ones.

So the model does not fail to accept rare transitives in passives. It fails
to reject rare intransitives there. This matches two earlier observations:
the *by* split's "bad side high" pairs (P-bysplit), and XTail intransitives
putting more mass on objects in actives (P-base).

The voice-matched "has been / has" contrast gives the same pattern, so this
is not specific to the auxiliary "was". A second, smaller effect runs the
other way: after "has", Head transitive participles are less likely than
their intransitive partners (M_has −1.07), and much less so for XTail
(−0.37). This offsets part of the passive-slot deficit in the plain
was-frame margin.

Caveats:
- L is a was–has interaction. The noun's role changes between the frames,
  as declared.
- Head vs XTail stays confounded with tokenization. Only the Tail vs XTail
  comparison at equal token counts escapes it.

## A2. Exact decomposition

- **Check.** The identity L₁ = (h_was − h_has) · (u_T1 − u_I1) holds to
  4.0e-5 nats for every pair and prefix. The first-subtoken log-probs
  rebuilt from h · u − LSE match to 2.7e-5.
- **The direction.** The mean was − has direction Δh̄ (norm 58.4) carries
  most of Δh: the per-context deviation share is 0.44.

| | Head | Tail | XTail |
|---|---|---|---|
| P = (u_T1 − u_I1) · Δh̄ (= mean L₁) | +3.84 [+3.16, +4.46] | +0.98 [+0.45, +1.52] | +0.06 [−0.30, +0.39] |
| mean L_rest (later subtokens) | 0 | +1.84 | +1.66 |
| L₁ share of L, multi-token pairs | — | 0.34 [0.16, 0.50] | 0.03 [−0.19, 0.22] |

- **Per verb, single-token participles** (affinity a_V = u_V · Δh̄):
  - transitive: Zipf slope +0.25 [−0.75, +1.13] (curated, 32 verbs) and
    +0.20 [−0.15, +0.58] (with pool verbs, 305);
  - intransitive: −1.58 [−2.09, −0.98] (33 verbs) and −1.55 [−2.03, −0.96]
    (39);
  - the intransitive trend is in the direction (cosine slope −0.031), not in
    the row norm;
  - Head T vs I separate along Δh̄ with AUC 0.96.
- **First-subtoken fragments** of multi-token verbs show the same
  intransitive trend (−1.06 [−1.54, −0.61]), even though a fragment is
  shared with other words.
- **XTail shortfall:** L̄_Head − L_XTail = 2.12 = (3.84 − 0.06) − 1.66. The
  first fragment carries none of Head's 3.84; the later subtokens win back
  1.66.

**Gate for A4** (single-token participles, curated + pool; all rare ones
are Tail):

| | Head T | rare T | Head I | rare I | D_T | D_I | D_L |
|---|---:|---:|---:|---:|---|---|---|
| gate population | 282 | 23 | 30 | 9 | +0.60 [+0.03, +1.20] | −1.49 [−2.43, −0.53] | **+2.09 [+1.00, +3.23]** |
| curated only | 26 | 6 | 26 | 7 | +0.98 | −1.82 | +2.80 [+1.13, +4.78] |
| Head Zipf ≤ 4.0 | 150 | 23 | 13 | 9 | +0.60 | −0.76 | +1.36 [+0.21, +2.50] |

**Declared decision.** Gates 1–3 are met, so **A4 runs**. A4 is a
Tail-singleton intervention: no XTail participle is a single token, so its
transfer to XTail stays untested. Its design is declared with A3–A5.

**Against the predictions.**
- Deviation share < 0.5: **held** (0.44).
- Head AUC ≥ 0.8: **held** (0.96).
- Gate 2 met: **held**. The contrast is driven mostly by the intransitive
  rows (D_I −1.49), not by low transitive affinity.
- The first subtoken carries 40–70% of L in multi-token pairs: **missed**
  (Tail 0.34, XTail 0.03).

**Reading.** Along one shared passive-slot direction, Pythia's output rows
for frequent intransitive participles point away from "was". For rare ones
they point much less away, while transitive rows are flat in frequency.
That is the unembedding-side form of A1's result.

For rare multi-token participles, the first fragment's row carries no
verb-specific passive preference, which is expected. The preference shows
up once the model has read the fragment and predicts the rest of the word,
so for those verbs it lives in the computation over earlier subtokens, not
in a single output row. An output-row edit (A4) can address only the
single-token minority.

## Deviations and checks (A1–A2)

- The declared reproduction check failed against the band-cross scores,
  which turned out to be bf16 (job 8811408). The curated `passive_1` pairs
  were rescored in fp32, and M_was then matches them to 3.3e-4 nats. The
  accounting uses the fp32 margins. This was declared in `plan.md` before
  any analysis.
- Two internal sanity tolerances were adjusted (not declared checks):
  - whole = verb + suffix holds to 5e-4, because of prefix batch noise;
  - the fit identity uses the run's own M_was. The scorer's margin gives
    the same coefficient to < 1e-3.

# Part B5–B6

## B5. Necessity on good passives

Natural good passives of the B7 items (2,048; 64 pairs), has-been frame
(primary). Share removed = mean(natural − patched) / mean(natural good −
bad) of log P(" by"). The natural gap is +0.71 [+0.36, +1.03] (gate
passes).

| Intervention (has been) | Δ " by" (nats) | share removed |
|---|---|---|
| (a) d → intransitive mean | −0.11 [−0.13, −0.09] | 0.15 [0.09, 0.31] |
| (b) top 10 → I means | −0.06 [−0.09, −0.02] | 0.08 [0.04, 0.15] |
| (b) top 50 → I means | −0.06 [−0.10, −0.02] | 0.09 [0.04, 0.15] |
| (c) both, top 50 | −0.14 [−0.18, −0.09] | 0.19 [0.13, 0.34] |
| random 50 / 10 (mean of 5) | 0.00 | 0.00 |
| weight-matched 50 | −0.01 | 0.01 |
| control: d → transitive mean | +0.01 [−0.00, +0.02] | — |
| control: top 50 → T means | +0.07 [+0.03, +0.12] | — |

- **Paired differences** (shares):
  - top 50 − d: −0.06 [−0.21, −0.00];
  - top 50 − random: +0.09 [+0.04, +0.15];
  - top 50 − weight-matched: +0.07 [+0.02, +0.13];
  - top 10 − random: +0.08;
  - (c) − top 50: +0.11 [+0.06, +0.22].
- **Equalized coordinate** (good and bad both set to (t_T + t_I)/2):
  - the top-50 good − bad difference falls from 1.33 to 0.52 w_by units
    (remaining 0.39 [0.14, 0.53]);
  - the " by" gap falls from 0.71 to 0.42 (remaining 0.59 [0.25, 0.72]).
- **Other frames and subsets.** The was frame gives the same picture: d
  removes 0.16, top 50 0.07, both 0.18. On D11's half-B pairs the neurons
  do a little better (top 50 removes 0.13, equal to d).

**Declared decisions** (has been, all pairs):
- d partly necessary (share ≥ 0.25): **no**.
- Neuron replacement removes more than coordinate replacement: **no**; it
  removes slightly less.
- Class information beyond the d coordinate: **yes**.
- Neuron specificity: **yes**.
- d adds beyond the clamped neurons: **adds** (+0.11).
- Controls: the d control is within ±0.1; the top-50 T-mean control is not
  (+0.07, CI to +0.12).

**Against the predictions.**
- share(a) ≈ 0.45: **missed** (0.15). d partly necessary: **missed**.
- share(top 50) ≈ 0.6 and share(top 10) ≈ 0.4: **missed** (0.09, 0.08).
- Neurons remove more than d: **missed**.
- Class information beyond d: **held** (0.39 of the neurons' contrast and
  0.59 of the " by" gap remain).
- Specificity: **held**.
- (c) − top 50 ≈ +0.1–0.2: **held** (+0.11).
- d control within ±0.1: **held**. The top-50 control failing: **held**
  (as anticipated from E13-T).

**Reading.** In round 4, moving an intransitive verb's coordinate up to
the transitive level supplied 0.44 of the " by" gap (sufficiency, A2).
Moving a transitive verb's coordinate down to the intransitive level
removes only 0.15. Clamping the switch neurons removes 0.09, and both
together 0.19.

The equalized-coordinate test shows why. When good and bad verbs sit at
the same point on d, the switch neurons still see about 40% of their class
contrast and " by" keeps about 60% of its gap. So the verb class reaches
MLPs 11 and 14, and the " by" output, through directions other than this
rank-1 coordinate.

The neurons are specific: random and weight-matched sets do nothing. But
50 neurons are only part of the route. The T-mean control also moves
" by" (+0.07), so part of what the clamp does is homogenize each verb, as
E13 / E13-T found. In short, d is a sufficient lever but a redundant,
non-necessary carrier of the class.

## B6. Serial chain into the late object MLPs

D10 rows (11,657 rows, 64 pairs), has been → has, site-8 T-donor
interchange. Shares of each late MLP's switch S_t (signed, pair bootstrap):

| Target | S | one-hop, all earlier MLPs | one-hop, mid MLPs | two-hop parallel | two-hop sequential | singles summed | remaining after freezing |
|---|---|---|---|---|---|---|---|
| 15 | +0.244 | 1.11 | 0.84 | **0.46** [0.40, 0.52] | 0.66 | 0.59 | **0.18** [0.15, 0.21] |
| 18 | +0.246 | 0.87 | 0.68 | **0.29** [0.23, 0.35] | 0.70 | 0.20 | **0.26** [0.23, 0.30] |
| 21 | +0.174 | 1.15 | 0.78 | **1.08** [0.97, 1.21] | 1.10 | 0.95 | **0.22** [0.19, 0.25] |
| 22 | +0.172 | 1.28 | 1.28 | **0.94** [0.82, 1.08] | 1.26 | 0.82 | **0.14** [0.10, 0.18] |

- **Neuron route.** Top-50 neurons only vs MLPs 11 + 14 (full outputs),
  as a share of S: 0.09 / 0.21 (15), −0.05 / −0.09 (18), 0.12 / 0.26 (21),
  0.18 / 0.37 (22). The ratio is 0.42–0.49 wherever the comparator is
  positive.
- **Checks.** Replacing every upstream component reproduces S to 1.9e-6.
  Zero-source and self-freeze checks are exact.
- **Robustness.** D11's half-B pairs give the same numbers to within
  0.05.

**Declared decisions.**
- The chain carries the late switch (two-hop ≥ 0.5 for ≥ 3 of 4):
  **no** (2 of 4: MLPs 21, 22).
- Anchor outputs of the mid MLPs are necessary: **yes** (4 of 4).
- The switch neurons carry the chain: **unresolved**. The comparator
  qualifies for only 2 targets. There, the neurons carry 0.45 and 0.49 of
  it, just under the 0.5 bar.

**Against the predictions.**
- One-hop mid 0.6–1.2: **held** for 3 of 4 (MLP22: 1.28).
- Two-hop ≈ 0.5 and the chain holds for 3 of 4: **missed** (2 of 4). The
  shares are 0.29–1.08, and the sequential version is larger by 0.2–0.4
  where two-hop is short (predicted ≈ 0.1).
- Freezing leaves ≤ 0.4 (≈ 0.5 for MLP15): **held**, and more strongly
  than predicted (0.14–0.26; 0.18 for MLP15).
- Top 50 < 0.5 of the 11 + 14 two-hop: **held** numerically (0.42–0.49),
  but the declared rule is unresolved.
- Joint ≈ summed within 20%: **held** for 21 and 22; **missed** for 15
  and 18.

**Reading.** Without the mid MLPs' outputs at the participle, the late
object MLPs lose three-quarters or more of their frame switch. So in
passives the late MLPs' silence (round 2 mechanism) is mostly inherited
from the mid MLPs.

For MLPs 21 and 22, the inheritance is the simple chain: heads switch the
mid MLPs, and those changes alone switch the late MLP. For MLPs 15 and 18
it is longer. The head-driven change in one mid MLP feeds further mid MLPs
before it reaches them (sequential 0.66–0.70 vs two-hop 0.29–0.46).

D11's switch neurons pass on a bit under half of what MLPs 11 and 14 send
to the late MLPs. They are a large part of the relay, not all of it.

## Deviations and checks (B5–B6)

- **B5 self-patch check.** The first two attempts stopped there. The check
  compared against the natural pass (other batches), and then against the
  same batch but with targets taken from other batches (1.7e-4 vs the
  declared 1e-4). The final run takes the self-patch targets and every
  baseline from the same batch's unpatched run: error 0.0. The natural
  pass differs from that run by fp32 batch noise of ≤ 5e-5. The decision
  quantities use the same-batch baselines.
- **B6** ran unchanged. Its checks: exactness 1.9e-6; zero-source and
  self-freeze exact.

# Part C8–C9

## C8. Filler-gap with non-DAS verbs as the primary set

fp32 rerun of round-4 B. Primary: kept non-DAS I bases (28 embedded, 34
matrix), non-DAS targets, natural gaps and T reference (97 T verbs),
half-B contexts. Commit `f2a7fff` holds the gates, written before any
patch run.

**Gates.** Both natural gates pass for the primary population:
- embedded: L +1.42 [+1.01, +1.84], O +0.78 [+0.46, +1.12];
- matrix: L +1.21 [+0.80, +1.64], O +0.97 [+0.67, +1.28].

Every matrix-verb and auxiliary stratum passes. An fp32 half-A I gate
keeps exactly round 4's I verbs. The base-form gate is 1.00 at every site.
One secondary population fails: the embedded DAS bases fail on O
(+0.33 [−0.17, +0.76]), so their patch readings are reported but not
interpreted.

**Readings, primary population (half B):**

| | site 4 | 6 | 8 | 10 | 12–17 |
|---|---|---|---|---|---|
| embedded, in range | mixed | surface | mixed | mixed | surface |
| embedded, T − I donors | surface | surface | surface | surface | surface |
| matrix, in range | **licensing-like** | mixed | **licensing-like** | **licensing-like** | surface |
| matrix, T − I donors | mixed | surface | surface | surface | surface |

- **What fails in embedded questions.** In range at site 8, with the
  filler, the closure and preposition parts behave as licensing predicts:
  "." +0.95, PREP −0.55, L +1.51, all decisive. But objects rise past the
  non-DAS transitive level: +0.37 [+0.01, +0.73] at 8 and +0.47 [+0.08,
  +0.85] at 6 (bounds 0.49 and 0.47). This fails the "not above"
  condition: "above" at 6, unresolved at 8.
- **In the matrix experiment** the O level stays at the T level:
  +0.14 [−0.19, +0.52] at 8.
- **Other populations** (sites 6 and 8):
  - round-4 population, fp32: embedded mixed / licensing-like; matrix
    licensing-like at both. This reproduces round 4.
  - DAS bases, either target pool: embedded licensing-like at 8 (its
    natural gate fails); matrix licensing-like at both.
  - multi-token non-DAS bases: embedded surface; matrix mixed.
  - matrix, all contexts: licensing-like at 6 and 8.
- **Strata** (in range, half B):
  - embedded: "forgot" licensing-like at 8 and 10; "know" surface at every
    site; "heard" and "remember" mixed or surface;
  - matrix: "Will" and "Would" licensing-like at 4–10, "Did" at 8–10,
    "Can" surface from site 6.

**Against the predictions.**
- Natural gates pass: **held**.
- Matrix licensing-like at 8: **held**. Matrix at 6, newly predicted
  licensing-like: **missed** (mixed on half B; licensing-like on all
  contexts).
- Embedded mixed at 6 and 8: **held at 8**; **missed at 6** (surface).
- T − I donors as round 4: embedded surface **held**. Matrix mixed at 6–8
  **missed**: it is surface, because the donors push O above the T level.
- Strata heterogeneous, "Can" surface, "know" / "heard" not
  licensing-like: **held**.

**Reading.** With verbs that never entered DAS training, the in-range
crossover survives in matrix questions at sites 4, 8 and 10. It does not
survive in embedded questions. There, the closure-vs-preposition part
holds, but the move pushes object expectation above the level natural
non-DAS transitive verbs reach. Round 4's embedded site-8 result
reappears only when the DAS-verb bases are included, and those bases fail
the natural gate on their own.

So the licensing-like reading of the early coordinate in filler-gap
contexts is robust for matrix questions only. In embedded questions, the
non-object (closure) side generalizes, but the object side overshoots. It
also depends on the matrix verb and auxiliary ("Can" and "know" read
surface).

## C9. Shared readers

D11's top-50 switch neurons (top 10 as sensitivity). A = mean over neurons
of sign(I_A) × the activation difference (T − I in filler-gap; good − bad
in passives). D_in = activation at t_T − at t_I under the site-8 in-range
patch. C8 primary population; half B; filler-gap bases are the kept
non-DAS I verbs.

| top 50 | natural A, "that" | natural A, "what" | D_in(A), "that" | D_in(A), "what" | ρ_nat (what / passive) | ρ_in (what / passive) |
|---|---|---|---|---|---|---|
| passive reference | — | +0.105 [+0.083, +0.126] | — | +0.080 [+0.074, +0.087] | — | — |
| embedded | −0.002 | +0.202 [+0.190, +0.213] | −0.003 | +0.125 [+0.116, +0.135] | 1.93 [1.58, 2.47] | 1.57 [1.41, 1.75] |
| matrix | −0.006 | +0.192 [+0.173, +0.212] | −0.006 | +0.114 [+0.102, +0.127] | 1.83 [1.48, 2.37] | 1.43 [1.25, 1.65] |

- **Top 10** gives the same pattern: ρ_nat 2.1–2.2, ρ_in 1.6.
- **Per neuron.** The sign of a neuron's "what"-frame T − I difference
  agrees with its passive difference for 0.62–0.68 of the 50 neurons. The
  cosine between the 50-neuron difference vectors is 0.55–0.58.
- **Dose.** The in-range step is larger in filler-gap than in passives
  (0.84 / 0.81 vs 0.34 z). Per z unit of coordinate change, the "what"
  response is 0.60–0.63 of the passive one (descriptive, not declared).
- **Writers.** The comparison was **not made** under the declared rule.
  The PREP logsumexp correction (−0.54, −0.48) exceeds a quarter of
  D_in(L̃) (+0.97, +1.14), so the linear L̃ does not stand in for L.
  Descriptive only:
  - profile overlap O = 0.38 (embedded) and 0.54 (matrix);
  - top-5 MLPs: filler-gap {8, 13, 15, 22, 23} / {11, 13, 14, 15, 19};
    passive {8, 10, 13, 14, 16}.
- **Checks.** The decomposition residual is ≤ 5.6e-5. The coordinates hit
  their targets to 4e-6. The two basis files agree (cos 1.0000).

**Declared decisions.**
- Class-reader recruitment: **shared** in both experiments (top 50 and
  top 10).
- Filler-conditioned recruitment: **yes** (patched "what" − "that" +0.13
  / +0.12, CI above 0 in both).
- Writer locations: **unresolved** (correction gate).

**Against the predictions.**
- The neurons respond, but less than in passives (intermediate):
  **missed**. They respond more, as raw ratios. Per unit of coordinate
  change they respond somewhat less (0.6), and the plan did not use that
  as the decision.
- No filler conditioning: **missed**. The response is almost entirely
  conditional on the filler.
- Different writers: **not tested** (the rule's precondition failed).

**Reading.** The switch neurons D11 found in passives also separate
transitive from intransitive verbs in wh questions, in both embedded and
matrix questions. They do so only when a wh filler is present ("what John
destroyed" vs "that John destroyed"). In the plain declarative frame they
do not separate the classes, just as they barely did in the active "has"
frame in D11.

So they are not passive-specific. They read the verb class when the
verb's object position is empty: after a passive auxiliary, or with a
fronted wh filler. That makes them a shared reader of a gap-sensitive
verb-class value, downstream of the early d coordinate. The in-range move
on d drives them in both constructions.

This holds for recruitment, not mediation: no neuron was intervened on in
the filler-gap frames. Which MLP layers write the filler-gap preference
stays open, because the declared linear readout did not track L there.

# Part B7

## B7. Switch neurons in the wild

D11's top switch neurons on 2,000 Pile-10k documents (722,625 tokens, fp32
post-activations). Full tables, examples and the annotation: `b7_wild.md`.
TSUBAME job: 9008705. Proxies are surface patterns; the labels come from
the blind annotation of each top-3 neuron's 40 strongest contexts.

**Top 100 by raw activation** (share [95% CI], document resampling with the
top list recomputed per draw). Participle-candidate baseline: be/get-form
0.26, " by" next 0.05.

| rank | neuron | (a) participle | (b) be/get in 3 | (c) wh/rel in 8 | (d) " by" next | frequent words |
|---:|---|---|---|---|---|---|
| 1 | MLP11.n3462 | 0.24 [0.15, 0.32] | 0.10 [0.04, 0.16] | 0.18 [0.10, 0.26] | 0.09 [0.03, 0.14] | use, used, to, see, want |
| 2 | MLP14.n2114 | 0.76 [0.68, 0.85] | 0.45 [0.34, 0.55] | 0.07 [0.02, 0.12] | 0.53 [0.42, 0.62] | used, provided, told, instructed |
| 3 | MLP14.n5407 | 0.89 [0.82, 0.95] | 0.53 [0.43, 0.62] | 0.01 [0.00, 0.04] | 0.78 [0.68, 0.86] | affected, followed, inspired, accompanied |

The top lists are spread over 88–95 documents. The one-per-document
versions agree to within 0.06 on every proxy (ranks 1–3).

**Annotation** (top 40 each; Claude and Codex, independent; 120 items, κ =
1.00):

| rank | neuron | verbal passive | object gap | other / adjectival | reading |
|---:|---|---:|---:|---:|---|
| 1 | MLP11.n3462 | 0.23 | 0.78 | 0 | **gap-sensitive participle reader** |
| 2 | MLP14.n2114 | 0.85 | 0 | 0.15 | **passive-participle detector** |
| 3 | MLP14.n5407 | 0.97 | 0 | 0.03 | **passive-participle detector** |

- **Rank 1** fires on a transitive verb whose object is missing after it.
  Most cases are relative clauses ("the only patient she sees", "the
  advice you've heard", "the work India does") and infinitival relatives ("no law
  to respect", "the module to use", "stuff to sell"). The other 0.23 are
  reduced passive relatives ("the taxes paid", "the rate achieved via ...").
  Every one of its 40 contexts is either a passive participle or an object
  gap.
- **Ranks 2–3** fire on passive participles with a by-agent coming:
  be-passives ("is dominated by", "has been approved by"), reduced
  relatives ("blood vessels surrounded by", "a startup headed by") and
  participial clauses ("Inspired by ...", "..., followed by"). Their
  other tokens sit inside a passive just after the participle ("adopted as
  such by", "trod upon by", "given a green light").
- **Ranks 4–10** (proxies and descriptions only):
  - 4, MLP11.n4925: stranded prepositions in relatives and questions ("the
    city you'd like to see the weather in", "did not take part in");
  - 5, MLP14.n4241: passive participles after modal or perfect "be"
    ("will be characterized", "can be resolved"), mostly academic text;
    be/get 0.82;
  - 6, MLP14.n496: the word before an agentive " by", participle or noun
    ("manufactured by", "photograph by", "designs by"); " by" next 0.52;
  - 7, MLP11.n1686: a mix of passive participles ("moderated by", "is run
    by") and infinitival gaps ("weeks to complete", "too long to cast");
  - 8, MLP14.n3517: participles with "with" / "to" complements
    ("entrusted to", "endowed with", "charged with") and object pronouns,
    plus German nouns and names (about 28 of the 100, from 6 documents);
  - 9, MLP11.n5408 (s_n −): raw top = verbs and participles before a
    stranded preposition ("the company I work for", "what it is made of");
    its switch-direction (lowest) contexts are punctuation and articles;
  - 10, MLP14.n503 (s_n −): weak (max 2.5), no clear pattern.

**Against the predictions.**
- Top-3 neurons are passive-participle detectors: **held for ranks 2 and
  3, missed for rank 1**. For 2 and 3, proxy (b) is also above the
  participle-candidate baseline (0.45 and 0.53 vs 0.26; CIs exclude it).
  For rank 1 it is below (0.10).
- Gap-sensitive reading possible for one of the three: **held** (rank 1,
  object gap 0.78, passive + gap 1.00).
- " by" follows in ≥ 20% of the top 100: **held for ranks 2 and 3** (0.53,
  0.78), **missed for rank 1** (0.09).

**Reading.** The largest switch neuron, MLP11.n3462, is not a passive
detector. On natural text it responds to a transitive verb whose object
position is empty: object relatives, infinitival relatives, and reduced
passives. This is the C9 result seen from the other side: in C9 the
neurons separated the verb classes only when a wh filler was present.

The two MLP14 neurons are passive-participle detectors and predict a
following " by". One hypothesis is a gap-sensitive reader at MLP11
feeding passive-specific units at MLP14. That would fit B6, where the late
object MLPs depend on MLPs 11–17. But neither B6 nor B7 tested this
neuron-to-neuron route. B7 describes what the neurons respond to; it is
not a causal test.

## Deviations and checks (B7)

- **Annotation guide.** The guide (`results/round5/wild/b7_annotation_guide.md`)
  added one precedence rule before labelling: when "active perfect or
  simple past" and "object gap" both apply ("the advice you've heard"),
  label object gap. It decides 3 of the 120 items ("money he used", "the
  take-away I've got", "the advice you've heard").
- **Blinding.** Codex saw only the guide and the shuffled contexts, in an
  empty directory. Claude also labelled the shuffled contexts without
  neuron identity, but knew the hypothesis (it wrote the plan) and had seen
  24 of the 120 contexts with their neuron in the describe output before
  labelling. Claude's labels were committed (`07df118`) before Codex
  labelled. Hashes: `results/round5/wild/b7_annotation_frozen.txt`.
- **κ = 1.00.** The two annotators agreed on all 120 items, edge cases
  included (a Spanish infinitival relative, a stranded preposition in a
  prepositional passive, one adjectival "overrun with"). Codex's notes are
  its own wording. The top contexts are unambiguous, but both annotators
  are language models with the same guide. κ is not a human-agreement
  measure.
- **Proxies.** (a)–(c) are defined only at word-final tokens: the run saved
  the preceding words there. Other tokens count as failing them, which
  lowers the all-token baseline. The participle-candidate baseline is
  unaffected. Proxy (b) misses reduced passives (no be-form), so it
  understates the passive share of ranks 2–3.
- **Token text.** 462 tokens contain a carriage return. They are written
  as "␍" in `b7_top_contexts.csv` so the CSV parses. The annotation items
  contain none, and their hash is unchanged.
