# Round 5: results

Plan, predictions and decision rules: `plan.md`, each part committed before
its runs (part 0 and A1–A2: `d3b6630`, builder fix `c9b673a`, analysis fixes
after code review `32feb71`, bf16 clarification `21671df`; B5–B6:
`c6be4ba`). Generated reports, one per experiment:
- 0: `../results_index.md`;
- A1–A2: `a1_frame.md`, `a2_decomposition.md`;
- B5–B6: `b5_necessity.md`, `b6_serial.md`;
- C8–C9: `c8_fillergap_gates.md`, `c8_fillergap.md`, `c9_shared_readers.md`;
- A3: `a3_counts.md`; A5: `a5_probe.md`;
- B7: `b7_wild.md`;
- D: `d10_scaleup.md`, `d10b_suite.md` (and `suite/`), `d10c_yesno.md`,
  `d11_verbs.md`.

TSUBAME jobs: 9007119 (A1–A2), 9007211 (curated `passive_1` rescored in
fp32), 9007340 (B5; earlier attempts 9007281 and 9007322 stopped at the
self-patch check, see below), 9007282 (B6), 9007314 (C8 natural pass),
9007336 (C8 patches), 9008384 (C9), 9008634 (A5), 9008705 (B7), 9008724 (D10), 9009081 (D10b, D10c; 9008725
stopped at a check, see D10b).

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
- **B7, the switch neurons in natural text.** The top three neurons'
  strongest Pile contexts, labelled blind by Claude and Codex (κ = 1.00):
  - MLP11.n3462 (rank 1) fires on transitive verbs with a missing object:
    object relatives and infinitival relatives (0.78), reduced passives
    (0.23);
  - MLP14.n2114 and MLP14.n5407 fire on passive participles (0.85, 0.97),
    mostly followed by " by" (0.53, 0.78 of the top 100).
  The rank-1 neuron's gap reading matches C9.
- **D10, scale-up.** 6.9B and 12B fit on one H100 in fp32 at round-2 batch
  sizes. A round-2-style DAS replication plus path patching costs about 3
  GPU-hours on 6.9B and 5 on 12B.

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

# Part A3–A5

## A3. Training-data predictor (Pile counts)

Exact infini-gram counts in the Pile training set (`v4_piletrain_llama`;
6,574 queries, none failed). Outcome: Δ_pass(V) from A1 (freq frames).
Report: `a3_counts.md`.

- **Audit.** Every slot query's tokens end with the overall query's
  tokens; no verb was dropped. No curated or pool verb has a zero count in
  any slot.
- **What "was V" counts** (hand audit, 10 verbs, up to 10 contexts each):
  - for transitives, 47 of 47 contexts are verbal passives;
  - for intransitives, 27 of 40 are passives, 24 of them prepositional
    ("was apologized to", "was glared at", "was flirted with") and 3 of a
    rare transitive sense ("the water was flirted away");
  - the rest are clause boundaries or nonstandard use ("whatever it was
    happened"; 10 of 10 for "happened");
  - "was hitchhiked" never occurs.

  So c_be is a passive count for T verbs and mainly a prepositional-
  passive count for I verbs.
- **Curated verbs** (primary; CV MSE, 10 folds grouped by pair):

| class | r(Zipf) | r(log c) | r(log c_be) | partial r(ρ_be \| log c) | CV ~log c | ~log c_be | ~log c + log c_be | difference [95% CI] | ~Zipf | decision |
|---|---|---|---|---|---:|---:|---:|---|---:|---|
| T | −0.14 | −0.14 | −0.06 | +0.16 [+0.01, +0.31] | 2.070 | 2.095 | 2.026 | +0.045 [−0.091, +0.198] | 2.077 | **unresolved** |
| I | −0.58 | −0.50 | −0.26 | +0.19 [+0.00, +0.37] | 2.649 | 3.350 | 2.606 | +0.043 [−0.212, +0.315] | 2.353 | **unresolved** |

  - The smoothing, capitalization and was + been sensitivities are all
    unresolved too.
  - For I verbs the slot count alone predicts much worse than overall
    frequency (3.35 vs 2.65). wordfreq Zipf predicts better than Pile log c
    (2.35; in-sample R² 0.34 vs 0.25).
- **Pool verbs** (secondary): T **slot frequency adds**: CV error 2.049 →
  1.636, difference +0.41 [+0.25, +0.61]; in-sample R² rises from 0.00
  to 0.20. It holds under every smoothing. I: unresolved (+0.34 [−0.09,
  +1.00]).
- **Pairs, L ~ T and I predictors:** unresolved (+0.18 [−0.43, +0.76]), in
  every sensitivity.
- **Conditional on overall frequency**, verbs seen more often after a
  be-form have a higher Δ_pass, i.e. they are more acceptable after "was".
  The partial r is positive in all four class × population cells (+0.16 to
  +0.45).

**Against the predictions.**
- I: overall frequency suffices: **missed**. The point estimate shows no
  gain (1.6% of the error), but the CI is too wide to call it
  negligible.
- T: slot frequency adds: **missed on the curated verbs** (unresolved).
  **Held in the secondary pool** (627 verbs).
- L: overall frequency suffices: **missed** (unresolved).
- Descriptive: Pile log c does not beat wordfreq Zipf.

**Reading.** With 126 verbs per class, the curated test cannot separate
"slot counts add" from "overall frequency suffices". The larger pool shows
that, at fixed overall frequency, a transitive participle's Pile passive
count predicts how acceptable Pythia finds it after "was". For
intransitives the picture fits preemption: overall frequency (Zipf) is the
best single predictor of how strongly they are rejected. Their "was V"
count mostly records prepositional passives and adds nothing reliable.

## A5. Non-LP readout: a grammaticality probe

Wang et al. 2026's probe, adapted (`plan.md`, A5): L2 logistic regression
on the last token's residual state, trained to separate 49,998 corpus
sentences (Gutenberg, WikiText-103) from perturbed copies. TSUBAME job
9008634. Report: `a5_probe.md`.

- **Probe quality.** Layer 13 has the best dev AUC (0.930). The baselines
  reach 0.569 (length, punctuation and casing) and 0.844 (LM log-prob per
  token). Three document splits give the same layer accuracy within 1 pp.
  **Informative** on all three criteria (AUC ≥ 0.85; +0.36 over the
  artifact baseline; Head `passive_1` 88.2%, CI above 50%).
- **Released `passive_1`** (accuracy, %, 95% CI):

| readout | Head | Tail | XTail | D = Head − XTail | Head − Tail |
|---|---|---|---|---|---|
| probe | 88.2 [86.2, 90.1] | 82.0 [79.6, 84.4] | 71.9 [69.1, 74.7] | +16.3 [+12.9, +19.8] | +6.2 [+3.0, +9.2] |
| LP | 79.1 [76.7, 81.7] | 79.2 [76.8, 81.6] | 69.4 [66.7, 72.1] | +9.7 [+5.9, +13.6] | −0.1 [−3.6, +3.6] |

  - Paired: D_LP − D_probe = −6.6 [−11.0, −2.0]. The probe's drop is
    *larger* than LP's. D_probe − 0.5 D_LP = +11.5 [+7.9, +14.9].
  - Good-verb clusters give the same: D_LP − D_probe −6.6 [−12.5, −0.8].
  - The probe-logit margin shrinks with rarity: +3.00 / +1.61 / +0.88.
  - Splits 1–2 give D_probe +15.9 and +16.8.
- **Secondary.**
  - `passive_2`: the same pattern (D_probe +19.8 vs D_LP +8.3).
  - Curated band-cross `passive_1`: probe 89 / 85 / 79%, LP 81 / 78 / 73%.
  - All 67 paradigms: mean Head accuracy is 77.6% for the probe and 79.2%
    for LP. The mean Head − XTail drop is 2.6 pp for the probe and 1.7 pp
    for LP. The two drops correlate at +0.64 across paradigms.

**Declared decisions.**
- Detectable decline: **yes**.
- Declines like LP: **yes**.
- Smaller decline than LP: **no**; the decline is larger.
- Practically no decline: **no**.

**Against the predictions.**
- The probe is informative: **held**.
- Its Head accuracy is below LP's: **missed** (88 vs 79%).
- Smaller decline than LP, not "declines like LP": **missed**. Both
  "declines like LP" and a significantly larger drop hold.

**Reading.** The rare-verb deficit is not confined to the participle's
output probability. A probe on the sentence-final state separates good
from bad passives better than LP on frequent verbs, and it loses more of
that advantage for rare verbs. It already drops at Tail, where LP does not.
So the representation at the end of the sentence carries less of the
good/bad distinction for rare verbs. That argues against a purely
output-side account of the deficit, which was the prediction's rationale.

One caveat: the probe was trained to detect perturbed text, and rare
words look unusual in both members of an XTail pair. That may compress its
margin through a familiarity signal as well as through grammaticality.
This experiment cannot separate the two.

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

# Part D

## D10. Scale-up feasibility (6.9B, 12B)

Timing only, with random bases and discarded updates (fp32, TF32 off, one
H100; TSUBAME job 9008724). Table and derivation: `d10_scaleup.md`.

| model | weights | DAS step / eval batch | DAS peak | path-patch batch (short / long) | path-patch peak |
|---|---:|---|---:|---|---:|
| 1.4B | 5.3 GiB | 0.057 / 0.19 s | 6.4 GiB | 0.41 / 0.46 s | 7.6 GiB |
| 6.9B | 25.5 GiB | 0.225 / 0.76 s | 27.5 GiB | 2.30 / 2.42 s | 37.0 GiB |
| 12B | 44.1 GiB | 0.360 / 1.33 s | 46.9 GiB | 4.39 / 4.63 s | 62.3 GiB |

- Everything fits on one GPU in fp32 at the round-2 batch sizes (64 for
  training, 512 for evaluation). No out-of-memory fallback was needed.
- **Per run** (round-2 settings: 31 steps and 2 evaluation batches per fit,
  15 fits):
  - one DAS run takes 2.1 min on 6.9B and 3.5 min on 12B;
  - a full round-2 final stage (3 ranks, 100 random bases × 2, shuffled
    labels) takes 1.4 h and 2.5 h;
  - an 8-site sweep takes 45 min and 1.2 h.
- **Path patching** (round 4's 11,657 rows, three frames): 14–19 min on
  6.9B and 28–42 min on 12B. The upper end scales for the deeper flagged
  layers at matched depth.
- **Calibration** on 1.4B:
  - the path-patching estimate (159 s) matches round 4's actual runs
    (164–180 s);
  - the DAS model gives 42 min for round 2's final job, which took 38 min.
- **Site plan** (matched relative depth):
  - 6.9B: 5, 8, 11, 13, 16, 19, 21, 23;
  - 12B: 6, 9, 12, 15, 18, 21, 24, 26.

**Reading.** A 6.9B replication of the round-2 DAS pipeline (sweep and
final) plus two path-patching frame sets costs about 3 GPU-hours. The same
on 12B costs about 5. Both are feasible on gpu_1 nodes without code
changes.

## D10b. A1/A2 across the Pythia suite

A1's runner on six more Pythia sizes (freq prefixes, fp32; TSUBAME job
9009081, 26 min of forward passes in total). Each model was analysed with
`analyze_frame_contrast.py --no-accounting`. Table: `d10b_suite.md`;
per-model reports: `suite/pythia-*/`.

| model | L: Head / Tail / XTail | ΔL (Head − XTail) | intransitive side −ΔΔ_pass(I) | transitive side ΔΔ_pass(T) | Head AUC | singleton D_L |
|---|---|---|---:|---:|---:|---|
| 160M | 3.10 / 2.08 / 1.53 | +1.57 [+0.76, +2.28] | +2.10 | −0.53 | 0.95 | +1.72 [+0.80, +2.62] |
| 410M | 3.86 / 3.05 / 1.42 | +2.44 [+1.66, +3.22] | +2.94 | −0.49 | 0.96 | +1.85 [+0.64, +3.05] |
| 1B | 4.02 / 3.14 / 1.73 | +2.29 [+1.45, +3.10] | +2.86 | −0.57 | 0.96 | +2.20 [+1.00, +3.38] |
| 1.4B | 3.84 / 2.82 / 1.72 | +2.12 [+1.26, +2.94] | +2.60 | −0.48 | 0.96 | +2.09 [+1.00, +3.23] |
| 2.8B | 3.74 / 2.99 / 2.00 | +1.74 [+0.84, +2.61] | +2.25 | −0.51 | 0.95 | +2.11 [+0.98, +3.26] |
| 6.9B | 3.87 / 2.85 / 1.59 | +2.27 [+1.41, +3.09] | +2.80 | −0.53 | 0.95 | +2.11 [+1.08, +3.19] |
| 12B | 3.92 / 3.06 / 2.04 | +1.88 [+0.91, +2.79] | +2.57 | −0.69 | 0.95 | +2.25 [+1.14, +3.43] |

- ΔL is a **deficit** (A1's rule) at every size, 160M included.
- At every size the deficit comes from the intransitive side: rare
  intransitives are rejected less after "was". The transitive side moves
  slightly the other way.
- ΔM_has is unresolved everywhere, so the A1 reading stays unresolved, as
  for 1.4B.
- Head AUC and the singleton contrast D_L hardly change with size.

**Against the prediction.** The deficit appears at every size ≥ 410M,
stays on the intransitive side, and is within ±50% of 1.4B's (1.06–3.18):
**held** for all five sizes (1.74–2.44). 160M shows it too.

**Reading.** Scale does not remove the rare-verb deficit at the passive
slot. Across a 75-fold range of parameters it stays about 2 nats, and its
source stays the same: frequent intransitive participles are rejected
after "was", rare ones are not. The models were all trained on the same
data, so this fits a deficit that tracks training exposure more than
model capacity.

**Checks and deviations (D10b).**
- Tolerance clarification (pre-run, `plan.md`, D10b): 160M's largest
  |logit| at the slot is 845, against 17–20 for every other size. That
  explains its larger fp32 errors (reconstruction 1.4e-3, identity
  2.7e-3), both within the 1e-2 tolerance. All other models are below
  6.4e-5.
- The planned timing block was skipped. The full runs took 26 min, well
  inside the 4 GPU-hour cap.

## D10c. FreqBLiMP passive_1 Yes/No on 1.4B, 6.9B and 12B

Task 4's Yes/No prompt on the released `passive_1` items (1,000 per band),
fp32. TSUBAME job 9009081. Full table: `d10c_yesno.md`.

| band | 1.4B | 6.9B | 12B |
|---|---|---|---|
| Head | 0.499 [0.468, 0.530] **at chance** | 0.450 [0.420, 0.481] **below** | 0.398 [0.368, 0.428] **below** |
| Tail | 0.512 [0.480, 0.544] **at chance** | 0.460 [0.428, 0.490] **below** | 0.474 [0.443, 0.505] unresolved |
| XTail | 0.546 [0.516, 0.576] **above** | 0.537 [0.506, 0.568] **above** | 0.485 [0.454, 0.515] **at chance** |

- **The readout says "Yes" to almost everything.** Yes > No for 97–100%
  of sentences in every model and band, good and bad alike. The mean
  good − bad margin is at most 0.05 nats.
- **Verb-specific shifts.** Per bad verb, accuracy ranges from 0.05 to
  0.88, and the ranking changes between models. With the 20 Head bad
  verbs resampled (descriptive, not declared), 12B's Head CI stays below
  0.5 ([0.32, 0.47]). 1.4B's and 6.9B's Head CIs include 0.5.
- **Model comparisons.** 12B is below 1.4B on Head (−0.10 [−0.14, −0.06])
  and XTail (−0.06). 6.9B is below 1.4B on Head and Tail (−0.05).

**Against the predictions.**
- 1.4B Head about 55%: **missed** (practically at chance, 0.50).
- 1.4B Tail and XTail at chance: **held for Tail, missed for XTail**
  (0.55, above chance).
- 6.9B and 12B above chance on Head (60–70%): **missed**. Both are below
  chance.
- Weaker on XTail: **missed**. XTail is the best band for 6.9B and 12B.

**Reading.** Read this way, the base models do not judge these sentences:
the prompt elicits "Yes" regardless. What separates good from bad items is
a small verb-specific shift that larger models do not improve. So the
Yes/No readout is not a usable second measure of the rare-verb deficit at
any of these sizes. The LP contrast (A1) and the probe (A5) remain the
measures. Task 4's 60.8% on curated Head contexts did not carry over to
the released items.

## D11. Expanded verb list

Built locally from FreqBLiMP `passive_1` / `passive_2` (released items),
the verb-expansion inventory, round 4's candidates and the 126 curated
pairs. Report: `d11_verbs.md`; files: `data/round5/verbs_expanded/`;
loader: `scripts/verbs_expanded.py`.

- **Ratings.** 1,233 in-band lemmas (of 1,312 pooled), rated T / I /
  reject by Claude and by Codex, each blind to the source class and to the
  other. Agreement 0.94, κ = 0.88. Accepted only when both said T or both
  said I. Accepted: Head 319 T / 47 I, Tail 395 / 133, XTail 118 / 126.
- **Pairs.** Matching within band: the maximum number of one-to-one pairs
  within the calipers (participle Zipf ≤ 0.35, lemma Zipf ≤ 0.25), then
  the minimum total participle gap, then lemma order (exact integer
  assignment).
  - Head 47 pairs, Tail 132, XTail 102: 281 in total.
  - Mean participle gap 0.04, mean lemma gap 0.14.
  - Every accepted Head intransitive is paired. Head intransitives are the
    bottleneck, as expected.
- **Token cells** (T × I, single / multi-token): Head is mostly
  single-single (43 of 47). Tail and XTail are mostly multi-multi (91 and
  101).
- **Split.** 93 pairs contain a lemma used in an earlier DAS training or
  selection run and are trainable only. Of the other 188, one third per
  band × token cell is held out: 62 pairs (Head 4, Tail 29, XTail 29).
  The loader asserts lemma-disjoint splits and no held-out lemma in any
  earlier DAS run.
- 164 pairs contain a curated verb (flagged). No model was run on the list.

**Rule** (≥ 128 pairs, ≥ 40 per band): **met** (281; Head 47).

**Deviations and checks (D11).**
- The registry of historical DAS verbs covers every earlier DAS training or
  selection run in the repo (rounds 1–2, the verb-expansion pairs and the
  round-3 Head train / dev items), not only the two inventories the plan
  names. It was widened after Codex's review, and it only moves pairs
  from held-out to trainable.
- The rating guide was revised after Codex's review, before either
  rating was frozen. The draft rejected verbs with common transitive and
  intransitive uses (break-type); the plan's T criterion admits them, so
  they are T with an `optional` flag. Claude's labels were drafted under
  the draft guide and revised to the final one before freezing (31 rejects
  became T optional, 1 I became reject).
- Token counts for 15 participles missing from the existing inventories
  were computed with the Pythia tokenizer on the TSUBAME login node
  (`sources/ntok_extra.csv`).
- The matching was checked against brute force on 400 small random
  instances (cardinality, total gap and tie-break all agree).
