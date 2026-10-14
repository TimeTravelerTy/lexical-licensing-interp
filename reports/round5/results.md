# Round 5: results

Plan, predictions and decision rules: `plan.md`, each part committed before
its runs (part 0 and A1–A2: `d3b6630`, builder fix `c9b673a`, analysis fixes
after code review `32feb71`, bf16 clarification `21671df`; B5–B6:
`c6be4ba`). Generated reports, one per experiment:
- 0: `../results_index.md`;
- A1–A2: `a1_frame.md`, `a2_decomposition.md`.

TSUBAME jobs: 9007119 (A1–A2), 9007211 (curated `passive_1` rescored in
fp32).

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
