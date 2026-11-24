# Paper outline: which results carry which claim (Nov 24, revised after Codex review)

This file sits under `docs/paper_plan.md`. It maps each claim to the results that carry it in the main text, the ones that go to the appendix, and the experiments still worth running before the Dec 11 freeze. Anything not listed as main or appendix is cut from this paper (it stays in the thesis).

## Contribution and question

**One sentence.** We identify verb-class information that supports causal transfer across verbs and constructions in three language models, measure how much of a verb's passive *by* preference it carries, dissect the voice-sensitive components that read it in Pythia, and use it to repair rare verbs' predictions.

**Organizing question.** What information about a verb's transitivity is reusable across verbs and constructions, and how does context change its effect on the next word? Lookup vs computation is not the frame (the thesis has the more nuanced version); the discussion says in a paragraph how the results bear on it. The paper does not claim a verdict: the in-sample vs cross-fitted LEACE difference is not an additive "per-verb" component (fit size, covariance and the scrub sequence all differ; Settled 13), and with more training verbs the cross-fitted share keeps rising (0.59 → 0.42 remaining on the same items, Settled 16).

## Thesis, with the limits next to it

- A verb-class signal at the verb, sensitive to surface NP cues (nonce, R4-C9), supports interventions that transfer across several constructions. The effects are model- and instrument-dependent: OLMo misses the declared passive margin, T3 flips between v and u_md, embedded wh is not uniform.
- Erasers fit on other verbs reduce the passive *by* gap by 0.53 [0.44, 0.67] (Pythia), 0.83 [0.61, 1.22] (Gemma) and 0.57 [0.45, 0.74] (OLMo). These are pooled reductions of a T − I gap, not per-verb fractions. Say "about half or more", never "at least half" (the Pythia and OLMo lower bounds are below 0.5).
- In Pythia, heads route voice into MLPs 11–17, and a passive-conjunction set in MLPs 11 / 14 carries most of the edit-induced *by* switch. These components are not necessary for natural behavior (R5-B5).
- Rare verbs keep class information that is decodable from their own verb state and transferable. Their deficit shows in predicting the participle itself, and writing the class read from the verb state into the participle's output row repairs it. R5-A5 rules out an exclusively output-side account.

Words to avoid: "one feature" (rank-1 success is not uniqueness; class information survives erasure), "every construction", "an NP follows this verb" as the meaning for known verbs, "circuit", "downstream" for the rare-verb deficit.

## Section plan (8 pages, 0.5 reserve)

| § | Title (working) | Pages | Carries |
|---|---|---|---|
| 1 | Introduction | 0.75 | Question, contributions |
| 2 | Setup | 1.0 | Minimal pairs, 3 models, instruments (v, u_md, LEACE), D / natural gap as the common unit, rules in one paragraph |
| 3 | Reusable class information: transfer and context | 1.75 | C1 + C2 |
| 4 | How much of the preference it carries | 1.25 | C3 |
| 5 | How Pythia reads it | 1.0 | C4 (0.6–0.8 if its checks fail) |
| 6 | Rare verbs: reading the class to repair the prediction | 0.75 | C5 (+ editing as application) |
| 7 | Related work, discussion, limitations | 1.25 | |
| | reserve | 0.25 | |

C4 is introduced as "one implementation of the context dependence observed above": it explains how the same verb-site edit produces different continuation effects in actives, passives and wh frames. It does not establish C5.

## Claims and evidence

### C1 + C2. Transfer and context (§3)

- **Main:** transfer of v and u_md to passive *by*, OR, TC, embedded and matrix wh in 3 models, as D / natural gap curves at representative sites with declared sites marked (R2-ptest, R4-A2, R4-B, R5-C8; Settled 9, 11, 12). u_md is training-free (the Kumon & Yanaka answer). New verbs: passive transfer holds on 74 unseen plain verbs in 7 of 8 declared readings (Settled 16, 20; these are site × instrument readings, not 8 constructions).
- **Main, one paragraph:** the Pythia split edit as the illusion defense. The on-axis piece carries 0.66–0.80 of *by*, the off-axis piece the O rise (Settled 2). In the same paragraph, the cross-model failures: Gemma R-A2 not held, OLMo R-A1 not held (Settled 10).
- **Main, nonce panel in the transfer figure:** class set from context, onset 0.12–0.31 of depth, driven by an adjacent NP (R4-C9; Settled 18). Frame invariance early, frame-specific with depth, with Gemma's non-monotone FI stated (Settled 5, 10).
- **Main, one clause each:** OLMo passive *by* misses the declared margin (0.30–1.19 of the gap at 0–11); T3 instrument flips.
- **Appendix:** full depth curves, in-range profiles, gap- and self-scaled classes, staging, R4-C8, DAS stability (exploratory; Pythia's declared reading failed its recompute check), reverse DAS R3-B5.

### C3. How much is verb-general (§4)

- **Main:** cross-fitted removal 0.53 / 0.83 / 0.57 (268 pairs); cross-frame: active-fit erasers remove 0.82 / 0.82 / 0.94 of what passive-fit ones do (Settled 13, 17); suppression reading, most closure from bad passives gaining *by* (Settled 8).
- **Main, needs matching:** the in-sample and permutation references are still the 126-pair runs. Rerun both on the 268-pair set before quoting "in-sample removes nearly all" next to the expanded headline. Until then, label the inventories.
- **Main, needs running:** a learning curve with a fixed evaluation population, repeated training subsets, and equal-size fits that include vs exclude the evaluated verbs. If the cross-fitted share keeps rising with n, the in-sample advantage is mostly fit size, and the text says so. Then rank-k erasure as a sensitivity test (Pythia first), refitting on the erased representation each step (re-applying one binary LEACE projection is not rank-k), with the removed rank, held-out residual class information and KL against rank-matched random controls. It constrains "shared in more directions"; it does not close the question.
- **Appendix:** 126-pair numbers, LOO, translation flag, centering fix, side splits, subset tables.

### C4. How Pythia reads it (§5; stored site-8 v)

- **Main (one figure):** R2-mech (object MLPs vanish in passives; MLPs 11–17 flip sign on *by*) → heads switch the *by* MLPs between voices, 67–115% (R3-B7), a voice switch (R4-D10) → the 2,782 passive-conjunction neurons in MLPs 11 / 14 carry 0.65 [0.60, 0.70] of the MLP11+14 switch on held-out pairs, and the top 10 carry 0.66 (exploratory) (R4-D11) → the same neurons read the class in wh-gap frames with a filler present (R5-C9).
- **Denominators in the figure:** 0.65 is a share of the MLP11+14 intervention-induced switch, not of natural behavior. Next to it: top-50 ablation removes 0.09 of the natural *by* gap (R5-B5); serial chain mixed (R5-B6); B7 missed the "few heads" prediction.
- **Main, one sentence:** in Pile text the top neurons fire on passives with *by* next and on transitive verbs missing an object (R5-B7).
- **Appendix:** dose (R2-dose), persistence (R3-A2), *got* frames (R3-B6), previous-token check (R4-A5), E13T (neurons beat random by +0.169 [0.107, 0.234]; the declared claim fails on the Head control and deficit closure, not on the random comparison).

### C5. Rare verbs: the class is read correctly but not used when predicting the verb (§6, 0.75 page)

The two positions are the point, not a confound. After reading a rare verb, the model has its class (site 8, along v). When predicting that verb after "was", it uses the participle's output row, and rare intransitives' rows sit too close to transitives' along the was − has direction (R5-A1 / A2). So the information exists on the input side and is missing on the output side. The edit shows it is usable: read the class from the verb's own state, write it into the output row, and the behavior is repaired.

- **Main:** the deficit is on the intransitive side at every size from 160M to 12B (R5-A1, R5-D10b, one sentence). The class is decodable from a rare verb's own site-8 z along v (R6-C4: 31 / 32 rare singletons, D11 Tail 44 / 44), and a rare-trained direction transfers to frequent verbs at 0.94–1.00 of v (Settled 15). **Classify-from-z edit** (R6-C4): moving each rare participle's output row to the Head mean of the class read from z raises affected passive margins by +0.53 [0.24, 0.83] nats, equal to the oracle, with Head pairs, unaffected items and Pile loss unchanged. Label-free (no class) +0.25 [−0.08, 0.58], unresolved.
- **Limits, in the text:** 12 affected Tail single-token pairs; the classification was seen before it was declared; R5-A5 (a sentence-final probe also shows a rarity drop, so not only output-side); not a matched localization with E13 (which intervenes after the participle). Gemma ties input and output rows (Settled 4), so the asymmetry is a Pythia statement unless ported.
- **Appendix:** R5-A2 decomposition, R5-A4 oracle conditions, R4-E13 / E13T, R2-freq, R3-A1, R4-D12.
- **Dynamics, if it lands:** one paragraph here. Behavior is done (R6-A6, R7-A8). Behavioral T / I separation appears in band order (Head 256, Tail 1000, XTail 3000, R7-A8d), so the open question is whether the verb-site signal follows the same order. Declare the prediction first and report either outcome.

## Figures

1. Transfer and context: constructions × models (v and u_md, D / gap), nonce onset panel.
2. Erasure: cross-fitted vs in-sample vs permutation (matched inventory), learning curve, rank-k panel.
3. Pythia readers: heads → MLP switch shares, conjunction-neuron share, with the necessity number beside it.
4. Rare verbs: classify-from-z and rare-trained transfer (plus a checkpoint panel if dynamics lands).

No separate summary table if it repeats the figures; the pass / fail table against declared rules is in the appendix.

## Still to run, by what it buys (17 days to freeze)

| Priority | Experiment | Buys | Cost |
|---|---|---|---|
| 1 | Expanded-set in-sample LEACE + 20-permutation control, 3 models | Matched references for the C3 headline | LEACE only |
| 1 | Learning curve, fixed eval population, include vs exclude at equal size, 3 models | Whether the in-sample advantage is fit size | LEACE only |
| 2 | Rank-k erasure with refitting and controls, Pythia first | Constrains "shared in more directions" | LEACE only |
| 2 | C4 on new verbs: frozen neuron set, class × voice interaction on the 142 new pairs (74 plain primary), Pythia | C4 is not about the original verbs | forward passes |
| 2 | C4 instrument check: repeat the key switch with stored v, P (already trained) and u_md at matched dose | C4 is about a class signal, not one DAS direction | < 1 GPU-hour |
| 3 | C4 mediation: under the verb-site edit, restore the selected neurons to unedited values (and the converse replay), with random-neuron controls; read final log P(*by*) | Necessity / sufficiency for the edit's effect, the bridge from C1 to C4 | ~1 GPU-hour |
| 3 | Classify-from-z edit on more verbs: D11 Tail singletons and the new single-token rare pairs, z from v, P and u_md, frozen θ from Head | Takes §6 beyond 12 affected pairs; shows the edit does not need the stored v | forward passes |
| 4 | Pythia-6.9B: natural gaps on the paper's items and u_md transfer; LEACE only with its controls | Scale | a few GPU-hours |
| 5 | Dynamics: a small declared checkpoint set bracketing steps 256 / 1000 / 3000, training-free verb-site measures, Pythia | The paragraph in §6 (sensei asked) | a few GPU-hours |
| — | Second-model conjunction neurons | Optional; not a substitute for the mediation test | ~1 GPU-hour + porting |

## Cut from this paper

- **Replacing v: decided (Nov 24), v2 replaces v** in the core chain of all three models (`reports/v2_retrain_rules.md`). The declared gate passed at every rule site (v2 − v held-out cross IIA ≥ 0 in point estimate everywhere, no CI below −0.02; `reports/v2/instrument_measures.md`). The downstream chain is running. The Pythia site-8 mechanism tail stays on v unless ported (decide by Dec 4).
- DAS at 6.9B and at checkpoints; alternating verbs; SAE lookup; further direction-check suites; staging as a headline claim; embedded wh beyond one limitation sentence.
