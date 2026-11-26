# Paper outline: which results carry which claim (Nov 24, revised after Codex review; C1 + C2 updated Nov 25 to v2 on the expanded set; C4 updated Nov 26 to v2; C5 restructured Nov 26; question, page plan, verb-position figure, §5 and dynamics revised Nov 26)

This file sits under `docs/paper_plan.md`. It maps each claim to the results that carry it in the main text, the ones that go to the appendix, and the experiments still worth running before the Dec 11 freeze. Anything not listed as main or appendix is cut from this paper (it stays in the thesis).

## Contribution and question

**One sentence.** We identify verb-class information that supports causal transfer across verbs and constructions in three language models, measure how much of a verb's passive *by* preference it carries, dissect the voice-sensitive components that read it in Pythia, and use it to repair rare verbs' predictions.

**Organizing question (reworded Nov 26 to what the paper answers).** Is a verb's transitivity represented in a form that is reused across verbs and constructions, how much of the behavior depends on it, and how does context turn it into different predictions? The "what information" part is answered descriptively: a graded position along the class direction (C1, verb-position figure), read by neurons that detect a transitive verb whose object is missing (C4). Do not claim more than that about its content. Lookup vs computation is not the frame (the thesis has the more nuanced version); the discussion says in a paragraph how the results bear on it. The paper does not claim a verdict: the in-sample vs cross-fitted LEACE difference is not an additive "per-verb" component (fit size, covariance and the scrub sequence all differ; Settled 13), and with more training verbs the cross-fitted share keeps rising (0.59 → 0.42 remaining on the same items, Settled 16).

## Thesis, with the limits next to it

- A verb-class signal at the verb, sensitive to surface NP cues (nonce, R4-C9), supports interventions that transfer across several constructions. On the held-out evaluation set (224 pairs; 89 plain primary) the v2 edit moves passive *by* by about the natural gap at the first rule site of Pythia and OLMo (D / G 1.05, 0.89), and by more than the gap in Gemma (1.8–2.5× from site 8). The object side is model- and construction-dependent: the edit matches the natural object-relative profile only at the first OR site of Pythia and OLMo, and overshoots the natural object effect elsewhere.
- Erasers fit on other verbs reduce the passive *by* gap by 0.53 [0.44, 0.67] (Pythia), 0.83 [0.61, 1.22] (Gemma) and 0.57 [0.45, 0.74] (OLMo). These are pooled reductions of a T − I gap, not per-verb fractions. Say "about half or more", never "at least half" (the Pythia and OLMo lower bounds are below 0.5).
- In Pythia, a passive-conjunction set in MLPs 11 / 14, selected with v2 on half the held-out pairs, carries 0.62 [0.58, 0.66] of those layers' edit-induced *by* switch on the other half. Clamping it mediates 0.19 [0.15, 0.23] of the edit's T − I effect on *by* (activation-matched neurons 0.01; all of MLPs 11 + 14 0.26). The training-free u_md gives the same numbers on the frozen set. Most of the edit's effect runs outside these two layers.
- The verb-class representation holds across frequency: a direction trained on frequent verbs only transfers to rare verbs, and one trained on rare verbs only transfers to frequent verbs (Pythia). Rare verbs' deficit shows mainly in predicting the participle itself, and writing the class read from the verb state into the participle's output row repairs it. R5-A5 rules out an exclusively output-side account.

Words to avoid: "one feature" (rank-1 success is not uniqueness; class information survives erasure), "every construction", "an NP follows this verb" as the meaning for known verbs, "circuit", "downstream" for the rare-verb deficit.

## Section plan (8 pages, 0.5 reserve)

| § | Title (working) | Pages | Carries |
|---|---|---|---|
| 1 | Introduction | 0.75 | Question, contributions |
| 2 | Setup | 1.0 | Minimal pairs, 3 models, instruments (v, u_md, LEACE), D / natural gap as the common unit, rules in one paragraph |
| 3 | Reusable class information: transfer and context | 1.75 | C1 + C2 |
| 4 | How much of the preference it carries | 1.0 | C3 (rank-k to the appendix) |
| 5 | How Pythia reads it | 1.0 | C4, built around what the core neurons detect (0.6–0.8 if its checks fail) |
| 6 | Rare verbs: reading the class to repair the prediction | 1.0 | C5 (+ editing as application, + training dynamics) |
| 7 | Related work, discussion, limitations | 1.25 | |
| | reserve | 0.25 | |

C4 is introduced as "one implementation of the context dependence observed above": it explains how the same verb-site edit produces different continuation effects in actives, passives and wh frames. It does not establish C5.

## Claims and evidence

### C1 + C2. Transfer and context (§3)

Source: `reports/v2_expanded_readings.md` (v2, 224 held-out pairs, primary 89 plain; criteria `reports/criteria_audit.md`). The paper reports only this set; it does not mention the 126-pair runs. Unit: D / natural gap G, with the T-arm recovery (T-patched − unpatched) / G beside it.

- **Main, passive *by* (Figure 1a):** D / G curves over all sites, three models, rule sites marked.
  - Pythia 8 / 12: 1.05 [0.77, 1.65] / 1.11; T arm 0.89 / 0.88. OLMo 9 / 11: 0.89 [0.61, 1.70] / 0.46; T arm 0.80 / 0.36. Wording: "about the size of the natural gap", not "reproduces" (the gap CIs are wide).
  - Gemma 5 / 8 / 12 / 15: 1.27 / 1.77 / 2.45 / 2.34 against a small natural gap (0.39 [0.15, 0.62]). Selective at 5 (T arm on bad bases +0.48, on good bases +0.05); at 15 partly a general *by* push (0.76 vs 0.39). Say both.
  - Deepest sites (Pythia 17, OLMo 12–13): *by* falls while O rises 3.6–4.1 nats. This is the object-next regime and belongs on the same curve.
  - Prepositional pairs (128) as their own panel or sentence: *by* RISE at every first and second rule site.
- **Main, depth profile (same figure):** *by* rise with small O, then both, then O with *by* suppressed, in depth order in Pythia (A 1–9, M 11–14, S− 16–17) and OLMo (A 1–4, M 6–11, S− 12–15). Gemma shows the first two regimes only (A 0–4, M 5–15). "Abstract" = O inside the small-effect margin, not zero: Pythia 8 O +0.29 [0.23, 0.36], 12% of the edit's active O effect. Disclose in methods that the equivalence-first classifier order was adopted after the output was read; OLMo's S1 holds only under it (frozen order in the appendix).
- **Main, OR / TC (Figure 1b), read against each construction's natural profile:**
  - At the first rule sites the edit raises the closure readout itself (OR +1.60 / +0.46 / +1.40 nats; TC +0.87 / +0.25 / +0.65, Pythia / Gemma / OLMo) and lowers PREP, so L rises in every model.
  - No detected object overshoot only for OR at Pythia 8 and OLMo 9 (D_O − G_O [−0.27, +0.21] and [−0.02, +0.50]; not equivalence). Elsewhere the object push exceeds the construction's natural O gap. New verbs alone overshoot slightly even at those two cells.
  - At the deepest sites L rises only because PREP falls faster than closure; do not call it licensing there.
- **Main, filler-gap (one paragraph, own frame items):** the audited frame interaction J_O (`results/v2_summary/t3_interaction.csv`). v2 is licensing-like in OLMo (matrix at every rule site, embedded from 11) and Gemma matrix 8–15; Pythia is licensing-like only at matrix 17, and its embedded frame goes the opposite way at 8 and 12. Model-dependent; say so.
- **Main, new verbs (one sentence + table row):** 30 new plain pairs, 23 lemma-fresh. D_new / D_orig CIs contain 1 at every rule site, and D_new − 0.5·D_orig is above 0 everywhere (Gemma 5 lemma-fresh lower bound 0.00).
- **Main, verb-position figure (opens §3; added Nov 26):** each verb's position along the class direction at the verb site, T vs I, by frequency band. Three points from one picture: the classes separate; positions vary within a class (the graded check below); frequency does not move a verb toward the other class (forward pointer to §6). Frequency on the y-axis and everywhere in §6 = log Pile count (c_total), bands only as a robustness line; counts exist for 108 of the draft's 344 verbs, fetch the rest (`fetch_pile_counts_r6.py`). The passive-frequency confound uses the same file's be_was / be_were / be_been counts.
  - Draft: `reports/figures/verb_positions_v_site8.png` (`scripts/plot_verb_positions.py`; Pythia site 8, frequent-trained v, 344 verbs in "has V-ed" frames, scale: training I mean 0, T mean 1). Exploratory, nothing declared. Within the intransitives, unaccusatives sit lowest (existed, died, happened, vanished, arrived) and verbs that take a clause or PP complement sit highest (replied, remarked, inquired, opted, responded); transitives with an object plus a second complement sit highest (appointed, awarded, assigned, designated). Worth a declared test. Passive-share check (Nov 26, exploratory; Pile counts of was / were / been + participle over c_total): within class, position tracks passive share at Spearman ρ 0.47 [0.37, 0.56] (T, 305) and 0.47 [0.21, 0.67] (I, 39); position vs log count ρ −0.05 (T) and −0.33 [−0.59, −0.01] (I). The intransitives that sit highest are ones that do occur after *be* (prepositional passives: *adhered to*, *remarked upon*). So the graded check and the ambitransitive prediction must enter object bias and passive share together, and the text may read the axis as passivizability rather than object-taking if passive share explains it.
  - Paper version: positions along v2 and u_md on the expanded eval verbs, 3 models (forward passes only).
  - **Ambitransitives in the paper version (Nov 26; idea already in `paper_plan.md` "Alternating verbs as a graded test"):** add alternating verbs, which were excluded by design, and predict they fall between the classes, ordered by their corpus transitivity bias. Bias from Pile counts with the existing query design: "P past the" (obj_ slots) vs "P past." (end_ slots), P in he / she / they / He / She / They; declare the bias measure and the prediction (rank correlation of position with bias among alternating verbs; their median between the I and T medians) before projecting. Keep the two alternation types apart: causative / inchoative (*broke*, *opened*: the intransitive use has the object as subject) vs unspecified object (*ate*, *read*), since they need not land in the same place. Selection rules for the alternating set are declared with it.
- **Main, nonce (one paragraph after the transfer results, no figure panel):** for a verb with no lexical history, the class at the verb is set by context: onset 0.12–0.31 of depth, driven by an adjacent NP (R4-C9; Settled 18). Framed with the position figure: known verbs are placed by identity, novel verbs by their context.
- **Main if it holds, graded check (declare before running):** within each class, correlate a verb's position along u_md / v2 with the model's own object expectation after it in actives (log P of an object-start continuation after "The X V-ed"), by rank correlation over verbs, both classes separately. A positive within-class correlation supports "how strongly this verb expects an object", not a binary label. Report either outcome; a null within class means the paper calls it a class signal and stops there.
- **Methods, one sentence each:** v2 replaces v (declared IIA gate); paired v2 − v on the same draws: equal or more *by*, equal or less O at the rule sites (Gemma 5 O push −41%). u_md as the training-free instrument (Kumon & Yanaka answer); its expanded-set numbers still to be quoted from the stored u_md runs.
- **Appendix:** split edit (decided Nov 25: appendix only, one pointer sentence in §3). It dissociates only at the second rule site (Pythia 12: Δ_by +0.34, Δ_O +0.61; OLMo 11: +0.81, +0.35), not at the first sites and not in Gemma, where the off-axis piece carries both readouts. Also: full per-site tables, frozen-order staging, gap- and self-scaled classes, v beside v2, DAS stability, reverse DAS R3-B5, R4-C8.

### C3. How much is verb-general (§4)

- **Main:** cross-fitted removal 0.53 / 0.83 / 0.57 (268 pairs); cross-frame: active-fit erasers remove 0.82 / 0.82 / 0.94 of what passive-fit ones do (Settled 13, 17); suppression reading, most closure from bad passives gaining *by* (Settled 8).
- **Main, needs matching:** the in-sample and permutation references are still the 126-pair runs. Rerun both on the 268-pair set before quoting "in-sample removes nearly all" next to the expanded headline. Until then, label the inventories.
- **Main, needs running:** a learning curve with a fixed evaluation population, repeated training subsets, and equal-size fits that include vs exclude the evaluated verbs. If the cross-fitted share keeps rising with n, the in-sample advantage is mostly fit size, and the text says so. Rank-k erasure (Pythia first) goes to the appendix if it runs: refitting on the erased representation each step (re-applying one binary LEACE projection is not rank-k), with the removed rank, held-out residual class information and KL against rank-matched random controls. It constrains "shared in more directions"; it does not close the question.
- **Appendix:** 126-pair numbers, LOO, translation flag, centering fix, side splits, subset tables.

### C4. How Pythia reads it (§5; v2 on the held-out set, `reports/pythia/v2_mech/readings.md`)

**Story (Nov 26):** the class signal at the verb is read in MLPs 11 / 14 by neurons whose own contexts say what they detect: a transitive verb whose object is missing. Passives are one case of that, filler-gap is another. Lead with the labelled core neurons and their causal shares, then the set-level mediation, then where the rest of the effect goes.

- **Main, core neurons (top 5 by half-A direct effect on the *by* switch; same five as v's set):** MLP11.n3462 reads an object missing after a verb (0.78 of its top 40 Pile contexts are object gaps, the rest passives: every one has the object missing; R5-B7); MLP14.n2114, n5407 and n4241 detect passive participles (0.85 / 0.97 / 0.72–0.75 of top contexts; n4241 labelled blind by two annotators, κ 0.91, `reports/pythia/v2_mech/b3a_annotation.md`); MLP11.n4925 is "other" under the declared rule (descriptively, prepositions with a missing object, plus object gaps 0.30–0.35). Shares of the half-B MLP11+14 switch: 0.20 / 0.14 / 0.11 / 0.05 / 0.05 (v's run). Per-neuron causal shares of the edit's effect and of the natural gap: running (phase 1b).
- **Main if it passes, declared:** the gap reader's response to verb class appears only when a filler is present ("I know *what* John V" vs "I know *that* John V"). Links §5 to the filler-gap paragraph of §3. The comparison with the passive detectors is descriptive only.
- **Main, running (phase 1b, `reports/v2_controls_mech_b_rules.md`):** clamp sweep over attention and MLP outputs at layers 8–23 (where the other ~three quarters of the effect goes; the MLP11+14 clamp must reproduce 0.26); natural mediation (the set swapped between the natural T and I sentences, with controls matched on the natural swap and a same-class swap); v2-coordinate swap in the natural pairs with the set held fixed, which links the model's own behavior to v2 and the neurons.

- **Main (one figure), all v2, half-B pairs (46; selection on 43):** 2,986 passive-conjunction neurons in MLPs 11 / 14 (sign agreement 0.99) carry **0.62 [0.58, 0.66] of the MLP11+14 *by* switch** (N1, N2, "at least a quarter") → **mediation:** clamping them to their I-run values removes **0.19 [0.15, 0.23] (0.15 nats) of the edit's T − I effect on *by*** against 0.01 (activation-matched), 0.03 (direct-effect-matched) and 0.02 (random); the converse restores 0.28 [0.23, 0.33] (M1, M2, both beyond de) → all of MLPs 11 + 14 mediate 0.26, so P is 0.73 of what the two layers carry, and most of the effect runs elsewhere.
- **One sentence each:** u_md on the frozen set: switch share 0.70, mediated 0.19, restored 0.26 (U1, U2, M1 / M2 hold). New and lemma-fresh half-B pairs: m(P) 0.22 / 0.23. MLP 14 alone 0.14, MLP 11 alone 0.06.
- **Denominators in the figure:** 0.62 is a share of the MLP11+14 direct-effect switch; 0.19 is a share of the edit's whole T − I effect on *by*. Neither is a share of natural behavior. Never write "carries most of the effect".
- **Exploratory, labelled:** top 10 neurons (half-A ranking) mediate 0.23, more than the full set (the rest partly cancels); top 100 0.28. Shuffled-label direction: the set responds at 0.56–0.60 of v2, in proportion to what that direction itself transmits (pc_frac 0.45 vs v2 0.80 in v2's final). Report it as proportional, not as a specificity control.
- **v's set, now covered:** the wh-gap reading (R5-C9) is replaced by the declared filler test above; Pile-text labels (R5-B7) carry over because the top 5 are the same neurons and the labels do not depend on the instrument; natural necessity is replaced by the natural mediation run. Heads (R3-B7, R4-D10) deferred: at most one labelled appendix sentence on v. No serial head → neuron → *by* claim.
- **Appendix:** dose (R2-dose), persistence (R3-A2), *got* frames (R3-B6), previous-token check (R4-A5), v / v2 set overlap (1,918 shared, Jaccard 0.50), control distributions and ranks (P first in all 60 control sets, both instruments).

### C5. Rare verbs: the class representation holds across frequency; the deficit is in predicting the verb (§6, 0.75 page; revised Nov 26)

Two steps, in this order. In the text, name directions by their training set ("trained on frequent verbs only", "trained on rare verbs only"), not v / v2; v2 (mixed frequency: 29 frequent + 44 rare pairs) stays the instrument of §3–5.

**Step 1: the verb-site class representation does not depend on frequency.** Transfer works both ways between directions trained at opposite ends of the frequency range:
- **Frequent → rare.** The stored v, trained on frequent verbs only (8 Head + 21 near-Head pairs), transfers to the new, mostly rare plain verbs as strongly as to the originals (Pythia D_new / D_orig 1.01 at sites 8 and 12, u_md 0.96 / 0.98; Gemma / OLMo 7 of 8 declared readings, Settled 20), and a rare verb's class can be read from its own site-8 state along v (R6-C4: 31 / 32 rare singletons; D11 Tail 44 / 44).
- **Rare → frequent.** A direction trained only on 42 rare plain pairs transfers to frequent verbs' passive *by* at 1.00 [0.89, 1.14] / 0.94 [0.85, 1.04] of v (sites 8 / 12; both sides rare 1.01 / 0.92; Settled 15). Disclose beside it: |cos| with v 0.80–0.83, and active IIA on frequent verbs is lower (0.63 vs 0.78 at 8). Pythia only; scope it so unless ported.
- **Supporting, one clause:** the verb-site passive gap barely tracks frequency (R2-freq / R3-A1: slope +0.065 per Zipf, split-half R ≥ 0.985).
- **Not claimed yet:** that rare- and frequent-trained directions differ no more than two random verb samples (the disjoint-sample curve, `reports/v2/instrument_measures.md` §2, gives 0.80 at n = 20 and 0.85 at n = 30, Pythia 8). The sizes are not matched (42 rare pairs); claim it only with a size-matched frequent-only vs rare-only comparison.

**Step 2: the deficit shows mainly when predicting the verb.** After reading a rare verb the model has its class (step 1). When predicting that verb after "was", it uses the participle's output row, and rare intransitives' rows sit too close to transitives' along the was − has direction (R5-A1 / A2). The edit shows the class is usable: read it from the verb's own state, write it into the output row, and the prediction is repaired.
- **Main:** the deficit is on the intransitive side at every size from 160M to 12B (R5-A1, R5-D10b, one sentence). **Classify-from-z edit** (R6-C4): moving each rare participle's output row to the Head mean of the class read from z raises affected passive margins by +0.53 [0.24, 0.83] nats, equal to the oracle, with Head pairs, unaffected items and Pile loss unchanged. Label-free (no class) +0.25 [−0.08, 0.58], unresolved.
- **Wording:** "mainly on the output side", never "only" or "elsewhere": R5-A5 (a sentence-final probe also shows a rarity drop).
- **Limits, in the text:** 12 affected Tail single-token pairs; the classification was seen before it was declared; not a matched localization with E13 (which intervenes after the participle). Gemma ties input and output rows (Settled 4), so the output-side asymmetry is a Pythia statement unless ported.
- **Appendix:** R5-A2 decomposition, R5-A4 oracle conditions, R4-E13 / E13T, R2-freq, R3-A1, R4-D12.
- **Dynamics (one paragraph, §6 gets 1.0 page for it; frequency as training exposure, not bands, Nov 26):** exposure E = the verb's Pile count (c_total) × step / 143,000, as in R7-A8; exact counts from Pythia's public batch order as a check at the early steps if cheap. Measure: Figure 0 over training, each verb's position along u_md recomputed per checkpoint (cross-fitted), on the same I = 0 / T = 1 scale, against its own exposure. Behavior side is done in these units (R7-A8: rare intransitives start rejecting the passive at lower own exposure, median log10 E at onset 3.57 / 3.07 / 2.65 for Head / Tail / XTail).
  - Candidate primary (to declare): **class before use, per verb**: the verb-site class onset is at or below the same verb's behavioral onset in own exposure (paired; R7-A8's onset definition with persistence).
  - Candidate secondary: **shared class**: like behavior, rarer verbs reach the class at lower own exposure (slope in log exposure between −1 and 0), which ties §6 to §4.
  - Dropped: "same checkpoint for rare and frequent" (rare verbs have orders of magnitude fewer occurrences by steps 256–1000).
  - If it does not land by the freeze, §6 goes back to 0.75 page and the page returns to §4.

## Figures

0. Verb positions along the class direction (T / I, frequency bands); compact, single column.
1. Transfer and context: passive *by* D / G depth curves (v2, three models, rule sites marked, staging regimes shaded), OR / TC against natural. Nonce in the text, no panel.
2. Erasure: cross-fitted vs in-sample vs permutation (matched inventory), learning curve (rank-k in the appendix).
3. Pythia readers (v2): core neurons with their labels and causal shares, then the set's switch share, mediated and restored shares against matched controls, with MLP 11 + 14 total beside them; u_md bars.
4. Rare verbs: transfer both ways (frequent-trained → rare, rare-trained → frequent), then classify-from-z (plus a checkpoint panel if dynamics lands).

No separate summary table if it repeats the figures; the pass / fail table against declared rules is in the appendix.

## Still to run, by what it buys (17 days to freeze)

| Priority | Experiment | Buys | Cost |
|---|---|---|---|
| 1 | Expanded-set in-sample LEACE + 20-permutation control, 3 models | Matched references for the C3 headline | LEACE only |
| 1 | Learning curve, fixed eval population, include vs exclude at equal size, 3 models | Whether the in-sample advantage is fit size | LEACE only |
| 4 | Rank-k erasure with refitting and controls, Pythia first (appendix only) | Constrains "shared in more directions" | LEACE only |
| 1 | Verb positions along v2 / u_md on the expanded verbs, 3 models | Figure 0 | forward passes |
| 2 | Graded check: within-class rank correlation of position with the model's own active object expectation (declare first) | The "what information" answer | forward passes |
| 1 | C4 phase 1b (running): clamp sweep, natural mediation, v2-coordinate swap, core-neuron shares, filler test | §5 story and where the rest of the effect goes | running |
| 1 | Shuffled-label direction beside the random controls in C1 (Pythia has it in v2's final; Gemma / OLMo with the illusion check) | C1 controls include a trained-on-noise direction, not only random ones | stored / small |
| 3 | Classify-from-z edit on more verbs: D11 Tail singletons and the new single-token rare pairs, z from v, P and u_md, frozen θ from Head | Takes §6 beyond 12 affected pairs; shows the edit does not need the stored v | forward passes |
| 4 | Pythia-6.9B: natural gaps on the paper's items and u_md transfer; LEACE only with its controls | Scale | a few GPU-hours |
| 2 | Dynamics in exposure units: per-checkpoint u_md positions per verb against own exposure; declared "class before use" (primary) and "shared class" (secondary); Pythia, R7-A8's 16 checkpoints | The paragraph in §6 (sensei asked) | a few GPU-hours |
| 1 | Pile counts for the remaining verbs (infini-gram) | Count axis for Figure 0 and §6 | API queries |
| — | Second-model conjunction neurons | Optional; not a substitute for the mediation test | ~1 GPU-hour + porting |

## Cut from this paper

- **Replacing v: decided (Nov 24), v2 replaces v** in the core chain of all three models (`reports/v2_retrain_rules.md`; declared IIA gate passed at every rule site, `reports/v2/instrument_measures.md`). The Nov 24 threshold readings (Gemma T1 failing by 0.02 nats, T3 by the "not above T" ceiling) are superseded by the audited criteria (`reports/criteria_audit.md`), under which *by* is read on the natural-gap scale and T3 by the frame interaction. The Pythia site-8 mechanism tail stays on v unless ported (decide by Dec 4).
- **Split edit** moves to the appendix (Nov 25); it no longer serves as the main-text illusion defense.
- DAS at 6.9B and at checkpoints; alternating verbs; SAE lookup; further direction-check suites; staging as a headline claim; embedded wh beyond one limitation sentence.
