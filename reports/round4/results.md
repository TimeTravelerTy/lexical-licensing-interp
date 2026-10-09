# Round 4: results

Plan, predictions and decision rules: `plan.md`, each part committed before
its runs (A: `47cdf9c`, fixes `260de13`; B: `801296a`; C: `74b2e41`; D and
E: `d27bf7c`).
Generated reports, one per experiment:
- A: `a1_a3_dose.md`, `a2_inrange.md`, `a4_bylever.md`, `a5_aux_heads.md`;
- B: `b_fillergap_gates.md`, `b_fillergap.md`;
- C: `c_nonce_cues.md`;
- D: `d10_voice.md`, `d11_conjunction.md`, `d12_translation_gain.md`;
- E: `e13_oracle.md`, `e13_deficit.md` (descriptive).

TSUBAME jobs: 8947213 (A2), 8947214 (A4), 8947215 (A5), 8947396 (B stage
1), 8947442 (B stage 2), 8947454 (C), 8947738 (D10–D12, E13). A1 and A3
ran locally on committed round-3 outputs.

## Summary

- **A, confirm current claims.**
  - Moved within each construction's own range, the early coordinate
    reproduces the object relative's and the tough construction's natural
    profile (A2). In passives it reaches about half of the natural " by"
    gap, and an excess of object log-probability remains.
  - The excess left by the T − I donors is not explained by how far they
    overshoot each construction's range (A1).
  - The passive-trained direction's " by" effect in actives is about half a
    generic "by lever" that label-free training also finds (A4).
  - The auxiliary heads carry the switch across an adverb and with "got",
    but the mid heads then read the adverb, not the auxiliary (A5).
- **B, filler-gap.** Both natural gates pass. Moved within range at sites
  4–10, the d coordinate gives the licensing-like crossover:
  - in matrix questions at 4, 6, 8 and 10;
  - in embedded questions at 8 and 10.
  The T − I donors overshoot and read surface or mixed. From site 12 on,
  every patch reads surface ("object next").
- **C, nonce cues.** A transitive context still moves a nonce participle
  along d when it uses a different inflection, while the object rise largely
  disappears. But the cue that moves d is a following noun phrase, not the
  verb having an object argument.
- **D, translation stage.**
  - The B7 switch is a voice switch. "has been V" vs "has V" reproduces it
    in all nine MLPs, with the same routing and the same heads (D10).
  - The mid-layer switch neurons exist and carry about two-thirds of the
    " by" switch in MLPs 11 and 14 (D11). Thousands pass the test, but ten
    neurons carry most of the effect.
  - Across pairs, what MLPs 11–17 write onto " by" (TG) does not track verb
    frequency. TG predicts the released *by* margin (r 0.57), and the d gap
    does not (r 0.02) (D12).
- **E, restoration.** Setting a rare verb's d coordinate to the
  frequent-verb class mean raises its " by" margin by only 0.05 nats. The
  same operation lowers frequent verbs' margins, so E13 does not close a real
  share and E14 is not run. The natural Head − XTail " by" deficit is itself
  not reliable, even on all 126 curated pairs.

# Part A: confirm current claims

## A1. Dose check for C9

Site 8, 64 primary pairs, pair bootstrap (paired across constructions).
The construction-level point values were already in round 3; A1 adds
intervals and paired tests.

| Construction | natural z: good / bad | ratio to the active gap | upper overshoot u = z_T − z_good | D(O) | natural O gap | excess E |
|---|---|---|---|---|---|---|
| passive | 0.45 / 0.09 | 0.42 [0.39, 0.45] | +0.55 [+0.53, +0.58] | +0.36 | −0.22 [−0.47, +0.05] | +0.58 [+0.33, +0.81] |
| TC | 0.60 / 0.06 | 0.62 [0.59, 0.65] | +0.40 [+0.37, +0.43] | +0.92 | +0.56 | +0.36 [+0.09, +0.63] |
| OR | 0.77 / 0.04 | 0.85 [0.82, 0.87] | +0.23 [+0.19, +0.27] | +1.41 | +1.49 | −0.08 [−0.33, +0.17] |

Paired: ratio passive − TC −0.20 [−0.24, −0.16], TC − OR −0.23 [−0.26,
−0.19]; u passive − TC +0.15 [+0.13, +0.17], TC − OR +0.17 [+0.15, +0.20];
E passive − TC +0.22 [−0.12, +0.54], TC − OR +0.44 [+0.18, +0.73]. The
sensitivity with contexts and donors resampled gives the same decisions.

**Declared decision:** the dose differs (ratio ordered) and the overshoot
is ordered, but the excess is not fully ordered (passive − TC includes 0):
**overshoot differs but does not explain the excess**.

**Against the predictions.** Ratios OR 0.8–0.9 (0.85), TC 0.55–0.7 (0.62),
passive ≈ 0.4 (0.42): **held**. E_passive > E_OR and E_TC > E_OR exclude
0: **held**; E_passive > E_TC may not: **as anticipated**, it does not.

## A2. In-range patch

Bad items; the coordinate is set to the construction's own transitive (t_T)
or intransitive (t_I) class mean in the same context. D_in = R(t_T) −
R(t_I), ratio = D_in / natural good − bad gap.

| Construction, site | " by" or MAIN / END | PREP | O | Decision |
|---|---|---|---|---|
| passive, 6 | by 0.52 [0.36, 0.97] | — | D_in +0.06, natural −0.22 (CI includes 0) | **reproduces** (only " by" decisive) |
| passive, 8 | by 0.44 [0.30, 0.82] | — | D_in +0.09; excess +0.31 [+0.07, +0.56] | **does not reproduce**; excess unresolved |
| OR, 6 | MAIN 0.93 | 0.72 | 0.67 | **reproduces** |
| OR, 8 | MAIN 0.89 | 0.67 | 0.76 | **reproduces** |
| TC, 6 | END 1.10 | 0.67 | 0.49 | **reproduces**; excess removed (−0.29 [−0.54, −0.03]) |
| TC, 8 | END 0.91 | 0.58 | 1.01 | **reproduces**; excess unresolved (+0.00 [−0.27, +0.28]) |

From site 12 on, even an in-range move pushes O past the natural gap (OR:
1.28–1.52×, TC: 2.0–2.5×) while MAIN / END collapse: the late coordinate
acts as "object next" whatever the range.

**Against the predictions.** Passive " by" ≈ 0.45 at site 8: **held**
(0.44); passive does not reproduce at 6: **missed narrowly** (0.52 passes
the 0.5 bar). OR reproduces: **held**. TC borderline: **better than
predicted** (reproduces at both sites). Excess removed in TC: **held at 6**,
unresolved at 8. Excess remains in passives: the point estimate is +0.31
with CI above 0, but the declared bound (lower > 0.2) is not met.

**Reading.** Within range, the early coordinate produces each gap
construction's own profile with the right signs and 0.5–1.1 of its size.
For passives it gives about half of the natural " by" gap and some object
log-probability that natural transitive passives do not show, so the
natural passive profile is not carried by d alone.

## A3. OR verb forms

One primary pair has simple past ≠ participle (tail/outdo/recede). Without
it, every site-8 OR estimate moves by ≤ 0.045 and the reading stays
"mixed": **unaffected** (predicted).

## A4. The B5 by lever

| Site | shuffled direction: by in actives (arm 1) | d_p: by | d_p⊥s: by (ratio) | d_p⊥s: O arm 1 / arm 3 | Decision |
|---:|---|---|---|---|---|
| 4 | +0.14 (below 0.2) | +0.82 | +0.39 (0.48) | +0.61 RISE / +0.50 RISE | shuffled does not raise by |
| 6 | +0.27 | +1.28 | +0.59 (0.46) | +0.55 RISE / +0.45 unresolved | **mixed** |
| 8 | +0.36 | +1.73 | +0.87 (0.50) | +0.59 RISE / +0.37 NO RISE | **mixed** |
| 12 | +0.52 | +2.87 | +1.57 (0.55) | +0.38 / +0.25 (no RISE under d_p) | mixed (dose-preserving control: separable) |
| 17 | +1.00 | +3.34 | +2.34 (0.70) | −0.01 / +0.02 | mixed |

|cos(d_p, s)| = 0.50–0.55 at every site; |cos(d_a, s)| = 0.03–0.23. The
shuffled direction never raises objects (NO RISE everywhere). All
effects lie far beyond their norm-matched nulls.

**Against the predictions.** Shuffled raises by at every site (≥ 0.3):
**partly** (sites 6–17; ≥ 0.3 from site 8). by falls ≥ 50% after
projection: **held at 4–8** (ratios 0.46–0.50; 0.38–0.42 for the
dose-preserving control), **missed at 10–17** (0.52–0.70). O RISE at 4–6
stays: **held in arm 1**; in arm 3 the O effect drops below the bound at 6–8
although the paired reduction is only 0.14–0.18 nats. So the declared
reading is **mixed**, not "separable".

**Reading.** About half of d_p's early " by" effect in actives is a generic
lever that training on the *by* readout finds without verb-class labels.
The other half, and the object transfer, lie in the class-specific part.
Late, the generic share is smaller.

## A5. Auxiliary heads

**(a)** Only 2 of 384 heads are previous-token heads (L1H11, L4H15). None
of the B7 top five is (L8H4 is the most local: 0.80 on random tokens, 0.46
on text).

**(b)**

| Frame | Gate | H1 | B7 heads retain their role | Attention to the auxiliary (L9H7, L10H2, L3H3, L1H6, L8H4) | Read the auxiliary across the adverb |
|---|---|---|---|---|---|
| got / has | passes | holds | **holds** (4 of the B7 five in the new top 10) | 0.44, 0.21, 0.74, 0.69, 0.53 | — |
| was quickly / has quickly | passes | holds | **holds** (all five in the new top 10) | 0.15, 0.21, 0.64, 0.33, 0.09 | **fails**: L9H7, L8H4, L10H2 attend to the adverb (0.31, 0.40, 0.23) |

The plain frame reproduces B7 to 1e-6 (code check).

**Against the predictions.** At most one previous-token head among the top
five: **held** (none). got: **held**. Adverb: same heads, **held**; they
read the auxiliary across it: **missed**. The early heads (L3H3, L1H6) keep
reading the auxiliary, and the mid heads move to the adverb, the opposite of
my guess.

**Reading.** The switch survives an intervening adverb and works with
"got". With the adverb, the mid heads that carry most of it read the adverb
position, whose residual has itself read the auxiliary. The switch is
therefore not tied to a fixed auxiliary position; it can arrive second-hand.

# Part B: filler-gap

## Natural gates (stage 1)

| | I verbs kept (half A) | interaction on L (half B) | interaction on O (half B) | Gate |
|---|---|---|---|---|
| B6 embedded ("I know that / what John V-ed") | 39 of 50 | +1.55 [+1.13, +1.97] | +0.63 [+0.33, +0.93] | **passes** |
| B7 matrix ("Did John V" / "What did John V") | 45 of 50 | +1.45 [+1.04, +1.88] | +0.96 [+0.68, +1.24] | **passes** |

With all 50 I verbs the interactions are similar (L +1.2, O +0.8–1.0), and
every matrix verb / auxiliary stratum passes on its own. The prepositions
the filler raises after I verbs are " for", " on", " about", " to", " over".
The B7 base-form gate passes at every site (AUC 1.00). Naturally, the T
verbs' coordinate on d drops with the filler only a little early (site 4:
0.99 → 0.89) but by half late (site 17: 1.03 → 0.44).

## Patches (stage 2; kept I verbs as bases; half-B contexts)

| Experiment | Patch | 4 | 6 | 8 | 10 | 12–17 |
|---|---|---|---|---|---|---|
| B6 embedded | in range | mixed | mixed | **licensing-like** | **licensing-like** | surface |
| B6 embedded | T − I donors | mixed | surface | surface | surface | surface |
| B7 matrix | in range | **licensing-like** | **licensing-like** | **licensing-like** | **licensing-like** | surface |
| B7 matrix | T − I donors | mixed | mixed | mixed | surface | surface |

In range, at site 8:
- **"that" / no-wh frame:** "." falls and O rises, as for a transitive verb.
- **"what" frame:** "." rises (B6 +1.01, B7 +1.19), PREP falls (−0.58, −0.62)
  and L rises.
- **O against the natural T level:** O rises too (+2.33, +2.25), but stays
  at that level (+0.19 [−0.10, +0.51], +0.09 [−0.24, +0.42]).

The T − I donors push O above the natural T level in the what frame (B6:
+0.51 [+0.20, +0.82]); this is the overshoot of A1 / A2.

**Sensitivities (sites 6, 8).** The in-range crossover at site 8 holds on
all contexts. It also holds in the matrix "Did", "Will" and "Would" frames
and the embedded "forgot" and "remember" frames. It does not hold in "Can"
(surface) or "know" / "heard" (mixed). It weakens to "mixed" with non-DAS
verbs only (B6; B7 keeps site 8) and with multi-token verbs only.

**Patch × frame interaction on O.**
- Matrix, in range: smaller with the filler (site 8: −0.26 [−0.49, −0.05]).
- Embedded: larger with the filler (+0.23 [+0.01, +0.46]), which a
  licensing variable should not show.
- Log-probability saturation differs between frames (I verbs start at O
  −4.5 with the filler vs −3.3 without), so this interaction is hard to
  read.

**Against the predictions.**
- Natural gates pass: **held**. I gate keeps ≥ 70%: **held** (78%, 90%).
- In range at 6–8 licensing-like: **held** for B7 (6 and 8) and for B6 at
  8; B6 site 6 is mixed.
- T − I donors licensing-like or mixed at 6–8: **held** for B7 (mixed);
  **missed** for B6 (surface, because O lands above the T level).
- Surface at 14–17: **held**.

**Reading.**
- **Early (sites 4–10):** moving only the d coordinate, within the natural
  range, makes an intransitive verb behave like a transitive one in both
  frames. Without a filler it now wants an object. With a filler, closure
  rises and stranded prepositions fall, and objects rise only to the level
  natural transitive verbs show there. This is the filler-sensitive pattern
  licensing predicts.
- **Late (12–17):** the same move makes "object next" regardless of the
  filler.
- **Caveat:** as declared, the crossover cannot separate a licensing
  variable from a lexical class feeding a learned, frame-sensitive
  continuation policy.
- **Fragility:** at sites 6–8 the result depends on the stratum and is
  weaker for non-DAS and multi-token verbs.

# Part C: nonce cues

Passive probe "The N was dakked", 80 nonce lemmas, per-lemma means.

## C8. Inflection mismatch

| Context form | z, site 6 | z, site 8 | " by" | O |
|---|---|---|---|---|
| ed (exact token; = round-3 C8) | +0.129 | +0.151 | +1.44 | +2.14 |
| ing ("is dakking the plate") | +0.091 | +0.087 | +0.60 | +0.99 |
| s ("daks the plate") | +0.056 | +0.062 | +0.59 | +0.70 |
| ing, other lemma (mismatched) | +0.019 | +0.012 | +0.29 | +0.64 |
| s, other lemma (mismatched) | +0.008 | −0.001 | +0.44 | +0.22 |

No lemma had the probe's final token in an ing or s context. **Declared
decisions (both forms, sites 6 and 8):** d shift survives, " by" survives,
the object rise mostly disappears (≤ half the ed value, paired difference
above 0), and the balanced (verb-specific) d and " by" effects survive.

**Against the predictions.** All **held**. The " by" rise is partly generic
(mismatched-lemma contexts give +0.3 to +0.4); the d shift is not
(≤ 0.02 z).

## C9. Which cue drives the shift

| Contrast | z, site 6 | z, site 8 | " by" | O |
|---|---|---|---|---|
| transitive + NP − bare ("dakked the plate") | +0.129 | +0.151 | +1.44 | +2.14 |
| transitive, no adjacent NP: relative T − I | +0.020 | +0.013 | +0.20 | +0.27 |
| transitive, no adjacent NP: question T − I | +0.008 | +0.016 | +0.17 | −0.09 |
| adjacent NP, intransitive: "the whole night" − "throughout the whole night" | +0.167 | +0.154 | +2.00 | +1.84 |
| adjacent NP − bare | +0.124 | +0.142 | +1.14 | +3.03 |

**Declared decisions.** d at sites 6 and 8: **NP-driven** (primary and
secondary contrasts). " by": **both**. O: **both** (primary), NP-driven
(secondary).

**Against the predictions.** d shift transitivity-driven: **missed**, it
is NP-driven. The object rise NP-driven: **held**.

**Reading.** The in-context signal that moves a nonce participle toward the
transitive side of d is a noun phrase right after the verb. "Dakked the
whole night" moves it as much as "dakked the plate", and the same duration
as a PP moves it not at all. A transitive use whose object has been
extracted (relative clause, question) barely moves it. So the early
coordinate is learned in context from surface adjacency, not from argument
structure. The declared caveat applies: an unknown verb followed by "the
whole night" may be read as transitive, so "NP-driven" may partly reflect
inferred transitivity. But the extracted-object contexts, which are
unambiguously transitive, do not move d.

# Part D: translation stage

## D10. Clean voice contrast

"The N has been V" (base) vs "The N has V" (counterfactual); B7 items,
rows and site-8 interchange; fp32. Exactness ≤ 3.5e-6 for every MLP.

| MLP | readout | S = has − has been (T donors) | B7 S | S_D (T − I donors) | S_sens (was − has been) | PE heads | PE earlier MLPs |
|---|---|---|---:|---|---:|---|---|
| 11 | by | −0.41 [−0.44, −0.38] | −0.41 | −0.42 | +0.005 | −0.39 | +0.11 |
| 13 | by | −0.26 [−0.30, −0.22] | −0.27 | −0.28 | +0.012 | −0.27 | +0.04 |
| 14 | by | −0.62 [−0.68, −0.57] | −0.62 | −0.65 | −0.007 | −0.46 | −0.00 |
| 16 | by | −0.46 [−0.50, −0.42] | −0.47 | −0.48 | +0.011 | −0.28 | −0.07 |
| 17 | by | −0.44 [−0.49, −0.39] | −0.44 | −0.45 | +0.008 | −0.35 | −0.02 |
| 15 | O | +0.24 [+0.23, +0.26] | +0.26 | +0.25 | −0.012 | −0.02 | +0.27 |
| 18 | O | +0.25 [+0.22, +0.27] | +0.28 | +0.24 | −0.030 | +0.03 | +0.21 |
| 21 | O | +0.17 [+0.16, +0.19] | +0.19 | +0.18 | −0.015 | −0.01 | +0.20 |
| 22 | O | +0.17 [+0.15, +0.20] | +0.16 | +0.18 | +0.012 | −0.03 | +0.22 |

**Declared decisions.** Gate 9/9: **passes**. Voice switch 9/9: **holds**.
Not an auxiliary-token effect 9/9: **holds**. Two-stage routing (heads
into all five " by" MLPs, earlier MLPs into all four object MLPs):
**replicates**. B7 heads retain their role: **holds**. The new top 5 are
L9H7, L3H3, L10H2, L8H4 and L3H5, so four of the five B7 heads return.

**Against the predictions.** All five **held**.

**Reading.** The switch does not depend on "was". Inserting "been" after
"has" gives the same MLP changes as replacing "has" with "was" (S vs B7 S
within 0.03 everywhere), and "was" vs "has been" moves the MLPs by ≤ 0.03.
I donors move nothing (|S_I| ≤ 0.03), so S_D ≈ S_T. From the participle's
last token, the top heads attend to "been" much more than to "has". For
example, L3H3 puts 0.76 of its attention on "been" and 0.02 on "has"; L1H6
puts 0.44 vs 0.21. So the heads read the passive auxiliary, whichever it
is.

## D11. Conjunction neurons in MLPs 11 and 14

I_n = (was T − I) − (has T − I) of each neuron's post-activation. Pairs
split within band: half A (31 pairs) selects, half B (33) tests.

- **Selected (half A, BH q < 0.05):** 9,344 of 16,384 neurons (MLP11
  4,742, MLP14 4,602). Categories: passive-conjunction 2,782 (1,415 in
  MLP11, 1,367 in MLP14), active-conjunction 4,083, graded 1,362, opposite
  signs 1,117.
- **Passive-conjunction set on half B:** sign agreement 0.99. Aligned I_n
  +0.036 [+0.034, +0.039], was T − I +0.041 [+0.039, +0.044], has T − I
  +0.005 [+0.004, +0.006].
- **Share of the MLP 11 + 14 " by" switch** (half B): 0.65 [0.60, 0.70].
  Share of the passive T − I effect: 1.22 [1.13, 1.32]; the other neurons
  partly cancel.
- **Generalization** (sign-aligned mean activation difference of the set):
  - natural good − bad passives: Head +0.033, Tail +0.028, XTail +0.028.
    XTail / Head is 0.84 [0.64, 1.21].
  - nonce probes: matched T − I +0.012 [+0.011, +0.013], balanced AB − BA
    +0.005 [+0.004, +0.006].

**Declared decisions.** Conjunction neurons exist: **yes**. They carry the
" by" switch: **yes**.

**Against the predictions.** They exist and carry ≥ 25%: **held** (65%).
"Tens of them, mostly in MLP14": **missed** as a count. Thousands pass,
split evenly between the two MLPs, though by effect MLP14 carries 0.60 of
the switch and MLP11 0.40. "Weaker for XTail (ratio 0.5–0.8)": **missed**;
the ratio is 0.84 and its CI includes 1. The nonce matched contrast:
**held**.

**Exploratory (not declared).**
- *Concentration.* Rank the passive-conjunction neurons by their half-A
  direct effect on the switch. On half B, the top 10 carry 0.66 of the whole
  MLP 11 + 14 switch, the top 50 0.86 and the top 100 0.95. Three neurons
  carry 0.44 between them: MLP11.n3462 (0.20), MLP14.n2114 (0.14) and
  MLP14.n5407 (0.11). Statistical selection finds thousands of neurons with
  a reliable interaction, but the " by" effect sits in tens of them, as
  predicted in spirit.
- *Rare verbs drive the switch neurons less.* Weight each neuron's natural
  good − bad activation difference by its " by" output weight. For the top
  10, this is Head +1.33 [+0.39, +2.66], Tail +1.23 [+0.68, +1.82] and XTail
  +0.35 [−0.19, +0.86]. For the top 50 it is +1.95, +1.70 and +0.85 (XTail /
  Head 0.43 [0.14, 1.03]). Under the patch, the same neurons respond equally
  in every band (top 10: 2.08, 2.24, 2.35). The whole of MLP 11 + 14 writes
  *more* good − bad " by" for XTail (unscaled: Head 0.68, Tail 1.51, XTail
  2.15). So rare verbs get their " by" preference in these MLPs through
  other neurons. The test is post hoc and the Head band has 9 pairs; Tail
  vs XTail (28 vs 27 pairs) is the better-powered contrast.
- *Gating.* Most neurons have negative mean post-activation in both frames
  (85% of all neurons). The passive-conjunction set is slightly more often
  negative in the has frame (89%) than in the was frame (77%). So part of
  the conjunction is the frame moving neurons across the GELU threshold.
  Pre-activations were not saved, so this is not separated further.

## D12. Translation gain per pair

TG = the direct effect of MLPs 11–17 on the centered " by" logit, natural
good − bad. 64 primary pairs. Means: TG +0.95 (MLP14 +0.55, MLP16 and 17
+0.13 each, MLP15 +0.10); z gap +0.36; gain +3.4. Split-half reliability
≥ 0.996 for every measure. Across pairs, TG does not rise with the d gap
(OLS β = −0.22). By band, TG is Head +1.04, Tail +0.64, XTail +1.03.

| Measure | slope on participle Zipf |
|---|---|
| TG | −0.17 [−0.80, +0.40] |
| gain | −0.81 [−2.25, +0.69] |
| TG_res | −0.15 [−0.75, +0.42] |
| " by" logit gap (all components) | +0.31 [−0.25, +0.93] |
| z gap (site 8) | +0.065 [+0.039, +0.101] |

| Behaviour | r(TG) | r(z gap) | r(TG) − r(z gap) | partial r(TG \| z gap) |
|---|---:|---:|---|---|
| single-prompt " by" preference (part–whole) | +0.65 | +0.10 | +0.54 [+0.25, +0.84] | +0.65 [+0.48, +0.79] |
| released *by* margin | +0.57 | +0.02 | +0.55 [+0.20, +0.88] | +0.57 [+0.37, +0.73] |
| curated LP margin | +0.17 | +0.04 | +0.13 [−0.31, +0.55] | +0.18 [−0.11, +0.46] |

**Declared decisions.** TG tracks frequency: **no**. Frequency dependence
beyond the d gap: **no** (Zipf coefficient −0.18 [−0.80, +0.46]). TG
predicts behaviour beyond the d gap: **yes, for the released *by*
margin**. Reading: no evidence of a frequency dependence of translation
beyond the d gap (not evidence that there is none).

**Against the predictions.**
- TG tracks frequency: **missed**. Only the d gap does.
- TG_res has no frequency slope: **held**.
- r(TG, " by" preference) 0.3–0.5: **exceeded** (0.65; part–whole).
- TG is weak with the LP margin: **held** (0.17).

**Reading.** The d gap tracks frequency but predicts no behaviour, while
TG predicts the *by* behaviours and does not track frequency. Within the
primary pairs, which verbs prefer " by" in natural text is set by what the
translation stage writes, and that is mostly not the d gap. D12 is
observational: TG is natural good − bad, so it includes every way the two
verbs differ.

# Part E: restoration

## E13. Oracle at d

Curated `passive_1` and `passive_2` sentences of the primary pairs with
curated scores (9 Head, 25 Tail, 25 XTail). The participle's d coordinate
is set to the Head mean of its true class (own pair and the basis's DAS
training pairs excluded; at least 18 target pairs). The natural scores
reproduce `pythia14b_scores.csv` (max |difference| 6e-6).

| Site | XTail Δ by | share of the by deficit | Head control Δ by | reverse (Head → XTail mean) Δ by | XTail Δ whole (passive_1) |
|---|---|---|---|---|---|
| 6 | +0.058 [+0.006, +0.114] | 0.11 [−0.95, +0.93] | −0.104 [−0.183, −0.040] | −0.196 [−0.269, −0.139] | +0.046 [−0.011, +0.106] |
| 8 | +0.045 [+0.003, +0.086] | 0.09 [−0.63, +0.66] | −0.175 [−0.286, −0.089] | −0.262 [−0.363, −0.187] | +0.043 [+0.001, +0.083] |

Natural " by" margin: Head +0.95 [−0.48, +2.17], XTail +0.42. The deficit
is +0.53 [−0.96, +1.84]. `passive_2` (the "." after the participle) moves
by ≤ 0.05 in every group.

**Declared decision.** Neither site closes a real share. Each fails the
deficit CI, the share CI and the Head control (outside ±0.1). **E14 is not
run**; per the plan, E13 is repeated at the translation stage (E13-T,
declared in `plan.md` before its run).

**Against the predictions.**
- " by" share 0.2–0.3: **missed** (0.09–0.11, uninformative CI).
- Whole-margin share ≤ 0.1: **held** (0.04).
- Head control moves < 0.1: **missed** (−0.10 and −0.18).
- Reverse control lowers the Head " by" margin: **held**.

**The deficit itself (descriptive, existing natural scores, no model; `e13_deficit.md`).**
With only 9 Head pairs, the E13 deficit is too noisy to test against. On
all 126 curated pairs (26 Head, 50 Tail, 50 XTail; pair-within-band ×
context bootstrap), the " by" margin is:
- Head +0.94 [+0.22, +1.66], Tail +0.96 [+0.60, +1.34], XTail +0.49 [+0.10,
  +0.82];
- Head − XTail +0.45 [−0.33, +1.27]; Tail − XTail +0.48 [−0.05, +1.02];
- released contexts: Head − XTail +0.63 [−0.21, +1.50], Tail − XTail +0.59
  [+0.06, +1.12];
- slope on the good participle's Zipf +0.16 [−0.18, +0.50] (curated).

So the part of the rare-verb deficit that a patch at the participle can
reach (the continuation) is at most marginal. Most of the whole-sentence
LP deficit is the participle's own log-probability, which the model
predicts before the patched position.

**Reading.** Moving a rare verb to the frequent-verb class mean on d
barely changes its " by" margin. The same move lowers frequent verbs'
margins, and under a common linear response to the coordinate the
leave-one-out Head control would sit near 0. So the response at d is not a
common function of the class value: each verb's own coordinate (or how
the rest of its representation interacts with it) matters. With D12 (the
d gap predicts no behaviour) and D11 (rare verbs drive the switch neurons
less, exploratory), the evidence points away from a weak early class value
as the reason rare verbs underperform.
