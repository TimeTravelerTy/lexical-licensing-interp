# Results index, rounds 1–5 (for paper planning)

One row per experiment. Pythia-1.4B throughout unless stated. Numbers are
copied from the cited report; CIs are 95% (bootstrap as in that report).
"Declared" says whether the experiment had a committed prediction and, if
so, whether its main prediction held. Tags are **suggestions only**:
- **main**: carries a claim in the paper body;
- **app**: appendix (support, robustness, or a necessary negative);
- **drop**: superseded, confounded, or not needed for the story.

Paths are relative to `reports/`. "—" = no figure. Notation as in
`round4/plan.md`: d = active-trained rank-1 DAS direction; site s = residual
output of layer s − 1; z: 0 = active intransitive level, 1 = transitive; D =
change after a transitive (T) minus an intransitive (I) active donor on a bad
passive; primary population = 64 plain-bad-verb pairs.

Round labels: **R1** = June DAS on aligned causative/inchoative templates
(v1, v2); **P** = September passive pilots and DAS prep (between R1 and R2);
**R2** = passive DAS round 2 and its follow-up; **R3**, **R4** = rounds 3–4.

## R1 (June): aligned causative/inchoative templates

No predictions were declared in this period.

| ID | Question | Key number [95% CI] | Declared | Report | Figure | Tag |
|---|---|---|---|---|---|---|
| R1-gate | Do Pythia 1.4/2.8/6.9B show the FreqBLiMP frequency decline on lexical-licensing paradigms? | 1.4B causative accuracy Head 0.80 → XTail 0.71 (NLL) / 0.80 → 0.70 (in-template LP), no CI; same pattern at 2.8B and 6.9B | — | `pythia_behavior_gate_20260617.md` | — | app |
| R1-patch | Where do attribution / exact patching localize the contrast (fixed-subject v1)? | Late residual, layers 16–23; exact top site L23 (normalized ≈ 1.0) | — | `v1_pythia_patching_report_20260617.md` | `figures/v1_localization_head_primary.png` | drop |
| R1-DAS | Does rank-1 DAS at L23 control the v1 proxy (" the" vs ".")? | Eval success 0.91, effect 9.8; random 0.03, shuffled −0.36 | — | `v1_das_control_report_20260617.md` | — | drop |
| R1-transfer | Does a Head-trained v1 direction transfer to Tail/XTail? | Success 1.00 / 0.86 / 0.79 (Head / Tail / XTail) | — | `v1_head_to_tail_transfer_report_20260618.md` | `figures/v1_synthesis_head_normalized.png` | drop |
| R1-v2DAS | Same with subject/frame decorrelated (v2), plus a red/blue readout control | Effect Head 12.5, low 11.2; red/blue, random, shuffled ≈ 0 | — | `v2_head_low_results_report.md`, `v2_red_blue_control_report.md` | — | app |
| R1-proj | Is the v2 direction a licensing axis or an object/no-object target axis? | Causative-calibrated inchoative margin negative at L18/20/23 (e.g. L23 Head −21.9): target axis | — | `das_projection_diagnostic_results_20260625.md` | — | app |
| R1-wpLP | Does whole-sentence LP recover the v2 contrast? | Acc 0.81–0.99; low > head is a subject-mix artifact for inchoatives (balanced 0.837 both) and verb-inventory difficulty for causatives (balanced 0.883 vs 0.982) | — | `v2_whole_pair_lp_report.md`, `v2_whole_pair_lp_regime_oddity_analysis.md` | — | drop |
| R1-wpDAS | Does whole-pair-trained DAS give a licensing axis (L16–23)? | Injection strong at every layer; read-side sign flips by subtask at every layer | — | `whole_pair_das_results_20260626.md`, `whole_pair_das_layer_sweep_20260626.md` | — | app |

R1 conclusion used later: proxy-trained directions in causative/inchoative
templates track the next-token target, not licensing. This motivated the
passive design (licensing and object expectation come apart in passives).

## P (September): passive behaviour and DAS prep

No predictions were declared in this period.

| ID | Question | Key number [95% CI] | Declared | Report | Figure | Tag |
|---|---|---|---|---|---|---|
| P-match1 | Rare-verb passive deficit with 8 shared frames? | Head − XTail margin −0.85 [−3.38, +1.69] (`passive_1`) | — | `matched_passives/report.md` | — | drop |
| P-match2 | Same with 126 curated, verb-specific contexts | Near ceiling: acc 99–100%; Head − XTail margin +2.15 [+0.81, +3.53] (`passive_1`) | — | `matched_passives_v2/report.md` | — | app |
| P-discrep | Why is the curated pilot easier than released FreqBLiMP? | Pythia released Head→XTail −8.7 pp (`passive_1`); same verbs, curated − released contexts +29.0 [+21.9, +36.6] pp (XTail, `passive_1`, pairs equally weighted) | — | `passive_discrepancy_20260923/analysis.md` | — | app |
| P-cross | Verb rarity vs noun/context rarity (verb band × context band) | Verb effect −13.0 [−21.1, −3.5] pp, whole margin −1.30 [−2.11, −0.39], verb margin −1.11 [−1.82, −0.41]; context effect +1.3 [−0.9, +3.4] pp | — | `passive_band_cross/report.md` | `figures/band_cross_accuracy.png` | **main** |
| P-fit | Does the deficit survive matched plausibility (LLM fit ratings)? | Participle margin XTail − Head −1.11 [−1.78, −0.46] (good + bad-patient fit; `passive_1`); whole margin −0.25 [−1.10, +0.63]; own-context advantage 21.6 → 5.0 pp | — | `../docs/fit_ratings.md`, `fit_ratings/report.md`, `fit_ratings/bad_fit_report.md` | `figures/fit_accuracy.png`, `figures/xtail_effect_matched.png` | **main** |
| P-base | Natural passive baseline (task 1) | Good − bad " by" ≈ +0.9 (Head, Tail), +0.49 [+0.11, +0.82] (XTail); no object preference in passives | — | `passive_das_prep/task1_baseline.md` | — | app |
| P-bysplit | Why some pairs have negative *by* margins (task 2) | Good side low: the transitive wants its own particle or PP (summed **up**, swaddled **in**); bad side high: the intransitive gets *by* (doff/**salivate**, debug/**tingle**) | — | `passive_das_prep/task2_by_split.md` | — | app |
| P-single | Single-prompt " by" readout as a proxy (task 3) | Spearman with *by* margin 0.41 [0.33, 0.49] per item, 0.54 per pair | — | `passive_das_prep/task3_single_prompt.md` | — | app |
| P-yesno | Yes/No prompting as a non-LP readout (task 4) | Gate fails: Head 60.8% [50.5, 69.8] vs LP 81–83%; r with LP −0.03 | — | `passive_das_prep/task4_yesno.md` | — | app |

## R2: rank-1 d on actives, passive test, follow-up

| ID | Question | Key number [95% CI] | Declared | Report | Figure | Tag |
|---|---|---|---|---|---|---|
| R2-DAS | Rank-1 DAS on actives ("object next"; 29 pairs: 8 original Head + 21 expansion), site 17 | Held-out IIA 0.97 [0.94, 0.99], gap fraction 1.06 [1.01, 1.11]; random 0.02, shuffled 0.34 | frozen config; no outcome prediction | `passive_das_prep/das_round2_results.md` | — | **main** |
| R2-sites | Active DAS at the other sites (4–16): how well does a rank-1 d control actives early? | Held-out IIA (I←T / T←I): site 4 0.54 [0.40, 0.66] / 0.36 [0.24, 0.48]; site 8 0.76 [0.65, 0.85] / 0.81 [0.71, 0.89]; site 12+ ≥ 0.87 | frozen configs; no outcome prediction | `passive_das_prep/das_round2_results_site{4,6,8,10,12,14,16}.md` | — | app |
| R2-ptest | Is d "object next" (surface) or "takes an object" (abstract) in passives? (sites 8, 12, 17) | Site 8: O +0.36 [0.30, 0.44] NO RISE, " by" +0.72 [0.63, 0.82] RISE → abstract; site 17: O +3.58 [3.36, 3.79] → surface | menu of outcomes; "surface at 17" lean held | `passive_das_prep/passive_test_results.md` | — | **main** |
| R2-ctrl | Random and shuffled-label controls at the passive test | Norm-matched random max \|D\| ≤ 0.13 on O, " by" and "." at every site (up to 0.20 on pronouns); shuffled-label = weaker copy of d (\|cos\| 0.69, ≈ 0.6× " by") | criteria declared; held | `passive_das_prep/passive_controls_site8.md` (+ other sites) | — | app |
| R2-depth | Passive test at all 8 DAS sites (4–17) | Abstract at 4, 6, 8; mixed at 10, 12; surface at 14, 16, 17 | 4 of 5 fill-in site calls held (14: surface, not mixed) | `passive_das_prep/layer_fillin_results.md` | `passive_das_prep/figures/depth_D.png` | **main** |
| R2-mech | Why no object rise in passives? Signed decomposition at site 8 | Object survival s = 0.46 [0.40, 0.50]; object MLPs 15/18/21/22 vanish in passives; MLPs 11, 13, 14, 16, 17 flip sign on " by" | story 1 predicted; held (at the boundary) | `passive_das_prep/site8_mechanism_results.md` | — | **main** |
| R2-dose | Dose-response of " by" and O along d at site 8 | " by" slope +0.92 [0.81, 1.04] (linear); object slope ratio 0.43 [0.34, 0.51]: intermediate | "coupled" predicted; missed (intermediate) | `passive_das_prep/followup_results.md` | — | app |
| R2-freq | Does the passive d gap track frequency and behaviour? | Site-8 gap slope on Zipf +0.065 [+0.037, +0.099]; r with behaviour across sites: Pearson −0.06 to +0.22, Spearman −0.03 to +0.24 | no prediction | `passive_das_prep/frequency_results.md` | — | **main** |

## R3

| ID | Question | Key number [95% CI] | Declared | Report | Figure | Tag |
|---|---|---|---|---|---|---|
| R3-A1 | Is the frequency/behaviour null a reliability failure? | Split-half R ≥ 0.985 for every per-pair measure; site 4 vs LP r +0.22 [+0.00, +0.43] | held | `passive_das_prep/round3_reliability.md` | — | app |
| R3-A2 | Is the site-8 separation still present late? | Passive d_8 gap flat to site 23 (retention 0.98–1.16); active d_8⊥17 gap → 0 by 22 | mostly held; rule unresolved at 14–17 | `passive_das_prep/round3_late_projection.md` | `passive_das_prep/figures/round3_late_projection.png` | app |
| R3-A3 | What drives the object acceleration above the passive range? | Acceleration +0.26 [0.21, 0.32]/z, of which final-LN scale +0.23 [0.19, 0.27] | partly; rules misleading | `passive_das_prep/round3_dose_decomp.md` | — | app |
| R3-A4 | Is low early IIA a rank or tokenization artifact? | Rank-4 gain +0.067 [+0.048, +0.087] IIA at site 4, +0.044 at 6; multi − single −0.03 to +0.10 (CIs ±0.3) | rank rule held; rank-4 gain < 0.05 missed at 4, held at 6; token deficit missed | `passive_das_prep/round3_token_rank.md` | — | drop |
| R3-B5 | Reverse DAS: passive-trained d_p tested on actives | \|cos(d_p, d_a)\| 0.57 (site 4) → 0.22 (17); O RISE in actives at 4–6 (+0.80); d_p raises " by" in actives at every site | partly (aligned at 4–6, specific at 16–17; " by" ≤ 0 missed) | `passive_das_prep/round3_reverse.md` | — | **main** |
| R3-B6 | Does the site-8 switch generalize from "was" to "got"? | D(O) got − was +0.15 [0.10, 0.19] (90% CI; inside ±0.49) | held | `passive_das_prep/round3_get_passive.md` | — | app |
| R3-B7 | Path patching: what switches the flagged MLPs between "was" and "has"? | Joint heads 67–115% of S in " by" MLPs 11–17; earlier MLPs carry the object MLPs 15/18/21/22 | H1 held; H2, H3 missed | `passive_das_prep/round3_path_patch.md` | — | **main** |
| R3-C8 | Nonce verbs: does transitive use in context move a later passive along d? | Matched T − I z +0.15 [0.14, 0.16] (site 8); balanced +0.11; " by" +0.56 [0.49, 0.62] (balanced) | held | `passive_das_prep/round3_nonce.md` | — | app |
| R3-C9 | Object relatives and tough constructions under the T − I patch | Gates pass (L_OR +2.26 [1.95, 2.58]); declared readings mixed; post hoc site-8 D reproduces each construction's profile | mostly missed (mixed) | `passive_das_prep/round3_constructions.md` | — | drop (superseded by R4-A2, R4-B) |

## R4

| ID | Question | Key number [95% CI] | Declared | Report | Figure | Tag |
|---|---|---|---|---|---|---|
| R4-A1 | Does donor overshoot explain the excess object mass? | Excess passive − TC +0.22 [−0.12, +0.54]: not ordered | held (as anticipated) | `round4/a1_a3_dose.md` | — | app |
| R4-A2 | In-range patch: does d reproduce each construction's natural profile? | OR, TC reproduce at 6 and 8; passive " by" 0.44 [0.30, 0.82] of natural at 8 | mostly held | `round4/a2_inrange.md` | — | **main** |
| R4-A3 | One OR pair with past ≠ participle | Every estimate moves ≤ 0.045 | held | `round4/a1_a3_dose.md` | — | drop |
| R4-A4 | Is d_p's " by" effect in actives a generic "by lever"? | Shuffled direction by +0.36 at 8; projecting it out halves d_p's " by" (ratio 0.50): mixed | missed (mixed, not separable) | `round4/a4_bylever.md` | — | app |
| R4-A5 | Auxiliary heads: previous-token heads? Do they survive adverb / got frames? | 0 of the B7 top 5 are previous-token heads; role retained in both frames; mid heads read the adverb | mostly held; "read across adverb" missed | `round4/a5_aux_heads.md` | — | app |
| R4-B | Filler-gap: is the in-range d move licensing-like (filler-sensitive)? | Gates pass (L interaction +1.55 [1.13, 1.97] embedded, +1.45 [1.04, 1.88] matrix); in range at site 8: "." +1.01 [0.82, 1.21], PREP −0.58 [−0.67, −0.49], O not above T (+0.19 [−0.10, +0.51]) → licensing-like; weaker for non-DAS verbs | mostly held | `round4/b_fillergap_gates.md`, `round4/b_fillergap.md` | `round4/figures/fillergap_layers.png` (untracked) | **main** |
| R4-C8 | Nonce: does the d shift survive an inflection mismatch? | s form z +0.062 [0.056, 0.067] (site 8); object rise falls to ≤ half | held | `round4/c_nonce_cues.md` | — | app |
| R4-C9 | Nonce: is the d shift driven by transitivity or an adjacent NP? | Adjacent NP (intrans.) +0.154 [0.147, 0.161] z vs transitive without NP (relative) +0.013 [0.011, 0.015]: NP-driven | missed | `round4/c_nonce_cues.md` | — | **main** |
| R4-D10 | Is the B7 switch a voice switch? ("has been V" vs "has V") | S reproduces B7 in 9 of 9 MLPs (within 0.03); was vs has been ≤ 0.03 | held (all five) | `round4/d10_voice.md` | — | **main** |
| R4-D11 | Do conjunction ("transitive AND passive") neurons exist in MLPs 11, 14? | Share of the " by" switch 0.65 [0.60, 0.70] (half B); top 10 carry 0.66 (exploratory) | held (count missed) | `round4/d11_conjunction.md` | — | **main** |
| R4-D12 | Does translation gain (TG) track frequency or behaviour? | TG Zipf slope −0.17 [−0.80, +0.40]; partial r(TG, released *by* margin \| z gap) +0.57 [+0.37, +0.73] | partly (TG–frequency missed) | `round4/d12_translation_gain.md` | — | **main** |
| R4-E13 | Oracle at d: set a rare verb to the Head class mean | XTail Δ by +0.045 [+0.003, +0.086] (site 8); Head control −0.18: rule not met | mostly missed | `round4/e13_oracle.md` | — | **main** (negative) |
| R4-E13d | Is the natural Head − XTail " by" deficit reliable (126 pairs)? | +0.45 [−0.33, +1.27] | descriptive | `round4/e13_deficit.md` | — | **main** |
| R4-E13T | Oracle at the translation stage (top-50 switch neurons) | XTail Δ by +0.165 [+0.103, +0.229]; vs random +0.169 [+0.107, +0.234]; Head control −0.094 [−0.208, +0.019]: rule not met | partly | `round4/e13t_oracle_translation.md` | — | **main** |

## R5: locating the deficit, mechanism tests, scale

Declared in `round5/plan.md`; results in `round5/results.md`. "Declared"
summarizes that part's predictions.

| ID | Question | Key number [95% CI] | Declared | Report | Figure | Tag |
|---|---|---|---|---|---|---|
| R5-A1 | Where is the participle deficit? (was vs has frame contrast L, 126 pairs) | L Head 3.84 / Tail 2.82 / XTail 1.72; ΔL +2.12 [+1.26, +2.94]; intransitive side +2.60, transitive −0.48 | mixed (deficit held; size, sides, ΔM_has missed) | `round5/a1_frame.md` | — | **main** |
| R5-A2 | Exact decomposition L = Δh · (u_T − u_I) | identity error 4e-5; I-row affinity Zipf slope −1.58; first subtoken carries XTail 0.03 / Tail 0.34 of L; singleton D_L +2.09 [+1.00, +3.23] | mostly held (subtoken share missed) | `round5/a2_decomposition.md` | — | **main** |
| R5-A3 | Do Pile passive-slot counts predict Δ_pass beyond overall frequency? | curated: unresolved (T +0.045 [−0.09, +0.20], I +0.043 [−0.21, +0.32] CV MSE); pool T: slot adds +0.41 [+0.25, +0.61]; I-verb "was V" = prepositional passives | missed (unresolved) | `round5/a3_counts.md` | — | app |
| R5-A4 | Output-side restoration of rare (Tail) single-token participles along the was − has direction | oracle +0.53 [+0.24, +0.83] nats on affected passives (I-only +0.33, T-only +0.20); Head and unaffected items unchanged; label-free −0.02 [−0.22, +0.19] vs half the oracle | held | `round5/a4_restore.md` | — | **main** |
| R5-A5 | Does a grammaticality probe (Wang et al. 2026) show the rarity decline? | probe Head 88.2 → XTail 71.9%, D +16.3 [+12.9, +19.8]; LP D +9.7; D_LP − D_probe −6.6 [−11.0, −2.0] | missed (larger, not smaller, decline) | `round5/a5_probe.md` | — | **main** |
| R5-B5 | Is d (or the switch neurons) necessary for good passives? | d → I mean removes 0.15 of " by"; top-50 neurons 0.09; both 0.19; class info beyond d (0.59 of the gap remains) | mostly missed (necessity) | `round5/b5_necessity.md` | — | **main** (negative) |
| R5-B6 | Serial chain heads → mid MLPs → late object MLPs | freeze leaves 0.14–0.26 of the late switch; strict two-hop 0.29–1.08 (chain holds 2 of 4) | mixed | `round5/b6_serial.md` | — | **main** |
| R5-C8 | Filler-gap licensing with non-DAS verbs | matrix licensing-like at sites 4, 8, 10; embedded mixed at 8, surface at 6 | mixed | `round5/c8_fillergap.md` | — | **main** |
| R5-C9 | Do the switch neurons read the class in wh-gap frames too? | ρ_nat 1.83–1.93, ρ_in 1.43–1.57 (what / passive); "that" frame ≈ 0: shared and filler-conditioned | missed (stronger than predicted) | `round5/c9_shared_readers.md` | — | **main** |
| R5-B7 | What do the top switch neurons respond to in Pile text? | MLP11.n3462: object gap 0.78 + passive 0.23 of top 40; MLP14.n2114 / n5407: passive 0.85 / 0.97, " by" next 0.53 / 0.78; κ = 1.00 | held for ranks 2–3, rank 1 gap reader | `round5/b7_wild.md` | — | **main** |
| R5-D10 | Feasibility on 6.9B / 12B | fp32 on one H100; DAS run 2.1 / 3.5 min; replication ≈ 3 / 5 GPU-h | — (feasibility) | `round5/d10_scaleup.md` | — | app |
| R5-D10b | A1/A2 across Pythia 160M–12B | ΔL 1.57–2.44 at every size (CI above 0), intransitive side throughout | held | `round5/d10b_suite.md` | — | **main** |
| R5-D10c | FreqBLiMP passive_1 Yes/No on 1.4B / 6.9B / 12B | Yes for 97–100% of sentences; Head 0.50 / 0.45 / 0.40 | missed | `round5/d10c_yesno.md` | — | app (negative) |
| R5-D11 | Expanded T/I verb list | 281 pairs (Head 47, Tail 132, XTail 102), 62 held out; κ = 0.88 | held (rule met) | `round5/d11_verbs.md` | — | app |

## Suggested storyline (for discussion, not a decision)

1. **Behaviour.** The passive frequency decline is a verb effect, not a
   noun/context effect (P-cross). Part of it is plausibility; about one
   log unit remains in the participle's own log-probability at matched fit
   (P-fit). Round 5 A1–A2 locates this residual.
2. **An early, construction-general verb-class value.** d at sites 4–8 acts
   as "takes an object": in passives it raises " by", not objects
   (R2-ptest, R2-depth); in range it reproduces the object relative, tough
   and filler-gap profiles (R4-A2, R4-B). Late, the same direction is
   "object next". Caveat: in context it is set by surface adjacency
   (R4-C9).
3. **Translation.** Heads read the passive auxiliary and switch mid MLPs
   (R3-B7, R4-D10); tens of switch neurons carry most of the " by" switch
   (R4-D11).
4. **Frequency.** The d gap tracks Zipf but predicts no behaviour (R2-freq,
   R3-A1); TG predicts *by* behaviour but does not track Zipf (R4-D12).
   Restoring d does nothing (R4-E13); restoring switch neurons helps a
   little (R4-E13T). The *by* deficit itself is unreliable (R4-E13d), so the
   rare-verb deficit is mostly not in the continuation.
5. **Round 5: where the deficit is.** It is in the passive slot itself and
   comes from intransitives: frequent intransitive participles are
   rejected after "was", rare ones much less (R5-A1). On the output side
   this is a compressed contrast along one was − has direction (R5-A2),
   and editing rare rows to the Head level recovers about half a nat
   (R5-A4). The deficit is the same at every Pythia size from 160M to 12B
   (R5-D10b). A grammaticality probe on the final state shows an even
   larger rarity drop (R5-A5), so the deficit is not only output-side.
   Pile slot counts do not reliably explain it for the curated verbs
   (R5-A3).
6. **Round 5: the mechanism is not passive-specific.** The early d value is
   far from necessary (R5-B5). The late object MLPs depend on the mid MLPs
   (R5-B6). The switch neurons read the verb class in wh-gap frames too,
   only with a filler present (R5-C9). In natural text the top neuron fires
   on transitive verbs with a missing object (R5-B7).
