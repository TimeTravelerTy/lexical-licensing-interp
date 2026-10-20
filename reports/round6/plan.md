# Round 6: declared plan (paper 1: frequency and entrenchment)

Builds on rounds 1–5 (`../results_index.md`). Round 6 serves paper 1 (the
rare-verb deficit: where it is, what exposure explains it, whether it is
specific to the passive, and whether it can be restored or preempted).
Paper-2 items (LEACE necessity, gap-neuron mediation, human annotation of B7,
6.9B DAS) are parked.

Same rules as round 5: each part is written and committed before any of its
runs, with predictions and decision rules stated before the data exist;
where a prediction leans on earlier outputs, that is said. An outside
reviewer (Codex) reviews each part's draft and code before the runs.

Run order: 0, A1, A2, B3, C4, C5, A6, A7. Parts are declared in that order,
in groups, each group in its own commit before its runs.

Notation as in rounds 4–5 (`../round5/plan.md`): T / I = transitive
(passivizable, "good") / intransitive ("bad") verb of a pair; Δ_pass(V, c) =
log P(V | "The N was") − log P(V | "The N has") (full participle log-prob,
summed over subtokens, no BOS); L = Δ_pass(T) − Δ_pass(I) per pair; a_V =
u_V1 · Δh̄ (passive-slot affinity of a verb's first subtoken); site s =
residual output of layer s − 1; z = a verb's site-8 coordinate on the
active-trained DAS direction d. D11 = the round-5 expanded inventory (281
T/I pairs: Head 47, Tail 132, XTail 102; 62 held out;
`data/round5/verbs_expanded/pairs.csv`, loader `scripts/verbs_expanded.py`).
"Curated" = the 126 band-cross pairs of rounds 4–5. CIs are 95%, 2,000
draws, seed 17, unless stated otherwise. Layout: plan and reports in
`reports/round6/`, regenerable inputs in `data/round6/<part>/`, outputs in
`results/round6/<part>/`.


# Part 0: housekeeping (declared 2026-10-20)

No model runs, no predictions.

1. **Results index.** Round 5 is already listed in `../results_index.md`
   (rows R5-*). Add a column tagging each experiment as paper 1 (P1:
   frequency / entrenchment), paper 2 (P2: mechanism) or both, and list the
   parked paper-2 items. Round-6 rows are added when round 6 reports.
2. **Human data.** Check the format of the FreqBLiMP human study
   (`freq-blimp/human_validation_ui`; returned three-annotator export, the
   corrected merge of 2026-08-25) for `passive_1` / `passive_2`.
   - Checked before this declaration (format only, no outcome tabulated
     beyond response counts): it is **forced choice** (A / B / can't
     decide, with reasons), followed by a 1–5 naturalness rating of the
     intended acceptable sentence only. There are no per-sentence ratings.
   - So, per the round-6 rule, **accuracy by band only, descriptive**; no
     bad-item acceptance. Per paradigm × condition (Original, Head, Tail,
     XTail; 15 pairs × 3 annotators): correct among decided responses,
     can't-decide share, accuracy with can't-decide counted as half, items
     with 2-of-3 support, mean naturalness of the good sentence; Pythia-1.4B
     LP accuracy on the same released items (round-5 A4 unedited scores)
     beside it. Items resampled within cell for 95% CIs (15 items per cell,
     so these are wide). Script `scripts/analyze_human_passives.py`; report
     `human_passives.md`; per-item table (no annotator-level data)
     `data/round6/human/passive_items.csv`. The raw export stays outside the
     repo; its sha256 is recorded.


# Part A1–A2: exposure counts and the D11 frame contrast across the suite (declared 2026-10-20)

Revised after an outside review (Codex) of the first draft and its code,
before any outcome analysis. Two disclosures: (1) the count fetch (data
collection only) started while this part was being reviewed; no count has
been looked at. (2) 433 of the 562 D11 participles (103 of the 124
held-out forms) were already scored in round 5's freq frames (as curated or
pool verbs), so D11 is not an unseen inventory; the predictions below lean
on round 5 where stated.

## What is already known

- Round 5 A1 (curated 126 pairs, 1.4B, freq frames): L Head 3.84 / Tail
  2.82 / XTail 1.72; the Head − XTail deficit (ΔL +2.12 [+1.26, +2.94]) is on
  the intransitive side (Δ_pass(I) Head −3.45, Tail −2.73, XTail −0.85;
  Δ_pass(T) shows no band trend). Tail − XTail in the same data: ΔL +1.10,
  of which the I side +1.88 and the T side −0.77. By token cell (round-5
  A2, curated): Tail − XTail L in 2×2 3.28 vs 1.50, 2×3 4.01 vs 2.22, 3×2
  2.24 vs 1.40.
- D10b: the Head − XTail L deficit is present at every Pythia size
  (ΔL 1.57–2.44), on the intransitive side.
- Round 5 A3 (Pile train counts, lowercase word-initial form only):
  r(Δ_pass(I), Zipf) = −0.58, r(Δ_pass(I), log c) = −0.50; for curated I
  verbs wordfreq Zipf predicted Δ_pass(I) better out of sample than Pile
  log c (CV MSE 2.35 vs 2.65); slot counts did not resolve for curated
  verbs; for the 627 pool T verbs, adding log c_be to log c lowered CV MSE
  (+0.41 [+0.25, +0.61]). Hand audit: I-verb "was V" hits are mostly
  prepositional passives.
- D11 token structure: Head is almost all single-token (T 44, I 45 of 47);
  Tail has 24 single-token T and 20 single-token I; XTail none (one T). So
  Head − XTail is confounded with token count. Tail and XTail overlap in
  the multi-token cells: exact (T tokens, I tokens) cells with ≥ 5 pairs in
  both bands are 2×2 (Tail 62 / XTail 23), 2×3 (8 / 28), 3×2 (15 / 20) and
  3×3 (5 / 17): 90 Tail and 88 XTail pairs. Held-out pairs in these cells:
  Tail 16 / 2 / 4 / 0, XTail 5 / 5 / 8 / 5.

## A1. Pile exposure: counts, BE/HAVE preference, model comparison

**Verbs.** The union of D11's 562 participles and the curated 252: 608
participles. Primary population: D11 (281 pairs); curated (126 pairs)
secondary, with the same rules, for comparison with round 5.

**Counts.** infini-gram API, index `v4_piletrain_llama` (the Pile training
set; Pythia saw just under one epoch of it). Exact counts of token sequences
(Llama-2 tokenizer) with no word boundary on the right; every query, its
token ids and its count are saved (`data/round6/counts/`). 3,655 counts
already fetched in round 5 for the identical query string on the same index
are reused (the builder checks the index and that round 5 has one count per
string). Per participle V (lowercase v, capitalized V̂, simple past p; p = v
except arisen → arose, striven → strove, browbeaten → browbeat, outdone →
outdid):
- **c_total** (primary, boundary-safe canonical count): word-initial
  lowercase "▁v" + word-initial capitalized "▁V̂" + the two forms directly
  after a newline ("\nv", "\nV̂"; the newline gives a left word boundary,
  and these tokens carry no word-initial marker). Full word strings are
  queried, so a multi-token participle is counted as its whole token
  sequence. Not counted: unmarked forms after quotes, brackets or hyphens,
  and all-caps forms.
  - Sensitivity 1 (marked only): "▁v" + "▁V̂".
  - Sensitivity 2 (upper bound): "▁v" + "▁V̂" + every unmarked occurrence
    (token-id counts of the ids that follow the newline). This has no left
    boundary and also counts other words that end in the same token
    sequence (e.g. "lied" inside "relied", "ached" inside "encroached"), so
    it is reported, not interpreted on its own.
  - Neither variant has a right boundary (a marked sequence can begin a
    longer word, e.g. "rated" in "ratedness").
- **c_be** (passive slot): Σ_X count("X v"), X ∈ {was, were, been, is,
  are, be, being}; per-auxiliary counts kept. It includes adjectival,
  resultative and prepositional-passive uses.
- **c_perf** (perfect slot): Σ_X count("X v"), X ∈ {has, have, had,
  having}; it includes possession + adjectival participle.
- **c_comp** (competing frame: the verb's finite active use). Primary:
  clause-anchored simple past, Σ count("a P p"), a ∈ {that, when, and},
  P ∈ {he, she, they}; auxiliary inversion is impossible after these
  anchors, so passive questions ("Was it abandoned?") and perfect questions
  ("Has she arrived?") cannot match. For an intransitive this is its
  intransitive active frame, the use that, under statistical preemption,
  competes with the passive; for a transitive it counts active uses with or
  without an object. It is a sample of the verb's active uses whose
  pronoun / anchor mix can differ by verb (animacy, genre), not a fixed
  fraction. Secondary: broad pronoun adjacency Σ count("P p"),
  P ∈ {he, she, it, they, He, She, It, They}, which also matches inverted
  questions and "found it abandoned"; its precision is audited (below).
- **Frame counts for B3** (collected now, analysed in B3): for
  P ∈ {he, she, they, He, She, They}, count("P p the") (object onset) and
  count("P p.") (clause end).
- **Zipf**: wordfreq 3.1.1, participle form (as D11).
Smoothing +0.5 on every count (sensitivity +0.1, +1); zero shares reported
per class and band.

**BE-versus-HAVE adjacency preference** (the operational proxy for the
brief's distinctive-collexeme strength of a verb for passive vs active;
Gries & Stefanowitsch 2004). For verb V, the 2 × 2 table V vs the other
inventory verbs × be-frame (c_be) vs have-frame (c_perf). The have-frame is
the active counterpart with the same participle form, so the table is the
corpus analogue of Δ_pass (was vs has), with round 5's voice caveat (be +
participle is not only the eventive passive).
- **Primary S: log-odds ratio** with Haldane +0.5 smoothing,
  S_V = log((c_be + .5) / (c_perf + .5)) − log((ΣC_be − c_be + .5) /
  (ΣC_perf − c_perf + .5)), totals over the 608 inventory forms (a fixed,
  selected reference, not English verbs in general). Up to a near-constant
  it is log(c_be / c_perf): a preference that does not grow with overall
  frequency.
- Secondary: Fisher-based strength, signed −log10 p (two-sided Fisher exact
  test, sign of the log-odds; saturates at 300); the log-odds with c_comp
  as the active column; and the log-odds of "was v" vs "has v" only (the
  exact frames of Δ_pass).

**Outcomes** (Pythia-1.4B, the 24 freq frames, mean over contexts):
Δ_pass(T), Δ_pass(I), L. D11 values come from the A2 run; curated values
are round 5's (`results/round5/frame/frame_scores.parquet`); verbs in both
must agree to ≤ 1e-3 nats (fail-fast). For single-token verbs the affinity
a_V equals Δ_pass(V) plus a verb-independent constant, so the a_V analysis
is the single-token subset of the Δ_pass analysis (an OLS model with an
intercept gives identical CV errors).

**Predictors** (log10 counts): Z (Zipf), C (c_total), B (c_be), P (c_perf),
K (c_comp, primary), S (BE/HAVE preference). OLS, per class.

**Cross-validated model comparison.**
- 10-fold CV with a frozen fold map: pairs shuffled within band (seed 17)
  and dealt to folds in turn, so each pair (and both its verbs) sits in one
  fold, shared by every comparison; curated pairs the same way. Every unit
  must receive an out-of-fold prediction (fail otherwise); rank-deficient
  designs are reported.
- Uncertainty: pairs resampled within band × the 24 contexts within
  animacy strata, 2,000 draws, seed 17. A draw's resampling multiplicities
  enter the fits and the error as weights; the fold map stays fixed, so a
  duplicated pair is never in both training and test. Intervals are
  conditional on the inventory, the contexts and the fold map.
- For base model a and model b, Δ = MSE_a − MSE_b (positive: b predicts
  better): **b better** = 95% CI of Δ above 0; **a better** = below 0;
  "(within 2%)" is added when the 90% CI of Δ / MSE_a also lies within
  ±2%; **equivalent** = not better either way and the 90% CI of Δ / MSE_a
  within ±2%; otherwise **unresolved**.
- **Direction.** For each added predictor the OLS coefficient (full data)
  is reported with its bootstrap CI. A prediction about a direction holds
  only if the comparison reads "b better" and the coefficient has the
  predicted sign with its 95% CI excluding 0.

Declared comparisons (D11 primary; curated with the same rules):

| ID | outcome | a → b | question | family |
|---|---|---|---|---|
| I1 | Δ_pass(I) | null → C | does total exposure predict intransitive rejection? (C coefficient < 0) | primary |
| I2 | Δ_pass(I) | C → K | competing-frame count vs total count (either direction) | secondary |
| I3 | Δ_pass(I) | C → C + K | does the competing frame add to total exposure? (K coefficient < 0) | primary |
| I4 | Δ_pass(I) | C → C + B | does the passive slot add? | secondary |
| I5 | Δ_pass(I) | Z → C | Pile counts vs wordfreq Zipf (either direction) | secondary |
| T1 | Δ_pass(T) | null → C | does total exposure predict transitive acceptance? | secondary |
| T2 | Δ_pass(T) | C → C + B | does the passive slot add? (B coefficient > 0) | primary |
| T3 | Δ_pass(T) | C → C + S | does the BE/HAVE preference add? (S coefficient > 0) | primary |
| T4 | Δ_pass(T) | Z → C | Pile vs Zipf | secondary |
| L1 | L | Z_T + Z_I → C_T + C_I | Pile vs Zipf at pair level | secondary |
| L2a | L | C_T + C_I → + S_T | does the T verb's preference add at pair level? | secondary |
| L2b | L | C_T + C_I → + K_I | does the I verb's competing frame add at pair level? | secondary |
| L2 | L | C_T + C_I → + S_T + K_I | both | secondary |

Claims rest on the primary family (I1, I3, T2, T3); for these a
Bonferroni 98.75% CI of Δ is reported beside the 95% CI. Secondary rows are
reported without correction. Associations here are observational; a
competing-frame effect would be consistent with statistical preemption but
would not identify it.

Sensitivities (each on the primary family and L1–L2): **token count**
(log participle token count added to both models; for L both verbs'
counts), since frequency and token count are confounded in D11;
canonicalization (marked-only and upper-bound c_total); smoothing 0.1 and
1; secondary K (broad pronoun) and secondary S measures in place of the
primary ones; for T3, C → C + B + P (whether a preference effect comes from
more BE or less HAVE). Also reported: correlations, in-sample R², design
condition numbers.

Single-token subset (a_V): I1, I3, T2, T3 on D11 single-token verbs (Head
and Tail: T 68, I 65; plus the one XTail single-token T), descriptive.

**Re-binning by counts.** Within each class, verbs ranked by c_total
(descending; ties broken by the form, alphabetically) and cut into bins
with the band sizes (47 / 132 / 102); pairs by mean log c_total the same
way. The bins are defined on this fixed inventory (descriptive), not
learned within folds. Tables of Δ_pass(T), Δ_pass(I) and L by Zipf band
and by count bin, with top − bottom bin differences. CV comparison B1:
band dummies (FreqBLiMP Zipf bands) → count-bin dummies, for Δ_pass(I),
Δ_pass(T) and L (secondary).

**Diagnosis of the curated-I case** (round 5: Zipf beat Pile log c):
- D1: refit I5 on curated I with the new c_total, with each sensitivity
  version, and with the round-5 lowercase-only count.
- D2: disagreement = residual of log10 c_total on Zipf (one regression over
  the 608 forms). The 10 curated I verbs with the largest positive and the
  10 with the largest negative residual; for each, Δ_pass(I) and the
  out-of-fold predictions of the Zipf and Pile models.
- D3 (why): for these 20 verbs, 20 occurrences each, allocated to the
  c_total variants in proportion to their counts (find + get_doc_by_rank at
  evenly spaced suffix-array ranks within a variant; descriptive, not a
  random sample): the Pile subset of each document (metadata) and a hand
  label of the use (verbal / adjectival / nominal or name / other, e.g.
  code or non-English), with the variant recorded. Controls: the 10
  curated I verbs with the smallest |residual|, chosen to match the bands
  of the 20 as closely as possible, sampled the same way. Labels by Claude;
  round-5 outcomes are already known, so this is a descriptive audit, not
  blind.
- Precision audit of the broad pronoun proxy: for 6 verbs (3 T, 3 I,
  spread over bands), up to 10 occurrences each of "it p" and "she p",
  labelled finite active / other (e.g. inverted question, small clause).

**Predictions** (ranges are about point estimates).
- I verbs: **I1 b better, C coefficient < 0**; I3 **b better, K
  coefficient < 0** (low confidence: C and K are highly correlated); I2 and
  I4: the slot does not improve on total exposure (I4 not "b better"); I5:
  not resolved in either direction on D11 (low confidence).
- T verbs: T1 not "b better" (no trend in Δ_pass(T)); **T2 b better, B
  coefficient > 0**; **T3 b better, S coefficient > 0** (both low
  confidence: pool T showed slot effects in round 5, curated T did not).
- L: L2a or L2b b better (low confidence).
- D1: the curated Zipf advantage (MSE_C − MSE_Z) is at most half its
  round-5 size with the new c_total, and does not reverse. D3: the
  Pile-over-represented verbs come disproportionately from technical
  subsets (PubMed, FreeLaw, USPTO, GitHub, ArXiv) or homographs.
- Count bins: the top − bottom difference in intransitive rejection,
  −Δ_pass(I), is at least as large as Head − XTail's.

## A2. D11 frame contrast across the Pythia suite

**Runs.** Round 5's A1 runner (`run_frame_contrast.py`) on the 562 D11
participles, every prefix set of round 5 (freq primary; been, cur — every
verb in all 126 curated prefixes, `--cur-all` — and they), for Pythia 160M,
410M, 1B, 1.4B, 2.8B, 6.9B and 12B, fp32, TF32 off, one job. Per model the
same numerical checks (tokenization, h shared within a prefix,
first-subtoken reconstruction, the identity L₁ = Δh · (u_T1 − u_I1) on every
D11 pair and prefix), tolerance 1e-3 for 1.4B and 1e-2 for the others (as
D10b); verbs also scored in round 5 must reproduce round 5's log-probs
(1e-3 for 1.4B, 1e-2 otherwise). Inputs: `data/round6/frame_d11/`.

**Measures** (per model; freq primary): band means of L, Δ_pass(T),
Δ_pass(I), M_has, L₁, L_rest; side contributions; readings as round 5's A1
(deficit / surplus / negligible / unresolved, bound 0.25 nats).
Bootstrap: pairs within band × split (within band × token cell × split for
the token-standardized contrasts) × contexts within animacy strata, 2,000
draws, seed 17; the same pair and context draws for every model (paired
across models).

**Primary comparison: Tail vs XTail, token-count standardized.** ΔL_tok =
Σ_s w_s (L̄_Tail,s − L̄_XTail,s) over the exact (T tokens, I tokens) cells
s ∈ {2×2, 2×3, 3×2, 3×3}, fixed weights w_s = 85 / 36 / 35 / 22 (out of
178; pairs per cell, both bands). The same contrast for Δ_pass(T) and
Δ_pass(I) gives the side contributions (ΔL_tok = ΔΔ_pass(T) − ΔΔ_pass(I)).
Equal token counts leave subtoken identity and frequency, morphology,
semantics and noun–verb fit uncontrolled; the 3×3 cell rests on 5 Tail
pairs. Sensitivities: OLS over all multi-multi Tail / XTail pairs, L ~
band + T tokens + I tokens; the 2×2 cell alone. Secondary: Head − XTail and
Head − Tail (token-confounded, as round 5), and by cell.

**DAS-unexposed (held-out) pairs.** The 62 held-out pairs (Head 4, Tail 29,
XTail 29) were never used in DAS training or selection, but most of their
verbs were scored in round 5 (above), so they are a subgroup, not an
untouched replication. Held-out Tail has no 3×3 pair, so the held-out
estimand is the **three-cell contrast** (2×2, 2×3, 3×2; fixed weights
85 / 36 / 35), reported beside the same three-cell contrast for all pairs
and for trainable pairs, plus held-out band means.

**A2 per-verb measures** (per model): Head AUC of a_V (T vs I, Head
single-token verbs); singleton D_L = (ā_T,Head − ā_I,Head) − (ā_T,Tail −
ā_I,Tail) on single-token verbs (Head 44 T / 45 I; Tail 24 T / 20 I), with
D_T and D_I; verbs resampled within role × band, Δh̄ re-drawn.

**Decision rules.** The round-5 A1 rules (bound 0.25 nats) applied to
ΔL_tok, the three-cell contrasts, ΔL (Head − XTail, Head − Tail) and ΔM_has,
per model. Side, only when ΔL_tok reads "deficit": **intransitive-side** if
−ΔΔ_pass(I) has a 95% CI above 0 and exceeds ΔΔ_pass(T) (paired difference
CI above 0); **transitive-side** in the mirror case; otherwise mixed. This
is dominance, not exclusivity. Without a deficit, the contributions are
reported descriptively.

**Predictions** (ranges about point estimates).
- 1.4B: ΔL_tok **deficit**, estimate in +0.5 to +2.0 nats,
  **intransitive-side**. Head − XTail: deficit (≈ +2, as round 5).
- Every size ≥ 410M: ΔL_tok deficit and intransitive-side; estimate within
  ±50% of 1.4B's. 160M: deficit or unresolved.
- Held-out three-cell contrast at 1.4B: positive; **deficit** or
  unresolved (22 Tail and 18 XTail pairs).
- Head AUC ≥ 0.9; singleton D_L > 0 with CI above 0 at 1.4B.

**Run clarification (A2, 2026-10-20, after job 9069457's checks, before
any analysis).** Pythia-160M stopped at the shared-prefix check: h differed
across the rows of a batch by up to 6.8e-4 (tolerance 1e-4); its
first-subtoken reconstruction error was 1.6e-3 and its largest |logit| at
the slot 845 (round 5's D10b saw the same scale at 160M). Scoring every
verb in the cur prefixes changes batch shapes relative to round 5, and the
fp32 rounding at this logit scale exceeds 1e-4; a hook or position error
would give differences of 0.1 or more. 1.4B, 410M, 1B and 2.8B passed with
h deviation 0. The runner gains `--h-tol` (default 1e-4, unchanged for
1.4B); suite models that fail only this check are rerun with 1e-2, the
tolerance D10b set for the reconstruction check, and the realized
deviation is reported.

## Order (A1–A2)

1. Commit this part.
2. Builders (done before commit, no outcomes): `build_frame_d11.py`
   (tokenizer only), `build_count_queries.py`.
3. Counts: `fetch_pile_counts_r6.py` locally (two shards, about 5 h). The
   D3 and precision-audit occurrences (`fetch_diag_contexts.py`) after the
   counts; labels before the A1 outcome analysis.
4. TSUBAME: one job, `run_frame_contrast.py --cur-all` on D11, seven
   models.
5. Analyses: `analyze_frame_d11.py` (A2), `analyze_exposure.py` (A1);
   reports `a2_suite_d11.md`, `a1_exposure.md`; summary in `results.md`.


# Part B3, C4, C5: other alternations, classify-from-z restore, in-context preemption (declared 2026-10-20)

Written while A1's counts were being fetched and before A2's analysis (no
A1/A2 outcome seen; A2's 1.4B scores exist but have not been analysed).
Revised after an outside review (Codex) of the first draft and its code,
before any run. **Disclosure (C4):** while smoke-testing the C4 analysis
code on a mock built from round-5 A4's stored outputs, the classification of
A4's 32 rare single-token verbs from their stored z was computed and seen:
balanced accuracy 0.94 (23 of 23 T and 8 of 9 I assigned correctly). C4's
classification result on that population is therefore known before
declaration; its restoration outcome is not (the mock copied the oracle
condition).

## B3. Other alternations: the symmetric entrenchment prediction

**Question.** In the passive, the rare-verb deficit sits on the side of the
verb that is (nearly) never used in the frame: rare intransitives are not
rejected after "was". Entrenchment predicts the same elsewhere: in an
object frame the dispreferred verb is the intransitive (rare I verbs should
leave room for an object), in a no-object frame it is the obligatorily
transitive verb (rare T verbs should leave room for a clause end). "Never
attested" is too strong for this inventory (111 of 281 I verbs carry a
"rare transitive" flag from at least one rater), so the hypothesis is
stated as preferred versus dispreferred frame; excluding flagged verbs is a
sensitivity.

**Verb inventory.** Checked before declaration:
- FreqBLiMP's alternation paradigms have few verb types on the side that
  matters: causative good (alternating) verbs 96 / 42 / 12 types (Head /
  Tail / XTail), inchoative good 96 / 42 / 12, drop_argument good 55 / 13 /
  19. The June causative / inchoative templates (`data/aligned_templates_v2`)
  have 17–36 lemmas per cell. Neither supports per-verb analysis by band.
- D11 (281 T/I pairs) does, so **D11 is the B3 inventory**. Its T verbs are
  non-alternating transitives, its I verbs non-alternating intransitives
  (the transitive / intransitive contrast behind the bad sides of
  FreqBLiMP's `transitive`, `intransitive`, `causative`, `inchoative` and
  `drop_argument`, not the causative alternation itself).

**Frames** (D11 verbs, the 24 A1 nouns, no BOS): primary simple past "The N
p"; secondary perfect "The N has v". The next-token distribution at the
verb's last subtoken:
- **O**, an object-onset proxy: round 2's object set (26 tokens). It also
  catches non-object continuations ("slept the entire afternoon", "laughed
  himself sick"), so claims are about object-onset expectation, and a
  strict set of unambiguous objective pronouns (him, them, me, us) is the
  declared sensitivity; an O conclusion is claimed only if the strict set
  agrees in sign.
- **Clause end**: "." (primary; commas, colons and newlines can precede an
  outstanding object); the punctuation set {".", ",", ";", ":", "!", "?",
  "\n"} as sensitivity.
- PREP (the prepositions of round 2's I set): reported by band.

**Populations.** M_obj on all 281 pairs. M_end on the **eligible pairs**:
the T verb flagged obligatory by both raters and the I verb forming a
complete bare clause ("The N V-ed." without its complement; Claude's
annotation `bare_ok` in `data/round6/preempt/i_frames.csv`, frozen before
any run; 29 I verbs need their complement, e.g. rely, consist, belong); 197
pairs. Sensitivities: all obligatory-T pairs; I verbs with a rare-transitive
flag excluded (any rater, both raters).

**Measures** (per pair, mean over nouns): M_obj = log P(O|T) − log P(O|I);
M_end = log P(.|I) − log P(.|T). Band contrasts as A2 (token-standardized
Tail − XTail over 2×2, 2×3, 3×2, 3×3 with weights 85 / 36 / 35 / 22; Head −
XTail), with side contributions.

**Side test (primary).** Per-verb OLS slopes of each side quantity on the
verb's Zipf, with the frame-specific log token count as covariate, on the
same pairs for both classes (all pairs for O, eligible pairs for "."):
- predicted side: log P(O | I) and log P(. | T) — **tracks frequency**
  (95% CI below 0), **reversed** (95% CI above 0), **flat** (90% CI within
  ±0.25 nats per Zipf unit) or unresolved;
- other side: log P(O | T), log P(. | I) — flat / slope ≠ 0 / unresolved;
- **side specificity**: the paired difference (predicted-side slope minus
  the other side's slope): **side-specific** if its 95% CI is below 0,
  **reversed** if above 0, otherwise not side-specific.
Sensitivities: token count as categories (1 / 2 / 3 / 4+); multi-token
Tail and XTail pairs only (common support); animate or inanimate nouns only
(inanimate subjects make some prefixes anomalous: "The report slept").
Bootstrap: pairs within band × nouns within animacy, 2,000 draws; identical
draws for every model.

**Against counts** (A1's counts; run after they are complete): log P(O | I)
~ C (A1's c_total) and C → C + log c_obj (c_obj = Σ count("P p the"), itself
an object-onset proxy); log P(. | T), eligible pairs, ~ C and C → C + log
c_end (Σ count("P p.")). Log token count is in every model. A1's CV rules;
predicted coefficients: C < 0, frame count > 0. Zero shares of the frame
counts by band.

**Released FreqBLiMP (descriptive).** Pythia-1.4B LP accuracy by band with
item CIs (already known: causative 79.3 / 86.7 / 78.8, transitive 75.3 /
75.5 / 69.2, inchoative 87.2 / 91.5 / 81.2, drop_argument 80.0 / 60.5 /
71.3, intransitive 67.5 / 73.6 / 69.8). Item-level association: each item's
critical verb (the one differing word) mapped to a D11 lemma via its
inflections; item margin regressed on the bad verb's side measure (log P(O |
I) for causative / transitive, log P(. | T) for inchoative / drop_argument /
intransitive) with band fixed effects and the bad verb's Zipf and log token
count as controls; verbs resampled within band. Analysed only with ≥ 3
mapped verb types in every band and ≥ 15 in total (checked: drop_argument
has 1 / 2 / 4 types and is descriptive only). It is labelled an
association: lexical, inflectional and good-verb differences are not
controlled, and it does not test the mechanism.

**Decision rules.** Band contrasts: A2's rules (bound 0.25 nats) on ΔM_obj
and ΔM_end; side labels only given a deficit (I-side / T-side when that
side's contribution has a CI above 0 and exceeds the other's, paired).
**The B3 prediction holds** for a frame if the predicted-side slope tracks
frequency and the side difference is side-specific (for O, with the strict
pronoun set agreeing in sign). "Unresolved" does not count as support.

**Predictions.**
- Object frame: log P(O | I) **tracks frequency**, **side-specific**; ΔM_obj
  Head − XTail **deficit, I-side**; token-standardized deficit or
  unresolved.
- Clause end (eligible pairs): log P(. | T) **tracks frequency**,
  **side-specific** (low confidence: an objectless obligatory transitive is
  rarely followed by a period even for rare verbs, so the effect may be
  small); ΔM_end Head − XTail deficit, T-side.
- Counts: C coefficients < 0 with "b better"; frame counts add (low
  confidence).
- Released items: negative associations for causative and transitive
  (descriptive).
- Suite (descriptive): the same side specificity at every size.

**Runs.** One TSUBAME job (with C4 and C5): the B3 prefixes (562 verbs × 24
nouns × 2 frames) on 1.4B and the six other suite models. Inputs
`data/round6/alternation/`; runner `run_next_sets.py`; analysis
`analyze_alternation.py`; report `b3_alternations.md`.

## C4. Classify-from-z restore

**Question.** Round-5 A4 restored rare single-token participles by moving
each unembedding row along Δh̄ to its class's Head mean (oracle: the class
label given) or to a value predicted linearly from the verb's site-8 z
(label-free; below half the oracle). Here the class is read from z: can the
verb's own early verb-class coordinate supply the label the oracle needs?

**Setup** (A4's, unchanged: model, Δh̄, edit rule, z = the mean over "She has
V", "He has V", "They have V" of the split-0 fold-scaled site-8 coordinate,
DAS verbs on their held-out fold; evaluations).
- Threshold θ: fit on A4's Head single-token verbs only (282 T, 30 I): the
  value maximizing balanced accuracy (Youden's J); ties → the midpoint of
  the first contiguous run of maximizing candidates.
- Assignment: T if z ≥ θ, else I. **classify** condition: move the row to
  the Head mean affinity of the assigned class (the oracle target of that
  class). The job recomputes z, checks it against A4's to 1e-4 and checks
  the frozen manifest (verb ↔ token, z, assignment).
- Sensitivity: θ fit on A4's Head plus D11's 17 extra Head singletons
  (reported, not edited).

**Populations.** Primary: A4's 32 rare single-token participles (Tail: 23 T,
9 I). Secondary (classification only): D11's Tail single-token verbs (24 T,
20 I; 28 not in A4's set, their z computed in the job).

**Measures.**
- Classification: per-class accuracy and balanced accuracy, 95% CIs with
  Head and rare verbs resampled within class and θ refit in every draw;
  Head in-sample and leave-one-out balanced accuracy (conditional on the
  existing z construction).
- Restoration (conditions unedited, oracle, classify): affected curated
  `passive_1` (primary) and `passive_2` margins, classify − oracle, share of
  the deficit closed; released passive items; specificity (curated Head
  pairs; unaffected FreqBLiMP items across all 67 paradigms; Pile-10k NLL)
  and NLL at the edited tokens on Pile-validation windows (A4's LM-loss
  evaluation, rerun because a misclassified verb gets the other class's
  target). Edited rows change every softmax normalizer, so specificity is
  measured, not assumed. Restoration CIs condition on the frozen θ,
  assignments and targets.

**Decision rules.**
- **z classifies rare verbs above chance**: balanced accuracy on the 32 has
  a 95% CI above 0.5 (known before declaration; see the disclosure).
- **Classify-restore improves rare passives**: affected curated `passive_1`
  margin change with 95% CI above 0 and ≥ 0.25 nats.
- **Achieves half the oracle** (interpreted only if the oracle improves, as
  it did in A4): Δ_classify − 0.5 Δ_oracle 95% CI above 0 (paired).
- **Matches the oracle**: 90% CI of Δ_classify − Δ_oracle within ±0.1 nats.
- Specific: A4's bounds (Head pairs ±0.05 nats; unaffected items ±0.5 pp;
  Pile-10k < +0.002 nats per token).

**Predictions.** Classification above chance (seen; not a prediction).
Given one misclassified I verb, classify-restore **improves** and
**achieves half the oracle**; it does not match the oracle if the
misclassified verb is in an affected pair (an I verb moved to the T mean
makes its bad passives more probable). Specific: yes. D11 Tail singletons
(secondary; partially seen: 16 of the 44 are in A4's set, 28 are new):
balanced accuracy above 0.5 (low confidence).

**Runs.** `run_restore.py` with the C4 manifest (`build_classify_manifest.py`),
conditions unedited / oracle / classify, LM loss included, extra z for
D11's Tail singletons. Analysis `analyze_classify.py`; report
`c4_classify.md`.

## C5. In-context preemption

**Question.** Does seeing a rare intransitive used intransitively, a few
times in the context, make the model reject it after "was" (statistical
preemption in context), beyond exposure to the word or to its exact form?

**Probe.** A1's frames after the context: "The N was" / "The N has"
(primary) and "The N has been" / "The N has" (secondary), the 24 A1 nouns
(none occurs in any context), no BOS. Δ_pass(V | arm, k) = log P(V | ctx +
was-prefix) − log P(V | ctx + has-prefix); raw log-probs per frame kept.

**Verbs.** Rare I: D11 Tail and XTail intransitives (incl. multi-token).
**Primary population**: those annotated dynamic (a natural -ing nominal and
progressive; 233 of 234; statives such as consist, belong, exist are
excluded). Head I (dynamic, 40): reference. Rare T control: 80 D11 T verbs
(40 Tail, 40 XTail; seed 17).

**Contexts** (k ∈ {1, 2, 4, 8}; k = 0 = no context, i.e. A1's scores).
Per-verb frames (`data/round6/preempt/i_frames.csv`; Claude's annotation,
frozen before any run): subject class (separate 8-noun pools per class:
person, animal, canine, large animal, object, kitchen item, event, weather /
conflict, show, quantity, substance, drink, light, fire, sound, machine,
vehicle, organic matter, protruding part, path, water body, place, plant,
document, time), plural flag (with correct plurals), optional complement
("relied on the map"), optional fixed subject ("the duck quacked"),
adjunct policy (none for e.g. "lasted for hours", "died"), bare_ok,
dynamic, theme (unaccusative-like sole argument). The generated sentences
were audited by reading them (the review found and the revision fixed
cases such as "The crisiss…", "The bell smoldered", "The path teemed with
fish").
- **preemptive**: k sentences "The S p (complement) (adjunct).";
- **exposure-only**: k nominal -ing mentions in eight carrier sentences
  ("Nobody noticed the V-ing.");
- **exact-form mention**: k metalinguistic mentions of the probed
  participle with its leading space ("Someone wrote glared on the board.",
  "The puzzle needed the word glared."): the exact form is repeated without
  a competing intransitive use;
- **filler**: k verbless fragments, each matched in length to its
  preemptive sentence;
- **other-verb**: the preemptive sentences of another intransitive of the
  same band with the same complement status and the closest length;
- **inflection mismatch**: "The S V-s …" (present) and "The S is V-ing …";
- **transitive control** (rare T): "The S p the O (adjunct).", with its
  filler.
Each verb's 8 sentence slots are put in a verb-specific random order, and
an arm's k-sentence context is the first k slots, so the dose does not add
the same subject or adjunct at the same k for every verb. Within a verb and
k, every arm is left-padded with verbless fragments to the longest arm's
length (± 1 token), so the arms compared have equal length and their own
sentences stay next to the probe. Checks: no be- or have-form directly
before any form of the target verb (except the "is V-ing" arm); no probe
noun in any context; joined tokenization = parts.

**Contrasts** (primary population; verb means over nouns; at k = 8):
E_arm = Δ_pass(arm) − Δ_pass(filler) for pre, exp, mention, other, vs, ing;
the paired differences E_pre − E_exp, E_pre − E_mention, E_pre − E_other,
E_pre − E_vs, E_pre − E_ing; the Head level (Head I under filler, k = 8),
the rare − Head gap G and the paired distance reduction D = G − |Δ_pass,rare(pre)
− Head level|; dose: slope of Δ_pass(pre) − Δ_pass(filler) on log2 k; T
control E_T = Δ_pass(trans) − Δ_pass(filler_t); raw log-probs (levels by
dose and k = 8 changes after was, has and has been). Bootstrap: verbs within
band × nouns within animacy, 2,000 draws. Breakdowns: band, token count,
complement; sensitivities: all rare I including statives; theme subjects.

**Decision rules** (bound 0.25 nats):
- E_pre **lowers** (95% CI below 0 and ≤ −0.25) / raises / negligible /
  unresolved;
- **beyond exposure**: E_pre − E_exp 95% CI below 0 and ≤ −0.25;
- **beyond exact-form mention**: E_pre − E_mention 95% CI below 0 and ≤
  −0.25;
- **verb-specific**: E_pre − E_other 95% CI below 0;
- **T control does not fall**: E_T 95% CI lower bound above −0.25;
- **in-context preemption is claimed** only if all five hold.
Secondary: **moves toward Head** = G 95% CI above 0 and D 95% CI above 0
(the share −E_pre / G is reported only when G is resolved above 0); dose and
mismatch comparisons reported with CIs. "Unresolved" never counts as
support. The theme-subject subset tests the stronger "affected-entity"
reading descriptively; for unergatives the context shows the verb's sole
argument as subject, which is contextual valency evidence rather than a
semantic competitor of the passive.

**Predictions** (the brief's hypothesis; low confidence, because an
in-context copy effect raises a repeated form's probability after any
auxiliary and can shrink Δ_pass toward 0 instead): E_pre **lowers**,
beyond exposure, beyond exact-form mention, verb-specific; the T control
does not fall → preemption claimed. Rare verbs move toward the Head level.
The mismatch arms lower Δ_pass less than the preemptive arm. Raw log-probs
rise after both auxiliaries in the preemptive and mention arms (copy), less
after "was" in the preemptive arm.

**Runs.** One TSUBAME job, Pythia-1.4B, fp32 (8,869 contexts × 72 probes ≈
640k sequences). Builder `build_preemption.py`, runner `run_preemption.py`,
analysis `analyze_preemption.py`; report `c5_preemption.md`.

## Order (B3, C4, C5)

1. Commit this part.
2. Builders (done before commit, tokenizer only): B3 prefixes, C4 manifest,
   C5 contexts.
3. TSUBAME: one job (C5, C4, B3 suite).
4. Analyses and reports; B3's count analyses after A1's counts.
