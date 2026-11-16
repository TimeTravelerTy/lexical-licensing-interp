# Transitivity mechanism paper: plan to ARR Jan 4

Started Nov 16, 2026; revised the same day against the repo (branch `paper`). This is the working guide for the rest of the project, and it is meant to change. Update the status table as results land, strike items that stop holding, and note which decision rule fired. Each experiment still gets its own plan file with rules and predictions, committed before it runs; this file sits above those.

## Goal and core claim

Submit a long paper (8 pages) to ARR by **Jan 4, 2027 AoE** (Jan 5, 20:59 JST) for ACL 2027, with Pythia-1.4B as the main model and Gemma-3-1B and OLMo-2-1B as the cross-family replications.

**Claim.** Language models compute verb transitivity as one shared, shallow feature at the verb, roughly "an NP follows this verb", set from both lexical identity and context. Construction-specific components then read it. Rare verbs get the feature right, and their deficit sits on the output side.

**Framing: H1/H2 answered as a split, not a winner.** The intro sets up lookup (H1) against computation (H2). The answer:

- The *format* is shared and computed. One direction is read by every construction, it is set from context for nonce verbs, and it encodes something shallower than argument structure.
- The *value* for a known verb is partly retrieved per verb. That is the part only an in-sample eraser removes.

With this framing, the necessity numbers answer the intro question instead of qualifying it.

**One number for "how shared"** (abstract): an eraser fit on other verbs removes about 41–45% of a new verb's *by* gap. An eraser fit on the verbs themselves removes almost all of it.

## Status per claim

Evidence is on branch `paper` (merged Nov 16 from `olmo-replication`, `leace-necessity` and `das-direction-check`). Paths are relative to `reports/`.

| Claim | Pythia-1.4B | Gemma-3-1B | OLMo-2-1B | Evidence (Pythia / Gemma) | Left to do |
| --- | --- | --- | --- | --- | --- |
| Shared direction transfers across 5 constructions (passive, OR, TC, embedded and matrix wh) | done | done | sweep only (frozen site 13) | R2-ptest, R4-A2, R4-B, R5-C8 / `gemma/transfer.md` | OLMo final DAS + transfer; u_md transfer next to v in all three |
| Shallow, context-derived class (nonce onset ~site 6, NP-driven) | done | done | not run | R4-C8/C9, round 3 nonce / `gemma/nonce/summary.md` | OLMo nonce, sites 0–8 |
| Abstract → surface staging | **at risk**, likely survives with u_md (see Settled 2) | **at risk** | not run | R2-depth, `pythia/das_direction_check.md` | Split-edit test, redraw with u_md |
| Necessity: shared + lexical (LEACE) | done | not run | not run | `pythia/leace_necessity.md`, `leace_permutation.md`, `leace_learning_curve.md`; R5-B5 | Rank-k erasure (Pythia); LEACE on Gemma, OLMo |
| Construction-specific readers | done | appendix if time | no | R3-B7, R4-D10/D11, R5-B6, R5-C9, R5-B7 | Writing only ("components", not "circuit") |
| Rare verbs: feature right, output-side deficit + edit | done | not run | not run | R6-C4 (31/32 rare singletons classified from z), R5-A1/A2, R5-A4 | Second model, or scope it to Pythia |
| *by* transfer passes the illusion check | done | not run | not run | `pythia/das_direction_check.md` | Rerun the direction check per model |
| Training dynamics (emergence order) | behavior done (R6-A6, R7-A8: 16 checkpoints); verb-site measures not run | n/a (no public checkpoints) | not run | `round6/a6_checkpoints*.md`, `round7/a8_trajectories.md` | Prediction, then 16 checkpoints per model |

Not committed anywhere: the one-off v / u_md side split ("about three-quarters of the v mean-ablation effect on the *by* gap comes from bad passives gaining *by*").

## Settled (Nov 16)

1. **The necessity numbers hold up, with wide CIs.** Cross-fitted (5-fold) scrub 0–12 leaves 0.59 [0.35, 0.70] of the natural *by* gap. LOO leaves 0.55 [0.29, 0.67], and the learning curve is flat from n = 75 to 125 by point estimate. The CIs at n = 75 and 100 are wide ([0.03, 0.72]), so quote LOO and cross-fitted, not the curve, for the shared share. In-sample scrub 0–12 leaves 0.04 [−0.46, 0.28]. That result is credible because of the permutation control: all 20 label permutations leave 0.96–1.00, and the true-label CI lies below their 5th percentile. Always print the CI next to 0.04.
2. **The direction check previews the split-edit outcome (Pythia).** On passives, u_md keeps most of v's *by* effect but little of its O effect:
   - D(by) ratio u_md/v: 0.88–0.93 at sites 6 and 8, 1.12 at site 12.
   - D(O) ratio: 0.34–0.36 at sites 6 and 8, 0.15 at site 12 (v +1.17 RISE, u_md +0.17 NO RISE).

   The off-axis part of v probably carries the passive O rise. If so, staging survives redrawn with u_md, with the abstract range running through site 12 and the surface stage at 17. The split-edit test still has to confirm it, with the prediction written down first.
3. **u_md is the weaker causal instrument on actives.** Its held-out active IIA is 0.30–0.41 of v's at sites 4–12, and its Δ log P(O) is 0.59–0.70 of v's. Use u_md as co-primary for *direction* claims (passive *by* transfer, staging) and report ratios to v; do not present its IIA as a replacement for v's. Mean ablation: v removes 0.37–0.41 of the passive *by* gap at sites 4–12, u_md 0.27–0.37, random directions about 0.
4. **Gemma ties input and output embeddings** (`scripts/model_arch.py`; `--fp32-unembed` unties the head). In Gemma, "the deficit lives in the output rows" means editing rows that are also input embeddings. Port the output-side edit on the untied head only, and say so.
5. **The LEACE / u_md cosine gap is probably a frame difference, not a puzzle.** For a binary label, a rank-1 LEACE edit vector is proportional to Cov(x, z), the class mean difference on its fit items. Those items were passive `was` prompts, while v and u_md come from actives. So |cos(u_md, edit)| of 0.29–0.49 (|cos(v, edit)| 0.19–0.56) mostly measures cos(u_md in passives, u_md in actives). This turns a loose end into a question for the "one shared feature" claim (task 3 below).
6. **Single-token and multi-token verbs separate sharply at site 20** (cross-fitted edit |cos| 0.69 vs 0.03). So the single-token × frequency confound is a concrete limitation, not a hypothetical one.

## Work, ordered by risk

Run whatever could break the claim first, then polish.

1. **Split-edit test, then the u_md staging curve.** For each v patch, δ = v·h_s − v·h_b. Run the full edit (δv), the on-axis piece (δa·u_md) and the off-axis piece (δb·ŵ), and report D(by), D(O) and Δ log P(O) for each. Compare the size of the ŵ push with the natural spread of ŵ coordinates. Pythia first, building on `scripts/run_das_direction_check.py`; then Gemma and OLMo. The prediction to declare is Settled 2.
2. **OLMo final runs.** Run final DAS, the passive test at all 16 sites, nonce at sites 0–8 and transfer at the rule-picked sites. This fills the empty column and settles relative depth vs absolute layer. Queue early, because the bottleneck is queue time, not code.
3. **Frame invariance of the class axis** (cheap; it bears on "one shared feature"). Per site: cos(u_md^active, u_md^passive); a one-line check that the LEACE edit vector equals the passive mean difference; transfer with u_md^passive vs u_md^active. A high cosine supports one feature. A low cosine combined with good transfer means the frames share a readable direction but not the class mean, which needs to be said in the paper.
4. **Rank-k erasure** (closes the main objection to "lexical"). Rank-1 LEACE removes one direction per site by construction. A reviewer can read the ~0.55 residue as shared class information in a few more directions rather than verb-specific information. The held-out probe still reads class at 0.67–0.78 after cross-fitted erasure, so the residue is real. Run cross-fitted rank-k erasure (iterated LEACE or INLP, k = 1…8) at site 8 and for scrub 0–12, with a random-direction reference.
5. **Replication criteria, declared once per claim.** State what counts as replicated in Gemma and OLMo before their remaining runs: sign, CI excluding 0, the site the declared rule picks, and whether depth is mapped by relative depth or absolute layer. The three-model table then reads as pass/fail against stated criteria.
6. **Training dynamics on 16 checkpoints per model.**
   - Write the prediction first (candidates below).
   - Pythia: the same 16 checkpoints as R6-A6 (step 64 to 143000, already used for behavior). Run u_md, the held-out probe and the passive *by* readout at the verb at all 16, plus DAS at all 16. DAS is about 1 min per site for 5 folds, so 16 checkpoints × a few sites is cheap.
   - Code: add `--revision` to `run_das_round2.Runner` (OLMo `rules.md` notes it is missing).
   - OLMo: 16 checkpoints matched to Pythia's by tokens seen, drawn from the 37 early-training checkpoints and the 195 stage-1 branches.
   - Check the TSUBAME HDD cache before downloading anything.
7. **LEACE on Gemma and OLMo.** In-sample, cross-fitted and the permutation control, reusing the Pythia pipeline. "Shared + lexical" is too central to rest on one model.
8. **Side splits.** Commit the v / u_md side split, then run the same split on the LEACE outputs. This tests the suppression framing below.
9. **Robustness.** DAS seed variance and random-subspace baselines for every reported number, plus the single-token × frequency confound (Settled 6). Pythia-6.9B transfer goes last.
10. **Writing, starting now.** Write the outline, figure list and methods now; results go in as each piece lands.
11. **Embedded wh in Pythia.** Pythia embedded wh is mixed at site 8 and surface at site 6 (R5-C8); Gemma's in-range result is mixed or unresolved. Hard stop Nov 18, then it goes to limitations.

### Dynamics prediction candidates (pick before running)

- **Feature before use, rare = frequent.** The class feature at the verb (u_md separation, held-out probe) appears at about the same checkpoint for rare and frequent verbs, while the rare-verb behavioral deficit (R6-A6: from step 256, peaking around steps 1000–3000) persists. Then the deficit is not a late-feature problem, which links "rare verbs get the feature right" to emergence. This also differs from Desai & Nair, who track sharing, not rare vs frequent onset of a feature vs its use.
- **Shared before lexical.** The cross-fitted (shared) share removed appears before the in-sample minus cross-fitted (lexical) share.
- **Abstract before surface.** The u_md *by* effect appears before the late O effect, following COMPASS.

## Possible framing: class information suppresses *by*

About three-quarters of the v mean-ablation effect on the passive *by* gap comes from bad passives gaining *by* (uncommitted one-off). If the LEACE gap also closes mostly from the bad side, class information mainly suppresses *by* after intransitives, and the default after a passive participle is to expect *by*. That ties into the output-side result: rare intransitives being under-rejected is weaker suppression. The test is task 8.

## Paper shape

Eight pages for six claims plus dynamics is tight. The main-text spine is one figure per claim:

1. Transfer: 5 constructions × 3 models, v and u_md side by side.
2. Shallow: nonce onset by depth.
3. Shared + lexical: LEACE in-sample vs cross-fitted (with rank-k if it lands).
4. Rare verbs: feature right (classify-from-z), output side wrong (restoration).
5. Emergence order.

Staging and the construction-specific readers get a paragraph each, with figures in the appendix, unless the split-edit test is clean enough for staging to carry Figure 1. If more has to shrink, shrink the edit section (sensei asked for dynamics).

## Related work that shapes the framing

- [Kumon & Yanaka](https://arxiv.org/abs/2604.22166) (ACL 2026) find that activation-patching mechanisms hold out of distribution while DAS overfits narrow linguistic data. Two defenses. Cross-construction transfer is itself an out-of-distribution test (trained on actives, tested on four other constructions). And u_md is training-free, so report its transfer next to v's.
- [Desai & Nair](https://preview.aclanthology.org/ingest-acl/2026.findings-acl.1737/) (Findings ACL 2026) run DAS over BabyLM checkpoints and call filler-gap sharing "shared, yet item-sensitive". That is close to "shared + lexical", so cite it. This paper quantifies the split and places the item-sensitive part (rare verbs, output side).
- [COMPASS](https://arxiv.org/html/2511.20910v2) reports semantic-role circuit structure stabilizing before it becomes functionally needed. It's the nearest training-dynamics comparison.
- [Boguraev, Potts & Mahowald](https://arxiv.org/pdf/2505.16002) (EMNLP 2025) used transitive/intransitive prediction as a control task. This paper makes their control the object of study.

## Cheap analyses worth adding

- **Alternating verbs as a graded test.** They were excluded by design. Projecting them onto u_md and correlating with corpus transitivity bias (Pile counts from R6-A1) would show the feature is graded. No training needed.
- **Gemma Scope 2 lookup.** Take the cosine of u_md with SAE decoder latents at the same sites. This gives unsupervised support without the attribution-graph work saved for the thesis.
- **More verbs for the training-free analyses.** 126 pairs caps the LEACE learning curve at "flat over the last 50 pairs". u_md, behavior and LEACE don't need the DAS splits.

## Timeline and decision rules

Freeze experiments on **Dec 4** and send the full draft to sensei by **Dec 11**. Coding is fast. Queue time on TSUBAME, reading results and sensei's turnaround are not.

| Week | Experiments | Writing |
| --- | --- | --- |
| Nov 16–22 | Split-edit + frame invariance (Pythia); side splits; queue OLMo finals; declare replication criteria and the dynamics prediction; embedded wh stop (Nov 18) | Outline, figure list, methods |
| Nov 23–29 | OLMo transfer / nonce / split-edit; Gemma split-edit; rank-k erasure; LEACE on Gemma, OLMo; checkpoint runs | Results sections as they land |
| Nov 30–Dec 4 | Dynamics analysis; seeds and random-subspace baselines; **freeze Dec 4** | Intro, related work |
| Dec 5–11 | — | Full draft to sensei **Dec 11** |
| Dec 12–Jan 4 | 6.9B transfer only if time | Revisions, appendix, checklist, submit **Jan 4 AoE** |

The week rows are a proposal; the original timeline chart did not survive the export. Fixed dates: Nov 18, Dec 4, Dec 11, Jan 4.

**Decision rules**

- **Off-axis piece carries the passive O rise:** redraw staging with u_md (or the on-axis piece) in every model, and report v as an upper bound.
- **On-axis piece carries it too:** keep staging as drawn with v, and note u_md's later boundary.
- **OLMo transfer fails:** the paper becomes "Pythia and Gemma, with a partial third-family check".
- **Rank-k residue keeps falling with k:** reword "lexical" as "shared, low-rank plus verb-specific", and report where it plateaus.
- **cos(u_md^active, u_md^passive) is low:** drop "one feature" in favor of "one direction every construction reads", and keep transfer as the evidence.
- **Training dynamics not clean by Dec 4:** report emergence order for the measures that are in, on one model if needed.
- **Embedded wh not solved by Nov 18:** it goes in limitations.
- **New idea after the freeze:** it goes in the thesis file.

## Admin checklist

- [x] Merge the result branches into one paper branch (`paper`, Nov 16)
- [ ] OpenReview profiles for both authors. Check the ARR CfP for the ORCID requirement and reviewer registration for all authors.
- [ ] Anonymized code link (repo scrubbed of names, TSUBAME paths and lab references)
- [ ] Cite FreqBLiMP in the third person
- [ ] Limitations section: embedded wh, single-token × frequency confound (Settled 6), English only, models of 1–7B
- [ ] Responsible NLP checklist

## Sources

- [ACL 2027 call for papers (via AI Weekly)](https://aiweekly.co/node/16101)
- [Kumon & Yanaka, Fine-Grained Analysis of Shared Syntactic Mechanisms in Language Models](https://arxiv.org/abs/2604.22166)
- [Desai & Nair, Filling in the Mechanisms](https://preview.aclanthology.org/ingest-acl/2026.findings-acl.1737/)
- [Emergence and Localisation of Semantic Role Circuits in LLMs](https://arxiv.org/html/2511.20910v2)
- [Boguraev, Potts & Mahowald, Causal Interventions Reveal Shared Structure Across English Filler-Gap Constructions](https://arxiv.org/pdf/2505.16002)
- In the repo: `reports/results_index.md` (R-numbers above), `reports/pythia/` (LEACE, direction check), `reports/gemma/`, `reports/olmo/`
