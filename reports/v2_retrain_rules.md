# Replacing v with v2, a DAS direction trained on more verbs (rules; declared Nov 24, amended the same day after Codex review, before any v2 run)

Runners: `scripts/run_das_v2.py` (finals), `scripts/run_das_v2_resolution.py` (verb-sample resolution). Downstream: the stored chain scripts, unchanged, pointed at a v2 root. Plan: `docs/paper_plan.md` (item 14, Settled 19), `docs/paper_outline.md`.

## Decision and why

v2 replaces v as the DAS instrument in every core-chain result of the paper, in all three models, unless the instrument gate below fails. The stored-v results move to the appendix as the original instrument, and every declared reading made with v stands as recorded (for example OLMo's T1 failure).

Reason: v was trained on 29 pairs, the ceiling of the October pool (Settled 14), and adding plain verbs moves the direction beyond the seed range at every rule site in every model (Settled 19: P vs v |cos| 0.72–0.92; stored-pair seed ranges 0.85–0.89 at Gemma 5 and 0.94–0.996 elsewhere). That shows v depends on its training verbs. It does not by itself show a retrained direction is better; the measures below test that. A direction estimated from 2.5× as many verbs is less tied to which verbs happened to train it, and the paper says that, not "better".

**Forking path.** P (Settled 19) was seen before this decision, including its 1.82× *by* at OLMo site 11. v2 is not P: it drops every new pair whose verbs occur in an evaluation set (below), so none of v2's results has been seen. The replacement is declared for all core-chain results at once, before any is rerun; no recipe change or further retrain follows the v2 results.

## v2 recipe

- **Candidate new pairs:** the plain pairs of the Nov 18 expansion batch (`expanded_pairs.csv`: batch `nov18`, not `das_train_overlap`, intransitive `das_class` plain): 62.
- **Exclusion (fixed now, one list for all three models):** a candidate pair is dropped if either verb's lemma or participle occurs as a cell value in any evaluation file of any model: passive-test items, construction items, filler-gap items, nonce-passive and nonce-cue prompts, and the classify-from-z files (`data/round5/restore/head_singletons.csv`, `data/round6/classify/*.csv`). This drops 18 pairs (lemmas browbeat, cache, certify, deface, desecrate, despoil, flinch, frolic, loiter, mosey, reemerge, relent, saunter, skulk, sleep, stage, tarry, tiptoe, waddle, waver), leaving **44**. The training file is fixed and committed: `data/das_v2/train_pairs_v2.csv` (29 stored + 44 new; sha256 in `data/das_v2/meta.json`, screen in `new_pair_screen.csv`). Each site job re-runs the disjointness check against that model's evaluation files on TSUBAME (including OLMo's nonce prompts, which are built there).
- **Training pairs:** for split k, fold f, the stored fold's training pairs (stored fold assignment of the 29 stored pairs) plus the 44 new pairs that pass this model's behavior filter (≥ 4 of the verb's items correct, both verbs), always in training.
- **Held-out active evaluation:** the stored held-out fold's items (29 stored pairs; 28 in Gemma), evaluated in the stored item order with the stored natural readouts, so held-out IIA is paired with v's. Uncertainty is over pairs.
- **Settings:** per site, the stored final's epochs, lr, batch size, n_cross, n_same, dtype, BOS and unembedding settings; seed 10000 + 100·split + fold; rank 1; 3 splits × 5 folds.
- **Controls in each final:** 100 random directions and their norm-matched versions per fold (generator seeded per fold from 900007, so no draws are paired with round 2's), and a shuffled-label direction trained on v2's training pool, saved (`shuffled_bases.pt`).
- **Evaluation sets are untouched:** v2's new training verbs occur in none of them, and v's 8 original training pairs keep their fold-matched bases. Every plan, gate and natural readout is the stored one.

## Sites

Every site with a stored final; the rule sites stay as declared for v (no v2 sweep, no re-picking):

| model | v2 sites | primary (final_strict) | declared rule sites | stored finals from |
|---|---|---|---|---|
| Pythia-1.4B | 0–14, 16, 17 | 17 | 8, 12 | `results/das_round2/final_strict*` (4, 6, 8, 10, 12, 14, 16, 17), `results/gemma/das/pythia_early/final_strict_site*` (0–3, 5, 7, 9, 11, 13) |
| Gemma-3-1B | 0–15 | 15 | 5, 8 (12) | `results/gemma/das/transfer/final_strict*` |
| OLMo-2-1B | 0–15 | 13 | 9, 11 (12) | `results/olmo/das/transfer/final_strict*` |

Output roots: `results/{pythia,gemma,olmo}/das_v2/` (smoke: `das_v2_smoke/`). Reports: `reports/{pythia,gemma,olmo}/v2/`.

## Is v2 a better instrument? (declared measures)

1. **Gate: held-out active IIA, non-inferiority.** Per rule site, v2 − v cross-IIA on the same held-out swaps, paired over stored pairs. v2 replaces v unless the 95% CI of the difference lies entirely below −0.02 at a declared rule site of that model; then that model keeps v and the paper says so. (Item 19's P was equal or higher; v2 has not been run.)
2. **Resolution (reported, not a gate).** At the declared rule sites (Pythia 8, 12; Gemma 5, 8; OLMo 9, 11), two directions are trained on disjoint random samples of n pairs from v2's training pool of split 0, fold f (n = 10, 20, 30; 3 draws per n and fold, folds 0–2), with the v2 recipe and the stored epochs. Measure |cos| between the two directions of each draw, each one's |cos| with the full v2 and stored v bases of that fold, and each one's held-out IIA on the stored fold. n = 20 is about v's per-fold training size (23 pairs); 30 is near the largest disjoint split of the pool (about 60–67 pairs). A |cos| that rises with n means a direction from more verbs is better determined, and the n = 20 value says how much a v-sized direction depends on which verbs trained it. Also descriptive: |cos| of v and v2 with u_md and with the cross-fitted LEACE edit vector at each rule site.
3. **Necessity (reported, not a gate).** Mean ablation along v2 vs v (the direction check's mean ablation: the passive *by* gap removed), paired on the same items, Pythia at the stored direction-check sites; Gemma and OLMo if the port of `run_das_direction_check.py` (site map, primary site) is done by Dec 1.

## Checks (hard: a failure stops that site)

1. Pairs: the training file's sha256 equals `data/das_v2/meta.json`, and its stored rows equal `data/das_round2/train_pairs.csv`.
2. Stored items equal and stored folds reproduce at every site.
3. Recompute: the stored v's held-out swaps recomputed in the same process, patched M and the natural M of base and source: median |Δ| ≤ 0.01 and max ≤ 0.5. *Amendment (Nov 24, after the first finals jobs, before any v2 result was read):* the stored finals kept `heldout_swaps.csv.gz` only at Pythia 8, 12, 17, Gemma 5, 8, 12, 15 and every OLMo site. At the other sites the recomputed per-fold summary (cross and same IIA, median patched fraction) is compared with the stored `summary.csv`: |Δ IIA| ≤ 0.01, |Δ median fraction| ≤ 0.02. The natural pass does not depend on the site, so its check at the sites with swaps covers the model.
4. Disjointness of the new training pairs (v's stored pairs are protected by their folds, as before), re-run in each site job on that model's mandatory file list (the job checks every file exists): zero hits.
5. Layout: each written final has the 15 keys r1_s{0..2}_f{0..4} (smoke: its own root only), heldout_swaps covering every (split, fold) with one-to-one (base, src) keys and finite values.
6. Downstream: each rerun's natural (unpatched) readouts equal the stored run's by prompt (max |Δ| ≤ 1e-6), as in `jobs/umd_tr.sh`; filler-gap gates equal the stored gates except the base-form gate along the direction.

## What is rerun with v2 (per model unless stated)

| result | v2 run |
|---|---|
| Held-out active IIA and controls | the v2 finals' `summary.csv` |
| Passive test at rule sites (T1) and depth curve at all v2 sites | `run_passive_test.py` with v2 site dirs, same prompts / plan / folds |
| OR / TC constructions (T2), in-range | `run_construction_test.py`, `run_inrange_patch.py` with v2 `bases_rank1.npz` |
| Filler-gap, embedded and matrix (T3) | `run_fillergap.py` natural / gates / patch (fp32, `_nondas` and `_all`) with v2 bases |
| Staging S1 and the split edit | `run_split_edit.py --root` v2 root, all sites; analysis compares against the v2 passive run |
| Nonce passive and cues | `run_nonce_passive.py`, `run_nonce_cues.py` with v2 bases, stored sites |
| Direction check and mean ablation | Pythia, v2 root (Gemma / OLMo if ported, item 3 above) |
| Expanded eval: new-verb transfer | v2 and v on the same pairs: the 126 originals and the new pairs with no v2 training verb (new plain pairs outside v2's 44, new prep pairs); IDs frozen in `results/{m}/das_v2/expanded_eval_pairs.csv` before the run. The stored 74-pair result stays attributed to v |
| Classify-from-z edit (R6-C4) | Pythia: z along v2 at site 8 for the same verbs, a new manifest built with the declared Head-only θ procedure, then unedited / oracle / classify scored against it |

Not rerun: reverse DAS (Settled 15) stays declared against v; its ratio to v2 is descriptive only.

**Margins.** Every T1 / T2 / T3 / S1 and R-A1 / R-A2 reading is reported three ways: each instrument under its own δ (v2's δ recomputed from v2's held-out active effect by the stored rule, `analyze_passive_test.py delta` for every analyzed site), both instruments under the common stored δ_v, and the raw paired effect with *by* / natural gap. The paper's primary reading for v2 uses its own δ, declared here. The u_md readings declared with δ_v stand; their δ_v2 version is labelled a recalibration.

**Disagreements and failures.** Where v and v2 readings differ, both are reported where the claim is stated. If v2 fails a criterion v passed, the claim is narrowed to what holds with the primary instrument (v2) and v's pass is reported beside it; if v2 passes where v failed (for example OLMo T1), the paper reports a v2 pass with v's failure beside it, never v2's pass alone. A cell whose checks fail or whose run misses Dec 11 is marked invalid or not available. R-A1 / R-A2 with no eligible site (no full-patch RISE) are reported as not applicable.

## Not rerun now

The Pythia site-8 mechanism results built on the stored v (R2-mech, R2-dose, R3-B7, R3-A2, R4-D11, R4-E13 / E13T, R5-B5, R5-B6, R5-C9, R5-A4) stay on v and are labelled as a case study of the stored site-8 intervention. They are not claimed to carry v2's effect unless the C4 instrument and mediation checks (`docs/paper_outline.md`) are run on v2. Whether to port them is decided by Dec 4, on cost. Training-free results (behavior, u_md transfer, all LEACE) do not use v.

## Cost

v2 finals: 49 site-finals (Pythia 17, Gemma 16, OLMo 16) in jobs of about 4 sites on `gpu_h`; resolution: 6 rule sites × 54 short trainings. Then per model the chain: passive test and depth curve, constructions and in-range, filler-gap, split edit, nonce, about 4 h of wall clock per model with stages in parallel.

## Amendment: the expanded set is the paper's evaluation set (Nov 25, before any v2 expanded run)

Decision (user, Nov 25): the paper evaluates transfer on the expanded set; the 126-pair readings above become the
reproduction check and are not presented separately.

- **Pairs (frozen): `data/das_v2/expanded_eval_pairs.csv`** (sha256 4a9229d7…). v2 is evaluated on every expanded pair except its 44 new training pairs:
  - all 126 originals. The 8 originals that contain one of v's stored training verbs are fold-protected, as for v: each is patched with the basis from the fold that held it out.
  - 98 new pairs: 30 plain and 68 prepositional.
  - Total 224 pairs; plain (primary) 59 + 30 = 89.

  The 44 are excluded in all three models, including pairs a model's behavior filter dropped from its training, so there is one list. The 44 training pairs' readings are computed and may be shown only as in-sample.
- **v beside v2:** the stored expanded v runs (`results/{m}/expanded_eval/passive_v`, `constructions_v`) use the same plan, prompts and draws. v is read on the same 224 pairs; v's 268-pair reading stays attributed to v.
- **Runs:** `run_passive_test.py` on the stored expanded plan with v2 site dirs at every v2 site (T1 at the rule sites, the S1 depth curve), and `run_construction_test.py` on the expanded constructions with v2 bases at the stored expanded sites (T2). Folds, prompts and batch settings are as in `jobs/item5.sh` / `item5m.sh`. Check: the natural (unpatched) readouts equal the stored expanded run's (max |Δ| ≤ 1e-6).
- **Not on the expanded set:**
  - T3 (filler-gap): its verb gates and in-range targets are per verb and declared on its own set.
  - Nonce: no real verbs.
  - (Split edit: see the next bullet.)
- **Split edit on the expanded set (decided Nov 25 after the criteria audit, before any expanded result was read):**
  - v2 only, on v2's evaluation plain items (89 pairs; the 44 training pairs are dropped from the items file);
  - sites: Pythia 4, 6, 8, 12, 17; Gemma 5, 8, 12, 15; OLMo 9, 11, 12, 13;
  - read by the audit's dissociation contrast (`reports/criteria_audit.md`, claim 5);
  - check: full-direction D on the same items equals the expanded v2 passive run's.
- **Readings:** the criteria are reviewed against the claims they support (`reports/criteria_audit.md`) before any expanded v2 result is read. That review may change a margin or a criterion for every instrument at once, never for one instrument or model.
