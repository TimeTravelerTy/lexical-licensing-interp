# Passive test of the round-2 direction: declaration (2026-10-06)

Written and committed before any passive has been patched or projected onto
a round-2 direction. Everything below is fixed now. The only passive data
seen so far are the unpatched readouts of the prep tasks (`README.md`, tasks
0-5).

## Question

On held-out actives, the rank-1 direction d at site 17 controls "object
next" (`das_round2_results.md`). The passive test asks which variable d
carries:
- **surface**: "an object comes next";
- **abstract**: "this verb takes an object".

Late sites tend to hold next-token features, so the frozen site 17 already
leans toward the surface reading. An abstract variable, if there is one, is
more likely to show up earlier.

## Sites

| Role | Site | Layer output | Epochs | Rank | Sweep IIA at those epochs |
|---|---:|---:|---:|---:|---:|
| **Primary** (frozen 2026-10-05) | 17 | 16 | 1 | 1 | 0.979 |
| Declared secondary | 12 | 11 | 2 | 1 | 0.917 |
| Declared secondary | 8 | 7 | 3 | 1 | 0.794 |

- **Configs:** `results/das_round2/frozen_config_site{8,12}.json`.
- **Recipe for sites 8 and 12:** the same as the frozen run
  (`run_das_round2.py final`):
  - the 29-pair strict set;
  - 5 folds × 3 split seeds;
  - the same seeds, learning rate, batch size and swap mix;
  - the same controls (100 random rank-1 subspaces, raw and norm-matched,
    and shuffled-label DAS).
- **Epochs:** the frozen epoch rule applied to each site's sweep curve
  (smallest epoch within 0.01 of the site's best mean held-out IIA). The
  best values are 0.803 at site 8 (epoch 8) and 0.921 at site 12 (epoch 5).
- **Rank:** fixed at 1. No sensitivity (named-set) run at the secondary
  sites.
- **Why 8 and 12:** they are the points where held-out IIA first reaches
  about 0.8 and about 0.9. They were chosen from the actives-only sweep;
  no passive result exists.
- **Held-out active metrics** are reported for each site in the same table
  as `das_round2_results.md`, before any passive number.
- **TOST bound per site:** δ_s = 0.2 × the held-out Δ log P(O)
  (intransitive base ← transitive source, mean over held-out pairs) from
  that site's final run.
  - Site 17: δ = 0.5716 (frozen).
  - Sites 8 and 12 get their own δ by this rule, and are also reported
    against 0.57.
- **Multiplicity:** site 17 carries the primary claim. Sites 8 and 12 are
  secondary analyses, reported as such, with no correction across sites.
  The depth pattern (below) is read from the three sites' classifications.

## Evaluation items

- **Passives:** "The N was <participle>", with the readout at the
  participle's last token. They come from two sources:
  - `data/passive_das/passives.jsonl`: the 126 original pairs in curated
    `passive_2` contexts of *other* pairs (15,708 items, good and bad
    prompt each);
  - the 19 new eval pairs (`data/verb_expansion/eval_pairs.csv`) in all
    126 curated contexts.
- **Same verbs in the active frame:** "<Subj> has/have <participle>", with
  the 7 round-2 subjects. This is the reference for step 1.
- **Primary population:** bad verb of class `plain`
  (`data/das_round2/intrans_classes_final.csv`). That is 59 original pairs
  (Head 9, Tail 25, XTail 25) plus 5 new pairs (Tail 3, XTail 2): **64
  pairs**. Results are reported pooled (equal weight per pair) and by band.
- **Cross-fitting:** the 8 `orig_head` training pairs are also Head
  passive pairs.
  - An item from one of these pairs uses, in each split, only the fold in
    which its pair was held out.
  - Any other item uses, in each split, one fold drawn uniformly at random
    per (item, split), with seed 17.
  - Every item is therefore evaluated under 3 bases (one per split).

## Step 1: projection of natural participles onto d (no patching)

- **Projection.** For every passive prompt (good and bad) and every active
  reference prompt, take the residual state at the last token at sites 8,
  12 and 17 and project it onto each fold basis.
- **Sign.** Each basis is oriented so that the fold's *training*
  transitive items project above its training intransitive items.
- **Scale.** Per basis, z = (p − μ_I) / (μ_T − μ_I).
  - μ_T and μ_I are the mean projections of the fold's *held-out*
    transitive and intransitive active items.
  - So z = 0 is the active intransitive level and z = 1 the active
    transitive level.
  - Raw projections are reported too.
- **Cross-fitting** as above: training-pair items average over the 3 bases
  in which their pair was held out; all other prompts average over all 15.
- **Separation, good vs bad**, by band, for:
  - the primary population;
  - all pairs;
  - prep_object pairs.
- **Metrics:**
  - mean difference in z (good − bad), with a two-way cluster bootstrap
    over verb pairs (within band) and contexts (2,000 draws, seed 17);
  - AUC over items;
  - AUC over verb means;
  - paired win rate (good > bad in the same context).
- **Active reference:** the same metrics for the same verbs in the active
  frame (subjects in place of contexts), and the mean z of good and bad
  passives against good and bad actives.
- **Reading.** If good and bad passives do not separate along d, the
  variable may not be computed in passives. The step-2 result is then read
  with that in mind. This step does not change anything in step 2.

## Step 2: passive transfer

**Interchange** at site s, on the participle's last token:
h ← h + ((h_src − h)·d) d.
- d is the basis assigned to the item for that split (cross-fitted).
- h_src is the donor's state at the same site and position.

**Donors.**
- For item × split: the held-out pairs of the assigned fold, excluding the
  item's own pair.
- Each such pair contributes its transitive and its intransitive verb once.
- The donor prompt is "<Subj> has/have <participle>", with the subject drawn
  at random from the 7 (seed 17). The transitive and intransitive verb of a
  donor pair share the drawn subject.
- Donor verbs are therefore never in the training set of the basis they are
  used with, and transitive and intransitive donors come in matched pairs.

**Conditions, on both the bad and the good passive of every item:**

| Condition | Donor |
|---|---|
| T | transitive active donor |
| I | intransitive active donor |
| same verb | the base verb's own active, "<Subj> has <its participle>" (voice change only) |
| passive swap | the other verb's passive in the same context (good → bad base, bad → good base) |
| matched donors (bad bases only) | verbs of the 15 participle-matched pairs (`train_pairs_participle_matched.csv`) not in the basis's training pairs, each once with a random subject |

**Readouts** at the participle's last token, as Δ = patched − unpatched:
- log P(" by");
- log P(O): the training object-start set;
- the split of O into determiners, pronouns and reflexives;
- log P(" the") and log P(" him");
- log P(".");
- log P(I);
- M.

**Primary contrast.**
D(b) = mean ΔR over T donors − mean ΔR over I donors, pooled over the 3
bases. It is averaged within pair, then across pairs.

**Inference.**
- Three-way cluster bootstrap over:
  - base verb pairs, resampled within band;
  - contexts;
  - donor verb pairs.
- 2,000 draws, seed 17.
- The 95% percentile CI is used for "rise"; the 90% CI for TOST.

**Classification of D, per readout (O primary; " by"; "."):**
- **RISE:** the 95% CI lies above 0 and the point estimate is ≥ δ_s.
  **FALL** is the mirror image.
- **NO RISE:** the 90% CI lies within ±δ_s (TOST).
- **Otherwise:** unresolved, reported with its CI.

**Outcome, per site:**

| O (object start) | " by" | Reading |
|---|---|---|
| RISE | not RISE | d = "object next" (surface) |
| NO RISE | RISE | d = "takes an object" (abstract) |
| RISE | RISE | mixed: not fully separated |
| NO RISE | NO RISE | nothing moves: d is specific to actives |
| any other combination | | unresolved |

**Depth pattern** across sites 8, 12 and 17:
- surface everywhere;
- abstract at 8/12 and surface at 17: the model converts "takes an object"
  into "object next" across depth;
- abstract everywhere;
- or as observed.

**Controls**, reported alongside D with the same bootstrap:
- **Voice-change baseline:**
  - E[ΔR | I → bad passive];
  - the same-verb version (the bad verb's own active → its passive).
- **good → good:**
  - E[ΔR | T → good passive];
  - the same-verb version.
- **D on good bases:** the contrast with intransitive donors patched into
  good passives.
- **Passive swap:** good passive → bad passive in the same context, and the
  reverse. This is a within-voice check that the site and readouts can carry
  the passive good/bad difference at all.

**Scales.**
- D(" by") is also given as a fraction of the natural good − bad gap in
  log P(" by") on the same items.
- D(O) is also given as a fraction of the site's active effect.

**Separate groups** (same contrast and bootstrap; descriptive, no outcome
classification):
1. the 6 new Head eval pairs (bad verbs all `prep_object`);
2. `prep_object` bad verbs by subtype (`pseudo_passive_ok`,
   `pseudo_passive_bad`), original + new, by band;
3. `contaminated_bad` (7 pairs), and the primary population pooled with
   them;
4. bad-side-high *by* pairs (11 bad verbs, plain and prep_object);
5. *by* reliability splits of the original pairs: reliable, negative,
   n.s.-positive;
6. familiarity robustness: D with the participle-matched donors only;
7. sensitivity rows for the primary population:
   - original pairs only;
   - without bet/appear;
   - the 8 cross-fitted `orig_head` pairs vs the rest.

**Not done here** (step 3, later): projections of the prep_object subtypes,
*ensue*, contaminated_bad and bad-side-high onto d. They are only
interpretable once the reading of d is known.

## Implementation

- `scripts/build_passive_test.py`: items, prompts and the full patch plan
  (`data/das_round2/passive_test/`). The plan is regenerated
  deterministically; its content hash is in `plan_meta.json`.
- `scripts/run_passive_test.py`: unpatched pass, projections and patches
  (GPU). It checks that a self-patch reproduces the unpatched readout and
  that it recomputes a sample of each site's held-out active swaps.
- `scripts/analyze_passive_test.py`: `delta` (δ_s for sites 8 and 12) and
  `passive` (steps 1 and 2).

## Order

1. Commit this declaration and the site 8/12 configs.
2. Train sites 8 and 12 (`run_das_round2.py final`). Report their held-out
   active metrics and δ_s.
3. Run the passive projection and transfer at sites 8, 12 and 17.
4. Analyse with the rules above. Report site 17 as primary and sites 8/12
   as declared secondary analyses.

## Secondary sites: training results (2026-10-06, before any passive evaluation)

Jobs 8916730 (site 8) and 8916732 (site 12), commit `64a0bb0`.
Reports: `das_round2_results_site8.md` and `das_round2_results_site12.md`.
Intransitive base ← transitive source, held-out pairs, CI over pairs:

| Site | IIA | Gap fraction (M) | Δ log P(O) | δ_s = 0.2 × Δ log P(O) | Random (norm-matched) IIA | Shuffled-label IIA |
|---:|---|---|---|---:|---:|---:|
| 8 | 0.76 [0.65, 0.85] | 0.75 [0.68, 0.82] | 2.46 [2.12, 2.79] | **0.4914** | 0.02 | 0.20 |
| 12 | 0.87 [0.80, 0.94] | 0.86 [0.80, 0.92] | 2.64 [2.30, 2.97] | **0.5281** | 0.02 | 0.24 |
| 17 | 0.97 [0.94, 0.99] | 1.06 [1.01, 1.11] | 2.86 [2.52, 3.19] | **0.5716** | 0.02 | 0.34 |

- At both secondary sites DAS beats the 95th percentile of norm-matched
  random subspaces in 15 of 15 fold × split runs.
- Within-site direction stability: median cosine 0.964 (site 8) and 0.966
  (site 12).
- The bounds are written into `frozen_config_site{8,12}.json` (`tost`).

## Addendum: passive-side controls at site 8 (2026-10-06, post hoc)

Added after the site-8 result (`passive_test_results.md`), so these are
post-hoc checks, not part of the declared test. The criteria below are
written before the controls are run. Script: `scripts/run_passive_controls.py`.
All checks use primary bad passive bases, T and I donors, and the same plan
rows as the main test.

- **Shuffled-label DAS.**
  - The 15 site-8 shuffled-label bases are retrained exactly as the
    `run_das_round2.py final` control (same permutation and training seeds).
  - They are run on all three splits and analysed with the declared
    bootstrap and rules (δ = 0.4914).
  - The site-8 call survives if the shuffled-label bases do *not* also give
    " by" RISE with O NO RISE.
  - It is a stronger result if the shuffled-label D(" by") is below the DAS
    D(" by") (difference reported with its CI).
- **Random directions.**
  - 100 draws, each with one random rank-1 direction per split-0 fold
    basis, raw and norm-matched (each patch's displacement rescaled to the
    DAS displacement norm).
  - Split-0 rows only. DAS is recomputed on the same rows for comparison.
  - The site-8 " by" effect is specific if DAS D(" by") exceeds the 95th
    percentile of norm-matched random D(" by") on the same rows.
  - The number of random draws with D(" by") ≥ δ is reported too.
