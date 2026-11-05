# Gemma replication: DAS runner port and Gemma-3-1B layer sweeps

Per-site tables: `das_sweeps.md` (script `scripts/analyze_gemma_das_sweeps.py`; CSV
`../../results/gemma/das/sweep_summary_by_site.csv`). Runs on TSUBAME `gpu_h`, branch `gemma-replication`:
- job 9270204: port checks (pre-map version) and the Gemma active sweep;
- job 9270266: port checks with the map, Pythia regressions, Gemma passive sweep;
- job 9270291: the pre-port code (commit 79ab4bf) on the same regressions.

Logs are in `../../results/gemma/logs/`. Selection only: no final runs, no passive test, no transfer.

## Port

- **Architecture map** `scripts/model_arch.py` (`ARCH`, keyed by `config.model_type`). Each entry gives:
  - the embedding hook (site 0), the layer list (site s = output of layer s − 1) and the final norm;
  - BOS and fp32-unembedding defaults;
  - where any embedding scaling happens.

  Entries: `gpt_neox` (Pythia: no BOS, bf16 logits, as in every stored run), `gemma3_text` (BOS, fp32 logits,
  √d scaling inside `embed_tokens`) and `olmo2` (BOS undecided, so the runner refuses to start without
  `--bos` / `--no-bos`). Adding a model is one entry.
- **`run_das_round2.Runner`** takes hooks and defaults from the map; `--bos/--no-bos` and
  `--fp32-unembed/--no-fp32-unembed` override them. The position rule (last participle subtoken =
  `attention_mask.sum(1) − 1`, right padding) and the interchange `h ← h + ((h_src − h)B)Bᵀ` are unchanged.
  `run_passive_das.py` (the passive *by* task) subclasses the Runner and gets the same behaviour.
- **Start-up site check:** on a real forward, the hooked output at sites 0..L−1 must equal
  `output_hidden_states` exactly (`model_arch.check_sites`). The last site is reported, not asserted.
- **Not yet ported:** other scripts still hard-code GPT-NeoX hooks. They fail on Gemma rather than misbehave.
  - Directly: `run_bylever`, `run_conjunction`, `run_shared_readers`, `run_necessity`, `run_site8_mechanism`,
    `run_path_patch(_frames)`, `run_oracle(_translation)`, `run_probe`, `run_neurons_wild`,
    `project_nonce_frames`, `project_eval_on_das_subspace`, and the June scripts `run_pythia_das_v1`,
    `run_pythia_exact_patching`, `run_pythia_whole_pair_das`.
  - Through `Mech` / `Capture` imports: `run_serial_chain`, `run_dose_decomp`, `run_translation_gain`.
  - `run_oracle(_translation)` and the frame builder used by `run_path_patch_frames` / `run_serial_chain` also
    build inputs without the BOS rule.

## Checks

- **Site consistency.** Sites 0..L−1 match exactly in both models (max |Δ| 0.0); the self-patch with an item's own
  state changes M by exactly 0 at every one of them.
  - **Gemma √d scaling at site 0:** consistent. The hooked `embed_tokens` output (which applies the scaling)
    and `hidden_states[0]` are identical, and the site-0 self-patch is 0.
  - **Last site** (Pythia 24, Gemma 26): mismatched in both, because HF's last hidden state is taken after the
    final norm, while the hook sees the pre-norm output. This is pre-existing and left unchanged. Gemma's IIA
    collapses there (0.51 active, 0.62 passive), so that site is excluded from the site picks below.
- **Pythia, ported runner vs stored outputs.**
  - Forward paths reproduce: natural M of all 406 training items matches to 2.2e-7, and the stored site-8
    basis re-evaluated on its 924 held-out swaps matches to 9e-16.
  - Training regressions, same seeds:
    - **Refactored vs pre-port code, both run today:** identical in every column (tolerance 0), for the
      passive sweep at site 8 (40 rows) and the round-2 final at site 17 (3,060 summary rows; ranks 1/2/4,
      random and shuffled-label controls). **The port does not move Pythia in these configurations.**
    - **Today vs the stored October runs:** close, not exact. Forward-only quantities match:
      - natural-pass τ and accuracy are identical;
      - raw random-direction controls: equal cross-class IIA in all 1,500 rows (1,448 rows equal in every
        column);
      - norm-matched random controls (scaled by the trained basis): equal IIA in 1,272 of 1,500 rows, max
        |Δ IIA| 0.008.

      Rows that involve training differ slightly:
      - final site 17: rank-1 DAS mean IIA 0.9728 vs 0.9730 (equal IIA in 19 of 45 DAS rows, max |Δ| 0.004);
        rank 1 chosen in both;
      - shuffled labels: mean 0.344 vs 0.341;
      - passive site 8: mean IIA by epoch within 0.0014, per fold ≤ 0.0093.

      Today's original code reproduces today's refactored code exactly, so the difference is not the port.
      The likely source is the training numerics of the environment (now torch 2.10.0+cu128; the stored runs'
      version was not recorded), but that is not established.
    - The stored passive sweep's `pc_frac_*` columns predate commit 96bf2b4 (gap-fraction denominator fix),
      so only its IIA and ΔM columns are comparable.
- **Gemma, batched vs one-at-a-time (bf16, 20 items):** natural M 0.038, patched M at site 9 0.053.
- **Gemma behaviour filters:**
  - active task keeps 28 of 29 pairs (drops transplant/ensue; Pythia keeps 29);
  - passive task keeps 24 (drops include/happen, infer/err, repeal/defect, send/die, transplant/ensue;
    Pythia keeps 26).
- **Tokenization covariance (Gemma only).** 11 of the 29 training pairs have a one-token transitive and a
  multi-token intransitive participle (T1-I2 9, T1-I3 2; Pythia: none). So at the last-subtoken position,
  class partly co-varies with "whole word vs word-final piece" (Gemma: 26 of 28 T participles are one token, 16
  of 28 I; Pythia: 14 of 29 T, 14 of 29 I, so no covariance). Tested in the token-pattern split below: Gemma's
  advantage survives on swaps with no length difference, so the covariance is not its sole explanation.

## Gemma-3-1B layer sweeps (rank 1, every site, split 0, 5 folds, round-2 settings)

Held-out cross-class IIA at the participle's last subtoken, best epoch per site, fold mean. Picks:
- **frozen rule:** the round-2 rule of `select_das_config.py`, i.e. the earliest site within 0.02 of the
  best IIA with positive-control fraction ≥ 0.5. It is declared for the active task and applied
  descriptively to the passive task.
- **best:** the highest IIA, excluding the last site.

| Task | Model | Sites | Frozen-rule site (depth) | IIA there | Best site (IIA) |
|---|---|---|---|---|---|
| active (round 2, "object next") | Gemma-3-1B | 0–26 | **15** (0.58) | 0.984 | 20 (1.000) |
| active | Pythia-1.4B (stored) | 0–24 | 17 (0.71) | 0.987 | 23 (0.996) |
| passive (*by* vs rest) | Gemma-3-1B | 0–26 | **15** (0.58) | 0.982 | 25 (0.994; site 24 0.9938) |
| passive | Pythia-1.4B (stored) | 4–17 only | 14 (0.58) | 0.931 | 17 (0.942) |

Depth = site / number of layers. Selected sites at matched depth:

| depth ≈ | Gemma site | active IIA | passive IIA | Pythia site | active IIA | passive IIA |
|---|---|---|---|---|---|---|
| 0.15–0.17 | 4 | 0.793 | 0.878 | 4 | 0.454 | 0.635 |
| 0.31–0.35 | 8 / 9 | 0.903 / 0.911 | 0.904 / 0.902 | 8 | 0.803 | 0.820 |
| 0.58 | 15 | 0.984 | 0.982 | 14 | 0.959 | 0.931 |

Descriptively:
- Gemma's IIA is higher than Pythia's at early sites on both tasks, at matched relative depth.
- Gemma's passive IIA stays at 0.90–0.91 from site 5 to 10.
- Same-class IIA on the passive task is about 0.75 in both models.
- Gemma's passive positive-control fractions are 1.1–2.7 from site 1 (last site excluded). The stored Pythia
  passive fractions use the pre-96bf2b4 definition, so the two are not compared.

### Token-pattern split (does a length cue drive Gemma's early-site IIA?)

`scripts/run_das_token_split.py` retrains every sweep site at its best epoch with the sweep's seeds and
evaluates each held-out swap. `scripts/analyze_das_token_split.py` splits cross-class IIA by the participles'
token counts. Full tables: `reports/gemma/das_token_split.md`; data: `results/gemma/das/token_split/`.
- **Reproduction:** Gemma's retrained fold IIA equals the sweep (max |Δ| 1.1e-16, both tasks). Pythia's is
  within 0.0099 per fold of the stored October sweeps, the same environment drift as the regression above.
- **Within Gemma, no length-cue signature.** T1-I1 swaps (both participles one token, so no length cue) are not
  significantly below T1-I2+ swaps at any site. T1-I1 − T1-I2+ has a CI including 0 everywhere (point estimates
  down to −0.06, active site 7) except active site 3 (+0.18 [0.00, 0.37], T1-I1 higher). The CIs are too wide to
  establish equivalence. At site 4, T1-I1 IIA is 0.842 (active) and 0.888 (passive), vs 0.436 / 0.616 for
  Pythia's T1-I1 swaps.
- **Same verb pairs in both models.** On common single-token pairs held out in both runs (active: 38 pairs,
  13 T × 13 I verbs; passive: 48 pairs, 10 × 10), Gemma − Pythia at nearest depth:
  - active: +0.22 to +0.46 at sites 0–5 (CI excludes 0 except site 2); +0.18 to +0.19 at Pythia sites 6–7
    (CI lower bound ≤ 0.00). From depth ≈ 0.33 it is ≤ 0.10 with CIs including 0. From depth ≈ 0.75 it is
    0.01–0.04 with lower bound 0.00, Gemma at ceiling, and −0.004 [−0.02, 0.00] at the last matched site
    (Pythia 23 / Gemma 25).
  - passive: +0.28 [0.10, 0.42] at site 4 and +0.25 [0.13, 0.35] at site 6. From site 8 / 9 it is ≤ 0.13, with
    a CI including 0 except at Pythia 14 / Gemma 15 (+0.06 [0.01, 0.11]; lower bound 0.00 at 17 / 18).
- **Pythia's own early sites go the other way.** On the active task at sites 0–2, Pythia's T1-I1 swaps beat
  T1-I2+ (+0.15 to +0.18, CIs exclude 0), the opposite of a length cue helping. Untested explanations: (a) the
  last subtoken carries less verb identity before mixing; (b) earlier verb pieces stay unpatched and keep the
  base verb's influence; (c) a cohort confound, since Pythia's multi-token I participles are rarer (mean
  participle Zipf 2.75 vs 3.87 for single-token ones).
- **Reading.** Gemma's higher early-site IIA holds on single-token verbs and on the same verb pairs as
  Pythia. So the length difference in the evaluated swaps is not its sole explanation. Training still uses
  every token pattern, so a contribution from the training-set covariance is not excluded. Nor does the result
  say what the early direction encodes. At site 0 the patched state is the contextless token embedding (√d-
  scaled for Gemma). But IIA is measured through the whole downstream model, so higher IIA can also mean
  Gemma's later layers respond more to lexical steering (possibly via frequency or morphological correlates),
  not only that the class is more separable in the patched representation.
- **Caveats:** small matched sets, wide CIs; one split (split 0); IIA at each site's best epoch, which
  selects on held-out IIA, the same as the sweep tables.

## OLMo-2-1B (no runs yet)

- **Model id:** `allenai/OLMo-2-0425-1B` (not gated).
  - 16 layers, hidden 2048, vocab 100,352, untied `lm_head`, checkpoint stored in fp32, `model_type` `olmo2`.
  - `GPT2Tokenizer`: `bos_token` = `eos_token` = `<|endoftext|>`, `pad_token` `<|pad|>`, no `add_bos_token`.
- **Intermediate checkpoints:** yes, as revisions (branches).
  - `allenai/OLMo-2-0425-1B` has 268 branches: `main`, 195 `stage1-step{N}-tokens{M}B` (step 0 to 1,907,359,
    mostly every 10k steps) and 72 `stage2-ingredient{1,2,3}-step{N}-tokens{M}B`.
  - `allenai/OLMo-2-0425-1B-early-training` adds 37 early `stage1` branches (steps 0–36,000, every 1,000).
- **What the runner needs:**
  1. The `olmo2` entry exists. Its hooks passed `check_sites` at sites 0..L−1 on a small randomly
     initialised OLMo-2, checked locally in this session (no saved artifact); the last site mismatches, as for
     the other two architectures.
  2. **Decide BOS.** The tokenizer adds none, and the only candidate is `<|endoftext|>` (= EOS). The map
     leaves it undecided so it can't be defaulted by accident. Settle it with the step-2 behavioural check
     run both ways.
  3. A `--revision` pass-through in `Runner` (it loads `args.model` only; `run_frame_contrast.py` already has
     `--revision`) for checkpoint runs; and download the weights (fp32, about 6 GB per revision) to the HDD
     cache on a compute node.
  4. Re-run the tokenizer checks for the participles and readout sets (`build_gemma_frame_inputs.py --model`,
     the runner's single-token readout asserts).
  5. Sites 0–16 (16 is the post-norm last site). `run_passive_das.py`'s default `--sites` includes 17, so pass
     the sites explicitly.
  6. The step-2 scorers take explicit `--bos` / `--fp32-unembed` flags rather than the map defaults; pass the
     decided settings.
