# OLMo-2-1B replication: model, checkpoints, settings and rules (written before any OLMo run)

Third model, same pipeline as the Gemma port (`reports/gemma/step1_refilter.md`, `step2_behavior.md`,
`das_port.md`, `das_sweeps.md`). Branch `olmo-replication`, off `gemma-replication` at 4e17a02.

## 1. Model and intermediate checkpoints (Hugging Face Hub, queried 2026-11-11)

- **Model id:** `allenai/OLMo-2-0425-1B`. Not gated. `main` = commit `a1847dff35000b4271fa70afc5db10fd29fedbdf`
  (last modified 2025-05-28).
  - `olmo2`, 16 layers, hidden 2048, vocab 100,352, untied `lm_head`, weights stored in fp32 (2 safetensors
    shards, 5.94 GB).
- **Checkpoints are revisions (branches): 268 = `main` + 195 stage-1 + 72 stage-2.**
  - **Stage 1 (pretraining):** 195 branches `stage1-step{N}-tokens{M}B`, step 0 to 1,907,359 (0 to 4,001 B
    tokens).
    - Every 10,000 steps (187 of the 194 gaps).
    - Off-grid steps: 300, 23,100 and 66,200; the last is 1,907,359.
    - No gap is larger than 10,000 steps.
  - **Stage 2 (mid-training):** 3 ingredients × 24 branches `stage2-ingredient{1,2,3}-step{N}-tokens{M}B`.
    - Steps 1,000 to 23,852, every 1,000 (3 to 51 B tokens).
  - **Early training:** `allenai/OLMo-2-0425-1B-early-training`, a separate repo, has 37 more `stage1`
    branches: steps 0 to 36,000, every 1,000 (0 to 76 B tokens). Its `main` is c70db05b.
  - Only `main` is used below. A checkpoint run would need a `--revision` pass-through in `run_das_round2.Runner`
    (`run_frame_contrast.py` already has one).

## 2. Tokenizer conventions (checked on TSUBAME, transformers 4.57.6)

- **Tokenizer:** `GPT2TokenizerFast`.
  - `bos_token` = `eos_token` = `<|endoftext|>` (id 100257).
  - `pad_token` `<|pad|>` (100277, distinct from EOS).
  - `add_bos_token` False; `padding_side` right.
- **Model config:** `bos_token_id` None, `eos_token_id` 100257.
- `" by"` is one token (555). `"The man was"` → 3 ids, with no special tokens added.
- **Padding:** the scorers and the DAS runner pad on the right and read the last real position, as for Gemma
  (Gemma's tokenizer defaults to left padding; OLMo's is already right).

## 3. BOS rule

OLMo-2, like Pythia, was trained on EOS-delimited documents with no BOS token. Its tokenizer adds none.

- **Primary: no BOS.** This is Pythia's convention, used in every stored Pythia run. `model_arch.ARCH["olmo2"].bos`
  is now `False`.
- **Sensitivity: `<|endoftext|>` prepended to every prompt** (`--bos`, which uses `bos_token_id` = EOS).
  Run in the behavioural check only.
- If the *by*-gap verdict (§5) differs between the two settings, stop and report before any DAS run.

## 4. Precision and gates

The same as Gemma:

- bf16 weights, logits computed in fp32 from the final-norm output (`--fp32-unembed`; the map default).
- **Gate before the full behavioural runs (both BOS settings):**
  - Items: OLMo fp32, batched vs one-at-a-time, on the step-2 check items:
    - 40 frame rows (the same 10 verbs × 2 nouns × was/has as Gemma's, with OLMo token ids);
    - 20 *by*-readout prompts;
    - 10 FreqBLiMP pairs.
  - Pass: max |Δ| ≤ 1e-3 on every scored column.
  - bf16 batched vs one-at-a-time is reported, not gated.
- **Gate before the DAS sweep:** `check_das_port.py` on OLMo (sites 0..15 self-patch exactly 0, as for Gemma).

## 5. Behavioural check and the *by*-gap rule

- **Measures:** Δ_pass, the T − I *by* gap, and FreqBLiMP passive accuracy. Definitions, verbs and bootstrap are
  as in `scripts/analyze_gemma_behavior.py`; the full D11 and curated inventories, with no token filter.
- **Tables:** next to Pythia and Gemma.
  - Pythia and Gemma rows are the stored step-2 scores, and their CIs must reproduce the stored step-2 tables
    exactly.
  - Splits by Gemma token bin and by OLMo token bin.
  - Script: `scripts/analyze_olmo_behavior.py`.
- **The *by* gap holds** if, for the primary (no-BOS) run, the pooled (all bands) T − I *by* gap has a 95% CI
  above 0 in both inventories (D11, curated). This is the criterion under which Gemma's gap held in step 2.
  Band cells are reported but are not part of the rule.

## 6. DAS layer sweep (only if §5 holds)

The Pythia round-2 / Gemma active-task sweep, unchanged:

- rank 1, every site 0..16 (16 = post-final-norm last site, excluded from conclusions);
- split seed 0, 5 folds, 8 epochs, bf16 + fp32 unembedding, no BOS;
- `run_das_round2.py sweep --model allenai/OLMo-2-0425-1B --pairs data/das_round2/train_pairs.csv`.

After the sweep:

- The round-2 frozen rule picks the site and epoch (`select_das_config.py`, unchanged). This is selection only.
- The per-site table goes next to Pythia and Gemma by relative depth (`analyze_gemma_das_sweeps.py --sweep ...`).
- Then stop and report: no final run, no passive test, no transfer.
