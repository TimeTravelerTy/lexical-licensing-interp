# Gemma replication, step 1: single-token refilter (stopped here)

Script: `scripts/build_gemma_verb_refilter.py` (tokenizers only, no model weights). Outputs:
`data/gemma/verb_refilter/{verbs,counts,pair_counts}.csv`, `meta.json`. Tokenizers: `google/gemma-3-1b-pt`
(gated download worked) and `EleutherAI/pythia-1.4b`; transformers 4.57.3, tokenizers 0.22.1 (the versions
the D11 builder used). No existing verb file or result was changed.

Form checked: `" V"` (participle with a leading space), the form `run_frame_contrast.py` scores after
"The N was" / "The N has" and that the curated *by* readout scores after the curated `was` prefix. For Gemma, a
verb counts as single-token only if `" V"` is one id **and** "prefix V" encodes to prefix ids + that id for all
48 freq prefixes (24 nouns × was/has). No single-token Gemma form failed the joined check. The stored Pythia
token counts reproduce exactly. Bands are the existing ones (not re-binned).

## Verbs per class × band (single-token participle)

D11 inventory (281 pairs; the Δ_pass inventory of round 6 A2):

| class | band | all verbs | Pythia single | Gemma single | intersection |
|---|---|---:|---:|---:|---:|
| T | Head | 47 | 44 | 47 | 44 |
| T | Tail | 132 | 24 | 60 | 24 |
| T | XTail | 102 | 1 | 2 | 1 |
| I | Head | 47 | 45 | 47 | 45 |
| I | Tail | 132 | 20 | 53 | 20 |
| I | XTail | 102 | **0** | **0** | **0** |

Curated inventory (126 pairs; the *by*-readout inventory of task 1 / R4-E13d):

| class | band | all verbs | Pythia single | Gemma single | intersection |
|---|---|---:|---:|---:|---:|
| T | Head | 26 | 26 | 26 | 26 |
| T | Tail | 50 | 6 | 25 | 6 |
| T | XTail | 50 | 0 | 0 | 0 |
| I | Head | 26 | 26 | 26 | 26 |
| I | Tail | 50 | 7 | 23 | 7 |
| I | XTail | 50 | **0** | **0** | **0** |

Pairs with both members single-token in both tokenizers: D11 Head 43 / Tail 3 / XTail 0; curated 26 / 1 / 0.
Gemma token counts of the XTail participles: D11 I 74 two-token, 28 three-token; T 2 / 67 / 32 / 1 (one to
four tokens).

**Flag: I × XTail = 0 (below 15) in the intersection, and in Gemma alone, for both inventories. Stopped
before step 2, as instructed.**

## Where the existing code disagrees with the task description

- **There is no single-token filter on the verb lists.** D11 (`data/round5/verbs_expanded/pairs.csv`,
  loader `scripts/verbs_expanded.py`) and the curated pairs keep multi-token participles and record the
  count (`t_ntok`, `t_single`, `tok_cell`). Δ_pass is the full participle log-prob summed over subtokens;
  single-token subsets are used only for per-verb affinity analyses (a_V, singleton D_L) and A6 stream counts.
  XTail is almost entirely multi-token under Pythia by construction (Zipf 1.2–2.2 forms in a 50k BPE), and
  Gemma's 262k vocabulary does not change that.
- **No BOS for Pythia.** All Pythia scoring (`run_frame_contrast.py`, `score_matched_passives.py`,
  `run_next_token_readout.py`) uses `add_special_tokens=False`, no BOS. Gemma needs BOS, so the two models
  will differ in that respect by design.
- **Gemma's tokenizer defaults to left padding** (`padding_side="left"`); the repo's scorers right-pad
  (`run_frame_contrast.py` pads with EOS and reads positions from the left), so any Gemma scorer must set
  right padding or index per row.
