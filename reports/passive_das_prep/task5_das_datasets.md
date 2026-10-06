# Task 5: DAS datasets (built, not trained)

Files in `data/passive_das/` (`actives.jsonl`, `passives.jsonl`, `manifest.json`), built by `scripts/build_das_passive_datasets.py`. Token indices are 0-based without special tokens: `*_verb_tokens` = [first, last] verb token, `*_readout_index` = last prompt token (its logits predict the next token), `*_next_index` = position of the next token. Baseline log-probs (`*_lp_the`, `*_lp_a`, `*_lp_by`, `*_lp_dot`) come from the task 1-4 readout.

## Train: actives ("The AGENT VERB-ed" -> " the" vs ".")

90 subjects (curated `passive_1` agents) x each verb pair. Head dev pairs (held out): celebrate/reply, confront/testify, include/happen, lease/apologize, lend/opt, rate/result. `LD` = log P(" the") - log P("."). Target behaviour: LD > 0 for the transitive verb, LD < 0 for the intransitive.

| Split | Items | Pairs | Trans LD | Intrans LD | Trans LD > 0 | Intrans LD < 0 | Both | Multi-token verb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| head_dev | 540 | 6 | 4.84 | -0.77 | 100.0% | 65.0% | 65.0% | 0.0% |
| head_train | 1,800 | 20 | 4.68 | -1.97 | 99.4% | 84.5% | 84.1% | 0.0% |
| transfer_tail | 4,500 | 50 | 3.86 | -2.00 | 96.2% | 80.1% | 76.7% | 98.0% |
| transfer_xtail | 4,500 | 50 | 3.55 | -1.53 | 96.1% | 71.4% | 67.9% | 100.0% |

## Eval: passives (curated `was` prefix + participle, other contexts)

Contexts whose prefix equals the pair's own prefix are dropped (42 items). Fit: Gemma-4-31B-it rating (1-7).

| Split | Band | Items | Pairs | Good fit (p2) | Bad patient fit | lp by good | lp by bad | lp the good | lp the bad |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| negative_by | head | 625 | 5 | 3.90 | 3.99 | -4.72 | -2.66 | -6.97 | -5.97 |
| negative_by | tail | 1,248 | 10 | 2.84 | 3.80 | -3.57 | -2.60 | -6.53 | -6.30 |
| negative_by | xtail | 1,498 | 12 | 2.23 | 3.06 | -3.61 | -2.32 | -6.26 | -6.15 |
| ns_pos | xtail | 375 | 3 | 2.03 | 2.54 | -2.48 | -2.59 | -5.59 | -6.65 |
| reliable | head | 2,622 | 21 | 4.49 | 4.08 | -2.06 | -3.71 | -5.76 | -6.15 |
| reliable | tail | 4,986 | 40 | 3.41 | 3.36 | -1.87 | -3.32 | -6.06 | -6.62 |
| reliable | xtail | 4,354 | 35 | 3.36 | 3.23 | -1.96 | -3.09 | -6.06 | -6.59 |
