# Round 5, D11: expanded verb list

Spec: `plan.md`, D11. Built by `build_verbs_expanded.py` (no model). Outputs: `data/round5/verbs_expanded/{candidates,verbs,pairs}.csv`, `meta.json`; loader `scripts/verbs_expanded.py`.

## Candidates and ratings

Sources pooled by lemma: 1312 lemmas, 1233 with the participle in a FreqBLiMP band (rated). Both raters labelled each rated lemma T / I / reject, blind to the source class and to each other. Agreement 0.94, Cohen's κ = 0.88.

| band | candidates | both T | both I | disagree | both reject |
|---|---:|---:|---:|---:|---:|
| head | 408 | 319 | 47 | 25 | 17 |
| tail | 564 | 395 | 133 | 26 | 10 |
| xtail | 261 | 118 | 126 | 17 | 0 |

Confusion (rows Claude, columns Codex):

| Claude \ Codex | I | T | reject |
|---|---:|---:|---:|
| I | 306 | 4 | 22 |
| T | 0 | 832 | 7 |
| reject | 5 | 30 | 27 |

## Pairs

Matching within band: maximum one-to-one pairs within the calipers (participle Zipf ≤ 0.35, lemma Zipf ≤ 0.25), then minimum total participle-Zipf gap.

| band | pairs | T1-I1 / T1-I2+ / T2+-I1 / T2+-I2+ | curated verb in pair | DAS-historical | held-out | mean part. gap | mean lemma gap |
|---|---:|---|---:|---:|---:|---:|---:|
| head | 47 | 43 / 1 / 2 / 1 | 27 | 33 | 4 | 0.03 | 0.15 |
| tail | 132 | 3 / 21 / 17 / 91 | 69 | 47 | 29 | 0.01 | 0.13 |
| xtail | 102 | 0 / 1 / 0 / 101 | 68 | 13 | 29 | 0.08 | 0.15 |
| all | 281 | 46 / 23 / 19 / 193 | 164 | 93 | 62 | 0.04 | 0.14 |

**Rule** (≥ 128 pairs and ≥ 40 per band): **met**.

Accepted but unpaired (T, I per band): {'head': {'T': 272, 'I': 0}, 'tail': {'T': 263, 'I': 1}, 'xtail': {'T': 16, 'I': 24}}.

