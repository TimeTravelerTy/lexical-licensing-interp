# B3a: construction annotation of core ranks 4–5 (R5-B7 protocol; Claude and Codex, independent)

80 items. Raw agreement 0.94; Cohen's κ = 0.91.

| neuron | annotator | verbal passive participle | adjectival / copular participle | active perfect or simple past | object gap after the verb | other | uncertain |
|---|---|---:|---:|---:|---:|---:|---:|
| MLP11.n4925 | claude | 0.00 | 0.00 | 0.00 | 0.35 | 0.62 | 0.03 |
| MLP11.n4925 | codex | 0.00 | 0.00 | 0.00 | 0.30 | 0.68 | 0.03 |
| MLP14.n4241 | claude | 0.75 | 0.00 | 0.07 | 0.07 | 0.10 | 0.00 |
| MLP14.n4241 | codex | 0.72 | 0.00 | 0.05 | 0.05 | 0.12 | 0.05 |

| neuron | verbal passive (Claude / Codex) | object gap (Claude / Codex) | reading |
|---|---|---|---|
| MLP11.n4925 | 0.00 / 0.00 | 0.35 / 0.30 | **other** |
| MLP14.n4241 | 0.75 / 0.72 | 0.07 / 0.05 | **passive-participle detector** |

Confusion (rows Claude, columns Codex):

| Claude \ Codex | active perfect or simple past | object gap after the verb | other | uncertain | verbal passive participle |
|---|---:|---:|---:|---:|---:|
| active perfect or simple past | 2 | 0 | 0 | 1 | 0 |
| object gap after the verb | 0 | 14 | 3 | 0 | 0 |
| other | 0 | 0 | 29 | 0 | 0 |
| uncertain | 0 | 0 | 0 | 1 | 0 |
| verbal passive participle | 0 | 0 | 0 | 1 | 29 |
