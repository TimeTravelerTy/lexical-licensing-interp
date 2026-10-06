# Task 0: consistency checks

## Own-context advantage (own - other), pooled over bands

Curated cross, WLS with shared good-fit slope and band dummies; two-way cluster bootstrap (verb pairs within band x contexts within band), 1000 draws, seed 17. `committed` = `reports/fit_ratings/bad_fit_models.csv`.

| Paradigm | Metric | Model | Recomputed | Committed |
|---|---|---|---:|---:|
| passive_1 | correct | own | 21.60 [16.04, 26.23] | 21.60 [16.04, 26.23] |
| passive_1 | correct | good+own | 5.01 [0.25, 9.15] | 5.01 [0.25, 9.15] |
| passive_1 | correct | good+bad_patient+own | 4.58 [-0.26, 8.76] | - |
| passive_1 | whole_margin | own | 5.56 [4.38, 6.63] | 5.56 [4.38, 6.63] |
| passive_1 | whole_margin | good+own | 3.51 [2.26, 4.65] | 3.51 [2.26, 4.65] |
| passive_1 | whole_margin | good+bad_patient+own | 3.47 [2.24, 4.62] | - |
| passive_1 | verb_margin | own | 3.79 [2.93, 4.67] | 3.79 [2.93, 4.67] |
| passive_1 | verb_margin | good+own | 2.38 [1.44, 3.22] | 2.38 [1.44, 3.22] |
| passive_1 | verb_margin | good+bad_patient+own | 2.32 [1.43, 3.14] | - |
| passive_2 | correct | own | 20.45 [13.70, 25.88] | 20.45 [13.70, 25.88] |
| passive_2 | correct | good+own | 7.12 [0.28, 13.08] | 7.12 [0.28, 13.08] |
| passive_2 | correct | good+bad_patient+own | 5.84 [-0.59, 11.79] | - |
| passive_2 | whole_margin | own | 3.80 [2.87, 4.67] | 3.80 [2.87, 4.67] |
| passive_2 | whole_margin | good+own | 2.41 [1.47, 3.30] | 2.41 [1.47, 3.30] |
| passive_2 | whole_margin | good+bad_patient+own | 2.29 [1.37, 3.13] | - |
| passive_2 | verb_margin | own | 3.79 [2.88, 4.70] | 3.79 [2.88, 4.70] |
| passive_2 | verb_margin | good+own | 2.46 [1.55, 3.33] | 2.46 [1.55, 3.33] |
| passive_2 | verb_margin | good+bad_patient+own | 2.36 [1.47, 3.18] | - |

## *by* margin reliability per verb pair (`passive_1`, released contexts)

Mean over 300 released contexts per pair; 95% bootstrap CI over contexts.

| Verb band | n | Reliable + | n.s. + | n.s. - | Reliable - | Mean < 0 |
|---|---:|---:|---:|---:|---:|---:|
| head | 26 | 21 | 0 | 0 | 5 | 5 |
| tail | 50 | 40 | 0 | 1 | 9 | 10 |
| xtail | 50 | 35 | 3 | 0 | 12 | 12 |

Negative head pairs: sum/compete (-3.24), bet/appear (-2.68), include/happen (-2.45), exhibit/proceed (-1.49), load/respond (-1.17)

Negative xtail pairs: shoo/quiver (-3.60), daub/resound (-2.85), doff/salivate (-2.09), wad/shimmer (-1.78), debug/tingle (-1.30), lard/fizz (-1.06), unlatch/burgeon (-0.97), disgorge/exult (-0.89), swaddle/decamp (-0.82), shoplift/eventuate (-0.34), prise/jut (-0.32), tithe/putter (-0.31)
