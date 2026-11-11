# OLMo replication, DAS layer sweep on the active task: OLMo-2-1B vs Pythia-1.4B and Gemma-3-1B-pt

Rules (written before the run): `rules.md` §4 and §6. The *by* gap held (`behavior.md`), so the sweep was run.
Per-site tables for all three models: `das_sweep_tables.md`. CSVs: `../../results/olmo/das/`.

## Setup

The Pythia round-2 and Gemma active-task sweep, unchanged:

- `run_das_round2.py sweep --model allenai/OLMo-2-0425-1B --pairs data/das_round2/train_pairs.csv`.
- Rank 1, every site 0..16. Site s is the residual output of layer s − 1. Site 16 holds post-final-norm source
  states and is excluded from conclusions.
- Split seed 0, 5 folds, 8 epochs, lr 0.01, batch 64.
- bf16 with fp32 unembedding, no BOS (`olmo2` map default).
- Pairs:
  - 29 training pairs (406 items).
  - None dropped by the behaviour filter: no verb is below 4 of 7 in either class. Gemma dropped
    `transplant/ensue`.
- Selection: `select_das_config.py`, unchanged. Selection only: no final run, no passive test, no transfer.

## Port checks (`check_das_port.py`, gate before the sweep): passed

| check | OLMo-2-1B | Gemma-3-1B-pt | Pythia-1.4B |
|---|---:|---:|---:|
| self-patch max \|ΔM\|, sites 0..L−1 | **0** (sites 0..15) | 0 | 0 |
| self-patch max \|ΔM\|, last site (post-final-norm) | 0.042 | 0.962 | 0.048 |
| natural M, batched vs one-at-a-time (bf16) | 0.050 | 0.038 | 0.096 |
| patched M at the check site, batched vs one-at-a-time (bf16) | 0.040 (site 8) | 0.053 (site 9) | 0.171 (site 8) |
| item accuracy, class 0 / class 1 | 0.985 / 0.985 | 0.966 / 1.000 | 0.985 / 0.975 |
| mean natural M, class 0 / class 1 | −3.12 / +1.74 | −2.80 / +2.07 | −2.93 / +1.92 |

The runner settings were `model_type olmo2`, `bos False` and `fp32_unembed True`. The site check gives 0.0 at sites
0..15.

OLMo token patterns of the 29 pairs (T, I subtokens):

| T1-I1 | T1-I2 | T2-I2 | T1-I3 | T2-I1 | T3-I1 |
|---:|---:|---:|---:|---:|---:|
| 9 | 9 | 8 | 1 | 1 | 1 |

## Sweep: held-out cross-class IIA by relative depth

Each value is the best epoch per site, averaged over 5 folds. Depth-matched values are interpolated linearly between
sites. The last site is excluded.

| model | layers | IIA at depth 0.25 | 0.50 | 0.75 | first site ≥ 0.5 | first ≥ 0.9 | first ≥ 0.95 |
|---|---:|---:|---:|---:|---|---|---|
| Pythia-1.4B | 24 | 0.669 | 0.921 | 0.989 | 5 (0.21) | 12 (0.50) | 14 (0.58) |
| Gemma-3-1B-pt | 26 | 0.856 | 0.954 | 0.998 | 1 (0.04) | 8 (0.31) | 12 (0.46) |
| **OLMo-2-1B** | 16 | **0.575** | **0.794** | **0.978** | 4 (0.25) | 11 (0.69) | 12 (0.75) |

The first-site columns give the site, with its relative depth in brackets.

OLMo per site:

| site | depth | IIA cross [fold min, max] | IIA same | pc frac |
|---:|---:|---|---:|---:|
| 0 | 0.00 | 0.129 [0.060, 0.204] | 0.966 | 0.20 |
| 2 | 0.12 | 0.290 [0.212, 0.343] | 0.980 | 0.47 |
| 4 | 0.25 | 0.575 [0.446, 0.649] | 0.986 | 0.64 |
| 6 | 0.38 | 0.712 [0.625, 0.794] | 0.991 | 0.74 |
| 8 | 0.50 | 0.794 [0.673, 0.877] | 0.993 | 0.77 |
| 10 | 0.62 | 0.873 [0.730, 0.927] | 0.993 | 0.83 |
| 11 | 0.69 | 0.940 [0.847, 0.989] | 0.993 | 0.96 |
| 12 | 0.75 | 0.978 [0.956, 0.988] | 0.993 | 1.08 |
| 13 | 0.81 | 0.993 [0.988, 1.000] | 0.991 | 1.21 |
| 14 | 0.88 | 0.993 [0.982, 1.000] | 0.988 | 1.46 |
| 15 | 0.94 | 0.997 [0.992, 1.000] | 0.983 | 1.67 |
| 16 (last) | 1.00 | 0.998 [0.992, 1.000] | 0.982 | 1.87 |

All sites, including the odd ones, are in `das_sweep_tables.md`.

- **Shape.** The curve rises monotonically from near chance to ceiling, like Pythia's. It does not show Gemma's
  early jump (Gemma reaches 0.69 at site 1).
- **Relative depth.** OLMo rises **latest** of the three models. At half depth it is at 0.79, against Pythia's 0.92
  and Gemma's 0.95. It reaches 0.9 only at 0.69 depth.
- **Absolute layer index.** OLMo tracks Pythia closely up to site 10. For example, at site 8 OLMo has 0.794 and
  Pythia 0.803. Both models have hidden size 2048, but OLMo has 16 layers and Pythia 24. From site 11 on, OLMo
  closes to ceiling over its last few layers.
- **Same-class IIA** is ≥ 0.966 at every site, as for the other two models.
- **Positive-control fraction** passes 0.5 from site 3 and 1.0 from site 12.

## Frozen-rule site and epoch

The rule: take the earliest site whose best IIA is within 0.02 of the overall best and whose positive-control
fraction is ≥ 0.5. The epoch is the smallest epoch within 0.01 of that site's best.

| model | frozen site (layer output) | depth | epochs | IIA at that epoch | pc frac | overall best IIA |
|---|---|---:|---:|---:|---:|---:|
| Pythia-1.4B | 17 (16) | 0.71 | 1 | 0.979 | 1.06 | 0.997 (site 24) |
| Gemma-3-1B-pt | 15 (14) | 0.58 | 2 | 0.977 | 1.20 | 1.000 (site 20) |
| **OLMo-2-1B** | **13 (12)** | **0.81** | **3** | **0.985** | **1.12** | 0.998 (site 16) |

OLMo's file: `results/olmo/das/frozen_config.json`. Sweep sha256 `f81e60bc0efb26ff…`.

**A near tie at the threshold.** `select_das_config.py` keeps the post-final-norm last site in the overall best:

- That gives a threshold of 0.99841 − 0.02 = 0.97841.
- Site 12's best IIA is 0.97792, below the threshold by 0.0005. So the declared rule picks **site 13**.
- `analyze_gemma_das_sweeps.py` leaves the last site out of the overall best (0.99665, site 15), so its version of the
  rule picks site 12. The tables now print both picks where they differ.
- For Pythia and Gemma the two versions agree.

Site 13 stands as the declared pick, because `rules.md` names `select_das_config.py` unchanged. Site 12 (depth 0.75)
is the same distance within the rule's tolerance. Either way, OLMo's frozen site is the deepest of the three models by
relative depth.

## Stopped here

As specified, the following were not run:

- the final run (rank selection);
- the passive *by* test;
- the transfer runs.

## Run

- TSUBAME `gpu_h` (H100), branch `olmo-replication`, commit 5809066, job 9354956.
- The sweep took 16 min, about 1 min per site.
- Log: `../../results/olmo/logs/o_das.9354956.out`.
