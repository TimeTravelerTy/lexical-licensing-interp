# Round 5, D10: scale-up feasibility (Pythia 1.4B / 6.9B / 12B)

Spec: `plan.md`, D10. Runs: `run_scaleup_feasibility.py` and `run_path_patch_frames.py been --feasibility 8` (TSUBAME job 9008724, one H100, fp32, TF32 off; warm-up, CUDA synchronization, median of 5 timed repetitions for DAS steps, 3 for evaluation and throughput; random bases, updates discarded). Table: `summarize_scaleup.py`.

## Measurements

| model | layers × width | load (s) | weights (GiB) | passive forward (prompts/s) | DAS step, batch 64 (s) | eval batch of 512 (s) | DAS peak (GiB) | path-patch batch of 32, short / long (s) | path-patch peak reserved (GiB) |
|---|---|---:|---:|---:|---|---|---:|---|---:|
| 1.4b | 24 × 2048 | 54 | 5.3 | 4341 | 0.057 (s4) / 0.053 (s8) / 0.043 (s17) | 0.19 / 0.19 / 0.19 | 6.4 | 0.41 / 0.46 | 7.6 |
| 6.9b | 32 × 4096 | 119 | 25.5 | 1006 | 0.225 (s5) / 0.206 (s11) / 0.162 (s23) | 0.76 / 0.76 / 0.76 | 27.5 | 2.30 / 2.42 | 37.0 |
| 12b | 36 × 5120 | 199 | 44.1 | 573 | 0.360 (s6) / 0.330 (s12) / 0.261 (s26) | 1.32 / 1.33 / 1.33 | 46.9 | 4.39 / 4.63 | 62.3 |

DAS columns list the earliest, middle and latest planned site of each model (1.4B: 4, 8, 17; 6.9B: 5, 11, 23; 12B: 6, 12, 26). No measurement needed the out-of-memory fallback (batch 64 and 512 throughout).

## Run-time estimates

A DAS fit = 31 training steps + 2 evaluation batches (slowest measured site); a DAS run = 15 fits. Final = round 2's final stage (3 ranks, 100 random bases × 2 on every fit, shuffled-label refit). Sweep = 8 sites × 5 folds × 8 epochs with per-epoch evaluation. Model load is excluded. Path patching: the measured three-frame run (182 + 184 batches) and, as an upper bound, the same scaled by the number of components below the flagged layers at matched depth (the timing run kept 1.4B's flagged layers 11–22).

| model | load | DAS run (15 fits) | round-2 final | 8-site sweep | path patching, 3 frames | 2 frames (≈ 2/3) |
|---|---:|---:|---:|---:|---|---|
| 1.4b | 54 s | 0.5 min | 21.3 min | 11.5 min | 2.7 min | 1.8 min |
| 6.9b | 119 s | 2.1 min | 1.41 h | 45.3 min | 14.4 min–19.2 min | 9.6 min–12.8 min |
| 12b | 199 s | 3.5 min | 2.45 h | 1.23 h | 27.5 min–41.7 min | 18.3 min–27.8 min |

**Calibration** (1.4B). The path-patching estimate (182 × 0.41 + 184 × 0.46 s = 159 s) matches round 4's actual D10 runs (164–180 s after setup; jobs 8947738 and 8947215). Round 2's final job (8901812, bf16) ran the primary final stage and the named-pair sensitivity (rank fixed, controls) in 38 min of wall time; the model gives 42 min for the same two stages in fp32.

## Site plan at matched relative depth

s' = ⌊s × L' / 24 + 0.5⌋ from the 1.4B sweep sites (4, 6, 8, 10, 12, 14, 16, 17):

| model | layers | sites |
|---|---:|---|
| 1.4b | 24 | 4, 6, 8, 10, 12, 14, 16, 17 |
| 6.9b | 32 | 5, 8, 11, 13, 16, 19, 21, 23 |
| 12b | 36 | 6, 9, 12, 15, 18, 21, 24, 26 |

±1 adjacent sites are a later sensitivity. Matched depth is a search heuristic, not evidence of homologous mechanisms.

