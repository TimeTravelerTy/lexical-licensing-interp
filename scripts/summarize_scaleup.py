#!/usr/bin/env python3
"""Round 5, D10: memory and runtime table for Pythia 1.4B / 6.9B / 12B (reports/round5/plan.md, D10). No model.

Inputs: results/round5/scaleup/{das,feasibility_been}_pythia-*.json (run_scaleup_feasibility.py,
run_path_patch_frames.py --feasibility). Output: reports/round5/d10_scaleup.md.

Run-time model (round-2 DAS settings, run_das_round2.py and results/das_round2/final_strict):
- a DAS fit trains on 4/5 of the 406 active items with 4 cross + 2 same-class swaps each, batch 64, so
  ceil(325 * 6 / 64) = 31 steps per epoch, and evaluates its held-out swaps (865 on average, 2 batches of 512);
- a DAS run = one site x 15 fold-split fits (rank 1, 1 epoch as frozen in round 2);
- round-2 final = ranks 1/2/4 (rank-2/4 steps counted as rank-1), then for the chosen rank 100 random bases x 2
  (plain and norm-matched) evaluated on every fit, and a shuffled-label refit;
- sweep = rank 1, every planned site, split seed 0, 5 folds, 8 epochs, evaluation after each epoch.
Path patching: one run = the round-4 D10 rows (11,657), 182 half-A batches (short prompts) and 184 half-B batches
(long prompts) of 32, three frames ("been"); two frames are estimated as 2/3 of that.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

RES = Path("results/round5/scaleup")
MODELS = ("1.4b", "6.9b", "12b")
STEPS = math.ceil(325 * 6 / 64)  # 31
EVAL_BATCHES = 2
FITS = 15
SITE_PLAN = {"1.4b": [4, 6, 8, 10, 12, 14, 16, 17], "6.9b": [5, 8, 11, 13, 16, 19, 21, 23],
             "12b": [6, 9, 12, 15, 18, 21, 24, 26]}
FLAGGED_14B = [11, 13, 14, 15, 16, 17, 18, 21, 22]
LAYERS = {"1.4b": 24, "6.9b": 32, "12b": 36}


def main():
    L = ["# Round 5, D10: scale-up feasibility (Pythia 1.4B / 6.9B / 12B)", "",
         "Spec: `plan.md`, D10. Runs: `run_scaleup_feasibility.py` and `run_path_patch_frames.py been --feasibility 8` "
         "(TSUBAME job 9008724, one H100, fp32, TF32 off; warm-up, CUDA synchronization, median of 5 timed repetitions "
         "for DAS steps, 3 for evaluation and throughput; random bases, updates discarded). Table: `summarize_scaleup.py`.",
         "", "## Measurements", "",
         "| model | layers × width | load (s) | weights (GiB) | passive forward (prompts/s) | DAS step, batch 64 (s) | "
         "eval batch of 512 (s) | DAS peak (GiB) | path-patch batch of 32, short / long (s) | path-patch peak reserved (GiB) |",
         "|---|---|---:|---:|---:|---|---|---:|---|---:|"]
    est = {}
    for m in MODELS:
        d = json.loads((RES / f"das_pythia-{m}.json").read_text())
        p = json.loads((RES / f"feasibility_been_pythia-{m}.json").read_text())
        sites = sorted(d["sites"], key=int)
        step = {s: d["sites"][s]["train"]["step_s"] for s in sites}
        ev = {s: d["sites"][s]["eval"]["eval_s"] for s in sites}
        assert all(d["sites"][s]["train"]["batch"] == 64 and d["sites"][s]["eval"]["batch"] == 512 for s in sites)
        peak = max(max(d["sites"][s]["train"]["peak_reserved_GiB"], d["sites"][s]["eval"]["peak_reserved_GiB"]) for s in sites)
        pp = p["median_full_batch_s"]
        L.append(f"| {m} | {d['layers']} × {d['hidden']} | {d['load_s']:.0f} | {d['load_alloc_GiB']:.1f} | "
                 f"{d['forward_passive']['prompts_per_s']:.0f} | "
                 + " / ".join(f"{step[s]:.3f} (s{s})" for s in sites) + " | "
                 + " / ".join(f"{ev[s]:.2f}" for s in sites) + f" | {peak:.1f} | {pp['0']:.2f} / {pp['1']:.2f} | "
                 f"{p['peak_reserved_GiB']:.1f} |")
        t_step, t_ev = max(step.values()), max(ev.values())  # slowest (earliest) site, conservative
        fit = STEPS * t_step + EVAL_BATCHES * t_ev
        das_run = FITS * fit
        final = 3 * FITS * STEPS * t_step + 3 * FITS * EVAL_BATCHES * t_ev + FITS * 100 * 2 * EVAL_BATCHES * t_ev + das_run
        sweep = len(SITE_PLAN[m]) * 5 * 8 * fit
        pp3 = 182 * pp["0"] + 184 * pp["1"]
        scale = sum(math.floor(l * LAYERS[m] / 24 + 0.5) for l in FLAGGED_14B) / sum(FLAGGED_14B)
        named = final - 2 * FITS * (STEPS * t_step + EVAL_BATCHES * t_ev)  # rank fixed: one rank plus controls
        est[m] = {"load": d["load_s"], "das_run": das_run, "final": final, "final_named": named, "sweep": sweep,
                  "pp3": pp3, "pp_scale": scale}
    L += ["", "DAS columns list the earliest, middle and latest planned site of each model (1.4B: 4, 8, 17; 6.9B: 5, 11, "
          "23; 12B: 6, 12, 26). No measurement needed the out-of-memory fallback (batch 64 and 512 throughout).", "",
          "## Run-time estimates", "",
          f"A DAS fit = {STEPS} training steps + {EVAL_BATCHES} evaluation batches (slowest measured site); a DAS run = "
          f"{FITS} fits. Final = round 2's final stage (3 ranks, 100 random bases × 2 on every fit, shuffled-label refit). "
          "Sweep = 8 sites × 5 folds × 8 epochs with per-epoch evaluation. Model load is excluded. Path patching: the "
          "measured three-frame run (182 + 184 batches) and, as an upper bound, the same scaled by the number of "
          "components below the flagged layers at matched depth (the timing run kept 1.4B's flagged layers 11–22).", "",
          "| model | load | DAS run (15 fits) | round-2 final | 8-site sweep | path patching, 3 frames | 2 frames (≈ 2/3) |",
          "|---|---:|---:|---:|---:|---|---|"]
    fmt = lambda s: f"{s / 60:.1f} min" if s < 3600 else f"{s / 3600:.2f} h"
    for m in MODELS:
        e = est[m]
        hi = e["pp3"] * e["pp_scale"]
        rng = lambda a, b: fmt(a) if abs(a - b) < 1 else f"{fmt(a)}–{fmt(b)}"
        L.append(f"| {m} | {e['load']:.0f} s | {fmt(e['das_run'])} | {fmt(e['final'])} | {fmt(e['sweep'])} | "
                 f"{rng(e['pp3'], hi)} | {rng(e['pp3'] * 2 / 3, hi * 2 / 3)} |")
    e = est["1.4b"]
    L += ["", "**Calibration** (1.4B). The path-patching estimate (182 × 0.41 + 184 × 0.46 s = 159 s) matches round 4's "
          "actual D10 runs (164–180 s after setup; jobs 8947738 and 8947215). Round 2's final job (8901812, bf16) ran the "
          "primary final stage and the named-pair sensitivity (rank fixed, controls) in 38 min of wall time; the model "
          f"gives {(e['final'] + e['final_named']) / 60:.0f} min for the same two stages in fp32.", "",
          "## Site plan at matched relative depth", "",
          "s' = ⌊s × L' / 24 + 0.5⌋ from the 1.4B sweep sites (4, 6, 8, 10, 12, 14, 16, 17):", "",
          "| model | layers | sites |", "|---|---:|---|"]
    for m in MODELS:
        L.append(f"| {m} | {LAYERS[m]} | {', '.join(map(str, SITE_PLAN[m]))} |")
    L += ["", "±1 adjacent sites are a later sensitivity. Matched depth is a search heuristic, not evidence of homologous "
          "mechanisms.", ""]
    Path("reports/round5/d10_scaleup.md").write_text("\n".join(L) + "\n")
    (RES / "estimates.json").write_text(json.dumps(est, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
