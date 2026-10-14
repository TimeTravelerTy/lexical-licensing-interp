#!/usr/bin/env python3
"""Round 5, D10b: A1/A2 across the Pythia suite (reports/round5/plan.md, D10b). No model.

Reads each model's results/round5/frame_suite/<model>/{a1_a2_summary.csv, a1_a2_decisions.json, frame_meta.json}
(analyze_frame_contrast.py --no-accounting) and the 1.4B run (results/round5/frame). Writes reports/round5/d10b_suite.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ORDER = ("pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b", "pythia-2.8b", "pythia-6.9b", "pythia-12b")


def main(args):
    lines = ["# Round 5, D10b: A1/A2 across the Pythia suite", "",
             "Spec: `plan.md`, D10b. A1's runner per model (`run_frame_contrast.py --model`), analysed with "
             "`analyze_frame_contrast.py --no-accounting`; freq prefixes (primary). ΔX = Head − XTail.", "",
             "| Model | L: Head / Tail / XTail | ΔL | −ΔΔ_pass(I) | ΔΔ_pass(T) | ΔM_has | reading | identity err | Head AUC | D_L (singletons) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for m in ORDER:
        d = Path(args.root) / m if m != "pythia-1.4b" else Path(args.ref)
        if not (d / "a1_a2_summary.csv").exists():
            lines.append(f"| {m} | not run | | | | | | | | |")
            continue
        S = pd.read_csv(d / "a1_a2_summary.csv")
        dec = json.loads((d / "a1_a2_decisions.json").read_text())
        meta = json.loads((d / "frame_meta.json").read_text())
        g = lambda q, st: S[(S.set == "freq") & (S.quantity == q) & (S.stat == st)].iloc[0]
        f = lambda r: f"{r.est:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]"
        L = "/".join(f"{g('L', b).est:.2f}" for b in ("head", "tail", "xtail"))
        dI = g("dI", "head-xtail")
        gate = dec["a4_gate"]["detail"]["curated + pool (gate)"]["D_L"]
        lines.append(f"| {m} | {L} | {f(g('L', 'head-xtail'))} | {-dI.est:+.2f} | {g('dT', 'head-xtail').est:+.2f} | "
                     f"{f(g('Mhas', 'head-xtail'))} | {dec['freq']['reading']} | {meta['checks']['max_identity_error']:.1e} | "
                     f"{dec['a4_gate']['auc_head']:.2f} | {gate[0]:+.2f} [{gate[1]:+.2f}, {gate[2]:+.2f}] |")
    Path(args.report).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/round5/frame_suite")
    ap.add_argument("--ref", default="results/round5/frame")
    ap.add_argument("--report", default="reports/round5/d10b_suite.md")
    main(ap.parse_args())
