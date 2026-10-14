#!/usr/bin/env python3
"""Round 5, B6 analysis (reports/round5/plan.md, B6). No model.

Input: results/round5/serial/serial_rows.parquet (+ meta). Per pair, every quantity is averaged over its rows;
shares are ratios of population means, sign(S_t) * mean(PE) / |mean(S_t)|, with a pair bootstrap within band
(2,000 draws, seed 17; shared draws for all quantities and targets).
Outputs: reports/round5/b6_serial.md, results/round5/serial/b6_summary.csv, b6_decisions.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_conjunction import pair_halves

TARGETS = (15, 18, 21, 22)


def weights(rng, labels, n_boot):
    labels = np.asarray(labels)
    w = np.ones((n_boot + 1, len(labels)))
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        w[1:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=n_boot)
    return w


def ci(d, lo=2.5, hi=97.5):
    return np.percentile(d[1:], lo), np.percentile(d[1:], hi)


def fmt(d, p=2):
    lo, hi = ci(d)
    return f"{d[0]:+.{p}f} [{lo:+.{p}f}, {hi:+.{p}f}]"


def main(args):
    rdir = Path(args.res_dir)
    R = pd.read_parquet(rdir / "serial_rows.parquet")
    meta = json.loads((rdir / "serial_meta.json").read_text())
    pm_all = R.drop(columns=["item_id", "donor", "ntok"]).groupby("pair").mean()
    z = np.load(args.d11, allow_pickle=True)
    halfB = set(np.array(z["pairs"])[pair_halves(np.array(z["bands"]), args.seed) == 1])
    lines = ["# Round 5, B6: serial chain into the late object MLPs", "",
             "Spec: `plan.md`, B6. Run: `run_serial_chain.py`; analysis: `analyze_serial_chain.py`. D10 rows "
             f"({meta['rows']} rows, {meta['pairs']} pairs); base \"has been\", counterfactual \"has\"; site-8 T-donor "
             "interchange; shares = mean PE / mean S per draw (pair bootstrap within band; signed).", "",
             f"Checks: {json.dumps(meta['checks'])}.", ""]
    rows, dec = [], {}
    for pop in ("all 64 pairs", "D11 half-B pairs"):
        pm = pm_all if pop.startswith("all") else pm_all[pm_all.index.isin(halfB)]
        band = np.array([p.split("/")[0] for p in pm.index])
        w = weights(np.random.default_rng(args.seed), band, args.n_boot)
        M = lambda col: (w * pm[col].to_numpy()[None]).sum(1) / w.sum(1)
        lines += [f"## {pop} ({len(pm)} pairs)", "",
                  "| Target | S | one-hop all MLPs | one-hop mid | two-hop parallel | two-hop sequential | sum of singles | "
                  "remainder S − two-hop | (a: 11+14) | neurons (c) | (c) / (a: 11+14) | remaining after freeze |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        dec_t = {}
        for t in TARGETS:
            S = M(f"S_{t}")
            ws = float(np.mean(np.sign(S[1:]) != np.sign(S[0])))
            sh = lambda col: M(col) / S
            mids = [m for m in range(11, 18) if m < t]
            singles = sum(M(f"single_{m}_{t}") for m in mids)
            q = {"one_mlps": sh(f"one_mlps_{t}"), "one_mid": sh(f"one_mid_{t}"), "two_par": sh(f"two_par_{t}"),
                 "two_seq": sh(f"two_seq_{t}"), "singles_sum": singles / S, "remainder": (S - M(f"two_par_{t}")) / S,
                 "two_1114": sh(f"two_1114_{t}"), "neu50": sh(f"neu50_{t}"), "freeze": sh(f"freeze_{t}")}
            q["neu_over_1114"] = M(f"neu50_{t}") / M(f"two_1114_{t}")
            lines.append(f"| {t} | {fmt(S, 3)} | " + " | ".join(fmt(q[k]) for k in
                         ("one_mlps", "one_mid", "two_par", "two_seq", "singles_sum", "remainder", "two_1114", "neu50",
                          "neu_over_1114", "freeze")) + " |")
            for k, v in q.items():
                rows.append({"pop": pop, "target": t, "quantity": k, "est": v[0], "lo": ci(v)[0], "hi": ci(v)[1]})
            rows.append({"pop": pop, "target": t, "quantity": "S", "est": S[0], "lo": ci(S)[0], "hi": ci(S)[1],
                         "S_wrong_sign": ws})
            elig = abs(S[0]) >= 0.05 and ws <= 0.025
            chain = elig and q["two_par"][0] >= 0.5 and ci(q["two_par"])[0] > 0
            necessary = elig and q["freeze"][0] <= 0.5 and ci(q["freeze"])[1] < 1
            comp_ok = elig and q["two_1114"][0] >= 0.25 and ci(q["two_1114"])[0] > 0
            neu = comp_ok and q["neu_over_1114"][0] >= 0.5
            dec_t[t] = {"eligible": bool(elig), "chain": bool(chain), "freeze_necessary": bool(necessary),
                        "comparator_ok": bool(comp_ok), "neurons_carry": bool(neu)}
        lines += ["", "Singles (parallel two-hop of each mid MLP alone, share of S):", ""]
        for t in TARGETS:
            S = M(f"S_{t}")
            mids = [m for m in range(11, 18) if m < t]
            lines.append(f"- MLP{t}: " + ", ".join(f"{m} {M(f'single_{m}_{t}')[0] / S[0]:+.2f}" for m in mids))
        ne = sum(d["eligible"] for d in dec_t.values())
        nc = sum(d["comparator_ok"] for d in dec_t.values())
        d = {"per_target": dec_t,
             "chain_carries_late_switch": ("unresolved" if ne < 3 else sum(v["chain"] for v in dec_t.values()) >= 3),
             "mid_anchor_outputs_necessary": ("unresolved" if ne < 3 else sum(v["freeze_necessary"] for v in dec_t.values()) >= 3),
             "switch_neurons_carry_chain": ("unresolved" if nc < 3 else sum(v["neurons_carry"] for v in dec_t.values()) >= 3)}
        dec[pop] = d
        lines += ["", "**Declared decisions" + (" (primary)" if pop.startswith("all") else "") + ":**",
                  f"- The chain carries the late switch (parallel two-hop ≥ 0.5, CI above 0, ≥ 3 of 4): "
                  f"**{d['chain_carries_late_switch']}** ({[t for t, v in dec_t.items() if v['chain']]}).",
                  f"- Anchor outputs of the mid MLPs are necessary (remaining ≤ 0.5, CI upper < 1, ≥ 3 of 4): "
                  f"**{d['mid_anchor_outputs_necessary']}** ({[t for t, v in dec_t.items() if v['freeze_necessary']]}; "
                  "t = 15 freezes 11–14).",
                  f"- The switch neurons carry the chain (where (a: 11+14) ≥ 0.25 with CI above 0, (c) ≥ 0.5 × it; ≥ 3 "
                  f"targets): **{d['switch_neurons_carry_chain']}** (comparator ok for "
                  f"{[t for t, v in dec_t.items() if v['comparator_ok']]}).", ""]
    Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.report_dir) / "b6_serial.md").write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(rdir / "b6_summary.csv", index=False)
    (rdir / "b6_decisions.json").write_text(json.dumps(dec, indent=1, default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-dir", default="results/round5/serial")
    ap.add_argument("--d11", default="results/round4/translation/conjunction.npz")
    ap.add_argument("--report-dir", default="reports/round5")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    main(ap.parse_args())
