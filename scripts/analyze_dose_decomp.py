#!/usr/bin/env python3
"""Decomposing the gradual push (round3_plan.md, A3).

Input: `dose_decomp_site8.npz` (`run_dose_decomp.py`). Per frame, interval j
(grid[j] -> grid[j+1]) and pair: sums of the carry, 256 heads, 16 MLPs,
LN-scale term, residual, actual change (centered logits; readouts " by", ".",
" the", " him", O-bar), Delta log-probs, Delta LSE terms, and fixed-scale
versions of the terms (`*_fx`, every interval divided by sigma at z = 0.10).

Interval groups: R_p = [0.10, 0.45] (intervals 0-1), [0.45, 0.90] (2-3),
[0.90, 1.50] (4-5). Per-z rate = summed change over the group / its width,
per pair, then the mean over pairs; 95% CIs: pair bootstrap (2,000 draws,
seed 17). Decision rules: `round3_plan.md`, A3.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RN = ["by", "dot", "the", "him", "Obar"]
LP = ["M", "O", "I", "det", "pron", "refl", "by", "dot", "the", "him"]
GROUPS = {"R_p [0.10, 0.45]": (0, 1), "[0.45, 0.90]": (2, 3), "[0.90, 1.50]": (4, 5)}
LATE = ("MLP15", "MLP18", "MLP21", "MLP22")
BYMID = ("MLP11", "MLP13", "MLP14", "MLP16", "MLP17")


def load(z, frame, fx=False):
    """-> per group: dict of per-pair rate arrays; components [pairs, 273, 5]."""
    grid = z["grid"]
    L0, nl = int(z["layers"][0]), len(z["layers"])
    names = ["carry"] + [f"L{L0 + l}H{h}" for l in range(nl) for h in range(16)] + [f"MLP{L0 + l}" for l in range(nl)]
    out = {}
    for g, js in GROUPS.items():
        width = grid[js[-1] + 1] - grid[js[0]]
        acc = {}
        for j in js:
            n = z[f"{frame}_{j}_n"]
            for k in ("carry", "heads", "mlp", "ln", "resid", "actual", "dlp", "dlse"):
                kk = f"{k}_fx" if fx and k in ("carry", "heads", "mlp") else k
                v = z[f"{frame}_{j}_{kk}"] / n.reshape((-1,) + (1,) * (z[f"{frame}_{j}_{kk}"].ndim - 1))
                acc[k] = acc.get(k, 0) + v
        P = acc["heads"].shape[0]
        comps = np.concatenate([acc["carry"][:, None], acc["heads"].reshape(P, -1, 5), acc["mlp"]], 1) / width
        out[g] = {"comps": comps, **{k: acc[k] / width for k in ("ln", "resid", "actual", "dlp", "dlse")}}
        if fx:  # fixed scale: the scale term is whatever the fixed-scale terms leave of the actual change
            out[g]["ln"] = out[g]["actual"] - comps.sum(1)
            out[g]["resid"] = np.zeros_like(out[g]["ln"])
    return out, names


def run(args):
    z = np.load(args.npz)
    rng = np.random.default_rng(args.seed)
    L = ["# A3. Decomposing the gradual push (site-8 dose-response)", "",
         "Spec: `round3_plan.md`, A3. Run: `run_dose_decomp.py` (fp32); analysis: `analyze_dose_decomp.py`. "
         "Coordinate-setting patches at site 8 (split-0 basis); each grid interval decomposed into the carry, 256 "
         "heads and 16 MLPs (layers 8–23), with the LN scale frozen at the interval's upper run (main) or at a fixed "
         "per-item scale (σ at z = 0.10, 'fixed σ'). Rates are per z unit; mean over pairs; 95% CI over pairs.", ""]
    rows = []
    for frame in ("bad_passive", "intrans_active"):
        if f"{frame}_pairs" not in z.files:
            continue
        for version, fx in (("main", False), ("fixed σ", True)):
            d, names = load(z, frame, fx)
            P = len(z[f"{frame}_pairs"])
            bidx = np.vstack([np.arange(P)[None], rng.integers(0, P, (args.n_boot, P))])
            ix = {n: i for i, n in enumerate(names)}
            q = lambda v: (float(v[0]), float(np.percentile(v[1:], 2.5)), float(np.percentile(v[1:], 97.5)))
            f3 = lambda t, k=2: f"{t[0]:+.{k}f} [{t[1]:+.{k}f}, {t[2]:+.{k}f}]"
            pm = {g: d[g]["comps"][bidx].mean(1) for g in GROUPS}  # [B+1, 273, 5]
            L += [f"## {frame}, {version}", "",
                  "| Interval | Readout | Actual | Carry | Downstream net | Downstream P | Downstream N | LN term | "
                  "Residual |", "|---|---|---|---|---|---|---|---|---|"]
            for g in GROUPS:
                for r in ("Obar", "by", "him", "the"):
                    j = RN.index(r)
                    c = pm[g][:, :, j]
                    ds = c[:, 1:]
                    act = d[g]["actual"][bidx].mean(1)[:, j]
                    vals = {"actual": q(act), "carry": q(c[:, 0]), "net": q(ds.sum(1)),
                            "P": q(np.clip(ds, 0, None).sum(1)), "N": q(np.clip(ds, None, 0).sum(1)),
                            "ln": q(d[g]["ln"][bidx].mean(1)[:, j]), "resid": q(d[g]["resid"][bidx].mean(1)[:, j])}
                    rows.append({"frame": frame, "version": version, "interval": g, "readout": r,
                                 **{f"{k}_{s}": v[i] for k, v in vals.items() for i, s in enumerate(("est", "lo", "hi"))}})
                    L.append(f"| {g} | {r} | {f3(vals['actual'])} | {f3(vals['carry'])} | {f3(vals['net'])} | "
                             f"{f3(vals['P'])} | {f3(vals['N'])} | {f3(vals['ln'], 3)} | {f3(vals['resid'], 3)} |")
            # log P and LSE rates
            L += ["", "| Interval | Δ log P(O) | Δ log P(\" by\") | Δ log P(pron) | Δ LSE_O | Δ LSE_all |",
                  "|---|---|---|---|---|---|"]
            for g in GROUPS:
                dl, dz = d[g]["dlp"][bidx].mean(1), d[g]["dlse"][bidx].mean(1)
                L.append(f"| {g} | {f3(q(dl[:, LP.index('O')]))} | {f3(q(dl[:, LP.index('by')]))} | "
                         f"{f3(q(dl[:, LP.index('pron')]))} | {f3(q(dz[:, 0]))} | {f3(q(dz[:, 1]))} |")
            if frame != "bad_passive":
                L.append("")
                continue
            # decisions on O-bar
            j = RN.index("Obar")
            lo_g, hi_g = "R_p [0.10, 0.45]", "[0.90, 1.50]"
            act_lo = d[lo_g]["actual"][bidx].mean(1)[:, j]
            act_hi = d[hi_g]["actual"][bidx].mean(1)[:, j]
            A = act_hi - act_lo
            ratio = act_hi[0] / act_lo[0]
            in_logits = q(act_lo)[1] > 0 and q(A)[1] > 0 and ratio >= 1.25
            carry_lo = pm[lo_g][:, 0, j]
            ds_lo = pm[lo_g][:, 1:, j]
            carry_share = carry_lo / act_lo
            P_lo, N_lo = np.clip(ds_lo, 0, None).sum(1), -np.clip(ds_lo, None, 0).sum(1)
            creep_carry = carry_share[0] >= 0.5 and P_lo[0] < carry_lo[0] and N_lo[0] < carry_lo[0]
            L += ["", "**Decisions on Ō (declared).**",
                  f"- Rate in R_p {f3(q(act_lo))}; in [0.90, 1.50] {f3(q(act_hi))}; A = {f3(q(A))}; ratio "
                  f"{ratio:.2f}. **Acceleration in the logits: {'yes' if in_logits else 'no'}.**",
                  f"- Carry share of the R_p change: {f3(q(carry_share))}; downstream P {P_lo[0]:.3f}, N "
                  f"{N_lo[0]:.3f} vs carry {carry_lo[0]:.3f}. **Creep = carry: {'yes' if creep_carry else 'no'}.**"]
            drate = pm[hi_g][:, :, j] - pm[lo_g][:, :, j]  # [B+1, 273]
            ln_d = d[hi_g]["ln"][bidx].mean(1)[:, j] - d[lo_g]["ln"][bidx].mean(1)[:, j]
            L.append(f"- Split of A (not part of the rule): carry {f3(q(drate[:, 0]))}, downstream net "
                     f"{f3(q(drate[:, 1:].sum(1)))}, LN-scale term {f3(q(ln_d))}.")
            if q(A)[1] > 0:
                late = sum(drate[:, ix[m]] for m in LATE)
                share = late / A
                L.append(f"- Late MLPs (15, 18, 21, 22): rate difference {f3(q(late))}, share of A {f3(q(share))}. "
                         f"**Late-MLP contribution: {'yes' if share[0] >= 0.5 else 'no'}.**")
            else:
                L.append("- A's CI is not above 0: no component shares computed.")
            for r in ("Obar", "by"):
                jj = RN.index(r)
                dr = pm[hi_g][:, :, jj] - pm[lo_g][:, :, jj]
                top = np.argsort(-np.abs(dr[0]))[:12]
                L += ["", f"Largest rate changes, [0.90, 1.50] − R_p, on {r} ({version}):", "",
                      "| Component | Rate in R_p | Rate in [0.90, 1.50] | Difference [CI] |", "|---|---:|---:|---|"]
                for t in top:
                    L.append(f"| {names[t]} | {pm[lo_g][0, t, jj]:+.3f} | {pm[hi_g][0, t, jj]:+.3f} | "
                             f"{f3(q(dr[:, t]), 3)} |")
                if r == "by":
                    mid = sum(dr[:, ix[m]] for m in BYMID)
                    L.append(f"\nMid MLPs 11, 13, 14, 16, 17 together: {f3(q(mid), 3)}; MLP23: "
                             f"{f3(q(dr[:, ix['MLP23']]), 3)}.")
            L.append("")
    pd.DataFrame(rows).to_csv(Path(args.out_dir) / "dose_decomp_summary.csv", index=False)
    # absolute levels (curve check)
    L += ["## Levels at the grid points (bad passives, mean over pairs)", "",
          "| z | z8 check | Ō | log P(O) | log P(\" by\") | LSE_O | LSE_all |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for j, zv in enumerate(z["grid"]):
        n = z[f"bad_passive_level{j}_n"][:, None]
        g = lambda k: (z[f"bad_passive_level{j}_{k}"] / n).mean(0)
        zc, lp, ls, z8 = g("zc"), g("lp"), g("lse"), g("z8")
        L.append(f"| {zv:.3f} | {z8[0]:.3f} | {zc[RN.index('Obar')]:+.3f} | {lp[LP.index('O')]:.3f} | "
                 f"{lp[LP.index('by')]:.3f} | {ls[0]:.3f} | {ls[1]:.3f} |")
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="results/das_round2/round3/dose_decomp_site8.npz")
    ap.add_argument("--out-dir", default="results/das_round2/round3")
    ap.add_argument("--report", default="reports/passive_das_prep/round3_dose_decomp.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
