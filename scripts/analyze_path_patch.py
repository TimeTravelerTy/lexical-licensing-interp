#!/usr/bin/env python3
"""Path patching of the frame switch (round3_plan.md, B7): analysis.

Input: `path_patch_b7.npz` (`run_path_patch.py`), per half (A = 0, B = 1) and
pair: sums of Delta^was, Delta^has, Delta^had, single-component path effects
and joint-group path effects for each flagged MLP, plus the was-run attention
from the participle to the auxiliary. Pair means first, then the mean over
pairs; 95% CIs: pair bootstrap (2,000 draws, seed 17).

Gate: sign(Delta_l^has) = sign of the step-1 active mean (from
`mechanism/decomp_site8.npz`, T donors) for >= 7 of 9 MLPs. Eligible MLPs:
|S_l| >= 0.05. H1 / alternative / H2 / H3 as declared.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_site8_mechanism import comp_names, frame_arrays

RN = ["by", "dot", "the", "him", "Obar"]


def pmean(z, key, half, idx=None):
    """Per-pair means for one half (or both halves pooled: half=None) -> [n_pairs, ...]."""
    n = z["n"] if half is None else z["n"][half:half + 1]
    v = z[key] if half is None else z[key][half:half + 1]
    num, den = v.sum(0), n.sum(0)
    ok = den > 0
    return num[ok] / den[ok].reshape((-1,) + (1,) * (num.ndim - 1))


def boot(x, B, rng):
    idx = np.vstack([np.arange(len(x))[None], rng.integers(0, len(x), (B, len(x)))])
    return x[idx].mean(1)


def q(d):
    return float(d[0]), float(np.percentile(d[1:], 2.5)), float(np.percentile(d[1:], 97.5))


def run(args):
    z = np.load(args.npz)
    meta = json.loads(Path(args.meta).read_text())
    flagged = [int(l) for l in z["flagged"]]
    readouts = dict(zip(flagged, z["readouts"]))
    top5 = [int(i) for i in z["top5"]]
    H = 16
    rng = np.random.default_rng(args.seed)
    # step-1 active means (T donors, held-out intransitive actives)
    m = np.load(args.step1, allow_pickle=True)
    fa = frame_arrays(m, "active", "T")
    names1 = comp_names(int(m["layers"][0]), len(m["layers"]), 16)
    act_mean = {l: float(fa["comps"][:, names1.index(f"MLP{l}"), RN.index(readouts[l])].mean()) for l in flagged}

    rows, L = [], ["# B7. Path patching of the frame switch in the flagged MLPs", "",
                   "Spec: `round3_plan.md`, B7. Run: `run_path_patch.py` (fp32); analysis: `analyze_path_patch.py`. "
                   f"{meta['rows']} patched rows ({meta['items']} items, {meta['pairs']} pairs; split-0 basis, T donors). "
                   "Units: the MLP's direct contribution to its readout (centered logit), final-LN scale of the was "
                   "patched run. Pair means, then the mean over pairs; 95% CI over pairs.", "",
                   f"Exactness (all components replaced vs S_l, max |error| per row): "
                   + ", ".join(f"MLP{l} {v:.1e}" for l, v in meta["max_exactness_error"].items()), ""]
    S = {}
    for l in flagged:
        d = {k: pmean(z, f"{k}_{l}", None) for k in ("dwas", "dhas", "dhad")}
        g = {k: pmean(z, f"grp_{k}_{l}", None) for k in ("heads", "mlps", "carry", "all")}
        singles = pmean(z, f"pe_{l}", None)
        s = d["dhas"] - d["dwas"]
        S[l] = float(s.mean())
        inter = s - g["heads"] - g["mlps"] - g["carry"]
        row = {"mlp": l, "readout": readouts[l], "step1_active": act_mean[l]}
        for k, v in (("d_was", d["dwas"]), ("d_has", d["dhas"]), ("d_had", d["dhad"]), ("S", s),
                     ("S_had", d["dhad"] - d["dwas"]), ("PE_heads", g["heads"]), ("PE_mlps", g["mlps"]),
                     ("PE_carry", g["carry"]), ("PE_all", g["all"]), ("interaction", inter),
                     ("sum_singles", singles.sum(1))):
            e, lo, hi = q(boot(v, args.n_boot, rng))
            row.update({k: e, f"{k}_lo": lo, f"{k}_hi": hi})
        rows.append(row)
    res = pd.DataFrame(rows).set_index("mlp")
    gate_n = int(sum(np.sign(res.d_has[l]) == np.sign(res.step1_active[l]) for l in flagged))
    elig = [l for l in flagged if abs(S[l]) >= 0.05]
    f3 = lambda r, k: f"{r[k]:+.3f} [{r[k + '_lo']:+.3f}, {r[k + '_hi']:+.3f}]"
    L += ["## The switch per flagged MLP (both halves)", "",
          "| MLP | Readout | Step-1 active | Δ was | Δ has | Δ had | S = has − was | PE heads (joint) | PE MLPs (joint) | "
          "PE carry | Interaction | Σ single PEs |", "|---|---|---:|---:|---:|---:|---|---|---|---|---|---:|"]
    for l in flagged:
        r = res.loc[l]
        L.append(f"| {l} | {readouts[l]} | {r.step1_active:+.3f} | {r.d_was:+.3f} | {r.d_has:+.3f} | {r.d_had:+.3f} | "
                 f"{f3(r, 'S')} | {f3(r, 'PE_heads')} | {f3(r, 'PE_mlps')} | {f3(r, 'PE_carry')} | "
                 f"{r.interaction:+.3f} | {r.sum_singles:+.3f} |")
    L += ["", f"**Gate:** Δ has has the step-1 active sign for {gate_n} of 9 MLPs "
              f"({'passes' if gate_n >= 7 else 'fails'}; needs ≥ 7). Eligible MLPs (|S| ≥ 0.05): {elig}.", ""]
    # H1 / alternative
    h1 = [l for l in elig if np.sign(S[l]) * res.PE_heads[l] >= 0.5 * abs(S[l])]
    alt = [l for l in elig if np.sign(S[l]) * res.PE_mlps[l] >= 0.5 * abs(S[l])]
    need = int(np.ceil(len(elig) / 2))
    L += [f"- **H1 (heads carry the switch):** sign(S)·PE_heads ≥ 0.5|S| for {len(h1)} of {len(elig)} eligible "
          f"({h1}) → **{'holds' if len(h1) >= need else 'fails'}** (needs ≥ {need}).",
          f"- **Alternative (earlier MLPs):** sign(S)·PE_MLPs ≥ 0.5|S| for {len(alt)} of {len(elig)} ({alt}) → "
          f"**{'holds' if len(alt) >= need else 'fails'}**."]
    # H2 / H3 on half B
    hb = {}
    for l in elig:
        hb[l] = {k: pmean(z, f"grp_{k}_{l}", 1).mean() for k in ("heads", "top5", "top5_aux", "top5_val", "top5_pat")}
        hb[l]["S"] = (pmean(z, f"dhas_{l}", 1) - pmean(z, f"dwas_{l}", 1)).mean()
    h2 = [l for l in elig if np.sign(S[l]) * hb[l]["top5"] >= 0.5 * np.sign(S[l]) * hb[l]["heads"]
          and np.sign(S[l]) * hb[l]["heads"] > 0]
    att = pmean(z, "att_aux_was", 1).mean(0)  # [L, H]
    att5 = {i: float(att[i // H, i % H]) for i in top5}
    h3a = all(v >= 0.3 for v in att5.values())
    # an MLP counts only if the top-5 joint PE is large enough to support a ratio (>= 0.05)
    h3b = [l for l in elig if np.sign(S[l]) * hb[l]["top5"] >= 0.05
           and np.sign(S[l]) * hb[l]["top5_aux"] >= 0.5 * np.sign(S[l]) * hb[l]["top5"]]
    L += [f"- **H2 (few heads), half B:** top 5 heads from half A = "
          + ", ".join(f"L{i // H}H{i % H}" for i in top5)
          + f"; their joint PE ≥ 0.5 × all heads for {len(h2)} of {len(elig)} ({h2}) → "
            f"**{'holds' if len(h2) >= need else 'fails'}**.",
          "- **H3 (they read the auxiliary), half B:** was-run attention to the auxiliary: "
          + ", ".join(f"L{i // H}H{i % H} {v:.2f}" for i, v in att5.items())
          + f" (all ≥ 0.3: {'yes' if h3a else 'no'}); aux-value-only ≥ 50% of the top-5 PE (counted only where "
            f"the top-5 PE is ≥ 0.05) for {len(h3b)} of "
            f"{len(elig)} ({h3b}) → **{'holds' if h3a and len(h3b) >= need else ('unresolved' if h3a else 'fails')}**.",
          "", "Half B, per eligible MLP (signed by S):", "",
          "| MLP | S (half B) | PE all heads | PE top 5 | aux value only | all values | pattern only |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for l in elig:
        s_ = np.sign(S[l])
        L.append(f"| {l} | {hb[l]['S']:+.3f} | {s_ * hb[l]['heads']:+.3f} | {s_ * hb[l]['top5']:+.3f} | "
                 f"{s_ * hb[l]['top5_aux']:+.3f} | {s_ * hb[l]['top5_val']:+.3f} | {s_ * hb[l]['top5_pat']:+.3f} |")
    # top single components per MLP and pooled head ranking
    L += ["", "## Largest single-component path effects (both halves, signed by S)", ""]
    for l in flagged:
        pe = pmean(z, f"pe_{l}", None).mean(0)
        nm = ["carry"] + [f"L{i // H}H{i % H}" for i in range(l * H)] + [f"MLP{i}" for i in range(l)]
        top = np.argsort(-np.sign(S[l]) * pe)[:8]
        L.append(f"- MLP{l} (S {S[l]:+.3f}): " + ", ".join(f"{nm[t]} {np.sign(S[l]) * pe[t]:+.3f}" for t in top))
    score = np.zeros(24 * H)
    for l in elig:
        pe = pmean(z, f"pe_{l}", None).mean(0)
        score[: l * H] += np.sign(S[l]) * pe[1:1 + l * H] / abs(S[l])
    att_all = pmean(z, "att_aux_was", None).mean(0)
    top = np.argsort(-score)[:15]
    L += ["", "Heads ranked over eligible MLPs (Σ sign(S)·PE/|S|, both halves) with their was-run attention to the "
              "auxiliary:", "", "| Head | Score | Attention to aux |", "|---|---:|---:|"]
    for t in top:
        L.append(f"| L{t // H}H{t % H} | {score[t]:+.3f} | {att_all[t // H, t % H]:.2f} |")
    res.to_csv(Path(args.out_dir) / "path_patch_summary.csv")
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="results/das_round2/round3/path_patch_b7.npz")
    ap.add_argument("--meta", default="results/das_round2/round3/path_patch_meta_b7.json")
    ap.add_argument("--step1", default="results/das_round2/mechanism/decomp_site8.npz")
    ap.add_argument("--out-dir", default="results/das_round2/round3")
    ap.add_argument("--report", default="reports/passive_das_prep/round3_path_patch.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
