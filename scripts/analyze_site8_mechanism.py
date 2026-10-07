#!/usr/bin/env python3
"""Analyse the site-8 mechanism run (followup_plan.md, steps 1 and 2).

Step 1: per frame (passive / active) and donor class, component contributions
are averaged within pair, then over pairs. Per readout: sum of positive terms
(P), sum of negative terms (N), net, LN-scale term, residual, actual; the
story rule on O-bar; component flags; downstream coordinates. 95% CIs:
bootstrap over pairs (each frame resampled on its own), 2,000 draws, seed 17.

Step 2: dose-response curves f_X(z) (pair means, then the mean over pairs),
two-way bootstrap over pairs and contexts (subjects for actives); slopes over
R_p = [0.10, 0.45] and R_a = [0.45, 1.50]; the overshoot / coupled rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_passive_test import Boot, cells, pair_values

RN = ["by", "dot", "the", "him", "Obar"]
RLAB = {"by": '" by"', "dot": '"."', "the": '" the"', "him": '" him"', "Obar": "Ō (object starts)"}
LP = ["M", "O", "I", "det", "pron", "refl", "by", "dot", "the", "him"]
COORDS = ["z8_u", "z8_p", "z12_u", "z12_p", "z17_u", "z17_p"]


def comp_names(L0, nl, H):
    return ["carry"] + [f"L{L0 + l}H{h}" for l in range(nl) for h in range(H)] + [f"MLP{L0 + l}" for l in range(nl)]


def frame_arrays(z, frame, cond):
    n = z[f"{frame}_{cond}_n"]
    keep = n > 0
    k = lambda name: z[f"{frame}_{cond}_{name}"][keep] / n[keep].reshape((-1,) + (1,) * (z[f"{frame}_{cond}_{name}"].ndim - 1))
    heads, mlp, carry = k("heads"), k("mlp"), k("carry")
    P = heads.shape[0]
    comps = np.concatenate([carry[:, None, :], heads.reshape(P, -1, 5), mlp], 1)  # [pairs, 273, 5]
    return {"comps": comps, "ln": k("ln"), "resid": k("resid"), "actual": k("actual"), "coords": k("coords"),
            "dlp": k("dlp"), "pairs": z["pairs"][keep]}


def psums(c):
    """c [..., n_comp] population-mean contributions -> P, N, net."""
    return np.clip(c, 0, None).sum(-1), np.clip(c, None, 0).sum(-1), c.sum(-1)


def boot_idx(n, B, rng):
    return np.vstack([np.arange(n)[None], rng.integers(0, n, (B, n))])


def ci(x):
    return {"est": float(x[0]), "lo": float(np.percentile(x[1:], 2.5)), "hi": float(np.percentile(x[1:], 97.5))}


def f3(d, k=2):
    return f"{d['est']:+.{k}f} [{d['lo']:+.{k}f}, {d['hi']:+.{k}f}]"


def decomposition(args, L):
    z = np.load(Path(args.out_dir) / "decomp_site8.npz", allow_pickle=True)
    L0, nl = int(z["layers"][0]), len(z["layers"])
    names = comp_names(L0, nl, 16)
    rng = np.random.default_rng(args.seed)
    fr = {(f, c): frame_arrays(z, f, c) for f in ("passive", "active") for c in ("T", "I")}
    idx = {k: boot_idx(len(v["pairs"]), args.n_boot, rng) for k, v in fr.items() if k[1] == "T"}
    # population-mean contributions per draw: [B+1, 273, 5]
    pm = {k: v["comps"][idx[(k[0], "T")]].mean(1) for k, v in fr.items()}
    rows = []
    L += ["## Step 1: signed direct-logit decomposition (T-donor patch)", "",
          "Centered logits through the final LayerNorm, with the LN scale frozen at the **patched** run (fp32). "
          "Terms: the carry (the site-8 displacement itself), 256 heads and 16 MLPs in layers 8–23. "
          f"Pairs: passive {len(fr[('passive', 'T')]['pairs'])} (primary bad passive bases), active "
          f"{len(fr[('active', 'T')]['pairs'])} (held-out intransitive actives). Mean over pairs; 95% CI over pairs.", "",
          "| Frame | Readout | P (positive terms) | N (negative terms) | Net | LN-scale term | Residual | Actual Δ |",
          "|---|---|---|---|---|---|---|---|"]
    for frame in ("active", "passive"):
        v = fr[(frame, "T")]
        ix = idx[(frame, "T")]
        for j, r in enumerate(RN):
            P_, N_, net = psums(pm[(frame, "T")][:, :, j].copy())
            lnv, res, act = (v[k][ix][:, :, j].mean(1) for k in ("ln", "resid", "actual"))
            row = {"frame": frame, "readout": r, "P": ci(P_), "N": ci(N_), "net": ci(net), "ln": ci(lnv),
                   "resid": ci(res), "actual": ci(act)}
            rows.append(row)
            L.append(f"| {frame} | {RLAB[r]} | {f3(row['P'])} | {f3(row['N'])} | {f3(row['net'])} | {f3(row['ln'])} | "
                     f"{f3(row['resid'], 3)} | {f3(row['actual'])} |")
    # story rule on O-bar
    j = RN.index("Obar")
    Pa, Na, _ = psums(pm[("active", "T")][:, :, j].copy())
    Pp, Np, _ = psums(pm[("passive", "T")][:, :, j].copy())
    s, ca, cp = Pp / Pa, -Na / Pa, -Np / Pp
    s_ci, dc_ci = ci(s), ci(cp - ca)
    if s[0] < 0.5 and cp[0] - ca[0] <= 0.2:
        story = "story 1: nothing downstream pushes toward objects in passives"
    elif s[0] >= 0.5 and cp[0] - ca[0] > 0.2:
        story = "story 2: object pushes survive but are cancelled"
    elif s[0] < 0.5:
        story = "both: pushes shrink and are cancelled"
    else:
        story = "neither (check LN term and residual)"
    L += ["", f"**Decision rule on Ō:** survival s = P_passive / P_active = {f3(s_ci)}; cancellation c = |N|/P: "
          f"active {ca[0]:.2f}, passive {cp[0]:.2f} (c_p − c_a = {f3(dc_ci)}). **Outcome: {story}.**", ""]
    # flags
    fl = []
    for r in ("Obar", "the", "him"):
        j = RN.index(r)
        a, p = pm[("active", "T")][0, :, j], pm[("passive", "T")][0, :, j]
        for i, nm in enumerate(names):
            rel_a, rel_p = abs(a[i]) >= 0.05, abs(p[i]) >= 0.05
            if not (rel_a or rel_p):
                continue
            flag = ""
            if rel_a and rel_p and np.sign(a[i]) != np.sign(p[i]):
                flag = "sign change"
            elif rel_a and abs(p[i]) < 0.25 * abs(a[i]):
                flag = "vanishes"
            elif rel_p and abs(a[i]) < 0.25 * abs(p[i]):
                flag = "appears"
            fl.append({"readout": r, "component": nm, "active": float(a[i]), "passive": float(p[i]), "flag": flag})
    fl = pd.DataFrame(fl)
    fl.to_csv(Path(args.out_dir) / "decomp_components_site8.csv", index=False)
    L += ["### Object-relevant components (|mean| ≥ 0.05 in either frame)", "",
          "| Readout | Relevant | Sign change | Vanishes | Appears | Same sign, kept |", "|---|---:|---:|---:|---:|---:|"]
    for r, g in fl.groupby("readout", sort=False):
        L.append(f"| {RLAB[r]} | {len(g)} | {(g.flag == 'sign change').sum()} | {(g.flag == 'vanishes').sum()} | "
                 f"{(g.flag == 'appears').sum()} | {(g.flag == '').sum()} |")
    L += ["", "Flagged and top components on Ō (sorted by |active|; full list in `decomp_components_site8.csv`):", "",
          "| Component | Active | Passive | Flag |", "|---|---:|---:|---|"]
    g = fl[fl.readout == "Obar"].assign(m=lambda d: d.active.abs()).sort_values("m", ascending=False)
    for r in g.head(args.top).itertuples():
        L.append(f"| {r.component} | {r.active:+.3f} | {r.passive:+.3f} | {r.flag} |")
    # same for " by"
    j = RN.index("by")
    a, p = pm[("active", "T")][0, :, j], pm[("passive", "T")][0, :, j]
    top = np.argsort(-np.maximum(abs(a), abs(p)))[:args.top]
    L += ["", 'Top components on " by" (by max |mean| over frames):', "", "| Component | Active | Passive |",
          "|---|---:|---:|"]
    for i in top:
        L.append(f"| {names[i]} | {a[i]:+.3f} | {p[i]:+.3f} |")
    # by layer
    L += ["", "Net contribution by layer (heads summed, MLP separate), T donors, mean over pairs:", "",
          "| Layer | Heads → Ō active | passive | MLP → Ō active | passive | Heads → \" by\" active | passive | "
          "MLP → \" by\" active | passive |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    jo, jb = RN.index("Obar"), RN.index("by")
    for l in range(nl):
        hs = slice(1 + l * 16, 1 + (l + 1) * 16)
        m = 1 + nl * 16 + l
        vals = []
        for jj in (jo, jb):
            for kind in ("h", "m"):
                for frame in ("active", "passive"):
                    c = pm[(frame, "T")][0, :, jj]
                    vals.append(c[hs].sum() if kind == "h" else c[m])
        L.append(f"| {L0 + l} | " + " | ".join(f"{x:+.3f}" for x in vals) + " |")
    L.append(f"| carry | | | | | | | | |")
    L[-1] = (f"| carry | {pm[('active', 'T')][0, 0, jo]:+.3f} (Ō) | {pm[('passive', 'T')][0, 0, jo]:+.3f} (Ō) | | | "
             f"{pm[('active', 'T')][0, 0, jb]:+.3f} (by) | {pm[('passive', 'T')][0, 0, jb]:+.3f} (by) | | |")
    # coordinates
    L += ["", "### Downstream d-coordinates under the site-8 patch", "",
          "z units of each site's basis for the same split and fold (0 = held-out active intransitive, 1 = transitive).",
          "", "| Frame | Donor | z8 before → after | z12 before → after | z17 before → after | Δz12/Δz8 | Δz17/Δz8 |",
          "|---|---|---|---|---|---|---|"]
    crow = []
    for frame in ("active", "passive"):
        for cond in ("T", "I"):
            v = fr[(frame, cond)]
            ix = idx[(frame, "T")] if cond == "T" else boot_idx(len(v["pairs"]), args.n_boot, rng)
            c = v["coords"][ix].mean(1)  # [B+1, 6]
            d8, d12, d17 = c[:, 1] - c[:, 0], c[:, 3] - c[:, 2], c[:, 5] - c[:, 4]
            r12, r17 = ci(d12 / d8), ci(d17 / d8)
            crow.append({"frame": frame, "cond": cond, **{k: float(c[0, i]) for i, k in enumerate(COORDS)},
                         "r12": r12, "r17": r17, "d17": ci(d17)})
            small = abs(d8[0]) < 0.05  # ratio undefined when the patch barely moves z8
            L.append(f"| {frame} | {cond} | {c[0, 0]:.2f} → {c[0, 1]:.2f} | {c[0, 2]:.2f} → {c[0, 3]:.2f} | "
                     f"{c[0, 4]:.2f} → {c[0, 5]:.2f} | {'—' if small else f3(r12)} | {'—' if small else f3(r17)} |")
    ct = {(r["frame"], r["cond"]): r for r in crow}
    ratio = ct[("passive", "T")]["r17"]["est"] / ct[("active", "T")]["r17"]["est"]
    L += ["", f"Passive/active ratio of Δz17/Δz8 (T donors): {ratio:.2f} (predicted ≤ 0.4); passive Δz17 = "
          f"{f3(ct[('passive', 'T')]['d17'])} (predicted ≈ 0.1).", ""]
    # fp32 vs bf16 main test
    tr = pd.read_csv(Path(args.passive_dir) / "transfer_summary.csv")
    tb = tr[(tr.population == "primary (plain)") & (tr.band == "all") & (tr.condition == "T_bad") & (tr.site == 8)] \
        .set_index("readout").est
    dl = fr[("passive", "T")]["dlp"].mean(0)
    L += ["fp32 check (passive, T donors, Δ log P, mean over pairs): " + ", ".join(
        f"{k} {dl[LP.index(k)]:+.3f} (bf16 main test {tb[k]:+.3f})" for k in ("O", "by", "dot")) + ".", ""]
    # D supplement
    L += ["### Supplement: D decomposition (T − I donors), P / N / net", "", "| Frame | Readout | P | N | Net |",
          "|---|---|---:|---:|---:|"]
    for frame in ("active", "passive"):
        cT, cI = fr[(frame, "T")], fr[(frame, "I")]
        common = np.intersect1d(cT["pairs"], cI["pairs"])
        dT = cT["comps"][np.isin(cT["pairs"], common)].mean(0)
        dI = cI["comps"][np.isin(cI["pairs"], common)].mean(0)
        for j, r in enumerate(RN):
            P_, N_, net = psums((dT - dI)[:, j])
            L.append(f"| {frame} | {RLAB[r]} | {P_:+.2f} | {N_:+.2f} | {net:+.2f} |")
    L.append("")
    pd.json_normalize(rows).to_csv(Path(args.out_dir) / "decomp_summary_site8.csv", index=False)
    pd.json_normalize(crow).to_csv(Path(args.out_dir) / "decomp_coords_site8.csv", index=False)
    return L


def dose(args, L):
    d = pd.read_parquet(Path(args.out_dir) / "dose_site8.parquet")
    d["context_id"] = d.item_id.str.split("|").str[1]
    grid = np.sort(d.z.unique())
    vals = LP + ["z12", "z17"]
    out = {}
    for frame, g in d.groupby("frame"):
        w = g.pivot_table(index=["item_id", "pair", "context_id"], columns="z", values=vals).reset_index()
        cols = [(v, zz) for v in vals for zz in grid]
        w.columns = ["item_id", "pair_id", "context_id"] + [f"{v}@{zz}" for v, zz in w.columns[3:]]
        vc = [f"{v}@{zz}" for v, zz in cols]
        pairs = sorted(w.pair_id.unique())
        ctxs = sorted(w.context_id.unique())
        bt = Boot(len(ctxs), args.n_boot, args.seed)
        v, m = cells(w.assign(dq=0), {p: i for i, p in enumerate(pairs)}, {c: i for i, c in enumerate(ctxs)},
                     {0: 0}, 1, vc)
        A = pair_values(v, m, bt.wc, np.ones((bt.B + 1, 1), np.float32))  # [B+1, P, V]
        wp = bt.pair_weights(np.zeros(len(pairs)))
        S = (A * wp[..., None]).sum(1) / wp.sum(1)[:, None]
        out[frame] = S.reshape(bt.B + 1, len(vals), len(grid))

    def slope(curve, lo, hi):
        sel = (grid >= lo - 1e-9) & (grid <= hi + 1e-9)
        x = grid[sel] - grid[sel].mean()
        return (curve[..., sel] * x).sum(-1) / (x ** 2).sum()

    L += ["## Step 2: dose-response at site 8", "",
          "The site-8 coordinate is set to z (split-0 basis, sign-aligned; 0 = active intransitive level, 1 = active "
          "transitive level). Curves: mean over pairs; 95% CIs over pairs and contexts (subjects for actives). "
          "R_p = [0.10, 0.45] (passive range), R_a = [0.45, 1.50].", "",
          "| Frame | S_by(R_p) | S_O(R_p) | S_O(R_a) | S_by(R_a) | S_O(R_p)/S_O(R_a) | R² by on R_p | Outcome |",
          "|---|---|---|---|---|---|---:|---|"]
    srow = []
    for frame in ("bad_passive", "good_passive", "intrans_active", "trans_active"):
        S = out[frame]
        by, O = S[:, vals.index("by")], S[:, vals.index("O")]
        sbp, sop, soa, sba = slope(by, .10, .45), slope(O, .10, .45), slope(O, .45, 1.5), slope(by, .45, 1.5)
        ratio = sop / soa
        sel = (grid >= .1 - 1e-9) & (grid <= .45 + 1e-9)
        yb = by[0, sel]
        fit = np.polyval(np.polyfit(grid[sel], yb, 1), grid[sel])
        r2 = 1 - ((yb - fit) ** 2).sum() / ((yb - yb.mean()) ** 2).sum()
        a = {k: ci(x) for k, x in (("sbp", sbp), ("sop", sop), ("soa", soa), ("sba", sba), ("ratio", ratio))}
        if a["sbp"]["lo"] > 0 and a["soa"]["lo"] > 0 and ratio[0] <= 0.25:
            oc = "overshoot"
        elif ratio[0] >= 0.5:
            oc = "coupled"
        else:
            oc = "intermediate"
        srow.append({"frame": frame, **{k: v["est"] for k, v in a.items()}, "r2_by_Rp": float(r2), "outcome": oc})
        L.append(f"| {frame} | {f3(a['sbp'])} | {f3(a['sop'])} | {f3(a['soa'])} | {f3(a['sba'])} | {f3(a['ratio'])} | "
                 f"{r2:.3f} | {oc if frame == 'bad_passive' else '(reference)'} |")
    L += ["", "Declared outcome applies to bad passives. Prediction: coupled; by ≈ +0.8/z and O ≈ +0.4/z; R² ≥ 0.95.", "",
          "Selected points (bad passives; log P, mean over pairs):", "",
          "| z | log P(\" by\") | log P(O) | log P(\".\") | log P(pron) | z12 | z17 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    S = out["bad_passive"][0]
    for zz in (-0.5, 0.0, 0.1, 0.25, 0.45, 0.6, 0.9, 1.2, 1.5, 2.0):
        k = int(np.argmin(abs(grid - zz)))
        L.append(f"| {grid[k]:.2f} | {S[vals.index('by'), k]:.2f} | {S[vals.index('O'), k]:.2f} | "
                 f"{S[vals.index('dot'), k]:.2f} | {S[vals.index('pron'), k]:.2f} | {S[vals.index('z12'), k]:.2f} | "
                 f"{S[vals.index('z17'), k]:.2f} |")
    L.append("")
    curves = pd.concat([pd.DataFrame(out[f][0].T, columns=vals).assign(frame=f, z=grid) for f in out])
    curves.to_csv(Path(args.out_dir) / "dose_curves_site8.csv", index=False)
    pd.DataFrame(srow).to_csv(Path(args.out_dir) / "dose_slopes_site8.csv", index=False)
    return L


def run(args):
    meta = json.loads((Path(args.out_dir) / "mechanism_meta_site8.json").read_text())
    L = ["# Site-8 mechanism: decomposition and dose-response", "",
         "Spec and predictions: `followup_plan.md` (steps 1–2), committed before the run. Run: "
         "`run_site8_mechanism.py` (fp32, LN scale frozen at the patched run); analysis: "
         f"`analyze_site8_mechanism.py`. {meta['n_patches']} decomposed patches.", ""]
    L = decomposition(args, L)
    L = dose(args, L)
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="results/das_round2/mechanism")
    ap.add_argument("--passive-dir", default="results/das_round2/passive_test")
    ap.add_argument("--report", default="reports/passive_das_prep/site8_mechanism_results.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--top", type=int, default=15)
    run(ap.parse_args())
