#!/usr/bin/env python3
"""Round 4, A5: auxiliary heads (reports/round4/plan.md, A5): analysis.

(a) previous-token scores (`run_prev_token.py`) for the B7 top 15 heads.
(b) per frame set (`run_path_patch_frames.py`: adverb, got; plain = code check against B7):
gate, eligible MLPs, H1, "the B7 heads retain their role" (i)-(iii), "read the auxiliary across
the adverb", and the same quantities by participle token group (single / multi-token).
Pair means, then the mean over pairs; 95% CIs: pair bootstrap (2,000 draws, seed 17).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_path_patch import RN, boot, q
from analyze_site8_mechanism import comp_names, frame_arrays

H = 16
B7_TOP = ("L9H7", "L10H2", "L3H3", "L1H6", "L8H4")
BY = (11, 13, 14, 16, 17)


def hix(name):
    l, h = name[1:].split("H")
    return int(l) * H + int(h)


def pm(z, key, half=None, grp=None):
    """Per-pair means; arrays [half, pair, group, ...]; half / grp None = pooled."""
    n, v = z["n"], z[key]
    if half is not None:
        n, v = n[half:half + 1], v[half:half + 1]
    if grp is not None:
        n, v = n[:, :, grp:grp + 1], v[:, :, grp:grp + 1]
    num, den = v.sum((0, 2)), n.sum((0, 2))
    ok = den > 0
    return num[ok] / den[ok].reshape((-1,) + (1,) * (num.ndim - 1))


def pm_b7(z, key, half=None):
    n = z["n"] if half is None else z["n"][half:half + 1]
    v = z[key] if half is None else z[key][half:half + 1]
    num, den = v.sum(0), n.sum(0)
    ok = den > 0
    return num[ok] / den[ok].reshape((-1,) + (1,) * (num.ndim - 1))


def analyse_frame(args, name, z, meta, act_mean, b7, rng):
    frames = list(z["frames"])
    base, cf, extra = frames
    flagged = [int(l) for l in z["flagged"]]
    readouts = dict(zip(flagged, z["readouts"]))
    out = {"frame": name, "frames": frames}
    rows = []
    S = {}
    for grp_label, grp in (("all", None), ("single-token", 0), ("multi-token", 1)):
        for l in flagged:
            d = {k: pm(z, f"d{k}_{l}", None, grp) for k in frames}
            g = {k: pm(z, f"grp_{k}_{l}", None, grp) for k in ("heads", "mlps", "carry")}
            s = d[cf] - d[base]
            hb = {k: pm(z, f"grp_{k}_{l}", 1, grp) for k in ("heads", "fixed", "fixed_aux", "fixed_adv", "fixed_val",
                                                             "fixed_pat", "new", "new_aux")}
            sB = pm(z, f"d{cf}_{l}", 1, grp) - pm(z, f"d{base}_{l}", 1, grp)
            sg = np.sign(s.mean())
            row = {"group": grp_label, "mlp": l, "readout": readouts[l], "step1_active": act_mean[l],
                   "n_pairs": len(s)}
            for k, v in (("d_base", d[base]), ("d_cf", d[cf]), ("d_extra", d[extra]), ("S", s),
                         ("S_extra", d[extra] - d[base]), ("PE_heads", g["heads"]), ("PE_mlps", g["mlps"]),
                         ("PE_carry", g["carry"]), ("S_halfB", sB)):
                e, lo, hi = q(boot(v, args.n_boot, rng))
                row.update({k: e, f"{k}_lo": lo, f"{k}_hi": hi})
            for k, v in hb.items():  # half B, signed by S
                e, lo, hi = q(boot(sg * v, args.n_boot, rng))
                row.update({f"B_{k}": e, f"B_{k}_lo": lo, f"B_{k}_hi": hi})
            rows.append(row)
            if grp is None:
                S[l] = float(s.mean())
    res = pd.DataFrame(rows)
    a = res[res.group == "all"].set_index("mlp")
    gate_n = int(sum(np.sign(a.d_cf[l]) == np.sign(a.step1_active[l]) for l in flagged))
    elig = [l for l in flagged if abs(S[l]) >= 0.05]
    need = int(np.ceil(len(elig) / 2))
    h1 = [l for l in elig if np.sign(S[l]) * a.PE_heads[l] >= 0.5 * abs(S[l])]
    qual = [l for l in BY if l in elig and a.B_heads[l] >= 0.05]
    qual_all = [l for l in elig if a.B_heads[l] >= 0.05]  # signed half-B all-heads PE >= 0.05
    out.update({"gate_n": gate_n, "gate": gate_n >= 7, "eligible": elig, "h1_mlps": h1,
                "h1": ("holds" if len(h1) >= need else "fails") if elig else "unresolved (no eligible MLP)",
                "qualifying_by": qual, "qualifying_all": qual_all})
    if len(qual) < 3:
        out["retain"] = "unresolved (fewer than 3 qualifying by MLPs)"
    else:
        c1 = [l for l in qual if np.sign(S[l]) * a.PE_heads[l] >= 0.5 * abs(S[l])]
        frac = {l: a.B_fixed[l] / a.B_heads[l] for l in qual}
        c2 = [l for l in qual if a.B_fixed_lo[l] > 0 and frac[l] >= 0.5 * b7[l]]
        sc = np.load(Path(args.dir) / f"head_scores_halfA_{name}.npy")
        top10 = [int(i) for i in np.argsort(-sc)[:10]]
        overlap = [n for n in B7_TOP if hix(n) in top10]
        ok = len(c1) >= 3 and len(c2) >= 3 and len(overlap) >= 3
        out.update({"retain_i": c1, "retain_ii": c2, "fraction": frac, "b7_fraction": {l: b7[l] for l in qual},
                    "top10": [f"L{i // H}H{i % H}" for i in top10], "overlap": overlap,
                    "retain": "holds" if ok else "fails"})
    att = pm(z, "att_aux_base").mean(0)  # [L, H]
    out["att_aux"] = {n: float(att[hix(n) // H, hix(n) % H]) for n in B7_TOP}
    out["att_prev"] = {n: float(pm(z, "att_prev_base").mean(0)[hix(n) // H, hix(n) % H]) for n in B7_TOP}
    for gl, gi in (("single", 0), ("multi", 1)):
        out[f"att_aux_{gl}"] = {n: float(pm(z, "att_aux_base", None, gi).mean(0)[hix(n) // H, hix(n) % H]) for n in B7_TOP}
    if "att_adv_base" in z.files:
        out["att_adv"] = {n: float(pm(z, "att_adv_base").mean(0)[hix(n) // H, hix(n) % H]) for n in B7_TOP}
        att_ok = sum(v >= 0.3 for v in out["att_aux"].values()) >= 3
        # qualifying MLPs (declared): eligible with signed all-heads PE >= 0.05; >= 3 qualifying by-MLPs needed.
        # An MLP counts if the fixed-five joint PE is >= 0.05 (as B7) and the aux value gives >= 50% of it.
        auxv = [l for l in qual_all if a.B_fixed[l] >= 0.05 and a.B_fixed_aux[l] >= 0.5 * a.B_fixed[l]]
        if len(qual) < 3:
            ra = "unresolved (fewer than 3 qualifying by MLPs)"
        else:
            ra = "holds" if att_ok and len(auxv) >= int(np.ceil(len(qual_all) / 2)) else "fails"
        out.update({"aux_value_mlps": auxv, "aux_value_candidates": qual_all, "read_aux": ra})
    return out, res


def run(args):
    d = Path(args.dir)
    rng = np.random.default_rng(args.seed)
    m = np.load(args.step1, allow_pickle=True)
    fa = frame_arrays(m, "active", "T")
    names1 = comp_names(int(m["layers"][0]), len(m["layers"]), 16)
    zb7 = np.load(args.b7)
    flagged = [int(l) for l in zb7["flagged"]]
    readouts = dict(zip(flagged, zb7["readouts"]))
    act_mean = {l: float(fa["comps"][:, names1.index(f"MLP{l}"), RN.index(readouts[l])].mean()) for l in flagged}
    S7 = {l: float((pm_b7(zb7, f"dhas_{l}", 1) - pm_b7(zb7, f"dwas_{l}", 1)).mean()) for l in flagged}
    b7 = {l: float((np.sign(S7[l]) * pm_b7(zb7, f"grp_top5_{l}", 1)).mean() /
                   (np.sign(S7[l]) * pm_b7(zb7, f"grp_heads_{l}", 1)).mean()) for l in flagged}
    pt = pd.read_csv(d / "prev_token_scores.csv")
    outs, tabs = {}, []
    for name in args.frames.split(","):
        z = np.load(d / f"path_patch_{name}.npz")
        meta = json.loads((d / f"path_patch_meta_{name}.json").read_text())
        o, r = analyse_frame(args, name, z, meta, act_mean, b7, rng)
        o["max_exactness_error"] = meta["max_exactness_error"]
        o["new_heads"] = meta["new_heads"]
        outs[name] = o
        tabs.append(r.assign(frame=name))
    tab = pd.concat(tabs, ignore_index=True)
    tab.to_csv(d / "aux_heads_summary.csv", index=False)
    (d / "aux_heads_decisions.json").write_text(json.dumps(outs, indent=2, default=str) + "\n")
    # plain frame = code check against B7 (both halves)
    check = None
    if "plain" in outs:
        p = tab[(tab.frame == "plain") & (tab.group == "all")].set_index("mlp")
        sb7 = {l: float((pm_b7(zb7, f"dhas_{l}") - pm_b7(zb7, f"dwas_{l}")).mean()) for l in flagged}
        hb7 = {l: float(pm_b7(zb7, f"grp_heads_{l}").mean()) for l in flagged}
        check = max(max(abs(p.S[l] - sb7[l]), abs(p.PE_heads[l] - hb7[l])) for l in flagged)
    write_report(args, outs, tab, pt, b7, check)


def write_report(args, outs, tab, pt, b7, check):
    f = lambda e, lo, hi: f"{e:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    L = ["# Round 4, A5. Auxiliary heads", "",
         "Spec: `plan.md`, A5. Runs: `run_prev_token.py`, `run_path_patch_frames.py` (fp32); analysis: "
         "`analyze_aux_heads.py`. Units as B7 (the MLP's direct contribution to its readout, final-LN scale of the base "
         "patched run). Pair means, then the mean over pairs; 95% CI over pairs.", "",
         "## (a) Generic previous-token scores", "",
         "| Head | random tokens | natural text | previous-token head | rank (natural, of 384) |", "|---|---:|---:|---|---:|"]
    pt = pt.assign(rank=pt.natural.rank(ascending=False).astype(int))
    b7_15 = ["L9H7", "L10H2", "L3H3", "L1H6", "L8H4", "L3H2", "L3H5", "L15H5", "L14H8", "L12H12", "L4H15", "L3H9",
             "L12H8", "L6H2", "L10H1"]
    pi = pt.set_index("name")
    for n in b7_15:
        r = pi.loc[n]
        L.append(f"| {n}{' (top 5)' if n in B7_TOP else ''} | {r.random:.2f} | {r.natural:.2f} | "
                 f"{'yes' if r.prev_token_head else 'no'} | {r['rank']} |")
    L += ["", f"Previous-token heads overall (≥ 0.5 on both): {int(pt.prev_token_head.sum())} of 384: "
          + ", ".join(pt[pt.prev_token_head].sort_values("natural", ascending=False).name.tolist()[:20]), ""]
    if check is not None:
        L += [f"Code check: the plain frame (\"was\" / \"has\") reproduces B7's S and joint-head PE to within "
              f"{check:.1e}.", ""]
    for name, o in outs.items():
        if name == "plain":
            continue
        t = tab[(tab.frame == name) & (tab.group == "all")].set_index("mlp")
        L += [f"## (b) {name} frame: {' / '.join(o['frames'])}", "",
              "| MLP | readout | step-1 active | Δ base | Δ cf | S | S (sensitivity) | PE heads | PE MLPs | "
              "half B: all heads | fixed top 5 | fixed aux value | fixed adverb value | fixed pattern |",
              "|---|---|---:|---:|---:|---|---:|---|---|---:|---|---:|---:|---:|"]
        for l, r in t.iterrows():
            L.append(f"| {l} | {r.readout} | {r.step1_active:+.3f} | {r.d_base:+.3f} | {r.d_cf:+.3f} | "
                     f"{f(r.S, r.S_lo, r.S_hi)} | {r.S_extra:+.3f} | {f(r.PE_heads, r.PE_heads_lo, r.PE_heads_hi)} | "
                     f"{f(r.PE_mlps, r.PE_mlps_lo, r.PE_mlps_hi)} | {r.B_heads:+.3f} | "
                     f"{f(r.B_fixed, r.B_fixed_lo, r.B_fixed_hi)} | {r.B_fixed_aux:+.3f} | {r.B_fixed_adv:+.3f} | "
                     f"{r.B_fixed_pat:+.3f} |")
        L += ["", f"- **Gate:** {o['gate_n']} of 9 → **{'passes' if o['gate'] else 'fails'}**. Eligible: {o['eligible']}.",
              f"- **H1:** {o['h1_mlps']} → **{o['h1']}**."]
        if "retain_i" in o:
            fr = ", ".join(f"MLP{l} {o['fraction'][l]:.2f} (B7 {o['b7_fraction'][l]:.2f})" for l in o["fraction"])
            L += [f"- **B7 heads retain their role:** (i) H1 on qualifying by-MLPs {o['retain_i']}; (ii) fixed-top-5 "
                  f"CI > 0 and fraction ≥ 0.5 × B7: {o['retain_ii']} (fractions: {fr}); (iii) new top 10 = "
                  f"{', '.join(o['top10'])}; B7 top 5 among them: {o['overlap']} → **{o['retain']}**."]
        else:
            L.append(f"- **B7 heads retain their role:** **{o['retain']}**.")
        L.append(f"- New top 5 (half A): {', '.join(o['new_heads'])}.")
        L.append("- Base-run attention from the participle's last token, fixed top 5: to the auxiliary "
                 + ", ".join(f"{k} {v:.2f}" for k, v in o["att_aux"].items())
                 + "; to the previous token " + ", ".join(f"{k} {v:.2f}" for k, v in o["att_prev"].items())
                 + ("; to the adverb " + ", ".join(f"{k} {v:.2f}" for k, v in o["att_adv"].items()) if "att_adv" in o else "")
                 + ".")
        L.append("- Attention to the auxiliary by participle tokens: single "
                 + ", ".join(f"{k} {v:.2f}" for k, v in o["att_aux_single"].items()) + "; multi "
                 + ", ".join(f"{k} {v:.2f}" for k, v in o["att_aux_multi"].items()) + ".")
        if "read_aux" in o:
            L.append(f"- **Read the auxiliary across the adverb:** aux value ≥ 50% of the fixed joint PE for "
                     f"{o['aux_value_mlps']} of {o['aux_value_candidates']} → **{o['read_aux']}**.")
        g = tab[(tab.frame == name) & (tab.group != "all")]
        L += ["", "By participle token group (S, joint heads, half-B fixed top 5, fixed aux value):", "",
              "| MLP | single: S | single: PE heads | single: fixed | single: aux value | multi: S | multi: PE heads | "
              "multi: fixed | multi: aux value |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for l in sorted(g.mlp.unique()):
            s1 = g[(g.mlp == l) & (g.group == "single-token")].iloc[0]
            s2 = g[(g.mlp == l) & (g.group == "multi-token")].iloc[0]
            L.append(f"| {l} | {s1.S:+.3f} | {s1.PE_heads:+.3f} | {s1.B_fixed:+.3f} | {s1.B_fixed_aux:+.3f} | "
                     f"{s2.S:+.3f} | {s2.PE_heads:+.3f} | {s2.B_fixed:+.3f} | {s2.B_fixed_aux:+.3f} |")
        L.append("")
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/round4/aux_heads")
    ap.add_argument("--frames", default="adverb,got,plain")
    ap.add_argument("--b7", default="results/das_round2/round3/path_patch_b7.npz")
    ap.add_argument("--step1", default="results/das_round2/mechanism/decomp_site8.npz")
    ap.add_argument("--report", default="reports/round4/a5_aux_heads.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
