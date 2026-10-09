#!/usr/bin/env python3
"""Round 4, D10: clean voice contrast ("has been V" vs "has V") (reports/round4/plan.md, D10).

From `run_path_patch_frames.py been` with T donors (B7-comparable) and with I donors. Reuses the A5 frame
analysis (gate, eligibility, H1, the B7-heads criteria) and adds the D10 rules: voice switch (S has B7's sign
and >= 50% of B7's |S| for >= 7 of 9 MLPs, and the T - I version S_D = S_T - S_I has the same sign for >= 7 of
9: the frames respond differently to the same coordinate change), not an auxiliary-token effect (|S_sens| <
0.5 |S| for >= 7 of 9), and the two-stage routing on eligible MLPs (|S| >= 0.05; heads >= 0.5 |S| for >= 3 of
the 5 " by" MLPs; joint earlier MLPs >= 0.5 |S| for >= 3 of the 4 object MLPs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_aux_heads import BY, analyse_frame, pm, pm_b7
from analyze_path_patch import RN
from analyze_site8_mechanism import comp_names, frame_arrays

OBJ = (15, 18, 21, 22)


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
    S7 = {l: float((pm_b7(zb7, f"dhas_{l}") - pm_b7(zb7, f"dwas_{l}")).mean()) for l in flagged}
    S7B = {l: float((pm_b7(zb7, f"dhas_{l}", 1) - pm_b7(zb7, f"dwas_{l}", 1)).mean()) for l in flagged}
    b7 = {l: float((np.sign(S7B[l]) * pm_b7(zb7, f"grp_top5_{l}", 1)).mean() /
                   (np.sign(S7B[l]) * pm_b7(zb7, f"grp_heads_{l}", 1)).mean()) for l in flagged}
    z = np.load(d / "path_patch_been.npz")
    meta = json.loads((d / "path_patch_meta_been.json").read_text())
    o, tab = analyse_frame(argparse.Namespace(n_boot=args.n_boot, dir=args.dir), "been", z, meta, act_mean, b7, rng)
    a = tab[tab.group == "all"].set_index("mlp")
    zI = np.load(d / "path_patch_been_I.npz")
    fr = list(z["frames"])
    SD, SDs, hD = {}, {}, {}
    for l in flagged:
        sT = pm(z, f"d{fr[1]}_{l}") - pm(z, f"d{fr[0]}_{l}")
        sI = pm(zI, f"d{fr[1]}_{l}") - pm(zI, f"d{fr[0]}_{l}")
        v = sT - sI
        bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(args.n_boot)]
        SD[l] = (float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
        SDs[l] = float(((pm(z, f"d{fr[2]}_{l}") - pm(z, f"d{fr[0]}_{l}")) - (pm(zI, f"d{fr[2]}_{l}") - pm(zI, f"d{fr[0]}_{l}"))).mean())
        hD[l] = float((pm(z, f"grp_heads_{l}") - pm(zI, f"grp_heads_{l}")).mean())
    elig = [l for l in flagged if abs(a.S[l]) >= 0.05]
    voice = [l for l in flagged if np.sign(a.S[l]) == np.sign(S7[l]) and abs(a.S[l]) >= 0.5 * abs(S7[l])
             and np.sign(SD[l][0]) == np.sign(a.S[l])]
    notaux = [l for l in flagged if abs(a.S_extra[l]) < 0.5 * abs(a.S[l])]
    h1by = [l for l in BY if l in elig and np.sign(a.S[l]) * a.PE_heads[l] >= 0.5 * abs(a.S[l])]
    altobj = [l for l in OBJ if l in elig and np.sign(a.S[l]) * a.PE_mlps[l] >= 0.5 * abs(a.S[l])]
    dec = {"gate": o["gate"], "gate_n": o["gate_n"], "voice_switch": len(voice) >= 7, "voice_mlps": voice,
           "S_D": SD, "S_D_sens": SDs, "PE_heads_D": hD, "eligible": elig,
           "not_aux": len(notaux) >= 7, "not_aux_mlps": notaux, "routing": len(h1by) >= 3 and len(altobj) >= 3,
           "h1_by": h1by, "alt_obj": altobj, "retain": o["retain"], "new_heads": meta["new_heads"],
           "att_aux": o["att_aux"], "att_adv": o.get("att_adv"), "exactness": meta["max_exactness_error"]}
    tab.to_csv(d / "voice_summary.csv", index=False)
    (d / "voice_decisions.json").write_text(json.dumps(dec, indent=2, default=str) + "\n")
    f = lambda r, k: f"{r[k]:+.3f} [{r[k + '_lo']:+.3f}, {r[k + '_hi']:+.3f}]"
    L = ["# Round 4, D10. Clean voice contrast (\"has been V\" vs \"has V\")", "",
         "Spec: `plan.md`, D10. Run: `run_path_patch_frames.py been` (fp32); analysis: `analyze_voice.py`. B7 items "
         "and rows (site-8 T-donor interchange in every frame). Base = \"The N has been V\", counterfactual = \"The N has "
         "V\", sensitivity = \"The N was V\". Units as B7; pair means, then the mean over pairs; 95% CI over pairs.", "",
         "Exactness (all components replaced vs S, max |error|): " +
         ", ".join(f"MLP{l} {v:.1e}" for l, v in meta["max_exactness_error"].items()), "",
         "| MLP | readout | step-1 active | Δ has been | Δ has | S = has − has been (T donors) | B7 S (has − was) | "
         "S_D (T − I donors) | S_sens = was − has been | PE heads | PE MLPs | half B: fixed top 5 / all heads |",
         "|---|---|---:|---:|---:|---|---:|---|---:|---|---|---:|"]
    for l, r in a.iterrows():
        L.append(f"| {l} | {r.readout} | {r.step1_active:+.3f} | {r.d_base:+.3f} | {r.d_cf:+.3f} | {f(r, 'S')} | "
                 f"{S7[l]:+.3f} | {SD[l][0]:+.3f} [{SD[l][1]:+.3f}, {SD[l][2]:+.3f}] | {r.S_extra:+.3f} | "
                 f"{f(r, 'PE_heads')} | {f(r, 'PE_mlps')} | {r.B_fixed:+.3f} / {r.B_heads:+.3f} |")
    L += ["", "## Declared decisions", "",
          f"- **Gate:** {o['gate_n']} of 9 → **{'passes' if o['gate'] else 'fails'}**.",
          f"- **Voice switch** (B7 sign, ≥ 50% of B7's |S|, and S_D of the same sign): {voice} → "
          f"**{'holds' if dec['voice_switch'] else 'fails'}**.",
          f"- **Not an auxiliary-token effect** (|S_sens| < 0.5 |S|): {notaux} → **{'holds' if dec['not_aux'] else 'fails'}**.",
          f"- **Two-stage routing:** heads ≥ 0.5|S| for by-MLPs {h1by}; earlier MLPs ≥ 0.5|S| for object MLPs {altobj} → "
          f"**{'replicates' if dec['routing'] else 'does not replicate'}**.",
          f"- **B7 heads retain their role:** **{o['retain']}**" + (f" (top 10: {', '.join(o['top10'])}; overlap "
                                                                  f"{o['overlap']})" if "top10" in o else "") + ".",
          f"- New top 5 (half A): {', '.join(meta['new_heads'])}.",
          "- Base-run attention from the participle's last token, fixed top 5: to \"has\" "
          + ", ".join(f"{k} {v:.2f}" for k, v in o["att_aux"].items()) + "; to \"been\" "
          + ", ".join(f"{k} {v:.2f}" for k, v in (o.get("att_adv") or {}).items()) + ".", ""]
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/round4/translation")
    ap.add_argument("--b7", default="results/das_round2/round3/path_patch_b7.npz")
    ap.add_argument("--step1", default="results/das_round2/mechanism/decomp_site8.npz")
    ap.add_argument("--report", default="reports/round4/d10_voice.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
