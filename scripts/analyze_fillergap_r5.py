#!/usr/bin/env python3
"""Round 5, C8: filler-gap gates and patches with non-DAS verbs as the primary set (reports/round5/plan.md, C8).

Reuses the round-4 machinery (`analyze_fillergap.py`) on the fp32 rerun in results/round5/fillergap.
Modes
- gates: natural gates (half B) for the primary population (non-DAS T, kept non-DAS I), the DAS-base
  sensitivities (pooled: all T with kept DAS I; non-DAS T with kept DAS I) and round 4's population (all T, kept
  I) in fp32; per stratum (matrix verb / auxiliary; half B primary, all 6 contexts secondary); whether an fp32
  half-A I gate would keep the same verbs; the B7 base-form gate. Writes gates.json (primary gate drives the
  patch stage), gates_r5.json, and reports/round5/c8_fillergap_gates.md.
- patch: readings per population, site and patch type (see plan.md C8), strata.
Kept I verbs: round 4's lists (data/round4/fillergap/kept_I_{exp}.csv).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_fillergap import RD, SITES, interaction, multi, readouts
from analyze_nonce_passive import z_table
from analyze_passive_test import auc, ci

EXPS = {"emb": "B6 embedded wh (\"I know that / what NAME V-ed\")",
        "mat": "B7 matrix wh (\"AUX NAME V\" / \"What AUX NAME V\")"}


def populations(e, kept):
    das = set(e[e.das_pair.fillna("") != ""].lemma)
    T = set(e[e.cls == "T"].lemma)
    return das, {
        "primary: non-DAS": (T - das, kept - das),
        "DAS bases, pooled": (T, kept & das),
        "DAS bases, non-DAS T": (T - das, kept & das),
        "round-4 population (all T, kept I)": (T, kept),
    }


def run_gates(args):
    ddir, rdir = Path(args.data_dir), Path(args.out_dir)
    items, P = pd.read_csv(ddir / "items.csv"), pd.read_csv(ddir / "prompts.csv")
    nat = pd.read_parquet(rdir / "natural.parquet").set_index("pid")
    x = readouts(items, nat)
    rng = np.random.default_rng(args.seed)
    out = {}
    f = lambda c: f"{c['est']:+.2f} [{c['lo95']:+.2f}, {c['hi95']:+.2f}]"
    L = ["# Round 5, C8: filler-gap natural gates (fp32, non-DAS primary)", "",
         "Spec: `plan.md`, C8. Run: `run_fillergap.py natural --dtype float32` (job 9007314); analysis: "
         "`analyze_fillergap_r5.py gates`. Interaction on L = [T − I]_what − [T − I]_that of L; on O = [T − I]_that − "
         "[T − I]_what of O. T and I verbs resampled independently, contexts jointly (2,000 draws, seed 17). Kept I "
         "verbs: round 4's half-A lists.", ""]
    for exp in ("emb", "mat"):
        e = x[x.exp == exp]
        kept = set(pd.read_csv(ddir / f"kept_I_{exp}.csv").lemma)
        das, pops = populations(e, kept)
        w = e.pivot_table(index=["lemma", "cls", "context_id", "half", "matrix"], columns="frame", values=list(RD))
        dif = pd.DataFrame({r: w[(r, "fill")] - w[(r, "nofill")] for r in RD}).reset_index()
        ctxA = sorted(e[e.half == 0].context_id.unique())
        ctxB = sorted(e[e.half == 1].context_id.unique())
        # fp32 I gate on half A (report only; the kept lists stay round 4's)
        wcA = multi(len(ctxA), args.n_boot, rng)
        g32 = []
        for lem, g in dif[(dif.cls == "I") & (dif.half == 0)].groupby("lemma"):
            v = np.zeros(len(ctxA))
            v[g.context_id.map({c: i for i, c in enumerate(ctxA)}).to_numpy()] = g.PREP.to_numpy()
            g32.append((lem, ci((wcA @ v) / wcA.sum(1))["lo95"] > 0))
        kept32 = {l for l, k in g32 if k}
        res = {}
        for label, (Tset, Iset) in pops.items():
            for cx_label, cx in (("half B", ctxB), ("all contexts", ctxA + ctxB)):
                per = interaction(dif, cx, Tset, Iset, args.n_boot, rng)
                iL = per["T"][:, RD.index("L")] - per["I"][:, RD.index("L")]
                iO = -(per["T"][:, RD.index("O")] - per["I"][:, RD.index("O")])
                res[f"{label} | {cx_label}"] = {"interaction_L": ci(iL), "interaction_O": ci(iO), "n_T": len(Tset),
                                                "n_I": len(Iset), "passes": bool(ci(iL)["lo95"] > 0 and ci(iO)["lo95"] > 0)}
        strata = {}
        Tp, Ip = pops["primary: non-DAS"]
        for mv in sorted(e.matrix.unique()):
            for cx_label, hv in (("half B", [1]), ("all 6", [0, 1])):
                cx = sorted(e[e.half.isin(hv) & (e.matrix == mv)].context_id.unique())
                per = interaction(dif, cx, Tp, Ip, args.n_boot, rng)
                iL = per["T"][:, RD.index("L")] - per["I"][:, RD.index("L")]
                iO = -(per["T"][:, RD.index("O")] - per["I"][:, RD.index("O")])
                strata[f"{mv} | {cx_label}"] = {"interaction_L": ci(iL), "interaction_O": ci(iO), "n_ctx": len(cx),
                                                "passes": bool(ci(iL)["lo95"] > 0 and ci(iO)["lo95"] > 0)}
        passes = res["primary: non-DAS | half B"]["passes"]
        out[exp] = {"passes": bool(passes), "kept": sorted(kept), "kept_nondas": sorted(kept - das),
                    "kept_das": sorted(kept & das), "das_verbs": sorted(das), "natural_gate": res, "strata": strata,
                    "fp32_I_gate": {"kept_fp32": sorted(kept32), "dropped_vs_round4": sorted(kept - kept32),
                                    "added_vs_round4": sorted(kept32 - kept)}}
        L += [f"## {EXPS[exp]}", "",
              f"Kept I verbs (round 4): {len(kept)} ({len(kept - das)} non-DAS, {len(kept & das)} DAS). An fp32 half-A "
              f"I gate would drop {sorted(kept - kept32) or 'none'} and add {sorted(kept32 - kept) or 'none'}.", "",
              "| Population | contexts | n T | n I | interaction on L | interaction on O | passes |",
              "|---|---|---:|---:|---|---|---|"]
        for k, r in res.items():
            pop, cx = k.split(" | ")
            L.append(f"| {pop} | {cx} | {r['n_T']} | {r['n_I']} | {f(r['interaction_L'])} | {f(r['interaction_O'])} | "
                     f"{'yes' if r['passes'] else 'no'} |")
        L += ["", f"**Natural gate (primary: non-DAS, half B): {'passes' if passes else 'FAILS: no patching'}.**", "",
              "Strata (primary population):", "", "| Stratum | contexts | interaction on L | interaction on O | passes |",
              "|---|---|---|---|---|"]
        for k, r in strata.items():
            mv, cx = k.split(" | ")
            L.append(f"| {mv} | {cx} ({r['n_ctx']}) | {f(r['interaction_L'])} | {f(r['interaction_O'])} | "
                     f"{'yes' if r['passes'] else 'no'} |")
        L.append("")
    # B7 base-form gate (fp32 projections)
    pz = np.load(rdir / "projections.npz")
    keys = [(int(c[1]), int(c.split("_f")[1])) for c in pz["keys"]]
    das_items = pd.read_csv(Path(args.root) / "final_strict" / "items.csv")
    pf = pd.read_csv(Path(args.root) / "final_strict" / "pairs_folds.csv")
    Pz = P.assign(kind=np.where(P.kind.isin(["emb", "mat", "base_ref"]), "real", P.kind))
    Z, _ = z_table(Pz, pz["proj"], list(pz["sites"]), keys, das_items, pf)
    cl = dict(zip(items.lemma, items.cls))
    bg = []
    for si, s in enumerate(pz["sites"]):
        br = P[P.kind == "base_ref"].assign(z=lambda d: Z[d.pid.to_numpy(), si])
        vm = br.groupby("lemma").z.mean()
        a = auc(vm[[cl[l] == "T" for l in vm.index]].to_numpy(), vm[[cl[l] == "I" for l in vm.index]].to_numpy())
        bg.append({"site": int(s), "auc_verbs": a, "pass": bool(a >= 0.8)})
    pd.DataFrame(bg).to_csv(rdir / "baseform_gate.csv", index=False)
    L += ["B7 base-form gate (\"NAME can V\", AUC over verb means along d_s): "
          + ", ".join(f"site {r['site']} {r['auc_verbs']:.2f}{'' if r['pass'] else ' (fails)'}" for r in bg), ""]
    # gates.json in round 4's format (read by run_fillergap.py patch) + the full record
    json.dump({exp: {"passes": o["passes"], "kept": o["kept"]} for exp, o in out.items()}, open(rdir / "gates.json", "w"),
              indent=2)
    json.dump(out, open(rdir / "gates_r5.json", "w"), indent=1, default=float)
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("gates", "patch"))
    ap.add_argument("--root", default="results/das_round2")
    ap.add_argument("--data-dir", default="data/round4/fillergap")
    ap.add_argument("--out-dir", default="results/round5/fillergap")
    ap.add_argument("--report", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    if a.report is None:
        a.report = "reports/round5/c8_fillergap_gates.md" if a.mode == "gates" else "reports/round5/c8_fillergap.md"
    if a.mode == "gates":
        run_gates(a)
    else:
        raise SystemExit("patch mode: see analyze_fillergap_r5.py run_patch (added before the patch run)")
