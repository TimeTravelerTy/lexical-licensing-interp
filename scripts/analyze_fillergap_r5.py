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

from analyze_fillergap import DECL, RD, SITES, interaction, multi, readouts, reading, stage2_stats
from analyze_passive_test import site_delta
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
        "non-DAS T, all non-DAS I (sensitivity)": (T - das, set(e[e.cls == "I"].lemma) - das),
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


def run_patch(args):
    ddir, rdir = Path(args.data_dir), Path(args.out_dir)
    items = pd.read_csv(ddir / "items.csv")
    nat = pd.read_parquet(rdir / "natural.parquet").set_index("pid")
    x = readouts(items, nat).set_index("item_id")
    bg = pd.read_csv(rdir / "baseform_gate.csv").set_index("site")["pass"]
    vt = pd.read_csv(rdir / "verb_tokens.csv")
    gates = json.loads((rdir / "gates_r5.json").read_text())
    rows = []
    for exp in ("emb", "mat"):
        if not gates[exp]["passes"]:
            print(f"{exp}: natural gate failed; no patch reading")
            continue
        plan = pd.read_csv(ddir / f"plan_{exp}.csv.gz")
        kept = set(gates[exp]["kept"])
        das = set(gates[exp]["das_verbs"])
        e = x[x.exp == exp].reset_index()
        T_all = e[e.cls == "T"]
        T_nd = T_all[~T_all.lemma.isin(das)]
        base = x.loc[plan.item_id]
        df = pd.DataFrame({"pair_id": base.lemma.to_numpy(), "context_id": base.context_id.to_numpy(),
                           "frame": base.frame.to_numpy(), "cond": plan.cond.to_numpy(), "half": base.half.to_numpy(),
                           "matrix": base.matrix.to_numpy(), "dq": plan.donor_pair.fillna("").to_numpy()})
        tok = vt[vt.exp == exp].set_index("lemma").n_tokens
        ctxB = sorted(e[e.half == 1].context_id.unique())
        ctxAll = sorted(e.context_id.unique())
        nd_I, das_I = sorted(kept - das), sorted(kept & das)
        for site in SITES:
            delta = site_delta(Path(args.root), site)
            dd = {}
            for pool in ("nondas", "all"):
                pat = pd.read_parquet(rdir / f"patches_{exp}_site{site}_{pool}.parquet")
                assert (pat.row.to_numpy() == plan.row.to_numpy()).all()
                v = pat.assign(num=pat["dot"] if exp == "emb" else pat["q"])
                v["L"] = v.num - v.PREP
                d = df.copy()
                d[list(RD)] = v[list(RD)].to_numpy() - base[list(RD)].to_numpy()
                d["Oabs"] = v.O.to_numpy()
                dd[pool] = d
            # T - I donor rows are identical in both pool files (same rows, donors, bases)
            tcols = list(RD)
            don = dd["nondas"].cond.isin(["T", "I"]).to_numpy()
            assert np.allclose(dd["nondas"].loc[don, tcols].to_numpy(), dd["all"].loc[don, tcols].to_numpy(), atol=1e-5)
            pops = {"primary: non-DAS, half B": ("nondas", nd_I, ctxB, T_nd),
                    "primary: non-DAS, all contexts": ("nondas", nd_I, ctxAll, T_nd),
                    "DAS bases, pooled targets": ("all", das_I, ctxB, T_all),
                    "DAS bases, non-DAS targets": ("nondas", das_I, ctxB, T_nd),
                    "round-4 population (fp32)": ("all", sorted(kept), ctxB, T_all),
                    "multi-token, non-DAS": ("nondas", [v_ for v_ in nd_I if tok[v_] > 1], ctxB, T_nd)}
            for mv in sorted(e.matrix.unique()):
                pops[f"stratum {mv}, half B"] = ("nondas", nd_I, sorted(e[(e.half == 1) & (e.matrix == mv)].context_id.unique()), T_nd)
                pops[f"stratum {mv}, all 6"] = ("nondas", nd_I, sorted(e[e.matrix == mv].context_id.unique()), T_nd)
            for pop, (pool, verbs, ctxs, Ti) in pops.items():
                if site not in DECL and not pop.startswith("primary") and not pop.startswith("stratum"):
                    continue
                ee = e[e.context_id.isin(ctxs)]
                natgap = {fr: ee[ee.lemma.isin(set(Ti.lemma)) & (ee.frame == fr)][list(RD)].mean()
                          - ee[(ee.cls == "I") & (ee.frame == fr) & ee.lemma.isin(verbs)][list(RD)].mean()
                          for fr in ("nofill", "fill")}
                for r in stage2_stats(dd[pool], None, Ti, verbs, ctxs, delta, natgap, args.n_boot, args.seed):
                    rows.append({"exp": exp, "site": site, "population": pop, "n_verbs": len(verbs), "n_ctx": len(ctxs), **r})
            print(f"{exp} site {site} done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(rdir / "c8_patch_summary.csv", index=False)
    write_patch(args, res, bg)


def gated(exp, site, rd, bg):
    return "inconclusive (base-form gate)" if exp == "mat" and not bool(bg.get(site, True)) else rd


def write_patch(args, res, bg):
    f = lambda r: f"{r.est:+.2f} [{r.lo95:+.2f}, {r.hi95:+.2f}]"
    L = ["# Round 5, C8: filler-gap patches with non-DAS verbs as the primary set (fp32)", "",
         "Spec: `plan.md`, C8. Runs: `run_fillergap.py patch --dtype float32 --target-pool {nondas, all}`; analysis: "
         "`analyze_fillergap_r5.py patch` (round-4 statistics and reading rule). D = after T − after I active donors; "
         "D_in = coordinate at t_T − at t_I. nofill = \"that\" / no wh; fill = \"what\". Primary: kept non-DAS I bases, "
         "non-DAS targets, natural gaps and T reference from non-DAS verbs, half-B contexts.", ""]
    dec = {}
    for exp in res.exp.unique():
        L += [f"## {EXPS[exp]}", ""]
        for pq, oq, nm in (("D", "T_only", "T − I donors"), ("D_in", "inT_only", "in range")):
            L += [f"### Primary, {nm} (half B)", "",
                  "| Site | nofill: . or ? | nofill: O | fill: L | fill: . or ? | fill: PREP | fill: O | fill: O level vs T | fill − nofill: O | Reading |",
                  "|---:|---|---|---|---|---|---|---|---|---|"]
            for s_ in SITES:
                sub = res[(res.exp == exp) & (res.site == s_) & (res.population == "primary: non-DAS, half B")]
                g = lambda fr, q, r: sub[(sub.frame == fr) & (sub.quantity == q) & (sub.readout == r)].iloc[0]
                rd, na, ab = reading(g, pq, oq)
                if exp == "mat" and not bool(bg.get(s_, True)):
                    rd = "inconclusive (base-form gate)"
                c = [g("nofill", pq, "num"), g("nofill", pq, "O"), g("fill", pq, "L"), g("fill", pq, "num"),
                     g("fill", pq, "PREP"), g("fill", pq, "O")]
                lev, it_ = g("fill", oq, "Olevel"), g("fill - nofill", pq, "O")
                L.append(f"| {s_} | " + " | ".join(f"{f(r)} {r['class']}" for r in c) +
                         f" | {f(lev)} {'not above' if na else 'above' if ab else 'unresolved'} | {f(it_)} | **{rd}** |")
                dec[f"{exp}|{nm}|{s_}"] = rd
            L.append("")
        L += ["### Readings by population (sites 6 and 8; strata at every site below)", "",
              "| Population | verbs | contexts | site | T − I donors | in range |", "|---|---:|---:|---:|---|---|"]
        for pop in res[res.exp == exp].population.unique():
            if pop.startswith("stratum"):
                continue
            for s_ in DECL:
                sub = res[(res.exp == exp) & (res.site == s_) & (res.population == pop)]
                g = lambda fr, q, r: sub[(sub.frame == fr) & (sub.quantity == q) & (sub.readout == r)].iloc[0]
                L.append(f"| {pop} | {int(sub.n_verbs.iloc[0])} | {int(sub.n_ctx.iloc[0])} | {s_} | "
                         f"{gated(exp, s_, reading(g, 'D', 'T_only')[0], bg)} | {gated(exp, s_, reading(g, 'D_in', 'inT_only')[0], bg)} |")
        L += ["", "### Frame strata (primary population; in range / T − I donors per site)", "",
              "| Stratum | contexts | " + " | ".join(f"site {s_}" for s_ in SITES) + " |", "|---|---|" + "---|" * len(SITES)]
        for pop in sorted(p for p in res[res.exp == exp].population.unique() if p.startswith("stratum")):
            cells = []
            for s_ in SITES:
                sub = res[(res.exp == exp) & (res.site == s_) & (res.population == pop)]
                g = lambda fr, q, r: sub[(sub.frame == fr) & (sub.quantity == q) & (sub.readout == r)].iloc[0]
                short = {"licensing-like crossover": "LIC", "surface (object next)": "surf", "mixed / unresolved": "mix",
                         "inconclusive (base-form gate)": "n/a"}
                ri, rd_ = gated(exp, s_, reading(g, "D_in", "inT_only")[0], bg), gated(exp, s_, reading(g, "D", "T_only")[0], bg)
                cells.append(f"{short[ri]} / {short[rd_]}")
                dec[f"{exp}|{pop}|{s_}"] = ri
            name, cx = pop.replace("stratum ", "").split(", ")
            L.append(f"| {name} | {cx} | " + " | ".join(cells) + " |")
        L += ["", "LIC = licensing-like crossover, surf = surface (object next), mix = mixed / unresolved.", ""]
    (Path(args.out_dir) / "c8_decisions.json").write_text(json.dumps(dec, indent=1))
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
    run_gates(a) if a.mode == "gates" else run_patch(a)
