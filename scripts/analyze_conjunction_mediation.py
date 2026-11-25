#!/usr/bin/env python3
"""Mediation analysis (reports/pythia/mechanism_v2_rules.md, stage 6) of `run_conjunction_mediation.py` outputs.

Per key (base item x donor pair), log P(" by"): D = Y[TT] - Y[II] (the C1 contrast);
B(S) = Y[TT] - Y[T<-I(S)] (removed by clamping S to the I run in the T run); R(S) = Y[I<-T(S)] - Y[II] (restored by
clamping S to the T run in the I run); Bnat(S) = D - (Y[T<-nat(S)] - Y[I<-nat(S)]) (secondary). Keys are averaged
within target pair, then over pairs; shares are ratios of those averages: m(S) = B / D, r(S) = R / D.
Pair bootstrap (half-B pairs, 2,000 draws, seed 17); controls enter as the mean over their 20 sets in each draw.
Declared (per instrument): gate D lo95 > 0; M1 lo95 m(P) > 0 and lo95(m(P) - mean m(mag)) > 0; M2 the same for r;
secondary comparators de (direct-effect-matched) and rand. P / ALL is shown only where lo95 m(ALL) > 0.
Check (v2): the T and I arms and D against the stored per-row v2 expanded passive run (bf16) on the same plan rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def per_pair(v, pair_codes, n_pairs):
    s = np.zeros(n_pairs)
    n = np.zeros(n_pairs)
    np.add.at(s, pair_codes, v)
    np.add.at(n, pair_codes, 1)
    return s / n


def load(d):
    z = np.load(d / "mediation.npz", allow_pickle=True)
    keys = pd.read_parquet(d / "mediation_keys.parquet")
    conds = [str(c) for c in z["conds"]]
    Y = pd.DataFrame(z["by"].astype(float), columns=conds)
    return keys, Y, [str(s) for s in z["sets"]], json.loads((d / "mediation_meta.json").read_text())


def analyze(d, args, subset=None):
    keys, Y, sets, meta = load(d)
    if subset is not None:
        m = keys.pair.isin(subset).to_numpy()
        keys, Y = keys[m].reset_index(drop=True), Y[m].reset_index(drop=True)
    pairs = np.array(sorted(keys.pair.unique()))
    pc = keys.pair.map({p: i for i, p in enumerate(pairs)}).to_numpy()
    nP = len(pairs)
    D = per_pair((Y.TT - Y.II).to_numpy(), pc, nP)
    q = {}
    for s in sets:
        q[("B", s)] = per_pair((Y.TT - Y[f"T<-I:{s}"]).to_numpy(), pc, nP)
        q[("R", s)] = per_pair((Y[f"I<-T:{s}"] - Y.II).to_numpy(), pc, nP)
        q[("Bnat", s)] = per_pair(((Y.TT - Y.II) - (Y[f"T<-nat:{s}"] - Y[f"I<-nat:{s}"])).to_numpy(), pc, nP)
    fam = {f: [s for s in sets if s.startswith(f + "_") and s[len(f) + 1:].isdigit()] for f in ("rand", "mag", "de")}
    rng = np.random.default_rng(args.seed)
    draws = [np.arange(nP)] + [rng.integers(0, nP, nP) for _ in range(args.n_boot)]

    def stat(fn):
        v = np.array([fn(ix) for ix in draws])
        return {"est": float(v[0]), "lo95": float(np.percentile(v[1:], 2.5)), "hi95": float(np.percentile(v[1:], 97.5))}

    out = {"pairs": nP, "keys": len(keys), "D": stat(lambda ix: D[ix].mean())}
    named = [s for s in sets if not any(s in v for v in fam.values()) and s != "empty"]
    for kind in ("B", "R", "Bnat"):
        o = {}
        for s in named:
            o[s] = {"nats": stat(lambda ix, s=s: q[(kind, s)][ix].mean()),
                    "share": stat(lambda ix, s=s: q[(kind, s)][ix].mean() / D[ix].mean())}
        for f, ss in fam.items():
            if not ss:
                continue
            o[f"{f}_mean"] = {"share": stat(lambda ix, ss=ss: np.mean([q[(kind, s)][ix].mean() for s in ss]) / D[ix].mean())}
            o[f"P_minus_{f}"] = {
                "nats": stat(lambda ix, ss=ss: q[(kind, "P")][ix].mean() - np.mean([q[(kind, s)][ix].mean() for s in ss])),
                "share": stat(lambda ix, ss=ss: (q[(kind, "P")][ix].mean()
                                                 - np.mean([q[(kind, s)][ix].mean() for s in ss])) / D[ix].mean())}
            ctrl = np.array([q[(kind, s)].mean() for s in ss])
            o[f"P_rank_among_{f}"] = int((ctrl >= q[(kind, "P")].mean()).sum()) + 1
            o[f"{f}_shares"] = [float(c / D.mean()) for c in ctrl]
        if "ALL" in named and o["ALL"]["share"]["lo95"] > 0:
            o["P_over_ALL"] = stat(lambda ix: q[(kind, "P")][ix].mean() / q[(kind, "ALL")][ix].mean())
        out[kind] = o
    gate = out["D"]["lo95"] > 0
    out["gate_D_positive"] = gate
    for kind, nm in (("B", "M1_mediates"), ("R", "M2_restores")):
        o = out[kind]
        out[nm] = bool(gate and o["P"]["share"]["lo95"] > 0 and o["P_minus_mag"]["share"]["lo95"] > 0) if "P_minus_mag" in o else None
        out[nm + "_beyond_de"] = bool(gate and o["P_minus_de"]["share"]["lo95"] > 0) if "P_minus_de" in o else None
    out["checks"] = meta["checks"]
    return out, keys, Y


def c1_check(keys, Y, stored):
    s = pd.read_parquet(stored, columns=["row", "by"]).set_index("row").by
    t, i = s.reindex(keys.plan_row_T).to_numpy(), s.reindex(keys.plan_row_I).to_numpy()
    dT, dI = np.abs(Y.TT.to_numpy() - t), np.abs(Y.II.to_numpy() - i)
    DT = (Y.TT - Y.II).mean() - np.nanmean(t - i)
    return {"missing": int(np.isnan(t).sum() + np.isnan(i).sum()), "T_median_abs": float(np.nanmedian(dT)),
            "T_max_abs": float(np.nanmax(dT)), "I_median_abs": float(np.nanmedian(dI)), "I_max_abs": float(np.nanmax(dI)),
            "D_pooled_diff": float(DT)}


def run(args):
    res = {}
    fz = pd.read_csv(args.frozen_pairs)
    items = pd.read_csv(args.items, low_memory=False).drop_duplicates("pair_id").set_index("pair_id")
    old = pd.read_csv(args.old_eval)
    old_l = set(old.trans_lemma) | set(old.intrans_lemma)
    new = set(items.index[items.source == "new"])
    fresh = {p for p in new if items.good_lemma[p] not in old_l and items.bad_lemma[p] not in old_l}
    for inst, d in (x.split("=") for x in args.runs.split(",")):
        d = Path(d)
        o, keys, Y = analyze(d, args)
        if inst == "v2" and args.stored:
            o["c1_check"] = c1_check(keys, Y, args.stored)
            c = o["c1_check"]
            o["c1_check_pass"] = bool(c["missing"] == 0 and c["T_median_abs"] <= 0.05 and c["I_median_abs"] <= 0.05
                                      and abs(c["D_pooled_diff"]) <= 0.02)
        o["descriptive_new"] = analyze(d, args, new)[0]
        o["descriptive_lemma_fresh"] = analyze(d, args, fresh)[0]
        res[inst] = o
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
    f = lambda c: f"{c['est']:+.3f} [{c['lo95']:+.3f}, {c['hi95']:+.3f}]"
    L = ["# Mediation of the site-8 edit by MLP 11 / 14 neuron sets (stage 6)", "",
         "Spec: `reports/pythia/mechanism_v2_rules.md`. Half-B pairs; log P(\" by\") nats; pair bootstrap.", ""]
    for inst, o in res.items():
        L += [f"## {inst}", "", f"Pairs {o['pairs']}, keys {o['keys']}. D = {f(o['D'])}. Checks: {json.dumps(o['checks'])}."]
        if "c1_check" in o:
            L += [f"C1 row check (stored bf16 run): {json.dumps(o['c1_check'])} → {'pass' if o['c1_check_pass'] else 'FAIL'}."]
        L += ["", "| set | B nats | m = B / D | R nats | r = R / D | Bnat / D |", "|---|---|---|---|---|---|"]
        for s in [k for k in o["B"] if isinstance(o["B"][k], dict) and "nats" in o["B"][k]]:
            L.append(f"| {s} | {f(o['B'][s]['nats'])} | {f(o['B'][s]['share'])} | {f(o['R'][s]['nats'])} | "
                     f"{f(o['R'][s]['share'])} | {f(o['Bnat'][s]['share'])} |")
        L += ["", f"P rank among controls (B): " + ", ".join(f"{k[len('P_rank_among_'):]} {v}" for k, v in o["B"].items()
                                                     if k.startswith("P_rank")),
              f"P / ALL: B {f(o['B']['P_over_ALL']) if 'P_over_ALL' in o['B'] else 'n/a (ALL unresolved)'}, "
              f"R {f(o['R']['P_over_ALL']) if 'P_over_ALL' in o['R'] else 'n/a'}.",
              f"**Declared:** gate D > 0 {o['gate_D_positive']}; M1 (P mediates, beyond mag) **{o['M1_mediates']}** "
              f"(beyond de: {o['M1_mediates_beyond_de']}); M2 (P restores) **{o['M2_restores']}** "
              f"(beyond de: {o['M2_restores_beyond_de']}).", ""]
        for nm in ("descriptive_new", "descriptive_lemma_fresh"):
            x = o[nm]
            L.append(f"- {nm} ({x['pairs']} pairs): D {f(x['D'])}; m(P) {f(x['B']['P']['share'])}; "
                     f"P − mag {f(x['B']['P_minus_mag']['share']) if 'P_minus_mag' in x['B'] else 'n/a'}; "
                     f"r(P) {f(x['R']['P']['share'])}.")
        L.append("")
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", required=True, help="inst=dir,... (v2 first)")
    ap.add_argument("--stored", default=None, help="stored per-row v2 patches_site8.parquet for the C1 check")
    ap.add_argument("--items", default="data/verb_expansion/eval/passive_test/items.csv")
    ap.add_argument("--frozen-pairs", default="data/das_v2/expanded_eval_pairs.csv")
    ap.add_argument("--old-eval", default="data/verb_expansion/eval_pairs.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
