#!/usr/bin/env python3
"""C4 phase 1b readings (reports/v2_controls_mech_b_rules.md, part B).

B1 (--components): component sweep of the v2 edit: m = mean(TT - T<-I:C) / mean(TT - II), r = mean(I<-T:C - II) / D;
   direct path 1 - m(all).
B2 (--natural): natural mediation: G = good - bad; m_nat(S) = mean(good - g<-b:S) / G, r_nat(S) = mean(b<-g:S - bad) / G;
   families rand / mag / de as phase 1; declared NM1 / NM2 (lo95 m_nat(P) > 0 and lo95(m_nat(P) - mean mag) > 0).
B3b (--core-edit, --natural): core sets (core_*, core_top3 / 5) against the per-layer controls ctl<layer>_*.
B3c (--fg, --natural): per core neuron, natural T - I of the post-activation in passives (half B, pairs) and in R4-C8's
   embedded fill / nofill frames (stored, lemmas), rho = fill / passive; declared D1 (gap reader: lo95 fill > 0 and
   lo95 fill - nofill > 0), D2 (lo95 rho(gap reader) - rho(detector) > 0 for each passive detector).
All shares are pair-averaged then ratios of averages; pair bootstrap 2,000 draws, seed 17 (C8 lemmas resampled within
class on the same draw index).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

GAP = "MLP11.n3462"


def per_pair(v, pc, n):
    s, c = np.zeros(n), np.zeros(n)
    np.add.at(s, pc, v)
    np.add.at(c, pc, 1)
    return s / c


class Boot:
    def __init__(self, n_pairs, n_boot, seed):
        rng = np.random.default_rng(seed)
        self.draws = [np.arange(n_pairs)] + [rng.integers(0, n_pairs, n_pairs) for _ in range(n_boot)]

    def stat(self, fn):
        v = np.array([fn(ix) for ix in self.draws])
        return {"est": float(v[0]), "lo95": float(np.percentile(v[1:], 2.5)), "hi95": float(np.percentile(v[1:], 97.5))}


def load(d, stem):
    z = np.load(Path(d) / f"{stem}.npz", allow_pickle=True)
    k = pd.read_parquet(Path(d) / f"{stem}_keys.parquet")
    Y = pd.DataFrame(z["by"].astype(float), columns=[str(c) for c in z["conds"]])
    return k, Y, json.loads((Path(d) / f"{stem}_meta.json").read_text())


def shares(Y, keys, base, ref, arms, sets, bt=None, subset=None):
    """base / ref: column names whose difference is the denominator; arms: (B column template, R column template)."""
    if subset is not None:
        m = keys.pair.isin(subset).to_numpy()
        keys, Y = keys[m].reset_index(drop=True), Y[m].reset_index(drop=True)
    pairs = np.array(sorted(keys.pair.unique()))
    pc = keys.pair.map({p: i for i, p in enumerate(pairs)}).to_numpy()
    n = len(pairs)
    bt = bt or Boot(n, 2000, 17)
    Dn = per_pair((Y[base] - Y[ref]).to_numpy(), pc, n)
    q = {}
    for s in sets:
        q[("B", s)] = per_pair((Y[base] - Y[arms[0].format(s)]).to_numpy(), pc, n)
        q[("R", s)] = per_pair((Y[arms[1].format(s)] - Y[ref]).to_numpy(), pc, n)
    return {"pairs": n, "D": Dn, "q": q, "bt": bt}


def summarize(S, named, fams, ref_set="P"):
    bt, Dn, q = S["bt"], S["D"], S["q"]
    out = {"pairs": S["pairs"], "denominator": bt.stat(lambda ix: Dn[ix].mean())}
    for kind in ("B", "R"):
        o = {}
        for s in named:
            o[s] = {"nats": bt.stat(lambda ix, s=s: q[(kind, s)][ix].mean()),
                    "share": bt.stat(lambda ix, s=s: q[(kind, s)][ix].mean() / Dn[ix].mean())}
        for f, ss in fams.items():
            if not ss or (kind, ref_set) not in q:
                continue
            cm = lambda ix, ss=ss: np.mean([q[(kind, s)][ix].mean() for s in ss])
            o[f"{f}_mean"] = {"share": bt.stat(lambda ix, cm=cm: cm(ix) / Dn[ix].mean())}
            o[f"{ref_set}_minus_{f}"] = {"share": bt.stat(lambda ix, cm=cm: (q[(kind, ref_set)][ix].mean() - cm(ix)) / Dn[ix].mean())}
            ctrl = np.array([q[(kind, s)].mean() for s in ss])
            o[f"{ref_set}_rank_among_{f}"] = int((ctrl >= q[(kind, ref_set)].mean()).sum()) + 1
        out[kind] = o
    return out


def fams_of(names, prefixes):
    return {f: [s for s in names if s.startswith(f + "_") and s[len(f) + 1:].isdigit()] for f in prefixes}


def b3c(acts, nkeys, fg_path, meta_path, core, labels, n_boot, seed):
    """Natural T - I per core neuron: passives (half-B pairs) and C8 embedded / matrix frames (lemmas)."""
    A = acts.merge(nkeys[["rid", "pair"]], on="rid")
    pairs = np.array(sorted(A.pair.unique()))
    pc = A.pair.map({p: i for i, p in enumerate(pairs)}).to_numpy()
    pas = {n: per_pair((A[f"good_{n}"] - A[f"bad_{n}"]).to_numpy(), pc, len(pairs)) for n in core}
    fg = pd.read_parquet(fg_path)
    cols = json.loads(Path(meta_path).read_text())["neuron_columns"]
    col = {n: f"a{cols.index(n)}" for n in core}
    frames = {}
    for (exp, fr), g in fg.groupby(["exp", "frame"]):
        lm = g.groupby(["lemma", "cls"])[[col[n] for n in core]].mean().reset_index()
        frames[f"{exp}_{fr}"] = {c: lm[lm.cls == c].reset_index(drop=True) for c in ("T", "I")}
    rng = np.random.default_rng(seed)
    nT, nI = len(frames["emb_fill"]["T"]), len(frames["emb_fill"]["I"])
    for f in frames:
        assert len(frames[f]["T"]) == nT and len(frames[f]["I"]) == nI
    draws = [(np.arange(len(pairs)), np.arange(nT), np.arange(nI))] + [
        (rng.integers(0, len(pairs), len(pairs)), rng.integers(0, nT, nT), rng.integers(0, nI, nI)) for _ in range(n_boot)]

    def stat(fn):
        v = np.array([fn(*d) for d in draws])
        return {"est": float(v[0]), "lo95": float(np.percentile(v[1:], 2.5)), "hi95": float(np.percentile(v[1:], 97.5))}

    ti = lambda f, n, iT, iI: frames[f]["T"][col[n]].to_numpy()[iT].mean() - frames[f]["I"][col[n]].to_numpy()[iI].mean()
    out = {"pairs": len(pairs), "lemmas_T": nT, "lemmas_I": nI, "neurons": {}}
    rho = lambda n, ip, iT, iI: ti("emb_fill", n, iT, iI) / pas[n][ip].mean()
    for n in core:
        o = {"label": labels.get(n, "unlabelled"), "passive": stat(lambda ip, iT, iI, n=n: pas[n][ip].mean())}
        for f in frames:
            o[f] = stat(lambda ip, iT, iI, f=f, n=n: ti(f, n, iT, iI))
        o["emb_fill_minus_nofill"] = stat(lambda ip, iT, iI, n=n: ti("emb_fill", n, iT, iI) - ti("emb_nofill", n, iT, iI))
        o["rho"] = stat(lambda ip, iT, iI, n=n: rho(n, ip, iT, iI))
        out["neurons"][n] = o
    g = out["neurons"][GAP]
    out["D1_gap_reader_filler_dependent"] = bool(g["emb_fill"]["lo95"] > 0 and g["emb_fill_minus_nofill"]["lo95"] > 0)
    det = [n for n in core if labels.get(n) == "passive-participle detector"]
    out["D2_detectors"] = det
    out["D2_rho_diff"] = {n: stat(lambda ip, iT, iI, n=n: rho(GAP, ip, iT, iI) - rho(n, ip, iT, iI)) for n in det}
    out["D2_dissociation"] = bool(det) and all(v["lo95"] > 0 for v in out["D2_rho_diff"].values())
    return out


def run(a):
    res = {}
    f = lambda c: f"{c['est']:+.3f} [{c['lo95']:+.3f}, {c['hi95']:+.3f}]"
    L = ["# C4 phase 1b: where the edit's effect goes, natural mediation, the core neurons", "",
         "Rules: `reports/v2_controls_mech_b_rules.md` (part B). Half-B pairs; log P(\" by\") nats; pair bootstrap.", ""]
    items = pd.read_csv(a.items, low_memory=False).drop_duplicates("pair_id").set_index("pair_id")
    old = pd.read_csv(a.old_eval)
    old_l = set(old.trans_lemma) | set(old.intrans_lemma)
    new = set(items.index[items.source == "new"])
    fresh = {p for p in new if items.good_lemma[p] not in old_l and items.bad_lemma[p] not in old_l}

    # B1
    k, Y, meta = load(a.components, "components")
    groups = [c.split(":", 1)[1] for c in Y.columns if c.startswith("T<-I:") and c != "T<-I:empty"]
    S = shares(Y, k, "TT", "II", ("T<-I:{}", "I<-T:{}"), groups)
    b1 = summarize(S, groups, {})
    singles = [g for g in groups if re.fullmatch(r"(attn|mlp)\d+", g)]
    b1["direct_path_share"] = S["bt"].stat(lambda ix: 1 - S["q"][("B", "all")][ix].mean() / S["D"][ix].mean())
    b1["sum_single_B_share"] = float(sum(S["q"][("B", g)].mean() for g in singles) / S["D"].mean())
    order = sorted(singles, key=lambda g: -S["q"][("B", g)].mean())
    b1["single_rank_B"] = {g: order.index(g) + 1 for g in ("mlp11", "mlp14")}
    b1["checks"] = meta["checks"]
    b1["check_pass"] = bool(max(meta["checks"]["T<-I:mlp11_14_vs_phase1_ALL_max"], meta["checks"]["I<-T:mlp11_14_vs_phase1_ALL_max"]) <= 1e-4
                            and meta["checks"]["T<-I:empty_vs_TT_max"] <= 1e-5 and meta["checks"]["I<-T:empty_vs_II_max"] <= 1e-5)
    res["B1"] = b1
    L += ["## B1. Where the v2 edit's effect on *by* goes (component clamps at the participle's last token)", "",
          f"Pairs {b1['pairs']}; D = {f(b1['denominator'])}. Checks: {json.dumps(meta['checks'])} → "
          f"{'pass' if b1['check_pass'] else 'INVALID'}.", "",
          "| component | m = B / D | r = R / D |", "|---|---|---|"]
    for g in groups:
        L.append(f"| {g} | {f(b1['B'][g]['share'])} | {f(b1['R'][g]['share'])} |")
    L += ["", f"Direct path (1 − m(all)): {f(b1['direct_path_share'])}; sum of single-component m: "
              f"{b1['sum_single_B_share']:+.3f}; rank among single components (B): {b1['single_rank_B']}.", ""]

    # B2 and B3b (natural)
    k, Y, meta = load(a.natural, "natural")
    names = [c.split(":", 1)[1] for c in Y.columns if c.startswith("g<-b:") and c != "g<-b:empty"]
    fam = fams_of(names, ("rand", "mag", "de"))
    named = [s for s in names if not any(s in v for v in fam.values()) and not s.startswith("ctl")]
    S = shares(Y, k, "good", "bad", ("g<-b:{}", "b<-g:{}"), names)
    b2 = summarize(S, named, fam)
    gate = b2["denominator"]["lo95"] > 0
    b2["gate_G_positive"] = bool(gate)
    for kind, nm in (("B", "NM1"), ("R", "NM2")):
        o = b2[kind]
        b2[nm] = bool(gate and o["P"]["share"]["lo95"] > 0 and o["P_minus_mag"]["share"]["lo95"] > 0)
        b2[nm + "_beyond_de"] = bool(gate and o["P_minus_de"]["share"]["lo95"] > 0)
        if o["ALL"]["share"]["lo95"] > 0:
            o["P_over_ALL"] = S["bt"].stat(lambda ix, kind=kind: S["q"][(kind, "P")][ix].mean() / S["q"][(kind, "ALL")][ix].mean())
    ck = meta["checks"]
    b2["checks"] = ck
    b2["check_pass"] = bool(ck["g<-b:empty_vs_good_max"] <= 1e-5 and ck["b<-g:empty_vs_bad_max"] <= 1e-5
                            and ck["stored_natural_missing"] == 0 and ck["stored_natural_median_abs"] <= 0.05)
    for nm, sub in (("descriptive_new", new), ("descriptive_lemma_fresh", fresh)):
        Ss = shares(Y, k, "good", "bad", ("g<-b:{}", "b<-g:{}"), ["P", "ALL"] + fam["mag"], subset=sub)
        b2[nm] = summarize(Ss, ["P", "ALL"], {"mag": fam["mag"]})
    res["B2"] = b2
    L += ["## B2. Natural mediation (good vs bad natural passives, same contexts)", "",
          f"Pairs {b2['pairs']}; G = {f(b2['denominator'])}. Checks: {json.dumps(ck)} → {'pass' if b2['check_pass'] else 'INVALID'}.", "",
          "| set | m_nat | r_nat |", "|---|---|---|"]
    for s in named + [x for x in b2["B"] if x.endswith("_mean") or "_minus_" in x]:
        L.append(f"| {s} | {f(b2['B'][s]['share'])} | {f(b2['R'][s]['share'])} |")
    L += ["", f"P rank among controls (B): " + ", ".join(f"{x[len('P_rank_among_'):]} {v}" for x, v in b2["B"].items() if x.startswith("P_rank")),
          f"P / ALL: B {f(b2['B']['P_over_ALL']) if 'P_over_ALL' in b2['B'] else 'n/a'}, R {f(b2['R']['P_over_ALL']) if 'P_over_ALL' in b2['R'] else 'n/a'}.",
          f"**Declared:** gate {gate}; NM1 **{b2['NM1']}** (beyond de {b2['NM1_beyond_de']}); NM2 **{b2['NM2']}** (beyond de {b2['NM2_beyond_de']}).", ""]
    for nm in ("descriptive_new", "descriptive_lemma_fresh"):
        x = b2[nm]
        L.append(f"- {nm} ({x['pairs']} pairs): G {f(x['denominator'])}; m_nat(P) {f(x['B']['P']['share'])}; "
                 f"P − mag {f(x['B']['P_minus_mag']['share'])}; r_nat(P) {f(x['R']['P']['share'])}.")
    L.append("")

    # B3b: core sets, edit (phase 1 runner with core sets) and natural
    core = pd.read_csv(a.switch).head(5).neuron.tolist()
    z = np.load(Path(a.core_edit) / "mediation.npz", allow_pickle=True)
    Ye = pd.DataFrame(z["by"].astype(float), columns=[str(c) for c in z["conds"]])
    ke = pd.read_parquet(Path(a.core_edit) / "mediation_keys.parquet")
    cnames = [str(s) for s in z["sets"] if str(s) != "empty"]
    Se = shares(Ye, ke, "TT", "II", ("T<-I:{}", "I<-T:{}"), cnames)
    Sn = S
    b3b = {}
    for n in core + ["core_top3", "core_top5"]:
        s = n if n.startswith("core_top") else f"core_{n}"
        row = {}
        for tag, SS in (("edit", Se), ("natural", Sn)):
            bt, Dn, q = SS["bt"], SS["D"], SS["q"]
            row[tag] = {kind: bt.stat(lambda ix, kind=kind, s=s: q[(kind, s)][ix].mean() / Dn[ix].mean()) for kind in ("B", "R")}
            if not n.startswith("core_top"):
                lay = n.split(".")[0][3:]
                ctl = [c for c in (cnames if tag == "edit" else names) if c.startswith(f"ctl{lay}_")]
                cv = np.array([q[("B", c)].mean() / Dn.mean() for c in ctl])
                row[tag]["ctrl_B_mean"] = float(cv.mean())
                row[tag]["ctrl_B_max"] = float(cv.max())
                row[tag]["rank_B_among_ctrl"] = int((cv >= q[("B", s)].mean() / Dn.mean()).sum()) + 1
        b3b[n] = row
    res["B3b"] = b3b
    L += ["## B3b. The core neurons' causal shares (edit: share of D; natural: share of G)", "",
          "| set | edit m | edit r | natural m_nat | natural r_nat | layer controls m (edit / natural) | rank (edit / natural) |",
          "|---|---|---|---|---|---|---|"]
    for n, row in b3b.items():
        c = (f"{row['edit'].get('ctrl_B_mean', float('nan')):+.3f} (max {row['edit'].get('ctrl_B_max', float('nan')):+.3f}) / "
             f"{row['natural'].get('ctrl_B_mean', float('nan')):+.3f}") if "ctrl_B_mean" in row["edit"] else ""
        rk = f"{row['edit'].get('rank_B_among_ctrl', '')} / {row['natural'].get('rank_B_among_ctrl', '')}" if "ctrl_B_mean" in row["edit"] else ""
        L.append(f"| {n} | {f(row['edit']['B'])} | {f(row['edit']['R'])} | {f(row['natural']['B'])} | {f(row['natural']['R'])} | {c} | {rk} |")
    L.append("")

    # B3c
    labels = {n: v["reading"] for n, v in json.loads(Path(a.b7_decisions).read_text())["neurons"].items()}
    if a.b3a_decisions and Path(a.b3a_decisions).exists():
        labels.update({n: v["reading"] for n, v in json.loads(Path(a.b3a_decisions).read_text())["neurons"].items()})
    acts = pd.read_parquet(Path(a.natural) / "natural_core_acts.parquet")
    b = b3c(acts, k, a.fg, a.fg_meta, core, labels, a.n_boot, a.seed)
    res["B3c"] = b
    L += ["## B3c. What the core neurons respond to (natural T − I of the post-activation)", "",
          f"Passives: {b['pairs']} half-B pairs. C8 frames: {b['lemmas_T']} T / {b['lemmas_I']} I lemmas.", "",
          "| neuron | label | passive | what-gap (emb fill) | that (emb nofill) | fill − nofill | matrix what | matrix did | ρ = fill / passive |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n, o in b["neurons"].items():
        L.append(f"| {n} | {o['label']} | {f(o['passive'])} | {f(o['emb_fill'])} | {f(o['emb_nofill'])} | "
                 f"{f(o['emb_fill_minus_nofill'])} | {f(o['mat_fill'])} | {f(o['mat_nofill'])} | {f(o['rho'])} |")
    L += ["", f"**Declared:** D1 (gap reader depends on the filler) **{b['D1_gap_reader_filler_dependent']}**; "
              f"D2 (ρ(gap reader) − ρ(detector) > 0 for {b['D2_detectors']}) **{b['D2_dissociation']}**: "
              + ", ".join(f"{n} {f(v)}" for n, v in b["D2_rho_diff"].items()) + ".", ""]
    Path(a.out).write_text(json.dumps(res, indent=1, default=float) + "\n")
    Path(a.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--components", required=True)
    ap.add_argument("--natural", required=True)
    ap.add_argument("--core-edit", required=True)
    ap.add_argument("--switch", default="results/pythia/das_v2/mech/conj_v2/conjunction_switch_neurons.csv")
    ap.add_argument("--fg", default="results/round5/shared/sr_natural_fg.parquet")
    ap.add_argument("--fg-meta", default="results/round5/shared/sr_meta.json")
    ap.add_argument("--b7-decisions", default="results/round5/wild/b7_decisions.json")
    ap.add_argument("--b3a-decisions", default="results/pythia/das_v2/mech_b/b3a_decisions.json")
    ap.add_argument("--items", default="data/verb_expansion/eval/passive_test/items.csv")
    ap.add_argument("--old-eval", default="data/verb_expansion/eval_pairs.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
