#!/usr/bin/env python3
"""C4 phase 1b readings (reports/v2_controls_mech_b_rules.md, part B). Fails closed: if any declared check fails, the
report says INVALID for that part and no decision is written for it.

B1 (--components): m = mean(TT - T<-I:C) / D, r = mean(I<-T:C - II) / D, D = TT - II; residual-only 1 - m(all), 1 - r(all).
B2 (--natural): G = good - bad; m_nat(S) = mean(good - g<-b:S) / G, r_nat(S) = mean(b<-g:S - bad) / G;
   B_same(S) = mean(good - g<-g2:S); coordinate swap C_g = good - gswap, C_b = bswap - bad and its mediation
   BC_g(S) = gswap|S - gswap, BC_b(S) = bswap - bswap|S. Declared NM1 / NM2 (vs magnat), NM-class, CS1, CS2 (vs mag).
B3b (--core-edit, --natural): core sets against the per-layer controls ctl<layer>_*.
B3c (--fg): per core neuron, natural T - I in passives (half B) and R4-C8 frames on a complete (class, lemma) grid;
   declared D1 for the gap reader; rho = fill / passive descriptive, gated on lo95(passive) > 0; C8-gated sensitivity.
Shares are pair-averaged then ratios of averages; pair bootstrap (--n-boot, --seed) on one draw set per part.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

GAP = "MLP11.n3462"
EXPECTED_LABELS = {"MLP11.n3462": "gap-sensitive participle reader", "MLP14.n2114": "passive-participle detector",
                   "MLP14.n5407": "passive-participle detector", "MLP14.n4241": "passive-participle detector",
                   "MLP11.n4925": "other"}


def per_pair(v, pc, n):
    s, c = np.zeros(n), np.zeros(n)
    np.add.at(s, pc, v)
    np.add.at(c, pc, 1)
    return s / c


class Boot:
    def __init__(self, n, n_boot, seed):
        rng = np.random.default_rng(seed)
        self.draws = [np.arange(n)] + [rng.integers(0, n, n) for _ in range(n_boot)]

    def stat(self, fn):
        v = np.array([fn(ix) for ix in self.draws])
        return {"est": float(v[0]), "lo95": float(np.percentile(v[1:], 2.5)), "hi95": float(np.percentile(v[1:], 97.5))}


def load(d, stem):
    z = np.load(Path(d) / f"{stem}.npz", allow_pickle=True)
    k = pd.read_parquet(Path(d) / f"{stem}_keys.parquet")
    Y = pd.DataFrame(z["by"].astype(float), columns=[str(c) for c in z["conds"]])
    assert len(k) == len(Y) and (k.rid.to_numpy() == np.arange(len(k))).all(), f"{d}: rows and keys misaligned"
    assert np.isfinite(Y.to_numpy()).all(), f"{d}: non-finite cells"
    return k, Y, json.loads((Path(d) / f"{stem}_meta.json").read_text())


class Pairs:
    """Per-pair averages of row-level contrasts, with one bootstrap over pairs. weight='item' averages rows within
    item first (edit keys: donors within item), then items within pair."""

    def __init__(self, keys, n_boot, seed, weight="row"):
        self.keys = keys
        self.pairs = np.array(sorted(keys.pair.unique()))
        self.pc = keys.pair.map({p: i for i, p in enumerate(self.pairs)}).to_numpy()
        self.n = len(self.pairs)
        self.bt = Boot(self.n, n_boot, seed)
        self.weight = weight
        if weight == "item":
            self.ic, it = pd.factorize(keys.item_id)
            self.item_pair = keys.groupby(self.ic).pair.first().map({p: i for i, p in enumerate(self.pairs)}).to_numpy()

    def __call__(self, v):
        v = np.asarray(v, float)
        if self.weight == "item":
            return per_pair(per_pair(v, self.ic, self.ic.max() + 1), self.item_pair, self.n)
        return per_pair(v, self.pc, self.n)

    def share(self, num, den):
        return self.bt.stat(lambda ix: num[ix].mean() / den[ix].mean())

    def nats(self, x):
        return self.bt.stat(lambda ix: x[ix].mean())


def fams_of(names, prefixes):
    return {f: [s for s in names if s.startswith(f + "_") and s[len(f) + 1:].isdigit()] for f in prefixes}


def control_block(pp, q, ref, fams, den):
    """P minus each family's mean (share of den), family mean share, and P's rank."""
    o = {}
    for f, ss in fams.items():
        if not ss:
            continue
        cm = lambda ix, ss=ss: np.mean([q[s][ix].mean() for s in ss])
        o[f"{f}_mean"] = pp.bt.stat(lambda ix, cm=cm: cm(ix) / den[ix].mean())
        o[f"P_minus_{f}"] = pp.bt.stat(lambda ix, cm=cm: (q[ref][ix].mean() - cm(ix)) / den[ix].mean())
        ctrl = np.array([q[s].mean() for s in ss])
        o[f"P_rank_among_{f}"] = int((ctrl >= q[ref].mean()).sum()) + 1
        o[f"{f}_max"] = float(ctrl.max() / den.mean())
    return o


def b3c(acts, nkeys, fg_path, meta_path, fg_data, core, labels, n_boot, seed):
    A = acts.merge(nkeys[["rid", "pair"]], on="rid", how="left", validate="one_to_one")
    assert len(A) == len(nkeys) and A.pair.notna().all(), "core activations do not cover the natural keys"
    pairs = np.array(sorted(A.pair.unique()))
    pc = A.pair.map({p: i for i, p in enumerate(pairs)}).to_numpy()
    pas = {n: per_pair((A[f"good_{n}"] - A[f"bad_{n}"]).to_numpy(), pc, len(pairs)) for n in core}
    fg = pd.read_parquet(fg_path)
    cols = json.loads(Path(meta_path).read_text())["neuron_columns"]
    core = [n for n in core if n in cols]  # always all five outside smoke (asserted by the caller)
    col = {n: f"a{cols.index(n)}" for n in core}
    out = {"pairs": len(pairs), "populations": {}}
    gated = {}
    for exp in ("emb", "mat"):
        e = fg[fg.exp == exp]
        das = set(e[e.das_pair.fillna("") != ""].lemma)
        kept = set(pd.read_csv(Path(fg_data) / f"kept_I_{exp}.csv").lemma)
        gated[exp] = (set(e[e.cls == "T"].lemma) - das) | (kept - das)
    for pop in ("all lemmas", "C8 gated"):
        grids = {}
        for (exp, fr), g in fg.groupby(["exp", "frame"]):
            if pop == "C8 gated":
                g = g[g.lemma.isin(gated[exp])]
            lm = g.groupby(["cls", "lemma"])[[col[n] for n in core]].mean()
            grids[f"{exp}_{fr}"] = lm
        idx = None
        for f, lm in grids.items():  # complete (class, lemma) grid across frames within each experiment
            ex = f.split("_")[0]
            ref = grids[f"{ex}_fill"].index
            assert lm.index.equals(ref), f"{pop}: {f} lemma grid differs from {ex}_fill"
        T = {f: lm.loc["T"] for f, lm in grids.items()}
        I = {f: lm.loc["I"] for f, lm in grids.items()}
        nT = {ex: len(T[f"{ex}_fill"]) for ex in ("emb", "mat")}
        nI = {ex: len(I[f"{ex}_fill"]) for ex in ("emb", "mat")}
        rng = np.random.default_rng(seed)
        draws = [(np.arange(len(pairs)), {ex: np.arange(nT[ex]) for ex in nT}, {ex: np.arange(nI[ex]) for ex in nI})] + [
            (rng.integers(0, len(pairs), len(pairs)), {ex: rng.integers(0, nT[ex], nT[ex]) for ex in nT},
             {ex: rng.integers(0, nI[ex], nI[ex]) for ex in nI}) for _ in range(n_boot)]

        def stat(fn):
            v = np.array([fn(*d) for d in draws])
            return {"est": float(v[0]), "lo95": float(np.percentile(v[1:], 2.5)), "hi95": float(np.percentile(v[1:], 97.5))}

        ti = lambda f, n, iT, iI: (T[f][col[n]].to_numpy()[iT[f.split("_")[0]]].mean()
                                   - I[f][col[n]].to_numpy()[iI[f.split("_")[0]]].mean())
        o = {"lemmas_T": nT, "lemmas_I": nI, "neurons": {}}
        for n in core:
            r = {"label": labels[n], "passive": stat(lambda ip, iT, iI, n=n: pas[n][ip].mean())}
            for f in grids:
                r[f] = stat(lambda ip, iT, iI, f=f, n=n: ti(f, n, iT, iI))
            r["emb_fill_minus_nofill"] = stat(lambda ip, iT, iI, n=n: ti("emb_fill", n, iT, iI) - ti("emb_nofill", n, iT, iI))
            if r["passive"]["lo95"] > 0:
                r["rho"] = stat(lambda ip, iT, iI, n=n: ti("emb_fill", n, iT, iI) / pas[n][ip].mean())
            o["neurons"][n] = r
        g = o["neurons"].get(GAP)
        o["D1_gap_reader_filler_dependent"] = (bool(g["emb_fill"]["lo95"] > 0 and g["emb_fill_minus_nofill"]["lo95"] > 0)
                                               if g is not None else None)
        out["populations"][pop] = o
    out["D1"] = out["populations"]["all lemmas"]["D1_gap_reader_filler_dependent"]
    return out


def run(a):
    res, inval = {}, {}
    f = lambda c: f"{c['est']:+.3f} [{c['lo95']:+.3f}, {c['hi95']:+.3f}]"
    L = ["# C4 phase 1b: where the edit's effect goes, natural mediation, the core neurons", "",
         "Rules: `reports/v2_controls_mech_b_rules.md` (part B). Half-B pairs; log P(\" by\") nats; pair bootstrap.", ""]
    items = pd.read_csv(a.items, low_memory=False).drop_duplicates("pair_id").set_index("pair_id")
    old = pd.read_csv(a.old_eval)
    old_l = set(old.trans_lemma) | set(old.intrans_lemma)
    new = set(items.index[items.source == "new"])
    fresh = {p for p in new if items.good_lemma[p] not in old_l and items.bad_lemma[p] not in old_l}
    core = pd.read_csv(a.switch).head(5).neuron.tolist()

    # ---------------- B1
    k, Y, meta = load(a.components, "components")
    ck = meta["checks"]
    ok = (max(ck["T<-I:mlp11_14_vs_phase1_ALL_max"], ck["I<-T:mlp11_14_vs_phase1_ALL_max"]) <= 1e-4
          and ck["TT_vs_phase1_max"] <= 1e-5 and ck["II_vs_phase1_max"] <= 1e-5
          and ck["T<-I:empty_vs_TT_max"] <= 1e-5 and ck["I<-T:empty_vs_II_max"] <= 1e-5
          and ck["resid_all_rel_max"] <= 1e-4 and ck["nan_cells"] == 0 and ck["hook_in_max"] <= 1e-5)
    L += ["## B1. Where the v2 edit's effect on *by* goes (component outputs at the participle's last token)", "",
          f"Checks: {json.dumps(ck)} → {'pass' if ok else 'INVALID'}.", ""]
    if ok:
        groups = [c.split(":", 1)[1] for c in Y.columns if c.startswith("T<-I:") and c != "T<-I:empty"]
        pp = Pairs(k, a.n_boot, a.seed)
        Dn = pp(Y.TT - Y.II)
        B = {g: pp(Y.TT - Y[f"T<-I:{g}"]) for g in groups}
        R = {g: pp(Y[f"I<-T:{g}"] - Y.II) for g in groups}
        b1 = {"pairs": pp.n, "D": pp.nats(Dn), "m": {g: pp.share(B[g], Dn) for g in groups},
              "r": {g: pp.share(R[g], Dn) for g in groups}}
        b1["residual_only_I_ref"] = pp.bt.stat(lambda ix: 1 - B["all"][ix].mean() / Dn[ix].mean())
        b1["residual_only_T_ref"] = pp.bt.stat(lambda ix: 1 - R["all"][ix].mean() / Dn[ix].mean())
        singles = [g for g in groups if re.fullmatch(r"(attn|mlp)\d+", g)]
        b1["sum_single_m"] = float(sum(B[g].mean() for g in singles) / Dn.mean())
        b1["sum_single_r"] = float(sum(R[g].mean() for g in singles) / Dn.mean())
        order = sorted(singles, key=lambda g: -B[g].mean())
        b1["single_rank_m"] = {g: order.index(g) + 1 for g in ("mlp11", "mlp14")}
        res["B1"] = b1
        L += [f"Pairs {pp.n}; D = {f(b1['D'])}.", "", "| component | m (I-ref) | r (T-ref) |", "|---|---|---|"]
        L += [f"| {g} | {f(b1['m'][g])} | {f(b1['r'][g])} |" for g in groups]
        L += ["", f"Residual-only: 1 − m(all) {f(b1['residual_only_I_ref'])}; 1 − r(all) {f(b1['residual_only_T_ref'])}. "
                  f"Sum of single-component m {b1['sum_single_m']:+.3f}, r {b1['sum_single_r']:+.3f} (clamps interact). "
                  f"Rank among single components (m): {b1['single_rank_m']}.", ""]
    else:
        inval["B1"] = ck

    # ---------------- B2 / B3b natural
    k, Y, meta = load(a.natural, "natural")
    ck = meta["checks"]
    ok = (ck["g<-b:empty_vs_good_max"] <= 1e-5 and ck["b<-g:empty_vs_bad_max"] <= 1e-5 and ck["g<-g2:empty_vs_good_max"] <= 1e-5
          and ck["gswap|empty_vs_gswap_max"] <= 1e-5 and ck["bswap|empty_vs_bswap_max"] <= 1e-5
          and ck["stored_natural_missing"] == 0 and ck["stored_natural_median_abs"] <= 0.05
          and abs(ck["stored_natural_pair_gap_mean_diff"]) <= 0.02 and ck["hook_out_max"] == 0 and ck["hook_in_max"] <= 1e-5
          and ck["nan_cells"] == 0 and ck["core"] == core)
    L += ["## B2. Natural mediation (good vs bad natural passives, same contexts)", "",
          f"Checks: {json.dumps(ck)} → {'pass' if ok else 'INVALID'}.", ""]
    names = [c.split(":", 1)[1] for c in Y.columns if c.startswith("g<-b:") and c != "g<-b:empty"]
    pp = Pairs(k, a.n_boot, a.seed)
    if ok:
        G = pp(Y.good - Y.bad)
        Bn = {s: pp(Y.good - Y[f"g<-b:{s}"]) for s in names}
        Rn = {s: pp(Y[f"b<-g:{s}"] - Y.bad) for s in names}
        Bs = {s: pp(Y.good - Y[f"g<-g2:{s}"]) for s in names}
        fam = fams_of(names, ("magnat", "denat", "mag", "de", "rand"))
        named = [s for s in names if not any(s in v for v in fam.values()) and not s.startswith("ctl")]
        b2 = {"pairs": pp.n, "G": pp.nats(G),
              "m_nat": {s: {"nats": pp.nats(Bn[s]), "share": pp.share(Bn[s], G)} for s in named},
              "r_nat": {s: {"nats": pp.nats(Rn[s]), "share": pp.share(Rn[s], G)} for s in named},
              "B_same": {s: {"nats": pp.nats(Bs[s]), "share": pp.share(Bs[s], G)} for s in named}}
        b2["m_nat_controls"] = control_block(pp, Bn, "P", fam, G)
        b2["r_nat_controls"] = control_block(pp, Rn, "P", fam, G)
        b2["B_same_controls"] = control_block(pp, Bs, "P", {"magnat": fam["magnat"]}, G)
        b2["nat_minus_same_P"] = pp.bt.stat(lambda ix: (Bn["P"][ix].mean() - Bs["P"][ix].mean()) / G[ix].mean())
        if b2["m_nat"]["ALL"]["share"]["lo95"] > 0:
            b2["P_over_ALL"] = pp.bt.stat(lambda ix: Bn["P"][ix].mean() / Bn["ALL"][ix].mean())
        # coordinate swap and its mediation
        Cg, Cb = pp(Y.good - Y.gswap), pp(Y.bswap - Y.bad)
        ssets = [c.split("|", 1)[1] for c in Y.columns if c.startswith("gswap|") and c != "gswap|empty"]
        BCg = {s: pp(Y[f"gswap|{s}"] - Y.gswap) for s in ssets}
        BCb = {s: pp(Y.bswap - Y[f"bswap|{s}"]) for s in ssets}
        sfam = fams_of(ssets, ("mag",))
        b2["swap"] = {"C_g": {"nats": pp.nats(Cg), "share_G": pp.share(Cg, G)},
                      "C_b": {"nats": pp.nats(Cb), "share_G": pp.share(Cb, G)},
                      "BC_g": {s: pp.share(BCg[s], Cg) for s in ssets if s in ("P", "ALL")},
                      "BC_b": {s: pp.share(BCb[s], Cb) for s in ssets if s in ("P", "ALL")},
                      "BC_g_controls": control_block(pp, BCg, "P", sfam, Cg),
                      "BC_b_controls": control_block(pp, BCb, "P", sfam, Cb)}
        gate = b2["G"]["lo95"] > 0
        mc, rc = b2["m_nat_controls"], b2["r_nat_controls"]
        sw = b2["swap"]
        dec = {"gate_G_positive": bool(gate),
               "NM1": bool(gate and b2["m_nat"]["P"]["share"]["lo95"] > 0 and mc["P_minus_magnat"]["lo95"] > 0),
               "NM2": bool(gate and b2["r_nat"]["P"]["share"]["lo95"] > 0 and rc["P_minus_magnat"]["lo95"] > 0),
               "NM1_beyond_denat": bool(gate and mc["P_minus_denat"]["lo95"] > 0),
               "NM1_beyond_edit_mag": bool(gate and mc["P_minus_mag"]["lo95"] > 0),
               "NM_class": bool(gate and b2["nat_minus_same_P"]["lo95"] > 0),
               "CS1": bool(gate and sw["C_g"]["share_G"]["lo95"] > 0),
               "CS2": bool(gate and sw["C_g"]["share_G"]["lo95"] > 0 and sw["BC_g"]["P"]["lo95"] > 0
                           and sw["BC_g_controls"]["P_minus_mag"]["lo95"] > 0)}
        b2["declared"] = dec
        for nm, sub in (("descriptive_new", new), ("descriptive_lemma_fresh", fresh)):
            m_ = k.pair.isin(sub).to_numpy()
            ks, Ys = k[m_].reset_index(drop=True), Y[m_].reset_index(drop=True)
            ps = Pairs(ks, a.n_boot, a.seed)
            Gs = ps(Ys.good - Ys.bad)
            BnP = ps(Ys.good - Ys["g<-b:P"])
            ctl = np.mean([ps(Ys.good - Ys[f"g<-b:{s}"]) for s in fam["magnat"]], 0)
            b2[nm] = {"pairs": ps.n, "G": ps.nats(Gs), "m_nat_P": ps.share(BnP, Gs),
                      "P_minus_magnat": ps.bt.stat(lambda ix: (BnP[ix].mean() - ctl[ix].mean()) / Gs[ix].mean())}
        res["B2"] = b2
        L += [f"Pairs {pp.n}; G = {f(b2['G'])}.", "", "| set | m_nat (share of G) | r_nat | B_same (same-class swap) |", "|---|---|---|---|"]
        L += [f"| {s} | {f(b2['m_nat'][s]['share'])} | {f(b2['r_nat'][s]['share'])} | {f(b2['B_same'][s]['share'])} |" for s in named]
        L += ["", "| controls (share of G) | m_nat | r_nat |", "|---|---|---|"]
        L += [f"| {x} | {f(mc[x]) if isinstance(mc[x], dict) else mc[x]} | {f(rc[x]) if isinstance(rc[x], dict) else rc[x]} |" for x in mc]
        L += ["", f"m_nat(P) in nats {f(b2['m_nat']['P']['nats'])}; P − same-class swap {f(b2['nat_minus_same_P'])}; "
                  f"P / ALL {f(b2['P_over_ALL']) if 'P_over_ALL' in b2 else 'n/a'}.", "",
              f"**Coordinate swap:** C_g {f(sw['C_g']['nats'])} nats = {f(sw['C_g']['share_G'])} of G; C_b {f(sw['C_b']['share_G'])}. "
              f"Held fixed during the swap, P blocks {f(sw['BC_g']['P'])} of C_g (ALL {f(sw['BC_g']['ALL'])}; "
              f"P − mag {f(sw['BC_g_controls']['P_minus_mag'])}); b-direction P {f(sw['BC_b']['P'])}.", "",
              f"**Declared:** {json.dumps(dec)}.", ""]
        for nm in ("descriptive_new", "descriptive_lemma_fresh"):
            x = b2[nm]
            L.append(f"- {nm} ({x['pairs']} pairs): G {f(x['G'])}; m_nat(P) {f(x['m_nat_P'])}; P − magnat {f(x['P_minus_magnat'])}.")
        L.append("")
    else:
        inval["B2"] = ck

    # ---------------- B3b
    ke, Ye, me = load(a.core_edit, "mediation") if (Path(a.core_edit) / "mediation_keys.parquet").exists() else (None, None, None)
    ph = pd.read_parquet(Path(a.phase1_dir) / "mediation_keys.parquet")
    zc = np.load(Path(a.phase1_dir) / "mediation.npz", allow_pickle=True)
    cke = me["checks"]
    ok_e = (ke[["item_id", "donor_pair", "plan_row_T", "plan_row_I"]].equals(ph[["item_id", "donor_pair", "plan_row_T", "plan_row_I"]])
            and float(np.abs(Ye.TT.to_numpy() - zc["by"][:, list(zc["conds"]).index("TT")]).max()) <= 1e-5
            and float(np.abs(Ye.II.to_numpy() - zc["by"][:, list(zc["conds"]).index("II")]).max()) <= 1e-5
            and cke["hook_out_max"] == 0 and cke["hook_in_max"] <= 1e-6 and cke["nan_cells"] == 0
            and all(v <= 1e-5 for kk, v in cke.items() if "empty_vs" in kk))
    L += ["## B3b. The core neurons' causal shares", "", f"Core-edit checks (keys and TT / II vs phase 1, hooks): {'pass' if ok_e else 'INVALID'}.", ""]
    if ok_e and "B2" in res:
        cn = [c.split(":", 1)[1] for c in Ye.columns if c.startswith("T<-I:") and c != "T<-I:empty"]
        pe, pi = Pairs(ke, a.n_boot, a.seed), Pairs(ke, a.n_boot, a.seed, weight="item")
        De, Di = pe(Ye.TT - Ye.II), pi(Ye.TT - Ye.II)
        G = pp(Y.good - Y.bad)
        b3b = {}
        for n in core + ["core_top3", "core_top5"]:
            s = n if n.startswith("core_top") else f"core_{n}"
            row = {"edit_m": pe.share(pe(Ye.TT - Ye[f"T<-I:{s}"]), De), "edit_r": pe.share(pe(Ye[f"I<-T:{s}"] - Ye.II), De),
                   "edit_m_item_weighted": pi.share(pi(Ye.TT - Ye[f"T<-I:{s}"]), Di),
                   "nat_m": pp.share(pp(Y.good - Y[f"g<-b:{s}"]), G), "nat_r": pp.share(pp(Y[f"b<-g:{s}"] - Y.bad), G)}
            if not n.startswith("core_top"):
                lay = n.split(".")[0][3:]
                ce = np.array([pe(Ye.TT - Ye[f"T<-I:{c}"]).mean() / De.mean() for c in cn if c.startswith(f"ctl{lay}_")])
                cnn = np.array([pp(Y.good - Y[f"g<-b:{c}"]).mean() / G.mean() for c in names if c.startswith(f"ctl{lay}_")])
                row.update({"edit_ctrl_mean": float(ce.mean()), "edit_ctrl_max": float(ce.max()),
                            "edit_rank": int((ce >= row["edit_m"]["est"]).sum()) + 1,
                            "nat_ctrl_mean": float(cnn.mean()), "nat_ctrl_max": float(cnn.max()),
                            "nat_rank": int((cnn >= row["nat_m"]["est"]).sum()) + 1})
            b3b[n] = row
        res["B3b"] = b3b
        L += ["| set | edit m | edit r | edit m (item-weighted) | natural m_nat | natural r_nat | layer controls m: edit mean / max, natural mean / max | rank edit / natural (of 21) |",
              "|---|---|---|---|---|---|---|---|"]
        for n, r in b3b.items():
            c = (f"{r['edit_ctrl_mean']:+.3f} / {r['edit_ctrl_max']:+.3f}, {r['nat_ctrl_mean']:+.3f} / {r['nat_ctrl_max']:+.3f}"
                 if "edit_ctrl_mean" in r else "")
            rk = f"{r['edit_rank']} / {r['nat_rank']}" if "edit_rank" in r else ""
            L.append(f"| {n} | {f(r['edit_m'])} | {f(r['edit_r'])} | {f(r['edit_m_item_weighted'])} | {f(r['nat_m'])} | {f(r['nat_r'])} | {c} | {rk} |")
        L.append("")
    else:
        inval["B3b"] = cke

    # ---------------- B3c
    labels = {n: v["reading"] for n, v in json.loads(Path(a.b7_decisions).read_text())["neurons"].items()}
    b3a = {n: v["reading"] for n, v in json.loads(Path(a.b3a_decisions).read_text())["neurons"].items()}
    assert not set(b3a) & set(labels), "B3a must not overwrite B7 labels"
    labels.update(b3a)
    if a.smoke:  # smoke: the smoke core (2 contexts) need not be the declared core
        labels = {n: labels.get(n, "unlabelled") for n in core}
    else:
        assert {n: labels.get(n) for n in core} == {n: EXPECTED_LABELS[n] for n in core}, "core labels differ from the rules"
    if "B2" in res:
        acts = pd.read_parquet(Path(a.natural) / "natural_core_acts.parquet")
        if not a.smoke:
            cols = json.loads(Path(a.fg_meta).read_text())["neuron_columns"]
            assert all(n in cols for n in core), "a core neuron has no stored C8 column"
        b = b3c(acts, k, a.fg, a.fg_meta, a.fg_data, core, labels, a.n_boot, a.seed)
        res["B3c"] = b
        L += ["## B3c. Frame profiles of the core neurons (natural T − I of the post-activation; reused activations)", ""]
        for pop, o in b["populations"].items():
            L += [f"**{pop}** (passives: {b['pairs']} half-B pairs; C8 lemmas T {o['lemmas_T']}, I {o['lemmas_I']}):", "",
                  "| neuron | label | passive | what (emb fill) | that (emb nofill) | fill − nofill | matrix what | matrix did | ρ = fill / passive |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for n, r in o["neurons"].items():
                L.append(f"| {n} | {r['label']} | {f(r['passive'])} | {f(r['emb_fill'])} | {f(r['emb_nofill'])} | "
                         f"{f(r['emb_fill_minus_nofill'])} | {f(r['mat_fill'])} | {f(r['mat_nofill'])} | "
                         f"{f(r['rho']) if 'rho' in r else 'n/a (passive lo95 ≤ 0)'} |")
            L += ["", f"D1 in this population: {o['D1_gap_reader_filler_dependent']}.", ""]
        L += [f"**Declared:** D1 (gap reader's class response depends on a filler; all lemmas) **{b['D1']}**.", ""]
    if inval:
        L += ["## INVALID parts (no decision written)", "", json.dumps(inval, indent=1), ""]
    Path(a.out).write_text(json.dumps({"results": res, "invalid": inval}, indent=1, default=float) + "\n")
    Path(a.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    if inval:
        raise SystemExit(f"INVALID parts: {sorted(inval)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--components", required=True)
    ap.add_argument("--natural", required=True)
    ap.add_argument("--core-edit", required=True)
    ap.add_argument("--phase1-dir", required=True)
    ap.add_argument("--switch", required=True)
    ap.add_argument("--fg", default="results/round5/shared/sr_natural_fg.parquet")
    ap.add_argument("--fg-meta", default="results/round5/shared/sr_meta.json")
    ap.add_argument("--fg-data", default="data/round4/fillergap")
    ap.add_argument("--b7-decisions", default="results/round5/wild/b7_decisions.json")
    ap.add_argument("--b3a-decisions", default="results/pythia/das_v2/mech_b/b3a_decisions.json")
    ap.add_argument("--items", default="data/verb_expansion/eval/passive_test/items.csv")
    ap.add_argument("--old-eval", default="data/verb_expansion/eval_pairs.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--smoke", action="store_true")
    run(ap.parse_args())
