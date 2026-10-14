#!/usr/bin/env python3
"""Round 5, C9 analysis: shared readers (reports/round5/plan.md, C9). No model.

Inputs: results/round5/shared/{sr_natural_fg, sr_natural_pass, sr_patch_fg, sr_patch_pass}.parquet, sr_meta.json;
data/round4/fillergap/kept_I_{exp}.csv. Populations: C8 primary (non-DAS T, kept non-DAS I). Contexts: half B
(primary) and all 24. Bootstrap (2,000 draws, seed 17): filler-gap, T and I verbs resampled independently x
contexts; passives, pairs within band x contexts. Numerator and denominator of every cross-construction ratio are
re-drawn in each draw (independent samples).
Outputs: reports/round5/c9_shared_readers.md, results/round5/shared/c9_summary.csv, c9_decisions.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LAYERS = tuple(range(8, 24))


def weights(rng, labels, n_boot):
    labels = np.asarray(labels)
    w = np.ones((n_boot + 1, len(labels)))
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        w[1:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=n_boot)
    return w


def ci(d, lo=2.5, hi=97.5):
    return np.percentile(d[1:], lo), np.percentile(d[1:], hi)


def fmt(d, p=2):
    lo, hi = ci(d)
    return f"{d[0]:+.{p}f} [{lo:+.{p}f}, {hi:+.{p}f}]"


class VC:
    """Mean of a per-item value over units (verbs or pairs) x contexts, with unit and context weights."""

    def __init__(self, units, ctxs, unit_strata, ctx_list, rng, n_boot):
        self.u = sorted(set(units))
        self.c = list(ctx_list)
        ui = {v: i for i, v in enumerate(self.u)}
        self.ui = np.array([ui[x] for x in units])
        ci_ = {v: i for i, v in enumerate(self.c)}
        self.ci = np.array([ci_.get(x, -1) for x in ctxs])
        strata = pd.Series(unit_strata, index=units).groupby(level=0).first().loc[self.u].to_numpy()
        self.wu = weights(rng, strata, n_boot)
        self.wc = weights(rng, np.zeros(len(self.c)), n_boot)

    def mean(self, v, mask=None):
        m = (self.ci >= 0) if mask is None else (mask & (self.ci >= 0))
        M = np.zeros((len(self.u), len(self.c)))
        N = np.zeros_like(M)
        np.add.at(M, (self.ui[m], self.ci[m]), v[m])
        np.add.at(N, (self.ui[m], self.ci[m]), 1)
        # per unit: context-weighted mean over its contexts, then units weighted
        num = np.einsum("bc,uc->bu", self.wc, M)
        den = np.einsum("bc,uc->bu", self.wc, N)
        um = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        ww = self.wu * (den > 0)
        return (ww * um).sum(1) / ww.sum(1)


def main(args):
    rdir = Path(args.res_dir)
    meta = json.loads((rdir / "sr_meta.json").read_text())
    sgn = np.array(meta["neuron_sign_IA"])
    rank = np.array(meta["neuron_rank"])
    sets = {"top50": np.ones(len(rank), bool), "top10": rank <= 10}
    nfg = pd.read_parquet(rdir / "sr_natural_fg.parquet")
    npa = pd.read_parquet(rdir / "sr_natural_pass.parquet")
    pfg = pd.read_parquet(rdir / "sr_patch_fg.parquet")
    ppa = pd.read_parquet(rdir / "sr_patch_pass.parquet")
    acols = [f"a{j}" for j in range(len(rank))]
    rng = np.random.default_rng(args.seed)
    lines = ["# Round 5, C9: shared readers", "",
             "Spec: `plan.md`, C9. Run: `run_shared_readers.py`; analysis: `analyze_shared_readers.py`. A = mean over the "
             "switch neurons of sign(I_A) × activation difference (T − I in filler-gap; good − bad in passives); D_in = at "
             "t_T − at t_I (site-8 in-range patch on I bases / bad passives). Populations: C8 primary (non-DAS T, kept "
             "non-DAS I); passives: the B7 items.", "",
             f"Checks: {json.dumps(meta['checks'])}.", ""]
    rows, dec = [], {}
    # passive reference: natural good - bad and in-range D_in (pairs within band x contexts)
    pctx = sorted(npa.context_id.unique())
    g, b = npa[npa.side == "good"].set_index("item_id"), npa[npa.side == "bad"].set_index("item_id")
    b = b.loc[g.index]
    Vp = VC(g.pair_id.to_numpy(), g.context_id.to_numpy(), g.band.to_numpy(), pctx, rng, args.n_boot)
    Vpp = VC(ppa.pair_id.to_numpy(), ppa.context_id.to_numpy(), ppa.band.to_numpy(), pctx, rng, args.n_boot)
    Vpp.wu, Vpp.wc = Vp.wu, Vp.wc  # the same pair and context draws for natural and patched (same B7 items)
    assert Vpp.u == Vp.u
    ref = {}
    for sname, sm in sets.items():
        dnat = ((g[acols].to_numpy() - b[acols].to_numpy())[:, sm] * sgn[sm]).mean(1)
        din = ((ppa[[f"neuT_{j}" for j in range(len(rank))]].to_numpy() - ppa[[f"neuI_{j}" for j in range(len(rank))]].to_numpy())[:, sm]
               * sgn[sm]).mean(1)
        ref[sname] = {"nat": Vp.mean(dnat), "in": Vpp.mean(din)}
    pvec_nat = ((g[acols].to_numpy() - b[acols].to_numpy()) * sgn).mean(0)
    lines += ["## Passive reference", "",
              "| set | natural good − bad A | in-range D_in(A) |", "|---|---|---|"] + \
             [f"| {s} | {fmt(r['nat'], 3)} | {fmt(r['in'], 3)} |" for s, r in ref.items()] + [""]
    for exp in ("emb", "mat"):
        kept = set(pd.read_csv(Path(args.fg_data) / f"kept_I_{exp}.csv").lemma)
        e = nfg[nfg.exp == exp]
        das = set(e[e.das_pair.fillna("") != ""].lemma)
        Tset = set(e[e.cls == "T"].lemma) - das
        Iset = kept - das
        for cxl, hv in (("half B", [1]), ("all contexts", [0, 1])):
            ctxs = sorted(e[e.half.isin(hv)].context_id.unique())
            key = f"{exp} | {cxl}"
            out = {}
            lines += [f"## {exp} ({'embedded' if exp == 'emb' else 'matrix'}), {cxl}", ""]
            # (a) natural T - I and in-range D_in in each frame
            nat = {}
            WC = weights(rng, np.zeros(len(ctxs)), args.n_boot)  # contexts jointly for T, I and patched bases
            for frame in ("nofill", "fill"):
                ef = e[e.frame == frame]
                VT = VC(ef[ef.lemma.isin(Tset)].lemma.to_numpy(), ef[ef.lemma.isin(Tset)].context_id.to_numpy(),
                        np.zeros(int(ef.lemma.isin(Tset).sum())), ctxs, rng, args.n_boot)
                VI = VC(ef[ef.lemma.isin(Iset)].lemma.to_numpy(), ef[ef.lemma.isin(Iset)].context_id.to_numpy(),
                        np.zeros(int(ef.lemma.isin(Iset).sum())), ctxs, rng, args.n_boot)
                VT.wc = VI.wc = WC
                if frame == "nofill":
                    WT, WI = VT.wu, VI.wu
                VT.wu, VI.wu = WT, WI  # the same verb draws in both frames
                for sname, sm in sets.items():
                    aT = (ef[ef.lemma.isin(Tset)][acols].to_numpy()[:, sm] * sgn[sm]).mean(1)
                    aI = (ef[ef.lemma.isin(Iset)][acols].to_numpy()[:, sm] * sgn[sm]).mean(1)
                    nat[(frame, sname)] = VT.mean(aT) - VI.mean(aI)
                vT = ef[ef.lemma.isin(Tset)][acols].to_numpy().mean(0) if frame == "fill" else None
                if frame == "fill":
                    vI = ef[ef.lemma.isin(Iset)][acols].to_numpy().mean(0)
                    fvec = (vT - vI) * sgn
            pf = pfg[(pfg.exp == exp) & pfg.lemma.isin(Iset) & pfg.context_id.isin(ctxs)]
            din = {}
            for frame in ("nofill", "fill"):
                q = pf[pf.frame == frame]
                Vd = VC(q.lemma.to_numpy(), q.context_id.to_numpy(), np.zeros(len(q)), ctxs, rng, args.n_boot)
                Vd.wc = WC
                assert Vd.u == VI.u, "patched bases must be the kept non-DAS I verbs"
                Vd.wu = WI
                for sname, sm in sets.items():
                    dv = ((q[[f"neuT_{j}" for j in range(len(rank))]].to_numpy()
                           - q[[f"neuI_{j}" for j in range(len(rank))]].to_numpy())[:, sm] * sgn[sm]).mean(1)
                    din[(frame, sname)] = Vd.mean(dv)
                if frame == "fill":
                    Vfill = Vd
                    qfill = q
            lines += ["| set | natural A: that | natural A: what | D_in(A): that | D_in(A): what | ρ_nat (what) | ρ_in (what) | "
                      "natural what − that | patched what − that |", "|---|---|---|---|---|---|---|---|---|"]
            for sname in sets:
                rn = nat[("fill", sname)] / ref[sname]["nat"]
                ri = din[("fill", sname)] / ref[sname]["in"]
                lines.append(f"| {sname} | {fmt(nat[('nofill', sname)], 3)} | {fmt(nat[('fill', sname)], 3)} | "
                             f"{fmt(din[('nofill', sname)], 3)} | {fmt(din[('fill', sname)], 3)} | {fmt(rn)} | {fmt(ri)} | "
                             f"{fmt(nat[('fill', sname)] - nat[('nofill', sname)], 3)} | "
                             f"{fmt(din[('fill', sname)] - din[('nofill', sname)], 3)} |")
                out[sname] = {"rho_nat": rn, "rho_in": ri, "patched_inter": din[("fill", sname)] - din[("nofill", sname)],
                              "ref_ok": bool(ci(ref[sname]["nat"])[0] > 0 and ci(ref[sname]["in"])[0] > 0)}
                for nm, v in (("rho_nat", rn), ("rho_in", ri), ("nat_what", nat[("fill", sname)]),
                              ("din_what", din[("fill", sname)]), ("patched_inter", out[sname]["patched_inter"])):
                    rows.append({"exp": exp, "contexts": cxl, "set": sname, "quantity": nm, "est": v[0],
                                 "lo": ci(v)[0], "hi": ci(v)[1]})
            agree = float(np.mean(np.sign(fvec) == np.sign(pvec_nat)))
            cos = float(fvec @ pvec_nat / (np.linalg.norm(fvec) * np.linalg.norm(pvec_nat)))
            step_fg = float((pf.tT - pf.tI).abs().mean())
            step_p = float((ppa.tT - ppa.tI).abs().mean())
            lines += ["", f"Per-neuron sign agreement (what-frame natural T − I vs passive good − bad, top 50): {agree:.2f}; "
                          f"cosine of the 50-dim difference vectors {cos:+.2f}. Coordinate steps |t_T − t_I| (raw): filler-gap "
                          f"{step_fg:.2f}, passive {step_p:.2f}.", ""]
            # (b) writers: what frame, in-range decomposition
            q = qfill
            Vd = Vfill
            num = "lp_dot" if exp == "emb" else "lp_q"
            dL = Vd.mean(q[f"d_{num}"].to_numpy() - q.d_lp_prep.to_numpy())
            dLt = Vd.mean(q.actual_lin.to_numpy())
            corr = Vd.mean(q.dprep_corr.to_numpy())
            comp_ok = abs(corr[0]) < 0.25 * abs(dLt[0])
            DEf = np.stack([Vd.mean(q[f"mlp{l}"].to_numpy()) for l in LAYERS], 1)  # [draw, layer]
            DEp = np.stack([Vpp.mean(ppa[f"mlp{l}"].to_numpy()) for l in LAYERS], 1)
            ATf = np.stack([Vd.mean(q[f"attn{l}"].to_numpy()) for l in LAYERS], 1)
            ATp = np.stack([Vpp.mean(ppa[f"attn{l}"].to_numpy()) for l in LAYERS], 1)
            posf, posp = np.clip(DEf, 0, None), np.clip(DEp, 0, None)
            pf_, pp_ = posf / posf.sum(1, keepdims=True), posp / posp.sum(1, keepdims=True)
            O = np.minimum(pf_, pp_).sum(1)
            gate_f = posf.sum(1)[0] >= 0.2 and ci(posf.sum(1))[0] > 0
            gate_p = posp.sum(1)[0] >= 0.2 and ci(posp.sum(1))[0] > 0
            topf = np.argsort(-DEf[1:], 1)[:, :5]
            freq = {LAYERS[j]: float(np.mean((topf == j).any(1))) for j in range(len(LAYERS))}
            lines += [f"Writers (what frame, in-range): D_in(L) {fmt(dL)}; D_in(L̃) {fmt(dLt)}; PREP lse − mean correction "
                      f"{fmt(corr)} → writer comparison {'made' if comp_ok else 'NOT made (correction ≥ 0.25 |D_in(L̃)|)'}. "
                      f"Carry {fmt(Vd.mean(q.carry.to_numpy()))}; LN term {fmt(Vd.mean(q.ln.to_numpy()))}; residual max "
                      f"{float(np.abs(q.resid).max()):.1e}.", "",
                      "| layer | MLP DE: filler-gap L̃ | MLP DE: passive by | attn DE: filler-gap | attn DE: passive | top-5 frequency (fg) |",
                      "|---:|---|---|---|---|---:|"]
            for j, l in enumerate(LAYERS):
                lines.append(f"| {l} | {fmt(DEf[:, j])} | {fmt(DEp[:, j])} | {fmt(ATf[:, j])} | {fmt(ATp[:, j])} | {freq[l]:.2f} |")
            lines += ["", f"Positive MLP totals: filler-gap {fmt(posf.sum(1))} (negative {fmt(np.clip(DEf, None, 0).sum(1))}); "
                          f"passive {fmt(posp.sum(1))} (negative {fmt(np.clip(DEp, None, 0).sum(1))}). Magnitude gate: filler-gap "
                          f"{'passes' if gate_f else 'fails'}, passive {'passes' if gate_p else 'fails'}. Profile overlap O {fmt(O)}.", ""]
            out["writers"] = {"O": O, "comp_ok": bool(comp_ok), "gates": bool(gate_f and gate_p)}
            rows.append({"exp": exp, "contexts": cxl, "set": "-", "quantity": "overlap_O", "est": O[0], "lo": ci(O)[0], "hi": ci(O)[1]})
            dec[key] = out
    # decisions (half B, both experiments)
    D = {}
    for sname in ("top50", "top10"):
        rr = [dec[f"{e} | half B"][sname] for e in ("emb", "mat")]
        if not all(r["ref_ok"] for r in rr):
            D[sname] = "unresolved (passive reference not above 0)"
            continue
        shared = all(r[k][0] >= 0.5 and ci(r[k])[0] > 0 for r in rr for k in ("rho_nat", "rho_in"))
        small = all(-0.25 < ci(r[k], 5, 95)[0] and ci(r[k], 5, 95)[1] < 0.25 for r in rr for k in ("rho_nat", "rho_in"))
        rev = any(ci(r[k])[1] < 0 for r in rr for k in ("rho_nat", "rho_in"))
        D[sname] = "shared" if shared else "small relative to passives" if small else "reversed" if rev else "intermediate / imprecise"
        D[f"{sname}_filler_conditioned"] = all(ci(r["patched_inter"])[0] > 0 for r in rr)
    ww = [dec[f"{e} | half B"]["writers"] for e in ("emb", "mat")]
    if not all(w["comp_ok"] and w["gates"] for w in ww):
        D["writers"] = "unresolved (correction or magnitude gate)"
    elif all(w["O"][0] >= 0.5 and ci(w["O"])[0] >= 0.35 for w in ww):
        D["writers"] = "shared dominant locations"
    elif all(ci(w["O"])[1] <= 0.35 for w in ww):
        D["writers"] = "different dominant locations"
    else:
        D["writers"] = "unresolved"
    lines += ["## Declared decisions (half B, both experiments)", "",
              f"- Class-reader recruitment (top 50): **{D['top50']}**; top 10 (sensitivity): {D['top10']}.",
              f"- Filler-conditioned recruitment (patched what − that, CI above 0 in both): top 50 "
              f"**{D.get('top50_filler_conditioned')}**; top 10 {D.get('top10_filler_conditioned')}.",
              f"- Writer locations: **{D['writers']}**.", ""]
    Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.report_dir) / "c9_shared_readers.md").write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(rdir / "c9_summary.csv", index=False)
    (rdir / "c9_decisions.json").write_text(json.dumps(D, indent=1, default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-dir", default="results/round5/shared")
    ap.add_argument("--fg-data", default="data/round4/fillergap")
    ap.add_argument("--report-dir", default="reports/round5")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    main(ap.parse_args())
