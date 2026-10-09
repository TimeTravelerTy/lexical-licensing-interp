#!/usr/bin/env python3
"""Round 4, D11: conjunction neurons in MLPs 11 and 14 (reports/round4/plan.md, D11): analysis.

From `run_conjunction.py`: per context half x pair x frame (was, has) x donor class (T, I) mean
activations (context halves pooled here). Per neuron, per pair: I_n = (a[was,T] - a[was,I]) - (a[has,T] - a[has,I]).
Pairs are split in halves within band (seed 17):
- selection on pair half A: one-sample t-test across pairs (two-sided; Student t via the regularized
  incomplete beta), BH-FDR q < 0.05 over all 16,384 neurons; categories from the simple effects
  (passive-conjunction: was effect in the direction of I_n and |has| < 0.5 |was|);
- replication on pair half B for the passive-conjunction set: sign agreement, aligned I_n, was and has
  effects with pair-bootstrap CIs;
- their share of the MLPs' " by" switch (direct effect of I_n through one reference final-LN scale per pair),
  half B; and of the passive T - I effect;
- generalization: sign-aligned activation difference of the set, natural good - bad passives per band,
  and the C8 nonce probes (matched T - I, balanced AB - BA).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def betacf(a, b, x, iters=300):
    """Continued fraction for the regularized incomplete beta (Numerical Recipes), vectorized."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c = np.ones_like(x)
    d = 1 - qab * x / qap
    d = np.where(np.abs(d) < tiny, tiny, d)
    d = 1 / d
    h = d.copy()
    for m in range(1, iters + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = np.where(np.abs(d) < tiny, tiny, d)
        c = 1 + aa / c
        c = np.where(np.abs(c) < tiny, tiny, c)
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = np.where(np.abs(d) < tiny, tiny, d)
        c = 1 + aa / c
        c = np.where(np.abs(c) < tiny, tiny, c)
        d = 1 / d
        de = d * c
        h *= de
        if np.all(np.abs(de - 1) < 1e-12):
            break
    return h


def betainc(a, b, x):
    from math import lgamma

    x = np.clip(np.asarray(x, float), 0, 1)
    lbt = lgamma(a + b) - lgamma(a) - lgamma(b)
    with np.errstate(divide="ignore"):
        bt = np.exp(lbt + a * np.log(np.where(x > 0, x, 1)) + b * np.log(np.where(x < 1, 1 - x, 1)))
    bt = np.where((x == 0) | (x == 1), 0, bt)
    sw = x < (a + 1) / (a + b + 2)
    out = np.where(sw, bt * betacf(a, b, np.where(sw, x, 0.5)) / a,
                   1 - bt * betacf(b, a, np.where(sw, 0.5, 1 - x)) / b)
    return np.where(x == 0, 0, np.where(x == 1, 1, out))


def t_pvalue(t, df):
    return betainc(df / 2, 0.5, df / (df + t ** 2))


def bh(p, q):
    n = len(p)
    o = np.argsort(p)
    thr = q * np.arange(1, n + 1) / n
    passed = p[o] <= thr
    k = np.max(np.flatnonzero(passed)) + 1 if passed.any() else 0
    sel = np.zeros(n, bool)
    sel[o[:k]] = True
    return sel


def cell_means(z, key):
    """Pool the two context halves: [P, frame, cls, ...] means."""
    a, n = z[key].sum(0), z["cnt"].sum(0)
    return a / np.maximum(n, 1).reshape(n.shape + (1,) * (a.ndim - n.ndim))


def pair_halves(bands, seed):
    rng = np.random.default_rng(seed)
    half = np.zeros(len(bands), int)
    for b in np.unique(bands):
        ix = rng.permutation(np.flatnonzero(bands == b))
        half[ix[len(ix) // 2:]] = 1
    return half


def bci(v, B, rng):
    bs = np.r_[v.mean(), [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]]
    return {"est": float(bs[0]), "lo95": float(np.percentile(bs[1:], 2.5)), "hi95": float(np.percentile(bs[1:], 97.5))}


def run(args):
    d = Path(args.dir)
    z = np.load(d / "conjunction.npz", allow_pickle=True)  # band / lemma labels are object arrays
    layers = [int(l) for l in z["layers"]]
    rng = np.random.default_rng(args.seed)
    A = np.concatenate([cell_means(z, f"a_{l}") for l in layers], -1)  # [P, 2, 2, 2D]
    sig = cell_means(z, "sigma")[:, 0].mean(1)  # per pair: mean final-LN scale of the was-frame patched runs
    wby = np.concatenate([z[f"w_by_{l}"] for l in layers])
    was, has = A[:, 0, 0] - A[:, 0, 1], A[:, 1, 0] - A[:, 1, 1]  # [P, 2D]
    I = was - has
    DEI = I * wby[None] / sig[:, None]  # direct effect of the interaction (the switch) on " by"
    DEW = was * wby[None] / sig[:, None]
    bands = z["bands"]
    ph = pair_halves(bands, args.seed)
    A_, B_ = ph == 0, ph == 1
    D = 8192
    names = np.array([f"MLP{l}.n{n}" for l in layers for n in range(D)])
    xa = I[A_]
    t = xa.mean(0) / (xa.std(0, ddof=1) / np.sqrt(len(xa)))
    p = t_pvalue(t, len(xa) - 1)
    sel = bh(p, args.q)
    sgn = np.sign(xa.mean(0))
    wA, hA = (was[A_] * sgn).mean(0), (has[A_] * sgn).mean(0)
    cat = np.full(len(sel), "", object)
    cat[sel & (wA > 0) & (np.abs(hA) < 0.5 * wA)] = "passive-conjunction"
    cat[sel & (hA < 0) & (np.abs(wA) < 0.5 * np.abs(hA))] = "active-conjunction"
    cat[sel & (cat == "") & (np.sign(wA) == np.sign(hA))] = "graded (same sign in both frames)"
    cat[sel & (cat == "")] = "opposite signs"
    out = {"n_selected": int(sel.sum()), "pairs_A": int(A_.sum()), "pairs_B": int(B_.sum()),
           "by_layer": {f"MLP{l}": int(sel[i * D:(i + 1) * D].sum()) for i, l in enumerate(layers)},
           "categories": {c: int((cat == c).sum()) for c in sorted(set(cat[sel]))}}
    pc = cat == "passive-conjunction"
    rows = []
    if pc.any():
        xb = I[B_][:, pc] * sgn[pc]
        agree = float((np.sign(I[B_][:, pc].mean(0)) == sgn[pc]).mean())
        mI = bci(xb.mean(1), args.n_boot, rng)
        mw = bci((was[B_][:, pc] * sgn[pc]).mean(1), args.n_boot, rng)
        mh = bci((has[B_][:, pc] * sgn[pc]).mean(1), args.n_boot, rng)
        set_I, all_I = DEI[B_][:, pc].sum(1), DEI[B_].sum(1)
        set_W, all_W = DEW[B_][:, pc].sum(1), DEW[B_].sum(1)

        def share(a, b):
            n = len(a)
            bs = [(lambda ix: a[ix].mean() / b[ix].mean())(rng.integers(0, n, n)) for _ in range(args.n_boot)]
            return {"est": float(a.mean() / b.mean()), "lo95": float(np.percentile(bs, 2.5)), "hi95": float(np.percentile(bs, 97.5))}

        out.update({"halfB_sign_agreement": agree, "halfB_aligned_I": mI, "halfB_aligned_was": mw, "halfB_aligned_has": mh,
                    "switch_share": share(set_I, all_I), "passive_TI_share": share(set_W, all_W),
                    "de_switch_set": float(set_I.mean()), "de_switch_all": float(all_I.mean())})
        out["exist"] = bool(pc.sum() >= 10 and agree >= 0.8 and mI["lo95"] > 0 and mw["lo95"] > 0
                            and abs(mh["est"]) < 0.5 * mw["est"])
        sh = out["switch_share"]
        out["carry_by"] = bool(sh["est"] >= 0.25 and sh["lo95"] > 0)
        top = np.flatnonzero(pc)[np.argsort(-np.abs(t[pc]))][:args.top]
        for i in top:
            rows.append({"neuron": names[i], "t_A": float(t[i]), "p_A": float(p[i]), "I_A": float(xa[:, i].mean()),
                         "I_B": float(I[B_][:, i].mean()), "was_T_minus_I": float(was[:, i].mean()),
                         "has_T_minus_I": float(has[:, i].mean()), "w_by": float(wby[i])})
        nat = np.concatenate([z[f"nat_{l}"] / np.maximum(z["ncnt"], 1)[..., None] for l in layers], -1)  # [P, 2, 2D]
        gb = ((nat[:, 0] - nat[:, 1])[:, pc] * sgn[pc]).mean(1)
        gen = {}
        for b in ("head", "tail", "xtail"):
            gen[b] = {**bci(gb[bands == b], args.n_boot, rng), "n_pairs": int((bands == b).sum())}
        vh, vx = gb[bands == "head"], gb[bands == "xtail"]
        ratio = [vx[rng.integers(0, len(vx), len(vx))].mean() / vh[rng.integers(0, len(vh), len(vh))].mean()
                 for _ in range(args.n_boot)]
        gen["xtail_over_head"] = {"est": float(vx.mean() / vh.mean()), "lo95": float(np.percentile(ratio, 2.5)),
                                  "hi95": float(np.percentile(ratio, 97.5))}
        non = np.concatenate([z[f"non_{l}"] / np.maximum(z["ccnt"], 1)[..., None] for l in layers], -1)
        conds = list(z["nonce_conds"])
        for nm, (c1, c2) in (("nonce matched T - I", ("matched_T", "matched_I")),
                             ("nonce balanced AB - BA", ("balanced_AB", "balanced_BA"))):
            gen[nm] = bci(((non[:, conds.index(c1)] - non[:, conds.index(c2)])[:, pc] * sgn[pc]).mean(1), args.n_boot, rng)
        out["generalization"] = gen
        out["exploratory"] = exploratory(args, d, A, was, I, DEI, wby, nat, bands, A_, B_, pc, names)
    else:
        out["exist"], out["carry_by"] = False, False
    pd.DataFrame(rows).to_csv(d / "conjunction_top.csv", index=False)
    (d / "conjunction_decisions.json").write_text(json.dumps(out, indent=2) + "\n")
    write_report(args, out, pd.DataFrame(rows))


def exploratory(args, d, A, was, I, DEI, wby, nat, bands, A_, B_, pc, names):
    """Not declared (results.md, D11, exploratory): concentration of the switch in the passive-conjunction
    neurons ranked by their half-A direct effect; the top-k sets' natural good - bad activation difference
    weighted by the " by" output weight (unscaled) per band, against their patched was T - I response; mean
    activation levels per frame (GELU operating point). Writes the ranked set to
    `conjunction_switch_neurons.csv` (the E13-T neuron oracle uses its top 50)."""
    rng = np.random.default_rng(args.seed)
    deA, deB = DEI[A_].mean(0), DEI[B_].mean(0)
    tot = deB.sum()
    idx = np.flatnonzero(pc)
    order = idx[np.argsort(-deA[idx])]
    x = {"switch_DE_halfB_all": float(tot),
         "concentration_halfB": {str(k): float(deB[order[:k]].sum() / tot) for k in (10, 50, 100, len(order))},
         "layer_share_halfB": {"MLP11": float(deB[:len(deB) // 2].sum() / tot), "MLP14": float(deB[len(deB) // 2:].sum() / tot)}}
    gbw = (nat[:, 0] - nat[:, 1]) * wby
    x["natural_all_by_band"] = {b: float(gbw[bands == b].sum(1).mean()) for b in ("head", "tail", "xtail")}
    x["top_sets"] = {}
    for k in (10, 50, 100):
        s = order[:k]
        gb, pw = gbw[:, s].sum(1), (was[:, s] * wby[s]).sum(1)
        e = {}
        for b in ("head", "tail", "xtail"):
            v = gb[bands == b]
            bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(args.n_boot)]
            e[b] = {"est": float(v.mean()), "lo95": float(np.percentile(bs, 2.5)), "hi95": float(np.percentile(bs, 97.5))}
        vh, vx = gb[bands == "head"], gb[bands == "xtail"]
        r = [vx[rng.integers(0, len(vx), len(vx))].mean() / vh[rng.integers(0, len(vh), len(vh))].mean()
             for _ in range(args.n_boot)]
        e["xtail_over_head"] = {"est": float(vx.mean() / vh.mean()), "lo95": float(np.percentile(r, 2.5)),
                                "hi95": float(np.percentile(r, 97.5))}
        e["patched_was_by_band"] = {b: float(pw[bands == b].mean()) for b in ("head", "tail", "xtail")}
        x["top_sets"][str(k)] = e
    lw, lh = A[:, 0].mean((0, 1)), A[:, 1].mean((0, 1))
    x["negative_level"] = {nm: {"has": float((lh[m] < 0).mean()), "was": float((lw[m] < 0).mean())}
                           for nm, m in (("passive-conjunction", pc), ("all", np.ones_like(pc)))}
    pd.DataFrame({"rank": np.arange(1, len(order) + 1), "neuron": names[order],
                  "layer": [int(n.split(".")[0][3:]) for n in names[order]],
                  "index": [int(n.split(".n")[1]) for n in names[order]],
                  "de_switch_A": deA[order], "de_switch_B": deB[order], "share_B": deB[order] / tot,
                  "I_A": I[A_][:, order].mean(0), "I_B": I[B_][:, order].mean(0), "w_by": wby[order]}
                 ).to_csv(d / "conjunction_switch_neurons.csv", index=False)
    return x


def write_report(args, o, top):
    f = lambda c: f"{c['est']:+.3f} [{c['lo95']:+.3f}, {c['hi95']:+.3f}]"
    L = ["# Round 4, D11. Conjunction neurons in MLPs 11 and 14", "",
         "Spec: `plan.md`, D11. Run: `run_conjunction.py` (fp32); analysis: `analyze_conjunction.py`. B7 items "
         "(32 contexts per pair) with T and I donors, was and has frames, site-8 interchange. I_n = (was T − I) − (has T "
         "− I) of the neuron's post-activation at the participle's last token, per pair. Pairs split in halves within "
         f"band (seed 17): selection on half A ({o['pairs_A']} pairs; t-test, BH-FDR q < {args.q}), tests on half B "
         f"({o['pairs_B']} pairs). Direct effects on \" by\" through one reference final-LN scale per pair.", "",
         f"**Selected on half A:** {o['n_selected']} neurons ({', '.join(f'{k}: {v}' for k, v in o['by_layer'].items())}); "
         "categories: " + ", ".join(f"{k} {v}" for k, v in o["categories"].items()) + "."]
    if "halfB_aligned_I" in o:
        L += [f"- Passive-conjunction set on half B: sign agreement {o['halfB_sign_agreement']:.2f}; aligned I_n "
              f"{f(o['halfB_aligned_I'])}; was T − I {f(o['halfB_aligned_was'])}; has T − I {f(o['halfB_aligned_has'])}.",
              f"- **Conjunction neurons exist: {'yes' if o['exist'] else 'no'}** (≥ 10 passive-conjunction neurons; half B: "
              "≥ 80% sign agreement, aligned I_n and was effect CIs above 0, |has| < 0.5 × was).",
              f"- Share of the MLP 11 + 14 \" by\" switch (direct effect of I_n, half B): {f(o['switch_share'])} "
              f"({o['de_switch_set']:+.3f} of {o['de_switch_all']:+.3f}); share of the passive T − I effect: "
              f"{f(o['passive_TI_share'])} → **they carry the by switch: {'yes' if o['carry_by'] else 'no'}** "
              "(switch share ≥ 0.25, CI above 0).", "", "Generalization (sign-aligned mean activation difference of the "
              "passive-conjunction set):", ""]
        g = o["generalization"]
        L += [f"- natural good − bad passives: Head {f(g['head'])} ({g['head']['n_pairs']} pairs), Tail {f(g['tail'])}, "
              f"XTail {f(g['xtail'])}; XTail / Head {f(g['xtail_over_head'])}",
              f"- nonce probes: matched T − I {f(g['nonce matched T - I'])}; balanced AB − BA {f(g['nonce balanced AB - BA'])}", "",
              "Top passive-conjunction neurons (|t| on half A):", "",
              "| neuron | t (A) | I (A) | I (B) | was: T − I | has: T − I | w_by |", "|---|---:|---:|---:|---:|---:|---:|"]
        for r in top.itertuples():
            L.append(f"| {r.neuron} | {r.t_A:+.1f} | {r.I_A:+.3f} | {r.I_B:+.3f} | {r.was_T_minus_I:+.3f} | "
                     f"{r.has_T_minus_I:+.3f} | {r.w_by:+.4f} |")
        x = o["exploratory"]
        sw = pd.read_csv(Path(args.dir) / "conjunction_switch_neurons.csv").head(5)
        L += ["", "## Exploratory (not declared)", "",
              "Passive-conjunction neurons ranked by their half-A direct effect on the switch; shares on half B "
              f"(of the whole MLP 11 + 14 switch, {x['switch_DE_halfB_all']:+.3f}): " +
              ", ".join(f"top {k} {v:.2f}" for k, v in x["concentration_halfB"].items()) +
              f". MLP11 / MLP14 share of the switch: {x['layer_share_halfB']['MLP11']:.2f} / "
              f"{x['layer_share_halfB']['MLP14']:.2f}. Largest: " +
              ", ".join(f"{r.neuron} ({r.share_B:.2f})" for r in sw.itertuples()) + ".", "",
              "Natural good − bad activation difference × \" by\" output weight (unscaled), summed over the set, per "
              "band (pair bootstrap); patched was T − I in the same units:", "",
              "| set | Head | Tail | XTail | XTail / Head | patched was T − I (Head / Tail / XTail) |",
              "|---|---|---|---|---|---|"]
        for k, e in x["top_sets"].items():
            pw = e["patched_was_by_band"]
            L.append(f"| top {k} | {f(e['head'])} | {f(e['tail'])} | {f(e['xtail'])} | {f(e['xtail_over_head'])} | "
                     f"{pw['head']:+.2f} / {pw['tail']:+.2f} / {pw['xtail']:+.2f} |")
        na = x["natural_all_by_band"]
        nl = x["negative_level"]
        L += ["", f"All MLP 11 + 14 neurons, same quantity: Head {na['head']:+.2f}, Tail {na['tail']:+.2f}, XTail "
              f"{na['xtail']:+.2f}. Share of neurons with negative mean post-activation (has / was frame): "
              f"passive-conjunction {nl['passive-conjunction']['has']:.2f} / {nl['passive-conjunction']['was']:.2f}, "
              f"all {nl['all']['has']:.2f} / {nl['all']['was']:.2f}."]
    L.append("")
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/round4/translation")
    ap.add_argument("--report", default="reports/round4/d11_conjunction.md")
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
