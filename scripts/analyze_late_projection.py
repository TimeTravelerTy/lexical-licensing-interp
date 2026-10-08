#!/usr/bin/env python3
"""Is the site-8 separation still present late? (round3_plan.md, A2)

Input: `late_projection.npz` from `run_late_projection.py` (raw dot products of
the last-token residual at sites 4..23 with the 15 fold bases of d_8 and d_17,
and with the own-site bases at 4, 6, 10, 12, 14, 16).

Read-outs (fixed across sites; sign aligned once, at the read-out's home site,
with the DAS training items, so a later sign flip shows as a negative gap):
- `d8`: x_s . d_8 (sign from site 8);
- `d8perp17`: x_s . e, e = (d_8 - c d_17) / sqrt(1 - c^2), c = d_8 . d_17 per fold (sign from site 8);
- `d17`: x_s . d_17 (sign from site 17), descriptive;
- `own`: x_s . d_s at the DAS sites (sign from site s), descriptive.
Cross-fitting as in `frame_z` (DAS verbs only use bases where their pair is
held out). Raw projection units throughout.

Per site, read-out and frame: per-pair raw gap (passives: good - bad over
contexts; actives: same verbs over subjects), mean over pairs; retention
gap_s / gap_8; d' (items); AUC over items and over verb means; passive/active
ratio = mean passive gap / mean active gap. 95% CIs: pair bootstrap (2,000
draws, seed 17), the same pair resample at every site, so differences from
site 8 are paired. Decision rules: `round3_plan.md`, A2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_passive_test import frame_z, populations

DECLARED = (12, 14, 16, 17, 20, 23)
DAS_SITES = (4, 6, 8, 10, 12, 14, 16, 17)


def auc(pos, neg):
    x = np.r_[pos, neg]
    r = pd.Series(x).rank().to_numpy()
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def signs_at(x, das, pairs_folds, keys):
    """Per basis column: sign of (training transitive - training intransitive) mean projection."""
    out = {}
    for j, (k, f) in enumerate(keys):
        fo = das.pair_id.map(dict(zip(pairs_folds.pair_id, pairs_folds[f"fold_split{k}"]))).to_numpy()
        tr = (fo != f) & (das.subject != "David").to_numpy()
        xd = x[das.pid.to_numpy(), j]
        out[f"s{k}_f{f}"] = float(np.sign(xd[tr & (das.cls == 1).to_numpy()].mean()
                                          - xd[tr & (das.cls == 0).to_numpy()].mean()))
    return out


def das_reference(x, sg, das, pairs_folds, keys):
    """Held-out DAS actives: cross-fitted sign-aligned projection, transitive vs intransitive."""
    vals = np.zeros(len(das))
    for i, r in enumerate(das.itertuples()):
        fo = pairs_folds.set_index("pair_id").loc[r.pair_id]
        js = [j for j, (k, f) in enumerate(keys) if fo[f"fold_split{k}"] == f]
        vals[i] = np.mean([x[r.pid, j] * sg[f"s{keys[j][0]}_f{keys[j][1]}"] for j in js])
    t, n = vals[das.cls.to_numpy() == 1], vals[das.cls.to_numpy() == 0]
    return t.mean() - n.mean(), auc(t, n)


def run(args):
    z = np.load(args.npz)
    sites = list(z["sites"])
    keys = [(int(c[1]), int(c.split("_f")[1])) for c in z["keys"]]
    cols = [f"s{k}_f{f}" for k, f in keys]
    pid = z["pid"]
    items = pd.read_csv(args.items)
    P = pd.read_csv(args.prompts)
    assert (P.pid.to_numpy() == pid).all()
    root = Path(args.root)
    pairs_folds = pd.read_csv(root / "final_strict" / "pairs_folds.csv")
    das = pd.read_csv(root / "final_strict" / "items.csv")
    das["pid"] = das.prompt.map(dict(zip(P.prompt, P.pid)))
    pr, pops = populations(items)
    prim = list(pr.index[pops["primary (plain)"].to_numpy()])
    npz_b = np.load(args.bases)
    c = np.array([float(npz_b[f"das_site8_s{k}_f{f}"] @ npz_b[f"das_site17_s{k}_f{f}"]) for k, f in keys])
    p8, p17 = z["p8"], z["p17"]
    pe = (p8 - c[None, None] * p17) / np.sqrt(1 - c ** 2)[None, None]
    own_sites = list(z["own"])
    si = {s: i for i, s in enumerate(sites)}

    def proj(readout, s):
        if readout == "d8":
            return p8[:, si[s]]
        if readout == "d8perp17":
            return pe[:, si[s]]
        if readout == "d17":
            return p17[:, si[s]]
        if s == 8:
            return p8[:, si[8]]
        if s == 17:
            return p17[:, si[17]]
        return z["pown"][:, own_sites.index(s)]

    home = {"d8": 8, "d8perp17": 8, "d17": 17}
    rng = np.random.default_rng(args.seed)
    bidx = np.vstack([np.arange(len(prim))[None], rng.integers(0, len(prim), (args.n_boot, len(prim)))])
    rows, gaps = [], {}
    for readout in ("d8", "d8perp17", "d17", "own"):
        rsites = DAS_SITES if readout == "own" else sites
        for s in rsites:
            x = proj(readout, s)
            sg = signs_at(proj(readout, home.get(readout, s)), das, pairs_folds, keys)
            df = pd.DataFrame(x, columns=cols).assign(pid=pid)
            it, at = frame_z(df, items, P, pairs_folds, das_items_for(das), pr, signs=sg)
            it, at = it[it.pair_id.isin(prim)], at[at.pair_id.isin(prim)]
            gp = it.groupby("pair_id").rdiff.mean().reindex(prim).to_numpy()
            ga = at.groupby("pair_id").rdiff.mean().reindex(prim).to_numpy()
            gaps[(readout, s)] = (gp, ga)
            # verb-mean AUC: good verbs vs bad verbs (passive: over contexts; active: over subjects)
            vp_g, vp_b = it.groupby("good_lemma").rg.mean(), it.groupby("bad_lemma").rb.mean()
            at_l = at.merge(pr[["good_lemma", "bad_lemma"]], left_on="pair_id", right_index=True)
            va_g, va_b = at_l.groupby("good_lemma").rg.mean(), at_l.groupby("bad_lemma").rb.mean()
            dref_gap, dref_auc = das_reference(x, sg, das, pairs_folds, keys)
            for frame, g, pos, neg, vpos, vneg in (("passive", gp, it.rg, it.rb, vp_g, vp_b),
                                                   ("active", ga, at.rg, at.rb, va_g, va_b)):
                draws = g[bidx].mean(1)
                rows.append({"readout": readout, "site": s, "frame": frame, "gap": draws[0],
                             "gap_lo": np.percentile(draws[1:], 2.5), "gap_hi": np.percentile(draws[1:], 97.5),
                             "dprime": (pos.mean() - neg.mean()) / np.sqrt((pos.var() + neg.var()) / 2),
                             "auc_items": auc(pos.to_numpy(), neg.to_numpy()),
                             "auc_verbs": auc(vpos.to_numpy(), vneg.to_numpy()),
                             "das_ref_gap": dref_gap, "das_ref_auc": dref_auc})
    res = pd.DataFrame(rows)
    # ratios, retention and paired differences from site 8
    rat = []
    for readout in ("d8", "d8perp17", "d17", "own"):
        rsites = DAS_SITES if readout == "own" else sites
        ref = 8
        gp8, ga8 = gaps[(readout, ref)]
        r8 = gp8[bidx].mean(1) / ga8[bidx].mean(1)
        for s in rsites:
            gp, ga = gaps[(readout, s)]
            mp, ma = gp[bidx].mean(1), ga[bidx].mean(1)
            r = mp / ma
            q = lambda v: (v[0], np.percentile(v[1:], 2.5), np.percentile(v[1:], 97.5))
            rat.append({"readout": readout, "site": s, **dict(zip(("ratio", "ratio_lo", "ratio_hi"), q(r))),
                        **dict(zip(("dratio", "dratio_lo", "dratio_hi"), q(r - r8))),
                        **dict(zip(("ret_passive", "ret_passive_lo", "ret_passive_hi"), q(mp / gp8[bidx].mean(1)))),
                        **dict(zip(("ret_active", "ret_active_lo", "ret_active_hi"), q(ma / ga8[bidx].mean(1)))),
                        "mean_of_pair_ratios": float(np.mean(gp / ga))})
    rat = pd.DataFrame(rat)
    out = Path(args.out_dir)
    res.to_csv(out / "late_projection_summary.csv", index=False)
    rat.to_csv(out / "late_projection_ratios.csv", index=False)
    write_report(args, res, rat)


def das_items_for(das):
    return das.drop(columns=["pid"])


def write_report(args, res, rat):
    L = ["# A2. Is the site-8 separation still present late?", "",
         "Spec: `round3_plan.md`, A2. Run: `run_late_projection.py`; analysis: `analyze_late_projection.py`. No "
         "patching. Raw projection units; 64 primary pairs; 95% CIs: pair bootstrap (2,000 draws, seed 17), paired "
         "across sites. `d8perp17` = d_8 with the same fold's d_17 component removed (one fixed read-out at every "
         "site). Ratio = mean passive gap / mean active gap over pairs.", ""]
    for readout, title in (("d8", "d_8"), ("d8perp17", "d_8 ⊥ d_17"), ("d17", "d_17 (descriptive)"),
                           ("own", "own-site d_s (descriptive)")):
        r = res[res.readout == readout]
        q = rat[rat.readout == readout].set_index("site")
        L += [f"## Read-out: {title}", "",
              "| Site | Passive gap [CI] | AUC verbs (items) | Active gap [CI] | AUC verbs (items) | Retention P / A | "
              "Ratio [CI] | Ratio − ratio₈ [CI] | Present (P / A) |",
              "|---:|---|---|---|---|---|---|---|---|"]
        for s in sorted(r.site.unique()):
            p = r[(r.site == s) & (r.frame == "passive")].iloc[0]
            a = r[(r.site == s) & (r.frame == "active")].iloc[0]
            t = q.loc[s]
            pres = lambda x: "yes" if x.auc_verbs >= 0.8 and (x.gap_lo > 0 or x.gap_hi < 0) else "no"
            mark = "**" if s in DECLARED or s == 8 else ""
            L.append(f"| {mark}{s}{mark} | {p.gap:+.2f} [{p.gap_lo:+.2f}, {p.gap_hi:+.2f}] | {p.auc_verbs:.3f} "
                     f"({p.auc_items:.3f}) | {a.gap:+.2f} [{a.gap_lo:+.2f}, {a.gap_hi:+.2f}] | {a.auc_verbs:.3f} "
                     f"({a.auc_items:.3f}) | {t.ret_passive:.2f} / {t.ret_active:.2f} | {t.ratio:.2f} "
                     f"[{t.ratio_lo:.2f}, {t.ratio_hi:.2f}] | {t.dratio:+.2f} [{t.dratio_lo:+.2f}, {t.dratio_hi:+.2f}] "
                     f"| {pres(p)} / {pres(a)} |")
        L.append("")
    # declared decision
    e = rat[rat.readout == "d8perp17"].set_index("site")
    o = rat[rat.readout == "own"].set_index("site")
    L += ["## Decision (declared sites 12–17)", "",
          "| Site | d8⊥17 ratio − ratio₈ [CI] | within ±0.1? | own-site ratio − own₈ [CI] | own falls (CI < 0)? | Reading |",
          "|---:|---|---|---|---|---|"]
    for s in (12, 14, 16, 17):
        t, u = e.loc[s], o.loc[s]
        within = -0.1 < t.dratio_lo and t.dratio_hi < 0.1
        falls = u.dratio_hi < 0
        if within and falls:
            rd = "early separation retained alongside the conversion"
        elif t.dratio_hi < -0.1:
            rd = "early separation itself converted"
        else:
            rd = "unresolved"
        L.append(f"| {s} | {t.dratio:+.2f} [{t.dratio_lo:+.2f}, {t.dratio_hi:+.2f}] | {'yes' if within else 'no'} | "
                 f"{u.dratio:+.2f} [{u.dratio_lo:+.2f}, {u.dratio_hi:+.2f}] | {'yes' if falls else 'no'} | {rd} |")
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default="results/das_round2/round3/late_projection.npz")
    ap.add_argument("--bases", default="results/das_round2/bases_rank1.npz")
    ap.add_argument("--root", default="results/das_round2")
    ap.add_argument("--items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--prompts", default="data/das_round2/passive_test/prompts.csv")
    ap.add_argument("--out-dir", default="results/das_round2/round3")
    ap.add_argument("--report", default="reports/passive_das_prep/round3_late_projection.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
