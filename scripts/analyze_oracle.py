#!/usr/bin/env python3
"""Round 4, E13: oracle at d (reports/round4/plan.md, E13): analysis.

From `run_oracle.py`. Per sentence pair (pair x context x paradigm) and split: margins good - bad of the by
token (passive_1), the suffix, the verb and the whole sentence, natural and patched; Delta = patched - natural.
Per band and site; bootstrap over pairs (within band) x contexts (2,000 draws, seed 17).
Share closed (XTail, Head target) = Delta by margin / (natural Head by margin - natural XTail by margin); the
same for the whole margin (passive_2 and passive_1). Head control: Head items at the Head target (own pair
excluded). Reverse: Head items at the XTail target. Context draws are shared across bands and pair draws
across groups with the same pairs; inference is conditional on the targets. Decision rule: plan.md, E13.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PARTS = ("whole", "verb", "suffix", "first")


def margins(df):
    w = df.pivot_table(index=["paradigm", "context_id", "verb_pair", "band", "split", "target"], columns="side",
                       values=list(PARTS) + [f"nat_{p}" for p in PARTS])
    out = pd.DataFrame(index=w.index)
    for p in PARTS:
        out[p] = w[(p, "good")] - w[(p, "bad")]
        out[f"nat_{p}"] = w[(f"nat_{p}", "good")] - w[(f"nat_{p}", "bad")]
        out[f"d_{p}"] = out[p] - out[f"nat_{p}"]
    return out.reset_index()


def boot_stat(df, cols, wc, cix, wpair):
    """Mean of per-pair means with context weights wc [B+1, C] (shared across groups) and pair weights
    wpair {pair: [B+1]} (shared across groups with the same pairs). Returns [B+1, len(cols)]."""
    g = df.groupby(["verb_pair", "context_id"])[cols].mean().reset_index()
    pairs = sorted(g.verb_pair.unique())
    pix = {p: i for i, p in enumerate(pairs)}
    M = np.zeros((len(pairs), len(cix), len(cols)))
    N = np.zeros((len(pairs), len(cix)))
    M[g.verb_pair.map(pix), g.context_id.map(cix)] = g[cols].to_numpy()
    N[g.verb_pair.map(pix), g.context_id.map(cix)] = 1
    num = np.einsum("bc,pcr->bpr", wc, M * N[..., None])
    den = np.einsum("bc,pc->bp", wc, N)[..., None]
    pm = num / np.maximum(den, 1e-9)
    wp = np.stack([wpair[p] for p in pairs], 1)
    return np.einsum("bp,bpr->br", wp, pm) / wp.sum(1)[:, None]


def ci(x):
    return {"est": float(x[0]), "lo95": float(np.percentile(x[1:], 2.5)), "hi95": float(np.percentile(x[1:], 97.5))}


def run(args):
    d = Path(args.dir)
    sc = pd.read_parquet(d / "oracle_scores.parquet")
    rng = np.random.default_rng(args.seed)
    rows, dec = [], {}
    for site in sorted(sc.site.unique()):
        m = margins(sc[sc.site == site])
        for par in ("passive_1", "passive_2"):
            mp = m[m.paradigm == par]
            cols = [f"nat_{p}" for p in PARTS] + [f"d_{p}" for p in PARTS]
            ctxs = sorted(mp.context_id.unique())
            cix = {c: i for i, c in enumerate(ctxs)}
            wc = np.vstack([np.ones(len(ctxs)), rng.multinomial(len(ctxs), np.full(len(ctxs), 1 / len(ctxs)), args.n_boot)])
            wpair = {}
            for b in ("head", "tail", "xtail"):  # pairs resampled within band, shared by groups with the same pairs
                ps = sorted(mp[mp.band == b].verb_pair.unique())
                w = np.vstack([np.ones(len(ps)), rng.multinomial(len(ps), np.full(len(ps), 1 / len(ps)), args.n_boot)])
                wpair.update({p_: w[:, i] for i, p_ in enumerate(ps)})
            S = {}
            for key, sub in (("xtail@head", mp[(mp.band == "xtail") & (mp.target == "head")]),
                             ("tail@head", mp[(mp.band == "tail") & (mp.target == "head")]),
                             ("head@head", mp[(mp.band == "head") & (mp.target == "head")]),
                             ("head@xtail", mp[(mp.band == "head") & (mp.target == "xtail")])):
                S[key] = boot_stat(sub, cols, wc, cix, wpair)
                for j, c in enumerate(cols):
                    rows.append({"site": int(site), "paradigm": par, "group": key, "quantity": c, **ci(S[key][:, j])})
            # share of the Head - XTail deficit closed (same context draws for both bands)
            for p in ("first", "whole", "suffix"):
                jn, jd = cols.index(f"nat_{p}"), cols.index(f"d_{p}")
                deficit = S["head@head"][:, jn] - S["xtail@head"][:, jn]
                share = S["xtail@head"][:, jd] / deficit
                rows.append({"site": int(site), "paradigm": par, "group": "share closed", "quantity": p, **ci(share),
                             "deficit": float(deficit[0])})
                rows.append({"site": int(site), "paradigm": par, "group": "deficit", "quantity": p, **ci(deficit)})
        g = lambda par, grp, q: next(r for r in rows if r["site"] == site and r["paradigm"] == par and r["group"] == grp
                                     and r["quantity"] == q)
        sh, ctl = g("passive_1", "share closed", "first"), g("passive_1", "head@head", "d_first")
        dfc = g("passive_1", "deficit", "first")
        dec[int(site)] = {"by_share": sh, "head_control_d_by": ctl, "by_deficit": dfc,
                          "real_share": bool(sh["est"] >= 0.25 and sh["lo95"] > 0 and dfc["lo95"] > 0
                                             and ctl["lo95"] >= -0.1 and ctl["hi95"] <= 0.1)}
    res = pd.DataFrame(rows)
    res.to_csv(d / "oracle_summary.csv", index=False)
    any_real = any(v["real_share"] for v in dec.values())
    (d / "oracle_decisions.json").write_text(json.dumps({"sites": dec, "closes_real_share": any_real}, indent=2) + "\n")
    write_report(args, res, dec, any_real)


def write_report(args, res, dec, any_real):
    f = lambda r: f"{r['est']:+.3f} [{r['lo95']:+.3f}, {r['hi95']:+.3f}]"
    L = ["# Round 4, E13. Oracle at d", "",
         "Spec: `plan.md`, E13. Run: `run_oracle.py`; analysis: `analyze_oracle.py`. Curated `passive_1` (\"The N was V by "
         "the X.\") and `passive_2` (\"The N was V.\") sentences of the 64 primary pairs. The participle's coordinate on "
         "d_s is set to the Head class mean (true class, same context, own pair excluded); margins good − bad; Δ = "
         "patched − natural. by = the \" by\" token (passive_1). 95% CIs: pairs within band × contexts.", ""]
    for site in sorted(res.site.unique()):
        L += [f"## Site {site}", "", "| Paradigm | Group | natural by | Δ by | natural suffix | Δ suffix | natural verb | "
              "Δ verb | natural whole | Δ whole |", "|---|---|---|---|---|---|---|---|---|---|"]
        for par in ("passive_1", "passive_2"):
            for grp in ("xtail@head", "tail@head", "head@head", "head@xtail"):
                q = lambda c: res[(res.site == site) & (res.paradigm == par) & (res.group == grp) & (res.quantity == c)].iloc[0]
                L.append(f"| {par} | {grp} | {f(q('nat_first'))} | {f(q('d_first'))} | {f(q('nat_suffix'))} | "
                         f"{f(q('d_suffix'))} | {f(q('nat_verb'))} | {f(q('d_verb'))} | {f(q('nat_whole'))} | {f(q('d_whole'))} |")
        L.append("")
        for par in ("passive_1", "passive_2"):
            for p in ("first", "suffix", "whole"):
                q = res[(res.site == site) & (res.paradigm == par) & (res.group == "share closed") & (res.quantity == p)].iloc[0]
                L.append(f"- {par}, share of the Head − XTail {p if p != 'first' else 'by / first-suffix-token'} deficit "
                         f"({q.deficit:+.3f}) closed: {f(q)}")
        L.append("")
    L += ["## Declared decision", ""]
    for site, v in dec.items():
        L.append(f"- Site {site}: by-margin deficit {f(v['by_deficit'])}; share {f(v['by_share'])}; Head control Δ by "
                 f"{f(v['head_control_d_by'])} → "
                 f"**{'closes a real share' if v['real_share'] else 'does not close a real share'}**.")
    L += ["", f"**E13 closes a real share: {'yes → E14 runs' if any_real else 'no → E14 is not run; E13 is to be repeated at the translation stage'}**.", ""]
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/round4/oracle")
    ap.add_argument("--report", default="reports/round4/e13_oracle.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
