#!/usr/bin/env python3
"""Round 4, E13-T: oracle at the translation stage (reports/round4/plan.md, E13-T): analysis.

From `run_oracle_translation.py`. Margins good - bad (first suffix token: " by" in passive_1, "." in
passive_2; suffix; whole), natural and patched, Delta = patched - natural, per intervention (`mlp`,
`neurons`, `random1`..`random5`, their mean `random`, `wmatched`) and group (XTail / Tail at the Head target;
Head at the Head target = specificity control; Head at the XTail target = reverse control). Subsets: all 126
pairs (decisions), the primary pairs (as E13; comparison with E13 site 8, paired by sentence, E13's Delta
averaged over splits), and the pairs outside D11 (not primary passive-test pairs).
Bootstrap: pairs within band x contexts (2,000 draws, seed 17), the same draws for every intervention and
for E13 within a subset and paradigm, so differences are paired. Decision rules: plan.md, E13-T.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_oracle import PARTS, boot_stat, ci, margins

GROUPS = (("xtail@head", "xtail", "head"), ("tail@head", "tail", "head"), ("head@head", "head", "head"),
          ("head@xtail", "head", "xtail"))
SUBSETS = ("all", "primary", "outside D11")
SHOW = ("mlp", "neurons", "random", "wmatched")
FIRST = {"passive_1": '" by"', "passive_2": '"."'}


def draws(mp, rng, n_boot):
    ctxs = sorted(mp.context_id.unique())
    cix = {c: i for i, c in enumerate(ctxs)}
    wc = np.vstack([np.ones(len(ctxs)), rng.multinomial(len(ctxs), np.full(len(ctxs), 1 / len(ctxs)), n_boot)])
    wpair = {}
    for b in ("head", "tail", "xtail"):
        ps = sorted(mp[mp.band == b].verb_pair.unique())
        w = np.vstack([np.ones(len(ps)), rng.multinomial(len(ps), np.full(len(ps), 1 / len(ps)), n_boot)])
        wpair.update({p: w[:, i] for i, p in enumerate(ps)})
    return wc, cix, wpair


def load_e13(path, sc, prim):
    if not Path(path).exists():
        raise SystemExit(f"E13 scores not found: {path} (the E13 comparison is declared)")
    e = pd.read_parquet(path)
    e = e[e.site == 8].drop(columns="site")
    assert set(e.split) == {0, 1, 2}, "E13: expected splits 0-2"
    key = ["paradigm", "context_id", "verb_pair", "side"]
    a = e[(e.band == "xtail") & (e.target == "head")]
    b = sc[(sc.cond == "mlp") & (sc.band == "xtail") & (sc.target == "head") & sc.verb_pair.isin(prim)]
    ka, kb = set(map(tuple, a[key].drop_duplicates().to_numpy())), set(map(tuple, b[key].to_numpy()))
    assert ka == kb, f"E13 and E13-T primary XTail sentences differ ({len(ka ^ kb)} keys)"
    assert (a.groupby(key).split.nunique() == 3).all(), "E13: a sentence lacks a split"
    return margins(e)


def run(args):
    d = Path(args.dir)
    sc = pd.read_parquet(d / "oracle_t_scores.parquet")
    items = pd.read_csv(args.items)
    prim = set(items[items.bad_class == "plain"].pair_id)
    conds = list(dict.fromkeys(sc.cond))
    rands = [c for c in conds if c.startswith("random")]
    M = {c: margins(sc[sc.cond == c].drop(columns="cond")) for c in conds}
    e13 = load_e13(args.e13, sc, prim)
    cols = [f"nat_{p}" for p in PARTS] + [f"d_{p}" for p in PARTS]
    jn, jd = cols.index("nat_first"), cols.index("d_first")
    rng = np.random.default_rng(args.seed)
    rows = []

    def add(subset, par, cond, group, quantity, x, **kw):
        rows.append({"subset": subset, "paradigm": par, "cond": cond, "group": group, "quantity": quantity, **ci(x), **kw})

    for subset in SUBSETS:
        keep = {"all": lambda m: m, "primary": lambda m: m[m.verb_pair.isin(prim)],
                "outside D11": lambda m: m[~m.verb_pair.isin(prim)]}[subset]
        for par in ("passive_1", "passive_2"):
            sel = lambda m: keep(m[m.paradigm == par])
            wc, cix, wpair = draws(sel(M["mlp"]), rng, args.n_boot)
            S = {}
            for cond in conds:
                mp = sel(M[cond])
                for key, band, tgt in GROUPS:
                    S[(cond, key)] = boot_stat(mp[(mp.band == band) & (mp.target == tgt)], cols, wc, cix, wpair)
            for key, _, _ in GROUPS:  # the random sets' mean, per draw
                S[("random", key)] = np.mean([S[(c, key)] for c in rands], 0)
            for cond in conds + ["random"]:
                for key, _, _ in GROUPS:
                    for j, c in enumerate(cols):
                        add(subset, par, cond, key, c, S[(cond, key)][:, j])
                for p in ("first", "whole", "suffix"):
                    a, b = cols.index(f"nat_{p}"), cols.index(f"d_{p}")
                    deficit = S[(cond, "head@head")][:, a] - S[(cond, "xtail@head")][:, a]
                    add(subset, par, cond, "share closed", p, S[(cond, "xtail@head")][:, b] / deficit,
                        deficit=float(deficit[0]))
                    add(subset, par, cond, "deficit", p, deficit)
                    add(subset, par, cond, "patched gap", p, deficit - S[(cond, "xtail@head")][:, b])
            for other in ("random", "wmatched"):
                add(subset, par, f"neurons − {other}", "xtail@head", "d_first",
                    S[("neurons", "xtail@head")][:, jd] - S[(other, "xtail@head")][:, jd])
            if subset == "primary":
                me = sel(e13)
                se = boot_stat(me[(me.band == "xtail") & (me.target == "head")], cols, wc, cix, wpair)
                add(subset, par, "E13 site 8", "xtail@head", "d_first", se[:, jd])
                for cond in ("mlp", "neurons"):
                    add(subset, par, f"{cond} − E13 site 8", "xtail@head", "d_first", S[(cond, "xtail@head")][:, jd] - se[:, jd])
    res = pd.DataFrame(rows)
    res.to_csv(d / "oracle_t_summary.csv", index=False)

    def g(**kw):
        r = res[np.logical_and.reduce([res[k] == v for k, v in kw.items()])].iloc[0]
        return {k: v for k, v in r.items() if not (isinstance(v, float) and np.isnan(v))}

    dec = {}
    for cond in ("mlp", "neurons", "random", "wmatched"):
        q = dict(subset="all", paradigm="passive_1", cond=cond)
        xd, ctl = g(**q, group="xtail@head", quantity="d_first"), g(**q, group="head@head", quantity="d_first")
        sh, dfc = g(**q, group="share closed", quantity="first"), g(**q, group="deficit", quantity="first")
        thr = 0.25 * max(dfc["est"], 0.0)
        ctl_ok = ctl["lo95"] >= -0.1 and ctl["hi95"] <= 0.1
        raises = bool(xd["lo95"] > 0 and xd["est"] >= thr and ctl_ok)
        v = {"xtail_d_by": xd, "head_control_d_by": ctl, "by_share": sh, "by_deficit": dfc, "threshold": thr,
             "patched_gap": g(**q, group="patched gap", quantity="first"),
             "real_share": bool(sh["est"] >= 0.25 and sh["lo95"] > 0 and dfc["lo95"] > 0 and ctl_ok)}
        if cond == "neurons":
            for other in ("random", "wmatched"):
                v[f"minus_{other}"] = g(subset="all", paradigm="passive_1", cond=f"neurons − {other}",
                                        group="xtail@head", quantity="d_first")
            raises = raises and v["minus_random"]["lo95"] > 0
        v["raises"] = bool(raises)
        dec[cond] = v
    (d / "oracle_t_decisions.json").write_text(json.dumps(dec, indent=2) + "\n")
    write_report(args, res, dec)


def write_report(args, res, dec):
    f = lambda r: f"{r['est']:+.3f} [{r['lo95']:+.3f}, {r['hi95']:+.3f}]"
    q = lambda **kw: res[np.logical_and.reduce([res[k] == v for k, v in kw.items()])].iloc[0]
    nd = pd.read_csv(args.neurons)
    wb = nd.assign(a=nd.w_by.abs()).groupby("set").a.mean()
    L = ["# Round 4, E13-T. Oracle at the translation stage", "",
         "Spec: `plan.md`, E13-T. Run: `run_oracle_translation.py` (fp32); analysis: `analyze_oracle_translation.py`. "
         "Curated `passive_1` and `passive_2` sentences of all 126 band-cross pairs. At the participle's last token, set "
         "to the Head mean of the true class (same context, own pair excluded): `mlp` = MLP 11–17 outputs; `neurons` = "
         "D11's top-50 switch neurons; `random` = mean over 5 random sets of 50 (same layer split); `wmatched` = 50 "
         "neurons D11 did not select, matched one-to-one on |w_by|. Mean |w_by|: " +
         ", ".join(f"{k} {v:.3f}" for k, v in wb.items()) + ". Margins good − bad; Δ = patched − natural; first = the "
         "first suffix token (\" by\" in passive_1, \".\" in passive_2). 95% CIs: pairs within band × contexts, the "
         "same draws for every intervention.", ""]
    for subset in SUBSETS:
        L += [f"## {dict(zip(SUBSETS, ('All 126 pairs', 'Primary pairs (as E13)', 'Pairs outside D11')))[subset]}", ""]
        for par in ("passive_1", "passive_2"):
            for part, lab in (("first", f"first ({FIRST[par]})"), ("whole", "whole")):
                L += [f"### {par}: Δ {lab} margin", "",
                      "| Group | natural | " + " | ".join(SHOW) + " |", "|---|---|" + "---|" * len(SHOW)]
                for key, _, _ in GROUPS:
                    r = lambda c, x: q(subset=subset, paradigm=par, cond=c, group=key, quantity=x)
                    L.append(f"| {key} | {f(r('mlp', f'nat_{part}'))} | " +
                             " | ".join(f(r(c, f'd_{part}')) for c in SHOW) + " |")
                L.append("")
            for c in SHOW:
                s = q(subset=subset, paradigm=par, cond=c, group="share closed", quantity="first")
                pg = q(subset=subset, paradigm=par, cond=c, group="patched gap", quantity="first")
                L.append(f"- {c}: share of the Head − XTail {FIRST[par]} deficit ({s.deficit:+.3f}) closed {f(s)}; "
                         f"patched gap {f(pg)}")
            for other in ("random", "wmatched"):
                r = q(subset=subset, paradigm=par, cond=f"neurons − {other}", group="xtail@head", quantity="d_first")
                L.append(f"- XTail Δ {FIRST[par]}, neurons − {other}: {f(r)}")
            if subset == "primary":
                e = q(subset=subset, paradigm=par, cond="E13 site 8", group="xtail@head", quantity="d_first")
                L.append(f"- E13 site 8, XTail Δ {FIRST[par]} (same sentences and draws): {f(e)}; mlp − E13 "
                         f"{f(q(subset=subset, paradigm=par, cond='mlp − E13 site 8', group='xtail@head', quantity='d_first'))}; "
                         f"neurons − E13 {f(q(subset=subset, paradigm=par, cond='neurons − E13 site 8', group='xtail@head', quantity='d_first'))}")
            L.append("")
    L += ["## Declared decisions (all 126 pairs, passive_1, \" by\")", ""]
    for c, v in dec.items():
        extra = "".join(f"; neurons − {o} {f(v[f'minus_{o}'])}" for o in ("random", "wmatched") if f"minus_{o}" in v)
        decl = "" if c in ("mlp", "neurons") else " (control; reported)"
        L.append(f"- **{c}**{decl}: XTail Δ by {f(v['xtail_d_by'])} (threshold {v['threshold']:.3f}); Head control "
                 f"{f(v['head_control_d_by'])}{extra} → raises rare verbs' by margin: **{'yes' if v['raises'] else 'no'}**. "
                 f"Closes a real share (E13 rule; by deficit {f(v['by_deficit'])}, share {f(v['by_share'])}): "
                 f"**{'yes' if v['real_share'] else 'no'}**. Patched Head − XTail gap {f(v['patched_gap'])}.")
    L.append("")
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/round4/oracle_translation")
    ap.add_argument("--items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--neurons", default="data/round4/oracle_translation/neurons.csv")
    ap.add_argument("--e13", default="results/round4/oracle/oracle_scores.parquet")
    ap.add_argument("--report", default="reports/round4/e13t_oracle_translation.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
