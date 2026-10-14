#!/usr/bin/env python3
"""Round 5, A5 analysis: does a grammaticality probe's accuracy degrade with verb rarity like LP?
(reports/round5/plan.md, A5). No model.

Inputs: results/round5/probe/{probe_train.csv, probe_meta.json, probe_freqblimp.parquet, probe_curated.parquet}.
Primary: released FreqBLiMP passive_1; D_m = accuracy_Head - accuracy_XTail (m = probe, LP), paired on items; items
resampled within band (2,000 draws, seed 17); sensitivity: good-participle clusters resampled within band.
Outputs: reports/round5/a5_probe.md, results/round5/probe/a5_summary.csv, a5_decisions.json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = ("head", "tail", "xtail")


def ci(d, lo=2.5, hi=97.5):
    return np.percentile(d[1:], lo), np.percentile(d[1:], hi)


def fmt(d, p=1):
    lo, hi = ci(d)
    return f"{d[0]:+.{p}f} [{lo:+.{p}f}, {hi:+.{p}f}]"


def participle(sentence, paradigm):
    w = re.findall(r"[A-Za-z'-]+", sentence)
    if paradigm == "passive_1" and "by" in w:
        return w[w.index("by") - 1].lower()
    return w[-1].lower() if w else ""


def boot_acc(correct, band, cluster, rng, n_boot):
    """Per draw, accuracy (%) per band; units = clusters resampled within band (cluster = item for item bootstrap)."""
    out = {b: np.empty(n_boot + 1) for b in BANDS}
    for b in BANDS:
        m = band == b
        cl = cluster[m]
        u, inv = np.unique(cl, return_inverse=True)
        s = np.bincount(inv, weights=correct[m].astype(float))
        n = np.bincount(inv)
        out[b][0] = 100 * s.sum() / n.sum()
        for i in range(n_boot):
            pick = rng.integers(0, len(u), len(u))
            out[b][i + 1] = 100 * s[pick].sum() / n[pick].sum()
    return out


def main(args):
    rdir = Path(args.res_dir)
    meta = json.loads((rdir / "probe_meta.json").read_text())
    tr = pd.read_csv(rdir / "probe_train.csv")
    F = pd.read_parquet(rdir / "probe_freqblimp.parquet")
    lines = ["# Round 5, A5: grammaticality probe vs LP", "",
             "Spec: `plan.md`, A5. Run: `run_probe.py`; analysis: `analyze_probe.py`. Probe: L2 logistic regression on the "
             "last token's residual state (Wang et al. 2026 adaptation); a pair is correct if the good sentence has the "
             "higher probe logit (LP: higher log-prob).", ""]
    best = tr[tr.split == 0].groupby("layer").dev_auc.max()
    lines += [f"Selected layer {meta['layer']} (dev AUC {meta['dev_auc_best']:.3f}); C per split {meta['C']}. Dev AUC by "
              "layer (split 0): " + ", ".join(f"{l}: {a:.3f}" for l, a in best.items()) + ".",
              f"Baselines (dev AUC): length/punctuation {meta['baselines_dev_auc']['length_punct']:.3f}; LM log-prob "
              f"{meta['baselines_dev_auc']['lm_logprob']:.3f}; per token {meta['baselines_dev_auc']['lm_logprob_per_token']:.3f}.", ""]
    rng = np.random.default_rng(args.seed)
    rows, dec = [], {}
    for par in ("passive_1", "passive_2"):
        d = F[F.uid == par].reset_index(drop=True)
        band = d.band.to_numpy()
        item = np.arange(len(d))
        verb = np.array([participle(s, par) for s in d.good])
        cp = {k: (d[f"probe{k}_good"] > d[f"probe{k}_bad"]).to_numpy() for k in range(3)}
        cl = (d.lp_good > d.lp_bad).to_numpy()
        lines += [f"## Released `{par}`", "", "| Method | Head | Tail | XTail | D = Head − XTail | Head − Tail |", "|---|---|---|---|---|---|"]
        res = {}
        for name, corr in (("probe (split 0)", cp[0]), ("probe (split 1)", cp[1]), ("probe (split 2)", cp[2]), ("LP", cl)):
            r = np.random.default_rng(args.seed)  # the same item draws for every method (paired)
            acc = boot_acc(corr, band, item, r, args.n_boot)
            res[name] = acc
            lines.append(f"| {name} | " + " | ".join(fmt(acc[b]) for b in BANDS) +
                         f" | {fmt(acc['head'] - acc['xtail'])} | {fmt(acc['head'] - acc['tail'])} |")
        Dp = res["probe (split 0)"]["head"] - res["probe (split 0)"]["xtail"]
        Dl = res["LP"]["head"] - res["LP"]["xtail"]
        rv = np.random.default_rng(args.seed)
        accv_p = boot_acc(cp[0], band, verb, rv, args.n_boot)
        rv = np.random.default_rng(args.seed)
        accv_l = boot_acc(cl, band, verb, rv, args.n_boot)
        Dpv, Dlv = accv_p["head"] - accv_p["xtail"], accv_l["head"] - accv_l["xtail"]
        margin = (d.probe0_good - d.probe0_bad)
        lines += ["", f"Paired: D_probe − 0.5 D_LP {fmt(Dp - 0.5 * Dl)}; D_LP − D_probe {fmt(Dl - Dp)}. Verb-cluster "
                      f"sensitivity: D_probe {fmt(Dpv)}, D_LP {fmt(Dlv)}, D_LP − D_probe {fmt(Dlv - Dpv)}.",
                  "Mean probe-logit margin by band: " + ", ".join(f"{b} {margin[band == b].mean():+.2f}" for b in BANDS) + ".", ""]
        lo90, hi90 = ci(Dp, 5, 95)
        dd = {"detectable_decline": bool(ci(Dp)[0] > 0), "declines_like_LP": bool(ci(Dp - 0.5 * Dl)[0] > 0),
              "smaller_than_LP": bool(ci(Dl - Dp)[0] > 0), "practically_no_decline": bool(-4 < lo90 and hi90 < 4),
              "D_probe": [float(Dp[0]), *map(float, ci(Dp))], "D_LP": [float(Dl[0]), *map(float, ci(Dl))]}
        hd = res["probe (split 0)"]["head"]
        dd["informative"] = bool(meta["dev_auc_best"] >= 0.85 and
                                 meta["dev_auc_best"] - meta["baselines_dev_auc"]["length_punct"] >= 0.1 and ci(hd)[0] > 50)
        dec[par] = dd
        for name, acc in res.items():
            for b in BANDS:
                rows.append({"paradigm": par, "method": name, "band": b, "acc": acc[b][0], "lo": ci(acc[b])[0], "hi": ci(acc[b])[1]})
    # descriptive: all paradigms, Head - XTail drop, probe vs LP
    g = F.assign(cp=F.probe0_good > F.probe0_bad, cl=F.lp_good > F.lp_bad).groupby(["uid", "band"])[["cp", "cl"]].mean().unstack()
    drop = pd.DataFrame({"probe_head": 100 * g[("cp", "head")], "probe_drop": 100 * (g[("cp", "head")] - g[("cp", "xtail")]),
                         "lp_head": 100 * g[("cl", "head")], "lp_drop": 100 * (g[("cl", "head")] - g[("cl", "xtail")])})
    drop.to_csv(rdir / "a5_paradigm_drops.csv")
    lines += ["## All 67 paradigms (descriptive)", "",
              f"Mean Head accuracy: probe {drop.probe_head.mean():.1f}%, LP {drop.lp_head.mean():.1f}%. Mean Head − XTail drop: "
              f"probe {drop.probe_drop.mean():+.1f} pp, LP {drop.lp_drop.mean():+.1f} pp; correlation of the drops across "
              f"paradigms {np.corrcoef(drop.probe_drop, drop.lp_drop)[0, 1]:+.2f}.", ""]
    Cc = pd.read_parquet(rdir / "probe_curated.parquet")
    c1 = Cc[Cc.paradigm == "passive_1"]
    acc = c1.assign(cp=c1.probe0_good > c1.probe0_bad, cl=c1.lp_good > c1.lp_bad).groupby("verb_band")[["cp", "cl"]].mean() * 100
    lines += ["Curated band-cross `passive_1` (descriptive): " + "; ".join(
        f"{b}: probe {acc.loc[b, 'cp']:.1f}%, LP {acc.loc[b, 'cl']:.1f}%" for b in BANDS) + ".", ""]
    p1 = dec["passive_1"]
    lines += ["## Declared decisions (released passive_1)", "",
              f"- Probe informative: **{p1['informative']}**.",
              f"- Detectable decline: **{p1['detectable_decline']}**; declines like LP: **{p1['declines_like_LP']}**; smaller "
              f"decline than LP: **{p1['smaller_than_LP']}**; practically no decline: **{p1['practically_no_decline']}**.", ""]
    Path(args.report).write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(rdir / "a5_summary.csv", index=False)
    (rdir / "a5_decisions.json").write_text(json.dumps(dec, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-dir", default="results/round5/probe")
    ap.add_argument("--report", default="reports/round5/a5_probe.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    main(ap.parse_args())
