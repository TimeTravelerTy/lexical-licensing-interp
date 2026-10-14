#!/usr/bin/env python3
"""Round 5, B5 analysis (reports/round5/plan.md, B5). No model.

Inputs: results/round5/necessity/{necessity_natural.parquet, necessity_rows.parquet, necessity_meta.json};
results/round4/translation/conjunction.npz (D11 pairs, for the half-B subset).
Share removed = mean(by_nat,good - by_cond) / mean(by_nat,good - by_nat,bad) over the B7 items (ratio of population
means). Item weights = pair weight (pairs within band) x context weight (contexts within context band); 2,000
draws, seed 17; the same draws for every condition and frame, so differences are paired.
Outputs: reports/round5/b5_necessity.md, results/round5/necessity/b5_summary.csv, b5_decisions.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_conjunction import pair_halves

READ = ("by", "O", "dot", "prep_noby")
CONDS = ("d_I", "n10_I", "n50_I", "c10", "c50", "w50", "r10", "r50", "d_T", "n50_T")


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


def main(args):
    rdir = Path(args.res_dir)
    nat = pd.read_parquet(rdir / "necessity_natural.parquet")
    R = pd.read_parquet(rdir / "necessity_rows.parquet")
    for k in READ:  # random sets: mean over the five
        for s in ("r10", "r50"):
            R[f"{s}_{k}"] = R[[f"{s}_{i}_{k}" for i in range(1, 6)]].mean(1)
    z = np.load(args.d11, allow_pickle=True)
    halfB = set(np.array(z["pairs"])[pair_halves(np.array(z["bands"]), args.seed) == 1])
    lines = ["# Round 5, B5: necessity on good passives", "",
             "Spec: `plan.md`, B5. Run: `run_necessity.py`; analysis: `analyze_necessity.py`. Natural good passives of the "
             "B7 items; share removed = mean(natural − patched) / mean(natural good − bad) of log P(\" by\"). 95% CIs: pairs "
             "within band × contexts within context band (shared draws).", ""]
    meta = json.loads((rdir / "necessity_meta.json").read_text())
    lines += [f"Checks: {json.dumps(meta['checks'])}; minimum target pairs {meta['min_target_pairs']}.", ""]
    rows, dec = [], {}
    rng = np.random.default_rng(args.seed)
    for frame in ("hasbeen", "was"):
        nf = nat[(nat.frame == frame) & (nat.b7 == 1)].set_index(["item_id", "side"])
        r = R[R.frame == frame].reset_index(drop=True)
        good = nf.xs("good", level="side").loc[r.item_id]
        bad = nf.xs("bad", level="side").loc[r.item_id]
        pairs = good.pair_id.to_numpy()
        ctxs = good.context_id.to_numpy()
        uc = np.unique(ctxs)
        cband = pd.Series([c.split("/")[2] for c in ctxs], index=ctxs).groupby(level=0).first().loc[uc].to_numpy()
        wc = weights(np.random.default_rng(args.seed + 1), cband, args.n_boot)  # shared by frames, conditions, pops
        cix = pd.Index(uc).get_indexer(ctxs)

        for pop, mask in (("all 64 pairs", np.ones(len(r), bool)), ("D11 half-B pairs", np.isin(pairs, list(halfB)))):
            up = np.unique(pairs[mask])
            pband = pd.Series(good.band.to_numpy()[mask], index=pairs[mask]).groupby(level=0).first().loc[up].to_numpy()
            wv = weights(np.random.default_rng(args.seed), pband, args.n_boot)  # pairs within band, this population
            pi = pd.Index(up).get_indexer(pairs[mask])
            # indicator [item, pair] to average within pair (context-weighted) before averaging pairs
            Ind = np.zeros((int(mask.sum()), len(up)))
            Ind[np.arange(int(mask.sum())), pi] = 1.0
            wcm = wc[:, cix[mask]]  # [draw, item]
            den = wcm @ Ind  # [draw, pair]: summed context weights per pair

            def mean(v):
                num = (wcm * v[mask][None]) @ Ind  # [draw, pair]
                pmean = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
                ww = wv * (den > 0)
                return (ww * pmean).sum(1) / ww.sum(1)
            gap = {k: mean(good[k].to_numpy() - bad[k].to_numpy()) for k in READ}
            lines += [f"## Frame `{frame}`, {pop} ({len(np.unique(pairs[mask]))} pairs, {int(mask.sum())} items)", "",
                      f"Natural good − bad: \" by\" {fmt(gap['by'])}; O {fmt(gap['O'])}; \".\" {fmt(gap['dot'])}; "
                      f"PREP\\by {fmt(gap['prep_noby'])}.", ""]
            gate = ci(gap["by"])[0] > 0 and gap["by"][0] >= 0.2
            lines += [f"Gate (natural \" by\" gap CI above 0 and ≥ 0.2): {'passes' if gate else 'fails'}.", "",
                      "| Condition | Δ by (nats) | share removed | Δ O | Δ \".\" | Δ PREP\\by |", "|---|---|---|---|---|---|"]
            share, dby = {}, {}
            for c in CONDS:
                delta = {k: mean(r[f"{c}_{k}"].to_numpy() - good[k].to_numpy()) for k in READ}
                dby[c] = delta["by"]
                share[c] = -delta["by"] / gap["by"]
                lines.append(f"| {c} | {fmt(delta['by'])} | {fmt(share[c])} | {fmt(delta['O'])} | {fmt(delta['dot'])} | "
                             f"{fmt(delta['prep_noby'])} |")
                rows.append({"frame": frame, "pop": pop, "cond": c, "dby": delta["by"][0], "dby_lo": ci(delta["by"])[0],
                             "dby_hi": ci(delta["by"])[1], "share": share[c][0], "share_lo": ci(share[c])[0],
                             "share_hi": ci(share[c])[1]})
            # equalized coordinate
            eqw = mean(r.eq_good_wby50.to_numpy() - r.eq_bad_wby50.to_numpy())
            natw = mean(good.wby50.to_numpy() - bad.wby50.to_numpy())
            eqby = mean(r.eq_good_by.to_numpy() - r.eq_bad_by.to_numpy())
            remw, remby = eqw / natw, eqby / gap["by"]
            diffs = {"n50_I − d_I": share["n50_I"] - share["d_I"], "n50_I − r50": share["n50_I"] - share["r50"],
                     "n50_I − w50": share["n50_I"] - share["w50"], "n10_I − r10": share["n10_I"] - share["r10"],
                     "c50 − n50_I": share["c50"] - share["n50_I"], "c10 − n10_I": share["c10"] - share["n10_I"],
                     "interaction c50 − d_I − n50_I": share["c50"] - share["d_I"] - share["n50_I"]}
            lines += ["", "| Paired difference (shares) | estimate |", "|---|---|"] + \
                     [f"| {k} | {fmt(v)} |" for k, v in diffs.items()]
            lines += ["", f"Target contrasts (nats): R(d_T) − R(d_I) {fmt(dby['d_T'] - dby['d_I'])}; "
                          f"R(n50_T) − R(n50_I) {fmt(dby['n50_T'] - dby['n50_I'])}.",
                      f"Equalized coordinate (good and bad at (t_T + t_I)/2): top-50 good − bad (w_by units) natural "
                      f"{fmt(natw)}, equalized {fmt(eqw)}, remaining fraction {fmt(remw)}; \" by\" gap equalized {fmt(eqby)} "
                      f"(fraction {fmt(remby)}).", ""]
            U = (lambda v: v if gate else "unresolved (gate fails)")
            d = {"gate": bool(gate),
                 "d_partly_necessary": U(bool(ci(share["d_I"])[0] > 0 and share["d_I"][0] >= 0.25)),
                 "neurons_remove_more_than_d": U(bool(ci(diffs["n50_I − d_I"])[0] > 0)),
                 "class_info_beyond_d": bool(ci(eqw)[0] > 0 and remw[0] >= 0.25),
                 "specificity": U(bool(ci(diffs["n50_I − r50"])[0] > 0 and ci(diffs["n50_I − w50"])[0] > 0
                                       and ci(diffs["n10_I − r10"])[0] > 0)),
                 "controls_within_0.1": {c: bool(-0.1 < ci(dby[c])[0] and ci(dby[c])[1] < 0.1) for c in ("d_T", "n50_T")}}
            dc = diffs["c50 − n50_I"]
            lo90, hi90 = ci(dc, 5, 95)
            d["d_adds_beyond_neurons"] = U("adds" if ci(dc)[0] > 0 and dc[0] >= 0.1 else
                                            "nothing beyond" if -0.1 < lo90 and hi90 < 0.1 else "unresolved")
            dec[f"{frame}|{pop}"] = d
            lines += ["**Declared decisions" + (" (primary)" if frame == "hasbeen" and pop == "all 64 pairs" else "") + ":** "
                      + "; ".join(f"{k}: {v}" for k, v in d.items()), ""]
    Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.report_dir) / "b5_necessity.md").write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(rdir / "b5_summary.csv", index=False)
    (rdir / "b5_decisions.json").write_text(json.dumps(dec, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-dir", default="results/round5/necessity")
    ap.add_argument("--d11", default="results/round4/translation/conjunction.npz")
    ap.add_argument("--report-dir", default="reports/round5")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    main(ap.parse_args())
