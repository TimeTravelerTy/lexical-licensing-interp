#!/usr/bin/env python3
"""Round 4, E13 follow-up (descriptive, no model): the natural " by" margin deficit of rare verbs.

From the existing band-cross scores (`pythia14b_scores.csv`), `passive_1` sentences ("The N was V by the X."):
per pair and context the " by" margin good - bad; per band the mean of per-pair means; Head - XTail,
Tail - XTail and (Head + Tail) / 2 - XTail; the slope of the per-pair margin on the good participle's form
Zipf. Bootstrap: pairs within band x contexts (2,000 draws, seed 17). Curated and released contexts; all
126 band-cross pairs and the primary passive-test pairs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = ("head", "tail", "xtail")


def boot(w, band, zipf, n_boot, rng):
    """w: pairs x contexts margins (NaN = missing). Returns [B+1, 7]: head, tail, xtail, H-X, T-X, (H+T)/2-X, slope."""
    M, N = np.nan_to_num(w.to_numpy()), w.notna().to_numpy().astype(float)
    C = M.shape[1]
    out = []
    for b in range(n_boot + 1):
        cw = np.ones(C) if b == 0 else rng.multinomial(C, np.full(C, 1 / C))
        pm = (M * cw).sum(1) / np.maximum((N * cw).sum(1), 1e-9)
        ix = np.concatenate([np.flatnonzero(band == bd) if b == 0 else rng.choice(np.flatnonzero(band == bd), (band == bd).sum())
                             for bd in BANDS])
        r = {bd: pm[ix][band[ix] == bd].mean() for bd in BANDS}
        out.append([r["head"], r["tail"], r["xtail"], r["head"] - r["xtail"], r["tail"] - r["xtail"],
                    (r["head"] + r["tail"]) / 2 - r["xtail"], np.polyfit(zipf[ix], pm[ix], 1)[0]])
    return np.array(out)


def run(args):
    S = pd.read_csv(args.scores, low_memory=False)
    items = pd.read_csv(args.items)
    prim = set(items[items.bad_class == "plain"].pair_id)
    rng = np.random.default_rng(args.seed)
    names = ["Head", "Tail", "XTail", "Head − XTail", "Tail − XTail", "(Head + Tail)/2 − XTail", "slope on Zipf"]
    rows = []
    for cs in ("curated", "released"):
        s = S[(S.context_set == cs) & (S.paradigm == "passive_1")].assign(m=lambda x: x.good_by_lp - x.bad_by_lp)
        for subset in ("all", "primary"):
            ss = s if subset == "all" else s[s.verb_pair.isin(prim)]
            w = ss.pivot_table(index="verb_pair", columns="context_id", values="m")
            band = np.array(w.index.str.split("/").str[0])
            zipf = ss.groupby("verb_pair").good_form_zipf.mean().reindex(w.index).to_numpy()
            o = boot(w, band, zipf, args.n_boot, rng)
            n = {bd: int((band == bd).sum()) for bd in BANDS}
            for j, nm in enumerate(names):
                rows.append({"contexts": cs, "pairs": subset, "n_head": n["head"], "n_tail": n["tail"], "n_xtail": n["xtail"],
                             "quantity": nm, "est": o[0, j], "lo95": np.percentile(o[1:, j], 2.5),
                             "hi95": np.percentile(o[1:, j], 97.5)})
    res = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    f = lambda r: f"{r.est:+.3f} [{r.lo95:+.3f}, {r.hi95:+.3f}]"
    L = ["# Round 4, E13 follow-up. The natural \" by\" deficit of rare verbs", "",
         "Descriptive, no model (`analyze_by_deficit.py`, from `pythia14b_scores.csv`). `passive_1` \" by\" margin good − "
         "bad; per band the mean of per-pair means; 95% CIs: pairs within band × contexts (2,000 draws, seed 17). "
         "Slope: per-pair margin on the good participle's form Zipf.", "",
         "| contexts | pairs (Head / Tail / XTail) | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for (cs, sub), g in res.groupby(["contexts", "pairs"], sort=False):
        r0 = g.iloc[0]
        L.append(f"| {cs} | {sub} ({r0.n_head} / {r0.n_tail} / {r0.n_xtail}) | " +
                 " | ".join(f(g[g.quantity == nm].iloc[0]) for nm in names) + " |")
    L.append("")
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default="results/passive_band_cross/pythia14b_scores.csv")
    ap.add_argument("--items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--out", default="results/round4/oracle/by_deficit.csv")
    ap.add_argument("--report", default="reports/round4/e13_deficit.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
