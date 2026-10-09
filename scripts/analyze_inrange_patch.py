#!/usr/bin/env python3
"""Round 4, A2: in-range patch analysis (reports/round4/plan.md, A2).

Per construction (passive, OR, TC) and site, on bad items of the 64 primary pairs:
- D_in = R(t_T) - R(t_I) (mean over the 3 splits per item), and R(t_T) - R(unpatched),
  R(t_I) - R(unpatched);
- the natural good - bad gap of the same readout and items;
- two-way bootstrap (pairs within band, contexts; 2,000 draws, seed 17), the same draws for D_in
  and the natural gap, so the ratio of population means D_in / natural is formed per draw.
Decision rules (sites 6 and 8): reproduces the profile; excess removed / remains (passive, TC).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_passive_test import Boot, cells, ci, pair_values, summarize

SITES = (4, 6, 8, 10, 12, 14, 16, 17)
DECL = (6, 8)
BASE = ("O", "by", "dot", "PREP", "prep_noby", "MAIN", "END")
SIGN = {"passive": ("by", "dot", "O"), "OR": ("MAIN", "PREP", "O", "L"), "TC": ("END", "PREP", "O", "L")}
RATIO = {"passive": ("by",), "OR": ("MAIN", "PREP"), "TC": ("END", "PREP")}
NUM = {"OR": "MAIN", "TC": "END"}


def with_L(df, cons):
    df = df.copy()
    df["L"] = df[NUM[cons]] - df["PREP"] if cons in NUM else 0.0
    return df


def run(args):
    rdir = Path(args.out_dir)
    nat = pd.read_parquet(rdir / "natural.parquet").set_index("prompt")
    R = pd.read_parquet(rdir / "rows.parquet")
    items = pd.read_csv(rdir / "items.csv")
    rows = []
    for cons in ("passive", "OR", "TC"):
        it = items[items.construction == cons].reset_index(drop=True)
        cols = list(BASE) + ["L"]
        good = with_L(nat.loc[it.good_prompt, list(BASE)].reset_index(drop=True), cons)[cols].to_numpy()
        bad = with_L(nat.loc[it.bad_prompt, list(BASE)].reset_index(drop=True), cons)[cols].to_numpy()
        pr = it.drop_duplicates("pair_id").set_index("pair_id")
        pair_ix = {p: i for i, p in enumerate(pr.index)}
        ctx_ix = {c: i for i, c in enumerate(sorted(it.context_id.unique()))}
        bt = Boot(len(ctx_ix), args.n_boot, args.seed)
        wp = bt.pair_weights(pr.band.to_numpy())
        pix = np.arange(len(pr))
        one = np.ones((bt.B + 1, 1), np.float32)

        def stat(vals):
            df = it[["pair_id", "context_id"]].assign(dq=0)
            df[cols] = vals
            v, m = cells(df, pair_ix, ctx_ix, {0: 0}, 1, cols)
            return summarize(pair_values(v, m, bt.wc, one), wp, pix)

        S_nat = stat(good - bad)
        for site in SITES:
            pat = pd.read_parquet(rdir / f"patches_site{site}.parquet")
            assert (pat.row.to_numpy() == R.row.to_numpy()).all()
            sel = (R.construction == cons).to_numpy()
            x = R[sel][["item_id", "split", "target"]].copy()
            pv = with_L(pat.loc[sel, list(BASE)].reset_index(drop=True), cons)[cols]
            x[cols] = pv.to_numpy()
            m = x.groupby(["item_id", "target"])[cols].mean()  # mean over the 3 splits
            mT = m.xs("T", level="target").loc[it.item_id].to_numpy()
            mI = m.xs("I", level="target").loc[it.item_id].to_numpy()
            S = {"D_in": stat(mT - mI), "T_vs_unpatched": stat(mT - bad), "I_vs_unpatched": stat(mI - bad)}
            for j, rd in enumerate(cols):
                if cons == "passive" and rd == "L":
                    continue
                row = {"construction": cons, "site": site, "readout": rd,
                       **{f"nat_{k}": v for k, v in ci(S_nat[:, j]).items()}}
                for q, s in S.items():
                    row.update({f"{q}_{k}": v for k, v in ci(s[:, j]).items()})
                with np.errstate(divide="ignore", invalid="ignore"):
                    row.update({f"ratio_{k}": v for k, v in ci(S["D_in"][:, j] / S_nat[:, j]).items()})
                if rd == "O":
                    row.update({f"excess_{k}": v for k, v in ci(S["D_in"][:, j] - S_nat[:, j]).items()})
                rows.append(row)
            print(f"{cons} site {site} done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(rdir / "inrange_summary.csv", index=False)
    write_report(args, res)


def decide(res, cons, site, excess_prior):
    g = res[(res.construction == cons) & (res.site == site)].set_index("readout")
    notes, ok = [], True
    for rd in SIGN[cons]:
        r = g.loc[rd]
        if not ((r.nat_lo95 > 0 or r.nat_hi95 < 0) and abs(r.nat_est) >= 0.1):
            notes.append(f"{rd}: natural gap not decisive (skipped)")
            continue
        match = r.D_in_lo95 > 0 if r.nat_est > 0 else r.D_in_hi95 < 0
        ok &= bool(match)
        notes.append(f"{rd}: sign {'match' if match else 'FAIL'}")
    unresolved = False
    for rd in RATIO[cons]:
        r = g.loc[rd]
        if abs(r.nat_est) < 0.2:
            unresolved = True
            notes.append(f"{rd}: natural gap < 0.2 (unresolved)")
            continue
        inr = 0.5 <= r.ratio_est <= 1.5
        ok &= bool(inr)
        notes.append(f"{rd}: ratio {r.ratio_est:.2f} {'in' if inr else 'OUT of'} [0.5, 1.5]")
    verdict = "unresolved" if unresolved else ("reproduces the profile" if ok else "does not reproduce the profile")
    ex = None
    if cons in excess_prior:
        r = g.loc["O"]
        ex = "excess removed" if r.excess_hi95 <= 0.2 else "excess remains" if r.excess_lo95 > 0.2 else "unresolved"
    return verdict, notes, ex


def write_report(args, res):
    f = lambda e, lo, hi: f"{e:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    L = ["# Round 4, A2. In-range patch", "",
         "Spec: `plan.md`, A2. Run: `run_inrange_patch.py`; analysis: `analyze_inrange_patch.py`. Bad items of the 64 "
         "primary pairs; coordinate on the item's cross-fitted d_s set to the construction's transitive (t_T) or "
         "intransitive (t_I) class mean in the same context (leave own pair out, DAS training pairs excluded). "
         "D_in = R(t_T) − R(t_I), nats. 95% CIs: pairs within band × contexts; ratio = D_in / natural per draw.", ""]
    for cons in ("passive", "OR", "TC"):
        L += [f"## {cons}", "", "| Site | Readout | natural good − bad | D_in | ratio | t_T − unpatched | t_I − unpatched |",
              "|---:|---|---|---|---|---|---|"]
        for site in SITES:
            g = res[(res.construction == cons) & (res.site == site)]
            for r in g.itertuples():
                L.append(f"| {site} | {r.readout} | {f(r.nat_est, r.nat_lo95, r.nat_hi95)} | "
                         f"{f(r.D_in_est, r.D_in_lo95, r.D_in_hi95)} | {r.ratio_est:.2f} [{r.ratio_lo95:.2f}, {r.ratio_hi95:.2f}] | "
                         f"{f(r.T_vs_unpatched_est, r.T_vs_unpatched_lo95, r.T_vs_unpatched_hi95)} | "
                         f"{f(r.I_vs_unpatched_est, r.I_vs_unpatched_lo95, r.I_vs_unpatched_hi95)} |")
        L.append("")
        o = res[(res.construction == cons) & (res.readout == "O")]
        L += ["Excess object log-probability, D_in(O) − natural O: " +
              ", ".join(f"site {r.site} {f(r.excess_est, r.excess_lo95, r.excess_hi95)}" for r in o.itertuples()), ""]
    L += ["## Declared decisions", ""]
    for cons in ("passive", "OR", "TC"):
        for site in DECL:
            v, notes, ex = decide(res, cons, site, ("passive", "TC"))
            L.append(f"- **{cons}, site {site}: {v}**" + (f"; **{ex}**" if ex else "") + f" ({'; '.join(notes)}).")
    L.append("")
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="results/round4/inrange")
    ap.add_argument("--report", default="reports/round4/a2_inrange.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
