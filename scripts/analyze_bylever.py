#!/usr/bin/env python3
"""Round 4, A4: the B5 "by lever" (reports/round4/plan.md, A4): analysis.

Arms on bad active bases (B5 plan): arm 1 = G - B passive donors, arm 3 = T - I active donors.
For each intervention (dp, shuf, perp, dpminus) D with the B5 three-way bootstrap (base pairs within
band, subjects, donor pairs; 2,000 draws, seed 17); paired differences against dp on the same draws;
classification with the active-DAS delta_s; nulls (norm-matched to shuf / perp) on split-0 G / B rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_passive_test import Boot, cells, ci, classify, pair_values, site_delta, summarize

READ = ("M", "O", "I", "det", "pron", "refl", "by", "dot", "the", "him")
NAMES = ("dp", "shuf", "perp", "dpminus")
KEYS = [(k, f) for k in range(3) for f in range(5)]
ARMS = {"arm1": ("G", "B"), "arm3": ("T", "I")}


def cosines(rdir, act, site):
    z = np.load(Path(rdir) / f"final_site{site}" / "bases_np.npz")
    out = {"dp_s": [], "da_s": [], "perp_da": [], "dp_da": []}
    for k, f in KEYS:
        dp, s = z[f"r1_s{k}_f{f}"].reshape(-1), z[f"shuf_s{k}_f{f}"].reshape(-1)
        dp, s = dp / np.linalg.norm(dp), s / np.linalg.norm(s)
        da = act[f"das_site{site}_s{k}_f{f}"].reshape(-1)
        da = da / np.linalg.norm(da)
        e = dp - (dp @ s) * s
        e /= np.linalg.norm(e)
        out["dp_s"].append(abs(dp @ s))
        out["da_s"].append(abs(da @ s))
        out["perp_da"].append(abs(e @ da))
        out["dp_da"].append(abs(dp @ da))
    return {k: float(np.median(v)) for k, v in out.items()}


def run(args):
    ddir, odir = Path(args.data_dir), Path(args.out_dir)
    items = pd.read_csv(ddir / "items.csv")
    plan = pd.read_csv(ddir / "plan.csv.gz", dtype={c: "category" for c in ("item_id", "side", "cond", "donor_pair")})
    nat = pd.read_parquet(odir / "natural.parquet").set_index("pid")
    act = np.load(args.active_bases)
    pr = items.drop_duplicates("pair_id").set_index("pair_id")
    pair_ix = {p: i for i, p in enumerate(pr.index)}
    ctx_ix = {c: i for i, c in enumerate(sorted(items.context_id.unique()))}
    sites = [int(s) for s in args.sites.split(",")]
    rows, dec_rows = [], []
    for site in sites:
        bt = Boot(len(ctx_ix), args.n_boot, args.seed)
        wp = bt.pair_weights(pr.band.to_numpy())
        pix = np.arange(len(pr))
        delta = site_delta(Path(args.root), site)
        S, norms = {}, {}
        for n in NAMES:
            pat = pd.read_parquet(odir / f"patches_{n}_site{site}.parquet")
            pl = plan.set_index("row").loc[pat.row.to_numpy()].reset_index()
            df = pl[["item_id", "cond", "donor_pair"]].join(items.set_index("item_id")[["pair_id", "context_id"]],
                                                            on="item_id")
            df[list(READ)] = pat[list(READ)].to_numpy() - nat.loc[pl.base.to_numpy(), list(READ)].to_numpy()
            for arm, (pos, neg) in ARMS.items():
                x = df[df.cond.isin([pos, neg])].assign(dq=lambda y: y.donor_pair)
                qs = sorted(x.donor_pair.unique())
                qix = {q: i for i, q in enumerate(qs)}
                vP, mP = cells(x[x.cond == pos], pair_ix, ctx_ix, qix, len(qs), READ)
                vN, mN = cells(x[x.cond == neg], pair_ix, ctx_ix, qix, len(qs), READ)
                assert (mP == mN).all()
                S[(n, arm)] = summarize(pair_values(vP - vN, mP, bt.wc, bt.donor_weights(arm, len(qs))), wp, pix)
                norms[(n, arm)] = float(pat.disp_norm[df.cond.isin([pos, neg]).to_numpy()].mean())
        nulls = {(n, arm): np.load(odir / f"null_{n}_{arm}_site{site}.npz")
                 for n, arms in (("shuf", ("arm1",)), ("perp", ("arm1", "arm3")), ("dpminus", ("arm1", "arm3")))
                 for arm in arms}
        for (n, arm), s in S.items():
            for j, rd in enumerate(READ):
                row = {"site": site, "intervention": n, "arm": arm, "readout": rd, "delta": delta,
                       "disp_norm": norms[(n, arm)], **ci(s[:, j])}
                if rd in ("O", "by"):
                    row["class"] = classify(row, delta)
                if n != "dp":
                    row.update({f"diff_dp_{k}": v for k, v in ci(S[("dp", arm)][:, j] - s[:, j]).items()})
                if (n, arm) in nulls and rd in ("O", "by"):
                    z = nulls[(n, arm)]
                    jj = list(z["readouts"]).index(rd)
                    row["null_p95"] = float(np.percentile(z["draws"][:, jj], 95))
                    row["split0_point"] = float(z["das"][jj])
                rows.append(row)
        res_s = pd.DataFrame([r for r in rows if r["site"] == site])
        dec_rows.append({"site": site, **cosines(args.reverse_dir, act, site), **decide(res_s, delta)})
        print(f"site {site} done", flush=True)
    res, dec = pd.DataFrame(rows), pd.DataFrame(dec_rows)
    res.to_csv(odir / "bylever_summary.csv", index=False)
    dec.to_csv(odir / "bylever_decisions.csv", index=False)
    write_report(args, res, dec)


def decide(res, delta):
    g = lambda n, arm, rd: res[(res.intervention == n) & (res.arm == arm) & (res.readout == rd)].iloc[0]
    sb = g("shuf", "arm1", "by")
    raises = bool(sb.lo95 > 0 and sb.est >= 0.2 and sb.split0_point > sb.null_p95)
    out = {"shuf_raises_by": raises}
    for n in ("perp", "dpminus"):
        if not raises:
            out[f"{n}_reading"] = "not read (shuffled direction does not raise by)"
            continue
        by_red = g(n, "arm1", "by").est <= 0.5 * g("dp", "arm1", "by").est
        checked = [arm for arm in ("arm1", "arm3") if g("dp", arm, "O")["class"] == "RISE"]
        # retained: still a RISE and beyond its own null, in every arm where d_p's O effect was a RISE
        retained = all(g(n, arm, "O")["class"] == "RISE" and g(n, arm, "O").split0_point > g(n, arm, "O").null_p95
                       for arm in checked)
        # lost: a paired reduction (CI above 0, >= delta) in every arm where d_p's O effect was a RISE
        lost = bool(checked) and all(g(n, arm, "O").diff_dp_lo95 > 0 and g(n, arm, "O").diff_dp_est >= delta
                                     for arm in checked)
        if by_red and retained:
            reading = "by lever separable" + ("" if checked else " (no O RISE under d_p to preserve)")
        elif by_red and lost:
            reading = "the by lever carries the transfer"
        else:
            reading = "mixed"
        out[f"{n}_reading"] = reading
        out[f"{n}_by_ratio"] = float(g(n, "arm1", "by").est / g("dp", "arm1", "by").est)
    return out


def write_report(args, res, dec):
    g = lambda s, n, arm, rd: res[(res.site == s) & (res.intervention == n) & (res.arm == arm) & (res.readout == rd)].iloc[0]
    f = lambda r: f"{r.est:+.2f} [{r.lo95:+.2f}, {r.hi95:+.2f}]"
    L = ["# Round 4, A4. The B5 by lever", "",
         "Spec: `plan.md`, A4. Run: `run_bylever.py`; analysis: `analyze_bylever.py`. Bad active bases (\"She has "
         "emerged\"), 64 primary pairs. Arm 1: G − B passive donors; arm 3: T − I active donors. dp = d_p; shuf = "
         "shuffled-label passive basis s; perp = interchange along d_p⊥s; dpminus = d_p's displacement with its s "
         "component removed. 95% CIs: B5 three-way bootstrap.", "",
         "| Site | \\|cos(d_p,s)\\| | \\|cos(d_a,s)\\| | \\|cos(d_p⊥s,d_a)\\| | \\|cos(d_p,d_a)\\| |", "|---:|---:|---:|---:|---:|"]
    for r in dec.itertuples():
        L.append(f"| {r.site} | {r.dp_s:.2f} | {r.da_s:.2f} | {r.perp_da:.2f} | {r.dp_da:.2f} |")
    for arm, nm in (("arm1", "Arm 1 (passive donors)"), ("arm3", "Arm 3 (active donors)")):
        L += ["", f"## {nm}", "", "| Site | δ | dp: O | dp: by | shuf: O | shuf: by | perp: O | perp: by | dpminus: O | dpminus: by |",
              "|---:|---:|---|---|---|---|---|---|---|---|"]
        for s in sorted(res.site.unique()):
            c = [g(s, n, arm, rd) for n in NAMES for rd in ("O", "by")]
            L.append(f"| {s} | {c[0].delta:.2f} | " + " | ".join(f"{f(r)} {r['class']}" for r in c) + " |")
    L += ["", "Paired reductions against d_p (dp − X), arm 1 / arm 3:", "",
          "| Site | perp: O | perp: by | dpminus: O | dpminus: by | displacement norm dp / shuf / perp / dpminus (arm 1) |",
          "|---:|---|---|---|---|---|"]
    for s in sorted(res.site.unique()):
        cells_ = []
        for n in ("perp", "dpminus"):
            for rd in ("O", "by"):
                a, b = g(s, n, "arm1", rd), g(s, n, "arm3", rd)
                cells_.append(f"{a.diff_dp_est:+.2f} [{a.diff_dp_lo95:+.2f}, {a.diff_dp_hi95:+.2f}] / "
                              f"{b.diff_dp_est:+.2f} [{b.diff_dp_lo95:+.2f}, {b.diff_dp_hi95:+.2f}]")
        nn = " / ".join(f"{g(s, n, 'arm1', 'O').disp_norm:.2f}" for n in NAMES)
        L.append(f"| {s} | " + " | ".join(cells_) + f" | {nn} |")
    L += ["", "Nulls (split-0 rows, point estimate / 95th percentile of 100 random directions norm-matched to the "
              "intervention's displacement):", "",
          "| Site | shuf arm 1: by | shuf arm 1: O | perp arm 1: by | perp arm 1: O | perp arm 3: O | dpminus arm 1: O | "
          "dpminus arm 3: O |", "|---:|---|---|---|---|---|---|---|"]
    for s in sorted(res.site.unique()):
        c = [g(s, "shuf", "arm1", "by"), g(s, "shuf", "arm1", "O"), g(s, "perp", "arm1", "by"), g(s, "perp", "arm1", "O"),
             g(s, "perp", "arm3", "O"), g(s, "dpminus", "arm1", "O"), g(s, "dpminus", "arm3", "O")]
        L.append(f"| {s} | " + " | ".join(f"{r.split0_point:+.2f} / {r.null_p95:+.2f}" for r in c) + " |")
    L += ["", "## Declared decisions", ""]
    for r in dec.itertuples():
        extra = ""
        if r.shuf_raises_by:
            extra = (f"; d_p⊥s: **{r.perp_reading}** (by ratio {r.perp_by_ratio:.2f}); d_p − s-part: "
                     f"**{r.dpminus_reading}** (by ratio {r.dpminus_by_ratio:.2f})")
        L.append(f"- Site {r.site}: shuffled direction raises \" by\" in actives: **{'yes' if r.shuf_raises_by else 'no'}**"
                 + extra + ".")
    L.append("")
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results/das_round2")
    ap.add_argument("--data-dir", default="data/das_round2/reverse_test")
    ap.add_argument("--reverse-dir", default="results/das_round2/reverse")
    ap.add_argument("--out-dir", default="results/round4/bylever")
    ap.add_argument("--active-bases", default="results/das_round2/bases_rank1.npz")
    ap.add_argument("--report", default="reports/round4/a4_bylever.md")
    ap.add_argument("--sites", default="4,6,8,10,12,14,16,17")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
