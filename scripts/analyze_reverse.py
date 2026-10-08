#!/usr/bin/env python3
"""Reverse DAS (round3_plan.md, B5): passive-side training results and the active test.

Passive side (`reverse/final_site{s}/summary.csv`): held-out Delta M_p (bad
base <- good source), both directions, same-class preservation, IIA vs the
natural threshold accuracy; robustness gate = DAS pc_dM above the
norm-matched random 95th percentile in >= 12 of 15 runs.

Active test (`reverse_test/`): three arms on bad active bases, D = G - B (or
T - I) donors, the passive test's three-way bootstrap (base pairs within
band, subjects, donor pairs; 2,000 draws, seed 17):
  arm 1 d_p + passive donors; arm 2 d_a + the same passive donors;
  arm 3 d_p + active donors (dose control).
Classification with the active-DAS delta_s; random null (split-0 rows).
Directions: |cos(d_p, d_a)| per split and fold, mean-direction cosine, d_p
stability. Natural projection onto d_p: AUC, good vs bad actives.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_passive_test import Boot, auc, cells, ci, classify, pair_values, site_delta, summarize

READ = ("M", "O", "I", "det", "pron", "refl", "by", "dot", "the", "him")
KEYS = [(k, f) for k in range(3) for f in range(5)]


def passive_side(rdir, site):
    s = pd.read_csv(rdir / f"final_site{site}" / "summary.csv")
    das = s[s.control == "das"].set_index(["split", "fold"])
    rnd = s[s.control == "random_normmatched"].groupby(["split", "fold"]).pc_dM.quantile(0.95)
    shuf = s[s.control == "shuffled_labels"].set_index(["split", "fold"])
    beats = int((das.pc_dM > rnd.loc[das.index]).sum())
    out = {"site": site, "gate_runs": beats, "gate": beats >= 12}
    for k in ("pc_dM", "pc_dM_heldout_ctx", "rev_dM", "same_abs_dM", "pc_frac_median", "iia_cross", "iia_same",
              "nat_acc", "tau"):
        out[k] = float(das[k].mean())
    out["random_p95_pc_dM"] = float(rnd.mean())
    out["shuffled_pc_dM"] = float(shuf.pc_dM.mean())
    return out


def directions(rdir, site, act):
    b = np.load(rdir / f"final_site{site}" / "bases_np.npz")  # torch-free export written by the final job
    dp = {k: b[f"r1_s{k[0]}_f{k[1]}"] for k in KEYS}
    da = {k: act[f"das_site{site}_s{k[0]}_f{k[1]}"].reshape(-1) for k in KEYS}
    cos = np.array([abs(dp[k] @ da[k]) / np.linalg.norm(dp[k]) / np.linalg.norm(da[k]) for k in KEYS])

    def mean_dir(d):
        ref = d[KEYS[0]]
        m = sum(np.sign(v @ ref) * v / np.linalg.norm(v) for v in d.values())
        return m / np.linalg.norm(m)

    stab = [abs(dp[a] @ dp[c]) / np.linalg.norm(dp[a]) / np.linalg.norm(dp[c]) for a, c in combinations(KEYS, 2)]
    return {"cos_median": float(np.median(cos)), "cos_min": float(cos.min()), "cos_max": float(cos.max()),
            "cos_mean_dirs": float(abs(mean_dir(dp) @ mean_dir(da))), "dp_stability_median": float(np.median(stab))}


def projection_auc(tdir, ddir, site, items, P, pf):
    """Good vs bad active bases along d_p, cross-fitted; sign per basis from the fold's training-pair passives."""
    proj = pd.read_parquet(tdir / f"projections_site{site}.parquet").set_index("pid")
    tverb = {r.trans: r.pair_id for r in pf.itertuples()}
    iverb = {r.intrans: r.pair_id for r in pf.itertuples()}
    das_lemma = {**tverb, **iverb}
    pas = P[(P.kind == "passive") & P.lemma.isin(das_lemma)]
    vals = {}
    for k, f in KEYS:
        col = f"s{k}_f{f}"
        fo = dict(zip(pf.pair_id, pf[f"fold_split{k}"]))
        tr = pas[pas.lemma.map(das_lemma).map(fo) != f]
        x = proj.loc[tr.pid, col].to_numpy()
        t = tr.lemma.isin(tverb).to_numpy()
        sign = np.sign(x[t].mean() - x[~t].mean())
        vals[(k, f)] = (sign, fo)
    out = {}
    pid_of = dict(zip(P.prompt, P.pid))
    for side in ("good", "bad"):
        v = []
        for it in items.itertuples():
            pid = int(pid_of[getattr(it, f"{side}_prompt")])
            dp_ = it.das_pair if isinstance(it.das_pair, str) and it.das_pair else None  # "" is read back as NaN
            ks = [(k, f) for k, f in KEYS if dp_ is None or vals[(k, f)][1].get(dp_) == f]
            v.append(np.mean([vals[kf][0] * proj.loc[pid, f"s{kf[0]}_f{kf[1]}"] for kf in ks]))
        out[side] = np.array(v)
    items = items.assign(pg=out["good"], pb=out["bad"])
    verb_g, verb_b = items.groupby("good_lemma").pg.mean(), items.groupby("bad_lemma").pb.mean()
    return {"auc_items": auc(items.pg.to_numpy(), items.pb.to_numpy()),
            "auc_verbs": auc(verb_g.to_numpy(), verb_b.to_numpy())}


def arms(tdir, site, items, plan, nat, bt, pr):
    pair_ix = {p: i for i, p in enumerate(pr.index)}
    ctx_ix = {c: i for i, c in enumerate(sorted(items.context_id.unique()))}
    natv = nat.set_index("pid").loc[plan.base.to_numpy(), list(READ)].to_numpy()
    df = plan[["item_id", "side", "cond", "donor_pair"]].join(items.set_index("item_id")[["pair_id", "context_id"]],
                                                                on="item_id")
    out = {}
    for basis, fname in (("dp", f"patches_site{site}.parquet"), ("da", f"patches_da_site{site}.parquet")):
        pat = pd.read_parquet(tdir / fname)
        assert (pat.row.to_numpy() == plan.row.to_numpy()).all()
        d = df.copy()
        d[list(READ)] = pat[list(READ)].to_numpy() - natv
        for side in ("bad", "good"):
            for pos, neg, nm in (("G", "B", "passive"), ("T", "I", "active")):
                x = d[(d.side == side) & d.cond.isin([pos, neg])].assign(dq=lambda y: y.donor_pair)
                qs = sorted(x.donor_pair.unique())
                qix = {q: i for i, q in enumerate(qs)}
                vP, mP = cells(x[x.cond == pos], pair_ix, ctx_ix, qix, len(qs), READ)
                vN, mN = cells(x[x.cond == neg], pair_ix, ctx_ix, qix, len(qs), READ)
                assert (mP == mN).all()
                out[f"{basis}_{nm}_{side}"] = pair_values(vP - vN, mP, bt.wc, bt.donor_weights("q", len(qs)))
        for cond in ("same_verb", "active_swap"):
            x = d[(d.side == "bad") & (d.cond == cond)].assign(dq=0)
            v, m = cells(x, pair_ix, ctx_ix, {0: 0}, 1, READ)
            out[f"{basis}_{cond}_bad"] = pair_values(v, m, bt.wc, np.ones((bt.B + 1, 1), np.float32))
    return out


def run(args):
    rdir, tdir, ddir = Path(args.reverse_dir), Path(args.test_dir), Path(args.data_dir)
    items = pd.read_csv(ddir / "items.csv")
    P = pd.read_csv(ddir / "prompts.csv")
    plan = pd.read_csv(ddir / "plan.csv.gz", dtype={c: "category" for c in ("item_id", "side", "cond", "donor_pair")})
    nat = pd.read_parquet(tdir / "natural.parquet")
    pf = pd.read_csv(args.folds)
    act = np.load(args.active_bases)
    pr = items.drop_duplicates("pair_id").set_index("pair_id")
    sites = [int(s) for s in args.sites.split(",")]
    side_rows, rows, dir_rows = [], [], []
    for site in sites:
        side_rows.append(passive_side(rdir, site))
        dir_rows.append({"site": site, **directions(rdir, site, act), **projection_auc(tdir, ddir, site, items, P, pf)})
        bt = Boot(items.context_id.nunique(), args.n_boot, args.seed)
        A = arms(tdir, site, items, plan, nat, bt, pr)
        delta = site_delta(Path(args.root), site)
        rz = np.load(tdir / f"random_site{site}.npz")
        for band in ("all", "head", "tail", "xtail"):
            pix = np.flatnonzero(((pr.band == band) if band != "all" else pd.Series(True, index=pr.index)).to_numpy())
            wp = bt.pair_weights(pr.band.to_numpy()[pix])
            for k, a in A.items():
                s = summarize(a, wp, pix)
                for j, rd in enumerate(READ):
                    row = {"site": site, "band": band, "n_pairs": len(pix), "arm": k, "readout": rd, "delta": delta,
                           **ci(s[:, j])}
                    if rd in ("O", "by", "dot") and k.endswith("_bad"):
                        row["class"] = classify(row, delta)
                        if k == "dp_passive_bad" and band == "all":
                            j_ = list(rz["readouts"]).index(rd)
                            row["null_p95"] = float(np.percentile(rz["draws"][:, j_], 95))
                            row["null_p05"] = float(np.percentile(rz["draws"][:, j_], 5))
                            row["das_split0"] = float(rz["das"][j_])
                    rows.append(row)
        print(f"site {site} done", flush=True)
    res, side, dirs = pd.DataFrame(rows), pd.DataFrame(side_rows), pd.DataFrame(dir_rows)
    tdir.mkdir(parents=True, exist_ok=True)
    res.to_csv(tdir / "reverse_summary.csv", index=False)
    side.to_csv(tdir / "reverse_passive_side.csv", index=False)
    dirs.to_csv(tdir / "reverse_directions.csv", index=False)
    write_report(args, res, side, dirs)


def write_report(args, res, side, dirs):
    g = lambda site, arm, rd, band="all": res[(res.site == site) & (res.arm == arm) & (res.readout == rd)
                                             & (res.band == band)].iloc[0]
    f3 = lambda r: f"{r.est:+.2f} [{r.lo95:+.2f}, {r.hi95:+.2f}]"
    L = ["# B5. Reverse DAS: trained on passives, tested on actives", "",
         "Spec: `round3_plan.md`, B5. Training: `run_passive_das.py`; test: `run_reverse_test.py`; analysis: "
         "`analyze_reverse.py`. d_p = passive-trained, d_a = active-trained direction of the same site, split and fold.",
         "", "## Passive side (held-out pairs; 15 runs per site)", "",
         "| Site | Epochs | Δ M_p bad←good | random p95 | gate runs | T←I Δ M_p | same-class \\|Δ\\| | IIA (natural acc.) | "
         "gap fraction | shuffled Δ M_p |", "|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    for r in side.itertuples():
        ep = json.loads((Path(args.reverse_dir) / f"frozen_config_site{r.site}.json").read_text())["epochs"]
        L.append(f"| {r.site} | {ep} | {r.pc_dM:+.2f} | {r.random_p95_pc_dM:+.2f} | {r.gate_runs}/15 | {r.rev_dM:+.2f} | "
                 f"{r.same_abs_dM:.2f} | {r.iia_cross:.2f} ({r.nat_acc:.2f}) | {r.pc_frac_median:.2f} | "
                 f"{r.shuffled_pc_dM:+.2f} |")
    L += ["", "## Directions and natural projection", "",
          "| Site | \\|cos(d_p, d_a)\\| median [min, max] | mean directions | d_p stability | AUC actives along d_p: items "
          "(verbs) |", "|---:|---|---:|---:|---|"]
    for r in dirs.itertuples():
        L.append(f"| {r.site} | {r.cos_median:.2f} [{r.cos_min:.2f}, {r.cos_max:.2f}] | {r.cos_mean_dirs:.2f} | "
                 f"{r.dp_stability_median:.2f} | {r.auc_items:.3f} ({r.auc_verbs:.3f}) |")
    L += ["", "## Active test: D on bad active bases (\"She has emerged\"), 64 primary pairs", "",
          "| Site | δ | Arm 1 d_p + passive donors: O | by | Arm 2 d_a + passive donors: O | by | "
          "Arm 3 d_p + active donors: O | by | null p95 (O) | Reading |", "|---:|---:|---|---|---|---|---|---|---:|---|"]
    side_i = side.set_index("site")
    dir_i = dirs.set_index("site")
    for site in sorted(res.site.unique()):
        a1o, a1b = g(site, "dp_passive_bad", "O"), g(site, "dp_passive_bad", "by")
        a2o, a2b = g(site, "da_passive_bad", "O"), g(site, "da_passive_bad", "by")
        a3o, a3b = g(site, "dp_active_bad", "O"), g(site, "dp_active_bad", "by")
        rise1 = a1o["class"] == "RISE" and a1o.das_split0 > a1o.null_p95
        cosm, aucv = dir_i.cos_median[site], dir_i.auc_items[site]
        if not side_i.gate[site]:
            rd = "not interpretable (passive gate fails)"
        elif cosm >= 0.5 and rise1 and aucv >= 0.8:
            rd = "aligned, with cross-frame causal transfer"
        elif cosm < 0.3 and a1o["class"] == "NO RISE" and a3o["class"] == "NO RISE":
            rd = "passive-specific"
        elif a1o["class"] == "NO RISE" and a3o["class"] == "RISE":
            rd = "small transfer at the passive donor dose"
        else:
            rd = "mixed / unresolved"
        L.append(f"| {site} | {a1o.delta:.2f} | {f3(a1o)} {a1o['class']} | {f3(a1b)} | {f3(a2o)} {a2o['class']} | "
                 f"{f3(a2b)} | {f3(a3o)} {a3o['class']} | {f3(a3b)} | {a1o.null_p95:+.2f} | **{rd}** |")
    L += ["", "Voice-change and swap controls (d_p, bad bases, Δ from unpatched): O / by",
          "", "| Site | own passive → active | active swap (good → bad) |", "|---:|---|---|"]
    for site in sorted(res.site.unique()):
        s1o, s1b = g(site, "dp_same_verb_bad", "O"), g(site, "dp_same_verb_bad", "by")
        s2o, s2b = g(site, "dp_active_swap_bad", "O"), g(site, "dp_active_swap_bad", "by")
        L.append(f"| {site} | {s1o.est:+.2f} / {s1b.est:+.2f} | {s2o.est:+.2f} / {s2b.est:+.2f} |")
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results/das_round2")
    ap.add_argument("--reverse-dir", default="results/das_round2/reverse")
    ap.add_argument("--test-dir", default="results/das_round2/reverse_test")
    ap.add_argument("--data-dir", default="data/das_round2/reverse_test")
    ap.add_argument("--folds", default="results/das_round2/final_strict/pairs_folds.csv")
    ap.add_argument("--active-bases", default="results/das_round2/bases_rank1.npz")
    ap.add_argument("--sites", default="4,6,8,10,12,14,16,17")
    ap.add_argument("--report", default="reports/passive_das_prep/round3_reverse.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
