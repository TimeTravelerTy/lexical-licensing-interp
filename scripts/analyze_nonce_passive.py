#!/usr/bin/env python3
"""Nonce-verb passives (round3_plan.md, C8): analysis.

z per prompt and site: per fold basis, sign from the DAS training items and the
held-out DAS active scale (0 = intransitive, 1 = transitive level); nonce
prompts average all 15 bases; real-verb prompts of DAS pairs use only bases
where the pair is held out. Raw (sign-aligned) projections are reported too.

Per lemma (mean over its 4 slots), contrasts for passive and active probes:
matched T - I, mismatched T - I (two mappings), balanced AB - BA, and matched -
mean mismatched. Lemma bootstrap (2,000 draws, seed 17). Real-verb reference:
good - bad passives in the same long (neutral) context and alone. Causal check:
the matched-T probe's d_s coordinate into the matched-I probe (and reverse) at
sites 6 and 8. Decision rules: round3_plan.md, C8.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RD = ("by", "prep_noby", "O", "dot")
DECL = (6, 8)


def z_table(P, proj, sites, keys, das, pf):
    """[prompt, site] z and raw (sign-aligned), cross-fitted for DAS verbs."""
    pid_of = dict(zip(P.prompt, P.pid))
    das = das.assign(pid=das.prompt.map(pid_of))
    fold = {(k, f): das.pair_id.map(dict(zip(pf.pair_id, pf[f"fold_split{k}"]))).to_numpy() for k, f in keys}
    verb_pair = dict(zip(das.verb, das.pair_id))
    elig = np.ones((len(P), len(keys)), bool)
    pfi = pf.set_index("pair_id")
    for i, (kind, lemma) in enumerate(zip(P.kind, P.lemma)):
        if kind == "real" and lemma in verb_pair:
            elig[i] = [pfi.loc[verb_pair[lemma], f"fold_split{k}"] == f for k, f in keys]
    Z = np.zeros((len(P), len(sites)))
    R = np.zeros((len(P), len(sites)))
    for si in range(len(sites)):
        z = np.zeros((len(P), len(keys)))
        r = np.zeros((len(P), len(keys)))
        for j, (k, f) in enumerate(keys):
            x = proj[:, si, j]
            xd = x[das.pid.to_numpy()]
            tr = (fold[(k, f)] != f) & (das.subject != "David").to_numpy()
            te = fold[(k, f)] == f
            c = das.cls.to_numpy()
            sign = np.sign(xd[tr & (c == 1)].mean() - xd[tr & (c == 0)].mean())
            mt, mi = (sign * xd[te & (c == 1)]).mean(), (sign * xd[te & (c == 0)]).mean()
            z[:, j], r[:, j] = (sign * x - mi) / (mt - mi), sign * x
        Z[:, si] = (z * elig).sum(1) / elig.sum(1)
        R[:, si] = (r * elig).sum(1) / elig.sum(1)
    return Z, R


def boot(x, B, rng):
    x = np.asarray(x, float)
    return np.r_[x.mean(), x[rng.integers(0, len(x), (B, len(x)))].mean(1)]


def q(d):
    return float(d[0]), float(np.percentile(d[1:], 2.5)), float(np.percentile(d[1:], 97.5))


def run(args):
    ddir, rdir = Path(args.data_dir), Path(args.out_dir)
    P = pd.read_csv(ddir / "prompts.csv")
    nat = pd.read_parquet(rdir / "natural.parquet").set_index("pid").loc[P.pid]
    pz = np.load(rdir / "projections.npz")
    sites = [int(s) for s in pz["sites"]]
    keys = [(int(c[1]), int(c.split("_f")[1])) for c in pz["keys"]]
    das = pd.read_csv(Path(args.root) / "final_strict" / "items.csv")
    pf = pd.read_csv(Path(args.root) / "final_strict" / "pairs_folds.csv")
    Z, R = z_table(P, pz["proj"], sites, keys, das, pf)
    cols = {**{f"z{s}": Z[:, i] for i, s in enumerate(sites)}, **{f"raw{s}": R[:, i] for i, s in enumerate(sites)},
            **{k: nat[k].to_numpy() for k in RD}}
    T = P.assign(**cols)
    rng = np.random.default_rng(args.seed)
    measures = [f"z{s}" for s in sites] + [f"raw{s}" for s in sites] + list(RD)
    rows = []
    nonce = T[T.kind == "nonce"]
    lem = nonce.groupby(["probe_type", "lemma", "cond"])[measures].mean()
    contrasts = {"matched T - I": ("matched_T", "matched_I"), "mismatched1 T - I": ("mismatched1_T", "mismatched1_I"),
                 "mismatched2 T - I": ("mismatched2_T", "mismatched2_I"), "balanced AB - BA": ("balanced_AB", "balanced_BA")}
    for pt in ("passive", "active"):
        L = lem.loc[pt]
        diffs = {name: L.xs(a, level="cond") - L.xs(b, level="cond") for name, (a, b) in contrasts.items()}
        diffs["matched - mismatched"] = diffs["matched T - I"] - (diffs["mismatched1 T - I"] + diffs["mismatched2 T - I"]) / 2
        idx = np.vstack([np.arange(len(L.xs("none", level="cond")))[None],
                         rng.integers(0, len(L.xs("none", level="cond")), (args.n_boot, len(L.xs("none", level="cond"))))])
        for name, d in diffs.items():
            for m in measures:
                v = d[m].to_numpy()[idx].mean(1)
                e, lo, hi = q(v)
                rows.append({"probe": pt, "contrast": name, "measure": m, "est": e, "lo95": lo, "hi95": hi})
        for cond in ("none", "matched_T", "matched_I", "balanced_AB", "balanced_BA"):
            for m in measures:
                rows.append({"probe": pt, "contrast": f"level {cond}", "measure": m,
                             "est": float(L.xs(cond, level="cond")[m].mean()), "lo95": np.nan, "hi95": np.nan})
    # real-verb reference: good - bad passive, per pair (mean over slots)
    real = T[T.kind == "real"]
    for cond in ("none", "neutral"):
        r = real[real.cond == cond].groupby(["context_lemma", "probe_type"])[measures].mean()
        d = r.xs("real_good", level="probe_type") - r.xs("real_bad", level="probe_type")
        for m in measures:
            e, lo, hi = q(boot(d[m], args.n_boot, rng))
            rows.append({"probe": "real", "contrast": f"good - bad ({cond})", "measure": m, "est": e, "lo95": lo, "hi95": hi})
    # causal swap
    cs = pd.read_parquet(rdir / "causal_swap.parquet")
    natb = nat.loc[cs.base_pid.to_numpy()]
    for k in RD:
        cs[f"d_{k}"] = cs[k].to_numpy() - natb[k].to_numpy()
    for (site, bc), g in cs.groupby(["site", "base_cond"]):
        per = g.groupby("lemma")[[f"d_{k}" for k in RD]].mean()
        for k in RD:
            e, lo, hi = q(boot(per[f"d_{k}"], args.n_boot, rng))
            rows.append({"probe": "passive", "contrast": f"causal swap site {site}: {bc} base <- other", "measure": k,
                         "est": e, "lo95": lo, "hi95": hi})
    res = pd.DataFrame(rows)
    res.to_csv(rdir / "nonce_summary.csv", index=False)
    write_report(args, res, sites)


def write_report(args, res, sites):
    g = lambda pt, c, m: res[(res.probe == pt) & (res.contrast == c) & (res.measure == m)].iloc[0]
    f3 = lambda r, k=2: f"{r.est:+.{k}f} [{r.lo95:+.{k}f}, {r.hi95:+.{k}f}]"
    L = ["# C8. Nonce verbs: context-sensitive separation along d in passives", "",
         "Spec: `round3_plan.md`, C8. Run: `run_nonce_passive.py`; analysis: `analyze_nonce_passive.py`. 80 nonce "
         "lemmas × 4 slots; per-lemma means; 95% CIs: lemma bootstrap (2,000 draws, seed 17). z: 0 = held-out "
         "active intransitive, 1 = transitive level of each site's d (a fixed ruler).", "",
         "## Passive probe (\"The house was dakked\")", "",
         "| Site | matched T − I | mismatched1 | mismatched2 | balanced AB − BA | matched − mismatched | real good − bad (neutral ctx) | active probe: matched |",
         "|---:|---|---|---|---|---|---|---|"]
    for s in sites:
        m = f"z{s}"
        L.append(f"| {s} | {f3(g('passive', 'matched T - I', m))} | {f3(g('passive', 'mismatched1 T - I', m))} | "
                 f"{f3(g('passive', 'mismatched2 T - I', m))} | {f3(g('passive', 'balanced AB - BA', m))} | "
                 f"{f3(g('passive', 'matched - mismatched', m))} | {f3(g('real', 'good - bad (neutral)', m))} | "
                 f"{f3(g('active', 'matched T - I', m))} |")
    L += ["", "Log-probabilities at the passive probe (nats):", "",
          "| Contrast | log P(\" by\") | log P(PREP without by) | log P(O) | log P(\".\") |", "|---|---|---|---|---|"]
    for c in ("matched T - I", "mismatched1 T - I", "mismatched2 T - I", "balanced AB - BA", "matched - mismatched"):
        L.append(f"| {c} | {f3(g('passive', c, 'by'))} | {f3(g('passive', c, 'prep_noby'))} | "
                 f"{f3(g('passive', c, 'O'))} | {f3(g('passive', c, 'dot'))} |")
    L.append(f"| real good − bad (neutral ctx) | {f3(g('real', 'good - bad (neutral)', 'by'))} | "
             f"{f3(g('real', 'good - bad (neutral)', 'prep_noby'))} | {f3(g('real', 'good - bad (neutral)', 'O'))} | "
             f"{f3(g('real', 'good - bad (neutral)', 'dot'))} |")
    L += ["", "Causal check (d_s coordinate of the matched-T probe into the matched-I probe; Δ from unpatched):", "",
          "| Site | base | Δ log P(\" by\") | Δ log P(PREP\\by) | Δ log P(O) |", "|---:|---|---|---|---|"]
    for s in DECL:
        for bc in ("matched_I", "matched_T"):
            c = f"causal swap site {s}: {bc} base <- other"
            L.append(f"| {s} | {bc} | {f3(g('passive', c, 'by'))} | {f3(g('passive', c, 'prep_noby'))} | "
                     f"{f3(g('passive', c, 'O'))} |")
    L += ["", "## Declared decisions (passive probe)", ""]
    for s in DECL:
        m = f"z{s}"
        dm, db = g("passive", "matched T - I", m), g("passive", "balanced AB - BA", m)
        bby, bpp = g("passive", "balanced AB - BA", "by"), g("passive", "balanced AB - BA", "prep_noby")
        a = dm.lo95 > 0 and dm.est >= 0.05
        b = db.lo95 > 0 and db.est >= 0.05
        c = bby.lo95 > 0 and bby.est >= 0.1 and bby.est > bpp.est
        real = g("real", "good - bad (neutral)", m).est
        act = g("active", "matched T - I", m).est
        L.append(f"- Site {s}: context-sensitive separation along d: **{'yes' if a else 'no'}** (Δ_matched "
                 f"{dm.est:+.3f}); verb-specific: **{'yes' if b else 'no'}** (Δ_balanced {db.est:+.3f}); raises *by* "
                 f"verb-specifically: **{'yes' if c else 'no'}** (Δ_balanced by {bby.est:+.2f} vs PREP\\by "
                 f"{bpp.est:+.2f}). Scale: Δ_matched / real-verb gap = {dm.est / real:.2f}; passive / active "
                 f"Δ_matched = {dm.est / act:.2f}.")
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results/das_round2")
    ap.add_argument("--data-dir", default="data/nonce_passive")
    ap.add_argument("--out-dir", default="results/das_round2/nonce_passive")
    ap.add_argument("--report", default="reports/passive_das_prep/round3_nonce.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
