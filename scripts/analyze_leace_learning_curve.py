#!/usr/bin/env python3
"""Report for run_leace_learning_curve.py (Part A) and run_leace_postscrub.py (Parts B, C).
Rules: reports/pythia/leace_learning_curve_rules.md.

Share of the *by* gap remaining = mean erased gap / mean natural gap over held-out items (T - I log P(by), item level).
Probe accuracy = pooled over held-out test prompts. 95% CIs: the task-1 two-way cluster bootstrap
(analyze_das_prep.Boot, verb pairs within verb band x contexts within context band), 1000 draws, seed 17; erasers,
subsets and probe predictions fixed. Per n, the 5 subsets' values are averaged within each draw.

Writes reports/pythia/leace_learning_curve.md and leace_learning_curve_*.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_das_prep import Boot, table

CONDS = [("single_8", "single site 8"), ("scrub_0_12", "scrub 0–12")]
PCOLS = [("pred__single_8__s8", "single 8, site 8"), ("pred__scrub_0_12__s12", "scrub 0–12, site 12"),
         ("pred__scrub_0_12__s8", "scrub 0–12, site 8"), ("pred__natural__s8", "natural, site 8"),
         ("pred__natural__s12", "natural, site 12")]


def ci(d):
    return d[0], *np.nanpercentile(d[1:], [2.5, 97.5])


def f3(t, k=2):
    return f"{t[0]:.{k}f} [{t[1]:.{k}f}, {t[2]:.{k}f}]"


def run(args):
    lc, ps, out = Path(args.lc), Path(args.postscrub), Path(args.report_dir)
    P = pd.read_csv(lc / "subsets" / "prompts.csv.gz")
    N = len(P)
    cls = P.cls.to_numpy()
    # item table (both prompts of an item are always held out together: pair-level split)
    G = P[P.side == "good"].sort_values("item")
    B = P[P.side == "bad"].sort_values("item")
    it = G[["item", "verb_pair", "verb_band", "context_id", "context_band"]].reset_index(drop=True)
    g_row, b_row = G.row.to_numpy(), B.row.to_numpy()
    nI = len(it)
    bi = Boot(it, args.n_boot, np.random.default_rng(args.seed))
    bp = Boot(P, args.n_boot, np.random.default_rng(args.seed))  # prompt level, same cluster weights
    assert np.allclose(bi.wv, bp.wv) and np.allclose(bi.wc, bp.wc)

    gaps = pd.read_csv(args.gaps)
    gaps = gaps[gaps.subset == "all"].set_index("condition")
    rp = pd.read_csv(args.probes)
    rp = rp[rp.probe == "linear"]

    def stored_probe(cond, site):
        return float(rp[(rp.condition == cond) & (rp.site == site)].acc.iloc[0])

    # ---------------------------------------------------------------- Part A
    H = pd.read_csv(lc / "subsets" / "held.csv.gz")
    fits = pd.read_csv(lc / "subsets" / "fits.csv")
    diag = pd.read_csv(lc / "subsets" / "fit_diagnostics.csv")
    chkA = json.loads((lc / "subsets" / "checks.json").read_text())
    loo_dirs = sorted(lc.glob("loo_part*"))
    HL = pd.concat([pd.read_csv(d / "held.csv.gz") for d in loo_dirs], ignore_index=True)
    fl = pd.concat([pd.read_csv(d / "fits.csv") for d in loo_dirs], ignore_index=True)
    diag = pd.concat([diag] + [pd.read_csv(d / "fit_diagnostics.csv") for d in loo_dirs], ignore_index=True)
    assert len(HL) == N and (np.sort(HL.row.to_numpy()) == np.arange(N)).all(), "LOO does not cover every prompt once"
    assert sorted(fl.held_pair) == sorted(P.verb_pair.unique())
    LP = pd.read_csv(lc / "loo_probe" / "loo_probes.csv.gz").set_index("row").sort_index()

    def item_arrays(F):
        """Per fit: item-level gaps (NaN where not held out) and prompt-level probe correctness."""
        v = {c: np.full(N, np.nan) for c in ["natural"] + [c for c, _ in CONDS]}
        for c in v:
            v[c][F.row.to_numpy()] = F[f"lp_by__{c}"].to_numpy()
        gp = {c: v[c][g_row] - v[c][b_row] for c in v}
        mask = np.isfinite(gp["natural"])
        assert (np.isfinite(v["natural"][g_row]) == np.isfinite(v["natural"][b_row])).all()
        return gp, mask

    def share_draws(F):
        gp, mask = item_arrays(F)
        nat = bi.means(gp["natural"], mask)
        return {c: bi.means(gp[c], mask) / nat for c, _ in CONDS}

    def acc_draws(F, col):
        corr = np.full(N, np.nan)
        corr[F.row.to_numpy()] = (F[col].to_numpy() == cls[F.row.to_numpy()]).astype(float)
        m = np.isfinite(corr)
        return bp.means(np.nan_to_num(corr), m)

    rowsA, perfit = [], []
    sub = fits[fits.kind == "subset"]
    for n in sorted(sub.n.unique()):
        ids = sub[sub.n == n].fit_id.tolist()
        sh = {c: [] for c, _ in CONDS}
        ac = {c: [] for c, _ in PCOLS}
        kl = {c: [] for c, _ in CONDS}
        for fid in ids:
            F = H[H.fit_id == fid]
            s = share_draws(F)
            r = {"n": n, "fit_id": fid}
            for c, _ in CONDS:
                sh[c].append(s[c])
                kl[c].append(float(F[f"kl__{c}"].mean()))
                r[f"share__{c}"], r[f"kl__{c}"] = s[c][0], kl[c][-1]
            for c, _ in PCOLS:
                a = acc_draws(F, c)
                ac[c].append(a)
                r[c.replace("pred__", "acc__")] = a[0]
            perfit.append(r)
        row = {"n": str(n), "n_fits": len(ids), "n_held_pairs": str(int(sub[sub.n == n].n_held_pairs.iloc[0]))}
        for c, _ in CONDS:
            e = ci(np.mean(sh[c], axis=0))
            row.update({f"share__{c}": e[0], f"share__{c}_lo": e[1], f"share__{c}_hi": e[2],
                        f"share__{c}_min": min(x[0] for x in sh[c]), f"share__{c}_max": max(x[0] for x in sh[c]),
                        f"kl__{c}": float(np.mean(kl[c]))})
        for c, _ in PCOLS:
            e = ci(np.mean(ac[c], axis=0))
            k = c.replace("pred__", "acc__")
            row.update({k: e[0], f"{k}_lo": e[1], f"{k}_hi": e[2], f"{k}_min": min(x[0] for x in ac[c]),
                        f"{k}_max": max(x[0] for x in ac[c])})
        rowsA.append(row)
    # LOO
    s = share_draws(HL)
    row = {"n": "LOO (125)", "n_fits": len(fl), "n_held_pairs": "1"}
    for c, _ in CONDS:
        e = ci(s[c])
        row.update({f"share__{c}": e[0], f"share__{c}_lo": e[1], f"share__{c}_hi": e[2],
                    f"kl__{c}": float(HL[f"kl__{c}"].mean())})
    for c, _ in PCOLS:
        F = pd.DataFrame({"row": LP.index.to_numpy(), c: LP[c].to_numpy()})
        e = ci(acc_draws(F, c))
        k = c.replace("pred__", "acc__")
        row.update({k: e[0], f"{k}_lo": e[1], f"{k}_hi": e[2]})
    rowsA.append(row)
    A = pd.DataFrame(rowsA)
    A.to_csv(out / "leace_learning_curve_summary.csv", index=False)
    pd.DataFrame(perfit).to_csv(out / "leace_learning_curve_per_fit.csv", index=False)

    refs = {}
    for lab, pre, pr in (("cross-fitted (5-fold, ~101 pairs)", "", ""), ("in-sample (126)", "insample_", "insample_")):
        r = {}
        for c, _ in CONDS:
            q = gaps.loc[pre + c]
            r[c] = (q.share_remaining, q.share_remaining_lo, q.share_remaining_hi, q.kl_mean)
        r["probes"] = {"pred__single_8__s8": stored_probe(pr + "single", 8),
                       "pred__scrub_0_12__s12": stored_probe(pr + "scrub_all", 12),
                       "pred__scrub_0_12__s8": stored_probe(pr + "scrub_all", 8),
                       "pred__natural__s8": stored_probe(pr + "natural", 8),
                       "pred__natural__s12": stored_probe(pr + "natural", 12)}
        refs[lab] = r

    # ---------------------------------------------------------------- Parts B, C
    Q = pd.read_csv(ps / "preds.csv.gz")
    assert (Q.prompt.to_numpy() == P.prompt.to_numpy()).all()
    cosB = pd.read_csv(ps / "cosines.csv")
    chkP = json.loads((ps / "checks.json").read_text())
    single_tok = (Q.ntok_part == 1).to_numpy()
    submask = {"all": np.ones(N, bool), "single": single_tok, "multi": ~single_tok}

    def acc_ci(col, mask):
        corr = (Q[col].to_numpy() == cls).astype(float)
        return ci(bp.means(corr, mask))

    rowsB = []
    for k in ("crossfit", "insample"):
        for sname, mk in submask.items():
            c = cosB[(cosB.fitting == k) & (cosB.subset == sname)]
            a = acc_ci(f"pred__{k}__pre20__s20", mk)
            nn = acc_ci(f"pred__{k}__natural__s20", mk)
            rowsB.append({"fitting": k, "subset": sname, "n_prompts": int(mk.sum()),
                          "n_T_verbs": int(Q.verb[mk & (cls == 1)].nunique()),
                          "n_I_verbs": int(Q.verb[mk & (cls == 0)].nunique()),
                          "cos_edit_mean": c.cos_edit.mean(), "cos_edit_min": c.cos_edit.min(),
                          "cos_edit_max": c.cos_edit.max(), "cos_read_mean": c.cos_read.mean(),
                          "cos_read_min": c.cos_read.min(), "cos_read_max": c.cos_read.max(),
                          "probe_pre20": a[0], "probe_pre20_lo": a[1], "probe_pre20_hi": a[2],
                          "probe_natural20": nn[0], "probe_natural20_lo": nn[1], "probe_natural20_hi": nn[2]})
    Bdf = pd.DataFrame(rowsB)
    Bdf.to_csv(out / "leace_learning_curve_site20_split.csv", index=False)

    rowsC = []
    for s in range(12, 24):
        r = {"site": s}
        for k in ("insample", "crossfit"):
            for st in ("scrub_0_12", "natural"):
                e = acc_ci(f"pred__{k}__{st}__s{s}", submask["all"])
                r.update({f"{k}__{st}": e[0], f"{k}__{st}_lo": e[1], f"{k}__{st}_hi": e[2]})
        rowsC.append(r)
    C = pd.DataFrame(rowsC)
    C.to_csv(out / "leace_learning_curve_downstream_probes.csv", index=False)

    # ---------------------------------------------------------------- checks and flags
    flags = []
    fr = chkA.get("full_vs_insample_max_abs", {})
    if not fr or max(fr.values()) >= 1e-4:
        flags.append(f"Full fit does not reproduce the in-sample run (max |Δ| log P(by) {fr}).")
    if chkA.get("natural_vs_reference_max_abs", 1) >= 1e-4:
        flags.append("Natural log P(by) differs from the in-sample run.")
    pp = chkA.get("probe_pred_vs_probe_cv", {})
    if not pp or abs(pp["probe_cv"] - pp["probe_pred"]) > 0.002:
        flags.append(f"probe_pred differs from probe_cv: {pp}.")
    ranks = sorted(diag.erased_rank.unique().tolist())
    if ranks != [1]:
        flags.append(f"Eraser ranks {ranks} (all must be 1).")
    lpr = chkP["lp_by_reproduction"]
    if max(max(v["scrub_0_12_max_abs"], v["natural_max_abs"]) for v in lpr.values()) >= 1e-4:
        flags.append(f"Saved erasers do not reproduce log P(by): {lpr}.")
    rf = chkP["site20_refit_vs_saved_abs_cos"]
    worst = min(min(v["single_edit"], v["scrub_edit"]) for v in rf.values())
    if worst <= 0.999:
        flags.append(f"Site-20 refit on all verbs does not reproduce the saved erasers (min |cos| {worst:.4f}).")
    for k in ("insample", "crossfit"):
        m = chkP[f"probe_reproduction_{k}"]["max_abs"]
        if m > 0.005:
            flags.append(f"Stored {k} probe accuracies not reproduced (max |Δ| {m:.4f}).")

    # ---------------------------------------------------------------- report
    L = ["# LEACE learning curve and downstream re-derivation (Pythia-1.4B, exploratory)", "",
         "Rules, committed before the runs: `reports/pythia/leace_learning_curve_rules.md`. Scripts: "
         "`scripts/run_leace_learning_curve.py`, `scripts/run_leace_postscrub.py` (GPU), "
         "`scripts/analyze_leace_learning_curve.py` (this report). Numbers only.", "",
         "## Flags", ""]
    L += [f"- **{x}**" for x in flags] if flags else ["- None: every check below passes."]
    L += ["", "## Checks", "",
          f"- Fit on all 126 pairs through the learning-curve code vs the in-sample run, per-prompt log P(by): max |Δ| "
          + ", ".join(f"{k} {v:.2g}" for k, v in fr.items()) + ".",
          f"- Natural log P(by) vs the in-sample run: max |Δ| {chkA.get('natural_vs_reference_max_abs', float('nan')):.2g}.",
          f"- Probe function (per-prompt predictions) vs `probe_cv` on the first subset's natural site-8 split: "
          f"{pp.get('probe_pred', float('nan')):.4f} vs {pp.get('probe_cv', float('nan')):.4f}.",
          f"- Erasers: rank {ranks} over {diag.fit_id.nunique()} fits; top singular value of W Σ_xz per n "
          "(single 8 / scrub sites 0–12): " + "; ".join(
              f"{lab} {g.sv_top.min():.3f}–{g.sv_top.max():.3f}" for lab, g in
              diag.assign(lab=diag.fit_id.str.extract(r'^(n\d+|loo|full)')[0]).groupby("lab")) + ".",
          f"- LOO: {len(fl)} fits, every pair held out exactly once; every prompt covered once.",
          "- Saved erasers loaded for Parts B / C reproduce each run's per-prompt log P(by) of scrub 0–12: " +
          ", ".join(f"{k} {v['scrub_0_12_max_abs']:.2g} (natural {v['natural_max_abs']:.2g})" for k, v in lpr.items()) + ".",
          "- Site-20 refit on all verbs vs the saved erasers, edit-direction |cos|: " +
          ", ".join(f"{k} single {v['single_edit']:.5f} / scrub {v['scrub_edit']:.5f}" for k, v in rf.items()) + ".",
          "- Stored probe accuracies (natural at sites 12–23 and erased site 12) reproduced with the same splits: max |Δ| " +
          ", ".join(f"{k} {chkP[f'probe_reproduction_{k}']['max_abs']:.4f}" for k in ("insample", "crossfit")) + ".", ""]

    L += ["## Part A: learning curve", "",
          "Erasers fit on n training pairs (both prompts, all 125 contexts) and evaluated on the held-out pairs (all "
          "other pairs). n = 25, 50, 75, 100: 5 random subsets each; the value is the mean over the 5 subsets with a "
          "95% CI (bootstrap, subsets fixed), and the range over subsets in parentheses. LOO: 126 fits on 125 pairs, "
          "each evaluated on its held-out pair, pooled over all items. References: the 5-fold cross-fitted run and the "
          "in-sample run (fit on all 126 pairs, evaluated on the same items).", "",
          "### Share of the natural *by* gap remaining (held-out pairs)", ""]
    trs = []
    for r in rowsA:
        cells = [r["n"], r["n_held_pairs"]]
        for c, _ in CONDS:
            t = (r[f"share__{c}"], r[f"share__{c}_lo"], r[f"share__{c}_hi"])
            rng = f" ({r[f'share__{c}_min']:.2f}–{r[f'share__{c}_max']:.2f})" if f"share__{c}_min" in r else ""
            cells += [f3(t) + rng, f"{r[f'kl__{c}']:.3f}"]
        trs.append(cells)
    for lab, r in refs.items():
        cells = [lab, "—" if "in-sample" in lab else "~25"]
        for c, _ in CONDS:
            cells += [f3(r[c][:3]), f"{r[c][3]:.3f}"]
        trs.append(cells)
    L += table(["n training pairs", "held-out pairs per fit", "single 8: share remaining", "single 8: KL",
                "scrub 0–12: share remaining", "scrub 0–12: KL"], trs) + [""]

    L += ["### Held-out linear probe accuracy (class, T vs I) after erasure", "",
          "n-subsets: the held-out pairs of each fit are dealt into 5 groups; train on 4, test on the 5th "
          "(the eraser saw none of these verbs). **LOO and in-sample rows are leaky** (rules): the erased states the "
          "probe trains on were erased by erasers that saw the probe's test pairs, which pushes accuracy toward or "
          "below chance; they are not comparable to the n-subset rows. The cross-fitted reference uses nested "
          "within-fold splits (not leaky).", ""]
    trs = []
    for r in rowsA:
        cells = [r["n"] + (" (leaky)" if r["n"].startswith("LOO") else "")]
        for c, _ in PCOLS:
            k = c.replace("pred__", "acc__")
            rng = f" ({r[f'{k}_min']:.2f}–{r[f'{k}_max']:.2f})" if f"{k}_min" in r else ""
            cells.append(f3((r[k], r[f"{k}_lo"], r[f"{k}_hi"]), 3) + rng)
        trs.append(cells)
    for lab, r in refs.items():
        trs.append([lab + (" (leaky)" if "in-sample" in lab else "")] + [f"{r['probes'][c]:.3f}" for c, _ in PCOLS])
    L += table(["n training pairs"] + [lab for _, lab in PCOLS], trs) + [""]

    L += ["## Part B: site 20, split by verb token count", "",
          "Subset by the prompt's participle: single-token or multi-token (verb level). Erasers at site 20 refit on "
          "the subset: single-site on natural states, post-scrub on states after the saved all-verb erasers 0..19. "
          "|cos| of edit and read directions (cross-fitted: mean over the 5 folds [min, max]). Probe: linear, trained "
          "on all verbs as in the necessity runs, on site-20 states after scrubbing 0–19 (before eraser 20); accuracy "
          "over the subset's test prompts. Cross-fitted probes: nested within-fold splits; in-sample probes: "
          "verb-held-out split (leaky).", ""]
    trs = []
    for r in rowsB:
        mm = lambda a: f"{r[a + '_mean']:.3f}" + (f" [{r[a + '_min']:.3f}, {r[a + '_max']:.3f}]"
                                                  if r["fitting"] == "crossfit" else "")
        trs.append([r["fitting"].replace("crossfit", "cross-fitted").replace("insample", "in-sample (leaky probe)"),
                    r["subset"], f"{r['n_T_verbs']} / {r['n_I_verbs']}", str(r["n_prompts"]), mm("cos_edit"),
                    mm("cos_read"), f3((r["probe_pre20"], r["probe_pre20_lo"], r["probe_pre20_hi"]), 3),
                    f3((r["probe_natural20"], r["probe_natural20_lo"], r["probe_natural20_hi"]), 3)])
    L += table(["Fitting", "Subset", "T / I verbs", "prompts", "|cos| edit", "|cos| read",
                "probe acc after scrub 0–19", "probe acc, natural"], trs) + [""]

    sh_in = gaps.loc["insample_scrub_0_12"]
    sh_cf = gaps.loc["scrub_0_12"]
    L += ["## Part C: class probes downstream of scrub 0–12", "",
          f"In-sample scrub 0–12 leaves {f3((sh_in.share_remaining, sh_in.share_remaining_lo, sh_in.share_remaining_hi))} "
          f"of the *by* gap; cross-fitted scrub 0–12 leaves "
          f"{f3((sh_cf.share_remaining, sh_cf.share_remaining_lo, sh_cf.share_remaining_hi))}. States at sites 13–23 "
          "after scrub 0–12 (no erasure there); site 12 is the last erased site (for the in-sample split it is the "
          "leakage gauge). Linear probe, verb-held-out: in-sample with the in-sample run's 5-group split (leaky), "
          "cross-fitted with nested within-fold splits (leak-free supplement). Accuracy with 95% CIs.", ""]
    trs = []
    for r in rowsC:
        trs.append([str(r["site"]) + (" (erased)" if r["site"] == 12 else "")] +
                   [f3((r[f"{k}__{st}"], r[f"{k}__{st}_lo"], r[f"{k}__{st}_hi"]), 3)
                    for k in ("insample", "crossfit") for st in ("scrub_0_12", "natural")])
    L += table(["Site", "in-sample: after scrub 0–12", "in-sample split: natural", "cross-fitted: after scrub 0–12",
                "cross-fitted split: natural"], trs) + [""]

    L += ["## Files", "",
          "- `reports/pythia/leace_learning_curve_summary.csv`: Part A per n (share remaining, KL, probe accuracy, "
          "CIs, subset ranges).",
          "- `reports/pythia/leace_learning_curve_per_fit.csv`: Part A per subset fit.",
          "- `reports/pythia/leace_learning_curve_site20_split.csv`: Part B.",
          "- `reports/pythia/leace_learning_curve_downstream_probes.csv`: Part C.",
          f"- `{args.lc.lstrip('./')}/` and `{args.postscrub.lstrip('./')}/`: fits, fit diagnostics, checks, metadata, "
          "cosines; per-prompt files (`held.csv.gz`, `loo_probes.csv.gz`, `preds.csv.gz`, `prompts.csv.gz`) and "
          "`states.npz` not in git (on TSUBAME under `lli_leace/`).", ""]
    (out / "leace_learning_curve.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out / 'leace_learning_curve.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lc", default="results/pythia/leace_learning_curve")
    ap.add_argument("--postscrub", default="results/pythia/leace_postscrub")
    ap.add_argument("--gaps", default="reports/pythia/leace_necessity_gaps.csv")
    ap.add_argument("--probes", default="reports/pythia/leace_necessity_probes.csv")
    ap.add_argument("--report-dir", default="reports/pythia")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
