#!/usr/bin/env python3
"""Report for run_leace_necessity.py: the *by* gap under LEACE erasure of verb class at the verb position.

Gap = mean log P(" by" | T passive) - mean log P(" by" | I passive) over the task-1 items (analyze_das_prep.py),
all bands pooled (item-weighted) and per band. Share remaining = erased gap / natural gap (natural = same-batch fp32
recomputation). 95% CIs: the task-1 two-way cluster bootstrap (verb pairs within verb band x contexts within context
band; analyze_das_prep.Boot), the same draws for every condition. Fold assignment and erasers are fixed (not
resampled).

In-sample rows (if --insample exists: run_leace_necessity.py --insample, erasers fit on all verbs and evaluated on
the same items) are merged as conditions `insample_<condition>`; their natural readout must match the cross-fitted
run's.

Writes reports/pythia/leace_necessity.md and reports/pythia/leace_necessity_*.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_das_prep import BANDS, Boot, load_readout, table
from build_das_prep_prompts import pid

DAS_SITE = 8
NEC_TOP = 12  # sites 0..~12: the necessity-relevant range (later sites sit next to the readout)
# Round 5, B5 (reports/round5/b5_necessity.md), frame `hasbeen`, all 64 pairs: share of the natural good - bad
# " by" gap removed, with 95% CIs. Different items (B7 items, "The N has been V"), fp32.
REFS = (("d -> I mean (DAS site 8 coordinate)", "d_I", 0.15, 0.09, 0.31),
        ("top-50 switch neurons -> I mean", "n50_I", 0.09, 0.04, 0.15),
        ("both (d + top-50 neurons)", "c50", 0.19, 0.13, 0.34))


def ci(draws):
    """Estimate and percentile CI; draws whose resample holds no item of the subset (0/0) are dropped."""
    return draws[0], *np.nanpercentile(draws[1:], [2.5, 97.5])


def fmt(t, k=2, sign=True):
    s = "+" if sign else ""
    return f"{t[0]:{s}.{k}f} [{t[1]:{s}.{k}f}, {t[2]:{s}.{k}f}]"


def run(args):
    res_dir = Path(args.results)
    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(res_dir / "items.csv.gz")
    meta = json.loads((res_dir / "meta.json").read_text())
    checks = json.loads((res_dir / "checks.json").read_text())
    probes = pd.read_csv(res_dir / "probes.csv")
    diag = pd.read_csv(res_dir / "fit_diagnostics.csv")
    folds = pd.read_csv(res_dir / "pairs_folds.csv")
    assert not meta["args"].get("insample", False), "--results must be the cross-fitted run"
    diag["fitting"] = "cross_fit"
    erasers = {"cross_fit": np.load(res_dir / "erasers.npz") if (res_dir / "erasers.npz").exists() else None}
    prev = None
    if args.previous and Path(args.previous).exists():  # an earlier run of the same cross-fitted job
        R = pd.read_csv(args.previous)
        assert (R[["item", "side"]].to_numpy() == P[["item", "side"]].to_numpy()).all()
        common = [c for c in R.columns if (c.startswith("lp_by__") or c.startswith("kl__")) and c in P.columns]
        prev = {"n_columns": len(common), "max_abs": float(max((R[c] - P[c]).abs().max() for c in common)),
                "sha256": hashlib.sha256(Path(args.previous).read_bytes()).hexdigest()}
    ins_dir = Path(args.insample) if args.insample else None
    ins = None
    if ins_dir and (ins_dir / "items.csv.gz").exists():
        Q = pd.read_csv(ins_dir / "items.csv.gz")
        assert (Q[["item", "side", "prompt"]].to_numpy() == P[["item", "side", "prompt"]].to_numpy()).all()
        ins = {"meta": json.loads((ins_dir / "meta.json").read_text()),
               "checks": json.loads((ins_dir / "checks.json").read_text()),
               "natural_vs_crossfit_max_abs": float((Q.lp_by__natural - P.lp_by__natural).abs().max()),
               "items_sha256": hashlib.sha256((ins_dir / "items.csv.gz").read_bytes()).hexdigest()}
        assert ins["meta"]["args"]["insample"]
        # the in-sample rows are compared against P's natural baseline (and Q's KL used Q's), so they must agree
        assert np.isfinite(Q.filter(like="lp_by__").to_numpy()).all()
        assert ins["natural_vs_crossfit_max_abs"] < 1e-3, ins["natural_vs_crossfit_max_abs"]
        for c in Q.columns:
            if (c.startswith("lp_by__") or c.startswith("kl__")) and not c.endswith("__natural"):
                pre, cond = c.split("__", 1)
                P[f"{pre}__insample_{cond}"] = Q[c].to_numpy()
        qp = pd.read_csv(ins_dir / "probes.csv")
        qp["condition"] = "insample_" + qp.condition
        probes = pd.concat([probes, qp], ignore_index=True)
        qd = pd.read_csv(ins_dir / "fit_diagnostics.csv")
        qd["fitting"] = "insample"
        diag = pd.concat([diag, qd], ignore_index=True)
        erasers["insample"] = np.load(ins_dir / "erasers.npz") if (ins_dir / "erasers.npz").exists() else None
    conds = [c[len("lp_by__"):] for c in P.columns if c.startswith("lp_by__")]
    erased = [c for c in conds if c != "natural"]

    # item table: one row per (verb pair, context), good (T) and bad (I) side by side
    key = ["item", "verb_pair", "verb_band", "context_id", "context_band"]
    G = P[P.side == "good"].set_index("item")
    B = P[P.side == "bad"].set_index("item")
    assert (G.index == B.index).all()
    it = G[key[1:]].reset_index()
    for c in conds:
        it[f"gap__{c}"] = (G[f"lp_by__{c}"] - B[f"lp_by__{c}"]).to_numpy()
    it["pattern"] = ("T" + G.ntok_part.clip(upper=2).astype(str).str.replace("2", "2+") + "-I"
                     + B.ntok_part.clip(upper=2).astype(str).str.replace("2", "2+")).to_numpy()
    rng = np.random.default_rng(args.seed)
    boot = Boot(it, args.n_boot, rng)

    # stored bf16 task-1 readout, for the reproduction check
    repro = {}
    if args.readout and Path(args.readout).exists():
        ro = load_readout(args.readout)
        ids = P.prompt.map(lambda p: pid("passive", p))
        stored = ro.reindex(ids)["lp[ by]"].to_numpy()
        if not np.isnan(stored).any():
            diff = np.abs(stored - P["lp_by__natural"].to_numpy())
            Gs, Bs = stored[(P.side == "good").to_numpy()], stored[(P.side == "bad").to_numpy()]
            gs = Gs[np.argsort(P[P.side == "good"].item.to_numpy())] - Bs[np.argsort(P[P.side == "bad"].item.to_numpy())]
            repro = {"median_abs": float(np.median(diff)), "p99_abs": float(np.quantile(diff, .99)),
                     "max_abs": float(diff.max()), "gap_stored_bf16": ci(boot.means(gs, np.ones(len(it), bool))),
                     "readout_sha256": hashlib.sha256(Path(args.readout).read_bytes()).hexdigest()}

    allm = np.ones(len(it), bool)
    masks = {"all": allm, **{b: (it.verb_band == b).to_numpy() for b in BANDS},
             **{f"pattern {p}": (it.pattern == p).to_numpy() for p in sorted(it.pattern.unique())}}
    nat = {m: boot.means(it["gap__natural"].to_numpy(), mk) for m, mk in masks.items()}
    rows = []
    for c in conds:
        y = it[f"gap__{c}"].to_numpy()
        kl = P[f"kl__{c}"] if c != "natural" else pd.Series(np.zeros(len(P)))
        for m, mk in masks.items():
            g = boot.means(y, mk)
            r = {"condition": c, "subset": m, "n_items": int(mk.sum()), "n_pairs": int(it.verb_pair[mk].nunique()),
                 "n_valid_draws": int(np.isfinite(g[1:]).sum())}
            for name, dr in (("gap", g), ("delta_gap", g - nat[m]), ("share_remaining", g / nat[m])):
                e, lo, hi = ci(dr)
                r.update({name: e, f"{name}_lo": lo, f"{name}_hi": hi})
            if m == "all":
                r.update({"kl_mean": float(kl.mean()), "kl_median": float(kl.median()),
                          "kl_mean_T": float(kl[(P.side == "good").to_numpy()].mean()) if c != "natural" else 0.0,
                          "kl_mean_I": float(kl[(P.side == "bad").to_numpy()].mean()) if c != "natural" else 0.0})
            rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(out / "leace_necessity_gaps.csv", index=False)
    probes.to_csv(out / "leace_necessity_probes.csv", index=False)
    diag.to_csv(out / "leace_necessity_fit_diagnostics.csv", index=False)
    refs = pd.DataFrame([{"reference": n, "b5_condition": k, "share_removed": e, "share_removed_lo": lo,
                          "share_removed_hi": hi, "share_remaining": 1 - e, "share_remaining_lo": 1 - hi,
                          "share_remaining_hi": 1 - lo} for n, k, e, lo, hi in REFS])
    refs.to_csv(out / "leace_necessity_references.csv", index=False)

    # probe summary at the erased site(s)
    def probe_at(cond, kind):
        p = probes[(probes.probe == kind)]
        pre = "insample_" if cond.startswith("insample_") else ""
        base = cond[len(pre):]
        if base.startswith("single_"):
            s = int(base.split("_")[1])
            return f"{p[(p.condition == pre + 'single') & (p.site == s)].acc.iloc[0]:.3f}"
        if base == "natural":
            return ""
        if base.startswith("scrub_0_") and base != "scrub_0_8":
            # scrub 0..k: its states at sites <= k are scrub_all's (each erased state depends on erasers 0..s only)
            k = int(base.split("_")[2])
            q = p[(p.condition == pre + "scrub_all") & (p.site <= k)]
        else:
            q = p[(p.condition == cond) & p.erased_here]
        return f"{q.acc.min():.3f}–{q.acc.max():.3f}"

    def nat_probe(site, kind):
        p = probes[(probes.probe == kind) & (probes.condition == "natural") & (probes.site == site)]
        return f"{p.acc.iloc[0]:.3f}"

    A = S[S.subset == "all"].set_index("condition")
    nd = diag[diag.condition.isin(["single", "scrub"]) & (diag.fitting == "cross_fit")]
    n_fit = nd.groupby("fold").n_fit.first()
    n_verbs = nd.groupby("fold").n_verbs.first()
    d = 2048
    L = ["# LEACE necessity of verb class at the verb position (Pythia-1.4B)", "",
         "Scripts: `scripts/run_leace_necessity.py` (GPU), `scripts/analyze_leace_necessity.py` (this report). "
         f"Run: {meta['minutes']:.0f} min, concept-erasure {meta['concept_erasure_version']}, fp32 (TF32 off).", "",
         "## Setup", "",
         f"- Items: task 1 of `analyze_das_prep.py` (curated `was` prefixes + participle, other verbs' contexts): "
         f"{meta['n_pairs']} verb pairs x 125 contexts each (all of the {it.context_id.nunique()} curated contexts but the "
         f"pair's own) = {meta['n_items']:,} items, a T and an "
         f"I prompt each ({meta['n_prompts']:,} prompts). Readout: log P(\" by\") at the participle's last token.",
         "- Erasure position: the participle's last subtoken (= last prompt token, the DAS Runner's anchor; the "
         "readout is at the same position). Applied to T and I prompts alike.",
         "- Sites: DAS convention, site L = residual stream entering layer L (site 0 = embedding output), "
         f"L = 0..23. DAS site = {DAS_SITE}.",
         "- Concept: class T (1) vs I (0). LeaceFitter with package defaults (affine, shrinkage on, svd_tol 0.01, "
         "trace constraint on), float64 statistics.",
         "- Split: the 29-pair DAS split covers only 8 of these 126 pairs, so the DAS fold procedure "
         "(`run_das_round2.folds`: pairs shuffled within strata, dealt round-robin; 5 folds, seed 0) is applied to "
         "the 126 pairs, strata = verb band x token pattern (both participles single-token or not). Every verb "
         "is in one pair only. Erasers for fold f are fit on the other four folds and applied to fold f "
         "(cross-fitted), so every item is erased by an eraser that never saw its verb.",
         f"- Fitting samples per eraser (per fold, the same at every site): n = {int(n_fit.min()):,}–{int(n_fit.max()):,} "
         f"verb-position states ({int(n_verbs.min())}–{int(n_verbs.max())} distinct verbs) vs d = {d} (n/d = "
         f"{n_fit.min() / d:.1f}–{n_fit.max() / d:.1f}); shrinkage left on (package default).",
         "- Conditions: `single_L` = LEACE at site L only, fit on natural states. `scrub_all` = concept scrubbing "
         "at sites 0..23: per fold, eraser L is fit on the training items' site-L states computed with that "
         "fold's erasers 0..L-1 applied (and L not applied). `scrub_0_8` = erasers 0..8 of `scrub_all` "
         "(the sequential fit makes them identical).",
         "- In-sample conditions (`insample_*`, rows marked \"in-sample\"): the same three kinds of erasure, but one "
         "eraser set fit on all items (every verb, both prompts) and applied to those same items, so the evaluated "
         "items are the fitting data. Their probes: the same nested procedure with one outer fold, i.e. a 5-group "
         "verb-held-out split over all pairs (the eraser has seen the test verbs), and a linear probe trained and "
         "scored on all items (fit-set accuracy).",
         "- Probes (nested, verb-held-out): for each outer fold g, the states of fold g's items after erasure by "
         "fold g's erasers (fit on the other folds); the fold's pairs are dealt into 5 inner groups, a probe is "
         "trained on 4 groups and tested on the 5th, accuracy pooled over all test items; chance = 0.5. So neither "
         "the probe's training nor its test items were seen by the eraser, and probe train and test verbs differ. "
         "Natural states use the same splits. Linear = L2 logistic regression; MLP = one hidden layer of 256 ReLU "
         "units.",
         "- KL = mean over prompts (T and I) of KL(P_natural || P_erased) over the full vocabulary at the verb "
         "position, natural recomputed in the same batch.",
         f"- CIs: task-1 two-way cluster bootstrap ({args.n_boot} draws, seed {args.seed}), shared across conditions.",
         ""]
    L += ["## Checks", ""]
    rp = checks["scrub_replay"]
    L += [f"- Identity hook at site {DAS_SITE} vs same-batch natural: max |Δ log P| = {checks['self_patch_max_abs']:.2g}.",
          f"- Non-anchor positions after the erasure hooks (scrub_all, all sites): max |Δh| = "
          f"{checks['outside_anchor_max_abs']:.2g}.",
          f"- single_L: erased states inside the model vs offline erasure of the natural states: max |Δ| = "
          f"{checks['single_offline_max_abs']:.2g}.",
          f"- Same-batch natural vs the natural capture pass (different batching): max |Δ log P(by)| = "
          f"{checks['natural_vs_pass_max_abs']:.2g}.",
          f"- Scrub replay (training items, all 24 erasers applied): pre-erasure mean at each site vs the mean "
          f"each eraser was fit on, max |Δ| = {rp['mean_maxabs']:.2g}; cross-covariance relative difference "
          f"max {rp['xcov_rel']:.2g}; post-erasure / pre-erasure cross-covariance norm, max over sites and folds "
          f"{rp['post_xcov_resid_max']:.2g}.",
          f"- Erasers: top singular value of W Σ_xz min {nd.sv_top.min():.3f} (svd_tol 0.01: "
          f"{int((nd.sv_top <= 0.01).sum())} fits below it, i.e. no erasure); erased rank "
          f"{sorted(nd.erased_rank.unique().tolist())}; trace-constraint branch used in "
          f"{int(nd.trace_branch.sum())} fits; training cross-covariance residual max {nd.train_xcov_resid.max():.2g}."]
    hd = diag[(diag.condition == "single") & (diag.fitting == "cross_fit")].assign(r=lambda x: x.held_meandiff_post / x.held_meandiff_pre)
    hs = diag[(diag.condition == "scrub_all_heldout") & (diag.fitting == "cross_fit")].assign(r=lambda x: x.held_meandiff_post / x.held_meandiff_pre)
    L += [f"- Held-out class-mean difference, erased / natural norm: single_L {hd.r.min():.3f}–{hd.r.max():.3f}; "
          f"scrub_all {hs.r.min():.3f}–{hs.r.max():.3f} (range over sites and folds)."]
    if repro:
        L += [f"- Natural fp32 log P(by) vs the stored bf16 task-1 readout (`results/passive_das_prep/"
              f"pythia14b_readout.csv`, sha256 {repro['readout_sha256'][:12]}…): median |Δ| {repro['median_abs']:.3f}, "
              f"99th pct {repro['p99_abs']:.3f}, max {repro['max_abs']:.3f}; gap from the stored readout "
              f"{fmt(repro['gap_stored_bf16'])} vs fp32 {fmt(ci(nat['all']))}."]
    if prev:
        L += [f"- This cross-fitted run vs the first run of the same job (per-prompt log P(by) and KL, "
              f"{prev['n_columns']} shared columns; first run's items.csv.gz sha256 {prev['sha256'][:12]}…): "
              f"max |Δ| = {prev['max_abs']:.2g}."]
    if ins:
        ic, irp = ins["checks"], ins["checks"]["scrub_replay"]
        idg = diag[diag.fitting == "insample"]
        ifit = idg[idg.condition.isin(["single", "scrub"])]
        ih = {c: idg[idg.condition == c].assign(r=lambda x: x.held_meandiff_post / x.held_meandiff_pre)
              for c in ("single", "scrub_all_heldout", "scrub_0_8_heldout")}
        ih["scrub_0_8_heldout"] = ih["scrub_0_8_heldout"][ih["scrub_0_8_heldout"].site <= DAS_SITE]
        L += [f"- In-sample run ({ins['meta']['minutes']:.0f} min): n = {int(ifit.n_fit.min()):,} fitting states per "
              f"eraser ({int(ifit.n_verbs.min())} verbs; n/d = {ifit.n_fit.min() / d:.1f}); top singular value min "
              f"{ifit.sv_top.min():.3f}, erased rank {sorted(ifit.erased_rank.unique().tolist())}, trace branch "
              f"used in {int(ifit.trace_branch.sum())} fits; same checks as above: identity hook "
              f"{ic['self_patch_max_abs']:.2g}, non-anchor {ic['outside_anchor_max_abs']:.2g}, in-model vs offline "
              f"{ic['single_offline_max_abs']:.2g}, scrub replay mean {irp['mean_maxabs']:.2g} / cross-covariance "
              f"{irp['xcov_rel']:.2g}; natural log P(by) vs the cross-fitted run's max |Δ| "
              f"{ins['natural_vs_crossfit_max_abs']:.2g}.",
              f"- In-sample, on the evaluated items (= the fitting items): cross-covariance with class after / before "
              f"erasure, max over scrub sites {irp['post_xcov_resid_max']:.2g}, single_L max "
              f"{ifit[ifit.condition == 'single'].train_xcov_resid.max():.2g}; class-mean difference after / before "
              f"erasure: single_L max {ih['single'].r.max():.2g}, scrub 0–23 max {ih['scrub_all_heldout'].r.max():.2g}, "
              f"scrub 0–{DAS_SITE} (erased sites) max {ih['scrub_0_8_heldout'].r.max():.2g}."]
    L += [""]

    er = probes[probes.erased_here & (probes.condition != "natural") & ~probes.condition.str.startswith("insample_")]
    L += ["## Flags", ""]
    for kind in ("linear", "mlp"):
        q = er[er.probe == kind]
        L += [f"- Held-out {kind.replace('mlp', 'MLP')} probe after erasure is not at chance: {q.acc.min():.3f}–{q.acc.max():.3f} over every "
              f"erased site and condition (chance 0.5; natural {probes[(probes.condition == 'natural') & (probes.probe == kind)].acc.min():.3f}–"
              f"{probes[(probes.condition == 'natural') & (probes.probe == kind)].acc.max():.3f})." if q.acc.min() > 0.55 else
              f"- Held-out {kind.replace('mlp', 'MLP')} probe after erasure: {q.acc.min():.3f}–{q.acc.max():.3f} over the erased sites (chance 0.5)."]
    L += [f"- Readout proximity: *by* is predicted at the verb position itself, so late sites sit next to the "
          f"readout, and erasure there partly removes the readout computation itself rather than only verb-class "
          f"information the readout uses. Sites 0–~{NEC_TOP} are the necessity-relevant range; rows for later sites "
          f"(including the post hoc single-site minimum) do not test necessity in that sense."]
    if ins:
        ie = probes[probes.erased_here & probes.condition.str.startswith("insample_")
                    & (probes.condition != "insample_natural")]
        for kind, what in (("linear_fitset", "linear probe trained and scored on all (= fitting) items"),
                           ("linear", "verb-held-out linear probe (eraser saw the test verbs)"),
                           ("mlp", "verb-held-out MLP probe (eraser saw the test verbs)")):
            q = ie[ie.probe == kind]
            L += [f"- In-sample erasure, {what}: {q.acc.min():.3f}–{q.acc.max():.3f} over every erased site and "
                  f"condition (chance 0.5)."]
        L += ["- In-sample verb-held-out probes can fall below chance: the eraser zeroes the class cross-covariance "
              "over all verbs together, so whatever class signal remains in the probe's training verbs points the "
              "opposite way in its test verbs (the same effect as the pooled cross-fold probe in the first smoke "
              "test)."]
    L += [""]

    def row(c, label):
        r = A.loc[c]
        return [label, fmt((r.gap, r.gap_lo, r.gap_hi)), fmt((r.share_remaining, r.share_remaining_lo,
                                                             r.share_remaining_hi), sign=False),
                f"{r.kl_mean:.3f}", probe_at(c, "linear"), probe_at(c, "mlp")]

    hdr = ["Condition", "*by* gap (nats)", "share of natural gap remaining", "KL (nats)", "linear probe",
           "MLP probe"]
    L += ["## Main table", "",
          "Probe columns: held-out accuracy at the erased site (scrubbing: range over the erased sites); for the "
          "in-sample rows the test verbs are held out from the probe but not from the eraser. Rows for sites "
          f"after ~{NEC_TOP} (scrub 0–23, the post hoc single-site minimum) include erasure next to the readout "
          "(see Flags). The B5 "
          "reference rows are from `reports/round5/b5_necessity.md` (frame `hasbeen`, all 64 pairs; different "
          "items and frame: B7 items, \"The N has been V\"), converted to share remaining = 1 − share removed.", ""]
    trs = [row("natural", "natural")]
    trs += [row("scrub_all", "scrub, sites 0–23")]
    if f"scrub_0_{NEC_TOP}" in A.index:
        trs += [row(f"scrub_0_{NEC_TOP}", f"scrub, sites 0–{NEC_TOP}")]
    trs += [row("scrub_0_8", f"scrub, sites 0–{DAS_SITE}")]
    best = A.loc[[c for c in erased if c.startswith("single_")]].share_remaining.idxmin()
    trs += [row(f"single_{DAS_SITE}", f"single site {DAS_SITE} (DAS site)"),
            row(best, f"single site {best.split('_')[1]}: lowest of 24 (post hoc; pointwise CI)")]
    if ins:
        ibest = A.loc[[c for c in erased if c.startswith("insample_single_")]].share_remaining.idxmin()
        trs += [row("insample_scrub_all", "in-sample: scrub, sites 0–23")]
        if f"insample_scrub_0_{NEC_TOP}" in A.index:
            trs += [row(f"insample_scrub_0_{NEC_TOP}", f"in-sample: scrub, sites 0–{NEC_TOP}")]
        trs += [row("insample_scrub_0_8", f"in-sample: scrub, sites 0–{DAS_SITE}"),
                row(f"insample_single_{DAS_SITE}", f"in-sample: single site {DAS_SITE} (DAS site)"),
                row(ibest, f"in-sample: single site {ibest.split('_')[-1]}: lowest of 24 (post hoc; pointwise CI)")]
    for n, k, e, lo, hi in REFS:
        trs.append([f"ref (B5): {n}", "", f"{1 - e:.2f} [{1 - hi:.2f}, {1 - lo:.2f}]", "", "", ""])
    L += table(hdr, trs) + [""]

    L += ["## Single-site LEACE, every site", "",
          "Natural linear / MLP probe accuracy at the same site in the last two columns.", ""]
    trs = []
    for s in range(24):
        c = f"single_{s}"
        trs.append(row(c, str(s)) + [nat_probe(s, "linear"), nat_probe(s, "mlp")])
    L += table(["Site", "*by* gap", "share remaining", "KL", "linear probe", "MLP probe", "natural linear",
                "natural MLP"], trs) + [""]

    L += ["## Probe accuracy under scrubbing, by site", "",
          f"`scrub_0_8` is erased at sites 0–{DAS_SITE} only; its later sites are unerased (states downstream of "
          "the erasure). Probes stop at site 23, the input to the last layer; the state after the last layer "
          "(which feeds the readout) is not probed.", ""]
    trs = []
    pv = probes.pivot_table(index="site", columns=["condition", "probe"], values="acc")
    for s in range(24):
        trs.append([str(s)] + [f"{pv.loc[s, (c, k)]:.3f}" for c in ("natural", "scrub_all", "scrub_0_8")
                               for k in ("linear", "mlp")])
    L += table(["Site", "natural lin", "natural MLP", "scrub 0–23 lin", "scrub 0–23 MLP", "scrub 0–8 lin",
                "scrub 0–8 MLP"], trs) + [""]

    if ins:
        L += ["## In-sample erasure, every site", "",
              "Erasers fit on all items and evaluated on them. Single-site rows: `insample_single_L`; probe columns as "
              "in the main table, plus the fit-set linear probe. Last three columns: probe accuracy under in-sample "
              "scrubbing 0–23 at that site.", ""]
        pv = probes.pivot_table(index="site", columns=["condition", "probe"], values="acc")
        trs = []
        for s in range(24):
            trs.append(row(f"insample_single_{s}", str(s)) +
                       [f"{pv.loc[s, ('insample_single', 'linear_fitset')]:.3f}"] +
                       [f"{pv.loc[s, ('insample_scrub_all', k)]:.3f}" for k in ("linear_fitset", "linear", "mlp")])
        L += table(["Site", "*by* gap", "share remaining", "KL", "linear probe", "MLP probe", "fit-set linear",
                    "scrub 0–23 fit-set lin", "scrub 0–23 lin", "scrub 0–23 MLP"], trs) + [""]

    def scrub_name(k, pre=""):
        return pre + ("scrub_all" if k == 23 else f"scrub_0_{k}")

    curve_runs = [(lab, pre) for lab, pre in (("cross-fitted", ""), ("in-sample", "insample_"))
                  if all(scrub_name(k, pre) in A.index for k in range(24))]
    if curve_runs:
        L += ["## Cumulative scrub curve", "",
              "Scrubbing sites 0..k (the first k + 1 erasers of the sequential fit), share of the natural gap "
              f"remaining and KL, k = 0..23. Sites after ~{NEC_TOP} sit next to the readout (see Flags).", ""]
        trs, cur = [], []
        for k in range(24):
            r = [str(k)]
            for lab, pre in curve_runs:
                q = A.loc[scrub_name(k, pre)]
                r += [fmt((q.share_remaining, q.share_remaining_lo, q.share_remaining_hi), sign=False),
                      f"{q.kl_mean:.3f}"]
                cur.append({"run": lab, "k": k, "condition": scrub_name(k, pre), "share_remaining": q.share_remaining,
                            "share_remaining_lo": q.share_remaining_lo, "share_remaining_hi": q.share_remaining_hi,
                            "kl_mean": q.kl_mean})
            trs.append(r)
        L += table(["k"] + [h for lab, _ in curve_runs for h in (f"{lab}: share remaining", f"{lab}: KL")],
                   trs) + [""]
        pd.DataFrame(cur).to_csv(out / "leace_necessity_cumulative.csv", index=False)

    def cosines(npz):  # |cos| between single_L and scrub_L eraser vectors, [F, 24] per part
        res = {}
        for part in ("edit", "read"):
            a, b = npz[f"single_{part}"], npz[f"scrub_{part}"]
            res[part] = np.abs((a * b).sum(-1)) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
        return res

    cos = {k: cosines(v) for k, v in erasers.items() if v is not None}
    if cos:
        ref = int(best.split("_")[1])
        L += ["## Single-site vs sequential-scrub eraser at the same site", "",
              "Each rank-1 eraser maps x to x − ⟨r, x − b⟩ e. |cos| between the single-site eraser at site L (fit on "
              "natural site-L states) and the scrub eraser at site L (fit on site-L states after erasure at sites "
              "0..L−1), for the edit direction e (what is removed) and the read direction r (what it is keyed on). "
              "Cross-fitted: mean over the 5 folds [min, max]; in-sample: one eraser pair per site.", ""]
        sent = []
        for lab, c in cos.items():
            e, r_ = c["edit"][:, ref], c["read"][:, ref]
            sent.append(f"{lab.replace('_', '-')}: edit {e.mean():.3f}" + (f" [{e.min():.3f}, {e.max():.3f}]" if len(e) > 1 else "")
                        + f", read {r_.mean():.3f}" + (f" [{r_.min():.3f}, {r_.max():.3f}]" if len(r_) > 1 else ""))
        sc = A.loc[scrub_name(ref)] if scrub_name(ref) in A.index else None
        L += [f"Site {ref} (the post hoc single-site minimum; share remaining single {A.loc[best].share_remaining:.2f}"
              + (f", scrub 0–{ref} {sc.share_remaining:.2f}" if sc is not None else "")
              + f", scrub 0–23 {A.loc['scrub_all'].share_remaining:.2f}): |cos| " + "; ".join(sent) + ".", ""]
        if sc is not None and "cross_fit" in cos:
            e = cos["cross_fit"]["edit"][:, ref].mean()
            L += [f"So the scrub eraser at site {ref} removes a direction nearly orthogonal to the one the single-site "
                  f"eraser removes (edit |cos| {e:.3f}): after erasure at sites 0..{ref - 1}, the class-correlated "
                  f"direction in the fitting states at site {ref} is a different one, and the single-site direction is "
                  f"left in place. Of the gap between scrub 0–23 ({A.loc['scrub_all'].share_remaining:.2f}) and single "
                  f"site {ref} ({A.loc[best].share_remaining:.2f}), the cumulative curve puts "
                  f"{sc.share_remaining - A.loc[best].share_remaining:.2f} before site {ref + 1} (scrub 0–{ref} "
                  f"{sc.share_remaining:.2f}) and {A.loc['scrub_all'].share_remaining - sc.share_remaining:+.2f} in "
                  f"sites {ref + 1}–23.", ""]
        trs, crow = [], []
        for s_ in range(24):
            r = [str(s_)]
            for lab, c in cos.items():
                for part in ("edit", "read"):
                    v = c[part][:, s_]
                    r.append(f"{v.mean():.3f}" + (f" [{v.min():.3f}, {v.max():.3f}]" if len(v) > 1 else ""))
                    crow.append({"fitting": lab, "site": s_, "direction": part, "abs_cos_mean": v.mean(),
                                 "abs_cos_min": v.min(), "abs_cos_max": v.max()})
            trs.append(r)
        L += table(["Site"] + [f"{lab.replace('_', '-')} {part}" for lab in cos for part in ("edit", "read")], trs) + [""]
        pd.DataFrame(crow).to_csv(out / "leace_necessity_eraser_cosines.csv", index=False)

    L += ["## By verb band and token pattern", "",
          "Share of the natural gap remaining (natural gap in parentheses). Pattern: participle token counts "
          "of T and I (2+ = two or more).", ""]
    subsets = [m for m in masks if m != "all"]
    show = ["scrub_all", "scrub_0_8", f"single_{DAS_SITE}", best] + (["insample_scrub_all", "insample_scrub_0_8"]
                                                                      if ins else [])
    trs = []
    for m in subsets:
        n0 = ci(nat[m])
        sub = S[S.subset == m].set_index("condition")
        trs.append([m, str(int(sub.loc["natural"].n_pairs)), fmt(n0)] +
                   [fmt(tuple(sub.loc[c, ["share_remaining", "share_remaining_lo", "share_remaining_hi"]]), sign=False)
                    for c in show])
    L += table(["Subset", "pairs", "natural gap", "scrub 0–23", f"scrub 0–{DAS_SITE}", f"single {DAS_SITE}",
                f"single {best.split('_')[1]} (post hoc)"] +
                (["in-sample scrub 0–23", f"in-sample scrub 0–{DAS_SITE}"] if ins else []), trs)
    nv = S[S.subset.isin(subsets)].groupby("subset").n_valid_draws.min()
    short = nv[nv < args.n_boot]
    L += ["", "Bootstrap draws that resample no pair of a subset are dropped: " +
          (", ".join(f"{k} {v}/{args.n_boot} valid" for k, v in short.items()) if len(short) else "none.")] + [""]
    L += ["## Files", "",
          "- `reports/pythia/leace_necessity_gaps.csv`: gap, Δ gap, share remaining (with CIs) per condition and "
          "subset; KL (all-items rows).",
          "- `reports/pythia/leace_necessity_probes.csv`: probe accuracy per condition, site and probe type.",
          "- `reports/pythia/leace_necessity_fit_diagnostics.csv`: per eraser: n, distinct verbs, top singular "
          "value, rank, training cross-covariance residual, held-out class-mean difference before/after.",
          "- `reports/pythia/leace_necessity_references.csv`: the B5 reference numbers.",
          "- `reports/pythia/leace_necessity_cumulative.csv`: the cumulative scrub curve.",
          "- `reports/pythia/leace_necessity_eraser_cosines.csv`: single-site vs scrub eraser |cos| per site.",
          f"- `{args.results.lstrip('./')}/`: per-prompt log P(by) and KL per condition (`items.csv.gz`), fold assignment, "
          f"checks, run metadata. `items.csv.gz` is not in git (sha256 {hashlib.sha256((res_dir / 'items.csv.gz').read_bytes()).hexdigest()}); "
          "it is on TSUBAME at `/gs/fs/tga-sip_arase/tyrone/lli_leace/results/pythia/leace_necessity/` (the first run's file next to it as `items_run1.csv.gz`)."]
    if ins:
        L += [f"- `{args.insample.lstrip('./')}/`: the same files for the in-sample run (`items.csv.gz` not in git, "
              f"sha256 {ins['items_sha256']}; on TSUBAME under `lli_leace/{args.insample.lstrip('./')}/`)."]
    L += [""]
    (out / "leace_necessity.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out / 'leace_necessity.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/pythia/leace_necessity")
    ap.add_argument("--insample", default="results/pythia/leace_insample")
    ap.add_argument("--previous", default="", help="items.csv.gz of an earlier run of the cross-fitted job")
    ap.add_argument("--readout", default="results/passive_das_prep/pythia14b_readout.csv")
    ap.add_argument("--report-dir", default="reports/pythia")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
