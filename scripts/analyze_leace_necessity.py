#!/usr/bin/env python3
"""Report for run_leace_necessity.py: the *by* gap under LEACE erasure of verb class at the verb position.

Gap = mean log P(" by" | T passive) - mean log P(" by" | I passive) over the task-1 items (analyze_das_prep.py),
all bands pooled (item-weighted) and per band. Share remaining = erased gap / natural gap (natural = same-batch fp32
recomputation). 95% CIs: the task-1 two-way cluster bootstrap (verb pairs within verb band x contexts within context
band; analyze_das_prep.Boot), the same draws for every condition. Fold assignment and erasers are fixed (not
resampled).

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
        if cond.startswith("single_"):
            s = int(cond.split("_")[1])
            return f"{p[(p.condition == 'single') & (p.site == s)].acc.iloc[0]:.3f}"
        if cond == "natural":
            return ""
        q = p[(p.condition == cond) & p.erased_here]
        return f"{q.acc.min():.3f}–{q.acc.max():.3f}"

    def nat_probe(site, kind):
        p = probes[(probes.probe == kind) & (probes.condition == "natural") & (probes.site == site)]
        return f"{p.acc.iloc[0]:.3f}"

    A = S[S.subset == "all"].set_index("condition")
    nd = diag[diag.condition.isin(["single", "scrub"])]
    n_fit = nd.groupby("fold").n_fit.first()
    n_verbs = nd.groupby("fold").n_verbs.first()
    d = 2048
    L = ["# LEACE necessity of verb class at the verb position (Pythia-1.4B)", "",
         "Scripts: `scripts/run_leace_necessity.py` (GPU), `scripts/analyze_leace_necessity.py` (this report). "
         f"Run: {meta['minutes']:.0f} min, concept-erasure {meta['concept_erasure_version']}, fp32 (TF32 off).", "",
         "## Setup", "",
         f"- Items: task 1 of `analyze_das_prep.py` (curated `was` prefixes + participle, other verbs' contexts): "
         f"{meta['n_pairs']} verb pairs x {it.context_id.nunique()} contexts = {meta['n_items']:,} items, a T and an "
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
         f"- Fitting samples per eraser (per fold, the same at every site): n = {n_fit.min():,}–{n_fit.max():,} "
         f"verb-position states ({n_verbs.min()}–{n_verbs.max()} distinct verbs) vs d = {d} (n/d = "
         f"{n_fit.min() / d:.1f}–{n_fit.max() / d:.1f}); shrinkage left on (package default).",
         "- Conditions: `single_L` = LEACE at site L only, fit on natural states. `scrub_all` = concept scrubbing "
         "at sites 0..23: per fold, eraser L is fit on the training items' site-L states computed with that "
         "fold's erasers 0..L-1 applied (and L not applied). `scrub_0_8` = erasers 0..8 of `scrub_all` "
         "(the sequential fit makes them identical).",
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
    hd = diag[diag.condition == "single"].assign(r=lambda x: x.held_meandiff_post / x.held_meandiff_pre)
    hs = diag[diag.condition == "scrub_all_heldout"].assign(r=lambda x: x.held_meandiff_post / x.held_meandiff_pre)
    L += [f"- Held-out class-mean difference, erased / natural norm: single_L {hd.r.min():.3f}–{hd.r.max():.3f}; "
          f"scrub_all {hs.r.min():.3f}–{hs.r.max():.3f} (range over sites and folds)."]
    if repro:
        L += [f"- Natural fp32 log P(by) vs the stored bf16 task-1 readout (`results/passive_das_prep/"
              f"pythia14b_readout.csv`, sha256 {repro['readout_sha256'][:12]}…): median |Δ| {repro['median_abs']:.3f}, "
              f"99th pct {repro['p99_abs']:.3f}, max {repro['max_abs']:.3f}; gap from the stored readout "
              f"{fmt(repro['gap_stored_bf16'])} vs fp32 {fmt(ci(nat['all']))}."]
    L += [""]

    def row(c, label):
        r = A.loc[c]
        return [label, fmt((r.gap, r.gap_lo, r.gap_hi)), fmt((r.share_remaining, r.share_remaining_lo,
                                                             r.share_remaining_hi), sign=False),
                f"{r.kl_mean:.3f}", probe_at(c, "linear"), probe_at(c, "mlp")]

    hdr = ["Condition", "*by* gap (nats)", "share of natural gap remaining", "KL (nats)", "linear probe",
           "MLP probe"]
    L += ["## Main table", "",
          "Probe columns: held-out accuracy at the erased site (scrubbing: range over the erased sites). The B5 "
          "reference rows are from `reports/round5/b5_necessity.md` (frame `hasbeen`, all 64 pairs; different "
          "items and frame: B7 items, \"The N has been V\"), converted to share remaining = 1 − share removed.", ""]
    trs = [row("natural", "natural")]
    trs += [row("scrub_all", "scrub, sites 0–23"), row("scrub_0_8", f"scrub, sites 0–{DAS_SITE}")]
    best = A.loc[[c for c in erased if c.startswith("single_")]].share_remaining.idxmin()
    trs += [row(f"single_{DAS_SITE}", f"single site {DAS_SITE} (DAS site)"),
            row(best, f"single site {best.split('_')[1]}: lowest of 24 (post hoc; pointwise CI)")]
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

    L += ["## By verb band and token pattern", "",
          "Share of the natural gap remaining (natural gap in parentheses). Pattern: participle token counts "
          "of T and I (2+ = two or more).", ""]
    subsets = [m for m in masks if m != "all"]
    show = ["scrub_all", "scrub_0_8", f"single_{DAS_SITE}", best]
    trs = []
    for m in subsets:
        n0 = ci(nat[m])
        sub = S[S.subset == m].set_index("condition")
        trs.append([m, str(int(sub.loc["natural"].n_pairs)), fmt(n0)] +
                   [fmt(tuple(sub.loc[c, ["share_remaining", "share_remaining_lo", "share_remaining_hi"]]), sign=False)
                    for c in show])
    L += table(["Subset", "pairs", "natural gap", "scrub 0–23", f"scrub 0–{DAS_SITE}", f"single {DAS_SITE}",
                f"single {best.split('_')[1]} (post hoc)"], trs)
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
          f"- `{args.results}/`: per-prompt log P(by) and KL per condition (`items.csv.gz`), fold assignment, "
          "checks, run metadata.", ""]
    (out / "leace_necessity.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out / 'leace_necessity.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/pythia/leace_necessity")
    ap.add_argument("--readout", default="results/passive_das_prep/pythia14b_readout.csv")
    ap.add_argument("--report-dir", default="reports/pythia")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
