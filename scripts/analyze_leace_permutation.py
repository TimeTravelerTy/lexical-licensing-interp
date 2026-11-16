#!/usr/bin/env python3
"""Report for run_leace_permutation.py: in-sample LEACE with permuted verb labels vs the true-label in-sample numbers.

Per permutation: share of the natural *by* gap remaining = mean item gap (T - I log P(by)) after erasure / natural mean
item gap (all items, item-weighted, as the point estimates of analyze_leace_necessity.py), and mean KL over prompts.
Summary over permutations: mean, 5th percentile (numpy linear interpolation; with 20 permutations it lies between the
lowest and second-lowest), min, max. True-label rows: the in-sample run (reports/pythia/leace_necessity_gaps.csv,
95% cluster-bootstrap CIs).

Flag: true-label erasure counts as well below the permutations when its 95% CI upper bound is below the permutation
5th percentile; otherwise the condition is flagged.

Writes reports/pythia/leace_permutation.md, leace_permutation_per_perm.csv, leace_permutation_summary.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_das_prep import table

CONDS = [("scrub_0_8", "scrub, sites 0–8"), ("scrub_0_12", "scrub, sites 0–12"), ("scrub_all", "scrub, sites 0–23"),
         ("single_8", "single site 8")]


def run(args):
    parts = sorted(Path(args.results).glob("part*"))
    assert parts, f"no part* directories under {args.results}"
    frames, metas, checks, diags, labels = [], [], {}, [], []
    for p in parts:
        frames.append(pd.read_csv(p / "items.csv.gz"))
        metas.append(json.loads((p / "meta.json").read_text()))
        checks[p.name] = json.loads((p / "checks.json").read_text())
        diags.append(pd.read_csv(p / "fit_diagnostics.csv"))
        if (p / "labels.csv").stat().st_size > 1:
            labels.append(pd.read_csv(p / "labels.csv"))
    P = frames[0]
    for F in frames[1:]:
        assert (F.prompt.to_numpy() == P.prompt.to_numpy()).all()
        assert np.abs(F.lp_by__natural.to_numpy() - P.lp_by__natural.to_numpy()).max() < 1e-3
        P = pd.concat([P, F[[c for c in F.columns if "__p" in c]]], axis=1)
    diag = pd.concat(diags, ignore_index=True)
    lab = pd.concat(labels, ignore_index=True) if labels else pd.DataFrame()
    perms = sorted({c.split("__")[1] for c in P.columns if c.startswith("lp_by__p")}, key=lambda x: int(x[1:]))
    assert len(perms) == len(set(perms))

    G = P[P.side == "good"].set_index("item").sort_index()
    B = P[P.side == "bad"].set_index("item").sort_index()
    nat = float((G.lp_by__natural - B.lp_by__natural).mean())
    rows = []
    for name in perms + (["true"] if "lp_by__true__scrub_all" in P.columns else []):
        for c, _ in CONDS:
            gap = float((G[f"lp_by__{name}__{c}"] - B[f"lp_by__{name}__{c}"]).mean())
            rows.append({"labels": name, "condition": c, "gap": gap, "share_remaining": gap / nat,
                         "kl_mean": float(P[f"kl__{name}__{c}"].mean())})
    per = pd.DataFrame(rows)
    out = Path(args.report_dir)
    per.to_csv(out / "leace_permutation_per_perm.csv", index=False)

    T = pd.read_csv(args.true_gaps)
    T = T[T.subset == "all"].set_index("condition")
    summ, trs, flags = [], [], []
    for c, label in CONDS:
        q = per[(per.labels != "true") & (per.condition == c)]
        t = T.loc[f"insample_{c}"]
        p5 = float(np.percentile(q.share_remaining, 5))
        well = t.share_remaining_hi < p5
        r = {"condition": c, "n_perms": len(q), "true_share": t.share_remaining, "true_share_lo": t.share_remaining_lo,
             "true_share_hi": t.share_remaining_hi, "true_kl": t.kl_mean, "perm_share_mean": q.share_remaining.mean(),
             "perm_share_p5": p5, "perm_share_min": q.share_remaining.min(), "perm_share_max": q.share_remaining.max(),
             "perm_kl_mean": q.kl_mean.mean(), "perm_kl_min": q.kl_mean.min(), "perm_kl_max": q.kl_mean.max(),
             "n_perms_share_le_true": int((q.share_remaining <= t.share_remaining).sum()),
             "true_well_below_p5": bool(well)}
        summ.append(r)
        trs.append([label, f"{t.share_remaining:.2f} [{t.share_remaining_lo:.2f}, {t.share_remaining_hi:.2f}]",
                    f"{r['perm_share_mean']:.2f}", f"{p5:.2f}", f"{r['perm_share_min']:.2f}–{r['perm_share_max']:.2f}",
                    f"{r['n_perms_share_le_true']}/{len(q)}", f"{t.kl_mean:.3f}",
                    f"{r['perm_kl_mean']:.3f} ({r['perm_kl_min']:.3f}–{r['perm_kl_max']:.3f})",
                    "yes" if well else "**NO**"])
        if not well:
            flags.append(f"- **{label}: true-label erasure is not well below the permutations.** True share remaining "
                         f"{t.share_remaining:.2f} [{t.share_remaining_lo:.2f}, {t.share_remaining_hi:.2f}] vs "
                         f"permutation 5th percentile {p5:.2f} (mean {r['perm_share_mean']:.2f}; "
                         f"{r['n_perms_share_le_true']}/{len(q)} permutations at or below the true share).")
    S = pd.DataFrame(summ)
    S.to_csv(out / "leace_permutation_summary.csv", index=False)

    corr = {k: v for ch in checks.values() for k, v in ch["label_corr"].items() if k != "true"}
    tv = {k: ch["true_vs_insample_max_abs"] for k, ch in checks.items() if "true_vs_insample_max_abs" in ch}
    nv_cls = P.groupby("cls").verb.nunique()
    L = ["# Permutation control for in-sample LEACE (Pythia-1.4B)", "",
         "Scripts: `scripts/run_leace_permutation.py` (GPU), `scripts/analyze_leace_permutation.py` (this report). "
         f"{len(perms)} permutations in {len(parts)} jobs ({sum(m['minutes'] for m in metas):.0f} GPU-min).", "",
         "## Setup", "",
         f"- Same items ({P.verb_pair.nunique()} pairs x {P.context_id.nunique() - 1} contexts, T and I prompt each), position (verb's last token = readout "
         "position), sites, fitting and rank (1) as the in-sample conditions of `reports/pythia/leace_necessity.md`: "
         "one eraser set fit on all prompts and applied to the same prompts.",
         "- Concept: instead of verb class, a permuted verb label. Per permutation, exactly half of the "
         f"{nv_cls[1]} T verbs and half of the {nv_cls[0]} I verbs are flipped at random; every prompt takes its verb's label. Labels are balanced; "
         f"correlation with the true class over prompts: max |r| = {max(abs(v) for v in corr.values()):.3g} "
         "(< 0.05 required).",
         "- Conditions: sequential scrubbing over sites 0–8, 0–12 and 0–23 (prefixes of one sequential fit per "
         "permutation) and single-site LEACE at site 8, each fit with the permuted labels.",
         "- Measured on the true items: share of the natural *by* gap (T − I log P(by)) remaining, and mean KL "
         "(natural ‖ erased) at the verb position over all prompts. True-label rows: the in-sample run, with "
         "95% cluster-bootstrap CIs.",
         "- Permutation summary: mean, 5th percentile (linear interpolation; with 20 permutations it lies between "
         "the lowest and second-lowest), range. \"Well below\" = the true-label CI upper bound is below the "
         "permutation 5th percentile.", "",
         "## Checks", ""]
    for k, v in tv.items():
        L += [f"- True labels run through the permutation code ({k}) vs the in-sample run, per-prompt log P(by): max "
              f"|Δ| " + ", ".join(f"{c} {x:.2g}" for c, x in v.items()) + "."]
    nv = [ch.get("natural_vs_reference_max_abs") for ch in checks.values() if "natural_vs_reference_max_abs" in ch]
    if nv:
        L += [f"- Natural log P(by) vs the in-sample run: max |Δ| {max(nv):.2g} over jobs."]
    pd_ = diag[diag.labels != "true"]
    L += [f"- Permuted-label erasers: rank {sorted(pd_.erased_rank.unique().tolist())}; top singular value of "
          f"W Σ_xz {pd_.sv_top.min():.3f}–{pd_.sv_top.max():.3f} (true labels {diag[diag.labels == 'true'].sv_top.min():.3f}–"
          f"{diag[diag.labels == 'true'].sv_top.max():.3f}; svd_tol 0.01: {int((pd_.sv_top <= 0.01).sum())} fits "
          f"below it); training cross-covariance residual max {pd_.train_xcov_resid.max():.2g}.", ""]
    L += ["## Flags", ""] + (flags if flags else
                             ["- None: for every condition the true-label CI upper bound is below the permutation "
                              "5th percentile."]) + [""]
    L += ["## Share of the natural *by* gap remaining", "",
          f"Natural gap +{nat:.2f} nats.", ""]
    L += table(["Condition", "true labels (95% CI)", "perm mean", "perm 5th pct", "perm range",
                "perms ≤ true", "KL true", "KL perm mean (range)", "true well below perm 5th pct"], trs) + [""]
    L += ["## Files", "",
          "- `reports/pythia/leace_permutation_summary.csv`: the table above.",
          "- `reports/pythia/leace_permutation_per_perm.csv`: gap, share remaining and KL per permutation and "
          "condition (and the true labels rerun through this code).",
          f"- `{args.results.lstrip('./')}/part*/`: labels per verb, fit diagnostics, checks, metadata; per-prompt "
          "`items.csv.gz` not in git (sha256 " +
          ", ".join(f"{p.name} {hashlib.sha256((p / 'items.csv.gz').read_bytes()).hexdigest()[:16]}…" for p in parts) +
          "; on TSUBAME under `lli_leace/" + args.results.lstrip("./") + "/`).", ""]
    (out / "leace_permutation.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out / 'leace_permutation.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/pythia/leace_permutation")
    ap.add_argument("--true-gaps", default="reports/pythia/leace_necessity_gaps.csv")
    ap.add_argument("--report-dir", default="reports/pythia")
    run(ap.parse_args())
