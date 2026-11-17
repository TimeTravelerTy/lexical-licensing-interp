"""Exploratory: passive *by* classes with the margin scaled to *by*'s own natural gap.

The declared passive test classes D(by) against delta_s = 0.2 x v's active O effect at the site, a margin set by the
object readout. For the L readouts of OR / TC / filler-gap the declared margins are 0.2 x the readout's own natural
good - bad gap. This script applies that convention to *by*: delta_g = 0.2 x the unpatched good - bad gap in log
P(" by") on the same pairs (stored as condition nat_gap in the passive test). Same classifier
(analyze_passive_test.classify / outcome) on stored CIs; O keeps its declared class. Nothing is re-run.
Not a declared reading: chosen after seeing that OLMo's T1 failed with D(by) close to its natural gap.

The natural gap is itself uncertain (wide CI), so the class is also given with delta from the gap's upper 95% bound
(conservative).

Sources: passive test transfer_summary.csv (v, all sites; nat_gap); split edit summary_passive.csv (on, umd);
Pythia u_md from das_direction_check. Writes reports/gapscaled_by_classes.md, results/gapscaled_by_classes.csv.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_passive_test import classify, outcome  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["est", "lo95", "hi95", "lo90", "hi90"]
SHORT = {"surface (object next)": "surface", "abstract (takes an object)": "abstract", "mixed": "mixed",
         "nothing moves (active-specific)": "nothing", "unresolved": "unresolved"}
PT = {"pythia": "results/das_round2/passive_test", "gemma": "results/gemma/das/transfer/passive_test",
      "olmo": "results/olmo/das/transfer/passive_test"}
RULE = {"pythia": (8, 12), "gemma": (5, 8), "olmo": (9, 11)}


def ci(row, pre=""):
    return {k: float(row[pre + k]) for k in KEYS}


def main():
    recs, gaps, checks = [], {}, {}
    for m, p in PT.items():
        T = pd.read_csv(ROOT / p / "transfer_summary.csv")
        T = T[(T.population == "primary (plain)") & (T.band == "all")]
        g = T[(T.condition == "nat_gap") & (T.readout == "by")]
        assert g.est.nunique() == 1, "natural gap should not depend on the site"
        gap = ci(g.iloc[0])
        gaps[m] = gap
        dg, dg_hi = 0.2 * gap["est"], 0.2 * gap["hi95"]
        deltas = {int(k): v for k, v in json.loads((ROOT / p / "deltas.json").read_text()).items()}
        cond_rows = []
        for s in sorted(T.site.unique()):
            r = lambda ro: ci(T[(T.condition == "D_bad") & (T.readout == ro) & (T.site == s)].iloc[0])
            cond_rows.append(("v", int(s), r("O"), r("by"), deltas[int(s)]))
        # split-edit conditions (same pairs as the passive test; checked below via the full condition)
        se = ROOT / f"results/{m}/split_edit/summary_passive.csv"
        P = pd.read_csv(se)
        for s in sorted(P.site.unique()):  # the split edit stores delta_s at every site it ran
            deltas.setdefault(int(s), float(P[P.site == s].delta_s.iloc[0]))
        for s in sorted(P.site.unique()):
            q = lambda ro: P[(P.site == s) & (P.readout == ro)].iloc[0]
            # v at sites the passive test did not run: the split edit's full condition (same pairs, see checks)
            conds = ("on", "umd") if (T.site == s).any() else ("full", "on", "umd")
            for c in conds:
                if f"{c}_est" in P:
                    cond_rows.append(("v" if c == "full" else c, int(s), ci(q("O"), c + "_"), ci(q("by"), c + "_"),
                                      float(q("O").delta_s)))
            full = ci(q("by"), "full_")
            v = T[(T.condition == "D_bad") & (T.readout == "by") & (T.site == s)]
            if len(v):
                checks[f"{m}_site{s}_full_minus_v_by"] = round(full["est"] - float(v.est.iloc[0]), 6)
        if m == "pythia":
            D = pd.read_csv(ROOT / "results/pythia/das_direction_check/summary_passive_patching.csv")
            for s in sorted(D.site.unique()):
                q = lambda ro: D[(D.site == s) & (D.readout == ro)].iloc[0]
                cond_rows.append(("umd", int(s), ci(q("O"), "umd_"), ci(q("by"), "umd_"), deltas.get(int(s), float("nan"))))
        for cond, s, o, by, ds in cond_rows:
            cO = classify(o, ds) if ds == ds else "n/a"
            cb_decl = classify(by, ds) if ds == ds else "n/a"
            cb, cb_hi = classify(by, dg), classify(by, dg_hi)
            recs.append(dict(model=m, condition=cond, site=s, delta_s=ds, delta_g=dg, delta_g_hi95=dg_hi,
                             by_est=by["est"], by_lo95=by["lo95"], by_hi95=by["hi95"], by_over_gap=by["est"] / gap["est"],
                             class_O=cO, class_by_declared=cb_decl, class_by_gap=cb, class_by_gap_hi95=cb_hi,
                             outcome_declared=SHORT.get(outcome(cO, cb_decl), "n/a"),
                             outcome_gap=SHORT.get(outcome(cO, cb), "n/a")))
    df = pd.DataFrame(recs).sort_values(["model", "condition", "site"])
    df.to_csv(ROOT / "results/gapscaled_by_classes.csv", index=False)

    def t1(m, cond, col):
        sub = df[(df.model == m) & (df.condition == cond) & (df.site.isin(RULE[m]))]
        return bool(len(sub) == 2 and (sub[col] == "RISE").all()) if len(sub) == 2 else None

    L = ["# Passive *by* with a gap-scaled margin (exploratory)", "",
         "Generated by `scripts/gapscaled_by_classes.py` from stored summaries; nothing re-run. The declared passive "
         "test classes D(by) against δ_s = 0.2 × v's active O effect. Here D(by) is classed against δ_g = 0.2 × "
         "the natural good − bad gap in log P(\" by\") on the same pairs, the convention the declared rules already "
         "use for L readouts. O keeps its declared class. Same classifier. **Exploratory**: chosen after OLMo's T1 "
         "failed with D(by) near its natural gap. The gap's CI is wide, so δ from its upper 95% bound is also "
         "shown.", "",
         "## Natural gap and margins", "", "| Model | natural by gap | δ_g | δ_g (gap hi95) | δ_s at rule sites |",
         "|---|---|---:|---:|---|"]
    for m in PT:
        g = gaps[m]
        ds = df[(df.model == m) & (df.condition == "v") & df.site.isin(RULE[m])].set_index("site").delta_s
        L.append(f"| {m} | {g['est']:+.2f} [{g['lo95']:+.2f}, {g['hi95']:+.2f}] | {0.2 * g['est']:.2f} | "
                 f"{0.2 * g['hi95']:.2f} | " + ", ".join(f"{s}: {ds[s]:.2f}" for s in RULE[m]) + " |")
    L += ["", "## T1 (by RISE at both rule sites)", "", "| Model | Rule sites | declared | gap-scaled | gap-scaled (hi95) |",
          "|---|---|---|---|---|"]
    for m in PT:
        for c in ("v", "on", "umd"):
            a, b, h = t1(m, c, "class_by_declared"), t1(m, c, "class_by_gap"), t1(m, c, "class_by_gap_hi95")
            if b is None:
                continue
            L.append(f"| {m} {c} | {RULE[m][0]} / {RULE[m][1]} | {a} | {b} | {h} |")
    for m in PT:
        sub = df[df.model == m]
        L += ["", f"## {m}", "", "| Cond | Site | δ_s | D(by) [95%] | D(by) / gap | declared by | gap by | gap by (hi95) "
              "| O | outcome declared → gap |", "|---|---:|---:|---|---:|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            L.append(f"| {r.condition} | {r.site} | {r.delta_s:.2f} | {r.by_est:+.2f} [{r.by_lo95:+.2f}, {r.by_hi95:+.2f}] | "
                     f"{r.by_over_gap:.2f} | {r.class_by_declared} | {r.class_by_gap} | {r.class_by_gap_hi95} | "
                     f"{r.class_O} | {r.outcome_declared} → {r.outcome_gap} |")
    L += ["", "## Checks", "", "v rows at sites outside the passive test come from the split edit's full condition. Split-edit full condition minus stored passive-test v D(by) at shared sites "
          "(same pairs if 0): " + ", ".join(f"{k} {v:+.2g}" for k, v in checks.items()), ""]
    (ROOT / "reports/gapscaled_by_classes.md").write_text("\n".join(L))
    print("\n".join(L[:30]))
    print(checks)


if __name__ == "__main__":
    main()
