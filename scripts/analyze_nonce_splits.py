#!/usr/bin/env python3
"""Gemma replication, EXPLORATORY nonce comparison (rules: reports/gemma/exploratory.md, section 9).

Headline metrics and declared decisions of the two nonce analyses, for both models, on all 80 lemmas and on the
sensitivity splits (lemma lists in results/gemma/das/nonce/subsets/):
- all: Gemma results/gemma/das/nonce/{nonce_passive,nonce_cues}; Pythia the stored results/das_round2/nonce_passive
  and results/round4/nonce_cues;
- same_ed_count (67 lemmas, same -ed token count in both tokenizers), last_bare_ed / last_stem (per model: the -ed
  form's last piece is "ed" / carries stem letters): results/gemma/das/nonce/splits/<model>_<split>/round{3,4}.
Decision rules, copied from the analyses (unchanged):
- round-3 C8 (analyze_nonce_passive): (a) context-sensitive separation: Δ_matched z lo95 > 0 and est >= 0.05;
  (b) verb-specific: Δ_balanced z, same rule; (c) raises *by* verb-specifically: Δ_balanced by lo95 > 0, est >= 0.1
  and est > Δ_balanced PREP without by;
- round-4 C8 (analyze_nonce_cues): d shift survives in the -ing / -s form: form T - I z lo95 > 0 and est >= 0.05;
  C9 primary: transitivity rel_T - rel_I, adjacent NP adj_I - pp_I (z: >= 0.05; nats: >= 0.1), reading both /
  transitivity-driven / NP-driven / neither.
Decision sites: Gemma 1, 3, 5; Pythia 6, 8. The round-3 table also lists the later mapped sites (Gemma 7 / 8 / 12 /
15 for Pythia 10 / 12 / 14 / 17).

Also every declared-decision line of the generated reports (bold yes / no / reading values), compared between
the all-lemma report and each split's report, listing every value that changes.

Writes reports/gemma/nonce/summary.md and results/gemma/das/nonce/summary.csv.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

M = {"Gemma-3-1B": {"decl": (1, 3, 5), "late": (7, 8, 12, 15), "tag": "gemma",
                    "all": ("results/gemma/das/nonce/nonce_passive/nonce_summary.csv",
                            "results/gemma/das/nonce/nonce_cues/nonce_cues_summary.csv")},
     "Pythia-1.4B": {"decl": (6, 8), "late": (10, 12, 14, 17), "tag": "pythia",
                     "all": ("results/das_round2/nonce_passive/nonce_summary.csv",
                             "results/round4/nonce_cues/nonce_cues_summary.csv")}}
SPLITS = ("all", "same_ed_count", "last_bare_ed", "last_stem")
MIN = {"z": 0.05, "by": 0.1, "O": 0.1}


def paths(model, split, root):
    if split == "all":
        return M[model]["all"]
    t = M[model]["tag"]
    s = split if split == "same_ed_count" else f"{t}_{split}"
    d = Path(root) / f"{t}_{s}"
    return str(d / "round3" / "nonce_summary.csv"), str(d / "round4" / "nonce_cues_summary.csv")


REPORTS = {"gemma": ("reports/gemma/nonce/round3_nonce.md", "reports/gemma/nonce/round4_nonce_cues.md"),
           "pythia": ("reports/passive_das_prep/round3_nonce.md", "reports/round4/c_nonce_cues.md")}


def decisions(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("- ") and "**" in line:
            for i, v in enumerate(re.findall(r"\*\*([^*]+)\*\*", line)):
                out[(line.split(":")[0], i)] = (v, line)
    return out


def up(r, m):
    return bool(r.lo95 > 0 and r.est >= MIN["z" if m.startswith("z") else m])


def main(a):
    rows = []
    for model, cfg in M.items():
        for split in SPLITS:
            p3, p4 = paths(model, split, a.splits_root)
            r3, r4 = pd.read_csv(p3), pd.read_csv(p4)
            g3 = lambda pt, c, m: r3[(r3.probe == pt) & (r3.contrast == c) & (r3.measure == m)].iloc[0]
            g4 = lambda an, c, m: r4[(r4.analysis == an) & (r4.contrast == c) & (r4.measure == m)].iloc[0]
            n = int(g4("C8", "ed: T - I", "by").n_lemmas)
            base = {"model": model, "split": split, "n_lemmas": n}
            bb, bp = g3("passive", "balanced AB - BA", "by"), g3("passive", "balanced AB - BA", "prep_noby")
            c_by = bool(bb.lo95 > 0 and bb.est >= 0.1 and bb.est > bp.est)
            for s in cfg["decl"] + cfg["late"]:
                dm, db = g3("passive", "matched T - I", f"z{s}"), g3("passive", "balanced AB - BA", f"z{s}")
                r = {**base, "site": s, "decision_site": s in cfg["decl"],
                     "r3_matched_z": dm.est, "r3_matched_z_lo": dm.lo95, "r3_matched_z_hi": dm.hi95,
                     "r3_balanced_z": db.est, "r3_balanced_z_lo": db.lo95, "r3_balanced_z_hi": db.hi95,
                     "r3_a_context": up(dm, "z"), "r3_b_verb": up(db, "z")}
                if s in cfg["decl"]:
                    t, np_ = g4("C9", "transitivity, relative: rel_T - rel_I", f"z{s}"), g4("C9", "adjacent NP: adj_I - pp_I", f"z{s}")
                    r.update({"r4_ed_z": g4("C8", "ed: T - I", f"z{s}").est, "r4_ing_z": g4("C8", "ing: T - I", f"z{s}").est,
                              "r4_s_z": g4("C8", "s: T - I", f"z{s}").est,
                              "r4_ing_survives": up(g4("C8", "ing: T - I", f"z{s}"), "z"),
                              "r4_s_survives": up(g4("C8", "s: T - I", f"z{s}"), "z"),
                              "r4_c9_trans_z": t.est, "r4_c9_np_z": np_.est,
                              "r4_c9_reading": "both" if up(t, "z") and up(np_, "z") else "transitivity-driven" if up(t, "z")
                              else "NP-driven" if up(np_, "z") else "neither"})
                rows.append(r)
            # site-free log-prob rows
            lp = {**base, "site": -1}
            for c, k in (("matched T - I", "matched"), ("balanced AB - BA", "balanced")):
                for m in ("by", "prep_noby", "O"):
                    x = g3("passive", c, m)
                    lp.update({f"r3_{k}_{m}": x.est, f"r3_{k}_{m}_lo": x.lo95, f"r3_{k}_{m}_hi": x.hi95})
            lp["r3_c_by"] = c_by
            t, np_ = g4("C9", "transitivity, relative: rel_T - rel_I", "by"), g4("C9", "adjacent NP: adj_I - pp_I", "by")
            lp.update({"r4_c9_trans_by": t.est, "r4_c9_np_by": np_.est,
                       "r4_c9_by_reading": "both" if up(t, "by") and up(np_, "by") else "transitivity-driven" if up(t, "by")
                       else "NP-driven" if up(np_, "by") else "neither",
                       "r4_ing_by": g4("C8", "ing: T - I", "by").est, "r4_s_by": g4("C8", "s: T - I", "by").est,
                       "r4_ing_by_survives": up(g4("C8", "ing: T - I", "by"), "by"),
                       "r4_s_by_survives": up(g4("C8", "s: T - I", "by"), "by")})
            rows.append(lp)
    R = pd.DataFrame(rows)
    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(a.csv, index=False)
    yn = lambda v: "yes" if v else "no"
    f3 = lambda e, l, h: f"{e:+.3f} [{l:+.3f}, {h:+.3f}]"
    f2 = lambda e, l, h: f"{e:+.2f} [{l:+.2f}, {h:+.2f}]"
    L = ["# Nonce verbs, Gemma-3-1B next to Pythia-1.4B, with sensitivity splits (EXPLORATORY)", "",
         "Generated by `scripts/analyze_nonce_splits.py` (sources and decision rules in its docstring; rules in "
         "`exploratory.md`, section 9). Per-lemma means, lemma bootstrap 95% CIs (2,000 draws, seed 17). z: 0 = "
         "held-out active intransitive, 1 = transitive level of each site's d. Splits: all 80 lemmas; same_ed_count = "
         "67 lemmas with the same -ed token count in both tokenizers; last_bare_ed / last_stem = the model's own "
         "-ed last piece is \"ed\" / carries stem letters.", "",
         "## Round-3 C8, passive probe: z along d", "",
         "Decision sites: Gemma 1 / 3 / 5, Pythia 6 / 8; later mapped sites listed for reference. (a) context-"
         "sensitive separation (Δ_matched), (b) verb-specific (Δ_balanced).", "",
         "| model | split | lemmas | site | Δ_matched z | (a) | Δ_balanced z | (b) |", "|---|---|---:|---:|---|---|---|---|"]
    for r in R[R.site >= 0].itertuples():
        L.append(f"| {r.model} | {r.split} | {r.n_lemmas} | {r.site}{'' if r.decision_site else ' (ref.)'} | "
                 f"{f3(r.r3_matched_z, r.r3_matched_z_lo, r.r3_matched_z_hi)} | {yn(r.r3_a_context) if r.decision_site else ''} | "
                 f"{f3(r.r3_balanced_z, r.r3_balanced_z_lo, r.r3_balanced_z_hi)} | {yn(r.r3_b_verb) if r.decision_site else ''} |")
    L += ["", "## Round-3 C8, passive probe: log-probabilities (nats)", "",
          "(c) raises *by* verb-specifically: Δ_balanced by lo95 > 0, ≥ 0.1 and above Δ_balanced PREP without by.", "",
          "| model | split | lemmas | matched T − I: by | matched: O | balanced AB − BA: by | balanced: PREP\\by | balanced: O | (c) |",
          "|---|---|---:|---|---|---|---|---|---|"]
    for r in R[R.site < 0].itertuples():
        L.append(f"| {r.model} | {r.split} | {r.n_lemmas} | {f2(r.r3_matched_by, r.r3_matched_by_lo, r.r3_matched_by_hi)} | "
                 f"{f2(r.r3_matched_O, r.r3_matched_O_lo, r.r3_matched_O_hi)} | "
                 f"{f2(r.r3_balanced_by, r.r3_balanced_by_lo, r.r3_balanced_by_hi)} | "
                 f"{f2(r.r3_balanced_prep_noby, r.r3_balanced_prep_noby_lo, r.r3_balanced_prep_noby_hi)} | "
                 f"{f2(r.r3_balanced_O, r.r3_balanced_O_lo, r.r3_balanced_O_hi)} | {yn(r.r3_c_by)} |")
    L += ["", "## Round-4 C8 / C9 at the decision sites", "",
          "C8: T − I z in the -ed / -ing / -s forms; the d shift survives in -ing / -s if T − I z lo95 > 0 and ≥ 0.05. "
          "C9 primary: transitivity (rel_T − rel_I) and adjacent NP (adj_I − pp_I).", "",
          "| model | split | lemmas | site | ed z | ing z (survives) | s z (survives) | C9 transitivity z | C9 adjacent NP z | C9 reading (z) |",
          "|---|---|---:|---:|---:|---|---|---:|---:|---|"]
    for r in R[(R.site >= 0) & R.decision_site].itertuples():
        L.append(f"| {r.model} | {r.split} | {r.n_lemmas} | {r.site} | {r.r4_ed_z:+.3f} | {r.r4_ing_z:+.3f} ({yn(r.r4_ing_survives)}) | "
                 f"{r.r4_s_z:+.3f} ({yn(r.r4_s_survives)}) | {r.r4_c9_trans_z:+.3f} | {r.r4_c9_np_z:+.3f} | {r.r4_c9_reading} |")
    L += ["", "Round-4, *by* (nats): -ing / -s T − I (survives if lo95 > 0 and ≥ 0.1); C9 primary on *by*.", "",
          "| model | split | ing: by (survives) | s: by (survives) | C9 transitivity by | C9 adjacent NP by | C9 reading (by) |",
          "|---|---|---|---|---:|---:|---|"]
    for r in R[R.site < 0].itertuples():
        L.append(f"| {r.model} | {r.split} | {r.r4_ing_by:+.2f} ({yn(r.r4_ing_by_survives)}) | {r.r4_s_by:+.2f} "
                 f"({yn(r.r4_s_by_survives)}) | {r.r4_c9_trans_by:+.2f} | {r.r4_c9_np_by:+.2f} | {r.r4_c9_by_reading} |")
    L += ["", "## Declared decisions in every split", "",
          "Each bold decision value of the generated reports' declared-decision lines, split against all lemmas.", "",
          "| model | split | round | decision values | changed |", "|---|---|---|---:|---:|"]
    changes = []
    for model, cfg in M.items():
        t = cfg["tag"]
        for split in SPLITS[1:]:
            sp = split if split == "same_ed_count" else f"{t}_{split}"
            for rnd, base in zip(("round3", "round4"), REPORTS[t]):
                A, B = decisions(base), decisions(f"{a.reports_root}/splits/{t}_{sp}_{rnd}.md")
                assert A.keys() == B.keys(), (model, split, rnd)
                ch = [k for k in A if A[k][0] != B[k][0]]
                L.append(f"| {model} | {split} | {rnd} | {len(A)} | {len(ch)} |")
                changes += [(model, split, rnd, A[k][1], B[k][1]) for k in ch]
    L += [""] + (["Changed lines (all lemmas, then the split):", ""] + [x for m_, sp_, r_, la, lb in changes for x in
                 (f"- {m_}, {sp_}, {r_}:", f"  - all: {la[2:]}", f"  - split: {lb[2:]}")] if changes else ["None."])
    Path(a.report).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits-root", default="results/gemma/das/nonce/splits")
    ap.add_argument("--csv", default="results/gemma/das/nonce/summary.csv")
    ap.add_argument("--report", default="reports/gemma/nonce/summary.md")
    ap.add_argument("--reports-root", default="reports/gemma/nonce")
    main(ap.parse_args())
