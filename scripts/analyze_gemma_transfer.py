#!/usr/bin/env python3
"""Gemma replication: transfer of the active-trained direction, Gemma-3-1B next to Pythia-1.4B by relative depth.

Reads the summaries the existing analyses write (same scripts and statistics for both models; Gemma runs under
results/gemma/das/transfer, Pythia's stored runs) and lays out each construction's headline metrics, rows sorted by
relative depth (site / number of layers: Pythia 24, Gemma 26).
- passive (by) test (run / analyze_passive_test.py): primary population (plain bad verbs), all bands, D on bad
  passive bases (transitive minus intransitive active donor) on log P(O), log P(" by"), log P("."), with the
  declared classes (RISE: 95% CI > 0 and estimate >= δ_s; NO RISE: 90% CI within ±δ_s) and the declared outcome.
- object relatives (OR) and tough constructions (TC), T − I donors (run / analyze_construction_test.py): D_bad on
  O and on L (OR: MAIN − PREP; TC: END − PREP), with classes.
- in-range patch (run / analyze_inrange_patch.py): D_in = R(t_T) − R(t_I) and its ratio to the natural good − bad
  gap, for passive " by", OR MAIN / PREP, TC END / PREP, and O.
- filler-gap, embedded and matrix wh (run_fillergap.py, analyze_fillergap_r5.py; round-5 C8 protocol, fp32):
  primary population (non-DAS T and kept non-DAS I verbs, non-DAS targets, half-B contexts); the declared reading
  per site for T − I donors (D) and in range (D_in), with nofill O, fill L, fill O, fill − nofill O and O level.

Writes reports/gemma/transfer.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analyze_fillergap import reading
from analyze_passive_test import outcome

NL = {"Pythia-1.4B": 24, "Gemma-3-1B": 26}


def f2(r, k="est"):
    return f"{r[k]:+.2f} [{r[k.replace('est', 'lo95')]:+.2f}, {r[k.replace('est', 'hi95')]:+.2f}]"


def runs(args):
    return {"Pythia-1.4B": {"passive": [Path(p) for p in args.pythia_passive.split(",")],
                            "cons": Path(args.pythia_cons), "inrange": Path(args.pythia_inrange),
                            "fg": Path(args.pythia_fg)},
            "Gemma-3-1B": {"passive": [Path(args.gemma) / "passive_test"], "cons": Path(args.gemma) / "constructions",
                           "inrange": Path(args.gemma) / "inrange", "fg": Path(args.gemma) / "fillergap"}}


def passive_rows(R):
    out = []
    for model, r in R.items():
        for d in r["passive"]:
            tr = pd.read_csv(d / "transfer_summary.csv")
            deltas = pd.read_json(d / "deltas.json", typ="series")
            t = tr[(tr.population == "primary (plain)") & (tr.band == "all") & (tr.condition == "D_bad")]
            for site, g in t.groupby("site"):
                g = g.set_index("readout")
                o, b, dt = g.loc["O"], g.loc["by"], g.loc["dot"]
                out.append({"model": model, "site": int(site), "depth": site / NL[model],
                            "delta": float(deltas[str(site)] if str(site) in deltas else deltas[site]),
                            "O": f"{f2(o)} {o['class']}", "by": f"{f2(b)} {b['class']}", "dot": f"{f2(dt)} {dt['class']}",
                            "frac_natgap_by": b.frac_natgap_est, "outcome": outcome(o["class"], b["class"])})
    return pd.DataFrame(out).sort_values(["depth", "model"])


def cons_rows(R):
    out = []
    for model, r in R.items():
        c = pd.read_csv(r["cons"] / "constructions_summary.csv")
        c = c[c.quantity == "D_bad"]
        for (cons, site), g in c.groupby(["construction", "site"]):
            g = g.set_index("readout")
            o, l = g.loc["O"], g.loc["L"]
            out.append({"construction": cons, "model": model, "site": int(site), "depth": site / NL[model],
                        "delta_O": o.delta, "delta_L": l.delta_L, "O": f"{f2(o)} {o['class']}",
                        "L": f"{f2(l)} {l['class']}"})
    return pd.DataFrame(out).sort_values(["construction", "depth", "model"])


INR = {"passive": ("by", "O"), "OR": ("MAIN", "PREP", "O"), "TC": ("END", "PREP", "O")}


def inrange_rows(R):
    out = []
    for model, r in R.items():
        s = pd.read_csv(r["inrange"] / "inrange_summary.csv")
        for (cons, site), g in s.groupby(["construction", "site"]):
            g = g.set_index("readout")
            row = {"construction": cons, "model": model, "site": int(site), "depth": site / NL[model]}
            for rd in INR[cons]:
                x = g.loc[rd]
                row[rd] = f"{f2(x, 'D_in_est')} ({x.ratio_est:.2f})"
            out.append(row)
    return pd.DataFrame(out).sort_values(["construction", "depth", "model"])


def fg_rows(R, pop):
    out = []
    for model, r in R.items():
        s = pd.read_csv(r["fg"] / "c8_patch_summary.csv")
        for (exp, site), sub in s[s.population == pop].groupby(["exp", "site"]):
            g = lambda fr, q, rd: sub[(sub.frame == fr) & (sub.quantity == q) & (sub.readout == rd)].iloc[0]
            for pq, oq, nm in (("D", "T_only", "T − I donors"), ("D_in", "inT_only", "in range")):
                rd = reading(g, pq, oq)[0]
                out.append({"exp": exp, "patch": nm, "model": model, "site": int(site), "depth": site / NL[model],
                            "n_verbs": int(sub.n_verbs.iloc[0]),
                            "nofill_O": f2(g("nofill", pq, "O")), "fill_L": f2(g("fill", pq, "L")),
                            "fill_O": f2(g("fill", pq, "O")), "fill_minus_nofill_O": f2(g("fill - nofill", pq, "O")),
                            "O_level": f2(g("fill", oq, "Olevel")), "reading": rd})
    return pd.DataFrame(out).sort_values(["exp", "patch", "depth", "model"])


def table(df, cols, heads):
    L = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        L.append("| " + " | ".join(cells) + " |")
    return L


def main(args):
    R = runs(args)
    L = ["# Transfer of the active-trained direction: Gemma-3-1B next to Pythia-1.4B", "",
         "Script `scripts/analyze_gemma_transfer.py` (sources and definitions in its docstring). Same runners, "
         "builders and analyses for both models; Gemma at its rule-picked sites (15: frozen rule; 12: earliest within "
         "0.05 of the max active IIA; 5 and 8: Pythia's declared secondary-site rule, the first sites whose best active "
         "IIA reaches 0.8 and 0.9, which gives Pythia's 8 and 12), Pythia at its stored sites. Rows sorted by relative "
         "depth = site / layers "
         "(Pythia 24, Gemma 26). Estimates in nats with 95% CIs.", ""]
    P = passive_rows(R)
    L += ["## Passive (*by*) test", "",
          "Primary population (plain bad verbs), all bands; D on bad passive bases = after a transitive minus after an "
          "intransitive active donor. Outcome: surface = O RISE and *by* not; abstract = O NO RISE and *by* RISE.", ""]
    L += table(P, ["model", "site", "depth", "delta", "O", "by", "dot", "frac_natgap_by", "outcome"],
               ["model", "site", "depth", "δ_s", "D log P(O)", "D log P(\" by\")", "D log P(\".\")",
                "D(by) / natural gap", "outcome"]) + [""]
    C = cons_rows(R)
    L += ["## Object relatives (OR) and tough constructions (TC): T − I donors", "",
          "D on bad bases. L = MAIN − PREP (OR) or END − PREP (TC). Classes against δ_s (O) and δ_L (L).", ""]
    for cons, g in C.groupby("construction"):
        L += [f"### {cons}", ""] + table(g, ["model", "site", "depth", "delta_O", "O", "delta_L", "L"],
                                         ["model", "site", "depth", "δ_s", "D O", "δ_L", "D L"]) + [""]
    I = inrange_rows(R)
    L += ["## In-range patch (passive, OR, TC)", "",
          "D_in = readout at t_T − at t_I (coordinate set to the construction's own class means), with its ratio to "
          "the natural good − bad gap in parentheses.", ""]
    for cons, g in I.groupby("construction"):
        cols = list(INR[cons])
        L += [f"### {cons}", ""] + table(g, ["model", "site", "depth"] + cols,
                                         ["model", "site", "depth"] + [f"D_in {c} (ratio)" for c in cols]) + [""]
    for pop in (args.fg_population,):
        F = fg_rows(R, pop)
        L += ["## Filler-gap (round-5 C8 protocol, fp32)", "",
              f"Population: {pop}. B6 embedded wh (\"I know that / what NAME V-ed\"), B7 matrix wh (\"AUX NAME V\" / "
              "\"What AUX NAME V\"). Reading rule as round 4 (licensing-like crossover / surface / mixed).", ""]
        for (exp, patch), g in F.groupby(["exp", "patch"], sort=False):
            L += [f"### {'embedded' if exp == 'emb' else 'matrix'} wh, {patch}", ""]
            L += table(g, ["model", "site", "depth", "n_verbs", "nofill_O", "fill_L", "fill_O", "fill_minus_nofill_O",
                           "O_level", "reading"],
                       ["model", "site", "depth", "n I verbs", "nofill O", "fill L", "fill O", "fill − nofill O",
                        "O level vs T", "reading"]) + [""]
    Path(args.report).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gemma", default="results/gemma/das/transfer")
    ap.add_argument("--pythia-passive", default="results/das_round2/passive_test,results/das_round2/passive_test_fill")
    ap.add_argument("--pythia-cons", default="results/das_round2/constructions")
    ap.add_argument("--pythia-inrange", default="results/round4/inrange")
    ap.add_argument("--pythia-fg", default="results/round5/fillergap")
    ap.add_argument("--fg-population", default="primary: non-DAS, half B")
    ap.add_argument("--report", default="reports/gemma/transfer.md")
    main(ap.parse_args())
