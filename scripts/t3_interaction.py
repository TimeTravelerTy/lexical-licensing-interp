#!/usr/bin/env python3
"""T3 under the audited criterion (reports/criteria_audit.md, claim 3): the filler-gap frame interaction.

J_O = D_nofill(O) - D_fill(O) = -(fill - nofill O), per declared site and experiment, for v and v2, from each run's
c8_patch_summary.csv (primary population: non-DAS, half B). "licensing-like" iff lo95(J_O) > 0; "opposite" iff
hi95(J_O) < 0. Natural reference G_J = natural T - I O gap without a filler minus with one (point value; the summary
has no draws for it), and J_O / G_J descriptively. Also fill L, fill PREP and the filled O level vs T.
Writes results/v2_summary/t3_interaction.csv.
"""

import pandas as pd

DECL = {"pythia": [8, 12, 17], "gemma": [5, 8, 12, 15], "olmo": [9, 11, 12, 13]}
V = {"pythia": "results/round5/fillergap", "gemma": "results/gemma/das/transfer/fillergap",
     "olmo": "results/olmo/das/transfer/fillergap"}
POP = "primary: non-DAS, half B"

rows = []
for m, sites in DECL.items():
    for inst, d in (("v", V[m]), ("v2", f"results/{m}/das_v2/fillergap")):
        x = pd.read_csv(f"{d}/c8_patch_summary.csv")
        x = x[x.population == POP]
        for exp in ("emb", "mat"):
            for q, ql in (("D_in", "inT_only"), ("D", "T_only")):
                for s in sites:
                    g = x[(x.exp == exp) & (x.site == s)]
                    pick = lambda fr, qq, rd: g[(g.frame == fr) & (g.quantity == qq) & (g.readout == rd)].iloc[0]
                    j = pick("fill - nofill", q, "O")
                    gj = pick("nofill", q, "O").natgap - pick("fill", q, "O").natgap
                    lev = pick("fill", ql, "Olevel")
                    J = {"est": -j.est, "lo95": -j.hi95, "hi95": -j.lo95}
                    rows.append({"model": m, "instrument": inst, "exp": exp, "quantity": q, "site": s,
                                 "J_O": J["est"], "J_O_lo95": J["lo95"], "J_O_hi95": J["hi95"], "G_J": gj,
                                 "J_over_G": J["est"] / gj,
                                 "reading": "licensing-like" if J["lo95"] > 0 else "opposite" if J["hi95"] < 0 else "unresolved",
                                 "fill_L": pick("fill", q, "L").est, "fill_L_lo95": pick("fill", q, "L").lo95,
                                 "fill_PREP": pick("fill", q, "PREP").est, "fill_PREP_hi95": pick("fill", q, "PREP").hi95,
                                 "fill_O_level_vs_T": lev.est, "level_lo90": lev.lo90, "level_hi90": lev.hi90, "bound": lev.bound})
T = pd.DataFrame(rows)
T.to_csv("results/v2_summary/t3_interaction.csv", index=False)
pd.set_option("display.width", 250)
print(T[T.quantity == "D_in"][["model", "instrument", "exp", "site", "J_O", "J_O_lo95", "J_O_hi95", "G_J", "J_over_G",
                               "reading", "fill_L", "fill_O_level_vs_T"]].round(2).to_string(index=False))
