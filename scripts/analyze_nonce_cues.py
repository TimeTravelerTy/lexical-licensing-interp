#!/usr/bin/env python3
"""Round 4, Part C: nonce cue tests, analysis (reports/round4/plan.md, C8 / C9).

z per prompt and site as round-3 C8 (`analyze_nonce_passive.z_table`: all 15 bases averaged for
nonce prompts; 0 = held-out active intransitive, 1 = transitive level). Per lemma, the mean over
its 4 slots; lemma bootstrap (2,000 draws, seed 17).
- C8: per form (ed, ing, s) Delta_matched = T - I, Delta_balanced = AB - BA, and for ing / s the
  mismatched-lemma T - I and the paired ed - form differences, all on the lemmas retained for that
  form (lemmas whose probe-final token occurs in the form's contexts are dropped).
- C9, structure-matched contrasts: transitivity without an adjacent NP = rel_T - rel_I (primary),
  q_T - q_I (secondary); adjacent NP without transitivity = adj_I - pp_I (primary), adj_I - ed_I
  (secondary); plus T+NP - bare and the 2 x 2 main effects (descriptive).
Decision rules: plan.md, C8 / C9 (minimum effects 0.05 z, 0.1 nats).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_nonce_passive import z_table

RD = ("by", "O", "prep_noby", "dot")
MIN = {"z": 0.05, "by": 0.1, "O": 0.1, "prep_noby": 0.1, "dot": 0.1}
ZS = (4, 6, 8, 12, 17)


def boot(x, B, rng):
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.full(B + 1, np.nan)
    return np.r_[x.mean(), x[rng.integers(0, len(x), (B, len(x)))].mean(1)]


def q(d):
    if np.isnan(d[0]):
        return {"est": np.nan, "lo95": np.nan, "hi95": np.nan}
    return {"est": float(d[0]), "lo95": float(np.percentile(d[1:], 2.5)), "hi95": float(np.percentile(d[1:], 97.5))}


def run(args):
    ddir, rdir = Path(args.data_dir), Path(args.out_dir)
    P = pd.read_csv(ddir / "prompts.csv", keep_default_na=False)
    nat = pd.read_parquet(rdir / "natural.parquet").set_index("pid").loc[P.pid]
    pz = np.load(rdir / "projections.npz")
    sites = [int(s) for s in pz["sites"]]
    keys = [(int(c[1]), int(c.split("_f")[1])) for c in pz["keys"]]
    das = pd.read_csv(Path(args.root) / "final_strict" / "items.csv")
    pf = pd.read_csv(Path(args.root) / "final_strict" / "pairs_folds.csv")
    Z, _ = z_table(P, pz["proj"], sites, keys, das, pf)
    T = P.assign(**{f"z{s}": Z[:, i] for i, s in enumerate(sites)}, **{k: nat[k].to_numpy() for k in RD})
    meas = [f"z{s}" for s in sites] + list(RD)
    lem = T[T.kind == "nonce"].groupby(["lemma", "cond"])[meas].mean()
    tc = pd.read_csv(rdir / "token_check.csv")
    lemmas = sorted(T[T.kind == "nonce"].lemma.unique())
    drop = {form: sorted(tc[tc.cond.str.startswith(form + "_") & tc.in_context].lemma.unique()) for form in ("ing", "s")}
    keep = {"ed": lemmas, **{f: [l for l in lemmas if l not in drop[f]] for f in ("ing", "s")}}
    rng = np.random.default_rng(args.seed)
    rows = []
    cond = lambda c: lem.xs(c, level="cond").loc[lemmas]

    def add(analysis, contrast, diff, kl):
        d = diff.loc[kl]
        for m in meas:
            rows.append({"analysis": analysis, "contrast": contrast, "measure": m, "n_lemmas": len(kl),
                         **q(boot(d[m].to_numpy(), args.n_boot, rng))})

    ed = cond("ed_T") - cond("ed_I")
    edb = cond("ed_AB") - cond("ed_BA")
    add("C8", "ed: T - I", ed, lemmas)
    add("C8", "ed: AB - BA", edb, lemmas)
    for form in ("ing", "s"):
        kl = keep[form]
        dm = cond(f"{form}_T") - cond(f"{form}_I")
        db = cond(f"{form}_AB") - cond(f"{form}_BA")
        add("C8", f"{form}: T - I", dm, kl)
        add("C8", f"{form}: AB - BA", db, kl)
        add("C8", f"{form}: mismatched T - I", cond(f"{form}_mT") - cond(f"{form}_mI"), kl)
        add("C8", f"ed on {form} lemmas: T - I", ed, kl)
        add("C8", f"ed on {form} lemmas: AB - BA", edb, kl)
        add("C8", f"ed - {form} (T - I)", ed - dm, kl)
        add("C8", f"ed - {form} (AB - BA)", edb - db, kl)
    TNP, Ib = cond("ed_T"), cond("ed_I")
    add("C9", "transitivity, relative: rel_T - rel_I", cond("rel_T") - cond("rel_I"), lemmas)
    add("C9", "transitivity, question: q_T - q_I", cond("q_T") - cond("q_I"), lemmas)
    add("C9", "adjacent NP: adj_I - pp_I", cond("adj_I") - cond("pp_I"), lemmas)
    add("C9", "adjacent NP: adj_I - bare", cond("adj_I") - Ib, lemmas)
    add("C9", "duration PP: pp_I - bare", cond("pp_I") - Ib, lemmas)
    add("C9", "T+NP - bare", TNP - Ib, lemmas)
    add("C9", "rel_T - bare", cond("rel_T") - Ib, lemmas)
    add("C9", "2x2 transitivity main effect (relative)", 0.5 * ((TNP + cond("rel_T")) - (cond("adj_I") + Ib)), lemmas)
    add("C9", "2x2 NP main effect (relative)", 0.5 * ((TNP + cond("adj_I")) - (cond("rel_T") + Ib)), lemmas)
    res = pd.DataFrame(rows)
    res.to_csv(rdir / "nonce_cues_summary.csv", index=False)
    write_report(args, res, drop)


def write_report(args, res, drop):
    g = lambda a, c, m: res[(res.analysis == a) & (res.contrast == c) & (res.measure == m)].iloc[0]
    f = lambda r: "n/a" if np.isnan(r.est) else f"{r.est:+.3f} [{r.lo95:+.3f}, {r.hi95:+.3f}]"
    up = lambda r, m: (not np.isnan(r.est)) and r.lo95 > 0 and r.est >= MIN["z" if m.startswith("z") else m]
    hdr = "| Contrast | lemmas | " + " | ".join(f"z{s}" for s in ZS) + " | \" by\" | O | PREP\\by | \".\" |"
    sep = "|---|---:|" + "---|" * (len(ZS) + 4)
    row = lambda a, c: (f"| {c} | {int(g(a, c, 'z8').n_lemmas)} | " + " | ".join(f(g(a, c, f"z{s}")) for s in ZS) + " | "
                        + " | ".join(f(g(a, c, m)) for m in RD) + " |")
    L = ["# Round 4, Part C. Nonce cue tests", "",
         "Spec: `plan.md`, C8/C9. Run: `run_nonce_cues.py`; analysis: `analyze_nonce_cues.py`. 80 nonce lemmas × 4 "
         "slots, passive probe \"The N was A-ed\"; per-lemma means; lemma bootstrap (2,000 draws, seed 17). z: 0 = "
         "held-out active intransitive, 1 = transitive level of each site's d (fixed ruler).", "",
         f"Lemmas dropped because the probe's final token occurs in the form's contexts: ing {drop['ing'] or 'none'}; "
         f"s {drop['s'] or 'none'}.", "", "## C8. Inflection mismatch", "", hdr, sep]
    order = ["ed: T - I", "ed: AB - BA"]
    for form in ("ing", "s"):
        order += [f"{form}: T - I", f"{form}: AB - BA", f"{form}: mismatched T - I", f"ed on {form} lemmas: T - I",
                  f"ed - {form} (T - I)", f"ed - {form} (AB - BA)"]
    L += [row("C8", c) for c in order]
    L += ["", "**Declared decisions (C8):**", ""]
    for form in ("ing", "s"):
        for s in (6, 8):
            c = f"{form}: T - I"
            zs, by, o = g("C8", c, f"z{s}"), g("C8", c, "by"), g("C8", c, "O")
            if np.isnan(zs.est):
                L.append(f"- {form}, site {s}: **unresolved: no eligible lemmas**.")
                continue
            oed, od = g("C8", f"ed on {form} lemmas: T - I", "O"), g("C8", f"ed - {form} (T - I)", "O")
            vz, vb, vp = (g("C8", f"{form}: AB - BA", m) for m in (f"z{s}", "by", "prep_noby"))
            gone = o.est <= 0.5 * oed.est and od.lo95 > 0
            vby = up(vb, "by") and vb.est > vp.est
            L.append(f"- {form}, site {s}: d shift survives **{'yes' if up(zs, 'z') else 'no'}** ({zs.est:+.3f}); "
                     f"\" by\" survives **{'yes' if up(by, 'by') else 'no'}** ({by.est:+.2f}); object rise mostly "
                     f"disappears **{'yes' if gone else 'no'}** (O {o.est:+.2f} vs ed on the same lemmas {oed.est:+.2f}); "
                     f"verb-specific (balanced): z **{'yes' if up(vz, 'z') else 'no'}**, \" by\" "
                     f"**{'yes' if vby else 'no'}** ({vb.est:+.2f} vs PREP\\by {vp.est:+.2f}).")
    L += ["", "## C9. Which cue drives the shift", "", hdr, sep]
    L += [row("C9", c) for c in ("T+NP - bare", "transitivity, relative: rel_T - rel_I", "transitivity, question: q_T - q_I",
                                 "adjacent NP: adj_I - pp_I", "adjacent NP: adj_I - bare", "duration PP: pp_I - bare",
                                 "rel_T - bare", "2x2 transitivity main effect (relative)", "2x2 NP main effect (relative)")]
    L += ["", "**Declared decisions (C9; primary contrasts rel_T − rel_I and adj_I − pp_I; secondary q_T − q_I and "
              "adj_I − bare):**", ""]
    for tr, np_, tag in (("transitivity, relative: rel_T - rel_I", "adjacent NP: adj_I - pp_I", "primary"),
                         ("transitivity, question: q_T - q_I", "adjacent NP: adj_I - bare", "secondary")):
        for m, lab in (("z6", "z, site 6"), ("z8", "z, site 8"), ("by", "\" by\""), ("O", "O")):
            t, n = g("C9", tr, m), g("C9", np_, m)
            a, b = up(t, m), up(n, m)
            rd = "both" if a and b else "transitivity-driven" if a else "NP-driven" if b else "neither"
            L.append(f"- {tag}, {lab}: **{rd}** (transitivity {t.est:+.3f}, adjacent NP {n.est:+.3f}).")
    L.append("")
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results/das_round2")
    ap.add_argument("--data-dir", default="data/round4/nonce_cues")
    ap.add_argument("--out-dir", default="results/round4/nonce_cues")
    ap.add_argument("--report", default="reports/round4/c_nonce_cues.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
