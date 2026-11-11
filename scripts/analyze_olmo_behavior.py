#!/usr/bin/env python3
"""OLMo replication, behavioural check: OLMo-2-1B next to Pythia-1.4B and Gemma-3-1B-pt on the same verbs.

Same measures, verbs and definitions as the Gemma step 2 (scripts/analyze_gemma_behavior.py, whose functions are
reused): Delta_pass(V) over the 24 freq nouns; the T - I *by* gap (curated: task-1 contexts; D11: after
"The N was V", freq nouns); FreqBLiMP released passive accuracy. No token-count filter.

OLMo is scored twice (rule declared before the runs, reports/olmo/rules.md):
- "olmo-2-1b"      primary: no BOS (OLMo-2's tokenizer adds none; Pythia's convention, the same EOS-delimited
                   training format);
- "olmo-2-1b+eos"  sensitivity: every prompt prefixed with <|endoftext|> (OLMo-2's bos_token, = EOS).
Both: bf16 weights, logits in fp32 from the final-norm output (as Gemma).

Pythia and Gemma per-verb scores are the stored step-2 ones (results/gemma/behavior/verb_scores.csv). Their
cells / gaps are recomputed with the step-2 bootstrap draw order (stored groups first, OLMo appended), so their
CIs reproduce the stored cells.csv / gaps.csv exactly (asserted). Token bins: "gemma_bin" (the step-2 split) and,
for all three models, "olmo_bin" (token count of " V" under OLMo; separate bootstrap stream, seed + 1).

"by gap holds" (declared rule): the pooled (all bands) T - I *by* gap of the primary OLMo run has a 95% CI above
0 in both inventories (D11 and curated).

Outputs (--out-dir): verb_scores.csv, cells.csv, gaps.csv (gemma-bin split, all models), cells_olmo_bin.csv,
gaps_olmo_bin.csv, freqblimp.csv, meta.json; tables to --tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_gemma_behavior import (BANDS, BINS, Boot, ci, curated_by_items, curated_by_verb, fmt, frame_verb_scores,
                                    sha, tbin)

STORED_MODELS = ("pythia-1.4b", "gemma-3-1b-pt")
OLMO = {"olmo-2-1b": "", "olmo-2-1b+eos": "_eos"}  # model label -> file suffix
MODELS = STORED_MODELS + tuple(OLMO)


def cells_gaps(VS, B, bincol):
    """The step-2 loop (analyze_gemma_behavior.main), with the token-bin column as a parameter."""
    cells, gaps, dr, nn = [], [], {}, {}
    for (mdl, inv), g in VS.groupby(["model", "inventory"], sort=False):
        for meas in ("dpass", "by"):
            for gb in ("all",) + BINS:
                h = g if gb == "all" else g[g[bincol] == gb]
                for role in ("T", "I"):
                    for band in BANDS + ("all",):
                        x = h[(h.role == role) & ((h.band == band) | (band == "all"))][meas]
                        d = B.draws(x)
                        dr[(mdl, inv, meas, gb, role, band)] = d
                        nn[(mdl, inv, meas, gb, role, band)] = len(x)
                        cells.append({"model": mdl, "inventory": inv, "measure": meas, bincol: gb, "class": role,
                                      "band": band, "n": len(x), **dict(zip(("est", "lo", "hi"), ci(d)))})
                for band in BANDS + ("all",):
                    d = dr[(mdl, inv, meas, gb, "T", band)] - dr[(mdl, inv, meas, gb, "I", band)]
                    gaps.append({"model": mdl, "inventory": inv, "measure": meas, bincol: gb,
                                 "contrast": f"T-I {band}", "n_min": min(nn[(mdl, inv, meas, gb, r, band)] for r in "TI"),
                                 **dict(zip(("est", "lo", "hi"), ci(d)))})
                for role in ("T", "I"):
                    for b1 in ("head", "tail"):
                        d = dr[(mdl, inv, meas, gb, role, b1)] - dr[(mdl, inv, meas, gb, role, "xtail")]
                        gaps.append({"model": mdl, "inventory": inv, "measure": meas, bincol: gb,
                                     "contrast": f"{role} {b1}-xtail",
                                     "n_min": min(nn[(mdl, inv, meas, gb, role, b)] for b in (b1, "xtail")),
                                     **dict(zip(("est", "lo", "hi"), ci(d)))})
    C, G = pd.DataFrame(cells), pd.DataFrame(gaps)
    C.loc[C.n < 5, ["lo", "hi"]] = np.nan
    G.loc[G.n_min < 5, ["lo", "hi"]] = np.nan
    return C, G


def main(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(Path(args.olmo_data) / "pairs.csv", keep_default_na=False)
    overbs = pd.read_csv(Path(args.olmo_data) / "verbs.csv", keep_default_na=False).set_index("verb")

    stored_vs = pd.read_csv(args.stored_verb_scores, keep_default_na=False, dtype={"gbin": str})
    assert set(stored_vs.model) == set(STORED_MODELS)
    assert set(stored_vs.verb) == set(overbs.index)
    assert (overbs.gemma_ntok.reindex(stored_vs.verb).to_numpy() == stored_vs.gemma_ntok.to_numpy()).all()

    scores = pd.read_csv(args.band_cross_scores, low_memory=False)
    items = curated_by_items(scores)
    rows, inputs = [], {}
    for mdl, suf in OLMO.items():
        fp = Path(args.frame_dir) / f"olmo-2-1b{suf}/frame_scores.parquet"
        ro = Path(args.readout_dir) / f"olmo-2-1b{suf}_readout.csv"
        inputs[str(fp)], inputs[str(ro)] = sha(fp), sha(ro)
        f = frame_verb_scores(pd.read_parquet(fp)).reindex(overbs.index)
        assert f.notna().all().all() and (f.n_ctx == 24).all(), mdl
        byc = curated_by_verb(items, ro)[0].set_index("verb").by_cur
        for inv in ("d11", "curated"):
            P = pairs[pairs.inventory == inv]
            for side, role in (("t_verb", "T"), ("i_verb", "I")):
                for vp, band, v in zip(P.verb_pair, P.band, P[side]):
                    rows.append({"model": mdl, "inventory": inv, "verb_pair": vp, "band": band, "role": role,
                                 "verb": v, "gemma_ntok": int(overbs.gemma_ntok[v]),
                                 "pythia_ntok": int(overbs.pythia_ntok[v]), "dpass": f.dpass[v],
                                 "by": byc[v] if inv == "curated" else f.by_d11[v], "gbin": tbin(int(overbs.gemma_ntok[v]))})
    VS = pd.concat([stored_vs, pd.DataFrame(rows)], ignore_index=True)
    VS["olmo_ntok"] = VS.verb.map(overbs.n_tok).astype(int)
    VS["obin"] = VS.olmo_ntok.map(tbin)
    VS.to_csv(out / "verb_scores.csv", index=False)

    # gemma-bin split: same draw order as step 2 for the stored models, so their CIs reproduce exactly
    C, G = cells_gaps(VS, Boot(args.n_boot, args.seed), "gbin")
    sc = pd.read_csv(Path(args.stored_verb_scores).parent / "cells.csv", dtype={"gemma_bin": str})
    sg = pd.read_csv(Path(args.stored_verb_scores).parent / "gaps.csv", dtype={"gemma_bin": str})
    repro = {}
    for name, new, old in (("cells", C, sc), ("gaps", G, sg)):
        new = new[new.model.isin(STORED_MODELS)].rename(columns={"gbin": "gemma_bin"}).reset_index(drop=True)
        old = old.reset_index(drop=True)
        key = ["model", "inventory", "measure", "gemma_bin"] + (["contrast"] if name == "gaps" else ["class", "band"])
        assert len(new) == len(old) and (new[key].astype(str).to_numpy() == old[key].astype(str).to_numpy()).all(), name
        for c in ("est", "lo", "hi"):
            assert (new[c].isna() == old[c].isna()).all(), f"{name}: NaN pattern of {c} differs"
        d = max(float(np.nanmax(np.abs(new[c].to_numpy(float) - old[c].to_numpy(float)))) for c in ("est", "lo", "hi"))
        repro[name] = d
        assert d <= 1e-12, f"stored step-2 {name} not reproduced (max |diff| {d})"
    C.rename(columns={"gbin": "gemma_bin"}).to_csv(out / "cells.csv", index=False)
    G.rename(columns={"gbin": "gemma_bin"}).to_csv(out / "gaps.csv", index=False)
    CO, GO = cells_gaps(VS, Boot(args.n_boot, args.seed + 1), "obin")
    CO.rename(columns={"obin": "olmo_bin"}).to_csv(out / "cells_olmo_bin.csv", index=False)
    GO.rename(columns={"obin": "olmo_bin"}).to_csv(out / "gaps_olmo_bin.csv", index=False)

    # FreqBLiMP: stored Pythia / Gemma rows + OLMo
    FB = [pd.read_csv(Path(args.stored_verb_scores).parent / "freqblimp.csv")]
    rng = np.random.default_rng(args.seed)
    for mdl, suf in OLMO.items():
        p = Path(args.fb_dir) / f"olmo-2-1b{suf}_scores.csv"
        if not p.exists():
            continue
        inputs[str(p)] = sha(p)
        f = pd.read_csv(p, low_memory=False)
        rr = []
        for (par, band), h in f.groupby(["paradigm", "band"]):
            x = h.correct.to_numpy(float)
            d = np.concatenate([[x.mean()], x[rng.integers(0, len(x), (args.n_boot, len(x)))].mean(1)])
            rr.append({"model": mdl, "paradigm": par, "band": band, "n": len(x), **dict(zip(("est", "lo", "hi"), ci(d)))})
        FB.append(pd.DataFrame(rr))
    FB = pd.concat(FB, ignore_index=True)
    FB.to_csv(out / "freqblimp.csv", index=False)

    def pooled(mdl, inv):
        r = G[(G.model == mdl) & (G.inventory == inv) & (G.measure == "by") & (G.gbin == "all") & (G.contrast == "T-I all")].iloc[0]
        return {"est": r.est, "lo": r.lo, "hi": r.hi}

    verdict = {m: {inv: pooled(m, inv) for inv in ("d11", "curated")} for m in OLMO}
    for m in verdict:
        verdict[m]["holds"] = bool(all(verdict[m][inv]["lo"] > 0 for inv in ("d11", "curated")))
    write_tables(args, C.rename(columns={"gbin": "gemma_bin"}), G.rename(columns={"gbin": "gemma_bin"}),
                 CO.rename(columns={"obin": "olmo_bin"}), GO.rename(columns={"obin": "olmo_bin"}), FB, repro, verdict)
    meta = {"inputs": {**inputs, args.stored_verb_scores: sha(args.stored_verb_scores),
                       args.band_cross_scores: sha(args.band_cross_scores)},
            "stored_step2_reproduction_max_abs": repro, "by_gap_rule": verdict, "n_boot": args.n_boot, "seed": args.seed}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("stored_step2_reproduction_max_abs", "by_gap_rule")}, indent=1))


def _row(C, G, bincol, mdl, inv, meas, gb, role):
    def cell(band):
        r = C[(C.model == mdl) & (C.inventory == inv) & (C.measure == meas) & (C[bincol] == gb) & (C["class"] == role)
              & (C.band == band)].iloc[0]
        n = int(r.n)
        return f"{fmt((r.est, r.lo, r.hi))} (n={n})" if n >= 5 else (f"{r.est:+.2f} (n={n}, no CI)" if n else "— (n=0)")

    def gp(contrast):
        r = G[(G.model == mdl) & (G.inventory == inv) & (G.measure == meas) & (G[bincol] == gb) & (G.contrast == contrast)].iloc[0]
        n = int(r.n_min)
        return fmt((r.est, r.lo, r.hi)) if n >= 5 else (f"{r.est:+.2f} (n<5, no CI)" if n else "—")

    if role == "T-I":
        return f"| {mdl} | T − I | " + " | ".join(gp(f"T-I {b}") for b in BANDS + ("all",)) + " | | |"
    return (f"| {mdl} | {role} | " + " | ".join(cell(b) for b in BANDS + ("all",))
            + f" | {gp(f'{role} head-xtail')} | {gp(f'{role} tail-xtail')} |")


def write_tables(args, C, G, CO, GO, FB, repro, verdict):
    L = ["# OLMo-2-1B vs Pythia-1.4B vs Gemma-3-1B-pt: behavioural check on the D11 and curated verbs", "",
         "Script `scripts/analyze_olmo_behavior.py`; definitions in `scripts/analyze_gemma_behavior.py`. Same verbs "
         "for all models, no token-count filter. `olmo-2-1b` = no BOS (primary); `olmo-2-1b+eos` = `<|endoftext|>` "
         "prefix (sensitivity). Pythia / Gemma rows = stored step-2 scores; their CIs reproduce the stored step-2 "
         f"tables (max |Δ| cells {repro['cells']:.0e}, gaps {repro['gaps']:.0e}). 95% bootstrap CIs over verbs "
         f"within cell ({args.n_boot} draws); n < 5: no CI.", "",
         "*by*-gap rule (pooled T − I CI above 0 in both inventories): " + "; ".join(
             f"{m}: {'holds' if v['holds'] else 'does not hold'} (D11 {fmt((v['d11']['est'], v['d11']['lo'], v['d11']['hi']))}, "
             f"curated {fmt((v['curated']['est'], v['curated']['lo'], v['curated']['hi']))})" for m, v in verdict.items()), ""]
    for bincol, CC, GG, bname in (("gemma_bin", C, G, "Gemma token bin"), ("olmo_bin", CO, GO, "OLMo token bin")):
        L += [f"# Split by {bname}", ""]
        for inv, title in (("d11", "D11 (281 pairs)"), ("curated", "Curated (126 pairs)")):
            for meas, mt in (("dpass", "Δ_pass"), ("by", "log P(\" by\") after the passive participle"
                                                     + (" (task-1 contexts)" if inv == "curated" else " (freq nouns)"))):
                L += [f"## {title}: {mt}", ""]
                for gb in ("all",) + BINS:
                    if gb == "all" and bincol == "olmo_bin":
                        continue  # identical estimates to the gemma-bin "all" block (separate draws)
                    L += [f"**{'all verbs' if gb == 'all' else f'{bname} {gb}'}**", "",
                          "| Model | Class | Head | Tail | XTail | all bands | Head − XTail | Tail − XTail |",
                          "|---|---|---|---|---|---|---|---|"]
                    for mdl in MODELS:
                        for role in ("T", "I", "T-I"):
                            L.append(_row(CC, GG, bincol, mdl, inv, meas, gb, role))
                    L.append("")
    L += ["## FreqBLiMP released passives: accuracy (whole-sentence LP)", "",
          "| Model | Paradigm | Head | Tail | XTail |", "|---|---|---|---|---|"]
    for mdl in MODELS:
        for par in sorted(FB.paradigm.unique()):
            h = FB[(FB.model == mdl) & (FB.paradigm == par)].set_index("band")
            if len(h):
                L.append(f"| {mdl} | {par} | " + " | ".join(f"{h.est[b]:.3f} [{h.lo[b]:.3f}, {h.hi[b]:.3f}]" for b in BANDS) + " |")
    Path(args.tables).parent.mkdir(parents=True, exist_ok=True)
    Path(args.tables).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--olmo-data", default="data/olmo/frame_olmo")
    ap.add_argument("--stored-verb-scores", default="results/gemma/behavior/verb_scores.csv")
    ap.add_argument("--frame-dir", default="results/olmo/frame")
    ap.add_argument("--readout-dir", default="results/olmo/by_readout")
    ap.add_argument("--fb-dir", default="results/olmo/freqblimp_passives")
    ap.add_argument("--band-cross-scores", default="results/passive_band_cross/pythia14b_scores.csv")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out-dir", default="results/olmo/behavior")
    ap.add_argument("--tables", default="reports/olmo/behavior_tables.md")
    main(ap.parse_args())
