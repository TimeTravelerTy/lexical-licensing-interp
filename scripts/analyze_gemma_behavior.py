#!/usr/bin/env python3
"""Gemma replication, step 2: behavioural effects of Gemma-3-1B-pt next to Pythia-1.4B on the same verbs.

No token-count filter: every D11 (281 pairs) and curated (126 pairs) participle, full participle log-prob summed
over subtokens. Gemma prompts start with BOS; Pythia prompts do not (as in every existing Pythia run).

Measures (per verb, then class x band means; T = transitive / good, I = intransitive / bad):
1. Delta_pass(V) = log P(V | "The N was") - log P(V | "The N has"), averaged over the 24 freq nouns
   (run_frame_contrast.py / analyze_frame_contrast.py definition). Both inventories (D11 primary). Pythia: the
   stored round-6 A2 (D11) and round-5 A1 (curated) scores, fp32.
2. *by* gap = mean log P(" by") after T passives - after I passives.
   - curated (the task-1 definition, analyze_das_prep.task1): curated `was` prefix + participle, other verbs'
     contexts only (126 pairs x 125 contexts; passive_2 rows of the band-cross scores index the items);
     per-verb mean over its items. Pythia: the stored bf16 readout (results/passive_das_prep).
   - d11: log P(" by") after "The N was V" (freq nouns), lp_next_by of run_frame_contrast.py --next-token " by";
     a new run for both models (Pythia fp32).
3. FreqBLiMP released passive_1 / passive_2 accuracy (whole-sentence LP, score_matched_passives.py), by band.

CIs: 95% percentile bootstrap over verbs (2,000 draws, seed 17), verbs resampled within class x band (x Gemma
token bin) cell; a difference of two cells resamples each independently (unpaired). FreqBLiMP accuracy:
items resampled within paradigm x band. Every table overall and split by the Gemma token count of the
participle (1 / 2 / 3+); the same verbs in both models.

Outputs (--out-dir): verb_scores.csv (per verb x model x inventory), cells.csv (class x band [x bin]),
gaps.csv (T - I gaps and band contrasts), freqblimp.csv, meta.json; full tables to --tables (markdown).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_das_prep_prompts import pid

BANDS = ("head", "tail", "xtail")
BN = {"head": "Head", "tail": "Tail", "xtail": "XTail"}
BINS = ("1", "2", "3+")
MODELS = ("pythia-1.4b", "gemma-3-1b-pt")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tbin(n):
    return "1" if n == 1 else "2" if n == 2 else "3+"


# ---------------------------------------------------------------- per-verb scores

def frame_verb_scores(S, col="lp"):
    """Per verb: mean over freq contexts of (pass - act) for `col`, and the pass-frame mean of lp_next_by."""
    s = S[S["set"] == "freq"]
    lp = s.pivot_table(index="verb", columns=["frame", "context_id"], values=col)
    assert set(lp["pass"].columns) == set(lp["act"].columns), "pass / act contexts differ"
    d = lp["pass"] - lp["act"][lp["pass"].columns]
    assert d.notna().all().all(), "a verb lacks a pass or act score in some context"
    out = pd.DataFrame({"dpass": d.mean(1), "n_ctx": d.notna().sum(1)})
    if "lp_next_by" in s:
        out["by_d11"] = s[s.frame == "pass"].groupby("verb").lp_next_by.mean()
    return out


def curated_by_items(scores):
    """Task-1 items: passive_2 curated rows, other verbs' contexts (prefix shared with passive_1)."""
    cur = scores[(scores.context_set == "curated") & (scores.paradigm == "passive_2") & (scores.own_context == 0)]
    return cur[["verb_pair", "verb_band", "context_id", "prefix", "good_verb", "bad_verb"]].copy()


def curated_by_verb(items, readout):
    ro = pd.read_csv(readout, usecols=["prompt_id", "lp[ by]"], low_memory=False).set_index("prompt_id")["lp[ by]"]
    rows = []
    for side, role in (("good", "T"), ("bad", "I")):
        ids = (items.prefix + items[f"{side}_verb"]).map(lambda p: pid("passive", p))
        lp = ro.reindex(ids).to_numpy()
        assert np.isfinite(lp).all(), f"missing readout rows ({side})"
        rows.append(pd.DataFrame({"verb_pair": items.verb_pair, "band": items.verb_band, "verb": items[f"{side}_verb"],
                                  "role": role, "lp_by": lp}))
    r = pd.concat(rows)
    g = r.groupby(["verb_pair", "band", "verb", "role"]).lp_by.agg(["mean", "size"]).reset_index()
    return g.rename(columns={"mean": "by_cur", "size": "n_items"}), r


# ---------------------------------------------------------------- bootstrap

class Boot:
    def __init__(self, n, seed):
        self.n, self.rng = n, np.random.default_rng(seed)

    def draws(self, x):
        x = np.asarray(x, float)
        if len(x) == 0:
            return np.full(self.n + 1, np.nan)
        idx = self.rng.integers(0, len(x), (self.n, len(x)))
        return np.concatenate([[x.mean()], x[idx].mean(1)])


def ci(d):
    if np.isnan(d[0]):
        return (np.nan, np.nan, np.nan)
    return (float(d[0]), *map(float, np.percentile(d[1:], [2.5, 97.5])))


def fmt(t, k=2):
    if t is None or np.isnan(t[0]):
        return "—"
    return f"{t[0]:+.{k}f} [{t[1]:+.{k}f}, {t[2]:+.{k}f}]"


# ---------------------------------------------------------------- main

def main(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(Path(args.gemma_data) / "pairs.csv", keep_default_na=False)
    gverbs = pd.read_csv(Path(args.gemma_data) / "verbs.csv", keep_default_na=False).set_index("verb")
    pverbs = pd.read_csv(Path(args.pythia_data) / "verbs.csv", keep_default_na=False).set_index("verb")
    assert (gverbs.index == pverbs.index).all() and (gverbs.gemma_ntok == pverbs.gemma_ntok).all()

    # frame scores
    stored = pd.concat([pd.read_parquet(args.pythia_d11), pd.read_parquet(args.pythia_r5)])
    stored = stored[stored["set"] == "freq"].drop_duplicates(["set", "context_id", "frame", "verb"])
    new_p = pd.read_parquet(Path(args.frame_dir) / "pythia-1.4b/frame_scores.parquet")
    new_g = pd.read_parquet(Path(args.frame_dir) / "gemma-3-1b-pt/frame_scores.parquet")
    k = ["set", "context_id", "frame", "verb"]
    m = new_p.merge(stored[k + ["lp"]], on=k, suffixes=("", "_stored"))
    repro = {"n_rows_common": len(m), "n_rows_new": len(new_p),
             "max_abs_diff_lp": float((m.lp - m.lp_stored).abs().max())}
    assert len(m) == len(new_p), "new Pythia frame rows not all in the stored scores"
    assert repro["max_abs_diff_lp"] <= args.repro_tol, f"new Pythia frame scores differ from stored: {repro}"
    fv = {"pythia-1.4b": frame_verb_scores(stored).join(frame_verb_scores(new_p)[["by_d11"]]),
          "gemma-3-1b-pt": frame_verb_scores(new_g)}
    for mdl in fv:  # the stored round-5 scores also hold pool verbs
        fv[mdl] = fv[mdl].reindex(gverbs.index)
        assert fv[mdl].notna().all().all() and (fv[mdl].n_ctx == 24).all(), mdl

    # curated by readout
    scores = pd.read_csv(args.band_cross_scores, low_memory=False)
    items = curated_by_items(scores)
    cb = {"pythia-1.4b": curated_by_verb(items, args.pythia_readout),
          "gemma-3-1b-pt": curated_by_verb(items, args.gemma_readout)}

    # long per-verb table: model x inventory x verb (role / band from the inventory's pairs)
    rows = []
    for inv in ("d11", "curated"):
        P = pairs[pairs.inventory == inv]
        for mdl in MODELS:
            f = fv[mdl]
            byc = cb[mdl][0].set_index("verb").by_cur if inv == "curated" else None
            for side, role in (("t_verb", "T"), ("i_verb", "I")):
                for vp, band, v in zip(P.verb_pair, P.band, P[side]):
                    rows.append({"model": mdl, "inventory": inv, "verb_pair": vp, "band": band, "role": role,
                                 "verb": v, "gemma_ntok": int(gverbs.gemma_ntok[v]), "pythia_ntok": int(pverbs.pythia_ntok[v]),
                                 "dpass": f.dpass[v],
                                 "by": (byc[v] if byc is not None else f.by_d11[v])})
    VS = pd.DataFrame(rows)
    VS["gbin"] = VS.gemma_ntok.map(tbin)
    VS.to_csv(out / "verb_scores.csv", index=False)

    # cells and contrasts
    B = Boot(args.n_boot, args.seed)
    cells, gaps, dr, nn = [], [], {}, {}
    for (mdl, inv), g in VS.groupby(["model", "inventory"], sort=False):
        for meas in ("dpass", "by"):
            for gb in ("all",) + BINS:
                h = g if gb == "all" else g[g.gbin == gb]
                for role in ("T", "I"):
                    for band in BANDS + ("all",):
                        x = h[(h.role == role) & ((h.band == band) | (band == "all"))][meas]
                        d = B.draws(x)
                        dr[(mdl, inv, meas, gb, role, band)] = d
                        nn[(mdl, inv, meas, gb, role, band)] = len(x)
                        cells.append({"model": mdl, "inventory": inv, "measure": meas, "gemma_bin": gb, "class": role,
                                      "band": band, "n": len(x), **dict(zip(("est", "lo", "hi"), ci(d)))})
                for band in BANDS + ("all",):  # T - I within band (all = pooled over bands, verbs equally weighted)
                    d = dr[(mdl, inv, meas, gb, "T", band)] - dr[(mdl, inv, meas, gb, "I", band)]
                    gaps.append({"model": mdl, "inventory": inv, "measure": meas, "gemma_bin": gb,
                                 "contrast": f"T-I {band}", "n_min": min(nn[(mdl, inv, meas, gb, r, band)] for r in "TI"),
                                 **dict(zip(("est", "lo", "hi"), ci(d)))})
                for role in ("T", "I"):  # Head - XTail and Tail - XTail within class
                    for b1 in ("head", "tail"):
                        d = dr[(mdl, inv, meas, gb, role, b1)] - dr[(mdl, inv, meas, gb, role, "xtail")]
                        gaps.append({"model": mdl, "inventory": inv, "measure": meas, "gemma_bin": gb,
                                     "contrast": f"{role} {b1}-xtail",
                                     "n_min": min(nn[(mdl, inv, meas, gb, role, b)] for b in (b1, "xtail")),
                                     **dict(zip(("est", "lo", "hi"), ci(d)))})
    C, G = pd.DataFrame(cells), pd.DataFrame(gaps)
    C.loc[C.n < 5, ["lo", "hi"]] = np.nan  # no CI for cells with fewer than 5 verbs (also in the tables)
    G.loc[G.n_min < 5, ["lo", "hi"]] = np.nan
    C.to_csv(out / "cells.csv", index=False)
    G.to_csv(out / "gaps.csv", index=False)

    # FreqBLiMP released passives
    fb_rows = []
    rng = np.random.default_rng(args.seed)
    for mdl, path in (("pythia-1.4b", args.pythia_fb), ("gemma-3-1b-pt", args.gemma_fb)):
        if not path or not Path(path).exists():
            continue
        f = pd.read_csv(path, low_memory=False)
        for (par, band), h in f.groupby(["paradigm", "band"]):
            x = h.correct.to_numpy(float)
            d = np.concatenate([[x.mean()], x[rng.integers(0, len(x), (args.n_boot, len(x)))].mean(1)])
            fb_rows.append({"model": mdl, "paradigm": par, "band": band, "n": len(x), **dict(zip(("est", "lo", "hi"), ci(d)))})
    FB = pd.DataFrame(fb_rows)
    FB.to_csv(out / "freqblimp.csv", index=False)

    # task-1 point-estimate reproduction (Pythia, curated by gap): item means = verb means (125 items per verb)
    t1 = {b: float(cb["pythia-1.4b"][1].pivot_table(index=["verb_pair", "band"], columns="role", values="lp_by",
                                                     aggfunc="mean").reset_index().query("band == @b").eval("T - I").mean())
          for b in BANDS}

    write_summary(out, C, G, FB, repro, t1, args)
    meta = {"inputs": {p: sha(p) for p in (args.pythia_d11, args.pythia_r5, args.pythia_readout, args.gemma_readout,
                                           str(Path(args.frame_dir) / "pythia-1.4b/frame_scores.parquet"),
                                           str(Path(args.frame_dir) / "gemma-3-1b-pt/frame_scores.parquet"))},
            "pythia_frame_reproduction": repro, "task1_by_gap_point_estimates": t1,
            "n_boot": args.n_boot, "seed": args.seed}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("pythia_frame_reproduction", "task1_by_gap_point_estimates")}, indent=1))


def cell(C, mdl, inv, meas, gb, role, band):
    r = C[(C.model == mdl) & (C.inventory == inv) & (C.measure == meas) & (C.gemma_bin == gb) & (C["class"] == role)
          & (C.band == band)].iloc[0]
    return (r.est, r.lo, r.hi), int(r.n)


def gap(G, mdl, inv, meas, gb, contrast):
    r = G[(G.model == mdl) & (G.inventory == inv) & (G.measure == meas) & (G.gemma_bin == gb) & (G.contrast == contrast)].iloc[0]
    return (r.est, r.lo, r.hi), int(r.n_min)


def gfmt(t):
    (c, n) = t
    return fmt(c) if n >= 5 else (f"{c[0]:+.2f} (n<5, no CI)" if n else "—")


def write_summary(out, C, G, FB, repro, t1, args):
    L = ["# Gemma-3-1B-pt vs Pythia-1.4B: behavioural check on the D11 and curated verbs", "",
         "Script `scripts/analyze_gemma_behavior.py`; definitions in its docstring. Same verbs for both models, no "
         "token-count filter (full participle log-prob summed over subtokens). **BOS asymmetry:** every Gemma "
         "prompt starts with `<bos>`; Pythia prompts have no BOS, as in every existing Pythia run, so the Pythia "
         "numbers stay comparable with them. Gemma: bf16 weights, fp32 unembedding (logits computed in fp32 from "
         "the final-norm output). Pythia: stored scores (Δ_pass fp32; curated *by* readout bf16); D11 *by* readout "
         "new (fp32). 95% CIs: bootstrap over verbs within cell (unpaired), "
         f"{args.n_boot} draws, seed {args.seed}; cells with n < 5 get no CI. Gemma token bin = token count of \" V\" under Gemma.", "",
         f"Checks: new Pythia frame run vs stored, {repro['n_rows_common']} rows, max |Δ lp| "
         f"{repro['max_abs_diff_lp']:.1e}; curated *by* gap point estimates reproduce task 1 "
         f"(Head {t1['head']:+.2f}, Tail {t1['tail']:+.2f}, XTail {t1['xtail']:+.2f}).", ""]
    for inv, title in (("d11", "D11 (281 pairs)"), ("curated", "Curated (126 pairs)")):
        for meas, mt in (("dpass", "Δ_pass"), ("by", "log P(\" by\") after the passive participle" +
                                                 (" (task-1 contexts)" if inv == "curated" else " (freq nouns)"))):
            L += [f"## {title}: {mt}", ""]
            for gb in ("all",) + BINS:
                hdr = "all verbs" if gb == "all" else f"Gemma token bin {gb}"
                L += [f"**{hdr}**", "", "| Model | Class | Head | Tail | XTail | all bands | Head − XTail | Tail − XTail |",
                      "|---|---|---|---|---|---|---|---|"]
                for mdl in MODELS:
                    for role in ("T", "I"):
                        cs = [cell(C, mdl, inv, meas, gb, role, b) for b in BANDS + ("all",)]
                        L.append(f"| {mdl} | {role} | " + " | ".join((f"{fmt(c)} (n={n})" if n >= 5 else f"{c[0]:+.2f} (n={n}, no CI)" if n else "— (n=0)") for c, n in cs)
                                 + f" | {gfmt(gap(G, mdl, inv, meas, gb, f'{role} head-xtail'))}"
                                 + f" | {gfmt(gap(G, mdl, inv, meas, gb, f'{role} tail-xtail'))} |")
                    L.append(f"| {mdl} | T − I | " + " | ".join(gfmt(gap(G, mdl, inv, meas, gb, f'T-I {b}')) for b in BANDS + ("all",)) + " | | |")
                L.append("")
    if len(FB):
        L += ["## FreqBLiMP released passives: accuracy (whole-sentence LP)", "",
              "| Model | Paradigm | Head | Tail | XTail |", "|---|---|---|---|---|"]
        for mdl in MODELS:
            for par in sorted(FB.paradigm.unique()):
                h = FB[(FB.model == mdl) & (FB.paradigm == par)].set_index("band")
                if len(h):
                    L.append(f"| {mdl} | {par} | " + " | ".join(
                        f"{h.est[b]:.3f} [{h.lo[b]:.3f}, {h.hi[b]:.3f}]" for b in BANDS) + " |")
        L.append("")
    Path(args.tables).parent.mkdir(parents=True, exist_ok=True)
    Path(args.tables).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gemma-data", default="data/gemma/frame_gemma")
    ap.add_argument("--pythia-data", default="data/gemma/frame_pythia")
    ap.add_argument("--frame-dir", default="results/gemma/frame")
    ap.add_argument("--pythia-d11", default="results/round6/frame_d11/pythia-1.4b/frame_scores.parquet")
    ap.add_argument("--pythia-r5", default="results/round5/frame/frame_scores.parquet")
    ap.add_argument("--band-cross-scores", default="results/passive_band_cross/pythia14b_scores.csv")
    ap.add_argument("--pythia-readout", default="results/passive_das_prep/pythia14b_readout.csv")
    ap.add_argument("--gemma-readout", default="results/gemma/by_readout/gemma-3-1b-pt_readout.csv")
    ap.add_argument("--pythia-fb", default="results/freqblimp_original_passives/pythia14b_scores.csv")
    ap.add_argument("--gemma-fb", default="results/gemma/freqblimp_passives/gemma-3-1b-pt_scores.csv")
    ap.add_argument("--repro-tol", type=float, default=1e-4, help="new vs stored Pythia frame lp (same GPU type)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out-dir", default="results/gemma/behavior")
    ap.add_argument("--tables", default="reports/gemma/step2_tables.md")
    main(ap.parse_args())
