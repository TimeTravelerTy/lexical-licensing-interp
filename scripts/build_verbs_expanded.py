#!/usr/bin/env python3
"""Round 5, D11: expanded T/I verb inventory (reports/round5/plan.md, D11). No model.

Stages:
- `candidates`: pool every T / I participle of the sources, key by lemma, add wordfreq 3.1.1 participle and lemma
  Zipf, FreqBLiMP band (participle Zipf: Head 3.5-5.5, Tail 2.4-3.2, XTail 1.2-2.2), Pythia token count of
  " participle" (from the A1 and verb-expansion inventories, else `ntok_extra.csv`), curated / historical-DAS flags.
  Writes candidates.csv and the blind rating sheet rating_items.csv (in-band lemmas only: item, lemma, participle;
  no source class).
- `match` (after both frozen ratings rating_{claude,codex}.csv exist): a verb is T (I) only if both raters say T (I).
  Within band, the maximum number of one-to-one T-I pairs within the calipers (participle Zipf 0.35, lemma Zipf 0.25),
  then the minimum total participle-Zipf gap (ties by lemma order). Split: pairs with a lemma in a historical DAS
  training inventory are `trainable`; of the rest, one third per band x token-count cell (seed 17) is `heldout`, the
  others `trainable`. Writes pairs.csv, verbs.csv, meta.json and reports/round5/d11_verbs.md.

Sources: FreqBLiMP passive_1 / passive_2 (data/freqblimp_original_passives/pairs.jsonl, the released items; lemmas
from FreqBLiMP's vocabulary, sources/fb_lemmas.csv), data/verb_expansion/verbs.csv,
data/round4/fillergap/candidates.csv, the 126 curated pairs (data/round5/frame/pairs.csv).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from verbs_expanded import DAS_SOURCES, das_historical

OUT = Path("data/round5/verbs_expanded")
BANDS = {"head": (3.5, 5.5), "tail": (2.4, 3.2), "xtail": (1.2, 2.2)}
CAL_PART, CAL_LEMMA = 0.35, 0.25
SRC = {"fb": "data/freqblimp_original_passives/pairs.jsonl", "ve": "data/verb_expansion/verbs.csv",
       "c4": "data/round4/fillergap/candidates.csv", "cur": "data/round5/frame/pairs.csv",
       "a1": "data/round5/frame/verbs.csv", "fb_lemmas": str(OUT / "sources/fb_lemmas.csv")}
GUIDE = OUT / "rating_guide.md"
LABELS = ("T", "I", "reject")
CELLS = ("T1-I1", "T1-I2+", "T2+-I1", "T2+-I2+")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def band_of(z):
    for b, (lo, hi) in BANDS.items():
        if lo <= z <= hi:
            return b
    return ""


def das_files():
    import glob
    return sorted(f for pat, _, _ in DAS_SOURCES for f in glob.glob(pat))


def candidates(args):
    from wordfreq import zipf_frequency

    rows = []
    fb = pd.read_json(SRC["fb"], lines=True)
    lem = pd.read_csv(SRC["fb_lemmas"], keep_default_na=False).set_index("participle").lemma
    for role, col in (("T", "good_verb"), ("I", "bad_verb")):
        for (part, b, u), _ in fb.groupby([fb[col].str.lower(), "band", "paradigm"]):
            rows.append({"lemma": lem[part], "participle": part, "class_src": role, "source": f"fb_{u}", "src_band": b})
    ve = pd.read_csv(SRC["ve"], keep_default_na=False)
    rows += [{"lemma": r.lemma, "participle": r.participle, "class_src": "T" if r["class"] == "trans" else "I",
              "source": "verb_expansion", "src_band": r.band} for _, r in ve.iterrows()]
    c4 = pd.read_csv(SRC["c4"], keep_default_na=False)
    rows += [{"lemma": r.lemma, "participle": r.participle, "class_src": r.pool, "source": f"round4_{r.source}",
              "src_band": ""} for _, r in c4.iterrows()]
    cur = pd.read_csv(SRC["cur"], keep_default_na=False)
    for _, r in cur.iterrows():
        _, tl, il = r.verb_pair.split("/")
        rows += [{"lemma": tl, "participle": r.t_verb, "class_src": "T", "source": "curated", "src_band": r.band},
                 {"lemma": il, "participle": r.i_verb, "class_src": "I", "source": "curated", "src_band": r.band}]
    R = pd.DataFrame(rows)
    R["lemma"], R["participle"] = R.lemma.str.lower(), R.participle.str.lower()
    # one participle per lemma: FreqBLiMP's form first, then curated, verb-expansion, round 4
    pri = R.source.map(lambda s: 0 if s.startswith("fb") else 1 if s == "curated" else 2 if s == "verb_expansion" else 3)
    R = R.assign(pri=pri).sort_values(["lemma", "pri", "participle"])
    form = R.groupby("lemma").participle.first()
    alts = R.groupby("lemma").participle.agg(lambda x: " ".join(sorted(set(x))))
    agg = R.groupby("lemma").agg(class_src=("class_src", lambda x: "/".join(sorted(set(x)))),
                                 sources=("source", lambda x: " ".join(sorted(set(x)))),
                                 src_band=("src_band", lambda x: " ".join(sorted(set(y for y in x if y)))))
    C = agg.assign(participle=form, participles=alts).reset_index()
    C["part_zipf"] = C.participle.map(lambda w: round(zipf_frequency(w, "en"), 2))
    C["lemma_zipf"] = C.lemma.map(lambda w: round(zipf_frequency(w, "en"), 2))
    C["band"] = C.part_zipf.map(band_of)
    ntok = dict(zip(pd.read_csv(SRC["a1"], keep_default_na=False).verb, pd.read_csv(SRC["a1"]).n_tok))
    ntok.update({p: n for p, n in zip(ve.participle, ve.participle_ntok) if p not in ntok})
    extra = OUT / "sources/ntok_extra.csv"
    if extra.exists():
        ntok.update(dict(pd.read_csv(extra).set_index("participle").n_tok))
    C["n_tok"] = C.participle.map(ntok).astype("Int64")
    cur_l = set(R[R.source == "curated"].lemma)
    C["curated_verb"] = C.lemma.isin(cur_l).astype(int)
    hist = das_historical()
    C["das_historical"] = C.lemma.isin(hist).astype(int)
    C["das_sources"] = C.lemma.map(lambda l: " ".join(hist.get(l, [])))
    OUT.mkdir(parents=True, exist_ok=True)
    C.to_csv(OUT / "candidates.csv", index=False)
    missing = C[(C.band != "") & C.n_tok.isna()].participle.tolist()
    if missing:
        (OUT / "sources/ntok_needed.txt").write_text("\n".join(missing) + "\n")
        print(f"{len(missing)} in-band participles lack a token count: sources/ntok_needed.txt")
    inb = C[C.band != ""].sort_values("lemma").reset_index(drop=True)
    sheet = inb[["lemma", "participle"]].sample(frac=1, random_state=args.seed).reset_index(drop=True)
    sheet.insert(0, "item", np.arange(len(sheet)))
    sheet.to_csv(OUT / "rating_items.csv", index=False)
    import importlib.metadata as md
    man = {"inputs": {k: sha(v) for k, v in SRC.items()} | {f: sha(f) for f in das_files()}
           | {str(extra): sha(extra) if extra.exists() else None},
           "wordfreq": md.version("wordfreq"), "seed": args.seed,
           "outputs": {f: sha(OUT / f) for f in ("candidates.csv", "rating_items.csv")}}
    (OUT / "candidates_meta.json").write_text(json.dumps(man, indent=1))
    print(C.groupby(["band", "class_src"]).size().unstack(fill_value=0))
    print(f"rating items: {len(sheet)}")


def hungarian(cost):
    """Exact minimum-cost assignment for an integer cost matrix with rows <= cols (Kuhn-Munkres with potentials,
    int64 arithmetic). Returns the column assigned to each row."""
    cost = np.asarray(cost, dtype=np.int64)
    n, m = cost.shape
    assert n <= m
    INF = np.int64(2 ** 62)
    u, v = np.zeros(n + 1, np.int64), np.zeros(m + 1, np.int64)
    p, way = np.zeros(m + 1, np.int64), np.zeros(m + 1, np.int64)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = np.full(m + 1, INF, np.int64), np.zeros(m + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = cost[i0 - 1] - u[i0] - v[1:]
            free = ~used[1:]
            upd = free & (cur < minv[1:])
            minv[1:][upd], way[1:][upd] = cur[upd], j0
            cand = np.where(free, minv[1:], INF)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            ui = np.flatnonzero(used)
            u[p[ui]] += delta
            v[ui] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    ans = np.full(n, -1)
    for j in range(1, m + 1):
        if p[j]:
            ans[p[j] - 1] = j - 1
    return ans


def match_band(T, I):
    """Maximum number of one-to-one pairs within the calipers; among those, minimum total participle-Zipf gap (in
    hundredths); among those, lemma order: maximum sum of alphabetical-rank products, i.e. pairs aligned in
    alphabetical order. T and I must be sorted by lemma. Exact (integer costs)."""
    if len(T) == 0 or len(I) == 0:
        return []
    gp = np.rint(np.abs(T.part_zipf.to_numpy()[:, None] - I.part_zipf.to_numpy()[None, :]) * 100).astype(np.int64)
    gl = np.rint(np.abs(T.lemma_zipf.to_numpy()[:, None] - I.lemma_zipf.to_numpy()[None, :]) * 100).astype(np.int64)
    ok = (gp <= round(CAL_PART * 100)) & (gl <= round(CAL_LEMMA * 100))
    tie = np.arange(len(T))[:, None] * np.arange(len(I))[None, :]
    n = min(len(T), len(I))
    M = int(tie.max()) * n + 1  # one hundredth of gap outweighs any tie-break total
    BIG = (int(gp.max()) * n + 1) * M  # one more pair outweighs any gap total
    cost = np.where(ok, -BIG + gp * M - tie, 0)
    flip = len(T) > len(I)
    a = hungarian(cost.T if flip else cost)
    pairs = []
    for r, c in enumerate(a):
        t, i = (c, r) if flip else (r, c)
        if c >= 0 and ok[t, i]:
            pairs.append((t, i))
    return sorted(pairs)


def match(args):
    C = pd.read_csv(OUT / "candidates.csv", keep_default_na=False)
    items = pd.read_csv(OUT / "rating_items.csv", keep_default_na=False)
    man = json.loads((OUT / "candidates_meta.json").read_text())
    assert sha(OUT / "rating_items.csv") == man["outputs"]["rating_items.csv"], "rating items changed since freezing"
    assert sha(OUT / "candidates.csv") == man["outputs"]["candidates.csv"], "candidates changed since freezing"
    for who in ("claude", "codex"):
        r = pd.read_csv(OUT / f"rating_{who}.csv", keep_default_na=False)
        assert sorted(r.item) == sorted(items.item), who
        assert set(r.label) <= set(LABELS), (who, set(r.label) - set(LABELS))
        r = r.merge(items, on="item")
        C = C.merge(r[["lemma", "label", "flag", "reason"]].rename(columns={"label": f"{who}_label", "flag": f"{who}_flag",
                                                                         "reason": f"{who}_reason"}), on="lemma", how="left")
    C["accepted"] = np.where((C.claude_label == C.codex_label) & C.claude_label.isin(["T", "I"]), C.claude_label, "")
    C["n_tok"] = pd.to_numeric(C.n_tok, errors="coerce")
    rows = []
    for b in BANDS:
        T = C[(C.band == b) & (C.accepted == "T")].sort_values("lemma").reset_index(drop=True)
        I = C[(C.band == b) & (C.accepted == "I")].sort_values("lemma").reset_index(drop=True)
        for t, i in match_band(T, I):
            rt, ri = T.iloc[t], I.iloc[i]
            rows.append({"band": b, "t_lemma": rt.lemma, "t_participle": rt.participle, "i_lemma": ri.lemma,
                         "i_participle": ri.participle, "t_part_zipf": rt.part_zipf, "i_part_zipf": ri.part_zipf,
                         "t_lemma_zipf": rt.lemma_zipf, "i_lemma_zipf": ri.lemma_zipf,
                         "part_gap": round(abs(rt.part_zipf - ri.part_zipf), 2),
                         "lemma_gap": round(abs(rt.lemma_zipf - ri.lemma_zipf), 2),
                         "t_ntok": int(rt.n_tok), "i_ntok": int(ri.n_tok),
                         "t_single": int(rt.n_tok == 1), "i_single": int(ri.n_tok == 1),
                         "t_flag": f"{rt.claude_flag}|{rt.codex_flag}", "i_flag": f"{ri.claude_flag}|{ri.codex_flag}",
                         "t_curated": int(rt.curated_verb), "i_curated": int(ri.curated_verb),
                         "das_historical": int(rt.das_historical or ri.das_historical)})
    cols = ["band", "t_lemma", "t_participle", "i_lemma", "i_participle", "t_part_zipf", "i_part_zipf", "t_lemma_zipf",
            "i_lemma_zipf", "part_gap", "lemma_gap", "t_ntok", "i_ntok", "t_single", "i_single", "t_flag", "i_flag",
            "t_curated", "i_curated", "das_historical"]
    P = pd.DataFrame(rows, columns=cols)
    # token-count cell: T single / multi x I single / multi (orientation kept)
    P["tok_cell"] = np.where(P.t_single == 1, "T1", "T2+") + np.where(P.i_single == 1, "-I1", "-I2+")
    rng = np.random.default_rng(args.seed)
    P["split"] = "trainable"
    for (b, cell), g in P[P.das_historical == 0].sort_values(["band", "tok_cell", "t_lemma"]).groupby(["band", "tok_cell"],
                                                                                                         sort=True):
        k = int(round(len(g) / 3))
        P.loc[rng.permutation(g.index.to_numpy())[:k], "split"] = "heldout"
    P = P.sort_values(["band", "split", "t_lemma"]).reset_index(drop=True)
    P.insert(0, "pair_id", [f"{b}/{t}/{i}" for b, t, i in zip(P.band, P.t_lemma, P.i_lemma)])
    P.to_csv(OUT / "pairs.csv", index=False)
    C.to_csv(OUT / "verbs.csv", index=False)
    from verbs_expanded import load_pairs  # loader checks
    load_pairs("heldout"), load_pairs("trainable")

    po = (C.claude_label == C.codex_label)[C.band != ""].mean()
    rated = C[C.band != ""]
    cats = list(LABELS)
    pc = sum((rated.claude_label == c).mean() * (rated.codex_label == c).mean() for c in cats)
    kappa = (po - pc) / (1 - pc)
    by = P.groupby("band").size().reindex(BANDS, fill_value=0)
    ok = len(P) >= 128 and (by >= 40).all()
    import importlib.metadata as md
    meta = {"candidates_manifest": man, "guide_sha256": sha(GUIDE), "wordfreq": md.version("wordfreq"),
            "freqblimp_tsubame_sha256": json.loads((OUT / "sources/freqblimp_tsubame_sha256.json").read_text()),
            "calipers": {"participle": CAL_PART, "lemma": CAL_LEMMA}, "bands": BANDS,
            "seed": args.seed, "rated": int(len(rated)), "agreement": float(po), "kappa": float(kappa),
            "accepted": C[C.accepted != ""].groupby(["band", "accepted"]).size().unstack(fill_value=0).to_dict(),
            "pairs": int(len(P)), "pairs_by_band": by.to_dict(), "success": bool(ok),
            "ratings_sha256": {w: sha(OUT / f"rating_{w}.csv") for w in ("claude", "codex")}}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    report(C, P, meta, args)


def report(C, P, meta, args):
    rated = C[C.band != ""]
    L = ["# Round 5, D11: expanded verb list", "",
         "Spec: `plan.md`, D11. Built by `build_verbs_expanded.py` (no model). Outputs: "
         "`data/round5/verbs_expanded/{candidates,verbs,pairs}.csv`, `meta.json`; loader `scripts/verbs_expanded.py`.", "",
         "## Candidates and ratings", "",
         f"Sources pooled by lemma: {len(C)} lemmas, {len(rated)} with the participle in a FreqBLiMP band (rated). "
         f"Both raters labelled each rated lemma T / I / reject, blind to the source class and to each other. "
         f"Agreement {meta['agreement']:.2f}, Cohen's κ = {meta['kappa']:.2f}.", "",
         "| band | candidates | both T | both I | disagree | both reject |", "|---|---:|---:|---:|---:|---:|"]
    for b in BANDS:
        g = rated[rated.band == b]
        L.append(f"| {b} | {len(g)} | {(g.accepted == 'T').sum()} | {(g.accepted == 'I').sum()} | "
                 f"{(g.claude_label != g.codex_label).sum()} | {((g.claude_label == 'reject') & (g.codex_label == 'reject')).sum()} |")
    conf = pd.crosstab(rated.claude_label, rated.codex_label)
    L += ["", "Confusion (rows Claude, columns Codex):", "", "| Claude \\ Codex | " + " | ".join(conf.columns) + " |",
          "|---|" + "---:|" * len(conf.columns)]
    L += [f"| {i} | " + " | ".join(str(v) for v in r) + " |" for i, r in zip(conf.index, conf.to_numpy())]
    L += ["", "## Pairs", "",
          f"Matching within band: maximum one-to-one pairs within the calipers (participle Zipf ≤ {CAL_PART}, lemma "
          f"Zipf ≤ {CAL_LEMMA}), then minimum total participle-Zipf gap.", "",
          "| band | pairs | T1-I1 / T1-I2+ / T2+-I1 / T2+-I2+ | curated verb in pair | DAS-historical | held-out | mean part. gap | mean lemma gap |",
          "|---|---:|---|---:|---:|---:|---:|---:|"]
    for b in BANDS:
        g = P[P.band == b]
        cells = " / ".join(str((g.tok_cell == c).sum()) for c in CELLS)
        L.append(f"| {b} | {len(g)} | {cells} | {((g.t_curated + g.i_curated) > 0).sum()} | {g.das_historical.sum()} | "
                 f"{(g.split == 'heldout').sum()} | {g.part_gap.mean():.2f} | {g.lemma_gap.mean():.2f} |")
    L += [f"| all | {len(P)} | " + " / ".join(str((P.tok_cell == c).sum()) for c in CELLS)
          + f" | {((P.t_curated + P.i_curated) > 0).sum()} | {P.das_historical.sum()} | {(P.split == 'heldout').sum()} | "
          f"{P.part_gap.mean():.2f} | {P.lemma_gap.mean():.2f} |", "",
          f"**Rule** (≥ 128 pairs and ≥ 40 per band): **{'met' if meta['success'] else 'not met'}**."]
    if not meta["success"]:
        short = {b: max(0, 40 - int(meta["pairs_by_band"][b])) for b in BANDS}
        L.append(f"Shortfall per band (to 40): {short}; total shortfall to 128: {max(0, 128 - len(P))}.")
    unp = {b: {c: int(((C.band == b) & (C.accepted == c)).sum() - (P.band == b).sum()) for c in ("T", "I")} for b in BANDS}
    L += ["", f"Accepted but unpaired (T, I per band): {unp}.", ""]
    Path(args.report).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("candidates", "match"))
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--report", default="reports/round5/d11_verbs.md")
    a = ap.parse_args()
    candidates(a) if a.stage == "candidates" else match(a)
