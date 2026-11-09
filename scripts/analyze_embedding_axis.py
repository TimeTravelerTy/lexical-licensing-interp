#!/usr/bin/env python3
"""Gemma replication, exploratory: is the transitivity class axis already in the token embeddings?

Verbs: the DAS verbs of both models' final runs (items.csv of results/das_round2/final_strict and
results/gemma/das/transfer/final_strict, i.e. after each model's behaviour filter) whose participle is a single token
in BOTH tokenizers, in the DAS prompt context ("She has <participle>": tokens of the prompt minus tokens of
"She has", no special tokens). Each verb's vector is the row of that token.

Embedding matrices: Gemma-3-1B input embedding (tied with the unembedding; the raw weight rows, i.e. without the
sqrt(d) factor Gemma3TextScaledWordEmbedding applies, which changes no direction); Pythia-1.4B embed_in and embed_out
separately (untied). All in fp32 (cast from the stored weights).

(a) Verb-held-out probe, T (cls 1) vs I (cls 0). Folds: each model's own DAS folds (pairs_folds.csv, fold_split0-2:
    a verb is held out in the fold of its pair), 3 splits x 5 folds. Classifiers fit on the fold's training verbs:
    - logistic: L2 logistic regression (C = 1, lbfgs), features standardized on the training verbs;
    - mean-diff: projection on (mean T - mean I) of the training verbs, threshold at the midpoint of the two class
      means' projections.
    Accuracy = held-out correct / held-out verbs, pooled over a split's 5 folds; reported as the mean over the 3
    splits. 95% CI: verbs resampled with replacement (2,000 draws, seed 17) on each verb's accuracy averaged over the
    3 splits. Null: the same procedure on 200 permutations of the verbs' labels (seed 23); its mean and 95th pct.
    Supplementary (added after a test run showed the bootstrap collapsing at 26 / 26 correct): an exact
    Clopper-Pearson 95% interval for k = round(mean accuracy x n) correct of n verbs.
(b) Class axis u = unit(mean T - mean I) over all intersection verbs, per embedding matrix. |cos(u, b)| with the
    rank-1 DAS direction b of every split x fold (15) of the model's final run at each early site (Gemma 0-5;
    Pythia 0-8: stored final runs at 4 / 6 / 8, exploratory early-site runs at 0-3, 5, 7). The DAS direction's sign is
    arbitrary, hence |cos|. References per site, over the same 15 directions:
    - random: 10,000 Gaussian random unit directions (seed 29): mean |cos| and 95th pct; expected sqrt(2 / (pi d))
      for the mean, 1 / sqrt(d) for the rms;
    - shuffled-label axis: unit(mean - mean) of the same verbs under 1,000 random label permutations with the class
      sizes kept (seed 31): mean |cos| and 95th pct (controls for embedding anisotropy).

Writes <out-dir>/{verbs.csv, probe.csv, cosines.csv, meta.json} and a generated table <report>.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {"Gemma-3-1B": "google/gemma-3-1b-pt", "Pythia-1.4B": "EleutherAI/pythia-1.4b"}


def participle_token(tok, prompt):
    words = prompt.split()
    prefix = " ".join(words[:-1])
    full = tok.encode(prompt, add_special_tokens=False)
    pre = tok.encode(prefix, add_special_tokens=False)
    assert full[:len(pre)] == pre, (prompt, full, pre)
    return full[len(pre):]


def verb_table(items, toks):
    v = items[items.subject == "She"].drop_duplicates("verb").set_index("verb")
    rows = []
    for verb, r in v.iterrows():
        row = {"verb": verb, "cls": int(r.cls), "pair_id": r.pair_id, "participle": r.prompt.split()[-1]}
        for m, tk in toks.items():
            ids = participle_token(tk, r.prompt)
            row[f"ntok_{m}"], row[f"id_{m}"] = len(ids), ids[-1]
        rows.append(row)
    return pd.DataFrame(rows)


def folds_for(verbs, pairs_folds):
    pf = pd.read_csv(pairs_folds).set_index("pair_id")
    return {s: verbs.pair_id.map(pf[f"fold_split{s}"]).astype(int).to_numpy() for s in range(3)}


def cv_correct(X, y, folds, kind):
    """Per split, a 0/1 vector of held-out correctness for every verb."""
    out = []
    for s in range(3):
        c = np.zeros(len(y))
        for f in np.unique(folds[s]):
            te, tr = folds[s] == f, folds[s] != f
            if kind == "logistic":
                sc = StandardScaler().fit(X[tr])
                clf = LogisticRegression(C=1.0, max_iter=5000).fit(sc.transform(X[tr]), y[tr])
                pred = clf.predict(sc.transform(X[te]))
            else:
                mT, mI = X[tr][y[tr] == 1].mean(0), X[tr][y[tr] == 0].mean(0)
                w = mT - mI
                pred = (X[te] @ w > (mT @ w + mI @ w) / 2).astype(int)
            c[te] = pred == y[te]
        out.append(c)
    return np.array(out)


def probe(X, y, folds, kind, n_boot=2000, n_perm=200):
    c = cv_correct(X, y, folds, kind)
    acc = c.mean(1)
    per_verb = c.mean(0)
    rng = np.random.default_rng(17)
    boot = np.array([per_verb[rng.integers(len(y), size=len(y))].mean() for _ in range(n_boot)])
    rng = np.random.default_rng(23)
    null = np.array([cv_correct(X, rng.permutation(y), folds, kind).mean() for _ in range(n_perm)])
    n, k = len(y), int(round(acc.mean() * len(y)))
    cp = (beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0, beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)
    return {"acc": acc.mean(), "cp_k": k, "cp_lo95": cp[0], "cp_hi95": cp[1], "acc_split0": acc[0], "acc_split1": acc[1], "acc_split2": acc[2],
            "lo95": np.quantile(boot, 0.025), "hi95": np.quantile(boot, 0.975),
            "null_mean": null.mean(), "null_p95": np.quantile(null, 0.95)}


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def load_bases(d):
    b = torch.load(Path(d) / "bases.pt", map_location="cpu")
    keys = sorted(k for k in b if k.startswith("r1_"))
    assert len(keys) == 15, (d, keys)
    return keys, unit(np.stack([b[k][:, 0].float().numpy() for k in keys]))


def main(a):
    toks = {m: AutoTokenizer.from_pretrained(p, local_files_only=True) for m, p in MODELS.items()}
    items = {"Gemma-3-1B": pd.read_csv(Path(a.gemma_final) / "items.csv"),
             "Pythia-1.4B": pd.read_csv(Path(a.pythia_final) / "items.csv")}
    vt = {m: verb_table(it, toks) for m, it in items.items()}
    common = sorted(set(vt["Gemma-3-1B"].verb) & set(vt["Pythia-1.4B"].verb))
    V = vt["Pythia-1.4B"].set_index("verb").loc[common]
    G = vt["Gemma-3-1B"].set_index("verb").loc[common]
    assert (V.cls == G.cls).all() and (V[[c for c in V if c.startswith(("ntok", "id"))]] ==
                                       G[[c for c in G if c.startswith(("ntok", "id"))]]).all().all()
    V = V.reset_index()
    V["single_both"] = (V["ntok_Gemma-3-1B"] == 1) & (V["ntok_Pythia-1.4B"] == 1)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    V.to_csv(out / "verbs.csv", index=False)
    S = V[V.single_both].reset_index(drop=True)
    y = S.cls.to_numpy()
    print(f"verbs in both DAS sets {len(V)}; single-token in both tokenizers {len(S)} (T {y.sum()}, I {(1 - y).sum()})")

    mats = {}
    for m, p in MODELS.items():
        model = AutoModelForCausalLM.from_pretrained(p, torch_dtype=torch.float32, local_files_only=True)
        ids = S[f"id_{m}"].to_numpy()
        if m == "Gemma-3-1B":
            E, U = model.get_input_embeddings().weight, model.get_output_embeddings().weight
            assert E.data_ptr() == U.data_ptr(), "Gemma embedding expected tied"
            mats[(m, "embedding (tied)")] = E.detach().numpy()[ids]
        else:
            mats[(m, "embed_in")] = model.gpt_neox.embed_in.weight.detach().numpy()[ids]
            mats[(m, "embed_out")] = model.embed_out.weight.detach().numpy()[ids]
        del model

    folds = {"Gemma-3-1B": folds_for(S, Path(a.gemma_final) / "pairs_folds.csv"),
             "Pythia-1.4B": folds_for(S, Path(a.pythia_final) / "pairs_folds.csv")}
    P = []
    for (m, mat), X in mats.items():
        for kind in ("logistic", "mean-diff"):
            P.append({"model": m, "matrix": mat, "classifier": kind, "n_verbs": len(y), "n_T": int(y.sum()),
                      "majority": max(y.mean(), 1 - y.mean()), **probe(X, y, folds[m], kind)})
    P = pd.DataFrame(P)
    P.to_csv(out / "probe.csv", index=False)

    site_dirs = {"Gemma-3-1B": {s: f"{a.gemma_root}/final_strict_site{s}" for s in range(6)},
                 "Pythia-1.4B": {s: (f"{a.pythia_root}/final_strict_site{s}" if s in (4, 6, 8)
                                     else f"{a.pythia_early}/final_strict_site{s}") for s in range(9)}}
    rng_r, rng_p = np.random.default_rng(29), np.random.default_rng(31)
    C = []
    for (m, mat), X in mats.items():
        u = unit(X[y == 1].mean(0) - X[y == 0].mean(0))
        d = X.shape[1]
        R = unit(rng_r.standard_normal((10000, d)))
        perm = np.array([unit(X[pp == 1].mean(0) - X[pp == 0].mean(0))
                         for pp in (rng_p.permutation(y) for _ in range(1000))])
        for s, sd in site_dirs[m].items():
            if not (Path(sd) / "bases.pt").exists():
                print(f"missing {sd}")
                continue
            keys, B = load_bases(sd)
            c = np.abs(B @ u)
            rc, pc = np.abs(R @ B.T), np.abs(perm @ B.T)
            C.append({"model": m, "matrix": mat, "site": s, "d": d, "n_dirs": len(keys),
                      "abs_cos_mean": c.mean(), "abs_cos_min": c.min(), "abs_cos_max": c.max(),
                      "random_mean": rc.mean(), "random_p95": np.quantile(rc, 0.95),
                      "expected_mean": np.sqrt(2 / (np.pi * d)), "inv_sqrt_d": 1 / np.sqrt(d),
                      "shuffled_axis_mean": pc.mean(), "shuffled_axis_p95": np.quantile(pc, 0.95),
                      "bases": sd})
    C = pd.DataFrame(C)
    C.to_csv(out / "cosines.csv", index=False)
    (out / "meta.json").write_text(json.dumps({"n_verbs_both_das": len(V), "n_single_both": len(S),
                                               "verbs": S.verb.tolist(), "args": vars(a)}, indent=2))

    f = lambda x: f"{x:.3f}"
    L = ["# Embedding-level class axis (EXPLORATORY; generated by `scripts/analyze_embedding_axis.py`)", "",
         f"Verbs: {len(S)} DAS verbs (T {int(y.sum())}, I {int(len(y) - y.sum())}) whose participle is one token in "
         "both tokenizers; definitions in the script's docstring.", "", "## (a) Verb-held-out probe", "",
         "| model | matrix | classifier | n | majority | accuracy [95% CI, verb bootstrap] | Clopper-Pearson 95% "
         "(k / n) | splits | label-shuffled null mean / p95 |",
         "|---|---|---|---:|---:|---|---|---|---|"]
    for _, r in P.iterrows():
        L.append(f"| {r.model} | {r.matrix} | {r.classifier} | {r.n_verbs} | {f(r.majority)} | {f(r.acc)} "
                 f"[{f(r.lo95)}, {f(r.hi95)}] | [{f(r.cp_lo95)}, {f(r.cp_hi95)}] ({r.cp_k} / {r.n_verbs}) | {f(r.acc_split0)} / {f(r.acc_split1)} / {f(r.acc_split2)} | "
                 f"{f(r.null_mean)} / {f(r.null_p95)} |")
    L += ["", "## (b) |cos| between the class mean-difference axis and the DAS direction", "",
          "| model | matrix | site | \\|cos\\| mean (min-max, 15 folds) | random mean / p95 | sqrt(2/(pi d)) / 1/sqrt(d) | "
          "shuffled-label axis mean / p95 |", "|---|---|---:|---|---|---|---|"]
    for _, r in C.iterrows():
        L.append(f"| {r.model} | {r.matrix} | {r.site} | {f(r.abs_cos_mean)} ({f(r.abs_cos_min)}-{f(r.abs_cos_max)}) | "
                 f"{f(r.random_mean)} / {f(r.random_p95)} | {f(r.expected_mean)} / {f(r.inv_sqrt_d)} | "
                 f"{f(r.shuffled_axis_mean)} / {f(r.shuffled_axis_p95)} |")
    Path(a.report).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gemma-final", default="results/gemma/das/transfer/final_strict")
    ap.add_argument("--pythia-final", default="results/das_round2/final_strict")
    ap.add_argument("--gemma-root", default="results/gemma/das/transfer")
    ap.add_argument("--pythia-root", default="results/das_round2")
    ap.add_argument("--pythia-early", default="results/gemma/das/pythia_early")
    ap.add_argument("--out-dir", default="results/gemma/das/embedding_axis")
    ap.add_argument("--report", default="reports/gemma/embedding_axis.md")
    main(ap.parse_args())
