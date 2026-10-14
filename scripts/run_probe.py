#!/usr/bin/env python3
"""Round 5, A5: hidden-state grammaticality probe on Pythia-1.4B (reports/round5/plan.md, A5). fp32.

Training data: data/round5/probe/pairs.csv (build_probe_corpus.py): original (label 1) and perturbed (label 0).
Features: the last token's residual stream after layer l, l = 1..24 (hidden_states[l] for l < 24; the input of the
final LayerNorm for l = 24; the final-LN output is not used); no BOS; right padding.
Probe (Wang et al. 2026): per layer, z-scored with training statistics, L2 logistic regression (scikit-learn lbfgs,
1,000 iterations), C in 2^-5..2^2 chosen by dev AUC; the layer with the best dev AUC (split 0) is used. Splits 1 and
2 (other document splits) re-select C at that layer (stability).
Baselines on dev: logistic regression on length / punctuation counts; the LM's sentence log-prob (sum and per token).
Scoring: every FreqBLiMP pair (67 paradigms x 3 bands) and the curated band-cross passive_1 / passive_2 pairs: probe
logit of each sentence (three splits' probes) and the sentence log-prob (paper LP: all tokens but the first).

Outputs (--out-dir): probe_train.csv (dev AUC per layer x C), probe_meta.json, probe_freqblimp.parquet,
probe_curated.parquet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import string
import time
from pathlib import Path

import numpy as np
import pandas as pd

CGRID = [2.0 ** e for e in range(-5, 3)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Feats:
    def __init__(self, args):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.t = torch
        self.dev = torch.device(args.device)
        self.tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                          local_files_only=True).to(self.dev).eval()
        self.L = self.model.config.num_hidden_layers

    def run(self, sents, layers=None, batch=128):
        """Returns states [N, len(layers), d] (float16), sentence LP [N], token counts [N]."""
        t = self.t
        layers = list(range(1, self.L + 1)) if layers is None else layers
        S = np.empty((len(sents), len(layers), self.model.config.hidden_size), np.float16)
        lpo, nto = np.empty(len(sents)), np.empty(len(sents), int)
        cap = {}
        h = self.model.gpt_neox.final_layer_norm.register_forward_pre_hook(lambda m, i: cap.__setitem__("pre", i[0]))
        try:
            for i in range(0, len(sents), batch):
                enc = self.tok(sents[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(self.dev)
                with t.inference_mode():
                    o = self.model(**enc, output_hidden_states=True, use_cache=False)
                anc = enc.attention_mask.sum(1) - 1
                rows = t.arange(len(anc), device=self.dev)
                hs = [o.hidden_states[l] if l < self.L else cap["pre"] for l in layers]
                S[i:i + len(anc)] = t.stack([x[rows, anc].float() for x in hs], 1).half().cpu().numpy()
                lp = t.log_softmax(o.logits[:, :-1].float(), -1).gather(-1, enc.input_ids[:, 1:, None])[..., 0]
                lpo[i:i + len(anc)] = (lp * enc.attention_mask[:, 1:]).sum(1).double().cpu().numpy()
                nto[i:i + len(anc)] = enc.attention_mask.sum(1).cpu().numpy()
        finally:
            h.remove()
        return S, lpo, nto


def auc(y, s):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def fit_layer(X, y, tr, dv):
    from sklearn.linear_model import LogisticRegression

    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Z = (X - mu) / sd
    out, models = [], {}
    clf = LogisticRegression(solver="lbfgs", max_iter=1000, warm_start=True)
    for C in CGRID:
        clf.set_params(C=C)
        clf.fit(Z[tr], y[tr])
        out.append((C, auc(y[dv], clf.decision_function(Z[dv]))))
        models[C] = (clf.coef_.copy(), clf.intercept_.copy())
    return out, models, (mu, sd)


def run(args):
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(args.pairs)
    sents = list(P.original) + list(P.perturbed)
    y = np.r_[np.ones(len(P)), np.zeros(len(P))].astype(int)
    split = {k: np.r_[P[f"split{k}"].to_numpy(), P[f"split{k}"].to_numpy()] for k in range(3)}
    F = Feats(args)
    S, lp, nt = F.run(sents, batch=args.batch)
    print(f"features {S.shape} in {time.time() - t0:.0f}s", flush=True)
    rows = []
    tr0, dv0 = split[0] == "train", split[0] == "dev"
    best = (-1, None, None)
    for j in range(S.shape[1]):
        X = S[:, j].astype(np.float32)
        res, models, norm = fit_layer(X, y, tr0, dv0)
        for C, a in res:
            rows.append({"split": 0, "layer": j + 1, "C": C, "dev_auc": a})
        Cb, ab = max(res, key=lambda r: r[1])
        if ab > best[0]:
            best = (ab, j, (Cb, models[Cb], norm))
        print(f"layer {j + 1}: best C {Cb}, dev AUC {ab:.4f} ({time.time() - t0:.0f}s)", flush=True)
    layer = best[1] + 1
    probes = {0: best[2]}
    X = S[:, best[1]].astype(np.float32)
    for k in (1, 2):
        res, models, norm = fit_layer(X, y, split[k] == "train", split[k] == "dev")
        for C, a in res:
            rows.append({"split": k, "layer": layer, "C": C, "dev_auc": a})
        Cb, _ = max(res, key=lambda r: r[1])
        probes[k] = (Cb, models[Cb], norm)
    pd.DataFrame(rows).to_csv(out / "probe_train.csv", index=False)
    # baselines on split-0 dev
    from sklearn.linear_model import LogisticRegression
    punct = np.array([sum(ch in string.punctuation for ch in s) for s in sents])
    words = np.array([len(s.split()) for s in sents])
    B = np.column_stack([nt, words, punct]).astype(float)
    Bz = (B - B[tr0].mean(0)) / (B[tr0].std(0) + 1e-6)
    lb = LogisticRegression(max_iter=1000).fit(Bz[tr0], y[tr0])
    meta = {"args": vars(args), "pairs_sha256": sha(args.pairs), "n_sentences": len(sents), "layer": layer,
            "dev_auc_best": best[0], "C": {k: v[0] for k, v in probes.items()},
            "baselines_dev_auc": {"length_punct": auc(y[dv0], lb.decision_function(Bz[dv0])),
                                  "lm_logprob": auc(y[dv0], lp[dv0]), "lm_logprob_per_token": auc(y[dv0], lp[dv0] / (nt[dv0] - 1))}}
    print(json.dumps(meta), flush=True)
    del S

    def logits(states):
        return {k: (((states.astype(np.float32) - nm[0]) / nm[1]) @ w[0].T + w[1]).ravel()
                for k, (C, w, nm) in probes.items()}

    def score(df, gcol, bcol):
        res = {}
        for side, col in (("good", gcol), ("bad", bcol)):
            st, lps, _ = F.run(df[col].tolist(), layers=[layer], batch=args.batch)
            lg = logits(st[:, 0])
            for k in range(3):
                res[f"probe{k}_{side}"] = lg[k]
            res[f"lp_{side}"] = lps
        return res

    fq = []
    for band in ("head", "tail", "xtail"):
        for f in sorted((Path(args.freqblimp) / band).glob("*.jsonl")):
            d = pd.read_json(f, lines=True)
            fq.append(pd.DataFrame({"band": band, "uid": f.stem, "pairID": d.pairID.astype(str),
                                    "good": d.sentence_good, "bad": d.sentence_bad}))
    fq = pd.concat(fq, ignore_index=True)
    sc = score(fq, "good", "bad")
    pd.concat([fq, pd.DataFrame(sc)], axis=1).to_parquet(out / "probe_freqblimp.parquet", index=False)
    print(f"FreqBLiMP scored ({time.time() - t0:.0f}s)", flush=True)
    bca = pd.read_json(args.band_cross, lines=True)
    cur = bca[bca.context_set == "curated"].reset_index(drop=True)
    sc = score(cur, "sentence_good", "sentence_bad")
    keep = cur[["pair_id", "verb_pair", "verb_band", "context_id", "context_band", "paradigm", "own_context",
                "good_verb", "bad_verb"]]
    pd.concat([keep, pd.DataFrame(sc)], axis=1).to_parquet(out / "probe_curated.parquet", index=False)
    meta["seconds"] = round(time.time() - t0)
    (out / "probe_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="data/round5/probe/pairs.csv")
    ap.add_argument("--freqblimp", required=True)
    ap.add_argument("--band-cross", default="data/passive_band_cross/pairs.jsonl")
    ap.add_argument("--out-dir", default="results/round5/probe")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=128)
    run(ap.parse_args())
