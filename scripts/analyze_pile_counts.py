#!/usr/bin/env python3
"""Round 5, A3 analysis: does passive-slot frequency predict Delta_pass and L better than overall frequency?
(reports/round5/plan.md, A3). No model.

Inputs: data/round5/counts/pile_counts.csv (fetch_pile_counts.py), data/round5/frame/{verbs,pairs,prefixes}.csv,
results/round5/frame/frame_scores.parquet (A1: Delta_pass per verb x freq context).
Outcome per verb: Delta_pass(V) = log P(V | "The N was") - log P(V | "The N has"), mean over the 24 freq contexts.
Predictors: Zipf (wordfreq), log10(c + s), log10(c_be + s) (primary slot predictor), rho_be = log((c_be + s)/(c + s));
s = 0.5 (sensitivity 0.1, 1). Per class (T, I); curated verbs primary, pool verbs secondary; L per pair with T and I
predictors entered separately.
Decision: 10-fold CV (folds grouped by verb pair; pool: by verb) error of ~ log c + log c_be vs ~ log c, in every
draw of a verbs-within-band x contexts bootstrap (2,000 draws, seed 17).
Outputs: reports/round5/a3_counts.md, results/round5/counts/a3_summary.csv, a3_decisions.json, a3_verbs.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BE = ("was", "were", "been", "is", "are", "be")
PERF = ("has", "have", "had")


def weights(rng, labels, n_boot):
    labels = np.asarray(labels)
    w = np.ones((n_boot + 1, len(labels)))
    for lab in np.unique(labels):
        idx = np.flatnonzero(labels == lab)
        w[1:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1 / len(idx)), size=n_boot)
    return w


def ci(d, lo=2.5, hi=97.5):
    return np.percentile(d[1:], lo), np.percentile(d[1:], hi)


def fmt(d, p=3):
    lo, hi = ci(d)
    return f"{d[0]:+.{p}f} [{lo:+.{p}f}, {hi:+.{p}f}]"


def wls(X, y, w):
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return beta


def cv_mse(X, y, w, groups, folds):
    """Weighted 10-fold CV MSE; units with weight 0 are absent; groups stay within one fold."""
    err, tot = 0.0, 0.0
    for k in range(folds):
        te = groups % folds == k
        tr = ~te & (w > 0)
        tw = w * te
        if tw.sum() == 0 or tr.sum() < X.shape[1] + 1:
            continue
        b = wls(X[tr], y[tr], w[tr])
        err += float((tw * (y - X @ b) ** 2).sum())
        tot += float(tw.sum())
    return err / tot


def design(df, cols):
    return np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in cols])


def main(args):
    C = pd.read_csv(args.counts)
    V = pd.read_csv(args.data_dir / "verbs.csv", keep_default_na=False)
    pairs = pd.read_csv(args.data_dir / "pairs.csv", keep_default_na=False)
    pre = pd.read_csv(args.data_dir / "prefixes.csv", keep_default_na=False)
    S = pd.read_parquet(args.frame)
    assert C["count"].notna().all(), f"{C['count'].isna().sum()} failed queries"
    # audit: every slot query's token ids end with the overall query's ids
    ov = C[C.slot == "overall"].set_index("verb").token_ids
    C["suffix_ok"] = [str(t).split()[-len(str(ov[v]).split()):] == str(ov[v]).split() for v, t in zip(C.verb, C.token_ids)]
    bad = sorted(set(C[~C.suffix_ok].verb))
    W = C.pivot_table(index="verb", columns="slot", values="count", aggfunc="first")
    W["c"] = W["overall"]
    W["c_be"] = W[[f"be_{x}" for x in BE]].sum(1)
    W["c_perf"] = W[[f"perf_{x}" for x in PERF]].sum(1)
    W["c_be_cap"] = W["c_be"] + W[[f"capbe_{x}" for x in BE]].sum(1, min_count=len(BE))  # curated verbs only
    W["c_be_wasbeen"] = W["be_was"] + W["be_been"]
    V = V.join(W[["c", "c_be", "c_perf", "c_be_cap", "c_be_wasbeen"] + [f"be_{x}" for x in BE]], on="verb")
    V = V[~V.verb.isin(bad)].reset_index(drop=True)
    V["bandkey"] = V.band.str.split("|").str[0]
    # outcome: Delta_pass per verb x freq context
    ctx = list(pre[pre.set == "freq"].context_id)
    s = S[S.set == "freq"]
    lp = s.pivot_table(index=["verb", "frame"], columns="context_id", values="lp")[ctx]
    Dp = (lp.xs("pass", level="frame") - lp.xs("act", level="frame")).loc[V.verb].to_numpy()  # [verb, ctx]
    anim = pre.set_index("context_id").loc[ctx].animacy.to_numpy()
    rng = np.random.default_rng(args.seed)
    wc = weights(rng, anim, args.n_boot)
    Y = (wc @ Dp.T) / wc.sum(1, keepdims=True)  # [draw, verb]
    V["dpass"] = Y[0]
    lines = ["# Round 5, A3: training-data predictor (Pile counts)", "",
             "Spec: `plan.md`, A3. Counts: infini-gram `v4_piletrain_llama` (`fetch_pile_counts.py`); analysis: "
             "`analyze_pile_counts.py`. Outcome: Δ_pass(V) (A1, freq frames). Decision: 10-fold CV error (folds grouped by "
             "pair) of ~ log c + log c_be vs ~ log c; verbs within band × contexts bootstrap.", "",
             f"Audit: {len(bad)} verbs fail the token-suffix check and are dropped: {bad or 'none'}.", ""]
    rows, dec = [], {}

    def add_preds(df, sm):
        df = df.copy()
        df["logc"] = np.log10(df.c + sm)
        df["logcbe"] = np.log10(df.c_be + sm)
        df["rho"] = np.log((df.c_be + sm) / (df.c + sm))
        df["logcbe_cap"] = np.log10(df.c_be_cap + sm)
        df["logcbe_wb"] = np.log10(df.c_be_wasbeen + sm)
        return df

    cur_pair = {v: p for v, p in zip(V.verb, V.verb_pair)}
    for pop in ("curated", "pool"):
        base = V[(V.curated == 1)] if pop == "curated" else V[(V.curated == 0)]
        lines += [f"## {pop} verbs", "",
                  "| class | n | zero c_be | r(Zipf) | r(log c) | r(log c_be) | r(ρ_be) | CV MSE ~log c | ~log c + log c_be | "
                  "difference (CI) | ~Zipf | decision |", "|---|---:|---:|---|---|---|---|---|---|---|---|---|"]
        for role in ("T", "I"):
            for sm in (0.5, 0.1, 1.0):
                df = add_preds(base[base.role == role], sm)
                idx = df.index.to_numpy()
                y = Y[:, idx]
                wv = weights(np.random.default_rng(args.seed + (0 if role == "T" else 7)), df.bandkey.to_numpy(), args.n_boot)
                groups = pd.factorize(df.verb.map(cur_pair) if pop == "curated" else df.verb)[0]
                groups = np.random.default_rng(args.seed).permutation(groups.max() + 1)[groups]  # shuffled fold labels
                res = {k: np.empty(len(y)) for k in ("c", "cbe", "z", "cap", "wb")}
                rs = {k: np.empty(len(y)) for k in ("Zipf", "logc", "logcbe", "rho")}
                for b in range(len(y)):
                    w = wv[b]
                    res["c"][b] = cv_mse(design(df, ["logc"]), y[b], w, groups, args.folds)
                    res["cbe"][b] = cv_mse(design(df, ["logc", "logcbe"]), y[b], w, groups, args.folds)
                    res["z"][b] = cv_mse(design(df, ["zipf"]), y[b], w, groups, args.folds)
                    if sm == 0.5:
                        res["cap"][b] = (cv_mse(design(df, ["logc", "logcbe_cap"]), y[b], w, groups, args.folds)
                                         if pop == "curated" else np.nan)
                        res["wb"][b] = cv_mse(design(df, ["logc", "logcbe_wb"]), y[b], w, groups, args.folds)
                        for k, col in (("Zipf", "zipf"), ("logc", "logc"), ("logcbe", "logcbe"), ("rho", "rho")):
                            x = df[col].to_numpy(float)
                            rs[k][b] = np.corrcoef(np.repeat(x, w.astype(int)), np.repeat(y[b], w.astype(int)))[0, 1]
                diff = res["c"] - res["cbe"]
                rel = diff / res["c"]
                lo90, hi90 = ci(rel, 5, 95)
                d = ("slot frequency adds" if ci(diff)[0] > 0 else
                     "overall frequency suffices" if -0.02 < lo90 and hi90 < 0.02 else "unresolved")
                key = f"{pop}|{role}|s={sm}"
                dec[key] = d
                rows.append({"pop": pop, "role": role, "smooth": sm, "cv_logc": res["c"][0], "cv_logc_cbe": res["cbe"][0],
                             "cv_zipf": res["z"][0], "diff": diff[0], "diff_lo": ci(diff)[0], "diff_hi": ci(diff)[1],
                             "decision": d})
                if sm == 0.5:
                    zero = float((df.c_be == 0).mean())
                    lines.append(f"| {role} | {len(df)} | {zero:.2f} | {fmt(rs['Zipf'], 2)} | {fmt(rs['logc'], 2)} | "
                                 f"{fmt(rs['logcbe'], 2)} | {fmt(rs['rho'], 2)} | {res['c'][0]:.3f} | {res['cbe'][0]:.3f} | "
                                 f"{fmt(diff)} | {res['z'][0]:.3f} | **{d}** |")
                    extra = {"cap": res["cap"], "wb": res["wb"]}
                    for nm, v in extra.items():
                        if np.isnan(v).all():
                            dec[f"{pop}|{role}|{nm}"] = "not computed"
                            continue
                        dd = res["c"] - v
                        dec[f"{pop}|{role}|{nm}"] = ("slot frequency adds" if ci(dd)[0] > 0 else "no added value")
                        rows.append({"pop": pop, "role": role, "smooth": sm, "variant": nm, "diff": dd[0],
                                     "diff_lo": ci(dd)[0], "diff_hi": ci(dd)[1]})
            lines.append(f"| {role} | sensitivity | smoothing 0.1 / 1: {dec[f'{pop}|{role}|s=0.1']} / {dec[f'{pop}|{role}|s=1.0']}; "
                         f"capitalized auxiliaries: {dec[f'{pop}|{role}|cap']}; was + been only: {dec[f'{pop}|{role}|wb']} |"
                         + " |" * 10)
        lines.append("")
        # zero shares by band
        zb = base.assign(z=base.c_be == 0).groupby(["role", "bandkey"]).z.mean().unstack()
        lines += ["Share of verbs with c_be = 0, by class and band: " + "; ".join(
            f"{r} " + ", ".join(f"{b} {zb.loc[r, b]:.2f}" for b in zb.columns if not np.isnan(zb.loc[r, b])) for r in zb.index), ""]
    # pair level: L ~ T and I predictors
    P = pairs.copy()
    Vi = add_preds(V, 0.5).set_index("verb")
    keep = P.t_verb.isin(Vi.index) & P.i_verb.isin(Vi.index)
    P = P[keep].reset_index(drop=True)
    for r in ("t", "i"):
        for col in ("logc", "logcbe", "zipf"):
            P[f"{col}_{r}"] = Vi.loc[P[f"{r}_verb"], col].to_numpy()
    ti = Vi.index.get_indexer(P.t_verb)
    ii = Vi.index.get_indexer(P.i_verb)
    vix = {v: j for j, v in enumerate(V.verb)}
    YL = Y[:, [vix[v] for v in P.t_verb]] - Y[:, [vix[v] for v in P.i_verb]]
    wv = weights(np.random.default_rng(args.seed + 3), P.band.to_numpy(), args.n_boot)
    groups = np.random.default_rng(args.seed).permutation(len(P))
    mc, mcb, mz = (np.empty(len(YL)) for _ in range(3))
    for b in range(len(YL)):
        mc[b] = cv_mse(design(P, ["logc_t", "logc_i"]), YL[b], wv[b], groups, args.folds)
        mcb[b] = cv_mse(design(P, ["logc_t", "logc_i", "logcbe_t", "logcbe_i"]), YL[b], wv[b], groups, args.folds)
        mz[b] = cv_mse(design(P, ["zipf_t", "zipf_i"]), YL[b], wv[b], groups, args.folds)
    diff = mc - mcb
    lo90, hi90 = ci(diff / mc, 5, 95)
    dL = "slot frequency adds" if ci(diff)[0] > 0 else "overall frequency suffices" if -0.02 < lo90 and hi90 < 0.02 else "unresolved"
    dec["pairs|L"] = dL
    lines += ["## Pairs: L ~ T and I predictors (curated)", "",
              f"CV MSE: ~ log c_T + log c_I {mc[0]:.3f}; + log c_be,T + log c_be,I {mcb[0]:.3f}; difference {fmt(diff)}; "
              f"~ Zipf_T + Zipf_I {mz[0]:.3f}. **Decision: {dL}.**", ""]
    rows.append({"pop": "pairs", "role": "L", "smooth": 0.5, "cv_logc": mc[0], "cv_logc_cbe": mcb[0], "cv_zipf": mz[0],
                 "diff": diff[0], "diff_lo": ci(diff)[0], "diff_hi": ci(diff)[1], "decision": dL})
    # descriptive: Pile log c vs Zipf
    lines += ["Descriptive: CV MSE with wordfreq Zipf vs Pile log c is in the class tables (columns ~Zipf, ~log c).", ""]
    out = Path(args.res_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "a3_summary.csv", index=False)
    (out / "a3_decisions.json").write_text(json.dumps(dec, indent=1))
    V.to_csv(out / "a3_verbs.csv", index=False)
    Path(args.report).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="data/round5/counts/pile_counts.csv")
    ap.add_argument("--data-dir", type=Path, default=Path("data/round5/frame"))
    ap.add_argument("--frame", default="results/round5/frame/frame_scores.parquet")
    ap.add_argument("--res-dir", default="results/round5/counts")
    ap.add_argument("--report", default="reports/round5/a3_counts.md")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    main(ap.parse_args())
