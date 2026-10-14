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
Requires the finished hand audit (audit_labels.csv for every row of audit_contexts.csv).
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


def decide(base, alt):
    """base = CV error of ~log c (per draw), alt = with the slot predictor. 'adds': paired difference 95% CI above 0;
    'suffices': 90% CI of the relative difference within +-2%; both can hold (then 'adds, within 2%')."""
    diff = base - alt
    lo90, hi90 = ci(diff / base, 5, 95)
    adds, eq = ci(diff)[0] > 0, -0.02 < lo90 and hi90 < 0.02
    return ("slot frequency adds (within 2%)" if adds and eq else "slot frequency adds" if adds else
            "overall frequency suffices" if eq else "unresolved"), diff


def r2(X, y):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def wcorr(x, y, w):
    m = w > 0
    return np.corrcoef(np.repeat(x[m], w[m].astype(int)), np.repeat(y[m], w[m].astype(int)))[0, 1]


def wpartial(x, y, z, w):
    """weighted partial correlation of x and y given z (with intercept)."""
    Z = np.column_stack([np.ones(len(z)), z])
    rx = x - Z @ wls(Z, x, w)
    ry = y - Z @ wls(Z, y, w)
    return wcorr(rx, ry, w)


def main(args):
    C = pd.read_csv(args.counts)
    V = pd.read_csv(args.data_dir / "verbs.csv", keep_default_na=False)
    pairs = pd.read_csv(args.data_dir / "pairs.csv", keep_default_na=False)
    pre = pd.read_csv(args.data_dir / "prefixes.csv", keep_default_na=False)
    S = pd.read_parquet(args.frame)
    assert C["count"].notna().all(), f"{C['count'].isna().sum()} failed queries"
    # declared order: the hand audit of "was V" contexts is finished before any outcome analysis
    AU = pd.read_csv(args.audit_labels, keep_default_na=False)
    AC = pd.read_csv(args.audit_contexts, keep_default_na=False)
    assert len(AU) == len(AC) and set(AU.label) <= {"eventive passive", "adjectival", "other"}, "hand audit incomplete"
    # audit: every slot query's token ids end with the overall query's ids
    C = C[~C.slot.str.endswith("_or")]  # OR queries from an abandoned fetch configuration are not used
    ov = C[C.slot == "overall"].set_index("verb").token_ids
    simple = np.ones(len(C), bool)
    C["suffix_ok"] = True
    C.loc[simple, "suffix_ok"] = [str(t).split()[-len(str(ov[v]).split()):] == str(ov[v]).split()
                                  for v, t in zip(C.verb[simple], C.token_ids[simple])]
    bad = sorted(set(C[~C.suffix_ok].verb))
    W = C.pivot_table(index="verb", columns="slot", values="count", aggfunc="first")
    W["c"] = W["overall"]
    cur_v = set(V[V.curated == 1].verb)
    full_be = W[[f"be_{x}" for x in BE]].sum(1, min_count=len(BE))
    main_be = W[[f"be_{x}" for x in ("was", "were", "been")]].sum(1)
    W["c_be"] = np.where(W.index.isin(cur_v), full_be, main_be)  # curated: all six; pool: was + were + been
    W["c_perf"] = np.where(W.index.isin(cur_v), W[[f"perf_{x}" for x in PERF]].sum(1, min_count=len(PERF)), W["perf_has"])
    W["c_be_cap"] = np.where(W.index.isin(cur_v), W["c_be"] + W.get("capbe_was", np.nan) + W.get("capbe_were", np.nan), np.nan)
    W["c_be_wasbeen"] = np.where(W.index.isin(cur_v), W["be_was"] + W["be_been"], np.nan)
    V = V.join(W[["c", "c_be", "c_perf", "c_be_cap", "c_be_wasbeen"]], on="verb")
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
    au = AC.merge(AU[["verb", "k", "label"]], on=["verb", "k"])
    tab = au.groupby(["role", "band", "verb", "count"]).label.value_counts().unstack(fill_value=0)
    lines += ["**Hand audit of \"was V\" contexts** (up to 10 per verb, evenly spaced occurrences; labels by hand):", "",
              "| class | band | verb | \"was V\" count | " + " | ".join(tab.columns) + " |",
              "|---|---|---|---:|" + "---:|" * len(tab.columns)]
    lines += [f"| {r} | {b} | {v} | {c:,} | " + " | ".join(str(x) for x in row) + " |"
              for (r, b, v, c), row in zip(tab.index, tab.to_numpy())]
    tot = au.groupby("role").label.value_counts(normalize=True).unstack(fill_value=0)
    lines += ["", "Shares: " + "; ".join(f"{r}: " + ", ".join(f"{c} {tot.loc[r, c]:.2f}" for c in tot.columns)
                                         for r in tot.index), ""]
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
                  "| class | n | r(Zipf) | r(log c) | r(log c_be) | partial r(ρ_be \\| log c) | in-sample R²: log c / log c_be / both / Zipf | "
                  "CV MSE: ~log c | ~log c_be | ~log c + log c_be | difference (CI) | ~Zipf | decision |",
                  "|---|---:|---|---|---|---|---|---|---|---|---|---|---|"]
        for role in ("T", "I"):
            for sm in (0.5, 0.1, 1.0):
                df = add_preds(base[base.role == role], sm)
                idx = df.index.to_numpy()
                y = Y[:, idx]
                wv = weights(np.random.default_rng(args.seed + (0 if role == "T" else 7)), df.bandkey.to_numpy(), args.n_boot)
                groups = pd.factorize(df.verb.map(cur_pair) if pop == "curated" else df.verb)[0]
                groups = np.random.default_rng(args.seed).permutation(groups.max() + 1)[groups]  # shuffled fold labels
                models = {"c": ["logc"], "cbe": ["logcbe"], "both": ["logc", "logcbe"], "z": ["zipf"]}
                if sm == 0.5 and pop == "curated":
                    models |= {"cap": ["logc", "logcbe_cap"], "wb": ["logc", "logcbe_wb"]}
                res = {k: np.empty(len(y)) for k in models}
                rs = {k: np.empty(len(y)) for k in ("Zipf", "logc", "logcbe", "rho_partial")}
                for b in range(len(y)):
                    w = wv[b]
                    for k, cols in models.items():
                        res[k][b] = cv_mse(design(df, cols), y[b], w, groups, args.folds)
                    if sm == 0.5:
                        for k, col in (("Zipf", "zipf"), ("logc", "logc"), ("logcbe", "logcbe")):
                            rs[k][b] = wcorr(df[col].to_numpy(float), y[b], w)
                        rs["rho_partial"][b] = wpartial(df.rho.to_numpy(float), y[b], df.logc.to_numpy(float), w)
                d, diff = decide(res["c"], res["both"])
                key = f"{pop}|{role}|s={sm}"
                dec[key] = d
                rows.append({"pop": pop, "role": role, "smooth": sm, "cv_logc": res["c"][0], "cv_logcbe": res["cbe"][0],
                             "cv_logc_cbe": res["both"][0], "cv_zipf": res["z"][0], "diff": diff[0],
                             "diff_lo": ci(diff)[0], "diff_hi": ci(diff)[1], "decision": d})
                if sm == 0.5:
                    R2 = "/".join(f"{r2(design(df, cols), y[0]):.2f}" for cols in (["logc"], ["logcbe"], ["logc", "logcbe"], ["zipf"]))
                    lines.append(f"| {role} | {len(df)} | {fmt(rs['Zipf'], 2)} | {fmt(rs['logc'], 2)} | {fmt(rs['logcbe'], 2)} | "
                                 f"{fmt(rs['rho_partial'], 2)} | {R2} | {res['c'][0]:.3f} | {res['cbe'][0]:.3f} | "
                                 f"{res['both'][0]:.3f} | {fmt(diff)} | {res['z'][0]:.3f} | **{d}** |")
                    for nm in ("cap", "wb"):
                        if nm not in res:
                            dec[f"{pop}|{role}|{nm}"] = "not computed"
                            continue
                        dd_dec, dd = decide(res["c"], res[nm])
                        dec[f"{pop}|{role}|{nm}"] = dd_dec
                        rows.append({"pop": pop, "role": role, "smooth": sm, "variant": nm, "diff": dd[0],
                                     "diff_lo": ci(dd)[0], "diff_hi": ci(dd)[1], "decision": dd_dec})
            lines.append(f"| {role} | sensitivity | smoothing 0.1 / 1: {dec[f'{pop}|{role}|s=0.1']} / {dec[f'{pop}|{role}|s=1.0']}; "
                         f"capitalized auxiliaries: {dec[f'{pop}|{role}|cap']}; was + been only: {dec[f'{pop}|{role}|wb']} |"
                         + " |" * 11)
        lines.append("")
        # zero shares by class and band
        for col, nm in (("c", "c"), ("c_be", "c_be"), ("c_perf", "c_perf")):
            zb = base.assign(z=base[col] == 0).groupby(["role", "bandkey"]).z.mean().unstack()
            lines.append(f"Share with {nm} = 0: " + "; ".join(
                f"{r} " + ", ".join(f"{b} {zb.loc[r, b]:.2f}" for b in zb.columns if not np.isnan(zb.loc[r, b])) for r in zb.index))
        lines.append("")
    # pair level: L ~ T and I predictors (curated pairs)
    lines += ["## Pairs: L ~ T and I predictors (curated)", "",
              "| smoothing / variant | CV MSE ~log c_T + log c_I | ~log c_be,T + log c_be,I | ~both | difference (CI) | "
              "~Zipf_T + Zipf_I | decision |", "|---|---|---|---|---|---|---|"]
    vix = {v: j for j, v in enumerate(V.verb)}
    for sm, var in ((0.5, "logcbe"), (0.1, "logcbe"), (1.0, "logcbe"), (0.5, "logcbe_cap"), (0.5, "logcbe_wb")):
        P = pairs.copy()
        Vi = add_preds(V, sm).set_index("verb")
        P = P[P.t_verb.isin(Vi.index) & P.i_verb.isin(Vi.index)].reset_index(drop=True)
        for r in ("t", "i"):
            for col in ("logc", var, "zipf"):
                P[f"{col}_{r}"] = Vi.loc[P[f"{r}_verb"], col].to_numpy()
        YL = Y[:, [vix[v] for v in P.t_verb]] - Y[:, [vix[v] for v in P.i_verb]]
        wv = weights(np.random.default_rng(args.seed + 3), P.band.to_numpy(), args.n_boot)
        groups = np.random.default_rng(args.seed).permutation(len(P))
        M = {k: np.empty(len(YL)) for k in ("c", "cbe", "both", "z")}
        cols = {"c": ["logc_t", "logc_i"], "cbe": [f"{var}_t", f"{var}_i"],
                "both": ["logc_t", "logc_i", f"{var}_t", f"{var}_i"], "z": ["zipf_t", "zipf_i"]}
        for b in range(len(YL)):
            for k in M:
                M[k][b] = cv_mse(design(P, cols[k]), YL[b], wv[b], groups, args.folds)
        dL, diff = decide(M["c"], M["both"])
        tag = f"s={sm}" if var == "logcbe" else {"logcbe_cap": "capitalized", "logcbe_wb": "was + been"}[var]
        dec[f"pairs|L|{tag}"] = dL
        lines.append(f"| {tag} | {M['c'][0]:.3f} | {M['cbe'][0]:.3f} | {M['both'][0]:.3f} | {fmt(diff)} | {M['z'][0]:.3f} | "
                     f"{'**' + dL + '**' if tag == 's=0.5' else dL} |")
        rows.append({"pop": "pairs", "role": "L", "smooth": sm, "variant": tag, "cv_logc": M["c"][0],
                     "cv_logcbe": M["cbe"][0], "cv_logc_cbe": M["both"][0], "cv_zipf": M["z"][0], "diff": diff[0],
                     "diff_lo": ci(diff)[0], "diff_hi": ci(diff)[1], "decision": dL})
    dec["pairs|L"] = dec["pairs|L|s=0.5"]
    lines.append("")
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
    ap.add_argument("--audit-contexts", default="data/round5/counts/audit_contexts.csv")
    ap.add_argument("--audit-labels", default="data/round5/counts/audit_labels.csv")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    main(ap.parse_args())
