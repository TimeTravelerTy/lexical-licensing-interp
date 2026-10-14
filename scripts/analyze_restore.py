#!/usr/bin/env python3
"""Round 5, A4 analysis: output-side restoration (reports/round5/plan.md, A4). No model.

Inputs: results/round5/restore/{restore_curated, restore_freqblimp, restore_freqblimp_items, restore_lm}.parquet,
restore_zfit.csv, restore_meta.json; data/passive_band_cross/pairs.jsonl (curated metadata); data/round5/restore/
manifest.csv. Bootstrap 2,000 draws, seed 17: curated, pairs within band x contexts within context band; released
items within band x paradigm; LM loss, documents (Pile-10k) and verbs (Pile validation).
Outputs: reports/round5/a4_restore.md, results/round5/restore/a4_summary.csv, a4_decisions.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONDS = ("oracle", "oracle_T", "oracle_I", "labelfree")


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


def main(args):
    rdir = Path(args.res_dir)
    meta = json.loads((rdir / "restore_meta.json").read_text())
    man = pd.read_csv(args.manifest)
    C = pd.read_parquet(rdir / "restore_curated.parquet")
    bca = pd.read_json(args.pairs, lines=True)
    bca = bca[bca.context_set == "curated"].set_index("pair_id")
    M = C.pivot_table(index=["pair_id", "cond"], columns="side", values="whole").reset_index()
    M["margin"] = M.good - M.bad
    W = M.pivot(index="pair_id", columns="cond", values="margin")
    info = bca.loc[W.index, ["verb_pair", "verb_band", "context_id", "context_band", "paradigm"]]
    aff_pairs = set(man[man.verb_pair != ""].verb_pair)
    info["affected"] = info.verb_pair.isin(aff_pairs)
    rng = np.random.default_rng(args.seed)
    lines = ["# Round 5, A4: output-side restoration (Tail single-token participles)", "",
             "Spec: `plan.md`, A4. Run: `run_restore.py`; analysis: `analyze_restore.py`. Edits move each rare "
             "single-token participle's unembedding row along Δh̄ so its affinity becomes a*. 95% CIs: pairs within band × "
             "contexts (curated), items (released), documents / verbs (LM loss).", "",
             f"Checks: {json.dumps(meta['checks'])}.", "",
             "Label-free calibration (Head single-token verbs): " + json.dumps(meta["zfit"]), ""]
    rows, dec = [], {}
    for par in ("passive_1", "passive_2"):
        sel = (info.paradigm == par).to_numpy()
        I_ = info[sel]
        Wp = W[sel]
        pairs, ctxs = I_.verb_pair.to_numpy(), I_.context_id.to_numpy()
        up, uc = np.unique(pairs), np.unique(ctxs)
        pb = I_.groupby("verb_pair").verb_band.first().loc[up].to_numpy()
        cb = I_.groupby("context_id").context_band.first().loc[uc].to_numpy()
        wv, wc = weights(rng, pb, args.n_boot), weights(rng, cb, args.n_boot)
        pi, cix = pd.Index(up).get_indexer(pairs), pd.Index(uc).get_indexer(ctxs)

        def mean(v, mask):
            w = wv[:, pi[mask]] * wc[:, cix[mask]]
            return (w * v[mask][None]).sum(1) / w.sum(1)

        aff = I_.affected.to_numpy()
        head = (I_.verb_band == "head").to_numpy()
        u = Wp["unedited"].to_numpy()
        lines += [f"## Curated `{par}`", "",
                  f"Affected pairs: {len(np.unique(pairs[aff]))} ({sorted(np.unique(pairs[aff]))}).", "",
                  "| Condition | affected: Δ margin | affected: Δ accuracy (pp) | Head pairs: Δ margin | share of deficit closed |",
                  "|---|---|---|---|---|"]
        deficit = mean(u, head) - mean(u, aff)
        for cnd in CONDS:
            d = Wp[cnd].to_numpy() - u
            da = mean(d, aff)
            acc = mean(((Wp[cnd].to_numpy() > 0).astype(float) - (u > 0)) * 100, aff)
            dh = mean(d, head)
            share = da / deficit
            lines.append(f"| {cnd} | {fmt(da)} | {fmt(acc, 1)} | {fmt(dh)} | {fmt(share, 2)} |")
            for nm, v in (("d_margin_affected", da), ("d_acc_affected", acc), ("d_margin_head", dh), ("share_closed", share)):
                rows.append({"set": f"curated_{par}", "cond": cnd, "stat": nm, "est": v[0], "lo": ci(v)[0], "hi": ci(v)[1]})
            if par == "passive_1":
                dec[f"{cnd}_improves"] = bool(ci(da)[0] > 0 and da[0] >= 0.25)
                dec[f"{cnd}_head_within_0.05"] = bool(-0.05 < ci(dh)[0] and ci(dh)[1] < 0.05)
                dec[f"{cnd}_d_affected"] = da
        lines += ["", f"Unedited margins: Head pairs {fmt(mean(u, head))}, affected pairs {fmt(mean(u, aff))}; deficit "
                      f"{fmt(deficit)}.", "",
                  "Accuracy and margin by band (all pairs):", "", "| Condition | Head | Tail | XTail |", "|---|---|---|---|"]
        for cnd in ("unedited",) + CONDS:
            v = Wp[cnd].to_numpy()
            cells = []
            for b in ("head", "tail", "xtail"):
                m = (I_.verb_band == b).to_numpy()
                cells.append(f"{100 * mean((v > 0).astype(float), m)[0]:.1f}% / {mean(v, m)[0]:+.2f}")
            lines.append(f"| {cnd} | " + " | ".join(cells) + " |")
        lines.append("")
    # label-free vs half the oracle (affected curated passive_1)
    if dec.get("oracle_improves"):
        diff = dec["labelfree_d_affected"] - 0.5 * dec["oracle_d_affected"]
        dec["labelfree_half_oracle"] = bool(ci(diff)[0] > 0)
        lines += [f"Label-free − 0.5 × oracle (affected, passive_1): {fmt(diff)} → **{'achieved' if dec['labelfree_half_oracle'] else 'not achieved'}**.", ""]
    else:
        dec["labelfree_half_oracle"] = "not tested (oracle does not improve)"
    # released FreqBLiMP
    F = pd.read_parquet(rdir / "restore_freqblimp.parquet")
    items = pd.read_parquet(rdir / "restore_freqblimp_items.parquet")
    Fw = F.assign(margin=F.good_lp - F.bad_lp).pivot(index="row", columns="cond", values="margin")
    affm = F[F.cond == "unedited"].set_index("row").affected.loc[Fw.index].to_numpy()
    it = items.set_index("row").loc[Fw.index]
    strata = (it.band + "|" + it.uid).to_numpy()
    wi = weights(rng, strata, args.n_boot)
    lines += ["## Released FreqBLiMP", "",
              f"Items containing an edited token: {int(affm.sum())} ({', '.join(f'{k}: {v}' for k, v in it[affm].uid.value_counts().items())}).", "",
              "| Condition | passive_1 affected: Δ acc (pp) | passive_1 Tail: acc before → after | unaffected items pooled: Δ acc (pp) | max |Δ acc| over cells (pp) |",
              "|---|---|---|---|---|"]
    u = Fw["unedited"].to_numpy()
    for cnd in CONDS:
        dacc = ((Fw[cnd].to_numpy() > 0).astype(float) - (u > 0)) * 100
        m1 = (it.uid == "passive_1").to_numpy() & affm
        a1 = (wi[:, m1] * dacc[m1]).sum(1) / wi[:, m1].sum(1) if m1.any() else np.full(len(wi), np.nan)
        un = ~affm
        pool = (wi[:, un] * dacc[un]).sum(1) / wi[:, un].sum(1)
        cell = pd.Series(dacc[un]).groupby(strata[un]).mean().abs().max()
        tail = ((it.uid == "passive_1") & (it.band == "tail")).to_numpy()
        lines.append(f"| {cnd} | {fmt(a1, 1) if m1.any() else '—'} | {100 * (u[tail] > 0).mean():.1f}% → "
                     f"{100 * (Fw[cnd].to_numpy()[tail] > 0).mean():.1f}% | {fmt(pool, 2)} | {cell:.2f} |")
        dec[f"{cnd}_unaffected_within_0.5pp"] = bool(-0.5 < ci(pool)[0] and ci(pool)[1] < 0.5)
        rows.append({"set": "freqblimp_unaffected", "cond": cnd, "stat": "d_acc_pp", "est": pool[0], "lo": ci(pool)[0],
                     "hi": ci(pool)[1]})
    # LM loss
    Lm = pd.read_parquet(rdir / "restore_lm.parquet")
    lines += ["", "## LM loss", "", "| Condition | Pile-10k: Δ NLL / token | Pile-val: Δ NLL / token (token-weighted) | "
              "Pile-val: per-verb macro Δ | edited tokens: Δ NLL (be / perfect / other) |", "|---|---|---|---|---|"]
    base = Lm[Lm.cond == "unedited"].reset_index(drop=True)
    for cnd in CONDS:
        x = Lm[Lm.cond == cnd].reset_index(drop=True)
        assert (x.doc_ix.to_numpy() == base.doc_ix.to_numpy()).all()
        d = x.nll_sum - base.nll_sum
        p10 = (base.src == "pile10k").to_numpy()
        wd = weights(rng, np.zeros(int(p10.sum())), args.n_boot)
        r10 = (wd * d[p10].to_numpy()[None]).sum(1) / (wd * base.n_tok[p10].to_numpy()[None]).sum(1)
        pv = ~p10
        rpv = d[pv].sum() / base.n_tok[pv].sum()
        mac = (d[pv] / base.n_tok[pv]).groupby(base.verb[pv]).mean().mean()
        ed = {"be": [], "perf": [], "other": []}
        for (a, b_), (ca, cb_) in zip(zip(x.edited_nll[pv], base.edited_nll[pv]), zip(x.edited_ctx[pv], base.edited_ctx[pv])):
            for va, vb, c_ in zip(a, b_, ca):
                ed[c_].append(va - vb)
        eds = " / ".join(f"{np.mean(v):+.3f} (n {len(v)})" if v else "—" for v in ed.values())
        lines.append(f"| {cnd} | {fmt(r10, 4)} | {rpv:+.4f} | {mac:+.4f} | {eds} |")
        dec[f"{cnd}_pile10k_below_0.002"] = bool(ci(r10)[1] < 0.002)
    for cnd in CONDS:
        dec[f"{cnd}_specific"] = bool(dec.get(f"{cnd}_head_within_0.05") and dec.get(f"{cnd}_unaffected_within_0.5pp")
                                      and dec.get(f"{cnd}_pile10k_below_0.002"))
    for k in [k for k in dec if k.endswith("_d_affected")]:
        dec.pop(k)
    lines += ["", "## Declared decisions", "",
              f"- Oracle improves rare passives (affected curated passive_1, Δ margin CI above 0 and ≥ 0.25): **{dec['oracle_improves']}**.",
              f"- Specific (Head ±0.05; unaffected ±0.5 pp; Pile-10k < +0.002): oracle **{dec['oracle_specific']}**, label-free "
              f"{dec['labelfree_specific']}.",
              f"- Label-free achieves half the oracle: **{dec['labelfree_half_oracle']}**.",
              f"- Side arms (reported): T only improves {dec['oracle_T_improves']}; I only improves {dec['oracle_I_improves']}.", ""]
    Path(args.report).write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(rdir / "a4_summary.csv", index=False)
    (rdir / "a4_decisions.json").write_text(json.dumps(dec, indent=1, default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--res-dir", default="results/round5/restore")
    ap.add_argument("--manifest", default="data/round5/restore/manifest.csv")
    ap.add_argument("--pairs", default="data/passive_band_cross/pairs.jsonl")
    ap.add_argument("--report", default="reports/round5/a4_restore.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    main(ap.parse_args())
