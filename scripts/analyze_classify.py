#!/usr/bin/env python3
"""Round 6, C4 analysis: classify-from-z restore (reports/round6/plan.md, C4). No model.

Inputs: data/round6/classify/{manifest.csv, extra_z.csv, meta.json}; the C4 run (results/round6/classify/: restore_curated,
restore_freqblimp, restore_freqblimp_items parquet, restore_zfit.csv, restore_zfit_extra.csv, restore_meta.json);
data/passive_band_cross/pairs.jsonl; D11 verbs.
Classification: theta = the balanced-accuracy-maximizing threshold over A4's Head single-token verbs (T if z >= theta);
balanced accuracy on A4's 32 rare singletons and on D11's Tail singletons; 95% CIs with Head and rare verbs resampled
within class and theta refit in every draw; Head in-sample and leave-one-out balanced accuracy; sensitivity theta from
A4's Head plus D11's extra Head singletons.
Restoration (conditions oracle and classify vs unedited): affected curated passive_1 / passive_2 pairs (pairs within band
x contexts within context band), released passive items, unaffected FreqBLiMP items (specificity).
Outputs: reports/round6/c4_classify.md, results/round6/classify/c4_decisions.json, c4_summary.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_restore import ci, fmt, weights
from build_classify_manifest import youden_threshold

CONDS = ("oracle", "classify")


def bal_acc(z, is_t, theta):
    return 0.5 * ((z[is_t] >= theta).mean() + (z[~is_t] < theta).mean())


def boot_ba(rng, zh, th, zr, tr, n_boot):
    """Balanced accuracy and per-class accuracies on (zr, tr) with theta refit on resampled Head verbs (verbs resampled
    within class); row 0 = point estimate. Returns [draw, 3]: balanced, T, I."""
    out = np.empty((n_boot + 1, 3))

    def accs(zr_, tr_, th_):
        a_t, a_i = (zr_[tr_] >= th_).mean(), (zr_[~tr_] < th_).mean()
        return 0.5 * (a_t + a_i), a_t, a_i

    out[0] = accs(zr, tr, youden_threshold(zh, th)[0])
    for b in range(1, n_boot + 1):
        ih = np.concatenate([rng.choice(np.flatnonzero(th == c), (th == c).sum()) for c in (True, False)])
        ir = np.concatenate([rng.choice(np.flatnonzero(tr == c), (tr == c).sum()) for c in (True, False)])
        out[b] = accs(zr[ir], tr[ir], youden_threshold(zh[ih], th[ih])[0])
    return out


def main(args):
    rng = np.random.default_rng(args.seed)
    rdir = Path(args.res_dir)
    meta = json.loads((rdir / "restore_meta.json").read_text())
    man = pd.read_csv(Path(args.data_dir) / "manifest.csv", keep_default_na=False)
    mmeta = json.loads((Path(args.data_dir) / "meta.json").read_text())
    Z = pd.read_csv(rdir / "restore_zfit.csv", keep_default_na=False)
    X = pd.read_csv(rdir / "restore_zfit_extra.csv", keep_default_na=False)
    head, rare = Z[Z.group == "head"], Z[Z.group == "rare"]
    zh, th = head.z.to_numpy(float), (head.role == "T").to_numpy()
    theta, ba_head = youden_threshold(zh, th)
    assert abs(theta - mmeta["theta"]) < 1e-4, "theta differs from the frozen manifest"
    zr, tr = rare.z.to_numpy(float), (rare.role == "T").to_numpy()
    live = pd.Series(np.where(zr >= theta, "T", "I"), index=rare.verb)
    assert (live.loc[man.verb].to_numpy() == man.assigned.to_numpy()).all(), "assignments from the live z differ from the manifest"
    lines = ["# Round 6, C4: classify-from-z restore", "",
             "Spec: `plan.md`, C4. Run: `run_restore.py --classify-manifest` (A4's setup; conditions unedited, oracle, "
             "classify; LM loss included); analysis: `analyze_classify.py`. The classify condition moves each rare single-token "
             "participle's row to the Head mean affinity of the class read from its site-8 z (T if z ≥ θ; θ maximizes "
             "balanced accuracy over A4's Head single-token verbs).", "",
             f"Checks: {json.dumps(meta['checks'])}.", "", "## Classification by z", ""]
    loo = np.array([((zh[i] >= youden_threshold(np.delete(zh, i), np.delete(th, i))[0]) == th[i]) for i in range(len(zh))])
    loo_ba = 0.5 * (loo[th].mean() + loo[~th].mean())
    br = boot_ba(rng, zh, th, zr, tr, args.n_boot)
    ba_r, at_r, ai_r = br[:, 0], br[:, 1], br[:, 2]
    acc_t, acc_i = at_r[0], ai_r[0]
    dec = {"theta": theta, "ba_rare": [ba_r[0], *ci(ba_r)], "above_chance": bool(ci(ba_r)[0] > 0.5),
           "acc_T_rare": [at_r[0], *ci(at_r)], "acc_I_rare": [ai_r[0], *ci(ai_r)], "ba_head_in_sample": ba_head,
           "ba_head_loo": loo_ba}
    V = pd.read_csv(args.d11_verbs, keep_default_na=False)
    d11t = V[(V.n_tok == 1) & (V.band == "tail")][["verb", "role"]]
    zall = pd.concat([Z[["verb", "z"]], X[["verb", "z"]]]).drop_duplicates("verb").set_index("verb").z
    zd, td = zall.loc[d11t.verb].to_numpy(float), (d11t.role == "T").to_numpy()
    bd = boot_ba(rng, zh, th, zd, td, args.n_boot)
    ba_d = bd[:, 0]
    dec["ba_d11_tail"] = [ba_d[0], *ci(ba_d)]
    dec["acc_T_d11_tail"], dec["acc_I_d11_tail"] = [bd[0, 1], *ci(bd[:, 1])], [bd[0, 2], *ci(bd[:, 2])]
    xh = X[X.band == "head"]
    zh2, th2 = np.concatenate([zh, xh.z.to_numpy(float)]), np.concatenate([th, (xh.role == "T").to_numpy()])
    theta2 = youden_threshold(zh2, th2)[0]
    dec["theta_sensitivity"] = theta2
    dec["ba_rare_theta2"] = bal_acc(zr, tr, theta2)
    dec["ba_d11_tail_theta2"] = bal_acc(zd, td, theta2)
    lines += [f"- θ = {theta:.3f} (Head: {int(th.sum())} T, {int((~th).sum())} I); Head balanced accuracy in sample "
              f"{ba_head:.3f}, leave-one-out {loo_ba:.3f}.",
              f"- **A4's rare singletons** ({int(tr.sum())} T, {int((~tr).sum())} I): accuracy T {fmt(at_r, 2)}, I {fmt(ai_r, 2)}; "
              f"balanced accuracy {fmt(ba_r)} → **{'above chance' if dec['above_chance'] else 'not above chance'}** (θ refit on "
              "resampled Head verbs in every draw).",
              f"- D11 Tail singletons ({int(td.sum())} T, {int((~td).sum())} I; secondary): accuracy T {fmt(bd[:, 1], 2)}, I "
              f"{fmt(bd[:, 2], 2)}; balanced accuracy {fmt(ba_d)}.",
              f"- Sensitivity θ' = {theta2:.3f} (Head of A4 + {len(xh)} D11 Head singletons): balanced accuracy rare "
              f"{dec['ba_rare_theta2']:.3f}, D11 Tail {dec['ba_d11_tail_theta2']:.3f}.", "",
              "z by verb (rare singletons; assigned class):", "",
              "| verb | class | z | assigned |", "|---|---|---|---|"]
    for r in man.sort_values(["role", "z"]).itertuples(index=False):
        lines.append(f"| {r.verb} | {r.role} | {r.z:+.2f} | {r.assigned}{'' if r.assigned == r.role else ' ✗'} |")
    # ---- restoration
    C = pd.read_parquet(rdir / "restore_curated.parquet")
    bca = pd.read_json(args.pairs, lines=True)
    bca = bca[bca.context_set == "curated"].set_index("pair_id")
    M = C.pivot_table(index=["pair_id", "cond"], columns="side", values="whole").reset_index()
    M["margin"] = M.good - M.bad
    W = M.pivot(index="pair_id", columns="cond", values="margin")
    info = bca.loc[W.index, ["verb_pair", "verb_band", "context_id", "context_band", "paradigm"]]
    info["affected"] = info.verb_pair.isin(set(man[man.verb_pair != ""].verb_pair))
    rows = []
    lines += ["", "## Restoration (curated passives)", "",
              "CIs condition on the frozen θ, assignments and Head targets (the classification CIs above do not).", ""]
    for par in ("passive_1", "passive_2"):
        sel = (info.paradigm == par).to_numpy()
        I_, Wp = info[sel], W[sel]
        pairs, ctxs = I_.verb_pair.to_numpy(), I_.context_id.to_numpy()
        up, uc = np.unique(pairs), np.unique(ctxs)
        wv = weights(rng, I_.groupby("verb_pair").verb_band.first().loc[up].to_numpy(), args.n_boot)
        wc = weights(rng, I_.groupby("context_id").context_band.first().loc[uc].to_numpy(), args.n_boot)
        pi, cix = pd.Index(up).get_indexer(pairs), pd.Index(uc).get_indexer(ctxs)

        def mean(v, mask):
            w = wv[:, pi[mask]] * wc[:, cix[mask]]
            return (w * v[mask][None]).sum(1) / w.sum(1)

        aff, hd = I_.affected.to_numpy(), (I_.verb_band == "head").to_numpy()
        u = Wp["unedited"].to_numpy()
        deficit = mean(u, hd) - mean(u, aff)
        lines += [f"### `{par}` (affected pairs: {len(np.unique(pairs[aff]))})", "",
                  "| Condition | affected: Δ margin | affected: Δ accuracy (pp) | Head pairs: Δ margin | share of deficit closed |",
                  "|---|---|---|---|---|"]
        d_aff = {}
        for cnd in CONDS:
            d = Wp[cnd].to_numpy() - u
            da, dh = mean(d, aff), mean(d, hd)
            dacc = mean(((Wp[cnd].to_numpy() > 0).astype(float) - (u > 0)) * 100, aff)
            share = (da - dh) / deficit
            d_aff[cnd] = da
            lines.append(f"| {cnd} | {fmt(da)} | {fmt(dacc, 1)} | {fmt(dh)} | {fmt(share, 2)} |")
            for nm, v in (("d_margin_affected", da), ("d_acc_affected", dacc), ("d_margin_head", dh), ("share", share)):
                rows.append({"set": par, "cond": cnd, "stat": nm, "est": v[0], "lo": ci(v)[0], "hi": ci(v)[1]})
            if par == "passive_1":
                dec[f"{cnd}_improves"] = bool(ci(da)[0] > 0 and da[0] >= 0.25)
                dec[f"{cnd}_head_within_0.05"] = bool(-0.05 < ci(dh)[0] and ci(dh)[1] < 0.05)
        diff_half = d_aff["classify"] - 0.5 * d_aff["oracle"]
        diff = d_aff["classify"] - d_aff["oracle"]
        lines += ["", f"classify − oracle: {fmt(diff)} (90% CI [{ci(diff, 5, 95)[0]:+.3f}, {ci(diff, 5, 95)[1]:+.3f}]); "
                      f"classify − 0.5 × oracle: {fmt(diff_half)}.", ""]
        if par == "passive_1":
            dec["half_oracle"] = bool(ci(diff_half)[0] > 0) if dec["oracle_improves"] else "not tested (oracle does not improve)"
            dec["matches_oracle"] = bool(-0.1 < ci(diff, 5, 95)[0] and ci(diff, 5, 95)[1] < 0.1)
            dec["classify_minus_oracle"] = [diff[0], *ci(diff)]
    # released FreqBLiMP
    F = pd.read_parquet(rdir / "restore_freqblimp.parquet")
    items = pd.read_parquet(rdir / "restore_freqblimp_items.parquet")
    Fw = F.assign(margin=F.good_lp - F.bad_lp).pivot(index="row", columns="cond", values="margin")
    affm = F[F.cond == "unedited"].set_index("row").affected.loc[Fw.index].to_numpy()
    it = items.set_index("row").loc[Fw.index]
    wi = weights(rng, (it.band + "|" + it.uid).to_numpy(), args.n_boot)
    wmean = lambda v, m: (wi[:, m] * v[m][None]).sum(1) / wi[:, m].sum(1)
    u = Fw["unedited"].to_numpy()
    lines += ["## Released FreqBLiMP", "", "| paradigm | band | items (affected) | unedited acc (%) | " +
              " | ".join(f"Δ acc {c} (pp)" for c in CONDS) + " | affected items: acc unedited → " + " / ".join(CONDS) + " |",
              "|---|---|---|---|" + "---|" * len(CONDS) + "---|"]
    for par in ("passive_1", "passive_2"):
        for b in ("head", "tail", "xtail"):
            m = ((it.uid == par) & (it.band == b)).to_numpy()
            ma = m & affm
            cells = [fmt(wmean(((Fw[c].to_numpy() > 0).astype(float) - (u > 0)) * 100, m), 1) for c in CONDS]
            affc = "—" if not ma.any() else f"{100 * (u[ma] > 0).mean():.0f}% → " + " / ".join(
                f"{100 * (Fw[c].to_numpy()[ma] > 0).mean():.0f}%" for c in CONDS)
            lines.append(f"| {par} | {b} | {int(m.sum())} ({int(ma.sum())}) | {fmt(wmean((u > 0).astype(float) * 100, m), 1)} | "
                         + " | ".join(cells) + f" | {affc} |")
    for cnd in CONDS:
        pool = wmean(((Fw[cnd].to_numpy() > 0).astype(float) - (u > 0)) * 100, ~affm)
        dec[f"{cnd}_unaffected_within_0.5pp"] = bool(-0.5 < ci(pool)[0] and ci(pool)[1] < 0.5)
        dec[f"{cnd}_specific"] = bool(dec[f"{cnd}_head_within_0.05"] and dec[f"{cnd}_unaffected_within_0.5pp"])
        lines.append(f"\nUnaffected items pooled, Δ accuracy ({cnd}): {fmt(pool, 2)} pp.")
    lmf = rdir / "restore_lm.parquet"
    assert lmf.exists(), "LM-loss results are required for the specificity decision"
    if lmf.exists():
        Lm = pd.read_parquet(lmf)
        base = Lm[Lm.cond == "unedited"].reset_index(drop=True)
        p10 = (base.src == "pile10k").to_numpy()
        wd = weights(rng, np.zeros(int(p10.sum())), args.n_boot)
        tok2verb = dict(zip(man.first_id.astype(int), man.verb))
        lines += ["", "## LM loss", "", "| Condition | Pile-10k: Δ NLL / token | edited tokens (Pile-val): Δ NLL, per-verb macro |",
                  "|---|---|---|"]
        for cnd in CONDS:
            x = Lm[Lm.cond == cnd].reset_index(drop=True)
            assert (x.doc_ix.to_numpy() == base.doc_ix.to_numpy()).all()
            d = (x.nll_sum - base.nll_sum).to_numpy()
            r10 = (wd * d[p10][None]).sum(1) / (wd * base.n_tok[p10].to_numpy()[None]).sum(1)
            recs = [(tok2verb.get(int(t_), str(t_)), va - vb) for a, b_, tk in zip(x.edited_nll[~p10], base.edited_nll[~p10],
                                                                                 x.edited_tok[~p10]) for va, vb, t_ in zip(a, b_, tk)]
            E = pd.DataFrame(recs, columns=["ev", "d"])
            pe = E.groupby("ev").d.mean()
            we = weights(rng, np.zeros(len(pe)), args.n_boot)
            mac = (we * pe.to_numpy()[None]).sum(1) / we.sum(1)
            dec[f"{cnd}_pile10k_below_0.002"] = bool(ci(r10)[1] < 0.002)
            dec[f"{cnd}_specific"] = bool(dec[f"{cnd}_specific"] and dec[f"{cnd}_pile10k_below_0.002"])
            lines.append(f"| {cnd} | {fmt(r10, 4)} | {fmt(mac, 3)} |")
    lines += ["", "## Declared decisions", "",
              f"- z classifies rare verbs above chance (balanced accuracy CI above 0.5): **{dec['above_chance']}**.",
              f"- Classify-restore improves rare passives (affected curated passive_1, CI above 0 and ≥ 0.25): **{dec['classify_improves']}** "
              f"(oracle: {dec['oracle_improves']}).",
              f"- Achieves half the oracle: **{dec['half_oracle']}**; matches the oracle (90% CI within ±0.1): **{dec['matches_oracle']}**.",
              f"- Specific (Head ±0.05 nats; unaffected ±0.5 pp; Pile-10k < +0.002 nats / token): classify {dec['classify_specific']}, "
              f"oracle {dec['oracle_specific']}.", ""]
    Path(args.report).write_text("\n".join(lines) + "\n")
    pd.DataFrame(rows).to_csv(rdir / "c4_summary.csv", index=False)
    (rdir / "c4_decisions.json").write_text(json.dumps(dec, indent=1, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res-dir", default="results/round6/classify")
    ap.add_argument("--data-dir", default="data/round6/classify")
    ap.add_argument("--d11-verbs", default="data/round6/frame_d11/verbs.csv")
    ap.add_argument("--pairs", default="data/passive_band_cross/pairs.jsonl")
    ap.add_argument("--report", default="reports/round6/c4_classify.md")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    main(ap.parse_args())
