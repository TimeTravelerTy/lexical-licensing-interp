#!/usr/bin/env python3
"""Analyse the passive test of the round-2 direction (spec: reports/passive_das_prep/passive_test_plan.md).

Modes
- `delta`: for the declared secondary sites, compute the held-out active
  effect and the TOST bound delta_s = 0.2 x Delta log P(O) (intransitive base
  <- transitive source, mean over held-out pairs; as for site 17) and write it
  into `frozen_config_site{s}.json`. Run before any passive evaluation.
- `passive`: step 1 (projection) and step 2 (transfer) at every site, from
  `run_passive_test.py` outputs; writes summary tables and the report.

Inference (step 2): three-way cluster bootstrap over base verb pairs
(resampled within band), contexts and donor verb pairs; per draw, cells
(pair x context x donor pair) are averaged within pair with context x donor
weights, then pairs are averaged with pair weights (equal weight per pair).
Step 1 uses the same machinery without the donor dimension, and subjects in
place of contexts for the active frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_das_round2_results import boot, per_pair

READ = ("M", "O", "I", "det", "pron", "refl", "by", "dot", "the", "him")
LABEL = {"M": "M", "O": "log P(O)", "I": "log P(I)", "det": "log P(det)", "pron": "log P(pron)",
         "refl": "log P(refl)", "by": 'log P(" by")', "dot": 'log P(".")', "the": 'log P(" the")',
         "him": 'log P(" him")'}
SUBJ = ["She", "He", "They", "We", "I", "Maria", "David"]


# ---------------------------------------------------------------- delta mode
def run_delta(args):
    for site in (8, 12):
        d = Path(args.root) / f"final_strict_site{site}"
        det = pd.read_csv(d / "heldout_swaps.csv.gz")
        pp, _ = per_pair(det, 1)
        intr = pp[pp.base_cls == 0]
        est, lo, hi, n = boot(intr.d_O)
        cfg_path = Path(args.root) / f"frozen_config_site{site}.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["tost"] = {"quantity": "passive contrast D on delta log P(O) (transitive minus intransitive donor), "
                                   "after patching",
                       "active_effect_delta_logP_O": round(float(est), 4),
                       "active_effect_ci95": [round(float(lo), 4), round(float(hi), 4)], "n_pairs": int(n),
                       "fraction": 0.2, "delta": round(0.2 * float(est), 4),
                       "rule": "equivalence ('no rise') if the 90% CI of D lies within +/- delta"}
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
        print(site, cfg["tost"])


# ---------------------------------------------------------------- bootstrap machinery
class Boot:
    """Shared resampling weights: contexts, donor pairs (per donor set), and pairs (per population)."""

    def __init__(self, n_ctx, n_draws, seed):
        self.rng = np.random.default_rng(seed)
        self.B = n_draws
        self.wc = self._multi(n_ctx)
        self.wq = {}

    def _multi(self, n):
        w = self.rng.multinomial(n, np.full(n, 1 / n), size=self.B).astype(np.float32)
        return np.vstack([np.ones((1, n), np.float32), w])  # row 0 = point estimate

    def donor_weights(self, key, n):
        if key not in self.wq:
            self.wq[key] = self._multi(n) if n > 1 else np.ones((self.B + 1, 1), np.float32)
        return self.wq[key]

    def pair_weights(self, bands):
        """bands: array of band labels for the population's pairs; multinomial within band."""
        w = np.zeros((self.B + 1, len(bands)), np.float32)
        w[0] = 1
        for b in np.unique(bands):
            ix = np.flatnonzero(bands == b)
            w[1:, ix] = self.rng.multinomial(len(ix), np.full(len(ix), 1 / len(ix)), size=self.B)
        return w


def pair_values(v, m, wc, wq, chunk=100):
    """v: [P, C, Q, R] cell means (NaN-free, 0 where absent), m: [P, C, Q] presence.

    Returns A [B+1, P, R]: per draw, per pair, the context x donor weighted mean."""
    P, C, Q, R = v.shape
    out = np.empty((wc.shape[0], P, R), np.float32)
    vf = v.reshape(P, C, Q * R).transpose(1, 0, 2).reshape(C, P * Q * R)
    mf = m.reshape(P, C, Q).transpose(1, 0, 2).reshape(C, P * Q)
    for i in range(0, wc.shape[0], chunk):
        c = wc[i:i + chunk]
        num = (c @ vf).reshape(len(c), P, Q, R)
        den = (c @ mf).reshape(len(c), P, Q)
        q = wq[i:i + chunk][:, None, :]
        with np.errstate(invalid="ignore", divide="ignore"):  # pairs with no cells in this condition -> NaN
            out[i:i + chunk] = np.einsum("bpqr,bq->bpr", num, wq[i:i + chunk]) / (den * q).sum(-1)[..., None]
    return out


def cells(df, pair_ix, ctx_ix, q_ix, n_q, value_cols):
    """Cell means [P, C, Q, R] and presence [P, C, Q] from rows with columns pair/ctx/q indices."""
    P, C = len(pair_ix), len(ctx_ix)
    p = df.pair_id.map(pair_ix).to_numpy()
    c = df.context_id.map(ctx_ix).to_numpy()
    q = df.dq.astype(str).map(q_ix).to_numpy(int) if n_q > 1 else np.zeros(len(df), int)
    s = np.zeros((P, C, n_q, len(value_cols)))
    n = np.zeros((P, C, n_q))
    np.add.at(s, (p, c, q), df[list(value_cols)].to_numpy())
    np.add.at(n, (p, c, q), 1)
    pres = n > 0
    v = np.where(pres[..., None], s / np.maximum(n, 1)[..., None], 0.0)
    return v.astype(np.float32), pres.astype(np.float32)


def summarize(A, wp, pix):
    """A [B+1, P, R] -> population statistic per draw [B+1, R] (pair weights wp over pair indices pix)."""
    a = A[:, pix, :]
    ok = ~np.isnan(a)
    w = wp[..., None] * ok
    return np.nansum(a * wp[..., None], 1) / w.sum(1)


def ci(x):
    e = x[0]
    d = x[1:]
    return {"est": float(e), "lo95": float(np.percentile(d, 2.5)), "hi95": float(np.percentile(d, 97.5)),
            "lo90": float(np.percentile(d, 5)), "hi90": float(np.percentile(d, 95))}


def classify(s, delta):
    if s["lo95"] > 0 and s["est"] >= delta:
        return "RISE"
    if s["hi95"] < 0 and s["est"] <= -delta:
        return "FALL"
    if s["lo90"] > -delta and s["hi90"] < delta:
        return "NO RISE"
    return "unresolved"


def outcome(o, by):
    if o == "RISE" and by != "RISE":
        return "surface (object next)"
    if o == "NO RISE" and by == "RISE":
        return "abstract (takes an object)"
    if o == "RISE" and by == "RISE":
        return "mixed"
    if o == "NO RISE" and by == "NO RISE":
        return "nothing moves (active-specific)"
    return "unresolved"


def auc(x, y):
    x, y = np.asarray(x), np.asarray(y)
    r = pd.Series(np.concatenate([x, y])).rank().to_numpy()
    return float((r[:len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))


# ---------------------------------------------------------------- populations
def populations(items):
    """name -> boolean mask over pairs (one row per pair)."""
    pr = items.drop_duplicates("pair_id").set_index("pair_id")
    plain = pr.bad_class == "plain"
    pops = {
        "primary (plain)": plain,
        "primary, original pairs only": plain & (pr.source == "original"),
        "primary, without bet/appear": plain & (pr.index != "head/bet/appear"),
        "primary, cross-fitted orig_head pairs": plain & pr.das_pair.notna(),
        "primary, non-training pairs": plain & pr.das_pair.isna(),
        "primary + contaminated_bad": plain | (pr.bad_class == "contaminated_bad"),
        "primary, by reliable": plain & (pr.by_split == "reliable"),
        "primary, by negative": plain & (pr.by_split == "negative_by"),
        "new Head eval pairs (prep_object)": (pr.source == "new") & (pr.band == "head"),
        "prep_object, pseudo_passive_ok": pr.prep_subtype == "pseudo_passive_ok",
        "prep_object, pseudo_passive_bad": pr.prep_subtype == "pseudo_passive_bad",
        "contaminated_bad": pr.bad_class == "contaminated_bad",
        "bad-side-high by": pr.bad_side_high.astype(bool),
        "original, by reliable": (pr.source == "original") & (pr.by_split == "reliable"),
        "original, by negative": (pr.source == "original") & (pr.by_split == "negative_by"),
        "original, by n.s. positive": (pr.source == "original") & (pr.by_split == "ns_pos"),
        "all pairs": pd.Series(True, index=pr.index),
    }
    return pr, pops


# ---------------------------------------------------------------- step 1
def projection(args, site, items, P, pairs_folds, das_items, bt, pr, pops):
    proj = pd.read_parquet(Path(args.out_dir) / f"projections_site{site}.parquet").set_index("pid")
    das = das_items.assign(pid=das_items.prompt.map(dict(zip(P.prompt, P.pid))))
    verb_pair = dict(zip(das.verb, das.pair_id))
    z = np.zeros((len(P), 15))
    raw = np.zeros((len(P), 15))
    keys = []
    for j, col in enumerate(sorted(proj.columns)):
        k, f = int(col[1]), int(col.split("_f")[1])
        keys.append((k, f))
        fo = das.pair_id.map(dict(zip(pairs_folds.pair_id, pairs_folds[f"fold_split{k}"])))
        x = proj[col].to_numpy()
        xd = x[das.pid.to_numpy()]
        tr = (fo != f).to_numpy() & (das.subject != "David").to_numpy()
        te = (fo == f).to_numpy()
        sign = np.sign(xd[tr & (das.cls == 1).to_numpy()].mean() - xd[tr & (das.cls == 0).to_numpy()].mean())
        x, xd = x * sign, xd * sign
        mt, mi = xd[te & (das.cls == 1).to_numpy()].mean(), xd[te & (das.cls == 0).to_numpy()].mean()
        z[:, j], raw[:, j] = (x - mi) / (mt - mi), x
    # cross-fitting: DAS verbs use only bases where their pair is held out
    elig = np.ones((len(P), 15), bool)
    for i, lemma in enumerate(P.lemma):
        if lemma in verb_pair:
            pid_ = verb_pair[lemma]
            fo = pairs_folds.set_index("pair_id").loc[pid_]
            elig[i] = [fo[f"fold_split{k}"] == f for k, f in keys]
    Z = (z * elig).sum(1) / elig.sum(1)
    RAW = (raw * elig).sum(1) / elig.sum(1)
    pidx = dict(zip(P.prompt, P.pid))
    it = items.assign(zg=items.good_prompt.map(pidx).map(lambda i: Z[i]),
                      zb=items.bad_prompt.map(pidx).map(lambda i: Z[i]),
                      rg=items.good_prompt.map(pidx).map(lambda i: RAW[i]),
                      rb=items.bad_prompt.map(pidx).map(lambda i: RAW[i]))
    it["diff"], it["rdiff"] = it.zg - it.zb, it.rg - it.rb
    # active frame, same verbs
    act = P[P.kind == "active"].set_index(["lemma", "subject"]).pid
    arows = []
    for p in pr.itertuples():
        for s in SUBJ:
            g, b = act.get((p.good_lemma, s)), act.get((p.bad_lemma, s))
            arows.append({"pair_id": p.Index, "context_id": s, "zg": Z[g], "zb": Z[b], "rg": RAW[g], "rb": RAW[b]})
    at = pd.DataFrame(arows)
    at["diff"], at["rdiff"] = at.zg - at.zb, at.rg - at.rb

    pair_ix = {p: i for i, p in enumerate(pr.index)}
    rows = []
    for frame, df, cl in (("passive", it, sorted(items.context_id.unique())), ("active", at, SUBJ)):
        cix = {c: i for i, c in enumerate(cl)}
        sub = Boot(len(cl), args.n_boot, args.seed) if frame == "active" else bt
        df = df.assign(dq=0)
        v, m = cells(df, pair_ix, cix, {0: 0}, 1, ["diff", "rdiff", "zg", "zb"])
        A = pair_values(v, m, sub.wc, np.ones((sub.B + 1, 1), np.float32))
        for name in ("primary (plain)", "all pairs", "prep_object, pseudo_passive_ok", "prep_object, pseudo_passive_bad"):
            mask = pops[name]
            for band in ("all", "head", "tail", "xtail"):
                sel = mask & ((pr.band == band) if band != "all" else True)
                pix = np.flatnonzero(sel.to_numpy())
                if len(pix) == 0:
                    continue
                wp = sub.pair_weights(pr.band.to_numpy()[pix])
                S = summarize(A, wp, pix)
                d = df[df.pair_id.isin(pr.index[pix])]
                gv = d.groupby("pair_id")[["zg", "zb"]].mean()
                rows.append({"site": site, "frame": frame, "population": name, "band": band, "n_pairs": len(pix),
                             "diff_z": ci(S[:, 0]), "diff_raw": ci(S[:, 1]), "good_z": float(S[0, 2]),
                             "bad_z": float(S[0, 3]), "auc_items": auc(d.zg, d.zb), "auc_verbs": auc(gv.zg, gv.zb),
                             "win_rate": float((d.zg > d.zb).mean())})
    return rows


# ---------------------------------------------------------------- step 2
def transfer(args, site, items, plan, nat, bt, pr, pops, delta):
    pat = pd.read_parquet(Path(args.out_dir) / f"patches_site{site}.parquet")
    if not (pat.row.to_numpy() == plan.row.to_numpy()).all():
        raise SystemExit("patch rows out of order")
    df = plan[["item_id", "side", "cond", "base", "donor_pair"]].copy()
    natv = nat.set_index("pid").loc[plan.base.to_numpy(), list(READ)].to_numpy()
    df[list(READ)] = pat[list(READ)].to_numpy() - natv
    it = items.set_index("item_id")[["pair_id", "context_id"]]
    df = df.join(it, on="item_id")
    pair_ix = {p: i for i, p in enumerate(pr.index)}
    ctx_ix = {c: i for i, c in enumerate(sorted(items.context_id.unique()))}
    das_q = sorted(df[df.cond == "T"].donor_pair.dropna().unique())
    m_q = sorted(df[df.cond == "matched_T"].donor_pair.dropna().unique())
    out = {}

    def arm(side, conds, qset, key):
        d = df[(df.side == side) & df.cond.isin(conds)].assign(dq=lambda x: x.donor_pair)
        if qset is None:
            v, m = cells(d, pair_ix, ctx_ix, {0: 0}, 1, READ)
            return pair_values(v, m, bt.wc, np.ones((bt.B + 1, 1), np.float32))
        qix = {q: i for i, q in enumerate(qset)}
        v, m = cells(d, pair_ix, ctx_ix, qix, len(qset), READ)
        return pair_values(v, m, bt.wc, bt.donor_weights(key, len(qset)))

    def paired_d(side):
        d = df[(df.side == side) & df.cond.isin(["T", "I"])].assign(dq=lambda x: x.donor_pair)
        qix = {q: i for i, q in enumerate(das_q)}
        dT, dI = d[d.cond == "T"], d[d.cond == "I"]
        vT, mT = cells(dT, pair_ix, ctx_ix, qix, len(das_q), READ)
        vI, mI = cells(dI, pair_ix, ctx_ix, qix, len(das_q), READ)
        if not (mT == mI).all():
            raise SystemExit("T and I donor cells are not paired")
        return pair_values(vT - vI, mT, bt.wc, bt.donor_weights("das", len(das_q)))

    out["D_bad"] = paired_d("bad")
    out["D_good"] = paired_d("good")
    for side in ("bad", "good"):
        out[f"T_{side}"] = arm(side, ["T"], das_q, "das")
        out[f"I_{side}"] = arm(side, ["I"], das_q, "das")
        out[f"same_{side}"] = arm(side, ["same_verb"], None, None)
        out[f"swap_{side}"] = arm(side, ["passive_swap"], None, None)
    out["Dm_bad"] = arm("bad", ["matched_T"], m_q, "matched") - arm("bad", ["matched_I"], m_q, "matched")
    # natural good - bad gap on the same items
    g = items[["pair_id", "context_id", "good_prompt", "bad_prompt"]].copy()
    pid = nat.set_index("pid")
    P = args._P
    pidx = dict(zip(P.prompt, P.pid))
    gv = pid.loc[g.good_prompt.map(pidx), list(READ)].to_numpy() - pid.loc[g.bad_prompt.map(pidx), list(READ)].to_numpy()
    g[list(READ)] = gv
    v, m = cells(g.assign(dq=0), pair_ix, ctx_ix, {0: 0}, 1, READ)
    out["nat_gap"] = pair_values(v, m, bt.wc, np.ones((bt.B + 1, 1), np.float32))

    rows = []
    for name, mask in pops.items():
        for band in ("all", "head", "tail", "xtail"):
            sel = mask & ((pr.band == band) if band != "all" else True)
            pix = np.flatnonzero(sel.to_numpy())
            if len(pix) == 0:
                continue
            wp = bt.pair_weights(pr.band.to_numpy()[pix])
            S = {k: summarize(A, wp, pix) for k, A in out.items()}
            for k, s in S.items():
                for r, rd in enumerate(READ):
                    row = {"site": site, "population": name, "band": band, "n_pairs": len(pix), "condition": k,
                           "readout": rd, **ci(s[:, r])}
                    if k == "D_bad" and rd in ("O", "by", "dot"):
                        row["class"] = classify(row, delta)
                    if k == "D_bad":
                        frac = s[:, r] / S["nat_gap"][:, r]
                        row.update({f"frac_natgap_{a}": b for a, b in ci(frac).items()})
                    rows.append(row)
    return rows


def run_passive(args):
    root = Path(args.root)
    items = pd.read_csv(args.items)
    P = pd.read_csv(args.prompts)
    args._P = P
    plan = pd.read_csv(args.plan, dtype={c: "category" for c in ("item_id", "side", "cond", "donor_pair")})
    nat = pd.read_parquet(Path(args.out_dir) / "natural.parquet")
    pairs_folds = pd.read_csv(root / "final_strict" / "pairs_folds.csv")
    das_items = pd.read_csv(root / "final_strict" / "items.csv")
    pr, pops = populations(items)
    deltas = {17: json.loads((root / "frozen_config_final.json").read_text())["tost"]["delta"]}
    for s in (8, 12):
        deltas[s] = json.loads((root / f"frozen_config_site{s}.json").read_text())["tost"]["delta"]
    proj_rows, tr_rows = [], []
    for site in (17, 12, 8):
        bt = Boot(items.context_id.nunique(), args.n_boot, args.seed)
        proj_rows += projection(args, site, items, P, pairs_folds, das_items, bt, pr, pops)
        tr_rows += transfer(args, site, items, plan, nat, bt, pr, pops, deltas[site])
        print(f"site {site} done", flush=True)
    proj = pd.json_normalize(proj_rows)
    tr = pd.DataFrame(tr_rows)
    proj.to_csv(Path(args.out_dir) / "projection_summary.csv", index=False)
    tr.to_csv(Path(args.out_dir) / "transfer_summary.csv", index=False)
    (Path(args.out_dir) / "deltas.json").write_text(json.dumps(deltas, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("delta", "passive"))
    ap.add_argument("--root", default="results/das_round2")
    ap.add_argument("--items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--prompts", default="data/das_round2/passive_test/prompts.csv")
    ap.add_argument("--plan", default="data/das_round2/passive_test/plan.csv.gz")
    ap.add_argument("--out-dir", default="results/das_round2/passive_test")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    run_delta(args) if args.mode == "delta" else run_passive(args)
