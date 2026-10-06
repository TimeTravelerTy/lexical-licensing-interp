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


def abs_probs(args, items, plan, nat):
    """Mean probabilities (not logs) on primary bases: unpatched, and after T / I donors at each site."""
    prim = set(items[items.bad_class == "plain"].item_id)
    rd = ["O", "det", "pron", "refl", "the", "him", "by", "dot", "I"]
    n = nat.set_index("pid")
    rows = []
    for side in ("bad", "good"):
        sel = plan.item_id.isin(prim).to_numpy() & (plan.side == side).to_numpy()
        rows.append({"site": "", "side": side, "condition": "unpatched",
                     **np.exp(n.loc[plan[sel].base.unique(), rd]).mean().to_dict()})
        for site in (8, 12, 17):
            pat = pd.read_parquet(Path(args.out_dir) / f"patches_site{site}.parquet", columns=["row"] + rd)
            for c in ("T", "I"):
                r = plan[sel & (plan.cond == c).to_numpy()].row.to_numpy()
                rows.append({"site": site, "side": side, "condition": f"{c} donor",
                             **np.exp(pat.loc[r, rd]).mean().to_dict()})
    pd.DataFrame(rows).to_csv(Path(args.out_dir) / "abs_probs.csv", index=False)


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
    abs_probs(args, items, plan, nat)
    proj = pd.json_normalize(proj_rows)
    tr = pd.DataFrame(tr_rows)
    proj.to_csv(Path(args.out_dir) / "projection_summary.csv", index=False)
    tr.to_csv(Path(args.out_dir) / "transfer_summary.csv", index=False)
    (Path(args.out_dir) / "deltas.json").write_text(json.dumps(deltas, indent=2) + "\n")


# ---------------------------------------------------------------- report
def fmt(r, k=2):
    return f"{r['est']:.{k}f} [{r['lo95']:.{k}f}, {r['hi95']:.{k}f}]"


def write_report(args):
    out = Path(args.out_dir)
    tr = pd.read_csv(out / "transfer_summary.csv")
    pj = pd.read_csv(out / "projection_summary.csv")
    ab = pd.read_csv(out / "abs_probs.csv")
    deltas = {int(k): v for k, v in json.loads((out / "deltas.json").read_text()).items()}
    meta = json.loads((out / "run_meta.json").read_text())
    sites = (17, 12, 8)
    role = {17: "primary", 12: "secondary", 8: "secondary"}

    def get(pop, band, cond, rd, site):
        x = tr[(tr.population == pop) & (tr.band == band) & (tr.condition == cond) & (tr.readout == rd)
               & (tr.site == site)]
        return x.iloc[0]

    P = "primary (plain)"
    L = ["# Passive test of the round-2 direction: results", "",
         "Spec, declared and committed before any passive evaluation: `passive_test_plan.md`. "
         "Run: `run_passive_test.py` (TSUBAME job 8916904, commit `f769ec9`); analysis: "
         "`analyze_passive_test.py` (2,000 draws, seed 17; three-way cluster bootstrap over base verb pairs "
         "within band, contexts and donor verb pairs). Tables: `results/das_round2/passive_test/*_summary.csv`.",
         "", "Fidelity checks (per site): a self-patch reproduces the unpatched readout exactly "
         "(max |dM| = 0); recomputing 512 held-out active swaps of each site's own run gives median |dM| "
         + ", ".join(f"{meta['checks'][str(s)]['heldout_recompute_median_abs_dM']:.3f}" for s in sites)
         + " and max " + ", ".join(f"{meta['checks'][str(s)]['heldout_recompute_max_abs_dM']:.2f}" for s in sites)
         + " (sites 17, 12, 8; bf16 batch effects, M is on a scale of about +-5).", ""]

    # verdict
    L += ["## Verdict (primary population: 64 pairs with a plain bad verb, all bands pooled)", "",
          "D = mean change after a transitive active donor minus after an intransitive active donor, on bad "
          "passive bases (\"The house was emerged\"), in nats.", "",
          "| Site | Role | δ_s | D log P(O) | class | D log P(\" by\") | class | D log P(\".\") | class | Outcome |",
          "|---:|---|---:|---|---|---|---|---|---|---|"]
    for st in sites:
        o, b, d = (get(P, "all", "D_bad", r, st) for r in ("O", "by", "dot"))
        L.append(f"| {st} | {role[st]} | {deltas[st]:.2f} | {fmt(o)} | {o['class']} | {fmt(b)} | {b['class']} | "
                 f"{fmt(d)} | {d['class']} | **{outcome(o['class'], b['class'])}** |")
    L += ["", "Rules (declared): RISE = 95% CI above 0 and estimate >= δ_s; NO RISE = 90% CI within ±δ_s; "
          "outcome from the O and \" by\" classes.", ""]

    # absolute probabilities
    L += ["### In probabilities", "",
          "Mean probability over primary bad passive bases (and, for reference, good passives unpatched).", "",
          "| Condition | P(O) | P(pronouns) | P(\" the\") | P(\" by\") | P(\".\") | P(I) |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for r in ab.itertuples():
        if r.side == "good" and r.condition != "unpatched":
            continue
        name = f"{r.side} passive, unpatched" if r.condition == "unpatched" else f"site {int(float(r.site))}, {r.condition}"
        L.append(f"| {name} | {r.O:.3f} | {r.pron:.4f} | {r.the:.3f} | {r.by:.3f} | {r.dot:.3f} | {r.I:.3f} |")
    L.append("")

    # components
    L += ["### Where the change goes (D on bad bases, primary)", "",
          "| Readout | " + " | ".join(f"site {s}" for s in sites) + " |", "|---|" + "---|" * len(sites)]
    for rd in ("O", "det", "pron", "refl", "the", "him", "by", "dot", "I", "M"):
        L.append(f"| {LABEL[rd]} | " + " | ".join(fmt(get(P, "all", "D_bad", rd, s)) for s in sites) + " |")
    L += ["", "D(\" by\") as a fraction of the natural good − bad gap in log P(\" by\") on the same items "
          f"({get(P, 'all', 'nat_gap', 'by', 17)['est']:.2f} nats): "
          + ", ".join(f"site {s} {get(P, 'all', 'D_bad', 'by', s)['frac_natgap_est']:.2f}" for s in sites)
          + ". D(O) as a fraction of the site's active effect: "
          + ", ".join(f"site {s} {get(P, 'all', 'D_bad', 'O', s)['est'] / (5 * deltas[s]):.2f}" for s in sites)
          + ".", ""]

    # controls
    L += ["## Controls (primary, Δ from unpatched)", "",
          "| Condition | " + " | ".join(f"site {s}: O / by / ." for s in sites) + " |", "|---|" + "---|" * 3]
    names = [("T_bad", "T donor → bad passive"), ("I_bad", "I donor → bad passive (voice-change baseline)"),
             ("same_bad", "own active → bad passive (same verb)"), ("swap_bad", "good passive → bad passive"),
             ("T_good", "T donor → good passive (good → good)"), ("I_good", "I donor → good passive"),
             ("same_good", "own active → good passive (same verb)"), ("swap_good", "bad passive → good passive"),
             ("D_good", "D on good bases"), ("Dm_bad", "D, participle-matched donors"),
             ("nat_gap", "natural good − bad (unpatched)")]
    for c, n in names:
        L.append(f"| {n} | " + " | ".join(" / ".join(f"{get(P, 'all', c, r, s)['est']:+.2f}" for r in ("O", "by", "dot"))
                                       for s in sites) + " |")
    L.append("")

    # step 1
    L += ["## Step 1: projection of natural participles onto d", "",
          "z: 0 = mean of held-out intransitive actives, 1 = held-out transitive actives (per basis, "
          "cross-fitted). Gap = good − bad (two-way bootstrap over pairs and contexts / subjects).", "",
          "| Site | Frame | Band | Pairs | Gap in z | good z | bad z | AUC items | AUC verbs | Win rate |",
          "|---:|---|---|---:|---|---:|---:|---:|---:|---:|"]
    for st in sites:
        for fr in ("passive", "active"):
            for band in ("all", "head", "tail", "xtail"):
                x = pj[(pj.site == st) & (pj.frame == fr) & (pj.band == band) & (pj.population == P)]
                if len(x) == 0:
                    continue
                r = x.iloc[0]
                L.append(f"| {st} | {fr} | {band} | {r.n_pairs} | {r['diff_z.est']:.2f} [{r['diff_z.lo95']:.2f}, "
                         f"{r['diff_z.hi95']:.2f}] | {r.good_z:.2f} | {r.bad_z:.2f} | {r.auc_items:.3f} | "
                         f"{r.auc_verbs:.3f} | {r.win_rate:.3f} |")
    L.append("")

    # bands and groups
    L += ["## By band (primary) and separate groups (descriptive)", "",
          "D on bad bases, O / \" by\" / \".\" (point estimates; CIs in `transfer_summary.csv`).", "",
          "| Population | Band | Pairs | " + " | ".join(f"site {s}" for s in sites) + " |", "|---|---|---:|" + "---|" * 3]
    pops = [P] * 4 + list(dict.fromkeys(tr.population))
    seen = set()
    for pop in pops:
        bands = ("head", "tail", "xtail") if pop == P and (pop, "head") not in seen else ("all",)
        for band in bands:
            if (pop, band) in seen or pop == P and band == "all":
                continue
            seen.add((pop, band))
            if len(tr[(tr.population == pop) & (tr.band == band)]) == 0:
                continue
            n = get(pop, band, "D_bad", "O", 17).n_pairs
            cells_ = []
            for s in sites:
                cells_.append(" / ".join(f"{get(pop, band, 'D_bad', r, s)['est']:+.2f}" for r in ("O", "by", "dot")))
            L.append(f"| {pop} | {band} | {n} | " + " | ".join(cells_) + " |")
    L.append("")
    Path(args.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))



# ---------------------------------------------------------------- site-8 controls (post hoc)
def run_controls(args, site=8):
    out = Path(args.out_dir)
    items = pd.read_csv(args.items)
    plan = pd.read_csv(args.plan)
    nat = pd.read_parquet(out / "natural.parquet").set_index("pid")
    delta = json.loads((Path(args.root) / f"frozen_config_site{site}.json").read_text())["tost"]["delta"]
    meta = json.loads((out / f"controls_meta_site{site}.json").read_text())
    pr, pops = populations(items)
    prim = pops["primary (plain)"]
    pix = np.flatnonzero(prim.to_numpy())
    pair_ix = {p: i for i, p in enumerate(pr.index)}
    ctx_ix = {c: i for i, c in enumerate(sorted(items.context_id.unique()))}
    item_pair = items.set_index("item_id")[["pair_id", "context_id"]]
    das_pat = pd.read_parquet(out / f"patches_site{site}.parquet")

    def deltas_for(pat):
        if "M" not in pat:
            pat = pat.assign(M=pat.O - pat.I)  # M = log P(O) - log P(I) exactly
        d = plan.loc[pat.row.to_numpy(), ["item_id", "split", "cond", "base", "donor_pair"]].reset_index(drop=True)
        d[list(READ)] = pat[list(READ)].to_numpy() - nat.loc[d.base.to_numpy(), list(READ)].to_numpy()
        return d.join(item_pair, on="item_id")

    def d_boot(d):
        bt = Boot(len(ctx_ix), args.n_boot, args.seed)
        q = sorted(d.donor_pair.unique())
        qix = {x: i for i, x in enumerate(q)}
        dd = d.assign(dq=d.donor_pair)
        vT, mT = cells(dd[dd.cond == "T"], pair_ix, ctx_ix, qix, len(q), READ)
        vI, mI = cells(dd[dd.cond == "I"], pair_ix, ctx_ix, qix, len(q), READ)
        assert (mT == mI).all()
        A = pair_values(vT - vI, mT, bt.wc, bt.donor_weights("das", len(q)))
        wp = bt.pair_weights(pr.band.to_numpy()[pix])
        return summarize(A, wp, pix)

    sh_pat = pd.read_parquet(out / f"controls_shuffled_site{site}.parquet")
    sh = deltas_for(sh_pat)
    das = deltas_for(das_pat[das_pat.row.isin(set(sh_pat.row))].reset_index(drop=True))  # same plan rows
    S_sh, S_das = d_boot(sh), d_boot(das)
    res = []
    for r, rd in enumerate(READ):
        a, b = ci(S_sh[:, r]), ci(S_das[:, r])
        diff = ci(S_das[:, r] - S_sh[:, r])  # same weights per draw (same seed), so the difference is paired
        res.append({"readout": rd, "das": b, "shuffled": a, "das_minus_shuffled": diff,
                    "class_shuffled": classify(a, delta) if rd in ("O", "by", "dot") else "",
                    "class_das": classify(b, delta) if rd in ("O", "by", "dot") else ""})

    # random null (split 0), D per item -> pair mean -> mean over primary pairs
    rnd = pd.read_parquet(out / f"controls_random_site{site}.parquet")
    rnd["M"] = rnd.O - rnd.I
    rnd = rnd.join(item_pair, on="item_id")
    rnd = rnd[rnd.pair_id.isin(pr.index[pix])]
    null = rnd.groupby(["control", "draw", "pair_id"])[list(READ)].mean().groupby(["control", "draw"]).mean()
    d0 = das[das.split == 0]
    g = d0.groupby(["item_id", "cond"])[list(READ)].mean().unstack("cond")
    D0 = pd.DataFrame({k: g[(k, "T")] - g[(k, "I")] for k in READ}).join(item_pair)
    das0 = D0[D0.pair_id.isin(pr.index[pix])].groupby("pair_id")[list(READ)].mean().mean()
    nrows = []
    for c, x in null.groupby(level="control"):
        for rd in ("O", "by", "dot", "pron", "the", "M"):
            v = x[rd].to_numpy()
            nrows.append({"control": c, "readout": rd, "das_split0": float(das0[rd]), "null_mean": float(v.mean()),
                          "null_p95": float(np.percentile(v, 95)), "null_p5": float(np.percentile(v, 5)),
                          "null_max_abs": float(np.abs(v).max()), "n_draws": len(v),
                          "frac_ge_das": float((v >= das0[rd]).mean()),
                          "n_ge_delta": int((v >= delta).sum())})
    nd = pd.DataFrame(nrows)
    nd.to_csv(out / f"controls_random_summary_site{site}.csv", index=False)
    pd.json_normalize(res).to_csv(out / f"controls_shuffled_summary_site{site}.csv", index=False)

    # active metrics of the retrained shuffled bases vs the original run
    orig = pd.read_csv(Path(args.root) / f"final_strict_site{site}" / "summary.csv")
    orig = orig[orig.control == "shuffled_labels"].set_index(["split", "fold"])
    act = pd.DataFrame(meta["shuffled_heldout_active"]).set_index(["split", "fold"])

    L = [f"# Site-{site} passive-side controls (post hoc)", "",
         "Criteria written before running: `passive_test_plan.md`, addendum. Script "
         "`run_passive_controls.py`; analysis `analyze_passive_test.py controls`. Primary population (64 pairs), "
         f"bad passive bases, T vs I donors; δ = {delta:.4f}.", "",
         "## Shuffled-label DAS (all splits, declared bootstrap)", "",
         f"Retrained shuffled-label bases, held-out active IIA {act.iia_cross.mean():.3f} (original run's control: "
         f"{orig.iia_cross.mean():.3f}; per-fold max |difference| "
         f"{(act.iia_cross - orig.iia_cross.reindex(act.index)).abs().max():.3f}).", "",
         "| Readout | DAS D | class | Shuffled-label D | class | DAS − shuffled |", "|---|---|---|---|---|---|"]
    for r in res:
        L.append(f"| {LABEL[r['readout']]} | {fmt(r['das'])} | {r['class_das']} | {fmt(r['shuffled'])} | "
                 f"{r['class_shuffled']} | {fmt(r['das_minus_shuffled'])} |")
    o = {r["readout"]: r for r in res}
    L += ["", f"Shuffled-label outcome: **{outcome(o['O']['class_shuffled'], o['by']['class_shuffled'])}**.", "",
          "## Random rank-1 directions (split 0, 100 draws)", "",
          "D per draw = mean over primary pairs (point estimate). DAS recomputed on the same split-0 rows.", "",
          "| Control | Readout | DAS (split 0) | Null mean | Null 5th–95th pct | Max abs | Draws ≥ DAS | Draws ≥ δ |",
          "|---|---|---:|---:|---|---:|---:|---:|"]
    for r in nd.itertuples():
        L.append(f"| {r.control} | {LABEL[r.readout]} | {r.das_split0:+.3f} | {r.null_mean:+.3f} | "
                 f"[{r.null_p5:+.3f}, {r.null_p95:+.3f}] | {r.null_max_abs:.3f} | {r.frac_ge_das:.2f} | "
                 f"{r.n_ge_delta} |")
    L.append("")
    Path(args.controls_report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("delta", "passive", "report", "controls"))
    ap.add_argument("--root", default="results/das_round2")
    ap.add_argument("--items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--prompts", default="data/das_round2/passive_test/prompts.csv")
    ap.add_argument("--plan", default="data/das_round2/passive_test/plan.csv.gz")
    ap.add_argument("--out-dir", default="results/das_round2/passive_test")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--report", default="reports/passive_das_prep/passive_test_results.md")
    ap.add_argument("--controls-report", default="reports/passive_das_prep/passive_controls_site8.md")
    args = ap.parse_args()
    if args.mode == "delta":
        run_delta(args)
    elif args.mode == "passive":
        run_passive(args)
        write_report(args)
    elif args.mode == "controls":
        run_controls(args)
    else:
        write_report(args)
