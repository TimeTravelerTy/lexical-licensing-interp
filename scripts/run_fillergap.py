#!/usr/bin/env python3
"""Round 4, Part B: filler-gap natural pass and patches (reports/round4/plan.md, B6 / B7).

Modes
- natural: every prompt of `build_fillergap.py items`: readouts at the last token and projections
  onto each DAS site's 15 fold bases -> `natural.parquet`, `projections.npz`. Token checks: in
  both frames the verb's tokens are the prompt's final tokens and identical across the two frames
  of an item pair; " that", " what", "?", "." are single tokens.
- patch: rows of `plan_{exp}.csv.gz` (kept I verbs as bases) at every site:
    T / I: interchange along the item's fold basis with the donor's state (h <- h + ((h_src - h).d) d);
    inT / inI: coordinate set to t_T (mean projection of the T verbs in the same experiment, frame
      and context) or t_I (mean over the kept I verbs other than the base verb), both over verbs
      outside the basis's DAS training pairs.
  -> `patches_{exp}_site{s}.parquet` (row + readouts), `targets_{exp}.parquet`.

Readouts (log-probs): O (27 object starts), PREP (26-token C9 set), "." , "?", END (".", ",", "!",
"?", ";"), " by", M.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_construction_test import SETS
from run_passive_test import PassiveRunner

SITES = (4, 6, 8, 10, 12, 14, 16, 17)
KEYS = [(k, f) for k in range(3) for f in range(5)]
READ = ("M", "O", "PREP", "dot", "q", "END", "by")


class FGRunner(PassiveRunner):
    def __init__(self, args):
        super().__init__(args)
        torch = self.torch
        for name, toks in (("PREP", SETS["PREP"]), ("END", SETS["END"]), ("q", ("?",))):
            ids = [self.tok.encode(t, add_special_tokens=False) for t in toks]
            assert all(len(i) == 1 for i in ids), name
            self.ids[name] = torch.tensor([i[0] for i in ids], device=self.device)

    def readout(self, logits, anchors, full=False):
        torch = self.torch
        rows = torch.arange(logits.shape[0], device=logits.device)
        z = logits[rows, anchors].float()
        m = torch.logsumexp(z[:, self.ids["O"]], -1) - torch.logsumexp(z[:, self.ids["I"]], -1)
        if not full:
            return m, None
        lp = torch.log_softmax(z, -1)
        parts = {k: torch.logsumexp(lp[:, self.ids[k]], -1) for k in READ[1:]}
        parts["_prep_tokens"] = lp[:, self.ids["PREP"]]
        return m, parts


def load_bases(path, device):
    import torch

    npz = np.load(path)
    B = {s: {kf: torch.tensor(npz[f"das_site{s}_s{kf[0]}_f{kf[1]}"].reshape(-1), dtype=torch.float32, device=device)
             for kf in KEYS} for s in SITES}
    assert max(abs(float(b.norm()) - 1) for s in SITES for b in B[s].values()) < 1e-3
    return B, hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_checks(tok, items):
    bad = []
    for (exp, lemma, ctx), g in items.groupby(["exp", "lemma", "context_id"]):
        form = g.form.iloc[0]
        v = tok.encode(" " + form, add_special_tokens=False)
        ids = {fr: tok.encode(p, add_special_tokens=False) for fr, p in zip(g.frame, g.prompt)}
        for fr, x in ids.items():
            if x[-len(v):] != v:
                bad.append((exp, lemma, ctx, fr))
    if bad:
        raise SystemExit(f"{len(bad)} items whose verb tokens are not the prompt's final tokens: {bad[:5]}")
    for t in (" that", " what", "?", ".", " What"):
        assert len(tok.encode(t, add_special_tokens=False)) == 1, t


def run_natural(args):
    import torch

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runner = FGRunner(args)
    ddir = Path(args.data_dir)
    P, items = pd.read_csv(ddir / "prompts.csv"), pd.read_csv(ddir / "items.csv")
    assert (P.pid.to_numpy() == np.arange(len(P))).all()
    token_checks(runner.tok, items)
    vt = items.drop_duplicates(["exp", "lemma"])[["exp", "lemma", "cls", "form"]].copy()
    vt["n_tokens"] = [len(runner.tok.encode(" " + f, add_special_tokens=False)) for f in vt.form]
    vt.to_csv(out / "verb_tokens.csv", index=False)
    bases, sha = load_bases(args.bases, runner.device)
    Bst = torch.stack([torch.stack([bases[s][kf] for kf in KEYS], 1) for s in SITES])
    prompts = P.prompt.tolist()
    nat = {k: np.empty(len(P), np.float32) for k in READ}
    proj = np.empty((len(P), len(SITES), 15), np.float32)
    ptok = np.empty((len(P), len(SETS["PREP"])), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(P), args.batch):
            enc, anchors = runner.encode(prompts[i:i + args.batch])
            o = runner.model(**enc, output_hidden_states=True, use_cache=False)
            m, parts = runner.readout(o.logits, anchors, full=True)
            rows = torch.arange(len(anchors), device=runner.device)
            x = torch.stack([o.hidden_states[s][rows, anchors].float() for s in SITES], 1)
            proj[i:i + len(m)] = torch.einsum("bsd,sdk->bsk", x, Bst).cpu().numpy()
            nat["M"][i:i + len(m)] = m.cpu().numpy()
            for k in READ[1:]:
                nat[k][i:i + len(m)] = parts[k].cpu().numpy()
            ptok[i:i + len(m)] = parts["_prep_tokens"].cpu().numpy()
    pd.DataFrame({"pid": P.pid, **nat}).to_parquet(out / "natural.parquet", index=False)
    np.savez_compressed(out / "projections.npz", pid=P.pid.to_numpy(), sites=np.array(SITES),
                        keys=np.array([f"s{k}_f{f}" for k, f in KEYS]), proj=proj)
    np.savez_compressed(out / "natural_prep_tokens.npz", pid=P.pid.to_numpy(), tokens=np.array(SETS["PREP"]), lp=ptok)
    meta = {"args": vars(args), "prompts": len(P), "items": len(items), "token_checks": "passed",
            "bases_sha256": sha, "seconds": round(time.time() - t0, 1)}
    (out / "natural_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps(meta, default=str), flush=True)


def targets(items, plan, proj, pf, kept, site_ix):
    """Per plan row (inT / inI), the target raw projection under the row's basis."""
    fo = pf.set_index("pair_id")
    it = items.set_index("item_id")
    out = np.full(len(plan), np.nan)
    nmin = np.full(len(plan), np.nan)
    sub = plan[plan.cond.isin(["inT", "inI"])]
    for (k, f), g in sub.groupby(["split", "fold"]):
        j = KEYS.index((k, f))
        train = set(fo.index[fo[f"fold_split{k}"] != f])
        pool = items[(~items.das_pair.fillna("").isin(train))]
        pool = pool.assign(x=proj[pool.pid.to_numpy(), site_ix, j])
        T = pool[pool.cls == "T"].groupby(["exp", "frame", "context_id"]).x.agg(["mean", "count"])
        Ipool = pool[(pool.cls == "I") & pool.lemma.isin(kept)]
        Isum = Ipool.groupby(["exp", "frame", "context_id"]).x.agg(["sum", "count"])
        own = Ipool.set_index("item_id").x
        b = it.loc[g.item_id]
        key = list(zip(b.exp, b.frame, b.context_id))
        tT = T.loc[key]["mean"].to_numpy()
        nT = T.loc[key]["count"].to_numpy()
        s = Isum.loc[key]
        o = own.reindex(g.item_id).fillna(0).to_numpy()  # leave the base verb out (if it is in the pool)
        n = s["count"].to_numpy() - own.reindex(g.item_id).notna().to_numpy()
        tI = (s["sum"].to_numpy() - o) / n
        out[g.index.to_numpy()] = np.where(g.cond.to_numpy() == "inT", tT, tI)
        nmin[g.index.to_numpy()] = np.where(g.cond.to_numpy() == "inT", nT, n)
    return out, nmin


def run_patch(args):
    import torch

    out = Path(args.out_dir)
    gates = json.loads((out / "gates.json").read_text())
    if not gates[args.exp]["passes"]:
        raise SystemExit(f"{args.exp}: the natural gate did not pass; no patching (plan.md, B6/B7)")
    nmeta = json.loads((out / "natural_meta.json").read_text())
    if hashlib.sha256(Path(args.bases).read_bytes()).hexdigest() != nmeta["bases_sha256"]:
        raise SystemExit("bases differ from the stage-1 natural pass")
    runner = FGRunner(args)
    ddir = Path(args.data_dir)
    P, items = pd.read_csv(ddir / "prompts.csv"), pd.read_csv(ddir / "items.csv")
    plan = pd.read_csv(ddir / f"plan_{args.exp}.csv.gz")
    kept = set(json.loads((ddir / f"plan_{args.exp}_meta.json").read_text())["kept_I_verbs"])
    pf = pd.read_csv(args.folds)
    proj = np.load(out / "projections.npz")["proj"]
    bases, sha = load_bases(args.bases, runner.device)
    prompts = P.prompt.tolist()
    donors = np.unique(plan.donor[plan.donor >= 0].to_numpy())
    dstate = torch.zeros(len(P), len(SITES), runner.model.config.hidden_size, device=runner.device)
    with torch.no_grad():
        for i in range(0, len(donors), args.batch):
            ids = donors[i:i + args.batch]
            enc, anchors = runner.encode([prompts[j] for j in ids])
            o = runner.model(**enc, output_hidden_states=True, use_cache=False)
            rr = torch.arange(len(anchors), device=runner.device)
            dstate[torch.as_tensor(ids, device=runner.device)] = torch.stack(
                [o.hidden_states[s][rr, anchors].float() for s in SITES], 1)
    tg = {}
    nat_M = pd.read_parquet(out / "natural.parquet").set_index("pid").M.reindex(np.arange(len(P))).to_numpy()
    meta = {"args": vars(args), "plan_rows": len(plan), "bases_sha256": sha, "checks": {},
            "plan_content_sha256": hashlib.sha256(pd.util.hash_pandas_object(plan, index=False).values.tobytes()).hexdigest()}
    for si, site in enumerate(SITES):
        t0 = time.time()
        t, nmin = targets(items, plan, proj, pf, kept, si)
        isin = plan.cond.isin(["inT", "inI"]).to_numpy()
        if np.nanmin(nmin[isin]) < 5:
            raise SystemExit(f"site {site}: an in-range target has fewer than 5 eligible verbs")
        tg[f"t_site{site}"] = t
        # self-set check: setting the coordinate to the base's own projection reproduces the natural readout
        pick = plan[isin].drop_duplicates("base").head(256)
        with torch.no_grad():
            ms = []
            for (k, f), g in pick.groupby(["split", "fold"]):
                d = bases[site][(k, f)]
                own = torch.as_tensor(proj[g.base.to_numpy(), si, KEYS.index((k, f))], device=runner.device)
                m, _ = runner.patched([prompts[j] for j in g.base], own[:, None] * d[None], site, d[:, None])
                ms.append(np.abs(m.cpu().numpy() - nat_M[g.base.to_numpy()]))
        meta["checks"][str(site)] = {"self_set_max_abs_dM": float(np.concatenate(ms).max()),
                                     "min_eligible_target_verbs": int(np.nanmin(nmin[isin]))}
        res = {k: np.full(len(plan), np.nan, np.float32) for k in READ}
        with torch.no_grad():
            for (k, f), g in plan.groupby(["split", "fold"]):
                d = bases[site][(k, f)]
                for i in range(0, len(g), args.batch):
                    gg = g.iloc[i:i + args.batch]
                    rows = gg.row.to_numpy()
                    don = gg.donor.to_numpy()
                    is_set = gg.cond.isin(["inT", "inI"]).to_numpy()
                    src = dstate[torch.as_tensor(np.where(is_set, 0, don), device=runner.device), si].clone()
                    if is_set.any():
                        tt = torch.as_tensor(t[rows[is_set]], dtype=torch.float32, device=runner.device)
                        src[torch.as_tensor(np.flatnonzero(is_set), device=runner.device)] = tt[:, None] * d[None]
                    m, parts = runner.patched([prompts[j] for j in gg.base], src, site, d[:, None], full=True)
                    res["M"][rows] = m.cpu().numpy()
                    for key in READ[1:]:
                        res[key][rows] = parts[key].cpu().numpy()
        assert not np.isnan(res["M"]).any()
        pd.DataFrame({"row": plan.row, **res}).to_parquet(out / f"patches_{args.exp}_site{site}.parquet", index=False)
        print(f"{args.exp} site {site}: {len(plan)} rows in {time.time() - t0:.0f}s; {meta['checks'][str(site)]}",
              flush=True)
    pd.DataFrame({"row": plan.row, **tg}).to_parquet(out / f"targets_{args.exp}.parquet", index=False)
    (out / f"patch_meta_{args.exp}.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("natural", "patch"))
    ap.add_argument("--exp", choices=("emb", "mat"))
    ap.add_argument("--data-dir", default="data/round4/fillergap")
    ap.add_argument("--out-dir", default="results/round4/fillergap")
    ap.add_argument("--bases", default="results/das_round2/bases_rank1.npz")
    ap.add_argument("--folds", default="results/das_round2/final_strict/pairs_folds.csv")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch", type=int, default=1024)
    a = ap.parse_args()
    run_natural(a) if a.mode == "natural" else run_patch(a)
