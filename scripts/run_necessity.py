#!/usr/bin/env python3
"""Round 5, B5: necessity on good (transitive) passives (reports/round5/plan.md, B5). fp32.

Frames (token ids, as run_path_patch_frames.py): "The N has been V" (hasbeen, primary) and "The N was V" (was).
1. Natural pass over every primary passive-test item (good and bad prompt, both frames): readouts at the
   participle's last token, the site-8 projection on the five split-0 fold bases, and the post-activations of
   every captured neuron (top 50 switch neurons, the E13-T random1-5 and wmatched sets, five layer-matched random
   10-neuron sets drawn here) in MLPs 11 and 14.
2. Targets per item and frame (context c, eligible pairs = primary pairs other than the item's own and outside
   the DAS training pairs of its split-0 basis): t_I / t_T = mean projection of the eligible bad / good items in c;
   per neuron, the mean activation over the eligible bad (I) / good (T) items in c.
3. On the B7 items' good prompts (32 contexts per pair, seed 17), one at a time: d_I, d_T (site-8 coordinate set),
   n10_I, n50_I, n50_T, r10_k, r50_k (k = 1..5), w50 (neurons set), c10, c50 (both); and the equalized coordinate
   (good and bad prompts with the coordinate set to (t_T + t_I) / 2), capturing the top-50 activations.
Checks: own natural values as targets reproduce the natural readouts; patched entries equal their targets and
nothing else in the hooked tensor changes at that hook.

Outputs (--out-dir): necessity_natural.parquet, necessity_rows.parquet, necessity_neurons.csv, necessity_meta.json.
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
from run_das_round2 import O_TOKENS, normalize_output, replace_output
from run_passive_test import PassiveRunner
from run_site8_mechanism import SITE, Mech, load_bases

LAYERS = (11, 14)
FRAMES = {"hasbeen": (" has", " been"), "was": (" was",)}
READ = ("by", "O", "dot", "prep_noby")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def neuron_sets(args, rng):
    sw = pd.read_csv(args.switch)
    top = sw.sort_values("rank")
    rows = []
    for k in (10, 50):
        for r in top[top["rank"] <= k].itertuples():
            rows.append(("top%d" % k, int(r.layer), int(r.index)))
    nd = pd.read_csv(args.neurons)
    for s in [f"random{i}" for i in range(1, 6)] + ["wmatched"]:
        for r in nd[nd.set == s].itertuples():
            rows.append((s.replace("random", "r50_") if s != "wmatched" else "w50", int(r.layer), int(r.index)))
    t50 = {(l, i) for nm, l, i in rows if nm == "top50"}
    n_l = {l: sum(1 for nm, ll, _ in rows if nm == "top10" and ll == l) for l in LAYERS}  # 4 in MLP11, 6 in MLP14
    for k in range(1, 6):
        for l in LAYERS:
            pool = np.array([i for i in range(args.d_mlp) if (l, i) not in t50])
            for i in rng.choice(pool, n_l[l], replace=False):
                rows.append((f"r10_{k}", l, int(i)))
    S = pd.DataFrame(rows, columns=["set", "layer", "index"])
    assert not S.duplicated().any()
    return S


def build_ids(tok, items, side):
    """Token ids of the good or bad prompt in both frames; the participle ids are the prompt's tail."""
    was = tok.encode(" was", add_special_tokens=False)[0]
    frame_tok = {f: [tok.encode(t, add_special_tokens=False)[0] for t in ts] for f, ts in FRAMES.items()}
    out = {f: [] for f in FRAMES}
    for pre, prompt in zip(items.prefix, items[f"{side}_prompt"]):
        pre = pre.rstrip()
        assert pre.endswith(" was")
        ctx = tok.encode(pre[: -len(" was")], add_special_tokens=False)
        full = tok.encode(prompt, add_special_tokens=False)
        assert full[:len(ctx)] == ctx and full[len(ctx)] == was
        verb = full[len(ctx) + 1:]
        for f in FRAMES:
            ids = ctx + frame_tok[f] + verb
            assert tok.encode(tok.decode(ids), add_special_tokens=False) == ids
            out[f].append(ids)
    assert out["was"] == [tok.encode(p, add_special_tokens=False) for p in items[f"{side}_prompt"]]
    return out


class Patcher:
    def __init__(self, runner, S):
        self.r, self.t = runner, runner.torch
        self.layers = runner.model.gpt_neox.layers
        t, dev = self.t, runner.device
        self.S = S.reset_index(drop=True)
        # captured neurons: all distinct (layer, index); per layer its indices and their columns in the capture
        self.cap_cols = self.S.drop_duplicates(["layer", "index"]).reset_index(drop=True)
        self.nidx = {l: t.as_tensor(self.cap_cols[self.cap_cols.layer == l]["index"].to_numpy(), device=dev) for l in LAYERS}
        self.ncol = {l: np.flatnonzero((self.cap_cols.layer == l).to_numpy()) for l in LAYERS}
        col = {(int(l), int(i)): j for j, (l, i) in enumerate(zip(self.cap_cols.layer, self.cap_cols["index"]))}
        self.set_cols = {s: np.array([col[(int(l), int(i))] for l, i in zip(g.layer, g["index"])])
                         for s, g in self.S.groupby("set")}
        self.ids = {}
        one = lambda x: runner.tok.encode(x, add_special_tokens=False)
        self.ids["by"] = t.tensor(one(" by"), device=dev)
        self.ids["dot"] = t.tensor(one("."), device=dev)
        self.ids["O"] = t.tensor([one(x)[0] for x in O_TOKENS], device=dev)
        prep = [one(x) for x in SETS["PREP"] if x != " by"]
        assert all(len(x) == 1 for x in prep)
        self.ids["prep_noby"] = t.tensor([x[0] for x in prep], device=dev)

    def run(self, seqs, d=None, tcoord=None, nset=None, ntgt=None, verify=None):
        """Forward token-id sequences (right padded). d [B, H] unit bases and tcoord [B]: set the site-8 coordinate;
        nset (a set name) and ntgt [B, n_cap]: set those neurons (columns of the capture) to ntgt.
        Returns readouts {name: [B]}, site-8 projections on d (pre-patch, if d given), captured activations [B, n_cap]."""
        t, r = self.t, self.r
        B, T = len(seqs), max(map(len, seqs))
        x = t.full((B, T), r.tok.eos_token_id, dtype=t.long)
        att = t.zeros((B, T), dtype=t.long)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = t.tensor(s)
            att[i, :len(s)] = 1
        x, att = x.to(r.device), att.to(r.device)
        anc = att.sum(1) - 1
        rows = t.arange(B, device=r.device)
        cap, hs = {}, []

        def site8(m, i, o):
            h = normalize_output(o)
            cap["proj"] = (h[rows, anc].float() * d).sum(-1)
            if tcoord is None:
                return None
            hn = h.clone()
            v = h[rows, anc].float()
            new = v + (tcoord - (v * d).sum(-1))[:, None] * d
            hn[rows, anc] = new.to(h.dtype)
            if verify is not None:
                verify["site8_target"] = max(verify.get("site8_target", 0.0),
                                             float(((new * d).sum(-1) - tcoord).abs().max()))
                mask = t.ones(hn.shape[:2], dtype=t.bool, device=hn.device)
                mask[rows, anc] = False
                verify["site8_outside"] = max(verify.get("site8_outside", 0.0), float((hn - h)[mask].abs().max()))
            return replace_output(o, hn)

        if d is not None:
            hs.append(self.layers[SITE - 1].register_forward_hook(site8))
        for l in LAYERS:
            def pre(m, i, l=l):
                a = i[0]
                cap[("neu", l)] = a[rows, anc][:, self.nidx[l]].float()
                if nset is None:
                    return None
                cols = self.set_cols[nset]
                sel = [c for c in cols if c in set(self.ncol[l])]
                if not sel:
                    return None
                pos = {c: j for j, c in enumerate(self.ncol[l])}
                idx = self.nidx[l][[pos[c] for c in sel]]
                tg = ntgt[:, sel]
                an = a.clone()
                an[rows[:, None], anc[:, None], idx[None]] = tg.to(a.dtype)
                if verify is not None:
                    verify["neuron_target"] = max(verify.get("neuron_target", 0.0),
                                                  float((an[rows[:, None], anc[:, None], idx[None]].float() - tg).abs().max()))
                    mask = t.ones_like(a, dtype=t.bool)
                    mask[rows[:, None], anc[:, None], idx[None]] = False
                    verify["neuron_outside"] = max(verify.get("neuron_outside", 0.0), float((an - a)[mask].abs().max()))
                return (an,) + tuple(i[1:])
            hs.append(self.layers[l].mlp.dense_4h_to_h.register_forward_pre_hook(pre))
        try:
            with t.no_grad():
                lg = r.model(input_ids=x, attention_mask=att, use_cache=False).logits
        finally:
            for h in hs:
                h.remove()
        lp = t.log_softmax(lg[rows, anc].float(), -1)
        out = {k: t.logsumexp(lp[:, self.ids[k]], -1) for k in READ}
        neu = t.zeros(B, len(self.cap_cols), device=r.device)
        for l in LAYERS:
            neu[:, self.ncol[l]] = cap[("neu", l)]
        return out, cap.get("proj"), neu


def run(args):
    import torch

    args.dtype = "float32"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runner = PassiveRunner(args)
    dev = runner.device
    args.d_mlp = runner.model.gpt_neox.layers[11].mlp.dense_4h_to_h.in_features
    rng = np.random.default_rng(args.seed)
    S = neuron_sets(args, np.random.default_rng(args.seed))
    P = Patcher(runner, S)
    mech = Mech(runner, args)
    w_by = {l: (mech.R[0] @ runner.model.gpt_neox.layers[l].mlp.dense_4h_to_h.weight.float()) for l in LAYERS}
    wcap = torch.zeros(len(P.cap_cols), device=dev)
    for l in LAYERS:
        wcap[torch.as_tensor(P.ncol[l], device=dev)] = w_by[l][P.nidx[l]]

    items = pd.read_csv(args.items)
    prim = items[items.bad_class == "plain"].reset_index(drop=True)
    plan = pd.read_csv(args.plan, usecols=["item_id", "split", "fold"]).drop_duplicates(["item_id", "split"])
    fold0 = plan[plan.split == 0].set_index("item_id").fold
    prim["fold"] = prim.item_id.map(fold0).astype(int)
    prim["das_pair"] = prim.das_pair.fillna("")
    pf = pd.read_csv(args.folds).set_index("pair_id")
    bases = load_bases(Path(args.site_dir) / "bases.pt", dev)
    D = torch.stack([bases[(0, f)] for f in range(5)])  # [5, H]
    assert (D.norm(dim=1) - 1).abs().max() < 1e-3
    # B7 items (same seed and selection as run_path_patch.py / run_path_patch_frames.py)
    keep = []
    for _, g in prim.groupby("pair_id"):
        ids = sorted(g.item_id)
        keep += list(rng.choice(ids, min(args.contexts, len(ids)), replace=False))
    keep = set(keep)

    ids = {side: build_ids(runner.tok, prim, side) for side in ("good", "bad")}
    n_cap = len(P.cap_cols)
    # 1. natural pass
    nat = {(f, s): {k: np.empty(len(prim)) for k in READ} for f in FRAMES for s in ("good", "bad")}
    proj = {(f, s): np.empty((len(prim), 5)) for f in FRAMES for s in ("good", "bad")}
    acts = {(f, s): np.empty((len(prim), n_cap), np.float32) for f in FRAMES for s in ("good", "bad")}
    for f in FRAMES:
        for s in ("good", "bad"):
            seqs = ids[s][f]
            for i in range(0, len(prim), args.batch):
                sl = slice(i, min(i + args.batch, len(prim)))
                rd, _, neu = P.run(seqs[sl])
                for k in READ:
                    nat[(f, s)][k][sl] = rd[k].double().cpu().numpy()
                acts[(f, s)][sl] = neu.cpu().numpy()
            # site-8 states for projections (one extra pass with hidden states)
            for i in range(0, len(prim), args.batch):
                sl = slice(i, min(i + args.batch, len(prim)))
                sq = seqs[sl]
                B, T = len(sq), max(map(len, sq))
                x = torch.full((B, T), runner.tok.eos_token_id, dtype=torch.long)
                att = torch.zeros((B, T), dtype=torch.long)
                for j, q in enumerate(sq):
                    x[j, :len(q)] = torch.tensor(q)
                    att[j, :len(q)] = 1
                x, att = x.to(dev), att.to(dev)
                with torch.no_grad():
                    o = runner.model(input_ids=x, attention_mask=att, output_hidden_states=True, use_cache=False)
                anc = att.sum(1) - 1
                h8 = o.hidden_states[SITE][torch.arange(B, device=dev), anc].float()
                proj[(f, s)][sl] = (h8 @ D.T).double().cpu().numpy()
    print(f"natural pass {time.time() - t0:.0f}s", flush=True)

    # 2. targets (leave own pair out; DAS training pairs of the item's basis excluded)
    tgt = {}
    for f in FRAMES:
        tI = np.full(len(prim), np.nan)
        tT = np.full(len(prim), np.nan)
        aI = np.full((len(prim), n_cap), np.nan)
        aT = np.full((len(prim), n_cap), np.nan)
        nmin = np.full(len(prim), np.nan)
        for fo in range(5):
            train = set(pf.index[pf["fold_split0"] != fo])
            elig = ~prim.das_pair.isin(train).to_numpy()
            key = prim.context_id.to_numpy()
            df = pd.DataFrame({"key": key, "n": elig.astype(float),
                               "g": proj[(f, "good")][:, fo] * elig, "b": proj[(f, "bad")][:, fo] * elig})
            sm = df.groupby("key")[["g", "b", "n"]].transform("sum")
            n = (sm.n - df.n).to_numpy()
            use = (prim.fold == fo).to_numpy()
            tT[use] = ((sm.g - df.g).to_numpy() / n)[use]
            tI[use] = ((sm.b - df.b).to_numpy() / n)[use]
            nmin[use] = n[use]
            for side, arr in (("good", aT), ("bad", aI)):
                A = acts[(f, side)].astype(np.float64) * elig[:, None]
                Ad = pd.DataFrame(A).groupby(key).transform("sum").to_numpy()
                arr[use] = ((Ad - A) / n[:, None])[use]
        assert not np.isnan(tI).any() and not np.isnan(aI).any()
        tgt[f] = {"tI": tI, "tT": tT, "aI": aI, "aT": aT, "nmin": nmin}
    meta = {"args": vars(args), "items": len(prim), "b7_items": len(keep), "neuron_sets": S.groupby("set").size().to_dict(),
            "min_target_pairs": {f: int(np.nanmin(tgt[f]["nmin"])) for f in FRAMES},
            "inputs": {k: sha(v) for k, v in (("items", args.items), ("plan", args.plan), ("switch", args.switch),
                                               ("neurons", args.neurons), ("bases", Path(args.site_dir) / "bases.pt"))},
            "checks": {}}

    # 3. interventions on the B7 good items
    conds = [("d_I", "tI", None, None), ("d_T", "tT", None, None), ("n10_I", None, "top10", "aI"),
             ("n50_I", None, "top50", "aI"), ("n50_T", None, "top50", "aT"), ("c10", "tI", "top10", "aI"),
             ("c50", "tI", "top50", "aI"), ("w50", None, "w50", "aI")] + \
            [(f"r10_{k}", None, f"r10_{k}", "aI") for k in range(1, 6)] + [(f"r50_{k}", None, f"r50_{k}", "aI") for k in range(1, 6)]
    sel = np.flatnonzero(prim.item_id.isin(keep).to_numpy())
    rows = []
    t1 = time.time()
    for f in FRAMES:
        T = tgt[f]
        for i in range(0, len(sel), args.batch):
            b = sel[i:i + args.batch]
            dB = D[torch.as_tensor(prim.fold.to_numpy()[b], device=dev)]
            seqs_g = [ids["good"][f][j] for j in b]
            rec = {"item_id": prim.item_id.to_numpy()[b], "frame": f}
            for cname, tc, ns, ak in conds:
                ver = {}
                tcoord = torch.as_tensor(T[tc][b], dtype=torch.float32, device=dev) if tc else None
                ntgt = torch.as_tensor(T[ak][b], dtype=torch.float32, device=dev) if ns else None
                rd, _, _ = P.run(seqs_g, d=dB if tc else None, tcoord=tcoord, nset=ns, ntgt=ntgt, verify=ver)
                for k in READ:
                    rec[f"{cname}_{k}"] = rd[k].double().cpu().numpy()
                for kk, v in ver.items():
                    meta["checks"][kk] = max(meta["checks"].get(kk, 0.0), v)
            # self patch: own natural coordinate and own natural top-50 activations
            own = torch.as_tensor(proj[(f, "good")][b, prim.fold.to_numpy()[b]], dtype=torch.float32, device=dev)
            owna = torch.as_tensor(acts[(f, "good")][b], dtype=torch.float32, device=dev)
            rd, _, _ = P.run(seqs_g, d=dB, tcoord=own, nset="top50", ntgt=owna)
            err = max(float(np.abs(rd[k].double().cpu().numpy() - nat[(f, "good")][k][b]).max()) for k in READ)
            meta["checks"]["self_patch_max_abs"] = max(meta["checks"].get("self_patch_max_abs", 0.0), err)
            # equalized coordinate: good and bad both at (tT + tI) / 2; capture top-50 activations
            tmid = torch.as_tensor((T["tT"][b] + T["tI"][b]) / 2, dtype=torch.float32, device=dev)
            for side in ("good", "bad"):
                rd, _, neu = P.run([ids[side][f][j] for j in b], d=dB, tcoord=tmid)
                for k in READ:
                    rec[f"eq_{side}_{k}"] = rd[k].double().cpu().numpy()
                rec[f"eq_{side}_wby50"] = (neu[:, torch.as_tensor(P.set_cols["top50"], device=dev)]
                                           * wcap[torch.as_tensor(P.set_cols["top50"], device=dev)]).sum(1).double().cpu().numpy()
            rows.append(pd.DataFrame(rec))
        print(f"frame {f} interventions done ({time.time() - t1:.0f}s)", flush=True)
    R = pd.concat(rows, ignore_index=True)
    assert meta["checks"]["self_patch_max_abs"] <= 1e-4, meta["checks"]
    assert meta["checks"].get("site8_target", 0) <= 1e-3 and meta["checks"].get("neuron_target", 0) <= 1e-5
    assert meta["checks"].get("site8_outside", 0) == 0 and meta["checks"].get("neuron_outside", 0) == 0
    # natural table (all primary items) with natural top-50 w_by activation sums and targets for the B7 items
    t50 = torch.as_tensor(P.set_cols["top50"])
    wc = wcap.cpu().numpy()[t50.numpy()]
    natural = []
    for f in FRAMES:
        for s in ("good", "bad"):
            natural.append(pd.DataFrame({"item_id": prim.item_id, "pair_id": prim.pair_id, "band": prim.band,
                                         "context_id": prim.context_id, "fold": prim.fold, "frame": f, "side": s,
                                         "b7": prim.item_id.isin(keep).astype(int), **{k: nat[(f, s)][k] for k in READ},
                                         "proj": proj[(f, s)][np.arange(len(prim)), prim.fold.to_numpy()],
                                         "tI": tgt[f]["tI"], "tT": tgt[f]["tT"],
                                         "wby50": acts[(f, s)][:, t50.numpy()] @ wc}))
    pd.concat(natural, ignore_index=True).to_parquet(out / "necessity_natural.parquet", index=False)
    R.to_parquet(out / "necessity_rows.parquet", index=False)
    S.to_csv(out / "necessity_neurons.csv", index=False)
    meta["seconds"] = round(time.time() - t0, 1)
    (out / "necessity_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps(meta["checks"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", default="data/das_round2/passive_test/items.csv")
    ap.add_argument("--plan", default="data/das_round2/passive_test/plan.csv.gz")
    ap.add_argument("--folds", default="results/das_round2/final_strict/pairs_folds.csv")
    ap.add_argument("--site-dir", default="results/das_round2/final_strict_site8")
    ap.add_argument("--switch", default="results/round4/translation/conjunction_switch_neurons.csv")
    ap.add_argument("--neurons", default="data/round4/oracle_translation/neurons.csv")
    ap.add_argument("--out-dir", default="results/round5/necessity")
    ap.add_argument("--contexts", type=int, default=32)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=256)
    run(ap.parse_args())
