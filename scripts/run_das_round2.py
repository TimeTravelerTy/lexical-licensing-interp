#!/usr/bin/env python3
"""DAS round 2: a rank-k "object next" subspace learned on perfect-frame actives.

Items: "<Subj> has/have <participle>" for both verbs of each training pair and
7 subjects (`build_das_round2_prompts.SUBJECTS`); the verb's last subtoken is
the last prompt token, so it is both the intervention and readout position.
Training uses 6 subjects; "David" is held out as a template check.

Interchange at site s (0 = embeddings, i = output of layer i-1), last token:
    h <- h + ((h_src - h) B) B^T,   B = orthonormal basis (d x k)
Target: M = logsumexp(logits[O]) - logsumexp(logits[I]); loss = BCE(sigmoid(M),
class of the source verb). Cross-class swaps (both directions) and same-class
swaps; base and source share the subject.

Verbs whose unpatched behaviour does not match their class (fewer than 4 of 7
items on the right side of 0) are dropped with their pair before any split.

Modes
- sweep: rank 1, every site, split seed 0, 5 folds; held-out metrics per epoch.
- final: frozen site and epochs (from `--config`), ranks 1/2/4 on 3 split
  seeds x 5 folds, rank chosen by the prespecified rule, then random-subspace
  and shuffled-label controls for the chosen rank. Saves bases.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from build_das_round2_prompts import SUBJECTS

TRAIN_SUBJECTS = [s for s, _ in SUBJECTS if s != "David"]
DET = (" the", " a", " an", " his", " her", " their", " its", " my", " our", " your", " this", " these",
       " those", " some", " every", " each", " several")
PRON = (" him", " them", " it", " me", " us", " you")
REFL = (" himself", " herself", " themselves", " itself")
O_TOKENS = DET + PRON + REFL
I_TOKENS = (".", ",", "\n", " and", " to", " in", " with", " on", " at", " for", " from", " as", " into",
            " over", " about", " of", " upon", " against", " through", " after", " under", " around",
            " across", " toward", " here", " there", " again", " already", " never", " not", " so", " just",
            " well", " but", " or", " because", " when", " while", " since", " before", " until", " like",
            " without", " during", " behind", " near", " onto", " together", " alone", " too", "!", "?", ";",
            ":", '."', ',"')


def normalize_output(output):
    return output[0] if isinstance(output, tuple) else output


def replace_output(output, hidden):
    return (hidden,) + tuple(output[1:]) if isinstance(output, tuple) else hidden


class Runner:
    def __init__(self, args):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device(args.device)
        self.tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        self.model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype),
                                                          local_files_only=True).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.ids = {}
        for name, toks in (("O", O_TOKENS), ("I", I_TOKENS), ("det", DET), ("pron", PRON), ("refl", REFL)):
            ids = []
            for t in toks:
                enc = self.tok.encode(t, add_special_tokens=False)
                if len(enc) != 1:
                    raise SystemExit(f"token {t!r} is not a single token")
                ids.append(enc[0])
            self.ids[name] = torch.tensor(ids, device=self.device)
        self.n_sites = self.model.config.num_hidden_layers + 1

    def encode(self, prompts):
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        anchors = enc.attention_mask.sum(1) - 1
        return enc, anchors

    def readout(self, logits, anchors, full=False):
        torch = self.torch
        rows = torch.arange(logits.shape[0], device=logits.device)
        z = logits[rows, anchors].float()
        m = torch.logsumexp(z[:, self.ids["O"]], -1) - torch.logsumexp(z[:, self.ids["I"]], -1)
        if not full:
            return m, None
        lp = torch.log_softmax(z, -1)
        parts = {k: torch.logsumexp(lp[:, self.ids[k]], -1) for k in ("O", "I", "det", "pron", "refl")}
        return m, parts

    def natural(self, prompts, batch=256):
        """Unpatched M, set log-probs, and last-token states at every site."""
        torch = self.torch
        ms, parts, reps = [], {k: [] for k in ("O", "I", "det", "pron", "refl")}, []
        with torch.no_grad():
            for i in range(0, len(prompts), batch):
                enc, anchors = self.encode(prompts[i:i + batch])
                out = self.model(**enc, output_hidden_states=True, use_cache=False)
                m, p = self.readout(out.logits, anchors, full=True)
                ms.append(m)
                for k in parts:
                    parts[k].append(p[k])
                rows = torch.arange(len(anchors), device=self.device)
                reps.append(torch.stack([h[rows, anchors].float() for h in out.hidden_states], 1))
        return torch.cat(ms), {k: torch.cat(v) for k, v in parts.items()}, torch.cat(reps)

    def patched(self, prompts, src_h, site, basis, full=False, match_norm=None):
        """Forward the base prompts with the last-token state at `site` interchanged along `basis`.

        match_norm: optional reference basis; the displacement is rescaled to the norm of the
        displacement the reference basis would produce (norm-matched random control).
        """
        torch = self.torch
        enc, anchors = self.encode(prompts)
        rows = torch.arange(len(anchors), device=self.device)

        def patch(hidden):
            h = hidden.float()
            base = h[rows, anchors]
            delta = ((src_h - base) @ basis) @ basis.T
            if match_norm is not None:
                ref = ((src_h - base) @ match_norm) @ match_norm.T
                delta = delta * (ref.norm(dim=-1, keepdim=True) / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6))
            h = h.clone()
            h[rows, anchors] = base + delta
            return h.to(hidden.dtype)

        if site == 0:
            handle = self.model.gpt_neox.embed_in.register_forward_hook(lambda m, i, o: patch(o))
        else:
            handle = self.model.gpt_neox.layers[site - 1].register_forward_hook(
                lambda m, i, o: replace_output(o, patch(normalize_output(o))))
        try:
            out = self.model(**enc, use_cache=False)
        finally:
            handle.remove()
        return self.readout(out.logits, anchors, full=full)


def load_items(pairs_csv):
    pairs = pd.read_csv(pairs_csv)
    pairs["pair_id"] = pairs.trans + "/" + pairs.intrans
    items = []
    for p in pairs.itertuples():
        for side, cls in (("trans", 1), ("intrans", 0)):
            verb, part = getattr(p, side), getattr(p, f"{side}_part")
            for subj, aux in SUBJECTS:
                items.append({"verb": verb, "cls": cls, "pair_id": p.pair_id, "source": p.source, "tok": p.tok,
                              "subject": subj, "prompt": f"{subj} {aux} {part}"})
    return pairs, pd.DataFrame(items)


def folds(pairs, n_folds, seed):
    """Stratified fold assignment: pairs shuffled within (source, token count), then dealt round-robin."""
    rng = random.Random(seed)
    order = []
    for _, g in sorted(pairs.groupby(["source", "tok"]), key=lambda kv: str(kv[0])):
        ids = sorted(g.pair_id)
        rng.shuffle(ids)
        order += ids
    return {pid: i % n_folds for i, pid in enumerate(order)}


def make_swaps(items, idx, labels, rng, n_cross, n_same):
    """Training swaps: for each base item, sources with the same subject and a different verb."""
    by = {}
    for i in idx:
        by.setdefault((items.subject[i], labels[i]), []).append(i)
    swaps = []
    for b in idx:
        s, c = items.subject[b], labels[b]
        cross = by.get((s, 1 - c), [])
        same = [j for j in by.get((s, c), []) if items.verb[j] != items.verb[b]]
        for j in rng.sample(cross, min(n_cross, len(cross))):
            swaps.append((b, j, labels[j]))
        for j in rng.sample(same, min(n_same, len(same))):
            swaps.append((b, j, labels[j]))
    rng.shuffle(swaps)
    return swaps


def eval_swaps(items, idx):
    """All held-out swaps: same subject, different verb."""
    out = []
    for b in idx:
        for j in idx:
            if items.subject[b] == items.subject[j] and items.verb[b] != items.verb[j]:
                out.append((b, j))
    return out


def train(runner, items, reps, idx, labels, site, rank, epochs, args, seed, eval_fn=None):
    torch = runner.torch
    torch.manual_seed(seed)
    rng = random.Random(seed)
    d = reps.shape[-1]
    raw = torch.nn.Parameter(torch.randn(d, rank, device=runner.device) * 0.02)
    opt = torch.optim.Adam([raw], lr=args.lr)
    history = []
    prompts = items.prompt.tolist()
    for epoch in range(1, epochs + 1):
        swaps = make_swaps(items, idx, labels, rng, args.n_cross, args.n_same)
        for i in range(0, len(swaps), args.batch_size):
            chunk = swaps[i:i + args.batch_size]
            b = [x[0] for x in chunk]
            src = reps[[x[1] for x in chunk], site]
            y = torch.tensor([float(x[2]) for x in chunk], device=runner.device)
            basis, _ = torch.linalg.qr(raw)
            m, _ = runner.patched([prompts[k] for k in b], src, site, basis)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(m, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if eval_fn is not None:
            basis = torch.linalg.qr(raw.detach())[0]
            history.append({"epoch": epoch, **eval_fn(basis)})
    return torch.linalg.qr(raw.detach())[0], history


def evaluate(runner, items, reps, nat, idx, site, basis, batch=512, match_norm=None, detail=False):
    torch = runner.torch
    swaps = eval_swaps(items, idx)
    prompts = items.prompt.tolist()
    rows = []
    with torch.no_grad():
        for i in range(0, len(swaps), batch):
            chunk = swaps[i:i + batch]
            b = [x[0] for x in chunk]
            j = [x[1] for x in chunk]
            m, parts = runner.patched([prompts[k] for k in b], reps[j, site], site, basis, full=detail,
                                      match_norm=match_norm)
            m = m.cpu().numpy()
            for t, (bb, jj) in enumerate(chunk):
                r = {"base": int(bb), "src": int(jj), "M_patched": float(m[t])}
                if detail:
                    for k in ("O", "I", "det", "pron", "refl"):
                        r[f"lp_{k}_patched"] = float(parts[k][t])
                rows.append(r)
    df = pd.DataFrame(rows)
    nm = nat["M"]
    df["base_cls"] = items.cls.values[df.base]
    df["src_cls"] = items.cls.values[df.src]
    df["M_base"] = nm[df.base]
    df["M_src"] = nm[df.src]
    df["cross"] = df.base_cls != df.src_cls
    df["iia"] = (df.M_patched > 0) == (df.src_cls == 1)
    df["frac"] = (df.M_patched - df.M_base) / (df.M_src - df.M_base)
    return df


def summarize(df, items):
    cross = df[df.cross]
    pc = cross[cross.base_cls == 0]
    held = items.subject.values[df.base] == "David"
    return {"iia_cross": float(cross.iia.mean()), "iia_same": float(df[~df.cross].iia.mean()),
            "pc_frac_median": float(pc.frac.median()), "pc_frac_mean": float(pc.frac.clip(-2, 3).mean()),
            "rev_frac_median": float(cross[cross.base_cls == 1].frac.median()),
            "iia_cross_david": float(cross[held[df.cross.values]].iia.mean()), "n_swaps": int(len(df))}


def setup(args):
    import torch

    runner = Runner(args)
    pairs, items = load_items(args.pairs)
    m, parts, reps = runner.natural(items.prompt.tolist())
    items["M_nat"] = m.cpu().numpy()
    items["correct"] = np.where(items.cls == 1, items.M_nat > 0, items.M_nat < 0)
    vp = items.groupby("verb").correct.sum() >= 4
    keep = pairs[pairs.trans.map(vp) & pairs.intrans.map(vp)].pair_id
    dropped = sorted(set(pairs.pair_id) - set(keep))
    sel = items.pair_id.isin(set(keep)).to_numpy()
    items = items[sel].reset_index(drop=True)
    reps = reps[torch.tensor(sel, device=reps.device)]
    nat = {"M": items.M_nat.to_numpy()}
    for k, v in parts.items():
        nat[k] = v.cpu().numpy()[sel]
    pairs = pairs[pairs.pair_id.isin(set(keep))].reset_index(drop=True)
    return runner, pairs, items, reps, nat, dropped


def fold_indices(items, assign, f):
    test = items.pair_id.map(assign).to_numpy() == f
    tr = np.flatnonzero(~test & items.subject.isin(TRAIN_SUBJECTS).to_numpy())
    return tr, np.flatnonzero(test)


def run_sweep(args):
    runner, pairs, items, reps, nat, dropped = setup(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    assign = folds(pairs, args.n_folds, 0)
    sites = range(runner.n_sites) if not args.sites else [int(s) for s in args.sites.split(",")]
    rows = []
    for site in sites:
        t0 = time.time()
        for f in range(args.n_folds):
            tr, te = fold_indices(items, assign, f)
            labels = items.cls.to_numpy()
            fn = lambda basis: summarize(evaluate(runner, items, reps, nat, te, site, basis), items)
            _, hist = train(runner, items, reps, tr, labels, site, 1, args.epochs, args, seed=1000 * site + f,
                            eval_fn=fn)
            for h in hist:
                rows.append({"site": site, "split": 0, "fold": f, **h})
        pd.DataFrame(rows).to_csv(out / "sweep.csv", index=False)
        print(f"site {site} done in {time.time() - t0:.0f}s", flush=True)
    meta = {"pairs": len(pairs), "dropped_pairs": dropped, "items": len(items), "mode": "sweep",
            "args": vars(args)}
    (out / "sweep_meta.json").write_text(json.dumps(meta, indent=2))


def run_final(args):
    import torch

    cfg = json.loads(Path(args.config).read_text())
    site, epochs = cfg["site"], cfg["epochs"]
    ranks = [cfg["rank"]] if cfg.get("rank") else [1, 2, 4]
    runner, pairs, items, reps, nat, dropped = setup(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = items.cls.to_numpy()
    summ, bases, details = [], {}, []
    for rank in ranks:
        for split in range(args.n_splits):
            assign = folds(pairs, args.n_folds, split)
            for f in range(args.n_folds):
                tr, te = fold_indices(items, assign, f)
                basis, _ = train(runner, items, reps, tr, labels, site, rank, epochs, args,
                                 seed=10000 * rank + 100 * split + f)
                df = evaluate(runner, items, reps, nat, te, site, basis, detail=True)
                df["rank"], df["split"], df["fold"], df["control"] = rank, split, f, "das"
                details.append(df)
                summ.append({"rank": rank, "split": split, "fold": f, "control": "das", **summarize(df, items)})
                bases[(rank, split, f)] = basis.cpu()
                print(f"rank {rank} split {split} fold {f}: {summ[-1]}", flush=True)
    s = pd.DataFrame(summ)
    by_rank = s.groupby("rank").pc_frac_median.mean()
    chosen = cfg.get("rank") or (int(by_rank.idxmax()) if by_rank.max() - by_rank.get(1, -np.inf) > 0.1 else 1)
    # Controls for the chosen rank.
    gen = torch.Generator(device="cpu").manual_seed(7)
    for split in range(args.n_splits):
        assign = folds(pairs, args.n_folds, split)
        for f in range(args.n_folds):
            tr, te = fold_indices(items, assign, f)
            basis = bases[(chosen, split, f)].to(runner.device)
            for r in range(args.n_random):
                rnd = torch.linalg.qr(torch.randn(basis.shape, generator=gen))[0].to(runner.device)
                for name, mn in (("random", None), ("random_normmatched", basis)):
                    df = evaluate(runner, items, reps, nat, te, site, rnd, match_norm=mn)
                    summ.append({"rank": chosen, "split": split, "fold": f, "control": name, "draw": r,
                                 **summarize(df, items)})
            vcls = items.groupby("verb").cls.first()
            tv = sorted(set(items.verb.values[tr]))
            shuffled = dict(zip(tv, np.random.default_rng(100 * split + f).permutation(vcls[tv].to_numpy())))
            perm = np.array([shuffled.get(v, c) for v, c in zip(items.verb, labels)])
            sb, _ = train(runner, items, reps, tr, perm, site, chosen, epochs, args, seed=777 + 100 * split + f)
            df = evaluate(runner, items, reps, nat, te, site, sb)
            summ.append({"rank": chosen, "split": split, "fold": f, "control": "shuffled_labels",
                         **summarize(df, items)})
            print(f"controls split {split} fold {f} done", flush=True)
    pd.DataFrame(summ).to_csv(out / "summary.csv", index=False)
    det = pd.concat(details, ignore_index=True)
    for col in ("verb", "pair_id", "subject", "source", "tok"):
        det[f"base_{col}"] = items[col].values[det.base]
        det[f"src_{col}"] = items[col].values[det.src]
    for k in ("O", "I", "det", "pron", "refl"):
        det[f"lp_{k}_base"] = nat[k][det.base]
        det[f"lp_{k}_src"] = nat[k][det.src]
    det.to_csv(out / "heldout_swaps.csv.gz", index=False)
    torch.save({f"r{k[0]}_s{k[1]}_f{k[2]}": v for k, v in bases.items()}, out / "bases.pt")
    items.drop(columns=["prompt"]).assign(prompt=items.prompt).to_csv(out / "items.csv", index=False)
    assigns = {split: folds(pairs, args.n_folds, split) for split in range(args.n_splits)}
    pairs.assign(**{f"fold_split{sp}": pairs.pair_id.map(a) for sp, a in assigns.items()}) \
        .to_csv(out / "pairs_folds.csv", index=False)
    meta = {"site": site, "epochs": epochs, "ranks_trained": ranks, "chosen_rank": int(chosen),
            "rank_rule": "rank 1 unless a larger rank raises mean held-out median pc_frac by > 0.1",
            "pc_frac_by_rank": by_rank.to_dict(), "pairs": len(pairs), "dropped_pairs": dropped,
            "items": len(items), "config": cfg, "args": vars(args)}
    (out / "final_meta.json").write_text(json.dumps(meta, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("sweep", "final"))
    ap.add_argument("--pairs", default="data/das_round2/train_pairs.csv")
    ap.add_argument("--config", default="results/das_round2/frozen_config.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--sites", default="")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--n-random", type=int, default=100)
    ap.add_argument("--n-cross", type=int, default=4)
    ap.add_argument("--n-same", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-2)
    args = ap.parse_args()
    run_sweep(args) if args.mode == "sweep" else run_final(args)
