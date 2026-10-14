#!/usr/bin/env python3
"""Round 5, A4: output-side restoration on single-token rare participles (reports/round5/plan.md, A4). fp32.

Edit (per edited participle V, token t): u_t <- u_t + (a* - a_V) dbar / |dbar|^2 (dbar = A2's mean was - has direction,
data/round5/restore/dbar.npy). Conditions: unedited, oracle (a* = Head class mean affinity), oracle_T, oracle_I (one
class only), labelfree (a* = alpha + beta z_V; z_V = site-8 coordinate on d in "She has V" / "He has V" / "They have V",
z per split-0 fold basis with its own sign and held-out active levels, mean over folds, DAS verbs on their held-out fold;
alpha, beta from OLS over the Head single-token verbs).
Evaluations per condition: curated band-cross passive_1 / passive_2 pairs (whole / verb / suffix LP, as
score_matched_passives.py); released FreqBLiMP, 67 paradigms x 3 bands (whole-sentence LP; items containing an edited
token flagged); LM loss on Pile-validation windows containing the edited verbs and on 500 Pile-10k documents (truncated
to 512 tokens), with NLL at the edited tokens split by the preceding word.
Checks: the edited rows' new affinity equals a*; unedited rows unchanged; the unedited condition reproduces the
curated fp32 scores (results/round5/frame/curated_p1_fp32_scores.csv) for passive_1.

Outputs (--out-dir): restore_curated.parquet, restore_freqblimp.parquet, restore_lm.parquet, restore_zfit.csv,
restore_meta.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from score_matched_passives import score_batch

KEYS = [(0, f) for f in range(5)]
BE = {"was", "were", "been", "is", "are", "be", "being", "am"}
PERF = {"has", "have", "had", "having"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sentence_lp(model, tok, dev, sents, batch):
    """Whole-sentence log-prob (all tokens but the first, no BOS) and token ids, as the paper scorer."""
    import torch

    out_lp, out_ids = [], []
    for i in range(0, len(sents), batch):
        enc = tok(sents[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.inference_mode():
            lg = model(**enc, use_cache=False).logits[:, :-1].float()
        lp = torch.log_softmax(lg, -1).gather(-1, enc.input_ids[:, 1:, None])[..., 0] * enc.attention_mask[:, 1:]
        out_lp.append(lp.sum(1).double().cpu().numpy())
        m = enc.attention_mask.sum(1).cpu().numpy()
        out_ids += [enc.input_ids[j, :m[j]].cpu().numpy() for j in range(len(m))]
    return np.concatenate(out_lp), out_ids


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    dev = torch.device(args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, local_files_only=True).to(dev).eval()
    W = model.embed_out.weight
    W0 = W.detach().clone()
    dbar = torch.tensor(np.load(args.dbar), dtype=torch.float32, device=dev)
    man = pd.read_csv(args.manifest)
    head = pd.read_csv(args.head)
    meta = {"args": vars(args), "checks": {}, "inputs": {k: sha(v) for k, v in
                                                         (("manifest", args.manifest), ("head", args.head), ("dbar", args.dbar))}}
    # affinities recomputed from the live weights must match A2's
    for df in (man, head):
        a = (W0[torch.as_tensor(df.first_id.to_numpy(), device=dev)] @ dbar).double().cpu().numpy()
        meta["checks"]["affinity_reproduces_max"] = max(meta["checks"].get("affinity_reproduces_max", 0.0),
                                                         float(np.abs(a - df.a.to_numpy()).max()))
    assert meta["checks"]["affinity_reproduces_max"] < 1e-3, meta["checks"]

    # ---- z_V for the label-free target: site-8 coordinate in active frames
    npz = np.load(args.bases)
    D = torch.stack([torch.tensor(npz[f"das_site8_s0_f{f}"].reshape(-1), dtype=torch.float32, device=dev) for _, f in KEYS])
    das = pd.read_csv(args.das_items)
    pf = pd.read_csv(args.folds)
    fold0 = dict(zip(pf.pair_id, pf.fold_split0))

    def site8(prompts):
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        with torch.inference_mode():
            o = model(**enc, output_hidden_states=True, use_cache=False)
        anc = enc.attention_mask.sum(1) - 1
        return o.hidden_states[8][torch.arange(len(prompts), device=dev), anc].float()

    xs = torch.cat([site8(das.prompt.tolist()[i:i + 256]) for i in range(0, len(das), 256)]) @ D.T  # [n, 5]
    xs = xs.double().cpu().numpy()
    scale = {}
    fo = das.pair_id.map(fold0).to_numpy()
    c = das.cls.to_numpy()
    for j, (_, f) in enumerate(KEYS):
        tr = (fo != f) & (das.subject != "David").to_numpy()
        te = fo == f
        sgn = np.sign(xs[tr & (c == 1), j].mean() - xs[tr & (c == 0), j].mean())
        mt, mi = (sgn * xs[te & (c == 1), j]).mean(), (sgn * xs[te & (c == 0), j]).mean()
        scale[f] = (sgn, mi, mt)
    verb_fold = {}  # participle -> the held-out split-0 fold of its DAS pair
    for col in ("trans_part", "intrans_part"):
        for v, f in zip(pf[col], pf.fold_split0):
            assert verb_fold.get(v, f) == f, f"inconsistent fold for {v}"
            verb_fold[v] = int(f)
    allv = pd.concat([head.assign(group="head"), man.assign(group="rare")], ignore_index=True)
    meta["das_participles_held_out_fold"] = sorted(v for v in allv.verb if v in verb_fold)
    frames = ("She has {}", "He has {}", "They have {}")
    Z = np.zeros(len(allv))
    for fr in frames:
        h8 = torch.cat([site8([fr.format(v) for v in allv.verb.tolist()[i:i + 256]]) for i in range(0, len(allv), 256)])
        x = (h8 @ D.T).double().cpu().numpy()
        z = np.stack([(scale[f][0] * x[:, j] - scale[f][1]) / (scale[f][2] - scale[f][1]) for j, (_, f) in enumerate(KEYS)], 1)
        zz = np.where([v in verb_fold for v in allv.verb], z[np.arange(len(allv)), [verb_fold.get(v, 0) for v in allv.verb]],
                      z.mean(1))
        Z += zz / len(frames)
    allv["z"] = Z
    H_ = allv[allv.group == "head"]
    X = np.column_stack([np.ones(len(H_)), H_.z])
    beta = np.linalg.lstsq(X, H_.a.to_numpy(), rcond=None)[0]
    rng = np.random.default_rng(args.seed)
    fid = rng.permutation(len(H_)) % 10

    def cv_r2(zv):
        err = err0 = 0.0
        y = H_.a.to_numpy()
        for k in range(10):
            te, tr = fid == k, fid != k
            b = np.linalg.lstsq(np.column_stack([np.ones(tr.sum()), zv[tr]]), y[tr], rcond=None)[0]
            err += ((y[te] - b[0] - b[1] * zv[te]) ** 2).sum()
            err0 += ((y[te] - y[tr].mean()) ** 2).sum()
        return 1 - err / err0

    r2 = cv_r2(H_.z.to_numpy())
    shuf = [cv_r2(rng.permutation(H_.z.to_numpy())) for _ in range(200)]
    res = H_.a.to_numpy() - X @ beta
    allv["a_labelfree"] = beta[0] + beta[1] * allv.z
    allv.to_csv(out / "restore_zfit.csv", index=False)
    meta["zfit"] = {"alpha": float(beta[0]), "beta": float(beta[1]), "cv_r2_head": float(r2),
                    "cv_r2_shuffled_95": float(np.percentile(shuf, 95)),
                    "resid_mean_T": float(res[H_.role == "T"].mean()), "resid_mean_I": float(res[H_.role == "I"].mean()),
                    "resid_zipf_slope": float(np.polyfit(H_.zipf, res, 1)[0]),
                    "z_range_head": [float(H_.z.min()), float(H_.z.max())],
                    "z_range_rare": [float(allv[allv.group == "rare"].z.min()), float(allv[allv.group == "rare"].z.max())]}
    print(json.dumps(meta["zfit"]), flush=True)
    rare = allv[allv.group == "rare"].reset_index(drop=True)
    targets = {"unedited": {},
               "oracle": dict(zip(rare.first_id, rare.a_oracle)),
               "oracle_T": dict(zip(rare[rare.role == "T"].first_id, rare[rare.role == "T"].a_oracle)),
               "oracle_I": dict(zip(rare[rare.role == "I"].first_id, rare[rare.role == "I"].a_oracle)),
               "labelfree": dict(zip(rare.first_id, rare.a_labelfree))}
    edited_ids = set(int(t) for t in rare.first_id)
    nd2 = float(dbar @ dbar)

    def apply(cond):
        with torch.no_grad():
            W.copy_(W0)
            for t, a_star in targets[cond].items():
                cur = float(W0[int(t)] @ dbar)
                W[int(t)] += (a_star - cur) * dbar / nd2
            if targets[cond]:
                ids = torch.as_tensor(list(targets[cond]), device=dev)
                got = (W[ids] @ dbar).double().cpu().numpy()
                want = np.array(list(targets[cond].values()))
                meta["checks"][f"edit_target_err_{cond}"] = float(np.abs(got - want).max())
                mask = torch.ones(W.shape[0], dtype=torch.bool, device=dev)
                mask[ids] = False
                meta["checks"][f"unedited_rows_max_{cond}"] = float((W[mask] - W0[mask]).abs().max())

    # ---- data
    bca = pd.read_json(args.pairs, lines=True)
    cur = bca[bca.context_set == "curated"].reset_index(drop=True)
    fq = []
    for band in ("head", "tail", "xtail"):
        for f in sorted((Path(args.freqblimp) / band).glob("*.jsonl")):
            d = pd.read_json(f, lines=True)
            fq.append(pd.DataFrame({"band": band, "uid": f.stem, "pairID": d.pairID.astype(str),
                                    "good": d.sentence_good, "bad": d.sentence_bad}))
    fq = pd.concat(fq, ignore_index=True)
    docs = [json.loads(l) for l in Path(args.pileval_docs).read_text().splitlines()]
    p10 = pd.read_parquet(args.pile10k).text
    pick = np.sort(np.random.default_rng(args.seed).choice(len(p10), 500, replace=False))
    lm_texts = [("pileval", d["verb"], d["doc_ix"], d["text"], d["occ_char"]) for d in docs] + \
               [("pile10k", "", int(i), p10.iloc[i], -1) for i in pick]

    def window(text, occ):
        """Pythia token ids and char offsets: the first lm_len tokens, or lm_len tokens centred on the occurrence."""
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc.input_ids, enc.offset_mapping
        if occ < 0 or len(ids) <= args.lm_len:
            st = 0
        else:
            j = next((k for k, (a, b) in enumerate(offs) if b > occ), len(ids) - 1)
            st = max(0, min(j - args.lm_len // 2, len(ids) - args.lm_len))
        return ids[st:st + args.lm_len], offs[st:st + args.lm_len]

    lm_win = [window(t, occ) for _, _, _, t, occ in lm_texts]
    lm_ids = [w[0] for w in lm_win]
    # the selected occurrence must be a scored edited token
    tid = dict(zip(man.verb, man.first_id))
    sel_pos = []
    for (src, verb, _, text, occ), (ids, offs) in zip(lm_texts, lm_win):
        if src != "pileval":
            sel_pos.append(-1)
            continue
        js = [k for k, (a, b) in enumerate(offs) if a <= occ < b or (a == occ - 1 and text[a] == " ")]
        js = [k for k in js if k >= 1 and ids[k] == int(tid[verb])]
        sel_pos.append(js[0] if js else -2)
    meta["pileval_selected_occurrence_missing"] = int(sum(p == -2 for p in sel_pos))
    meta["n_pileval_windows"] = len(docs)
    print(f"setup {time.time() - t0:.0f}s; curated {len(cur)}, FreqBLiMP {len(fq)}, LM texts {len(lm_texts)}", flush=True)

    rows_c, rows_f, rows_l = [], [], []
    for cond in targets:
        apply(cond)
        t1 = time.time()
        # curated passive pairs
        for i in range(0, len(cur), args.batch):
            g = cur.iloc[i:i + args.batch]
            for side in ("good", "bad"):
                sc = score_batch(model, tok, dev, [{"sentence": r[f"sentence_{side}"], "prefix": r.prefix,
                                                    "verb": r[f"{side}_verb"], "paradigm": r.paradigm} for r in g.itertuples()])
                for r, x in zip(g.itertuples(), sc):
                    rows_c.append({"cond": cond, "pair_id": r.pair_id, "side": side, "whole": x["whole_lp"],
                                   "verb": x["verb_lp"], "suffix": x["suffix_lp"], "by": x["by_lp"]})
        # FreqBLiMP
        lg, ig = sentence_lp(model, tok, dev, fq.good.tolist(), args.batch)
        lb, ib = sentence_lp(model, tok, dev, fq.bad.tolist(), args.batch)
        aff = np.array([bool(edited_ids & (set(a.tolist()) | set(b.tolist()))) for a, b in zip(ig, ib)])
        rows_f.append(pd.DataFrame({"cond": cond, "row": np.arange(len(fq)), "good_lp": lg, "bad_lp": lb, "affected": aff}))
        # LM loss
        for i in range(0, len(lm_ids), 16):
            chunk = lm_ids[i:i + 16]
            T = max(len(x) for x in chunk)
            x = torch.full((len(chunk), T), tok.eos_token_id, dtype=torch.long)
            att = torch.zeros((len(chunk), T), dtype=torch.long)
            for j, q in enumerate(chunk):
                x[j, :len(q)] = torch.tensor(q)
                att[j, :len(q)] = 1
            x, att = x.to(dev), att.to(dev)
            with torch.inference_mode():
                lgt = model(input_ids=x, attention_mask=att, use_cache=False).logits[:, :-1].float()
            nll = -torch.log_softmax(lgt, -1).gather(-1, x[:, 1:, None])[..., 0]
            for j, q in enumerate(chunk):
                n = len(q) - 1
                v = nll[j, :n].double().cpu().numpy()
                src, verb, dix, text, _ = lm_texts[i + j]
                offs = lm_win[i + j][1]
                tgt_pos = [p for p in range(1, len(q)) if q[p] in edited_ids]
                ctxw = []
                for p in tgt_pos:  # the full preceding surface word (character offsets), not a BPE piece
                    m = re.search(r"([A-Za-z]+)[^A-Za-z]*$", text[:offs[p][0]])
                    prev = m.group(1).lower() if m else ""
                    ctxw.append("be" if prev in BE else "perf" if prev in PERF else "other")
                rows_l.append({"cond": cond, "src": src, "verb": verb, "doc_ix": dix, "n_tok": n, "nll_sum": float(v.sum()),
                               "edited_nll": [float(v[p - 1]) for p in tgt_pos], "edited_ctx": ctxw,
                               "edited_tok": [int(q[p]) for p in tgt_pos], "selected_pos": sel_pos[i + j]})
        print(f"{cond}: {time.time() - t1:.0f}s", flush=True)
    with torch.no_grad():
        W.copy_(W0)
    C = pd.DataFrame(rows_c)
    # check: unedited curated passive_1 reproduces the fp32 scores
    ref = pd.read_csv(args.cur_ref).set_index("pair_id")
    u = C[(C.cond == "unedited")].pivot_table(index="pair_id", columns="side", values="whole")
    u = u.loc[u.index.intersection(ref.index)]
    meta["checks"]["unedited_reproduces_fp32_max"] = float(np.abs((u.good - u.bad) - ref.loc[u.index].whole_margin).max())
    C.to_parquet(out / "restore_curated.parquet", index=False)
    F = pd.concat(rows_f, ignore_index=True)
    F.to_parquet(out / "restore_freqblimp.parquet", index=False)
    fq.drop(columns=["good", "bad"]).assign(row=np.arange(len(fq))).to_parquet(out / "restore_freqblimp_items.parquet", index=False)
    pd.DataFrame(rows_l).to_parquet(out / "restore_lm.parquet", index=False)
    meta["seconds"] = round(time.time() - t0)
    (out / "restore_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps(meta["checks"]), flush=True)
    assert meta["checks"]["unedited_reproduces_fp32_max"] < 1e-3
    assert all(v < 1e-3 for k, v in meta["checks"].items() if k.startswith("edit_target_err"))
    assert all(v == 0 for k, v in meta["checks"].items() if k.startswith("unedited_rows_max"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/round5/restore/manifest.csv")
    ap.add_argument("--head", default="data/round5/restore/head_singletons.csv")
    ap.add_argument("--dbar", default="data/round5/restore/dbar.npy")
    ap.add_argument("--pileval-docs", default="data/round5/restore/pileval_docs.jsonl")
    ap.add_argument("--pile10k", required=True)
    ap.add_argument("--pairs", default="data/passive_band_cross/pairs.jsonl")
    ap.add_argument("--cur-ref", default="results/round5/frame/curated_p1_fp32_scores.csv")
    ap.add_argument("--freqblimp", required=True)
    ap.add_argument("--bases", default="results/das_round2/bases_rank1.npz")
    ap.add_argument("--das-items", default="results/das_round2/final_strict/items.csv")
    ap.add_argument("--folds", default="results/das_round2/final_strict/pairs_folds.csv")
    ap.add_argument("--out-dir", default="results/round5/restore")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lm-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
