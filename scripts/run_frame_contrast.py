#!/usr/bin/env python3
"""Round 5, A1-A2: frame-contrast scores (reports/round5/plan.md, A1-A2). fp32, TF32 off.

Inputs from `build_frame_contrast.py` (--data-dir): prefixes.csv, verbs.csv, pairs.csv.
For every prefix (set x context) and frame (pass = "was"/"were", act = "has"/"have") and verb, the log-prob of
each participle subtoken given the prefix and the earlier subtokens (no BOS, as score_matched_passives.py).
freq and they prefixes: every verb (curated + pool); cur prefixes: the curated verbs only (every verb with --cur-all,
round 6).
Also, per prefix and frame, the final-LN output h at the prefix's last token and the log-normalizer there; and
the unembedding rows of the verbs' first subtokens.

Checks (fail-fast):
- tokenization: prefix ids + participle ids (from verbs.csv) equal the joined string's ids;
- h is identical across the rows of a batch (shared prefix) to 1e-4;
- h . u_t - LSE reproduces every verb's first-subtoken log-prob to 1e-3 nats;
- the A2 identity L1 = (h_was - h_has) . (u_T1 - u_I1) on every curated pair and prefix to 1e-3 nats.

Output (--out-dir): frame_scores.parquet (one row per set x context x frame x verb), frame_h.npz, frame_meta.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

FRAMES = ("pass", "act")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(args.device)
    data = Path(args.data_dir)
    pre = pd.read_csv(data / "prefixes.csv", keep_default_na=False)
    V = pd.read_csv(data / "verbs.csv", keep_default_na=False)
    pairs = pd.read_csv(data / "pairs.csv", keep_default_na=False)
    vids = [list(map(int, s.split())) for s in V.ids]
    # every pair verb must be scored in every prefix set (else the identity check fails after the expensive scoring)
    scored_cur = set(V.verb) if args.cur_all else set(V.verb[V.curated == 1])
    missing = (set(pairs.t_verb) | set(pairs.i_verb)) - scored_cur
    assert not missing, f"{len(missing)} pair verbs not scored in the cur prefixes (use --cur-all?): {sorted(missing)[:5]}"
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                                 local_files_only=True).to(dev).eval()
    W = model.embed_out.weight  # [vocab, H]; GPT-NeoX unembedding has no bias
    assert getattr(model.embed_out, "bias", None) is None
    cap = {}
    model.gpt_neox.final_layer_norm.register_forward_hook(lambda m, i, o: cap.__setitem__("h", o))

    for p in list(pre.pass_prefix) + list(pre.act_prefix):
        pid = tok.encode(p, add_special_tokens=False)
        for k in range(0, len(V), max(1, len(V) // 40)):
            assert tok.encode(p + " " + V.verb[k], add_special_tokens=False) == pid + vids[k], (p, V.verb[k])

    cur_idx = np.flatnonzero(V.curated.to_numpy() == 1)
    all_idx = np.arange(len(V))
    rows, H, LSE, keys = [], [], [], []
    max_h_dev = max_first_err = max_logit = 0.0
    t0 = time.time()
    with torch.inference_mode():
        for _, pr in pre.iterrows():
            vi = cur_idx if pr["set"] == "cur" and not args.cur_all else all_idx
            for frame in FRAMES:
                prefix = pr[f"{frame}_prefix"]
                pid = tok.encode(prefix, add_special_tokens=False)
                n = len(pid)
                hs, ls = [], []
                for s in range(0, len(vi), args.batch):
                    b = vi[s:s + args.batch]
                    seqs = [pid + vids[j] for j in b]
                    T = max(map(len, seqs))
                    x = torch.full((len(b), T), tok.eos_token_id, dtype=torch.long)
                    att = torch.zeros((len(b), T), dtype=torch.long)
                    for r, sq in enumerate(seqs):
                        x[r, :len(sq)] = torch.tensor(sq)
                        att[r, :len(sq)] = 1
                    x, att = x.to(dev), att.to(dev)
                    logits = model(input_ids=x, attention_mask=att, use_cache=False).logits.float()
                    lp = torch.log_softmax(logits, -1)
                    h = cap["h"][:, n - 1].float()
                    hs.append(h)
                    ls.append(torch.logsumexp(logits[:, n - 1], -1))
                    max_logit = max(max_logit, float(logits[:, n - 1].abs().max()))
                    for r, j in enumerate(b):
                        t = torch.tensor(vids[j], device=dev)
                        sub = lp[r, n - 1:n - 1 + len(vids[j])].gather(-1, t[:, None]).squeeze(-1).double().cpu().numpy()
                        rows.append((pr["set"], pr["context_id"], frame, V.verb[j], float(sub.sum()), float(sub[0]),
                                     len(vids[j]), " ".join(f"{v:.6f}" for v in sub)))
                h = torch.cat(hs)
                max_h_dev = max(max_h_dev, float((h - h[0]).abs().max()))
                lse = torch.cat(ls)
                # first-subtoken log-prob from h . u - LSE
                fid = torch.tensor([vids[j][0] for j in vi], device=dev)
                rec = (h[0] @ W[fid].float().T) - lse[0]
                got = torch.tensor([r[5] for r in rows[-len(vi):]], device=dev, dtype=torch.float32)
                max_first_err = max(max_first_err, float((rec - got).abs().max()))
                H.append(h[0].cpu().numpy())
                LSE.append(float(lse[0]))
                keys.append((pr["set"], pr["context_id"], frame))
        print(f"forward passes done in {time.time() - t0:.0f}s; max h deviation {max_h_dev:.2e}; "
              f"max first-subtoken reconstruction error {max_first_err:.2e}; max |logit| at the slot {max_logit:.1f}", flush=True)
        assert max_h_dev <= 1e-4, max_h_dev
        assert max_first_err <= args.tol, max_first_err

        S = pd.DataFrame(rows, columns=["set", "context_id", "frame", "verb", "lp", "lp1", "n_tok", "lp_sub"])
        H = np.stack(H).astype(np.float32)
        kidx = {k: i for i, k in enumerate(keys)}
        uniq = sorted(set(V.first_id.astype(int)))
        U = W[torch.tensor(uniq, device=dev)].float().cpu().numpy()
        urow = {t: i for i, t in enumerate(uniq)}

        # A2 identity on curated pairs: L1 from log-probs vs (h_was - h_has) . (u_T1 - u_I1)
        first = V.set_index("verb").first_id.astype(int)
        lp1 = S.set_index(["set", "context_id", "frame", "verb"]).lp1
        max_id_err = 0.0
        for (st, cid) in pre[["set", "context_id"]].itertuples(index=False):
            dh = H[kidx[(st, cid, "pass")]] - H[kidx[(st, cid, "act")]]
            for t, i in zip(pairs.t_verb, pairs.i_verb):
                L1 = (lp1[(st, cid, "pass", t)] - lp1[(st, cid, "act", t)]) - (lp1[(st, cid, "pass", i)] - lp1[(st, cid, "act", i)])
                pred = float(dh.astype(np.float64) @ (U[urow[first[t]]].astype(np.float64) - U[urow[first[i]]].astype(np.float64)))
                max_id_err = max(max_id_err, abs(L1 - pred))
        print(f"max A2 identity error {max_id_err:.2e}", flush=True)
        assert max_id_err <= args.tol, max_id_err

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    S.to_parquet(out / "frame_scores.parquet", index=False)
    np.savez(out / "frame_h.npz", H=H, LSE=np.array(LSE), keys=np.array(["|".join(k) for k in keys]),
             U=U, U_ids=np.array(uniq))
    meta = {"model": args.model, "dtype": "float32", "tf32": False, "n_rows": len(S), "cur_all": bool(args.cur_all),
            "inputs": {f: sha(data / f) for f in ("prefixes.csv", "verbs.csv", "pairs.csv")},
            "checks": {"max_h_deviation": max_h_dev, "max_first_subtoken_error": max_first_err,
                       "max_identity_error": max_id_err, "max_abs_logit": max_logit, "tol": args.tol},
            "outputs": {f: sha(out / f) for f in ("frame_scores.parquet", "frame_h.npz")}}
    (out / "frame_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/round5/frame")
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--out-dir", default="results/round5/frame")
    ap.add_argument("--cur-all", action="store_true", help="score every verb in the cur prefixes (round 6, D11)")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="tolerance of the reconstruction and identity checks (1e-3 for 1.4B; 1e-2 for the D10b suite)")
    main(ap.parse_args())
