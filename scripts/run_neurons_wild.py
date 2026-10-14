#!/usr/bin/env python3
"""Round 5, B7: D11's switch neurons on natural text (reports/round5/plan.md, B7). fp32.

2,000 documents sampled (seed 17) from NeelNanda/pile-10k, each truncated to 512 Pythia tokens. For the switch neurons
of ranks 1-10 (conjunction_switch_neurons.csv), the post-activation (input of dense_4h_to_h) at every token.
Per token also: the decoded token, whether it ends a word, the full word it ends, the 8 preceding words and the next
token (word segmentation by character offsets; words = runs of letters and apostrophes).

Output (--out-dir): wild_tokens.parquet (one row per token), wild_meta.json.
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

WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    dev = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, local_files_only=True).to(dev).eval()
    sw = pd.read_csv(args.switch)
    top = sw[sw["rank"] <= 10].sort_values("rank").reset_index(drop=True)
    layers = sorted(set(top.layer))
    cap = {}
    hooks = [model.gpt_neox.layers[l].mlp.dense_4h_to_h.register_forward_pre_hook(
        lambda m, i, l=l: cap.__setitem__(l, i[0].float())) for l in layers]
    texts = pd.read_parquet(args.pile10k).text
    pick = np.sort(np.random.default_rng(args.seed).choice(len(texts), args.docs, replace=False))
    rows = []
    for k in range(0, len(pick), args.batch):
        idx = pick[k:k + args.batch]
        encs = [tok(texts.iloc[i], add_special_tokens=False, return_offsets_mapping=True, truncation=True,
                    max_length=args.max_len) for i in idx]
        T = max(len(e.input_ids) for e in encs)
        x = torch.full((len(idx), T), tok.eos_token_id, dtype=torch.long)
        att = torch.zeros((len(idx), T), dtype=torch.long)
        for j, e in enumerate(encs):
            x[j, :len(e.input_ids)] = torch.tensor(e.input_ids)
            att[j, :len(e.input_ids)] = 1
        with torch.inference_mode():
            model(input_ids=x.to(dev), attention_mask=att.to(dev), use_cache=False)
        acts = np.stack([cap[l][:, :, int(i)].cpu().numpy() for l, i in zip(top.layer, top["index"])], -1)  # [B, T, 10]
        for j, (di, e) in enumerate(zip(idx, encs)):
            text = texts.iloc[di]
            words = [(m.start(), m.end(), m.group()) for m in WORD.finditer(text)]
            ends = {w[1]: n for n, w in enumerate(words)}
            n = len(e.input_ids)
            for p in range(n):
                a, b = e.offset_mapping[p]
                wi = ends.get(b)
                nxt_starts_word = p + 1 < n and text[e.offset_mapping[p + 1][0]:e.offset_mapping[p + 1][1]][:1].isalpha() and \
                    e.offset_mapping[p + 1][0] == b
                ends_word = wi is not None and not nxt_starts_word
                rows.append({"doc": int(di), "pos": p, "tok_id": int(e.input_ids[p]), "start": int(a), "end": int(b),
                             "next_id": int(e.input_ids[p + 1]) if p + 1 < n else -1, "tok": text[a:b], "ends_word": ends_word,
                             "word": words[wi][2] if ends_word else "",
                             "prev_words": " ".join(w[2] for w in words[max(0, wi - 8):wi]) if ends_word else "",
                             "next_tok": text[e.offset_mapping[p + 1][0]:e.offset_mapping[p + 1][1]] if p + 1 < n else "",
                             **{f"a{r}": float(acts[j, p, r]) for r in range(len(top))}})
        print(f"{k + len(idx)}/{len(pick)} docs ({time.time() - t0:.0f}s)", flush=True)
    for h in hooks:
        h.remove()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out / "wild_tokens.parquet", index=False)
    meta = {"args": vars(args), "pile10k_sha256": hashlib.sha256(Path(args.pile10k).read_bytes()).hexdigest(), "neurons": top[["rank", "neuron", "layer", "index", "I_A"]].to_dict("records"),
            "docs": [int(i) for i in pick], "tokens": len(df), "switch_sha256": hashlib.sha256(Path(args.switch).read_bytes()).hexdigest(),
            "seconds": round(time.time() - t0)}
    (out / "wild_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("tokens", "seconds")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pile10k", required=True)
    ap.add_argument("--switch", default="results/round4/translation/conjunction_switch_neurons.csv")
    ap.add_argument("--out-dir", default="results/round5/wild")
    ap.add_argument("--docs", type=int, default=2000)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
