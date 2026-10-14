#!/usr/bin/env python3
"""Round 5, D10c: FreqBLiMP passive_1 Yes/No readout on Pythia 1.4B / 6.9B / 12B (reports/round5/plan.md, D10c). fp32.

Prompt: task 4's YES_NO_BASE_PROMPT (scripts/build_das_prep_prompts.py). Score per sentence = LSE(Yes logits) -
LSE(No logits) at the answer position, Yes = {"Yes", " Yes"}, No = {"No", " No"} (single tokens, checked); also the
Yes + No probability mass. Items: released FreqBLiMP passive_1, head / tail / xtail (1,000 each).

Output: --out parquet (band, pairID, good, bad, yn_good, yn_bad, mass_good, mass_bad).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_das_prep_prompts import YES_NO_BASE_PROMPT


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, local_files_only=True).to(dev).eval()
    ids = {}
    for name, toks in (("yes", ("Yes", " Yes")), ("no", ("No", " No"))):
        enc = [tok.encode(t, add_special_tokens=False) for t in toks]
        assert all(len(e) == 1 for e in enc), (name, enc)
        ids[name] = torch.tensor([e[0] for e in enc], device=dev)
    rows = []
    for band in ("head", "tail", "xtail"):
        d = pd.read_json(Path(args.freqblimp) / band / "passive_1.jsonl", lines=True)
        for side in ("good", "bad"):
            prompts = [YES_NO_BASE_PROMPT.format(sentence=s) for s in d[f"sentence_{side}"]]
            sc, ms = [], []
            for i in range(0, len(prompts), args.batch):
                enc = tok(prompts[i:i + args.batch], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
                with torch.inference_mode():
                    lg = model(**enc, use_cache=False).logits
                anc = enc.attention_mask.sum(1) - 1
                z = lg[torch.arange(len(anc), device=dev), anc].float()
                lp = torch.log_softmax(z, -1)
                y, n = torch.logsumexp(z[:, ids["yes"]], -1), torch.logsumexp(z[:, ids["no"]], -1)
                sc.append((y - n).double().cpu().numpy())
                ms.append((torch.logsumexp(lp[:, ids["yes"]], -1).exp() + torch.logsumexp(lp[:, ids["no"]], -1).exp()).cpu().numpy())
            d[f"yn_{side}"], d[f"mass_{side}"] = np.concatenate(sc), np.concatenate(ms)
        rows.append(d.assign(band=band)[["band", "pairID", "sentence_good", "sentence_bad", "yn_good", "yn_bad", "mass_good",
                                         "mass_bad"]])
    out = pd.concat(rows, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    acc = out.assign(c=np.where(out.yn_good > out.yn_bad, 1.0, np.where(out.yn_good == out.yn_bad, 0.5, 0.0))).groupby("band").c.mean()
    print(json.dumps({"model": args.model, "acc": acc.round(3).to_dict(), "mass_median": float(out.mass_good.median())}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--freqblimp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    run(ap.parse_args())
