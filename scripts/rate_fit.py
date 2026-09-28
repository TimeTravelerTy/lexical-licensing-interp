#!/usr/bin/env python3
"""Rate event plausibility of passive sentences with an instruction-tuned LM.

One forward pass per (item, prompt): the rating is the expected value of the
next-token distribution over the digits 1-7 after the assistant turn starts,
renormalized over those digits. `digit_mass` is the unnormalized probability
on the digits; low mass means the model did not answer in the expected format.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROMPTS = {
    "A": ("Rate how plausible the situation described by the sentence is, based on world "
          "knowledge: could this event realistically happen, with these participants in these "
          "roles? Judge the meaning only. Do not penalize rare or formal words, and ignore "
          "grammar and style. If you do not know a word, judge using its most likely meaning.\n\n"
          "Sentence: {sentence}\n\n"
          "Answer with a single digit from 1 (completely implausible) to 7 (completely plausible)."),
    "B": ("How likely is it that the following sentence describes a realistic event? Consider "
          "only who does what to whom, not how common or sophisticated the words are.\n\n"
          "\"{sentence}\"\n\n"
          "Reply with one number: 1 = impossible or absurd, 7 = entirely natural and expected."),
}
DIGITS = "1234567"


def build_prompt(tokenizer, template: str, sentence: str) -> str:
    messages = [{"role": "user", "content": template.format(sentence=sentence)}]
    # enable_thinking is read by Qwen3 templates and ignored by others.
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def run(args):
    import torch

    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = [json.loads(line) for line in Path(args.items).open(encoding="utf-8")]
    if args.limit:
        items = items[:args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=not args.allow_download)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    digit_ids = []
    for d in DIGITS:
        ids = tokenizer.encode(d, add_special_tokens=False)
        if len(ids) != 1:
            raise SystemExit(f"Digit {d!r} is not a single token: {ids}")
        digit_ids.append(ids[0])
    kwargs = {"dtype": getattr(torch, args.dtype), "device_map": args.device_map,
              "local_files_only": not args.allow_download}
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()
    except ValueError:
        # Multimodal checkpoints such as Gemma 4 register only an image-text class.
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(args.model, **kwargs).eval()
    print(f"Loaded {type(model).__name__}", flush=True)
    device = next(model.parameters()).device

    def last_logits(enc):
        try:
            return model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, :]
        except TypeError:
            return model(**enc, use_cache=False).logits[:, -1, :]

    jobs = [(item, name, build_prompt(tokenizer, PROMPTS[name], item["sentence"]))
            for item in items for name in args.prompts]
    jobs.sort(key=lambda j: len(j[2]))
    print(f"{len(items)} items x {len(args.prompts)} prompts = {len(jobs)} forward passes", flush=True)
    print("Example prompt:\n" + jobs[0][2], flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["item_id", "kind", "paradigm", "sentence", "prompt", "rating", "argmax", "digit_mass"] + \
             [f"p{d}" for d in DIGITS]
    digits_t = torch.tensor([int(d) for d in DIGITS], dtype=torch.float32)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for start in range(0, len(jobs), args.batch_size):
            batch = jobs[start:start + args.batch_size]
            enc = tokenizer([j[2] for j in batch], return_tensors="pt", padding=True,
                            add_special_tokens=False).to(device)
            with torch.inference_mode():
                logits = last_logits(enc).float()
            probs = torch.softmax(logits, dim=-1)[:, digit_ids].cpu()
            mass = probs.sum(dim=-1)
            norm = probs / mass.unsqueeze(-1)
            expected = norm @ digits_t
            for (item, name, _), p, m, e in zip(batch, norm, mass, expected):
                row = {"item_id": item["item_id"], "kind": item["kind"], "paradigm": item["paradigm"],
                       "sentence": item["sentence"], "prompt": name, "rating": round(float(e), 4),
                       "argmax": int(p.argmax()) + 1, "digit_mass": round(float(m), 4)}
                row.update({f"p{d}": round(float(v), 4) for d, v in zip(DIGITS, p)})
                writer.writerow(row)
            done = min(start + len(batch), len(jobs))
            if done % (args.batch_size * 20) < args.batch_size or done == len(jobs):
                print(f"Rated {done}/{len(jobs)}", flush=True)
    print(f"Wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", default="data/fit_ratings/items.jsonl")
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--prompts", nargs="+", default=list(PROMPTS), choices=list(PROMPTS))
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--out", default="results/fit_ratings/gemma4_31b_it.csv")
    run(ap.parse_args())
