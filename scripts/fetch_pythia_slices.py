#!/usr/bin/env python3
"""Round 6, A6: a limited stationarity diagnostic for verb occurrences in Pythia's training stream
(reports/round6/plan.md, A6). No model.

The pre-shuffled Pile that Pythia (standard) trained on is public in training order
(EleutherAI/pile-standard-pythia-preshuffled): raw uint16 token ids, 20 shards of 15e9 tokens plus a final partial
shard; each step stores 1024 records of 2049 ids (inputs and shifted targets), so step t starts at stored token
t x 1024 x 2049 (the 143,000 steps fill the files exactly). This script reads --slice-tokens stored tokens starting at
each step of --steps with HTTP range requests (crossing shard boundaries; the 206 status and Content-Range are checked
before reading; no full download) and counts the token ids of the D11 single-token participles (" V", Pythia tokenizer).
Output: data/round6/checkpoints/slice_counts.csv (slice, step, start_token, n_tokens, verb, role, band, token_id, count),
slice_meta.json.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://huggingface.co/datasets/EleutherAI/pile-standard-pythia-preshuffled/resolve/main/document-{:05d}-of-00020.bin"
SHARD_TOKENS = 15_000_000_000
LAST_SHARD_TOKENS = 39_168_000  # document-00020 (78,336,000 bytes)
RECORD_TOKENS = 1024 * 2049
TOTAL_STEPS = 143_000


def read_range(shard, start_tok, n_tok, tries=10):
    a, b = start_tok * 2, (start_tok + n_tok) * 2 - 1
    for t in range(tries):
        try:
            req = urllib.request.Request(URL.format(shard), headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=300) as r:
                cr = r.headers.get("Content-Range", "")
                m = re.match(r"bytes (\d+)-(\d+)/", cr)
                if r.status != 206 or not m or (int(m.group(1)), int(m.group(2))) != (a, b):
                    raise RuntimeError(f"range not honoured: status {r.status}, Content-Range {cr!r}")
                chunks, got = [], 0
                while got < n_tok * 2:
                    c = r.read(min(1 << 24, n_tok * 2 - got))
                    if not c:
                        break
                    chunks.append(c)
                    got += len(c)
            buf = b"".join(chunks)
            assert len(buf) == n_tok * 2, (len(buf), n_tok * 2)
            return np.frombuffer(buf, dtype=np.uint16)
        except Exception as e:  # noqa: BLE001
            print(f"retry {t}: {e}", flush=True)
            time.sleep(min(2 ** t, 60))
    raise SystemExit("range read failed")


def read_stream(g, n):
    """n stored tokens from global position g, across shard boundaries."""
    parts = []
    while n > 0:
        shard, off = int(g // SHARD_TOKENS), int(g % SHARD_TOKENS)
        cap = LAST_SHARD_TOKENS if shard == 20 else SHARD_TOKENS
        k = min(n, cap - off)
        parts.append(read_range(shard, off, k))
        g, n = g + k, n - k
    return np.concatenate(parts)


def main(args):
    V = pd.read_csv(args.verbs, keep_default_na=False)
    V = V[V.n_tok == 1]
    ids = V.first_id.astype(int).to_numpy()
    rows = []
    total = TOTAL_STEPS * RECORD_TOKENS
    for i, st in enumerate(args.steps):
        g = st * RECORD_TOKENS
        n = min(args.slice_tokens, total - g)
        t0 = time.time()
        x = read_stream(g, n)
        cnt = np.bincount(x, minlength=65536)
        for r, t in zip(V.itertuples(index=False), ids):
            rows.append({"slice": i, "step": st, "start_token": int(g), "n_tokens": int(n), "verb": r.verb, "role": r.role,
                         "band": r.band, "token_id": int(t), "count": int(cnt[t])})
        print(f"slice {i} (step {st}): {n} tokens, {time.time() - t0:.0f}s", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "slice_counts.csv", index=False)
    (out / "slice_meta.json").write_text(json.dumps({"url": URL, "steps": args.steps, "slice_tokens": args.slice_tokens,
                                                     "record_tokens": RECORD_TOKENS, "total_steps": TOTAL_STEPS,
                                                     "verbs": len(V), "date": time.strftime("%Y-%m-%d")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbs", default="data/round6/frame_d11/verbs.csv")
    ap.add_argument("--steps", type=int, nargs="+",
                    default=[128, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000, 130000, 142000])
    ap.add_argument("--slice-tokens", type=int, default=25_000_000)
    ap.add_argument("--out-dir", default="data/round6/checkpoints")
    main(ap.parse_args())
