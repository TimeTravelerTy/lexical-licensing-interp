#!/usr/bin/env python3
"""Round 6, C4: frozen classify-from-z manifest (reports/round6/plan.md, C4). No model.

From round-5 A4's stored z (results/round5/restore/restore_zfit.csv): the threshold theta maximizing balanced accuracy
(Youden's J) over A4's Head single-token verbs (282 T, 30 I; ties -> midpoint of the tied interval); each rare verb is
assigned T if z >= theta, else I, and its classify target a* is the Head mean affinity of the assigned class (the
oracle target of that class). The sensitivity threshold (Head of A4 plus D11's extra Head singletons) is computed in the
analysis, after the job supplies their z. Accuracies are not printed here (they are reported by analyze_classify.py).

Outputs: data/round6/classify/manifest.csv (A4's rare manifest + z, assigned class, a_classify), extra_z.csv (D11 Tail
and Head single-token verbs without an A4 z: verb, role, band, first_id), meta.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def youden_threshold(z, is_t):
    """Threshold maximizing balanced accuracy for 'T if z >= theta'; ties -> midpoint of the tied interval."""
    zs = np.sort(np.unique(z))
    cand = np.concatenate([[zs[0] - 1], (zs[:-1] + zs[1:]) / 2, [zs[-1] + 1]])
    ba = np.array([0.5 * ((z[is_t] >= c).mean() + (z[~is_t] < c).mean()) for c in cand])
    best = np.flatnonzero(np.isclose(ba, ba.max()))
    # tied candidates form one or more intervals; take the midpoint of the first maximal run
    run = [best[0]]
    for b in best[1:]:
        if b == run[-1] + 1:
            run.append(b)
        else:
            break
    return float((cand[run[0]] + cand[run[-1]]) / 2), float(ba.max())


def main(args):
    Z = pd.read_csv(args.zfit, keep_default_na=False)
    head = Z[Z.group == "head"]
    rare = Z[Z.group == "rare"].copy()
    assert len(head) == 312 and len(rare) == 32
    theta, _ = youden_threshold(head.z.to_numpy(float), (head.role == "T").to_numpy())
    mean_a = head.groupby("role").a.mean()
    rare["assigned"] = np.where(rare.z.astype(float) >= theta, "T", "I")
    rare["a_classify"] = rare.assigned.map(mean_a)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rare[["verb", "role", "bandkey", "curated", "pool", "verb_pair", "zipf", "first_id", "a", "a_oracle", "z", "assigned",
          "a_classify"]].to_csv(out / "manifest.csv", index=False)
    V = pd.read_csv(args.d11_verbs, keep_default_na=False)
    extra = V[(V.n_tok == 1) & V.band.isin(["head", "tail"]) & ~V.verb.isin(Z.verb)]
    extra[["verb", "role", "band", "first_id"]].to_csv(out / "extra_z.csv", index=False)
    meta = {"theta": theta, "head_class_mean_a": mean_a.to_dict(), "n_head": {k: int(v) for k, v in head.role.value_counts().items()},
            "n_rare": len(rare), "n_extra": {f"{b}/{r}": int(n) for (b, r), n in extra.groupby(["band", "role"]).size().items()},
            "inputs": {args.zfit: sha(args.zfit), args.d11_verbs: sha(args.d11_verbs)},
            "outputs": {f: sha(out / f) for f in ("manifest.csv", "extra_z.csv")}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("n_head", "n_rare", "n_extra")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zfit", default="results/round5/restore/restore_zfit.csv")
    ap.add_argument("--d11-verbs", default="data/round6/frame_d11/verbs.csv")
    ap.add_argument("--out-dir", default="data/round6/classify")
    main(ap.parse_args())
