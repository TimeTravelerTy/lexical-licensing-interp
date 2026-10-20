#!/usr/bin/env python3
"""Round 6, A1: occurrence samples for the D3 audit and the broad-pronoun precision audit (reports/round6/plan.md, A1).
Uses counts and Zipf only (no model outcome).

D3: the 10 curated I verbs with the largest positive and the 10 with the largest negative residual of log10 c_total on
Zipf (one regression over the 608 forms), and 10 controls: curated I verbs with the smallest |residual|, chosen band by
band to match the bands of the 20 as closely as possible. Per verb, 20 occurrences allocated to the four c_total
variants ("v", "V", "\\nv", "\\nV") in proportion to their counts (largest remainder), at evenly spaced suffix-array
ranks within a variant (find + get_doc_by_rank). Precision audit: 6 verbs (3 T, 3 I; curated; one per class x band,
seed 17), up to 10 occurrences each of "it p" and "she p".
Output: data/round6/counts/diag_contexts.csv (audit, group, verb, variant, query, k, count, doc_ix, pile_set, snippet)
and an empty label sheet diag_labels.csv (labels: D3 verbal / adjectival / nominal / other; precision finite active /
other), filled by hand before the A1 outcome analysis.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_pile_counts_r6 import API  # noqa: F401  (same endpoint)

PAST = {"arisen": "arose", "striven": "strove", "browbeaten": "browbeat", "outdone": "outdid"}


def call(body, tries=30, delay=0.8):
    import urllib.error
    import urllib.request

    for t in range(tries):
        try:
            req = urllib.request.Request("https://api.infini-gram.io/", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.loads(r.read())
            if "error" in d:
                raise RuntimeError(d["error"])
            time.sleep(delay)
            return d
        except urllib.error.HTTPError as e:
            time.sleep(60 if e.code in (429, 403) else min(2 ** t, 60))
        except Exception:  # noqa: BLE001
            time.sleep(min(2 ** t, 60))
    raise SystemExit(f"failed: {body}")


def sample(index, query, n, needle):
    f = call({"index": index, "query_type": "find", "query": query})
    segs, cnt = f["segment_by_shard"], int(f["cnt"])
    out = []
    if cnt == 0 or n == 0:
        return cnt, out
    sizes = [b - a for a, b in segs]
    bounds = np.cumsum(sizes)
    for k, i in enumerate(np.unique(np.linspace(0, cnt - 1, min(n, cnt)).round().astype(int))):
        s = int(np.searchsorted(bounds, i, side="right"))
        r = segs[s][0] + int(i - (bounds[s - 1] if s else 0))
        d = call({"index": index, "query_type": "get_doc_by_rank", "s": s, "rank": r, "max_disp_len": 80, "query": ""})
        spans = d["spans"]
        text = "".join(sp[0] for sp in spans) if isinstance(spans, list) else str(spans)
        meta = json.loads(d.get("metadata", "{}") or "{}")
        pile = meta.get("metadata", {}).get("meta", {}).get("pile_set_name", "")
        occ = [m.start() for m in re.finditer(needle, text)]
        o = min(occ, key=lambda x: abs(x - len(text) / 2)) if occ else len(text) // 2
        snip = (text[max(0, o - 150):o] + "[[" + text[o:o + 40] + "]]" + text[o + 40:o + 120]).replace("\n", " ⏎ ")
        out.append({"k": k, "doc_ix": int(d["doc_ix"]), "pile_set": pile, "snippet": snip})
    return cnt, out


def main(args):
    C = pd.read_csv(args.counts, keep_default_na=False)
    C["count"] = C["count"].astype(float)
    W = C.pivot(index="verb", columns="slot", values="count")
    ct = W.overall + W.overall_cap + W.nl_low + W.nl_cap
    Vc = pd.read_csv(args.cur_verbs, keep_default_na=False).query("curated == 1").set_index("verb")
    Vd = pd.read_csv(args.d11_verbs, keep_default_na=False).set_index("verb")
    z = pd.concat([Vd.zipf, Vc.zipf])
    z = z[~z.index.duplicated()].loc[ct.index].astype(float)
    lc = np.log10(ct + 0.5)
    b = np.polyfit(z, lc, 1)
    resid = lc - np.polyval(b, z)
    ci_ = Vc[Vc.role == "I"].index
    r = resid.loc[ci_]
    top, bot = r.nlargest(10).index.tolist(), r.nsmallest(10).index.tolist()
    need = Vc.loc[top + bot].band.value_counts()
    pool = r.drop(top + bot).abs().sort_values()
    ctrl = []
    for band, n in need.items():
        ctrl += [v for v in pool.index if Vc.loc[v, "band"] == band][: max(1, round(10 * n / 20))]
    ctrl = sorted(ctrl, key=lambda v: pool[v])[:10]
    rows = []
    for group, verbs in (("Pile > Zipf", top), ("Pile < Zipf", bot), ("control", ctrl)):
        for v in verbs:
            cap = v[0].upper() + v[1:]
            var = {"lower": (v, W.loc[v, "overall"]), "Cap": (cap, W.loc[v, "overall_cap"]),
                   "nl lower": ("\n" + v, W.loc[v, "nl_low"]), "nl Cap": ("\n" + cap, W.loc[v, "nl_cap"])}
            tot = sum(c for _, c in var.values())
            raw = {k: args.per_verb * c / tot for k, (_, c) in var.items()}
            alloc = {k: int(np.floor(x)) for k, x in raw.items()}
            for k in sorted(raw, key=lambda k: raw[k] - alloc[k], reverse=True)[: args.per_verb - sum(alloc.values())]:
                alloc[k] += 1
            for name, (q, c) in var.items():
                cnt, occ = sample(args.index, q, alloc[name], rf"(?<![A-Za-z]){re.escape(q.strip())}(?![A-Za-z])")
                rows += [{"audit": "D3", "group": group, "verb": v, "variant": name, "query": q, "count": cnt, **o} for o in occ]
            print(group, v, alloc, flush=True)
    rng = np.random.default_rng(args.seed)
    prec = []
    for role in ("T", "I"):
        for band in ("head", "tail", "xtail"):
            cand = sorted(Vc[(Vc.role == role) & (Vc.band == band)].index)
            prec.append(rng.choice(cand))
    for v in prec:
        p = PAST.get(v, v)
        for pron in ("it", "she"):
            q = f"{pron} {p}"
            cnt, occ = sample(args.index, q, 10, rf"(?<![A-Za-z]){re.escape(q)}(?![A-Za-z])")
            rows += [{"audit": "precision", "group": Vc.loc[v, "role"], "verb": v, "variant": pron, "query": q, "count": cnt, **o}
                     for o in occ]
        print("precision", v, flush=True)
    D = pd.DataFrame(rows)
    out = Path(args.out_dir)
    D.to_csv(out / "diag_contexts.csv", index=False)
    lab = D[["audit", "group", "verb", "variant", "k", "snippet"]].assign(label="")
    if not (out / "diag_labels.csv").exists():
        lab.to_csv(out / "diag_labels.csv", index=False)
    meta = {"index": args.index, "regression": list(map(float, b)), "pile_gt_zipf": top, "pile_lt_zipf": bot,
            "controls": ctrl, "precision_verbs": list(map(str, prec)), "rows": len(D), "date": time.strftime("%Y-%m-%d")}
    (out / "diag_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--counts", default="data/round6/counts/pile_counts.csv")
    ap.add_argument("--cur-verbs", default="data/round5/frame/verbs.csv")
    ap.add_argument("--d11-verbs", default="data/round6/frame_d11/verbs.csv")
    ap.add_argument("--index", default="v4_piletrain_llama")
    ap.add_argument("--per-verb", type=int, default=20)
    ap.add_argument("--out-dir", default="data/round6/counts")
    ap.add_argument("--seed", type=int, default=17)
    main(ap.parse_args())
