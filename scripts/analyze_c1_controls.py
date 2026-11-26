#!/usr/bin/env python3
"""C1 trained-direction controls (reports/v2_controls_mech_b_rules.md, part A).

For one model: v2's passive run at its rule sites (results/{m}/das_v2/expanded_eval/passive_rule) and control runs of
`run_passive_test.py` on the same plan (--controls name=dir,...; each dir holds patches_site{s}.parquet for its sites).
On one set of joint draws per site (contexts, DAS donor pairs, pairs within band; seed and n_boot as
analyze_v2_expanded.py), per population and readout (by, O): D for v2 and each control, the natural gap G, D / G, the
paired difference v2 - control and the ratio control / v2.
Declared (first rule site, by, eval plain): A1 lo95(D_v2 - D_shuf) > 0; A2 for each random draw, the 95% CI of D within
+-0.1 G (lo95(D + 0.1 G) > 0 and hi95(D - 0.1 G) < 0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_passive_test as apt
from analyze_v2_expanded import CFG, d_bad_all, populations

RULE_SITES = {"pythia": (8, 12), "gemma": (5, 8), "olmo": (9, 11)}
POPS = ("eval plain", "eval new plain", "eval new plain, lemma-fresh")


def nat_gap(items, nat, P, bt, pair_ix, ctx_ix):
    """Natural good - bad per pair and draw [B+1, P, 2] (by, O), as analyze_passive_test.transfer."""
    g = items[["pair_id", "context_id", "good_prompt", "bad_prompt"]].copy()
    pid, pidx = nat.set_index("pid"), dict(zip(P.prompt, P.pid))
    g[["by", "O"]] = (pid.loc[g.good_prompt.map(pidx), ["by", "O"]].to_numpy()
                      - pid.loc[g.bad_prompt.map(pidx), ["by", "O"]].to_numpy())
    v, m = apt.cells(g.assign(dq=0), pair_ix, ctx_ix, {0: 0}, 1, ["by", "O"])
    return apt.pair_values(v, m, bt.wc, np.ones((bt.B + 1, 1), np.float32))


def main(a):
    cfg = CFG[a.model]
    data = Path(cfg["data"])
    items = pd.read_csv(data / "items.csv", low_memory=False)
    P = pd.read_csv(data / "prompts.csv")
    plan = pd.read_csv(data / "plan.csv.gz", dtype={c: "category" for c in ("item_id", "side", "cond", "donor_pair")})
    old = pd.read_csv(a.old_eval)
    pr, pops = populations(items, pd.read_csv(a.frozen), set(old.trans_lemma) | set(old.intrans_lemma))
    v2dir = Path(f"results/{a.model}/das_v2/expanded_eval/passive_rule")
    nat = pd.read_parquet(v2dir / "natural.parquet")
    ctrl = dict(x.split("=") for x in a.controls.split(","))
    s0, s1 = RULE_SITES[a.model]
    expected = {"shuffled": (s0, s1), **{f"random_{k}": (s0,) for k in range(3)}}
    checks, bad = {}, []
    assert (plan.row.to_numpy() == np.arange(len(plan))).all(), "plan row column is not its position"
    vn = nat.set_index("pid").sort_index()
    for name, sites in expected.items():  # fail closed: every expected control x site, checked before any decision
        if name not in ctrl:
            bad.append(f"{name}: missing run")
            continue
        d = Path(ctrl[name])
        cn = pd.read_parquet(d / "natural.parquet").set_index("pid").sort_index()
        same = cn.index.equals(vn.index)
        fin = bool(np.isfinite(cn[["by", "O"]].to_numpy(float)).all())
        dmax = float(np.nanmax(np.abs(cn[["by", "O"]].to_numpy(float) - vn[["by", "O"]].to_numpy(float)))) if same else float("inf")
        rm = json.loads((d / "run_meta.json").read_text())
        checks[name] = {"natural_same_pids": bool(same), "natural_finite": fin, "natural_max_abs": dmax}
        if not (same and fin and dmax <= 1e-6):
            bad.append(f"{name}: natural readouts")
        for s in sites:
            pf = d / f"patches_site{s}.parquet"
            mp = d.parent / "bases" / f"{name}_site{s}" / "control_meta.json"
            if not pf.exists() or not mp.exists():
                bad.append(f"{name} site {s}: missing patches or control_meta")
                continue
            pat = pd.read_parquet(pf, columns=["row", "by", "O"])
            sp = float(rm["checks"][str(s)]["self_patch_max_abs_dM"])
            sha_ok = rm["bases_sha256"][str(s)] == json.loads(mp.read_text())["sha256"]
            fin_p = bool(np.isfinite(pat[["by", "O"]].to_numpy(float)).all())
            checks[f"{name}_site{s}"] = {"self_patch_max_abs_dM": sp, "sha_matches_control_meta": sha_ok,
                                         "patched_finite": fin_p, "rows": len(pat)}
            if not (sp <= 0.1 and sha_ok and fin_p and len(pat) == len(plan)):
                bad.append(f"{name} site {s}: self-patch / sha / finite / rows")
    if bad:
        Path(a.out_dir).mkdir(parents=True, exist_ok=True)
        (Path(a.out_dir) / "c1_controls_decisions.json").write_text(json.dumps({"INVALID": bad, "checks": checks}, indent=1) + "\n")
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(f"# C1 trained-direction controls: {a.model}\n\nINVALID: {bad}\n\nChecks: {json.dumps(checks)}\n")
        raise SystemExit(f"INVALID: {bad}")
    pair_ix = {p: i for i, p in enumerate(pr.index)}
    ctx_ix = {c: i for i, c in enumerate(sorted(items.context_id.unique()))}
    das_q = sorted(plan[plan.cond == "T"].donor_pair.dropna().astype(str).unique())
    rows = []
    for s in RULE_SITES[a.model]:
        have = {n: Path(d) for n, d in ctrl.items() if (Path(d) / f"patches_site{s}.parquet").exists()}
        if not have:
            continue
        bt = apt.Boot(items.context_id.nunique(), a.n_boot, a.seed)
        A = {"v2": d_bad_all(v2dir, s, plan, items, nat, bt, pair_ix, ctx_ix, das_q)}
        for n, d in have.items():
            A[n] = d_bad_all(d, s, plan, items, nat, bt, pair_ix, ctx_ix, das_q)
        Gp = nat_gap(items, nat, P, bt, pair_ix, ctx_ix)
        for pop in POPS:
            pix = np.flatnonzero(pops[pop].to_numpy())
            wp = bt.pair_weights(pr.band.to_numpy()[pix])
            S = {k: apt.summarize(x, wp, pix) for k, x in A.items()}
            G = apt.summarize(Gp, wp, pix)
            for ri, rd in enumerate(("by", "O")):
                v, g = S["v2"][:, ri], G[:, ri]
                for n in S:
                    x = S[n][:, ri]
                    r = {"model": a.model, "site": s, "population": pop, "n_pairs": len(pix), "readout": rd,
                         "instrument": n, **{f"D_{k}": q for k, q in apt.ci(x).items()},
                         **{f"DG_{k}": q for k, q in apt.ci(x / g).items()},
                         **{f"G_{k}": q for k, q in apt.ci(g).items()}}
                    if n != "v2":
                        r.update({f"v2_minus_{k}": q for k, q in apt.ci(v - x).items()})
                        r.update({f"v2_minus_G_{k}": q for k, q in apt.ci((v - x) / g).items()})
                        r.update({f"ratio_to_v2_{k}": q for k, q in apt.ci(x / v).items()})
                        r.update({f"plus10G_{k}": q for k, q in apt.ci(x + 0.1 * g).items()})
                        r.update({f"minus10G_{k}": q for k, q in apt.ci(x - 0.1 * g).items()})
                    rows.append(r)
    T = pd.DataFrame(rows)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / "c1_controls.csv", index=False)
    s0 = RULE_SITES[a.model][0]
    base = T[(T.site == s0) & (T.population == "eval plain") & (T.readout == "by")].set_index("instrument")
    dec = {"checks": checks, "site": s0}
    if "shuffled" in base.index:
        dec["A1_v2_exceeds_shuffled"] = bool(base.loc["shuffled", "v2_minus_lo95"] > 0)
    rnd = [n for n in base.index if n.startswith("random")]
    dec["A2_random_within_10pct_G"] = {n: bool(base.loc[n, "plus10G_lo95"] > 0 and base.loc[n, "minus10G_hi95"] < 0) for n in rnd}
    dec["abs_cos_with_v2"] = {}
    for n, d in ctrl.items():  # bases dirs: <run dir>/../bases/<name>_site<s>/control_meta.json
        for s in RULE_SITES[a.model]:
            mp = Path(d).parent / "bases" / f"{n}_site{s}" / "control_meta.json"
            if mp.exists():
                m = json.loads(mp.read_text())
                dec["abs_cos_with_v2"][f"{n}_site{s}"] = {"mean": round(m["abs_cos_mean"], 3),
                                                          "min": round(min(m["abs_cos_with_v2"].values()), 3),
                                                          "max": round(max(m["abs_cos_with_v2"].values()), 3)}
    (out / "c1_controls_decisions.json").write_text(json.dumps(dec, indent=1) + "\n")
    f = lambda r, p: f"{r[p + '_est']:+.2f} [{r[p + '_lo95']:+.2f}, {r[p + '_hi95']:+.2f}]"
    L = [f"# C1 trained-direction controls: {a.model}", "",
         "Rules: `reports/v2_controls_mech_b_rules.md` (part A). Same plan, draws and populations as v2's expanded run; "
         "nats.", "", f"Checks: {json.dumps(checks)}. |cos| with v2: {json.dumps(dec['abs_cos_with_v2'])}.", "",
         "| site | population | readout | instrument | D | D / G | v2 − control (nats) | (v2 − control) / G | control / v2 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in T.to_dict("records"):
        L.append(f"| {r['site']} | {r['population']} ({r['n_pairs']}) | {r['readout']} | {r['instrument']} | {f(r, 'D')} | "
                 f"{f(r, 'DG')} | {f(r, 'v2_minus') if r['instrument'] != 'v2' else ''} | "
                 f"{f(r, 'v2_minus_G') if r['instrument'] != 'v2' else ''} | "
                 f"{f(r, 'ratio_to_v2') if r['instrument'] != 'v2' else ''} |")
    L += ["", f"**Declared (site {s0}, by, eval plain):** A1 v2 exceeds shuffled **{dec.get('A1_v2_exceeds_shuffled')}**; "
              f"A2 random within ±0.1 G {dec['A2_random_within_10pct_G']}.", ""]
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=list(CFG))
    ap.add_argument("--controls", required=True, help="name=run_dir,... (shuffled, random_0, ...)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--frozen", default="data/das_v2/expanded_eval_pairs.csv")
    ap.add_argument("--old-eval", default="data/verb_expansion/eval_pairs.csv")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    main(ap.parse_args())
