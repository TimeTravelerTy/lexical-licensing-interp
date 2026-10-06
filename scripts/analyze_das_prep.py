#!/usr/bin/env python3
"""Tasks 1-4 of the passive DAS groundwork (Pythia-1.4B, next-token readout).

1. Natural baseline: next-token log-probs after the participle (curated
   `was` prefixes, other verbs' contexts) and after the verb in actives.
2. *by* margin split into log P(by | good) and log P(by | bad) per verb pair
   (released `passive_1` contexts), with z-scores against reliable pairs.
3. Single-prompt readout: log-odds(" by" vs ".") after the participle,
   correlated with the two-sentence *by* margin.
4. Yes/No readout (FreqBLiMP base prompt) on the curated cross, and whether
   the fit effects seen under LP persist.

Intervals: two-way cluster bootstrap (verb pairs within verb band x contexts
within context band), as in the band-cross and fit analyses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_bad_fit import fit
from analyze_fit_ratings import mean_rating, weights
from build_das_prep_prompts import PAST, YES_NO_BASE_PROMPT, pid

BANDS = ("head", "tail", "xtail")
OBJ = (" the", " a", " an", " his", " her", " their", " its", " this", " that", " some",
       " him", " them", " it")


# ---------------------------------------------------------------- helpers

def lse(frame, toks):
    return np.logaddexp.reduce(np.column_stack([frame[f"lp[{t}]"] for t in toks]), axis=1)


def load_readout(path):
    r = pd.read_csv(path, low_memory=False)
    r["lp_det"] = lse(r, (" the", " a", " an"))
    r["lp_obj"] = lse(r, OBJ)
    r["lp_yes"] = lse(r, ("Yes", " Yes"))
    r["lp_no"] = lse(r, ("No", " No"))
    return r.set_index("prompt_id")


def attach(df, readout, kind, prompt_col, side, cols):
    ids = df[prompt_col].map(lambda p: pid(kind, p))
    sub = readout.reindex(ids)[cols]
    if sub.isna().any().any():
        raise SystemExit(f"missing readout rows for {kind}/{side}")
    for c in cols:
        df[f"{side}_{c.replace('lp[', 'lp').replace(']', '').replace(' ', '_')}"] = sub[c].to_numpy()
    return df


class Boot:
    """Two-way cluster bootstrap weights for one item table."""

    def __init__(self, df, n_boot, rng, ctx_band="context_band"):
        verbs = df.drop_duplicates("verb_pair").sort_values("verb_pair")
        ctxs = df.drop_duplicates("context_id").sort_values("context_id")
        self.vi = df.verb_pair.map({v: i for i, v in enumerate(verbs.verb_pair)}).to_numpy()
        self.ci = df.context_id.map({c: i for i, c in enumerate(ctxs.context_id)}).to_numpy()
        self.wv = weights(rng, verbs.verb_band.to_numpy(), n_boot)
        self.wc = weights(rng, ctxs[ctx_band].to_numpy(), n_boot)
        self.band = df.verb_band.to_numpy()

    def means(self, y, mask):
        w = self.wv[:, self.vi[mask]] * self.wc[:, self.ci[mask]]
        return (w * y[mask]).sum(1) / w.sum(1)

    def cells(self, y):
        """Per-band means and XTail - Head, each as (estimate, lo, hi)."""
        y = np.asarray(y, float)
        d = {b: self.means(y, self.band == b) for b in BANDS}
        d["xtail-head"] = d["xtail"] - d["head"]
        return {k: (v[0], *np.percentile(v[1:], [2.5, 97.5])) for k, v in d.items()}

    def spearman(self, x, y, n):
        """Spearman on resampled items (cartesian product of drawn clusters)."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        out = [spear(x, y)]
        for b in range(1, min(n, self.wv.shape[0] - 1) + 1):
            rep = (self.wv[b, self.vi] * self.wc[b, self.ci]).astype(int)
            idx = np.repeat(np.arange(len(x)), rep)
            out.append(spear(x[idx], y[idx]))
        return out[0], *np.percentile(out[1:], [2.5, 97.5])


def spear(x, y):
    return float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])


def boot_spearman_units(x, y, n, rng):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = [spear(x[i], y[i]) for i in rng.integers(0, len(x), (n, len(x)))]
    return spear(x, y), *np.percentile(d, [2.5, 97.5])


def f3(t, k=2):
    return f"{t[0]:.{k}f} [{t[1]:.{k}f}, {t[2]:.{k}f}]"


def table(header, rows):
    """Markdown table; columns whose cells all start like a number are right-aligned."""
    def numeric(j):
        cells = [str(r[j]) for r in rows if str(r[j])]
        return bool(cells) and all(c.lstrip("+-")[:1].isdigit() for c in cells)
    align = ["---:" if numeric(j) else "---" for j in range(len(header))]
    return ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"] + \
        ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def topk_mass(frames):
    """Mean probability of tokens over prompts, from the stored top-10 (0 if absent)."""
    acc = {}
    for s in frames:
        for tok, lp in json.loads(s):
            acc[tok] = acc.get(tok, 0.0) + float(np.exp(lp))
    return sorted(((t, p / len(frames)) for t, p in acc.items()), key=lambda x: -x[1])


def show(tok):
    return json.dumps(tok)[1:-1].replace("|", "\\|")


# ---------------------------------------------------------------- data

def curated_items(scores, v2, paradigm):
    cur = scores[(scores.context_set == "curated") & (scores.paradigm == paradigm)].copy()
    agents = {f"curated/{r.paradigm}/{r.band}/{r.good_lemma}": r.agent for r in v2.itertuples()}
    cur["agent"] = cur.context_id.map(agents)
    return cur


# ---------------------------------------------------------------- task 1

def task1(scores, v2, ro, args, rng):
    cols = ["lp[ the]", "lp[ a]", "lp[ by]", "lp[.]", "lp_det", "lp_obj"]
    # passive_2 rows index (verb pair x curated context); the prefix equals passive_1's.
    pas = curated_items(scores, v2, "passive_2")
    pas = pas[pas.own_context == 0].copy()
    pas["good_prompt"], pas["bad_prompt"] = pas.prefix + pas.good_verb, pas.prefix + pas.bad_verb
    attach(pas, ro, "passive", "good_prompt", "good", cols + ["topk"])
    attach(pas, ro, "passive", "bad_prompt", "bad", cols + ["topk"])
    act = curated_items(scores, v2, "passive_1")
    act = act[act.own_context == 0].copy()
    act["good_prompt"] = "The " + act.agent + " " + act.good_verb.map(lambda v: PAST.get(v, v))
    act["bad_prompt"] = "The " + act.agent + " " + act.bad_verb
    acols = ["lp[ the]", "lp[ a]", "lp[.]", "lp_det", "lp_obj"]
    attach(act, ro, "active", "good_prompt", "good", acols + ["topk"])
    attach(act, ro, "active", "bad_prompt", "bad", acols + ["topk"])

    bp, ba = Boot(pas, args.n_boot, rng), Boot(act, args.n_boot, rng)
    names = {"lp_the": 'log P(" the")', "lp_a": 'log P(" a")', "lp_by": 'log P(" by")',
             "lp.": 'log P(".")', "lp_det": "log P(det: the/a/an)", "lp_obj": "log P(object start)"}
    rows, lines = [], ["# Task 1: natural passive baseline", "",
                       "Pythia-1.4B next-token log-probs (bf16, TSUBAME). Passives: curated `was` prefix + "
                       "participle, other verbs' contexts only (126 verb pairs x 125 contexts = "
                       f"{len(pas):,} items). Actives: \"The AGENT VERB-ed\" with the agent of the same "
                       "curated `passive_1` context. Object start = the, a, an, his, her, their, its, this, "
                       "that, some, him, them, it. 95% CIs: two-way cluster bootstrap "
                       f"({args.n_boot} draws).", ""]
    for label, df, boot, keys in (("passive", pas, bp, ["lp_the", "lp_a", "lp_by", "lp.", "lp_det", "lp_obj"]),
                                  ("active", act, ba, ["lp_the", "lp_a", "lp.", "lp_det", "lp_obj"])):
        lines += [f"## {label.capitalize()}: good (transitive) vs bad (intransitive) verb", ""]
        trs = []
        for k in keys:
            g, b = df[f"good_{k}"].to_numpy(), df[f"bad_{k}"].to_numpy()
            cg, cb, cd = boot.cells(g), boot.cells(b), boot.cells(g - b)
            for band in BANDS + ("xtail-head",):
                rows.append({"set": label, "metric": k, "band": band, "side": "good", "estimate": cg[band][0],
                             "ci_low": cg[band][1], "ci_high": cg[band][2]})
                rows.append({"set": label, "metric": k, "band": band, "side": "bad", "estimate": cb[band][0],
                             "ci_low": cb[band][1], "ci_high": cb[band][2]})
                rows.append({"set": label, "metric": k, "band": band, "side": "good-bad", "estimate": cd[band][0],
                             "ci_low": cd[band][1], "ci_high": cd[band][2]})
            for band in BANDS:
                trs.append([names[k], band, f"{cg[band][0]:.2f}", f"{cb[band][0]:.2f}", f3(cd[band])])
            trs.append([names[k], "XTail - Head", "", "", f3(cd["xtail-head"])])
        lines += table(["Next token", "Verb band", "Good", "Bad", "Good - bad"], trs) + [""]
    pd.DataFrame(rows).to_csv(Path(args.out_dir) / "task1_baseline.csv", index=False)

    # top-1 and pooled top-10
    lines += ["## Most probable next tokens", "",
              "Mean probability over items from each prompt's stored top-10 (a token absent from a "
              "prompt's top-10 counts as 0, so values are lower bounds).", ""]
    trs = []
    for label, df in (("passive", pas), ("active", act)):
        for band in BANDS:
            sub = df[df.verb_band == band]
            for side in ("good", "bad"):
                top = topk_mass(sub[f"{side}_topk"].tolist())[:8]
                trs.append([label, band, side, ", ".join(f"`{show(t)}` {p:.3f}" for t, p in top)])
    lines += table(["Set", "Verb band", "Verb", "Top tokens (mean p)"], trs) + [""]

    lines += ["## Sample prompts (top-10)", ""]
    for band in BANDS:
        vp = sorted(pas[pas.verb_band == band].verb_pair.unique())
        for pair in [vp[i] for i in rng.choice(len(vp), 2, replace=False)]:
            cand = pas[pas.verb_pair == pair].sort_values("context_id")
            r = cand.iloc[rng.integers(len(cand))]
            a = act[(act.verb_pair == pair) & (act.context_id.str.split("/").str[-1] == r.context_id.split("/")[-1])]
            a = a.iloc[0] if len(a) else act[act.verb_pair == pair].sort_values("context_id").iloc[0]
            for prompt, tk in ((r.good_prompt, r.good_topk), (r.bad_prompt, r.bad_topk),
                               (a.good_prompt, a.good_topk), (a.bad_prompt, a.bad_topk)):
                lines.append(f"- `{prompt}` → " + ", ".join(f"`{show(t)}` {np.exp(v):.3f}"
                                                            for t, v in json.loads(tk)))
            lines.append("")
    (Path(args.out_dir) / "task1_baseline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pas, act


# ---------------------------------------------------------------- task 2 + 3

def released_items(scores, ro):
    rel = scores[(scores.context_set == "released") & (scores.paradigm == "passive_1")].copy()
    rel["good_prompt"], rel["bad_prompt"] = rel.prefix + rel.good_verb, rel.prefix + rel.bad_verb
    cols = ["lp[ by]", "lp[.]", "lp[ the]", "topk"]
    attach(rel, ro, "released", "good_prompt", "good", cols)
    attach(rel, ro, "released", "bad_prompt", "bad", cols)
    return rel


def task2(rel, reliability, args):
    per = rel.groupby(["verb_band", "verb_pair"]).agg(
        good_by_lp=("good_by_lp", "mean"), bad_by_lp=("bad_by_lp", "mean"), by_margin=("by_margin", "mean"),
        good_dot=("good_lp.", "mean"), bad_dot=("bad_lp.", "mean")).reset_index()
    per = per.merge(reliability[["verb_pair", "ci_low", "ci_high", "by_class"]], on="verb_pair")
    ref = per[per.by_class == "reliable_pos"]
    for side in ("good", "bad"):
        c = f"{side}_by_lp"
        per[f"{side}_z"] = (per[c] - ref[c].mean()) / ref[c].std()
        band_ref = ref.groupby("verb_band")[c].agg(["mean", "std"])
        per[f"{side}_z_band"] = (per[c] - per.verb_band.map(band_ref["mean"])) / per.verb_band.map(band_ref["std"])
        per[f"{side}_dev"] = per[c] - ref[c].mean()
    # Margin shortfall vs mean reliable pair = good_dev - bad_dev.
    per["shortfall"] = per.by_margin - ref.by_margin.mean()
    per["driver"] = np.where(per.by_margin >= 0, "",
                             np.where(-per.good_dev > per.bad_dev, "good side low", "bad side high"))
    tops = {}
    for pair, g in rel.groupby("verb_pair"):
        tops[pair] = {s: ", ".join(f"`{show(t)}` {p:.2f}" for t, p in topk_mass(g[f"{s}_topk"].tolist())[:5])
                      for s in ("good", "bad")}
    per["top_good"] = per.verb_pair.map(lambda v: tops[v]["good"])
    per["top_bad"] = per.verb_pair.map(lambda v: tops[v]["bad"])
    per.drop(columns=["top_good", "top_bad"]).to_csv(Path(args.out_dir) / "task2_by_split_per_verb.csv",
                                                     index=False)

    lines = ["# Task 2: splitting the *by* margin", "",
             "Released `passive_1` contexts (300 per verb pair; mixed auxiliaries: was/were/is/are and "
             "negations). log P(by | prefix) is from the full-sentence scores (`good_by_lp`, `bad_by_lp`). "
             "z-scores are against the per-pair means of the reliable-positive pairs (all bands pooled; "
             "`_band` columns use the same band only). Reliable = 95% CI over contexts excludes 0.", "",
             "## Band means of per-pair log P(by)", ""]
    trs = []
    for band in BANDS:
        for cls, g in per[per.verb_band == band].groupby("by_class"):
            trs.append([band, cls, len(g), f"{g.good_by_lp.mean():.2f}", f"{g.bad_by_lp.mean():.2f}",
                        f"{g.by_margin.mean():.2f}"])
    lines += table(["Verb band", "Class", "n", "log P(by \\| good)", "log P(by \\| bad)", "Margin"], trs)
    lines += ["", f"Reliable-positive reference (n = {len(ref)}): good {ref.good_by_lp.mean():.2f} "
              f"(sd {ref.good_by_lp.std():.2f}), bad {ref.bad_by_lp.mean():.2f} (sd {ref.bad_by_lp.std():.2f}), "
              f"margin {ref.by_margin.mean():.2f}.", "",
              "## Pairs with a negative *by* margin", "",
              "`Good dev` and `Bad dev` are the deviations from the reliable mean; the margin shortfall is "
              "good dev - bad dev. `Driver` names the larger contributor.", ""]
    neg = per[per.by_margin < 0].sort_values(["verb_band", "by_margin"])
    trs = [[r.verb_band, r.verb_pair.split("/", 1)[1], f"{r.by_margin:.2f}", f"{r.good_by_lp:.2f}",
            f"{r.bad_by_lp:.2f}", f"{r.good_dev:+.2f}", f"{r.bad_dev:+.2f}", f"{r.good_z:+.1f}",
            f"{r.bad_z:+.1f}", r.driver] for r in neg.itertuples()]
    lines += table(["Band", "Pair", "Margin", "lp by good", "lp by bad", "Good dev", "Bad dev", "z good",
                    "z bad", "Driver"], trs)
    counts = neg.groupby(["verb_band", "driver"]).size().unstack(fill_value=0)
    lines += ["", "Driver counts: " + "; ".join(
        f"{b}: " + ", ".join(f"{k} {v}" for k, v in counts.loc[b].items()) for b in counts.index), "",
        "## Top next tokens after each prefix (negative pairs)", "",
        "Mean probability from the stored top-10, averaged over the 300 released contexts.", ""]
    trs = [[r.verb_band, r.verb_pair.split("/", 1)[1], r.top_good, r.top_bad] for r in neg.itertuples()]
    lines += table(["Band", "Pair", "After good participle", "After bad participle"], trs)
    ref_tops = {s: ", ".join(f"`{show(t)}` {p:.2f}" for t, p in topk_mass(
        rel[rel.verb_pair.isin(ref.verb_pair)][f"{s}_topk"].tolist())[:6]) for s in ("good", "bad")}
    lines += ["", f"Reliable-positive pairs, pooled: good → {ref_tops['good']}; bad → {ref_tops['bad']}", ""]
    (Path(args.out_dir) / "task2_by_split.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return per


def task3(rel, cur_p1, args, rng):
    rel = rel.copy()
    rel["lo_good"] = rel["good_lp_by"] - rel["good_lp."]
    rel["lo_bad"] = rel["bad_lp_by"] - rel["bad_lp."]
    rel["lo_diff"] = rel.lo_good - rel.lo_bad
    check = np.abs(rel.good_lp_by - rel.good_by_lp)
    boot = Boot(rel, args.n_boot, rng)
    lines = ["# Task 3: single-prompt readout", "",
             "log-odds(\" by\" vs \".\") after the participle within one prompt, against the two-sentence "
             "*by* margin log P(by | good) - log P(by | bad). Released `passive_1` contexts "
             f"({len(rel):,} items). Item-level CIs: two-way cluster bootstrap ({args.n_spear} draws); "
             f"verb-pair-level CIs: bootstrap over pairs ({args.n_boot} draws).", "",
             f"Sanity: readout log P(by | good) vs full-sentence `good_by_lp`: median |diff| "
             f"{check.median():.3f}, 99th pct {check.quantile(.99):.3f} (bf16, different batching).", "",
             "## Raw log-probs after the participle (band means, item level)", ""]
    trs = []
    for col, name in (("good_lp_by", 'log P(" by" \\| good)'), ("good_lp.", 'log P("." \\| good)'),
                      ("lo_good", "log-odds by vs . (good)"), ("bad_lp_by", 'log P(" by" \\| bad)'),
                      ("bad_lp.", 'log P("." \\| bad)'), ("lo_bad", "log-odds by vs . (bad)"),
                      ("by_margin", "two-sentence by margin")):
        c = boot.cells(rel[col].to_numpy())
        trs.append([name] + [f3(c[b]) for b in BANDS + ("xtail-head",)])
    lines += table(["Quantity", "Head", "Tail", "XTail", "XTail - Head"], trs)
    rows = []
    per = rel.groupby(["verb_band", "verb_pair"])[["lo_good", "lo_bad", "lo_diff", "by_margin", "good_lp_by",
                                                   "good_lp."]].mean().reset_index()
    pairs = (("lo_good", "log-odds (good prompt)"), ("lo_bad", "log-odds (bad prompt)"),
             ("lo_diff", "log-odds good - bad"), ("good_lp_by", 'log P(" by" \\| good) alone'),
             ("good_lp.", 'log P("." \\| good) alone'))
    for col, name in pairs:
        it = boot.spearman(rel[col], rel.by_margin, args.n_spear)
        vp = boot_spearman_units(per[col], per.by_margin, args.n_boot, rng)
        rows.append({"readout": col, "level": "item", "rho": it[0], "ci_low": it[1], "ci_high": it[2]})
        rows.append({"readout": col, "level": "verb_pair", "rho": vp[0], "ci_low": vp[1], "ci_high": vp[2]})
        for band in BANDS:
            pb = per[per.verb_band == band]
            vb = boot_spearman_units(pb[col], pb.by_margin, args.n_boot, rng)
            rows.append({"readout": col, "level": f"verb_pair_{band}", "rho": vb[0], "ci_low": vb[1],
                         "ci_high": vb[2]})
    rho = pd.DataFrame(rows)
    rho.to_csv(Path(args.out_dir) / "task3_single_prompt.csv", index=False)
    lines += ["", "## Spearman with the two-sentence *by* margin", ""]
    trs = []
    for col, name in pairs:
        g = rho[rho.readout == col].set_index("level")
        trs.append([name] + [f3(tuple(g.loc[l, ["rho", "ci_low", "ci_high"]])) for l in
                             ("item", "verb_pair", "verb_pair_head", "verb_pair_tail", "verb_pair_xtail")])
    lines += table(["Readout", "Item", "Verb pair (all)", "Pair: Head", "Pair: Tail", "Pair: XTail"], trs)
    if cur_p1 is not None:
        c = cur_p1.copy()
        c["lo_good"] = c["good_lp_by"] - c["good_lp."]
        c["lo_diff"] = c.lo_good - (c["bad_lp_by"] - c["bad_lp."])
        cb = Boot(c, args.n_boot, rng)
        lines += ["", f"Curated `passive_1`, other contexts ({len(c):,} items), item-level Spearman with the "
                  f"curated *by* margin: log-odds (good) {f3(cb.spearman(c.lo_good, c.by_margin, args.n_spear))}; "
                  f"good - bad {f3(cb.spearman(c.lo_diff, c.by_margin, args.n_spear))}."]
    (Path(args.out_dir) / "task3_single_prompt.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- task 4

def task4(scores, ro, args, rng):
    _, good = mean_rating(args.good_ratings)
    _, badr = mean_rating(args.bad_ratings)
    gl = pd.read_csv(args.good_links, dtype={"item_id": str})[["pair_id", "item_id"]]
    bl = pd.read_csv(args.bad_links, dtype={"item_id": str})
    bad = bl[bl.role == "patient"].merge(badr[["item_id", "rating"]], on="item_id") \
        .set_index("pair_id").rating.rename("bad_patient")
    cur = scores[scores.context_set == "curated"].merge(gl, on="pair_id") \
        .merge(good[["item_id", "rating"]].rename(columns={"rating": "good_fit"}), on="item_id") \
        .merge(bad, left_on="pair_id", right_index=True)
    for side in ("good", "bad"):
        ids = cur[f"sentence_{side}"].map(lambda s: pid("yesno", YES_NO_BASE_PROMPT.format(sentence=s)))
        sub = ro.reindex(ids)
        if sub.lp_yes.isna().any():
            raise SystemExit("missing yes/no readout rows")
        cur[f"{side}_lyes"], cur[f"{side}_lno"] = sub.lp_yes.to_numpy(), sub.lp_no.to_numpy()
        cur[f"{side}_yn"] = cur[f"{side}_lyes"] - cur[f"{side}_lno"]  # logit of P(yes)/(P(yes)+P(no))
    cur["yn_margin"] = cur.good_yn - cur.bad_yn
    cur["yn_correct"] = (cur.yn_margin > 0) + 0.5 * (cur.yn_margin == 0)
    cur["yn_mass"] = np.exp(np.logaddexp(cur.good_lyes, cur.good_lno))

    lines = ["# Task 4: Yes/No readout", "",
             "Prompt: FreqBLiMP base-model Yes/No prompt (blimp-rare `YES_NO_BASE_PROMPT`); score = "
             "P(Yes)/(P(Yes)+P(No)), Yes = {\"Yes\", \" Yes\"}, No = {\"No\", \" No\"}; a pair is correct if "
             "the good sentence scores higher. `yn_margin` = logit(score good) - logit(score bad). "
             f"Curated cross, both paradigms. 95% CIs: two-way cluster bootstrap ({args.n_boot} draws).", ""]
    rows, trs = [], []
    for paradigm, sub in cur.groupby("paradigm"):
        for own, s in sub.groupby("own_context"):
            b = Boot(s, args.n_boot, rng)
            for metric in ("yn_correct", "correct", "yn_margin", "whole_margin"):
                y = s[metric].to_numpy(float) * (100 if metric in ("yn_correct", "correct") else 1)
                c = b.cells(y)
                for band, v in c.items():
                    rows.append({"paradigm": paradigm, "own_context": own, "metric": metric, "band": band,
                                 "estimate": v[0], "ci_low": v[1], "ci_high": v[2]})
            ctx = "own" if own else "other"
            r = pd.DataFrame(rows)
            r = r[(r.paradigm == paradigm) & (r.own_context == own)]
            get = lambda m, band: f3(tuple(r[(r.metric == m) & (r.band == band)][["estimate", "ci_low", "ci_high"]].iloc[0]), 1)
            for band in BANDS + ("xtail-head",):
                trs.append([paradigm, ctx, band, get("yn_correct", band), get("correct", band),
                            get("yn_margin", band)])
    pd.DataFrame(rows).to_csv(Path(args.out_dir) / "task4_yesno_cells.csv", index=False)
    lines += ["## Accuracy (%) by verb band", "", "Head rows first answer the gating question "
              "(chance = 50%).", ""]
    lines += table(["Paradigm", "Contexts", "Verb band", "Yes/No acc", "LP acc", "Yes/No margin"], trs)
    lines += ["", f"Yes+No probability mass on good prompts: median {cur.yn_mass.median():.3f} "
              f"(10th pct {cur.yn_mass.quantile(.1):.3f}). Mean P(Yes) share, good vs bad: "
              f"{(1 / (1 + np.exp(-cur.good_yn))).mean():.3f} vs {(1 / (1 + np.exp(-cur.bad_yn))).mean():.3f}. "
              f"Spearman(yn_margin, whole_margin) over items: {spear(cur.yn_margin, cur.whole_margin):.3f}.", ""]

    # Fit models: same spec as analyze_bad_fit.py `good+bad_patient`.
    mrows = []
    for paradigm, sub in cur.groupby("paradigm"):
        verbs = sub.drop_duplicates("verb_pair").sort_values("verb_pair")
        ctxs = sub.drop_duplicates("context_id").sort_values("context_id")
        vi = sub.verb_pair.map({v: i for i, v in enumerate(verbs.verb_pair)}).to_numpy()
        ci = sub.context_id.map({c: i for i, c in enumerate(ctxs.context_id)}).to_numpy()
        wv, wc = weights(rng, verbs.verb_band.to_numpy(), args.n_boot), weights(rng, ctxs.context_band.to_numpy(), args.n_boot)
        t, x = (sub.verb_band == "tail").to_numpy(float), (sub.verb_band == "xtail").to_numpy(float)
        X = np.column_stack([np.ones(len(sub)), sub.good_fit - 4, sub.bad_patient - 4, t, x])
        names = ["intercept", "good_fit", "bad_patient", "tail", "xtail"]
        for metric in ("yn_correct", "correct", "yn_margin", "whole_margin"):
            y = sub[metric].to_numpy(float) * (100 if metric in ("yn_correct", "correct") else 1)
            d = fit(sub, y, X, vi, ci, wv, wc)
            for j, n in enumerate(names):
                mrows.append({"paradigm": paradigm, "metric": metric, "term": n, "estimate": d[0, j],
                              "ci_low": np.percentile(d[1:, j], 2.5), "ci_high": np.percentile(d[1:, j], 97.5)})
    m = pd.DataFrame(mrows)
    m.to_csv(Path(args.out_dir) / "task4_yesno_fit_models.csv", index=False)
    lines += ["## Fit effects: Yes/No vs LP", "",
              "WLS `y ~ good_fit + bad_patient + band` (the `good+bad_patient` spec of "
              "`analyze_bad_fit.py`), all curated items (own included, as there). Fit per rating point; "
              "accuracy in pp.", ""]
    trs = []
    for paradigm in sorted(m.paradigm.unique()):
        for term in ("good_fit", "bad_patient", "xtail"):
            g = m[(m.paradigm == paradigm) & (m.term == term)].set_index("metric")
            trs.append([paradigm, term] + [f3(tuple(g.loc[k, ["estimate", "ci_low", "ci_high"]]))
                                           for k in ("yn_correct", "correct", "yn_margin", "whole_margin")])
    lines += table(["Paradigm", "Term", "Yes/No acc", "LP acc", "Yes/No margin", "LP whole margin"], trs)
    (Path(args.out_dir) / "task4_yesno.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    scores = pd.read_csv(args.scores, low_memory=False)
    v2 = pd.DataFrame([json.loads(l) for l in Path(args.v2_pairs).open(encoding="utf-8")])
    v2 = v2[v2.frame_id == "was"]
    ro = load_readout(args.readout)
    reliability = pd.read_csv(Path(args.out_dir) / "task0_by_reliability.csv")
    tasks = set(args.tasks.split(","))
    if "1" in tasks:
        task1(scores, v2, ro, args, rng)
        print("task 1 done", flush=True)
    if tasks & {"2", "3"}:
        rel = released_items(scores, ro)
        if "2" in tasks:
            task2(rel, reliability, args)
            print("task 2 done", flush=True)
        if "3" in tasks:
            c = curated_items(scores, v2, "passive_1")
            c = c[c.own_context == 0].copy()
            c["good_prompt"], c["bad_prompt"] = c.prefix + c.good_verb, c.prefix + c.bad_verb
            attach(c, ro, "passive", "good_prompt", "good", ["lp[ by]", "lp[.]"])
            attach(c, ro, "passive", "bad_prompt", "bad", ["lp[ by]", "lp[.]"])
            task3(rel, c, args, rng)
            print("task 3 done", flush=True)
    if "4" in tasks:
        task4(scores, ro, args, rng)
        print("task 4 done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", default="results/passive_band_cross/pythia14b_scores.csv")
    ap.add_argument("--readout", default="results/passive_das_prep/pythia14b_readout.csv")
    ap.add_argument("--v2-pairs", default="data/matched_passives_v2/pairs.jsonl")
    ap.add_argument("--good-ratings", default="results/fit_ratings/gemma4_31b_it.csv")
    ap.add_argument("--good-links", default="data/fit_ratings/item_links.csv")
    ap.add_argument("--bad-ratings", default="results/fit_ratings/bad_gemma4_31b_it.csv")
    ap.add_argument("--bad-links", default="data/fit_ratings/bad_item_links.csv")
    ap.add_argument("--out-dir", default="reports/passive_das_prep")
    ap.add_argument("--tasks", default="1,2,3,4")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-spear", type=int, default=200)
    ap.add_argument("--seed", type=int, default=17)
    run(ap.parse_args())
