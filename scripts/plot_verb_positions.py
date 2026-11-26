"""Draft figure: verbs placed along the frequent-trained DAS direction (stored v) at Pythia site 8.

Input: results/round6/classify/restore_zfit.csv (z = projection along v of the participle's last token in
"She/He/They has V-ed", averaged over the three frames, scaled per fold so that the DAS training I mean is 0 and T mean
is 1; training verbs use the fold that held them out). Exploratory: no rule declared, nothing here is a test.

Frequency: --freq zipf (participle Zipf) or --freq count (log10 Pile count of the word-initial lowercase participle,
infini-gram v4_piletrain_llama; on the 608 round-6 verbs it is a median 0.978 of c_total and log r = 0.9995 with it;
round-6 counts plus data/verb_positions/counts for the remaining verbs).

Output: reports/figures/verb_positions_v_site8[_count].{png,pdf}
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

T_COL, I_COL = "#2a78d6", "#eb6834"  # reference palette slots 1, 2 (validated pair)
INK, MUTED = "#0b0b0b", "#52514e"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zfit", default="results/round6/classify/restore_zfit.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--freq", choices=("zipf", "count"), default="count")
    ap.add_argument("--counts", nargs="*", default=["data/round6/counts/pile_counts.csv",
                                                    "data/verb_positions/counts/pile_counts.csv"])
    ap.add_argument("--label", nargs="*", default=[
        "existed", "died", "vanished", "happened", "arrived", "laughed", "complained",
        "responded", "replied", "remarked", "inquired", "opted", "mutated",
        "appointed", "enclosed", "lasted", "starred", "tended"])
    args = ap.parse_args()
    d = pd.read_csv(args.zfit)
    d["rare"] = d.bandkey.eq("tail")
    if args.freq == "count":
        C = pd.concat([pd.read_csv(f) for f in args.counts if Path(f).exists()]
                      + [pd.read_json(f, lines=True) for f in sorted(Path("data/verb_positions/counts").glob("*.jsonl"))])
        C = C[C["count"].notna()].drop_duplicates(["verb", "slot"])
        W = C.pivot(index="verb", columns="slot", values="count")
        tot = W.overall  # word-initial lowercase; tracks c_total at log r = 0.9995 on the 608 round-6 verbs
        miss = sorted(set(d.verb) - set(tot.dropna().index))
        assert not miss, f"no count for {len(miss)} verbs: {miss[:10]}"
        d["y"] = np.log10(d.verb.map(tot) + 0.5)
        ylab, jit = "participle count in the Pile (log10)", 0.0
    else:
        d["y"], ylab, jit = d.zipf, "participle frequency (Zipf)", 0.015
    out = Path(args.out or f"reports/figures/verb_positions_v_site8{'_count' if args.freq == 'count' else ''}")

    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=200)
    rng = np.random.default_rng(0)
    for role, col, name in (("T", T_COL, "transitive"), ("I", I_COL, "intransitive")):
        for rare in ((False, True) if args.freq == "zipf" else (None,)):
            x = d[(d.role == role) & ((d.rare == rare) if rare is not None else True)]
            ax.scatter(x.z, x.y + rng.uniform(-jit, jit, len(x)) if jit else x.y, s=22 if role == "I" else 14,
                       facecolor="white" if rare is True else col, edgecolor=col, linewidth=1.1,
                       alpha=1.0 if role == "I" else 0.55, zorder=3 if role == "I" else 2,
                       label=(f"{name} (n={len(x)})" if rare is None else
                              f"{name}, {'rare (Tail)' if rare else 'frequent (Head)'} (n={len(x)})"))
    for v in (0, 1):
        ax.axvline(v, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    lo, hi = d.y.min(), d.y.max()
    pad = 0.06 * (hi - lo)
    ax.set_ylim(lo - pad, hi + 2.2 * pad)
    for v, t in ((0, "training I mean"), (1, "training T mean")):
        ax.text(v, hi + 1.3 * pad, t, ha="center", va="bottom", fontsize=7, color=MUTED)
    lab = d[d.verb.isin(args.label)]
    for _, r in lab.iterrows():
        left = r.verb in ("died",)
        ax.annotate(r.verb, (r.z, r.y), xytext=(-4 if left else 4, 3), textcoords="offset points", fontsize=6.5,
                    color=INK, ha="right" if left else "left")
    ax.set_xlabel("position along frequent-trained class direction (site 8, active frame)", fontsize=8, color=INK)
    ax.set_ylabel(ylab, fontsize=8, color=INK)
    ax.tick_params(labelsize=7, colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, handletextpad=0.2, columnspacing=1.0)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"))
    fig.savefig(out.with_suffix(".pdf"))
    print(out.with_suffix(".png"))


if __name__ == "__main__":
    main()
