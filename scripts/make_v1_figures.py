#!/usr/bin/env python3
"""Make v1 summary figures from committed result CSVs."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


FIG_DIR = Path("reports/figures")


def weighted_mean(df: pd.DataFrame, value: str) -> float:
    return float((df[value] * df["n"]).sum() / df["n"].sum())


def savefig(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        plt.savefig(FIG_DIR / f"{name}.{suffix}", bbox_inches="tight", dpi=300)
    plt.close()


def make_localization() -> None:
    df = pd.read_csv("results/attribution_patching/20260617-pythia14b-ap-primary.summary.csv")
    df = df[
        (df["regime"] == "head")
        & (df["subtask"].isin(["causative", "inchoative"]))
        & (df["site"].str.startswith("resid_post_layer_"))
    ].copy()
    df["layer"] = df["site"].map(lambda s: int(re.search(r"(\d+)$", s).group(1)))

    rows = []
    for layer, layer_df in df.groupby("layer"):
        rows.append(
            {
                "layer": int(layer),
                "mean_attribution": weighted_mean(layer_df, "mean_attribution"),
                "mean_abs_attribution": weighted_mean(layer_df, "mean_abs_attribution"),
            }
        )
    out = pd.DataFrame(rows).sort_values("layer")
    out.to_csv(FIG_DIR / "v1_localization_head_primary.csv", index=False)

    plt.figure(figsize=(7.2, 3.8))
    ax = plt.gca()
    ax.plot(out["layer"], out["mean_attribution"], color="#1f5a85", linewidth=2.2)
    markerline, stemlines, baseline = ax.stem(
        out["layer"],
        out["mean_attribution"],
        linefmt="#8eb5d1",
        markerfmt="o",
        basefmt=" ",
    )
    plt.setp(markerline, markersize=4.5, markerfacecolor="#1f5a85", markeredgewidth=0)
    plt.setp(stemlines, linewidth=1.0, alpha=0.65)
    ax.axvspan(16, 23, color="#d9e8f2", alpha=0.55, linewidth=0)
    peak = out.loc[out["mean_attribution"].idxmax()]
    ax.scatter([23], [float(out[out["layer"] == 23]["mean_attribution"].iloc[0])], s=70, color="#b13f2e", zorder=5)
    ax.annotate(
        "layer 23",
        xy=(23, float(out[out["layer"] == 23]["mean_attribution"].iloc[0])),
        xytext=(20.2, float(peak["mean_attribution"]) + 0.35),
        arrowprops={"arrowstyle": "->", "color": "#7a2c22", "lw": 1.1},
        fontsize=9,
        color="#7a2c22",
    )
    ax.text(16.15, ax.get_ylim()[0] + 0.18, "late residual band", fontsize=9, color="#496b80")
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Pythia 1.4B layer")
    ax.set_ylabel("Mean attribution patching effect")
    ax.set_title("Localization: head causative + inchoative")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    savefig("v1_localization_head_primary")


def make_synthesis() -> None:
    regimes = ["head", "tail", "xtail"]

    behavioral = {
        "head": (0.798 + 0.844) / 2,
        "tail": (0.774 + 0.813) / 2,
        "xtail": (0.699 + 0.740) / 2,
    }

    exact = pd.read_csv("results/exact_patching/20260617-pythia14b-ap-primary-exact.summary.csv")
    exact = exact[
        (exact["subtask"].isin(["causative", "inchoative"]))
        & (exact["site"] == "resid_post_layer_23")
    ]
    patching = {}
    for regime in regimes:
        patching[regime] = weighted_mean(exact[exact["regime"] == regime], "mean_exact_effect")

    transfer = pd.read_csv("results/das_v1_transfer/20260618-pythia14b-das-transfer-head-l23.eval_detail.csv")
    transfer_success = {}
    for regime in regimes:
        sub = transfer[transfer["regime"] == regime]
        transfer_success[regime] = float(sub["patched_success"].mean())

    raw = pd.DataFrame(
        {
            "regime": regimes,
            "behavioral_acc": [behavioral[r] for r in regimes],
            "patching_effect": [patching[r] for r in regimes],
            "transfer_success": [transfer_success[r] for r in regimes],
        }
    )
    norm = raw.copy()
    for col in ["behavioral_acc", "patching_effect", "transfer_success"]:
        norm[col] = norm[col] / float(norm.loc[norm["regime"] == "head", col].iloc[0])
    raw.to_csv(FIG_DIR / "v1_synthesis_raw.csv", index=False)
    norm.to_csv(FIG_DIR / "v1_synthesis_head_normalized.csv", index=False)

    x = range(len(regimes))
    plt.figure(figsize=(6.8, 3.9))
    ax = plt.gca()
    series = [
        ("behavioral_acc", "Behavioral acc.", "#2a6f4e", "o"),
        ("patching_effect", "Exact patching effect", "#1f5a85", "s"),
        ("transfer_success", "Head-trained DAS success", "#b13f2e", "^"),
    ]
    for col, label, color, marker in series:
        ax.plot(x, norm[col], marker=marker, linewidth=2.2, markersize=6, color=color, label=label)
    ax.axhline(1.0, color="#666666", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(list(x), regimes)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Head-normalized value")
    ax.set_title("Synthesis across frequency regimes")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower left")
    savefig("v1_synthesis_head_normalized")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    make_localization()
    make_synthesis()
    print(FIG_DIR)


if __name__ == "__main__":
    main()
