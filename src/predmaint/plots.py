"""Report figures used in the README."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config as C

ORDER = ["TWF", "HDF", "PWF", "OSF"]


def plot_eda(df: pd.DataFrame, path: Path | str) -> None:
    counts = df[C.TARGET].map(C.CLASS_NAMES).value_counts().reindex(list(C.CLASS_NAMES.values()))
    fails = df[df[C.TARGET] > 0].assign(Failure=lambda d: d[C.TARGET].map(C.CLASS_NAMES))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    ax.bar(counts.index, counts.values, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"])
    ax.set_yscale("log")
    ax.set_title("Class balance (log scale): 96.7% healthy")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}", ha="center", va="bottom")
    sns.histplot(fails, x="Torque", hue="Failure", hue_order=ORDER, bins=25, element="step", ax=axes[1])
    axes[1].set_title("Torque by failure mode")
    sns.histplot(fails, x="Tool wear", hue="Failure", hue_order=ORDER, bins=25, element="step", ax=axes[2])
    axes[2].set_title("Tool wear by failure mode")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_drift(drift: dict, path: Path | str) -> None:
    rows = []
    for batch, d in drift.items():
        for f in d["features"]:
            rows.append({"feature": f["feature"], "batch": batch, "wasserstein": f["wasserstein"]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(df, y="feature", x="wasserstein", hue="batch",
                palette={"stable": "#4C72B0", "heavy_load": "#C44E52"}, ax=ax)
    ax.axvline(0.1, ls="--", c="grey")
    ax.text(0.105, 2.5, "drift threshold 0.1", color="grey")
    ax.set_title("Feature drift vs baseline (normed Wasserstein distance)")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_leaderboard(board: pd.DataFrame, path: Path | str) -> None:
    df = board[["macro_f1", "accuracy"]].reset_index(names="model").melt("model")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(df, x="model", y="value", hue="variable", palette=["#55A868", "#BBBBBB"], ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy hides the problem; macro-F1 exposes it")
    ax.legend(loc="lower right", title="")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
