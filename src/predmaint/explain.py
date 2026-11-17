"""Class-specific explainability with SHAP."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import config as C

FAILURE_CLASSES = [1, 2, 3, 4]
COLORS = {1: "#DD8452", 2: "#55A868", 3: "#C44E52", 4: "#8172B3"}


def shap_by_class(model, X: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per feature for each failure class (features x classes)."""
    values = shap.TreeExplainer(model).shap_values(X)
    if isinstance(values, list):
        values = np.stack(values, axis=-1)
    return pd.DataFrame({C.CLASS_NAMES[c]: np.abs(values[:, :, c]).mean(axis=0)
                         for c in FAILURE_CLASSES}, index=X.columns)


def plot_shap_by_class(importance: pd.DataFrame, path: Path | str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, c in zip(axes.ravel(), FAILURE_CLASSES):
        s = importance[C.CLASS_NAMES[c]].sort_values()
        ax.barh(s.index, s.values, color=COLORS[c])
        ax.set_title(f"{C.CLASS_NAMES[c]} - {C.CLASS_DESCRIPTIONS[c]}")
        ax.set_xlabel("mean |SHAP value| (log-odds)")
    fig.suptitle("What drives each failure mode? (tuned XGBoost)", fontsize=14)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
