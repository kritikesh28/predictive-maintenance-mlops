"""Physics-informed feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add machine-type encoding plus two physically meaningful features.

    Power_W   = Torque [Nm] x angular velocity [rad/s]   (mechanical power)
    Temp_diff = Process temperature - Air temperature    (heat-dissipation headroom)
    """
    df = df.copy()
    unknown = set(df["Type"]) - set(C.TYPE_MAP)
    if unknown:
        raise ValueError(f"Unknown machine Type values: {unknown}")
    df["Type_enc"] = df["Type"].map(C.TYPE_MAP).astype("int64")
    df["Power_W"] = df["Torque"] * df["Rotational speed"] * 2 * np.pi / 60
    df["Temp_diff"] = df["Process temperature"] - df["Air temperature"]
    return df


def strain_index(df: pd.DataFrame) -> pd.Series:
    """Tool wear x Torque [min*Nm]: the overstrain indicator used for alerting."""
    return df["Tool wear"] * df["Torque"]
