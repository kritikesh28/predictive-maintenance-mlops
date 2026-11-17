"""Data loading and contract validation (Pandera)."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

from . import config as C

# Data contract: physical ranges every sensor batch must respect.
SENSOR_SCHEMA = DataFrameSchema(
    {
        "Type": Column(str, Check.isin(["L", "M", "H"])),
        "Air temperature": Column(float, Check.in_range(295.0, 305.0)),
        "Process temperature": Column(float, Check.in_range(305.0, 315.0)),
        "Rotational speed": Column("int64", Check.in_range(1000, 2900)),
        "Torque": Column(float, Check.in_range(3.0, 80.0)),
        "Tool wear": Column("int64", Check.in_range(0, 253)),
        "Failure_Type": Column("int64", Check.isin(C.CLASSES)),
    },
    strict=True,
)


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("Rotational speed", "Tool wear", C.TARGET):
        df[col] = df[col].astype("int64")
    return df


def load_csv(name: str, data_dir: Path | str = C.DATA_DIR) -> pd.DataFrame:
    return fix_dtypes(pd.read_csv(Path(data_dir) / name))


def load_batches(data_dir: Path | str = C.DATA_DIR) -> dict[str, pd.DataFrame]:
    return {
        "baseline": load_csv(C.BASELINE_CSV, data_dir),
        "stable": load_csv(C.STABLE_CSV, data_dir),
        "heavy_load": load_csv(C.HEAVY_LOAD_CSV, data_dir),
    }


def validate(df: pd.DataFrame, name: str = "batch") -> dict:
    """Validate a batch lazily and return a summary instead of raising."""
    try:
        SENSOR_SCHEMA.validate(df, lazy=True)
        return {"batch": name, "rows": len(df), "valid": True, "violations": 0}
    except pa.errors.SchemaErrors as err:
        return {"batch": name, "rows": len(df), "valid": False,
                "violations": int(len(err.failure_cases)),
                "details": err.failure_cases.groupby(["column", "check"]).size().to_dict()}
