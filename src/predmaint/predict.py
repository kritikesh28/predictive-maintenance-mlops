"""Inference helpers: score raw sensor readings with the saved production model."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from . import config as C
from .features import add_features, strain_index

DEFAULT_MODEL = C.MODELS_DIR / "xgb_predmaint.joblib"
STRAIN_ALERT = 10_000   # min*Nm; every overstrain failure in the baseline sits above ~11,000


def load_model(path: Path | str = DEFAULT_MODEL):
    return joblib.load(path)


def predict(model, readings: pd.DataFrame | list[dict]) -> pd.DataFrame:
    """Return predicted failure class, class probabilities and an overstrain flag."""
    df = pd.DataFrame(readings)
    missing = set(C.RAW_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    feats = add_features(df)
    proba = model.predict_proba(feats[C.FEATURES])
    out = pd.DataFrame(proba, columns=[f"p_{C.CLASS_NAMES[c]}" for c in C.CLASSES],
                       index=df.index)
    out.insert(0, "prediction", [C.CLASS_NAMES[i] for i in proba.argmax(axis=1)])
    out["strain_index"] = strain_index(df)
    out["strain_alert"] = out["strain_index"] > STRAIN_ALERT
    return out
