"""Production drift monitoring with Evidently."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from evidently.legacy.metrics.data_drift.column_drift_metric import ColumnDriftMetric
from evidently.legacy.metrics.data_drift.dataset_drift_metric import DatasetDriftMetric
from evidently.legacy.report import Report

from . import config as C


def drift_report(reference: pd.DataFrame, batch: pd.DataFrame, html_path: Path | str,
                 cols: list[str] = C.SENSOR_COLS) -> tuple[dict, pd.DataFrame]:
    """Dataset-level + per-feature Wasserstein drift. Saves an interactive HTML report."""
    report = Report(metrics=[DatasetDriftMetric(stattest="wasserstein")]
                    + [ColumnDriftMetric(column_name=c, stattest="wasserstein") for c in cols])
    report.run(reference_data=reference[cols], current_data=batch[cols])
    Path(html_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(html_path))

    res = report.as_dict()["metrics"]
    ds = res[0]["result"]
    summary = {
        "dataset_drift": bool(ds["dataset_drift"]),
        "drifted_features": int(ds["number_of_drifted_columns"]),
        "n_features": int(ds["number_of_columns"]),
    }
    rows = []
    for m in res[1:]:
        r = m["result"]
        col = r["column_name"]
        ref_mean, cur_mean = reference[col].mean(), batch[col].mean()
        rows.append({
            "feature": col,
            "drift": bool(r["drift_detected"]),
            "wasserstein": round(float(r["drift_score"]), 4),
            "ref_mean": round(float(ref_mean), 2),
            "batch_mean": round(float(cur_mean), 2),
            "delta_pct": round(float((cur_mean - ref_mean) / ref_mean * 100), 2),
        })
    table = pd.DataFrame(rows).sort_values("wasserstein", ascending=False).reset_index(drop=True)
    return summary, table
