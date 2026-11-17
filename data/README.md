# Data

| File | Rows | Role |
|---|---|---|
| `baseline.csv` | 6,993 | Labelled historical readings. Training and validation reference. |
| `batch_stable.csv` | 1,499 | Post-deployment batch under normal operation. |
| `batch_heavy_load.csv` | 1,499 | Post-deployment batch from a heavy-load production period (valid but shifted). |
| `ai4i2020_failure_types.csv` | 9,991 | Full source table with the original binary failure flags (TWF, HDF, PWF, OSF, RNF) and the derived multi-class label. |

**Columns:** `Type` (L/M/H product quality variant), `Air temperature` [K], `Process temperature` [K], `Rotational speed` [rpm], `Torque` [Nm], `Tool wear` [min], `Failure_Type` (target).

**Target classes:** 0 No Failure · 1 TWF Tool Wear · 2 HDF Heat Dissipation · 3 PWF Power · 4 OSF Overstrain.

## Source and licence
Derived from the **AI4I 2020 Predictive Maintenance Dataset**, UCI Machine Learning Repository
(https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

> S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications,"
> 2020 Third International Conference on Artificial Intelligence for Industries (AI4I), 2020, pp. 69-74.

The dataset is synthetic and reflects real industrial predictive maintenance patterns. No private or personal data is included.
