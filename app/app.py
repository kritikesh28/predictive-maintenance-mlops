"""Gradio demo: predict the failure mode of a machine from six sensor readings.

Self-contained on purpose (no project imports) so the same file runs as a
Hugging Face Space. It looks for the model next to this file first, then in ../models.
"""
from __future__ import annotations

from pathlib import Path

import gradio as gr
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

HERE = Path(__file__).resolve().parent
MODEL_PATH = next(p for p in (HERE / "xgb_predmaint.joblib",
                              HERE.parent / "models" / "xgb_predmaint.joblib") if p.exists())
MODEL = joblib.load(MODEL_PATH)
EXPLAINER = shap.TreeExplainer(MODEL)

CLASS_NAMES = ["No Failure", "TWF - Tool Wear", "HDF - Heat Dissipation",
               "PWF - Power", "OSF - Overstrain"]
TYPE_MAP = {"H": 0, "L": 1, "M": 2}
FEATURES = ["Type_enc", "Air temperature", "Process temperature",
            "Rotational speed", "Torque", "Tool wear", "Power_W", "Temp_diff"]
STRAIN_ALERT = 10_000  # min*Nm


def build_features(mtype, air_k, proc_k, rpm, torque, wear) -> pd.DataFrame:
    return pd.DataFrame([{
        "Type_enc": TYPE_MAP[mtype],
        "Air temperature": float(air_k),
        "Process temperature": float(proc_k),
        "Rotational speed": int(rpm),
        "Torque": float(torque),
        "Tool wear": int(wear),
        "Power_W": float(torque) * int(rpm) * 2 * np.pi / 60,
        "Temp_diff": float(proc_k) - float(air_k),
    }])[FEATURES]


def explain_plot(X: pd.DataFrame, cls: int):
    values = EXPLAINER.shap_values(X)
    values = np.stack(values, axis=-1) if isinstance(values, list) else values
    contrib = pd.Series(values[0, :, cls], index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.barh(contrib.index, contrib.values,
            color=["#C44E52" if v > 0 else "#4C72B0" for v in contrib.values])
    ax.axvline(0, c="grey", lw=0.8)
    ax.set_title(f"Why '{CLASS_NAMES[cls]}'? (SHAP, red pushes towards it)")
    ax.set_xlabel("contribution (log-odds)")
    fig.tight_layout()
    return fig


def predict(mtype, air_k, proc_k, rpm, torque, wear):
    X = build_features(mtype, air_k, proc_k, rpm, torque, wear)
    proba = MODEL.predict_proba(X)[0]
    cls = int(proba.argmax())
    power, dT, strain = X["Power_W"].iat[0], X["Temp_diff"].iat[0], float(torque) * int(wear)
    alert = ("**Overstrain alert:** tool wear x torque is above 10,000 min·Nm. "
             "Plan a tool change or reduce load." if strain > STRAIN_ALERT else
             "Strain index is within the normal range.")
    details = (f"| Derived signal | Value |\n|---|---|\n"
               f"| Mechanical power | {power:,.0f} W |\n"
               f"| Process - air temperature | {dT:.1f} K |\n"
               f"| Strain index (wear x torque) | {strain:,.0f} min·Nm |\n\n{alert}")
    return {n: float(p) for n, p in zip(CLASS_NAMES, proba)}, details, explain_plot(X, cls)


EXAMPLES = [
    ["L", 298.3, 308.1, 1528, 33.9, 82],    # healthy
    ["H", 300.5, 309.9, 1397, 45.9, 210],   # tool wear
    ["L", 303.5, 311.9, 1344, 49.3, 93],    # heat dissipation
    ["L", 301.5, 310.9, 2760, 8.0, 15],     # power (too low torque, very high speed)
    ["M", 302.8, 312.3, 1262, 70.5, 234],   # overstrain
]

with gr.Blocks(title="Predictive Maintenance: Failure Mode Classifier") as demo:
    gr.Markdown(
        "# Predictive Maintenance: Failure Mode Classifier\n"
        "Enter six sensor readings from a milling machine. The tuned XGBoost model predicts whether "
        "it is healthy or heading towards one of four failure modes, and SHAP shows which signals "
        "drove the call. Try the examples below.")
    with gr.Row():
        with gr.Column():
            mtype = gr.Radio(["L", "M", "H"], value="L", label="Machine quality variant (Low / Medium / High)")
            air = gr.Slider(295, 305, value=300, step=0.1, label="Air temperature [K]")
            proc = gr.Slider(305, 315, value=310, step=0.1, label="Process temperature [K]")
            rpm = gr.Slider(1000, 2900, value=1500, step=1, label="Rotational speed [rpm]")
            torque = gr.Slider(3, 80, value=40, step=0.1, label="Torque [Nm]")
            wear = gr.Slider(0, 253, value=100, step=1, label="Tool wear [min]")
            btn = gr.Button("Predict", variant="primary")
        with gr.Column():
            label = gr.Label(num_top_classes=5, label="Failure-mode probability")
            details = gr.Markdown()
            plot = gr.Plot(label="Explanation")
    inputs = [mtype, air, proc, rpm, torque, wear]
    btn.click(predict, inputs, [label, details, plot])
    gr.Examples(EXAMPLES, inputs, [label, details, plot], fn=predict, cache_examples=False,
                run_on_click=True)
    gr.Markdown(
        "Model: XGBoost tuned with Optuna on macro-F1 (validation macro-F1 0.767). "
        "Trained on the public AI4I 2020 Predictive Maintenance dataset (UCI, CC BY 4.0). "
        "Demo only; not for safety-critical use.")

if __name__ == "__main__":
    demo.launch()
