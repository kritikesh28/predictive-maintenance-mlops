"""End-to-end pipeline: validate -> train -> track -> tune -> monitor -> explain.

Usage:
    python -m predmaint.pipeline            # full run (about 2 minutes on a laptop CPU)
    python -m predmaint.pipeline --trials 5 # quick smoke run
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import joblib
import pandas as pd

from . import config as C
from .data import load_batches, validate
from .explain import plot_shap_by_class, shap_by_class
from .features import add_features, strain_index
from .monitor import drift_report
from .plots import plot_drift, plot_eda, plot_leaderboard
from .train import compare_models, evaluate, split_and_resample, tune_and_register

warnings.filterwarnings("ignore")


def banner(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def run(n_trials: int = C.N_TRIALS) -> dict:
    t0 = time.time()
    for d in (C.FIGURES_DIR, C.DRIFT_DIR, C.MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    banner("1/6  Data contract (Pandera)")
    batches = load_batches()
    checks = [validate(df, name) for name, df in batches.items()]
    for chk in checks:
        print(f"{chk['batch']:<11} rows={chk['rows']:<5} valid={chk['valid']}  violations={chk['violations']}")
    batches = {k: add_features(v) for k, v in batches.items()}
    base = batches["baseline"]
    plot_eda(base, C.FIGURES_DIR / "eda_overview.png")
    dist = base[C.TARGET].map(C.CLASS_NAMES).value_counts()
    print("\nClass balance (baseline):\n" + (dist / len(base) * 100).round(2).astype(str).add(" %").to_string())

    banner("2/6  Model comparison (MLflow)")
    X_res, y_res, X_val, y_val, y_tr = split_and_resample(base)
    board = compare_models(X_res, y_res, X_val, y_val)
    print(board[["macro_f1", "weighted_f1", "accuracy"]].round(4).to_string())
    plot_leaderboard(board, C.FIGURES_DIR / "leaderboard.png")

    banner(f"3/6  Optuna tuning ({n_trials} trials) + model registry")
    model, study, tuned, version = tune_and_register(X_res, y_res, X_val, y_val, n_trials)
    joblib.dump(model, C.MODELS_DIR / "xgb_predmaint.joblib")
    model.get_booster().save_model(str(C.MODELS_DIR / "xgb_predmaint.json"))
    base_f1 = board.loc["XGBoost", "macro_f1"]
    print(f"baseline XGBoost macro-F1 {base_f1:.4f} -> tuned {tuned['macro_f1']:.4f} "
          f"({tuned['macro_f1'] - base_f1:+.4f}); registered {C.REGISTERED_MODEL} v{version} @{C.PROD_ALIAS}")

    banner("4/6  Drift monitoring (Evidently)")
    drift = {}
    for name in ("stable", "heavy_load"):
        summary, table = drift_report(base, batches[name], C.DRIFT_DIR / f"drift_{name}.html")
        perf = evaluate(batches[name][C.TARGET], model.predict(batches[name][C.FEATURES]))
        drift[name] = {**summary, "features": table.to_dict(orient="records"),
                       "macro_f1": perf["macro_f1"], "accuracy": perf["accuracy"],
                       "f1_OSF": perf["f1_OSF"],
                       "osf_share_pct": float((batches[name][C.TARGET] == 4).mean() * 100),
                       "strain_gt_11000_pct": float((strain_index(batches[name]) > 11_000).mean() * 100)}
        print(f"\n[{name}] dataset drift={summary['dataset_drift']} "
              f"({summary['drifted_features']}/{summary['n_features']} features) | "
              f"model macro-F1={perf['macro_f1']:.4f}, OSF F1={perf['f1_OSF']:.4f}")
        print(table.to_string(index=False))

    plot_drift(drift, C.FIGURES_DIR / "drift_comparison.png")

    banner("5/6  Explainability (SHAP, per failure class)")
    importance = shap_by_class(model, base[C.FEATURES])
    plot_shap_by_class(importance, C.FIGURES_DIR / "shap_per_class.png")
    top = importance.idxmax().to_dict()
    for cls, feat in top.items():
        print(f"{cls:<4} top driver: {feat:<17} (mean |SHAP| {importance.loc[feat, cls]:.2f})")

    banner("6/6  Retraining decision")
    hl = drift["heavy_load"]
    drifted = [f["feature"] for f in hl["features"] if f["drift"]]
    osf_drivers = importance["OSF"].nlargest(2).index.tolist()
    retrain = hl["dataset_drift"] and bool(set(drifted) & set(osf_drivers))
    print(f"Drifted under heavy load: {drifted}")
    print(f"OSF top SHAP drivers    : {osf_drivers}")
    print(f"RETRAIN RECOMMENDED     : {retrain}")

    metrics = {
        "split": "stratified 80/20 of baseline, SMOTE on train split only, seed 42",
        "leaderboard": board.round(4).to_dict(orient="index"),
        "tuned_xgboost": {**{k: round(v, 4) for k, v in tuned.items()},
                          "best_params": study.best_params, "registry_version": version},
        "drift": drift,
        "shap_mean_abs": importance.round(4).to_dict(),
        "shap_top_driver": top,
        "retrain_recommended": retrain,
    }
    (C.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    pd.DataFrame(board).round(4).to_csv(C.REPORTS_DIR / "leaderboard.csv")
    print(f"\nDone in {time.time() - t0:.0f}s -> reports/, models/, mlflow.db")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=C.N_TRIALS, help="Optuna trials (default 30)")
    run(ap.parse_args().trials)


if __name__ == "__main__":
    main()
