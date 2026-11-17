"""Model comparison and hyper-parameter tuning, tracked with MLflow."""
from __future__ import annotations

import contextlib
import io
import logging

import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import config as C

for _name in ("alembic", "mlflow"):
    logging.getLogger(_name).setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def setup_mlflow(experiment: str) -> None:
    with contextlib.redirect_stderr(io.StringIO()):   # silence one-time DB migrations
        mlflow.set_tracking_uri(C.MLFLOW_URI)
        mlflow.set_experiment(experiment)


def split_and_resample(df: pd.DataFrame):
    """Stratified split first, then SMOTE on the training part only (no leakage)."""
    X, y = df[C.FEATURES], df[C.TARGET]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=C.VAL_SIZE, stratify=y, random_state=C.SEED)
    X_res, y_res = SMOTE(k_neighbors=C.SMOTE_K, random_state=C.SEED).fit_resample(X_tr, y_tr)
    return X_res, y_res, X_val, y_val, y_tr


def evaluate(y_true, y_pred) -> dict[str, float]:
    per_class = f1_score(y_true, y_pred, labels=C.CLASSES, average=None, zero_division=0)
    out = {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    for c, f in zip(C.CLASSES, per_class):
        out[f"f1_{C.CLASS_NAMES[c].replace(' ', '')}"] = float(f)
    return {k: float(v) for k, v in out.items()}


def candidate_models() -> dict:
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, random_state=C.SEED, class_weight="balanced")),
        ]),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=C.SEED,
                                               class_weight="balanced"),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=C.SEED,
                                 eval_metric="mlogloss", verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=C.SEED,
                                   class_weight="balanced", verbose=-1),
    }


def compare_models(X_res, y_res, X_val, y_val) -> pd.DataFrame:
    """Train each candidate, log params/metrics/model to MLflow, return a leaderboard."""
    setup_mlflow(C.EXP_SELECTION)
    rows = {}
    for name, model in candidate_models().items():
        with mlflow.start_run(run_name=name):
            model.fit(X_res, y_res)
            metrics = evaluate(y_val, model.predict(X_val))
            mlflow.log_params({"model": name, "smote_k_neighbors": C.SMOTE_K})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_val.iloc[:5])
        rows[name] = metrics
    return pd.DataFrame(rows).T.sort_values("macro_f1", ascending=False)


def _xgb_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
    }


def tune_and_register(X_res, y_res, X_val, y_val, n_trials: int = C.N_TRIALS):
    """Optuna TPE search on macro-F1, then log + register + alias the winner."""
    fixed = {"random_state": C.SEED, "eval_metric": "mlogloss", "verbosity": 0}

    def objective(trial):
        model = XGBClassifier(**_xgb_space(trial), **fixed).fit(X_res, y_res)
        return f1_score(y_val, model.predict(X_val), average="macro", zero_division=0)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=C.SEED))
    study.optimize(objective, n_trials=n_trials)

    model = XGBClassifier(**study.best_params, **fixed).fit(X_res, y_res)
    metrics = evaluate(y_val, model.predict(X_val))

    setup_mlflow(C.EXP_TUNING)
    with mlflow.start_run(run_name="XGBoost_Optuna_best") as run:
        mlflow.log_params({"model": "XGBoost_Optuna", "n_trials": n_trials, **study.best_params})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_val.iloc[:5])
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", C.REGISTERED_MODEL)
    MlflowClient().set_registered_model_alias(C.REGISTERED_MODEL, C.PROD_ALIAS, mv.version)
    return model, study, metrics, int(mv.version)
