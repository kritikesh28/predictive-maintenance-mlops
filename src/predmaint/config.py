"""Central configuration: paths, feature lists, class names and seeds."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DRIFT_DIR = REPORTS_DIR / "drift"
MODELS_DIR = ROOT / "models"

BASELINE_CSV = "baseline.csv"          # labelled historical data (training reference)
STABLE_CSV = "batch_stable.csv"        # post-deployment batch, normal operation
HEAVY_LOAD_CSV = "batch_heavy_load.csv"  # post-deployment batch, heavy-load period

SEED = 42
VAL_SIZE = 0.2
SMOTE_K = 3                  # rarest class has only ~24 training samples
N_TRIALS = 30                # Optuna trials

MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
EXP_SELECTION = "PredMaint_ModelSelection"
EXP_TUNING = "PredMaint_Optuna"
REGISTERED_MODEL = "PredMaint_XGBoost"
PROD_ALIAS = "production"

TARGET = "Failure_Type"
CLASS_NAMES = {0: "No Failure", 1: "TWF", 2: "HDF", 3: "PWF", 4: "OSF"}
CLASS_DESCRIPTIONS = {
    0: "Machine operating normally",
    1: "Tool Wear Failure",
    2: "Heat Dissipation Failure",
    3: "Power Failure",
    4: "Overstrain Failure",
}
CLASSES = list(CLASS_NAMES)

# Same ordering as sklearn's LabelEncoder (alphabetical) -> fixed, no pickle needed
TYPE_MAP = {"H": 0, "L": 1, "M": 2}

RAW_COLS = ["Type", "Air temperature", "Process temperature",
            "Rotational speed", "Torque", "Tool wear"]
SENSOR_COLS = ["Air temperature", "Process temperature",
               "Rotational speed", "Torque", "Tool wear"]
FEATURES = ["Type_enc", "Air temperature", "Process temperature",
            "Rotational speed", "Torque", "Tool wear", "Power_W", "Temp_diff"]
