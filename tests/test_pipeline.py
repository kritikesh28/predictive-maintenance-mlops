"""Fast unit tests: data contract, feature physics, inference contract."""
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from predmaint import config as C
from predmaint.data import load_batches, validate
from predmaint.features import add_features, strain_index
from predmaint.predict import predict

ROW = {"Type": "L", "Air temperature": 300.0, "Process temperature": 310.0,
       "Rotational speed": 1500, "Torque": 40.0, "Tool wear": 100, "Failure_Type": 0}


@pytest.fixture(scope="module")
def batches():
    return load_batches()


def test_all_batches_pass_contract(batches):
    for name, df in batches.items():
        assert validate(df, name)["valid"], name


def test_contract_rejects_out_of_range():
    bad = pd.DataFrame([{**ROW, "Torque": 120.0}, {**ROW, "Type": "X"}])
    bad["Rotational speed"] = bad["Rotational speed"].astype("int64")
    bad["Tool wear"] = bad["Tool wear"].astype("int64")
    res = validate(bad, "bad")
    assert not res["valid"] and res["violations"] >= 2


def test_power_and_temp_diff_physics():
    out = add_features(pd.DataFrame([ROW]))
    assert out["Power_W"].iat[0] == pytest.approx(40.0 * 1500 * 2 * np.pi / 60)
    assert out["Temp_diff"].iat[0] == pytest.approx(10.0)
    assert out["Type_enc"].iat[0] == C.TYPE_MAP["L"]


def test_unknown_machine_type_raises():
    with pytest.raises(ValueError):
        add_features(pd.DataFrame([{**ROW, "Type": "Z"}]))


def test_strain_index():
    assert strain_index(pd.DataFrame([ROW])).iat[0] == 4000


def test_predict_contract(batches):
    base = add_features(batches["baseline"]).sample(800, random_state=0)
    model = XGBClassifier(n_estimators=20, verbosity=0).fit(base[C.FEATURES], base[C.TARGET])
    out = predict(model, [ROW, {**ROW, "Torque": 60.0, "Tool wear": 220}])
    assert list(out.columns[:1]) == ["prediction"]
    assert np.allclose(out.filter(like="p_").sum(axis=1), 1.0, atol=1e-5)
    assert out["strain_alert"].tolist() == [False, True]
