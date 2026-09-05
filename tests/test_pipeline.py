import pandas as pd
import pytest

from src.data_loader import make_demo_data, validate_data
from src.preprocessing import build_preprocessor
from src.predict import predict
from src.train import train_models


def test_demo_data_has_valid_target():
    data = validate_data(make_demo_data(100))
    assert set(data["Attrition"]) == {0, 1}


def test_missing_required_column_is_rejected():
    with pytest.raises(ValueError, match="missing required"):
        validate_data(pd.DataFrame({"Attrition": ["Yes"]}))


def test_preprocessor_covers_numeric_and_categorical_features():
    data = make_demo_data(30)
    _, numeric, categorical = build_preprocessor(data)
    assert "Age" in numeric
    assert "JobRole" in categorical


def test_training_writes_model_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = train_models(make_demo_data(160), tmp_path / "models")
    assert result["best_model"] in result["models"]
    assert (tmp_path / "models" / "best_model.joblib").exists()
    assert (tmp_path / "data" / "processed" / "train.csv").exists()
    assert (tmp_path / "data" / "processed" / "test.csv").exists()
    prediction = predict({"Age": 35, "JobRole": "Sales Executive", "MonthlyIncome": 5000,
                          "OverTime": "No", "JobSatisfaction": 3, "YearsAtCompany": 5},
                         str(tmp_path / "models" / "best_model.joblib"))
    assert 0 <= prediction["attrition_probability"] <= 1