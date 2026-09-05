"""Train and compare attrition classifiers."""

from pathlib import Path
import argparse
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data_loader import TARGET, load_data, validate_data
from .preprocessing import build_preprocessor, save_processed


def metrics_for(model, x_test, y_test) -> dict[str, float]:
    prediction = model.predict(x_test)
    probability = model.predict_proba(x_test)[:, 1]
    return {"accuracy": accuracy_score(y_test, prediction), "precision": precision_score(y_test, prediction, zero_division=0),
            "recall": recall_score(y_test, prediction, zero_division=0), "f1": f1_score(y_test, prediction, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probability)}


def train_models(data: pd.DataFrame, output_dir: str | Path = "models") -> dict:
    data = validate_data(data).drop_duplicates().reset_index(drop=True)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_processed(data, "data/processed")
    x = data.drop(columns=TARGET)
    y = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    x_train.assign(**{TARGET: y_train}).to_csv(processed_dir / "train.csv", index=False)
    x_test.assign(**{TARGET: y_test}).to_csv(processed_dir / "test.csv", index=False)
    preprocessor, _, _ = build_preprocessor(data)
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42, n_jobs=-1),
    }
    try:
        from xgboost import XGBClassifier
        candidates["xgboost"] = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                                               colsample_bytree=0.8, eval_metric="logloss", random_state=42)
    except ImportError:
        pass
    results = {}
    fitted = {}
    for name, estimator in candidates.items():
        model = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])
        model.fit(x_train, y_train)
        results[name] = metrics_for(model, x_test, y_test)
        fitted[name] = model
    best_name = max(results, key=lambda name: (results[name]["f1"], results[name]["roc_auc"]))
    joblib.dump(fitted[best_name], output / "best_model.joblib")
    (output / "metrics.json").write_text(json.dumps({"best_model": best_name, "models": results}, indent=2), encoding="utf-8")
    return {"best_model": best_name, "models": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    args = parser.parse_args()
    print(json.dumps(train_models(load_data(args.data)), indent=2))


if __name__ == "__main__":
    main()