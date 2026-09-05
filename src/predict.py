"""Score a new employee record with the saved best model."""

import argparse
import json
from pathlib import Path
import joblib
import pandas as pd


def predict(record: dict, model_path: str = "models/best_model.joblib") -> dict:
    model = joblib.load(model_path)
    probability = float(model.predict_proba(pd.DataFrame([record]))[0, 1])
    return {"attrition_probability": probability, "risk_label": "high" if probability >= 0.5 else "low"}


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="JSON object containing employee features")
    group.add_argument("--input-file", help="Path to a JSON object containing employee features")
    args = parser.parse_args()
    raw_input = args.input if args.input is not None else Path(args.input_file).read_text(encoding="utf-8")
    print(json.dumps(predict(json.loads(raw_input)), indent=2))


if __name__ == "__main__":
    main()