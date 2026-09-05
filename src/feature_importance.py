"""Export feature importance for the saved pipeline."""

from pathlib import Path
import argparse
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from .data_loader import TARGET, load_data


def export_feature_importance(model_path: str = "models/best_model.joblib", output_dir: str = "reports/figures") -> pd.DataFrame:
    model = joblib.load(model_path)
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    values = getattr(classifier, "feature_importances_", None)
    if values is None:
        values = abs(classifier.coef_[0])
    result = pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "feature_importance.csv", index=False)
    top = result.head(10).sort_values("importance")
    top.plot.barh(x="feature", y="importance", legend=False, figsize=(8, 5), color="#0f766e")
    plt.tight_layout()
    plt.savefig(output / "top_feature_importance.png", dpi=150)
    plt.close()
    return result


def export_shap_summary(model_path: str, data_path: str, output_dir: str) -> bool:
    """Create an optional SHAP bar summary for the fitted classifier."""
    try:
        import shap
        model = joblib.load(model_path)
        data = load_data(data_path)
        features = data.drop(columns=TARGET)
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]
        transformed = pd.DataFrame(preprocessor.transform(features), columns=preprocessor.get_feature_names_out())
        explanation = shap.Explainer(classifier, transformed)(transformed)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        shap.plots.bar(explanation, max_display=10, show=False)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        return True
    except ImportError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best_model.joblib")
    parser.add_argument("--data", help="CSV used for optional SHAP analysis")
    args = parser.parse_args()
    print(export_feature_importance(args.model).head(10).to_string(index=False))
    if args.data:
        print(f"SHAP summary generated: {export_shap_summary(args.model, args.data, 'reports/figures')}")


if __name__ == "__main__":
    main()