# HR Attrition Prediction

An end-to-end, reproducible machine-learning project for predicting employee attrition and translating model results into HR-actionable insights.

## Project layout

```text
data/raw/          Original CSV (kept out of version control)
data/processed/    Train/test CSV exports
notebooks/         EDA, preprocessing, modeling, and interpretation notebooks
src/               Reusable pipeline modules and command-line scripts
models/            Saved model and metadata artifacts
reports/figures/   Generated charts
tests/             Fast unit tests
```

## Dataset

The code is designed for the IBM HR Analytics Employee Attrition & Performance CSV, commonly distributed as `WA_Fn-UseC_-HR-Employee-Attrition.csv`. Place that file at `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv`. The target column is `Attrition` (`Yes`/`No`). Do not commit the raw dataset; it may have its own license and distribution terms.

For a smoke test or demo, `src/data_loader.py` can generate a deterministic synthetic dataset with the same core schema.

## Setup and run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Use the supplied IBM CSV, or generate a local demo CSV
python -m src.data_loader --generate-demo
python -m src.train --data data/raw/demo_hr_attrition.csv
python -m src.evaluate
# Or pass a JSON file on Windows when shell quoting is inconvenient:
python -m src.predict --input-file employee.json

# Launch the interactive demo
streamlit run app.py
```

The training command compares Logistic Regression, Random Forest, and XGBoost when available, selects the model with the highest validation F1-score, writes `models/best_model.joblib` and `models/metrics.json`, and exports `data/processed/train.csv` and `test.csv`. XGBoost is optional at runtime; the other two models are always used.

## Deployment

The included `app.py` runs on Streamlit Community Cloud. Push this repository to GitHub, select `app.py` as the main file, and use `requirements.txt` for dependencies. The app bootstraps a deterministic demo model when no saved model artifact is present.

## Evaluation and interpretation

Metrics include accuracy, precision, recall, F1-score, and ROC-AUC. F1 and ROC-AUC are prioritized because attrition is typically the minority class. Run `python -m src.feature_importance --data data/raw/demo_hr_attrition.csv` after training to create coefficient/importances and an optional SHAP summary. Run `jupyter notebook` and open the notebooks in order for EDA and model interpretation.

## Limitations

This is a decision-support model, not an automated employment decision system. Validate performance, fairness, data governance, and local employment-law requirements before using predictions in an HR workflow. Predictions indicate statistical risk, not employee intent or worth.
