# HR Attrition Prediction

End-to-end machine-learning project for predicting employee attrition risk and identifying the factors associated with employee turnover.

## Live Demo

**Try the deployed Streamlit app:** [HR Attrition Risk Predictor](https://hr11attrition11prediction.streamlit.app/)

**Source repository:** [Sarikapatel02/hr-attrition-prediction](https://github.com/Sarikapatel02/hr-attrition-prediction)

The demo accepts age, job role, monthly income, overtime, job satisfaction, and tenure, then returns an estimated attrition probability. It uses a deterministic demo model when no private or local model artifact is available.

## What This Project Does

- Loads and validates HR attrition data.
- Performs exploratory analysis and saves charts.
- Encodes categorical features and scales numeric features.
- Compares Logistic Regression, Random Forest, and XGBoost.
- Selects the best model using F1-score, with ROC-AUC as a tie-breaker.
- Exports evaluation metrics and feature importance.
- Provides a command-line prediction tool and Streamlit interface.

## Pipeline

```text
Raw CSV
	-> validation and duplicate removal
	-> stratified train/test split
	-> imputation, scaling, and one-hot encoding
	-> model comparison
	-> metrics, feature importance, SHAP summary, and prediction
```

## Repository Structure

```text
data/
	raw/                  Input dataset, excluded from Git
	processed/            Validated train/test exports
notebooks/
	01_eda.ipynb          Exploratory analysis
	02_preprocessing.ipynb Preprocessing notes
	03_modeling.ipynb     Model comparison
	04_feature_importance.ipynb  Interpretation notes
src/
	data_loader.py        Loading, validation, and demo data
	preprocessing.py      Reusable feature transformer
	train.py              Model training and comparison
	evaluate.py           Metrics display
	feature_importance.py Ranked drivers and SHAP summary
	predict.py            CLI prediction API
app.py                  Streamlit deployment entry point
models/                 Local model artifacts, excluded from Git
reports/                Findings and generated figures
tests/                  Automated regression tests
requirements.txt        Python dependencies
```

## Dataset

The code is designed for the IBM HR Analytics Employee Attrition & Performance CSV, commonly distributed as `WA_Fn-UseC_-HR-Employee-Attrition.csv`. Place that file at `data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv`. The target column is `Attrition` (`Yes`/`No`). Do not commit the raw dataset; it may have its own license and distribution terms.

For a smoke test or demo, `src/data_loader.py` can generate a deterministic synthetic dataset with the same core schema.

## Setup and Run

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

## Reports and Evaluation

Metrics include accuracy, precision, recall, F1-score, and ROC-AUC. F1-score and ROC-AUC are prioritized because attrition is typically the minority class. Generated outputs include:

- `reports/figures/eda_summary.csv`
- `reports/figures/attrition_by_overtime.png`
- `reports/figures/feature_importance.csv`
- `reports/figures/top_feature_importance.png`
- `reports/figures/shap_summary.png`
- `models/metrics.json`

Run the following after training to generate interpretation outputs:

```powershell
python -m src.eda --data data/raw/demo_hr_attrition.csv
python -m src.feature_importance --data data/raw/demo_hr_attrition.csv
```

## Deployment

The application is deployed at [hr11attrition11prediction.streamlit.app](https://hr11attrition11prediction.streamlit.app/). To deploy your own copy on Streamlit Community Cloud:

1. Fork or clone this repository.
2. Select `app.py` as the main file.
3. Use `requirements.txt` for dependencies.
4. Deploy from the `main` branch.

The app bootstraps a fast deterministic Logistic Regression model, so deployment does not depend on ignored model binaries or private HR data.

## Responsible Use

This is a decision-support model, not an automated employment decision system. Validate performance, fairness, data governance, and local employment-law requirements before using predictions in an HR workflow. Predictions indicate statistical risk, not employee intent or worth.
