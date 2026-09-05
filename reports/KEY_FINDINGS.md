# Key Findings

Run the EDA and training commands against the supplied IBM HR CSV before filling in this section. The generated artifacts are:

- `reports/figures/eda_summary.csv` for data quality and descriptive statistics.
- `reports/figures/attrition_by_overtime.png` for the overtime comparison.
- `reports/figures/feature_importance.csv` and `top_feature_importance.png` for model drivers.
- `models/metrics.json` for the held-out evaluation comparison.

Interpret the highest-ranked features as associations in this dataset, not causal explanations. Review them with HR subject-matter experts, check subgroup performance, and avoid using the score as the sole basis for employment decisions.