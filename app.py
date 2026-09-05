"""Streamlit dashboard for the HR attrition prediction pipeline."""

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_loader import make_demo_data
from src.preprocessing import build_preprocessor

ROLE_OPTIONS = ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manager"]


@st.cache_resource
def load_model():
    data = make_demo_data(rows=400)
    features = data.drop(columns="Attrition")
    target = data["Attrition"].map({"Yes": 1, "No": 0})
    preprocessor, _, _ = build_preprocessor(data)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    model.fit(features, target)
    return model


def score(model, record: dict) -> float:
    return float(model.predict_proba(pd.DataFrame([record]))[0, 1])


def risk_band(probability: float) -> tuple[str, str, str]:
    if probability >= 0.65:
        return "Elevated", "#b42318", "Prioritize a contextual review"
    if probability >= 0.4:
        return "Watch", "#b54708", "Look for additional context"
    return "Lower", "#087443", "No immediate signal from this model"


def sensitivity(model, record: dict) -> pd.DataFrame:
    """Estimate directional sensitivity by changing one input at a time."""
    alternatives = {
        "Overtime": ("No", "Yes"),
        "Job satisfaction": (4, 1),
        "Years at company": (max(record["YearsAtCompany"], 10), 0),
        "Monthly income": (max(record["MonthlyIncome"], 7000), 1500),
    }
    current = score(model, record)
    rows = []
    for label, (lower_risk, higher_risk) in alternatives.items():
        field = {"Overtime": "OverTime", "Job satisfaction": "JobSatisfaction",
                 "Years at company": "YearsAtCompany", "Monthly income": "MonthlyIncome"}[label]
        lower_record = {**record, field: lower_risk}
        higher_record = {**record, field: higher_risk}
        rows.append({"Signal": label, "Current": current,
                     "Lower-risk scenario": score(model, lower_record),
                     "Higher-risk scenario": score(model, higher_record)})
    return pd.DataFrame(rows).sort_values("Higher-risk scenario", ascending=False)


st.set_page_config(page_title="HR Attrition Risk", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .block-container {max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem;}
    [data-testid="stMetricValue"] {font-size: 2.1rem;}
    .hero {padding: 1.5rem 1.8rem; border-radius: 14px; background: linear-gradient(120deg, #102a43, #1f4e5f); color: #f8fafc; margin-bottom: 1.4rem;}
    .hero h1 {margin: 0; font-size: 2.25rem; letter-spacing: 0;}
    .hero p {margin: .45rem 0 0; color: #d7e6ef; font-size: 1.02rem;}
    .risk-card {padding: 1rem 1.2rem; border-left: 5px solid var(--risk-color); background: #f8fafc; border-radius: 8px;}
    .risk-card h3 {margin: 0; color: var(--risk-color);}
    .risk-card p {margin: .35rem 0 0; color: #475569;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>HR Attrition Risk</h1>
    <p>A focused decision-support workspace for exploring employee attrition signals.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.subheader("About this demo")
    st.write("This interactive model uses a deterministic synthetic dataset for demonstration. It is not a personnel decision system.")
    st.divider()
    st.caption("Model")
    st.write("Balanced Logistic Regression")
    st.caption("Inputs")
    st.write("6 employee attributes")
    st.caption("Primary metric")
    st.write("F1-score and ROC-AUC")

st.subheader("Employee profile")
with st.form("employee_form"):
    left, right = st.columns(2)
    with left:
        age = st.slider("Age", 18, 65, 35)
        role = st.selectbox("Job role", ROLE_OPTIONS)
        income = st.number_input("Monthly income", min_value=1000, max_value=30000, value=5000, step=500)
    with right:
        overtime = st.selectbox("Overtime", ["No", "Yes"])
        satisfaction = st.slider("Job satisfaction", 1, 4, 3)
        years = st.slider("Years at company", 0, 40, 5)
    submitted = st.form_submit_button("Assess attrition risk", type="primary", use_container_width=True)

if submitted:
    model = load_model()
    record = {"Age": age, "JobRole": role, "MonthlyIncome": income, "OverTime": overtime,
              "JobSatisfaction": satisfaction, "YearsAtCompany": years}
    probability = score(model, record)
    label, color, guidance = risk_band(probability)

    st.divider()
    st.subheader("Assessment")
    metric, band, guidance_col = st.columns([1, 1, 2])
    with metric:
        st.metric("Estimated probability", f"{probability:.1%}")
    with band:
        st.metric("Risk band", label)
    with guidance_col:
        st.markdown(f'<div class="risk-card" style="--risk-color:{color}"><h3>{label} signal</h3><p>{guidance}</p></div>', unsafe_allow_html=True)
    st.progress(probability, text="Model score")

    insights_tab, scenarios_tab, notes_tab = st.tabs(["Signals", "Scenario lab", "Model notes"])
    with insights_tab:
        st.write("These one-variable comparisons show how the demo model responds to selected inputs. They are directional, not causal explanations.")
        table = sensitivity(model, record)
        display = table.copy()
        for column in ["Current", "Lower-risk scenario", "Higher-risk scenario"]:
            display[column] = display[column].map(lambda value: f"{value:.1%}")
        st.dataframe(display, hide_index=True, use_container_width=True)
    with scenarios_tab:
        baseline = {**record, "OverTime": "No", "JobSatisfaction": 4, "YearsAtCompany": max(years, 10),
                    "MonthlyIncome": max(income, 7000)}
        baseline_probability = score(model, baseline)
        comparison = pd.DataFrame({"Profile": ["Submitted profile", "Supportive baseline"],
                                   "Probability": [probability, baseline_probability]})
        st.bar_chart(comparison.set_index("Profile"), y="Probability", height=260)
        st.caption("The baseline is a comparison scenario, not a recommended employee profile or HR target.")
    with notes_tab:
        st.write("The production pipeline compares Logistic Regression, Random Forest, and XGBoost using stratified evaluation. This hosted demo uses Logistic Regression so the app starts quickly and does not require private model artifacts.")
        st.write("Use the score as one input for a careful, human-led review. Do not use it as the sole basis for employment decisions.")
else:
    st.info("Complete the profile and select Assess attrition risk to see the model dashboard.")