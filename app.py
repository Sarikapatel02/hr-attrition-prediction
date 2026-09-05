"""Streamlit demo for the HR attrition prediction pipeline."""

import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_loader import make_demo_data
from src.preprocessing import build_preprocessor


@st.cache_resource
def load_model():
    data = make_demo_data(rows=400)
    features = data.drop(columns="Attrition")
    target = data["Attrition"].map({"Yes": 1, "No": 0})
    preprocessor, _, _ = build_preprocessor(data)
    model = Pipeline([("preprocessor", preprocessor),
                      ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
    model.fit(features, target)
    return model


st.set_page_config(page_title="HR Attrition Risk", page_icon="📊", layout="centered")
st.title("HR Attrition Risk")
st.caption("Decision-support demo using a trained employee attrition classifier.")

with st.form("employee_form"):
    age = st.slider("Age", 18, 65, 35)
    role = st.selectbox("Job role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manager"])
    income = st.number_input("Monthly income", min_value=1000, max_value=30000, value=5000, step=500)
    overtime = st.selectbox("Overtime", ["No", "Yes"])
    satisfaction = st.slider("Job satisfaction", 1, 4, 3)
    years = st.slider("Years at company", 0, 40, 5)
    submitted = st.form_submit_button("Assess attrition risk")

if submitted:
    model = load_model()
    record = {"Age": age, "JobRole": role, "MonthlyIncome": income, "OverTime": overtime,
              "JobSatisfaction": satisfaction, "YearsAtCompany": years}
    probability = float(model.predict_proba(pd.DataFrame([record]))[0, 1])
    st.metric("Estimated attrition probability", f"{probability:.1%}")
    if probability >= 0.5:
        st.warning("Higher predicted risk. Review context with appropriate HR safeguards.")
    else:
        st.success("Lower predicted risk in this model.")
    st.caption("This estimate is not a judgment about an employee and must not be used as the sole basis for employment decisions.")