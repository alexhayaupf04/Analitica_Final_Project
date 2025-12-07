import streamlit as st
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils import get_tickers_data
from ml_models import (
    prepare_dataset,
    load_xgboost,
    load_r_forest
)
from xai import (
    prepare_shap_data,
    get_class_shap_values
)

st.set_page_config(
    page_title="Explainable AI (XAI)",
    layout="wide"
)

st.title("IBEX 35 – Explainable AI (SHAP)")

st.markdown(
    """
    This section explains how the models make trading decisions using **SHAP**.
    Global explanations show overall feature importance,
    while local explanations justify individual predictions.
    """
)

@st.cache_data
def load_data():
    price_data, _ = get_tickers_data()
    return prepare_dataset(price_data)

df = load_data()


col1, col2 = st.columns(2)

with col1:
    model_name = st.selectbox(
        "Select model:",
        ["XGBoost", "Random Forest"]
    )

with col2:
    class_map = {0: "Buy", 1: "Hold", 2: "Sell"}
    class_idx = st.selectbox(
        "Select class:",
        options=[0, 1, 2],
        format_func=lambda x: class_map[x]
    )


if model_name == "XGBoost":
    bundle = load_xgboost("models/xgboost.pkl")
else:
    bundle = load_r_forest("models/r_forest.pkl")


X_shap = prepare_shap_data(bundle, df)

if model_name == "Random Forest":
    X_shap = X_shap.sample(200, random_state=42)


@st.cache_resource
def compute_shap(bundle, X_shap):
    explainer = shap.TreeExplainer(bundle["model"])
    shap_values = explainer.shap_values(X_shap)
    return explainer, shap_values

explainer, shap_values = compute_shap(bundle, X_shap)
class_shap = get_class_shap_values(shap_values, class_idx)

st.header("Global Feature Importance")
st.markdown("Feature importance aggregated over the dataset.")

shap.summary_plot(
    class_shap,
    X_shap,
    plot_type="bar",
    show=False
)

st.pyplot(plt)
plt.close()


st.header("Local Explanation")
st.markdown("Explanation for a single model prediction.")

row_idx = st.slider(
    "Select an observation:",
    min_value=0,
    max_value=len(X_shap) - 1,
    value=0
)

base_value = (
    explainer.expected_value[class_idx]
    if isinstance(explainer.expected_value, (list, np.ndarray))
    else explainer.expected_value
)

shap.force_plot(
    base_value,
    class_shap[row_idx],
    X_shap.iloc[row_idx],
    matplotlib=True,
    show=False
)

st.pyplot(plt)
plt.close()
