import streamlit as st
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import get_tickers_data
from ml_models import prepare_dataset, load_xgboost, load_r_forest


st.set_page_config(page_title="XAI", layout="wide")
st.title("IBEX-35 – Explainable AI (SHAP)")


@st.cache_resource
def load_data():
    price_data, _ = get_tickers_data()
    return prepare_dataset(price_data)


@st.cache_resource
def load_model(name):
    if name == "XGBoost":
        return load_xgboost("models/xgboost.pkl")
    return load_r_forest("models/r_forest.pkl")


@st.cache_resource
def compute_shap(_model, X):
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


df = load_data()

model_name = st.selectbox("Model", ["XGBoost", "Random Forest"])
class_idx = st.selectbox("Class", [0, 1, 2], format_func=lambda x: ["Buy", "Hold", "Sell"][x])

bundle = load_model(model_name)
features = bundle["features"]
scaler = bundle["scaler"]

X = df[features].sample(200, random_state=42)
X_scaled = pd.DataFrame(
    scaler.transform(X),
    columns=features,
    index=X.index
)

explainer, shap_values = compute_shap(bundle["model"], X_scaled)

if isinstance(shap_values, list):
    class_shap = shap_values[class_idx]
else:
    class_shap = shap_values[:, :, class_idx]


st.header("Global Feature Importance")
shap.summary_plot(class_shap, X_scaled, plot_type="bar", show=False)
st.pyplot(plt)
plt.close()


st.header("Local Explanation")
row = st.slider("Observation", 0, len(X_scaled) - 1, 0)
base = explainer.expected_value[class_idx]

shap.force_plot(
    base,
    class_shap[row],
    X_scaled.iloc[row],
    matplotlib=True,
    show=False
)

st.pyplot(plt)
plt.close()
