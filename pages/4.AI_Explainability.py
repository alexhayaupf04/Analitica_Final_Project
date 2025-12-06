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

###########################################
# CONFIG
###########################################

st.set_page_config(
    page_title="Explainable AI (XAI)",
    layout="wide"
)

st.title("Explainable AI – SHAP Analysis")

###########################################
# UTILS SHAP (CLAVE)
###########################################

def prepare_shap_data(model_bundle, df):
    features = model_bundle["features"]
    scaler = model_bundle["scaler"]

    X = df[features]
    X_scaled = scaler.transform(X)

    return pd.DataFrame(X_scaled, columns=features, index=X.index)


def get_class_shap_values(shap_values, class_idx):
    """
    Normaliza SHAP para:
    - RandomForest: list[class]
    - XGBoost multiclase: array (n_samples, n_features, n_classes)
    """
    if isinstance(shap_values, list):
        return shap_values[class_idx]

    if len(shap_values.shape) == 3:
        return shap_values[:, :, class_idx]

    raise ValueError("Formato SHAP no soportado")


###########################################
# LOAD DATA
###########################################

price_data, _ = get_tickers_data()
df = prepare_dataset(price_data)

###########################################
# MODEL SELECTION
###########################################

model_name = st.selectbox(
    "Select model",
    ["XGBoost", "Random Forest"]
)

if model_name == "XGBoost":
    bundle = load_xgboost("models/xgboost.pkl")
else:
    bundle = load_r_forest("models/r_forest.pkl")

###########################################
# PREPARE SHAP INPUT
###########################################

X_shap = prepare_shap_data(bundle, df)

explainer = shap.TreeExplainer(bundle["model"])
shap_values = explainer.shap_values(X_shap)

class_map = {0: "Buy", 1: "Hold", 2: "Sell"}

class_idx = st.selectbox(
    "Select class",
    options=[0, 1, 2],
    format_func=lambda x: class_map[x]
)

class_shap = get_class_shap_values(shap_values, class_idx)

###########################################
# GLOBAL EXPLANATION
###########################################

st.header("Global Feature Importance")

shap.summary_plot(
    class_shap,
    X_shap,
    plot_type="bar",
    show=False
)

st.pyplot(plt)
plt.close()

###########################################
# LOCAL EXPLANATION
###########################################

st.header("Local Explanation")

row_idx = st.slider(
    "Select observation",
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
