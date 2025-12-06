import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils import get_tickers_data
from ml_models import (
    prepare_dataset,
    load_xgboost,
    load_r_forest
)

###########################################
# 1. PREPARAR DATOS PARA SHAP
###########################################

def prepare_shap_data(model_bundle, df):
    features = model_bundle["features"]
    scaler = model_bundle["scaler"]

    X = df[features]
    X_scaled = scaler.transform(X)

    return pd.DataFrame(X_scaled, columns=features, index=X.index)


###########################################
# 2. NORMALIZAR SHAP VALUES (CLAVE)
###########################################

def get_class_shap_values(shap_values, class_idx):
    """
    Normaliza salida SHAP para:
    - RandomForest (lista)
    - XGBoost multiclase (array 3D)
    """
    if isinstance(shap_values, list):
        return shap_values[class_idx]

    # XGBoost sklearn → (n_samples, n_features, n_classes)
    if len(shap_values.shape) == 3:
        return shap_values[:, :, class_idx]

    raise ValueError("Formato SHAP no soportado")


###########################################
# 3. SHAP GLOBAL
###########################################

def shap_global_summary(model_bundle, X_scaled_df, class_idx, out_file):
    model = model_bundle["model"]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled_df)

    class_shap = get_class_shap_values(shap_values, class_idx)

    shap.summary_plot(
        class_shap,
        X_scaled_df,
        plot_type="bar",
        show=False
    )

    plt.savefig(out_file, bbox_inches="tight")
    plt.close()


###########################################
# 4. SHAP LOCAL
###########################################

def shap_local_explanation(model_bundle, X_scaled_df, row_idx, class_idx, out_file):
    model = model_bundle["model"]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled_df)

    class_shap = get_class_shap_values(shap_values, class_idx)

    base_value = (
        explainer.expected_value[class_idx]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value
    )

    shap.force_plot(
        base_value,
        class_shap[row_idx],
        X_scaled_df.iloc[row_idx],
        matplotlib=True,
        show=False
    )

    plt.savefig(out_file, bbox_inches="tight")
    plt.close()


###########################################
# 5. MAIN
###########################################

if __name__ == "__main__":

    price_data, _ = get_tickers_data()
    df = prepare_dataset(price_data)

    # ---------- XGBOOST ----------
    print("Running SHAP for XGBoost...")
    xgb_bundle = load_xgboost("models/xgboost.pkl")
    X_shap_xgb = prepare_shap_data(xgb_bundle, df)

    shap_global_summary(xgb_bundle, X_shap_xgb, 0, "imgs/xgb_shap_buy.png")
    shap_global_summary(xgb_bundle, X_shap_xgb, 1, "imgs/xgb_shap_hold.png")
    shap_global_summary(xgb_bundle, X_shap_xgb, 2, "imgs/xgb_shap_sell.png")

    shap_local_explanation(
        xgb_bundle,
        X_shap_xgb,
        row_idx=100,
        class_idx=0,
        out_file="imgs/xgb_local_buy.png"
    )

    # ---------- RANDOM FOREST ----------
    print("Running SHAP for Random Forest...")
    rf_bundle = load_r_forest("models/r_forest.pkl")
    X_shap_rf = prepare_shap_data(rf_bundle, df)

    shap_global_summary(rf_bundle, X_shap_rf, 0, "rf_shap_buy.png")

    print("✅ XAI images generated successfully")
