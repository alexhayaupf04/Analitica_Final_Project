import streamlit as st
import yfinance as yf
import pandas as pd
import pickle
import shap
import numpy as np

st.set_page_config(page_title="ML Explainability", layout="wide")
st.title("🔍 SHAP Explainability – IBEX 35")
st.write("Selecciona un ticker y modelo para ver cómo cada feature influye en la predicción Buy/Hold/Sell.")

# -----------------------------
# Cargar modelos
# -----------------------------
@st.cache_resource
def load_models():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

models = load_models()
model_options = ["RandomForest", "XGBoost"]
selected_model = st.selectbox("Selecciona modelo:", model_options)
model = models["rf"] if selected_model=="RandomForest" else models["xgb"]
scaler = models["scaler"]
features = models["features"]

# -----------------------------
# Lista de tickers
# -----------------------------
IBEX_TICKERS = [
    "ACS.MC", "ACX.MC", "AENA.MC", "AMS.MC", "ANA.MC", "ANE.MC",
    "BBVA.MC", "BKT.MC", "CABK.MC", "CLNX.MC", "COL.MC", "ELE.MC",
    "ENG.MC", "FDR.MC", "FER.MC", "GRF.MC", "IAG.MC", "IBE.MC",
    "ITX.MC", "LOG.MC", "MAP.MC", "MRL.MC", "MTS.MC", "NTGY.MC",
    "PUIG.MC", "RED.MC", "SAB.MC", "SAN.MC", "TEF.MC", "UNI.MC"
]
selected_ticker = st.selectbox("Selecciona ticker:", IBEX_TICKERS)

# -----------------------------
# Feature engineering
# -----------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def build_features(df):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    df["return_1"] = df["Close"].pct_change()
    df["return_5"] = df["Close"].pct_change(5)
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["rsi"] = compute_rsi(df["Close"])
    
    df["vol_5"] = df["Close"].rolling(5).std()
    df["ma_ratio"] = df["ma_10"] / df["ma_20"]
    df["momentum_5"] = df["Close"] - df["Close"].shift(5)
    df["vol_change"] = df["Volume"].pct_change()
    
    df = df.dropna()
    return df

# -----------------------------
# Descargar datos ticker
# -----------------------------
@st.cache_data
def download_and_prepare(ticker):
    df = yf.download(ticker, period="1y")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df_feat = build_features(df)
    return df_feat

df_feat = download_and_prepare(selected_ticker)
if df_feat.empty:
    st.warning("No se pudieron descargar datos para este ticker.")
else:
    # Última fila para predicción
    X = df_feat[features].tail(1)
    X_scaled = scaler.transform(X)
    
    # Predicción
    pred = model.predict(X_scaled)[0]
    st.subheader(f"Predicción para {selected_ticker}: **{pred}**")

    # -----------------------------
    # SHAP explainability
    # -----------------------------
    explainer = shap.Explainer(model, X_scaled)
    shap_values = explainer(X_scaled)
    
    st.subheader("📊 Feature importance (SHAP values)")
    st.pyplot(shap.plots.bar(shap_values, show=False))

    st.subheader("🌟 Force plot (contribución de cada feature)")
    st.pyplot(shap.plots.force(shap_values[0], matplotlib=True, show=False))

    st.subheader("📈 Summary plot de SHAP (últimas 30 filas)")
    X_last30 = df_feat[features].tail(30)
    X_last30_scaled = scaler.transform(X_last30)
    explainer_last30 = shap.Explainer(model, X_last30_scaled)
    shap_values_last30 = explainer_last30(X_last30_scaled)
    st.pyplot(shap.summary_plot(shap_values_last30, X_last30, feature_names=features, show=False))
