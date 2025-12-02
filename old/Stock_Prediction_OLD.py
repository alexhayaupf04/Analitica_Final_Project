import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="ML IBEX35 Signals", layout="wide")
st.title("📈 Señales de Compra/Venta – IBEX 35")
st.write("Modelo entrenado offline usando RandomForest. Aquí se muestran señales recientes y resumen de oportunidades.")

IBEX_TICKERS = [
    "ACS.MC", "ACX.MC", "AENA.MC", "AMS.MC", "ANA.MC", "ANE.MC",
    "BBVA.MC", "BKT.MC", "CABK.MC", "CLNX.MC", "COL.MC", "ELE.MC",
    "ENG.MC", "FDR.MC", "FER.MC", "GRF.MC", "IAG.MC", "IBE.MC",
    "ITX.MC", "LOG.MC", "MAP.MC", "MRL.MC", "MTS.MC", "NTGY.MC",
    "PUIG.MC", "RED.MC", "SAB.MC", "SAN.MC", "TEF.MC", "UNI.MC"
]

# -----------------------------
# 1) CARGAR MODELO
# -----------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# -----------------------------
# 2) FEATURE ENGINEERING RAPIDA
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
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["return_1"] = df["Close"].pct_change()
    df["return_5"] = df["Close"].pct_change(5)
    df["ma_10"] = df["Close"].rolling(10).mean()
    df["ma_20"] = df["Close"].rolling(20).mean()
    df["rsi"] = compute_rsi(df["Close"])

    # Nuevas features
    df["vol_5"] = df["Close"].rolling(5).std()
    df["ma_ratio"] = df["ma_10"] / df["ma_20"]
    df["momentum_5"] = df["Close"] - df["Close"].shift(5)
    df["vol_change"] = df["Volume"].pct_change()

    df = df.dropna()
    return df

# -----------------------------
# 3) PREDICCION POR TICKER
# -----------------------------
signals = []

for ticker in IBEX_TICKERS:
    df = yf.download(ticker, period="1y")
    if df.empty:
        continue

    # MultiIndex fix
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df_feat = build_features(df)
    if df_feat.empty:
        continue

    X = df_feat[["return_1","return_5","ma_10","ma_20","rsi", "vol_5", "ma_ratio", "momentum_5", "vol_change"]].tail(1)
    pred = model.predict(X)[0]

    signals.append({"Ticker": ticker, "Signal": pred, "Last Close": df_feat["Close"].iloc[-1]})

signals_df = pd.DataFrame(signals)

# -----------------------------
# 4) MOSTRAR RESULTADOS
# -----------------------------
st.subheader("📊 Resumen de señales")
st.write(signals_df.groupby("Signal").count())

st.markdown("### 📈 Top 5 oportunidades de Compra")
top_buy = signals_df[signals_df["Signal"]=="Buy"].sort_values(by="Last Close", ascending=False).head(5)
st.dataframe(top_buy)

st.markdown("### 📉 Top 5 señales de Venta")
top_sell = signals_df[signals_df["Signal"]=="Sell"].sort_values(by="Last Close", ascending=True).head(5)
st.dataframe(top_sell)

st.subheader("🔎 Señales recientes")
st.dataframe(signals_df)
