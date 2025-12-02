import yfinance as yf
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
import xgboost as xgb



###########################################
# 1) DESCARGAR DATOS IBEX 35
###########################################
IBEX_TICKERS = [
    "ACS.MC", "ACX.MC", "AENA.MC", "AMS.MC", "ANA.MC", "ANE.MC",
    "BBVA.MC", "BKT.MC", "CABK.MC", "CLNX.MC", "COL.MC", "ELE.MC",
    "ENG.MC", "FDR.MC", "FER.MC", "GRF.MC", "IAG.MC", "IBE.MC",
    "ITX.MC", "LOG.MC", "MAP.MC", "MRL.MC", "MTS.MC", "NTGY.MC",
    "PUIG.MC", "RED.MC", "SAB.MC", "SAN.MC", "TEF.MC", "UNI.MC"
    ]

def download_data(tickers, period="5y"):
    data = {}
    for t in tickers:
        df = yf.download(t, period=period)
        if not df.empty:
            data[t] = df
    return data

###########################################
# 2) FEATURE ENGINEERING
###########################################
def build_features(df):
    df = df.copy()
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

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

###########################################
# 3) GENERAR ETIQUETAS BUY / HOLD / SELL
###########################################
def create_labels(df, future_horizon=5, thr=0.01):
    future = df["Close"].shift(-future_horizon)
    ret = (future - df["Close"]) / df["Close"]
    labels = []
    for r in ret:
        if r > thr:
            labels.append("Buy")
        elif r < -thr:
            labels.append("Sell")
        else:
            labels.append("Hold")
    df["label"] = labels
    df = df.dropna()
    return df

###########################################
# 4) UNIR TODO Y ENTRENAR
###########################################
def prepare_dataset(data_dict):
    rows = []
    for ticker, df in data_dict.items():
        df.columns = [c[0] for c in df.columns]
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df_feat = build_features(df)
        df_feat = create_labels(df_feat)
        df_feat["ticker"] = ticker
        rows.append(df_feat)
    full = pd.concat(rows)
    return full

###########################################
# 5) ENTRENAMIENTO
###########################################
def train_models(df):
    feature_cols = ["return_1","return_5","ma_10","ma_20","rsi",
                    "vol_5","ma_ratio","momentum_5","vol_change"]
    X = df[feature_cols]
    y = df["label"]

    # Oversampling para Buy/Sell
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)

    # Escalado
    scaler = StandardScaler()
    X_res_scaled = scaler.fit_transform(X_res)

    # RandomForest
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    rf.fit(X_res_scaled, y_res)
    print("RandomForest entrenado.")

    # XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=4,
                                  learning_rate=0.05, subsample=0.9,
                                  colsample_bytree=0.9, eval_metric="mlogloss",
                                  use_label_encoder=False)
    xgb_model.fit(X_res_scaled, y_res)
    print("XGBoost entrenado.")

    # Guardar ambos modelos + scaler
    with open("model.pkl", "wb") as f:
        pickle.dump({"rf": rf, "xgb": xgb_model, "scaler": scaler, "features": feature_cols}, f)

    print("Modelos guardados en model.pkl")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("Descargando datos...")
    data = download_data(IBEX_TICKERS)

    print("Construyendo dataset...")
    df = prepare_dataset(data)

    print("Entrenando modelos...")
    train_models(df)