import yfinance as yf
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


###########################################
# 1) DESCARGA
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
# 2) FEATURES
###########################################
def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def build_features(df):
    df = df.copy()
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


###########################################
# 3) LABELS BUY/HOLD/SELL
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
# 4) UNIR TODO
###########################################
def prepare_dataset(data_dict):
    rows = []
    for ticker, df in data_dict.items():
        df_feat = build_features(df)
        df_feat = create_labels(df_feat)
        df_feat["ticker"] = ticker
        rows.append(df_feat)
    full = pd.concat(rows)
    return full


###########################################
# 5) ENTRENAMIENTO + BACKTEST
###########################################
def train_classification(df):
    feature_cols = [
        "return_1","return_5","ma_10","ma_20","rsi",
        "vol_5","ma_ratio","momentum_5","vol_change"
    ]

    X = df[feature_cols]
    y = df["label"]

    # Codificación 0/1/2
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # BACKTEST: TimeSeriesSplit (NO mezclar futuro)
    tscv = TimeSeriesSplit(n_splits=5)

    print("\n=== BACKTESTING TimeSeriesSplit ===")

    fold_num = 1
    for train_idx, test_idx in tscv.split(X):

        print(f"\n--- Fold {fold_num} ---")
        fold_num += 1

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        # Oversampling solo en train
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X_train, y_train)

        # Escalado
        scaler = StandardScaler().fit(X_res)
        X_train_scaled = scaler.transform(X_res)
        X_test_scaled  = scaler.transform(X_test)

        # Modelo
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42
        )

        model.fit(X_train_scaled, y_res)
        preds = model.predict(X_test_scaled)

        print(classification_report(y_test, preds, target_names=le.classes_))


    #######################################
    # ENTRENAR MODELO FINAL (FULL DATA)
    #######################################
    print("\nEntrenando modelo final...")

    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y_encoded)

    scaler = StandardScaler().fit(X_res)
    X_res_scaled = scaler.transform(X_res)

    model_final = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42
    )

    model_final.fit(X_res_scaled, y_res)

    # Guardar
    with open("model_classif.pkl", "wb") as f:
        pickle.dump({
            "model": model_final,
            "scaler": scaler,
            "features": feature_cols,
            "label_encoder": le
        }, f)

    print("Modelo final guardado en model_classif.pkl")


###########################################
# MAIN
###########################################
if __name__ == "__main__":
    print("Descargando datos...")
    data = download_data(IBEX_TICKERS)

    print("Construyendo dataset...")
    df = prepare_dataset(data)

    print("Entrenando modelo...")
    train_classification(df)
