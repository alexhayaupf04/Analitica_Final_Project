import yfinance as yf
import pandas as pd
import numpy as np
import pickle
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from utils import get_tickers_data


###########################################
# 2. FEATURES
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
# 3. LABELS
###########################################
def create_labels(df, future_horizon=5, thr=0.01):
    future = df["Close"].shift(-future_horizon)
    ret = (future - df["Close"]) / df["Close"]

    labels = np.where(ret > thr, "Buy",
                      np.where(ret < -thr, "Sell", "Hold"))

    df["label"] = labels
    df = df.dropna()
    return df


###########################################
# 4. DATASET GLOBAL
###########################################
def prepare_dataset(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for ticker, df in data_dict.items():
        df_feat = build_features(df)
        df_feat = create_labels(df_feat)
        df_feat["ticker"] = ticker
        rows.append(df_feat)
    return pd.concat(rows)


###########################################
# 5. BACKTEST FUNCIONES PROFESIONALES
###########################################
def compute_equity_curve(ret):
    return (1 + ret.fillna(0)).cumprod()


def max_drawdown(cum_curve):
    roll_max = cum_curve.cummax()
    dd = (roll_max - cum_curve) / roll_max
    return dd.max()


def sharpe_ratio(ret, freq=252):
    mean = ret.mean()
    vol = ret.std()
    if vol == 0:
        return 0
    return (mean / vol) * np.sqrt(freq)


def backtest_fold(df_test, preds, le):
    df = df_test.copy()
    df["ret"] = df["Close"].pct_change()

    buy = le.transform(["Buy"])[0]
    sell = le.transform(["Sell"])[0]

    df["pred"] = preds

    df["strategy_ret"] = np.where(df["pred"] == buy, df["ret"],
                           np.where(df["pred"] == sell, -df["ret"], 0))

    df["cum_strategy"] = compute_equity_curve(df["strategy_ret"])
    df["cum_buyhold"] = compute_equity_curve(df["ret"])

    metrics = {
        "Strategy Return": df["cum_strategy"].iloc[-1] - 1,
        "BuyHold Return": df["cum_buyhold"].iloc[-1] - 1,
        "Max Drawdown": max_drawdown(df["cum_strategy"]),
        "Sharpe Ratio": sharpe_ratio(df["strategy_ret"])
    }

    return df, metrics


###########################################
# 6. TRAIN + BACKTEST + GUARDADO
###########################################
def train_xgboost(df):
    feature_cols = [
        "return_1","return_5","ma_10","ma_20","rsi",
        "vol_5","ma_ratio","momentum_5","vol_change"
    ]

    X = df[feature_cols]
    y = df["label"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    tscv = TimeSeriesSplit(n_splits=5)
    fold_results = []

    print("\n=== BACKTEST ===")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        print(f"\n--- FOLD {fold} ---")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="mlogloss", use_label_encoder=False,
            random_state=42
        )

        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        print(classification_report(y_test, preds, target_names=le.classes_))

        df_test = df.iloc[test_idx]
        bt_df, metrics = backtest_fold(df_test, preds, le)
        fold_results.append({"fold": fold, "metrics": metrics, "curve": bt_df})

        print(metrics)

    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)

    final_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="mlogloss", use_label_encoder=False,
        random_state=42
    )

    final_model.fit(X_scaled, y_train)
    return final_model, scaler, feature_cols, le, fold_results


def train_r_forest(df):
    feature_cols = [
        "return_1","return_5","ma_10","ma_20","rsi",
        "vol_5","ma_ratio","momentum_5","vol_change"
    ]

    X = df[feature_cols]
    y = df["label"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    tscv = TimeSeriesSplit(n_splits=5)
    fold_results = []

    print("\n=== BACKTEST ===")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        print(f"\n--- FOLD {fold} ---")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")

        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        print(classification_report(y_test, preds, target_names=le.classes_))

        df_test = df.iloc[test_idx]
        bt_df, metrics = backtest_fold(df_test, preds, le)
        fold_results.append({"fold": fold, "metrics": metrics, "curve": bt_df})

        print(metrics)

    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)

    final_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")

    final_model.fit(X_scaled, y_train)
    return final_model, scaler, feature_cols, le, fold_results


def save_xgboost(final_model, scaler, feature_cols, le, fold_results, filename="models/xgboost.pkl"):

    with open(filename, "wb") as f:
        pickle.dump({
            "model": final_model,
            "scaler": scaler,
            "features": feature_cols,
            "label_encoder": le,
            "backtest": fold_results
        }, f)

def save_r_forest(final_model, scaler, feature_cols, le, fold_results, filename="models/r_forest.pkl"):

    with open(filename, "wb") as f:
        pickle.dump({
            "model": final_model,
            "scaler": scaler,
            "features": feature_cols,
            "label_encoder": le,
            "backtest": fold_results
        }, f)


if __name__ == "__main__":

    price_data, info_data = get_tickers_data()
    print("Preparing data...")
    df = prepare_dataset(price_data)
    print("Training xgboost...")
    final_model, scaler, feature_cols, le, fold_results = train_xgboost(df)
    save_xgboost(final_model, scaler, feature_cols, le, fold_results)
    print("Saved xgboost...")
    print("Training random forest...")
    final_model2, scaler2, feature_cols2, le2, fold_results2 = train_r_forest(df)
    save_r_forest(final_model2, scaler2, feature_cols2, le2, fold_results2)
    print("Saved random forest...")