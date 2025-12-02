# ml_model.py
import os
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from joblib import dump, load

# ---------------------------
# Feature helpers (same as antes, robustificados)
# ---------------------------
def rolling_mean(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def rolling_std(series, window):
    return series.rolling(window=window, min_periods=1).std()

def compute_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=window, min_periods=1).mean()
    ma_down = down.rolling(window=window, min_periods=1).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, span_short=12, span_long=26, span_signal=9):
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(span=span_signal, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

# ---------------------------
# Cache loader (pickle)
# ---------------------------
def load_cache(filename="ibex_cache.pkl") -> Dict[str, Any]:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Cache file not found: {filename}")
    with open(filename, "rb") as f:
        return pickle.load(f)

# ---------------------------
# Feature builder per ticker
# ---------------------------
def build_features_for_df(df: pd.DataFrame, prefix: Optional[str]=None) -> pd.DataFrame:
    df = df.copy()
    # choose adjusted close
    if "Adj Close" in df.columns:
        price = df["Adj Close"].astype(float)
    elif "AdjClose" in df.columns:
        price = df["AdjClose"].astype(float)
    else:
        price = df["Close"].astype(float)

    features = pd.DataFrame(index=df.index)
    features["price"] = price
    features["return_1"] = price.pct_change(1)
    features["return_3"] = price.pct_change(3)
    features["return_5"] = price.pct_change(5)
    features["ma_5"] = rolling_mean(price, 5)
    features["ma_10"] = rolling_mean(price, 10)
    features["ma_21"] = rolling_mean(price, 21)
    features["std_10"] = rolling_std(price, 10)
    features["std_21"] = rolling_std(price, 21)
    features["rsi_14"] = compute_rsi(price, 14)
    macd, signal, hist = compute_macd(price)
    features["macd"] = macd
    features["macd_signal"] = signal
    features["macd_hist"] = hist
    features["momentum_7"] = price - price.shift(7)

    features = features.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill")
    if prefix:
        features = features.add_prefix(prefix + "_")
    return features

# ---------------------------
# Label creation (Buy/Sell/Hold)
# ---------------------------
def create_labels(df: pd.DataFrame, horizon: int=5, threshold: float=0.02) -> pd.Series:
    price = df["price"]
    future_price = price.shift(-horizon)
    future_return = (future_price - price) / price
    labels = pd.Series(index=df.index, dtype="object")
    labels.loc[future_return > threshold] = "Buy"
    labels.loc[future_return < -threshold] = "Sell"
    labels.loc[(future_return <= threshold) & (future_return >= -threshold)] = "Hold"
    return labels

# ---------------------------
# Build dataset combining tickers OR single ticker
# ---------------------------
def prepare_dataset_from_cache(cache: Dict[str, Any], tickers: Optional[list]=None,
                               min_rows: int=50, horizon: int=5, threshold: float=0.02
                              ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    price_data = cache.get("price_data", {})
    rows = []
    metas = []
    tickers_to_use = tickers if tickers else list(price_data.keys())
    for ticker in tickers_to_use:
        df = price_data.get(ticker)
        if df is None or df.empty:
            continue
        feats = build_features_for_df(df)
        if "price" not in feats.columns:
            continue
        labels = create_labels(feats, horizon=horizon, threshold=threshold)
        merged = feats.copy()
        merged["label"] = labels
        merged["ticker"] = ticker
        merged = merged.dropna(subset=["label"])
        if len(merged) < min_rows:
            continue
        rows.append(merged)
    if not rows:
        raise ValueError("No ticker had enough data to build dataset.")
    df_all = pd.concat(rows).sort_index()
    y = df_all["label"]
    X = df_all.drop(columns=["label"])
    meta = df_all[["ticker"]].copy()
    return X, y, meta

# ---------------------------
# Train / CV
# ---------------------------
def train_model(X: pd.DataFrame, y: pd.Series, n_splits: int=4, random_state: int=42
               ) -> Tuple[Pipeline, Dict[str, Any]]:
    # encode labels
    y_enc, classes = pd.factorize(y)
    class_mapping = {i: c for i, c in enumerate(classes)}
    # numeric columns
    X_num = X.select_dtypes(include=[np.number]).columns.tolist()

    preproc = ColumnTransformer([("num", StandardScaler(), X_num)], remainder="drop")
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1, class_weight="balanced")
    pipeline = Pipeline([("preproc", preproc), ("clf", clf)])

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_reports = []
    scores = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, preds)
        fold_reports.append({"fold": fold, "report": report, "confusion_matrix": cm})
        scores.append(report.get("macro avg", {}).get("f1-score", 0.0))

    pipeline.fit(X, y_enc)  # final fit on all data
    return pipeline, {"fold_reports": fold_reports, "cv_macro_f1_mean": float(np.mean(scores)), "class_mapping": class_mapping}

# ---------------------------
# Backtest (simple, fast)
# ---------------------------
def backtest_signals(pipeline: Pipeline, X: pd.DataFrame, meta: pd.DataFrame,
                     cache: Dict[str, Any], apply_holding_days: int = 1
                    ) -> Dict[str, Any]:
    """
    pipeline: trained pipeline (expects the same X columns)
    X: features DataFrame (index = date)
    meta: DataFrame with 'ticker' column aligned with X
    cache: original cache to get prices by ticker
    apply_holding_days: how many days to hold position for P&L calc (default 1)
    Returns dict with pnl series, sharpe, drawdown and per-ticker stats
    """
    # predict labels (encoded integers)
    preds_enc = pipeline.predict(X)
    # we need to map back to label names if pipeline was trained with factorize mapping saved separately.
    # Here, pipeline itself doesn't know mapping — the user should pass mapping separately for perfect mapping.
    # We'll assume preds_enc are integers and create position mapping: buy->+1, sell->-1, hold->0
    # Try to infer mapping if original model save included 'class_mapping.pkl' — try to load it.
    class_map = None
    if os.path.exists("training_report.pkl"):
        try:
            with open("training_report.pkl", "rb") as f:
                rep = pickle.load(f)
                class_map = rep.get("class_mapping")
        except Exception:
            class_map = None

    # attempt to reconstruct label strings
    label_names = None
    if class_map:
        label_names = [class_map[i] for i in sorted(class_map.keys())]

    # create a DataFrame with preds and metadata
    df = X.copy()
    df["pred_enc"] = preds_enc
    if meta is not None and "ticker" in meta.columns:
        df["ticker"] = meta["ticker"].values
    else:
        df["ticker"] = "UNKNOWN"

    # map enc->signal
    # if label_names known, map to 'Buy'/'Sell'/'Hold', else try heuristic by checking unique strings if preds are strings
    if label_names:
        inv_map = {i: label_names[i] for i in range(len(label_names))}
        df["pred_label"] = df["pred_enc"].map(inv_map)
    else:
        df["pred_label"] = df["pred_enc"].astype(str)

    # mapping to position values
    df["position"] = 0
    df.loc[df["pred_label"] == "Buy", "position"] = 1
    df.loc[df["pred_label"] == "Sell", "position"] = -1
    df.loc[df["pred_label"] == "Hold", "position"] = 0

    # compute pnl per-row using next-period returns for the corresponding ticker
    pnl_rows = []
    for ticker, grp in df.groupby("ticker"):
        price_df = cache["price_data"].get(ticker)
        if price_df is None or price_df.empty:
            continue
        # get daily return series aligned by index
        if "Adj Close" in price_df.columns:
            price = price_df["Adj Close"].astype(float)
        else:
            price = price_df["Close"].astype(float)
        daily_ret = price.pct_change().rename("ret_1")
        # align grp index with daily_ret
        joined = grp.join(daily_ret, how="left")
        # apply holding: simple approach - assume position at day t earns next day ret
        joined["pnl"] = joined["position"] * joined["ret_1"].shift(-0)  # position on day t applied to next day ret
        pnl_rows.append(joined[["ticker", "position", "ret_1", "pnl"]])

    if not pnl_rows:
        return {"error": "No prices available for backtest."}

    pnl_df = pd.concat(pnl_rows).sort_index().fillna(0.0)
    # daily aggregated pnl across tickers (sum)
    daily_pnl = pnl_df.groupby(pnl_df.index).pnl.sum()
    cum_returns = (1 + daily_pnl).cumprod()
    # Sharpe: use mean/std of daily_pnl
    mean = daily_pnl.mean()
    std = daily_pnl.std()
    sharpe = (mean / (std + 1e-9)) * np.sqrt(252) if std != 0 else np.nan
    # drawdown
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_dd = drawdown.min()

    return {
        "daily_pnl": daily_pnl,
        "cum_returns": cum_returns,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else None,
        "max_drawdown": float(max_dd) if np.isfinite(max_dd) else None,
        "pnl_df": pnl_df
    }

# ---------------------------
# Save/load pipeline & report
# ---------------------------
def save_pipeline(pipeline: Pipeline, filename: str = "model_pipeline.pkl"):
    dump(pipeline, filename)

def load_pipeline(filename: str = "model_pipeline.pkl") -> Pipeline:
    return load(filename)

def save_report(report: Dict[str, Any], filename: str = "training_report.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(report, f)

def load_report(filename: str = "training_report.pkl") -> Dict[str, Any]:
    if not os.path.exists(filename):
        return {}
    with open(filename, "rb") as f:
        return pickle.load(f)

# ---------------------------
# SHAP helper (sampled to be fast)
# ---------------------------
def compute_shap_explanations(pipeline: Pipeline, X_sample: pd.DataFrame, nsamples: int = 100):
    try:
        import shap
    except Exception as e:
        raise RuntimeError("shap not installed. Install `shap` to compute explanations.") from e

    clf = pipeline.named_steps["clf"]
    preproc = pipeline.named_steps["preproc"]
    X_proc = preproc.transform(X_sample)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_proc, nsamples=nsamples)
    return explainer, shap_values

# ---------------------------
# Train-by-ticker convenience
# ---------------------------
def train_by_ticker(cache: Dict[str, Any], ticker: str, min_rows: int = 60, horizon: int = 5,
                    threshold: float = 0.02, n_splits: int = 3):
    tickers = [ticker]
    X, y, meta = prepare_dataset_from_cache(cache, tickers=tickers, min_rows=min_rows, horizon=horizon, threshold=threshold)
    pipeline, report = train_model(X, y, n_splits=n_splits)
    return pipeline, report, X, y, meta
