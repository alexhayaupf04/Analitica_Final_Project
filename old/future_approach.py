# quant_pipeline.py
import os
import time
import math
from typing import Dict, Tuple, List, Any
import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from joblib import dump, load
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

from utils import get_tickers_data, get_ibex_tickers
# SHAP
import shap

# ---------------------------
# 0. CONFIG
# ---------------------------
SEED = 42
np.random.seed(SEED)

# Backtest / market params (ajusta según la realidad)
COMMISSION_PER_TRADE = 0.0005    # 0.05% por trade (adjust)
SLIPPAGE_PCT = 0.0005            # 0.05% slippage
MAX_POSITION_SIZE = 0.02         # max % equity per position (risk limit)
FREQ = 252                       # trading days per year

# ---------------------------
# 1. DATA INGEST (usa tu get_tickers_data)
#   Aquí dejo una función wrapper si no tienes

# ---------------------------
# 2. FEATURE ENGINEERING (avanzado y robusto)
# ---------------------------
def add_basic_feats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['close'] = df['Close']
    df['ret_1'] = df['Close'].pct_change()
    df['ret_5'] = df['Close'].pct_change(5)
    df['logret_1'] = np.log(df['Close']).diff()
    df['volume_change'] = df['Volume'].pct_change()
    # moving averages
    df['ma_5'] = df['Close'].rolling(5).mean()
    df['ma_10'] = df['Close'].rolling(10).mean()
    df['ma_20'] = df['Close'].rolling(20).mean()
    df['ma_ratio_5_20'] = df['ma_5'] / df['ma_20']
    # volatility
    df['vol_5'] = df['ret_1'].rolling(5).std()
    df['vol_20'] = df['ret_1'].rolling(20).std()
    # momentum
    df['mom_5'] = df['Close'] - df['Close'].shift(5)
    df['mom_20'] = df['Close'] - df['Close'].shift(20)
    # RSI
    df['rsi_14'] = compute_rsi(df['Close'], 14)
    # VWAP approximation for daily (typical price)
    df['typ_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['vwap_10'] = (df['typ_price'] * df['Volume']).rolling(10).sum() / (df['Volume'].rolling(10).sum() + 1e-9)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_advanced_feats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade features algo más sofisticadas como realised volatility, rolling skew/kurt, lagged features, etc.
    """
    df = df.copy()
    # Realized volatility (daily window)
    df['rv_10'] = df['ret_1'].rolling(10).std() * np.sqrt(252)
    # skewness / kurtosis rolling
    df['skew_20'] = df['ret_1'].rolling(20).skew()
    df['kurt_20'] = df['ret_1'].rolling(20).kurt()
    # lag features
    for lag in [1,2,3,5]:
        df[f'ret_lag_{lag}'] = df['ret_1'].shift(lag)
    # ratio price/volume features
    df['pv_ratio'] = (df['Close'] / (df['Volume'] + 1)).rolling(5).mean()
    return df

def build_features(df):
    df = df.copy()
    df = add_basic_feats(df)
    df = add_advanced_feats(df)
    df = df.dropna()
    return df

# ---------------------------
# 3. LABELS: triple-barrier method simplificado + meta-labeling
# ---------------------------
def triple_barrier_labels(df: pd.DataFrame, pt_sl: Tuple[float,float]=(0.02, 0.02),
                          t_days:int=5, min_ret_vol=0.001) -> pd.DataFrame:
    """
    Implementación simplificada del triple-barrier:
     - pt_sl: (profit_taker pct, stop_loss pct) relativos al precio actual
     - t_days: horizonte máximo
    Devuelve DataFrame con columnas: label (1 buy, -1 sell, 0 hold), tol_exit_date, exit_return
    """
    df = df.copy()
    N = len(df)
    labels = np.zeros(N, dtype=int)
    exit_dates = [pd.NaT]*N
    exit_rets = np.zeros(N, dtype=float)

    close = df['Close'].values

    for i in range(N):
        start_price = close[i]
        if np.isnan(start_price):
            continue
        pt = start_price * (1 + pt_sl[0])
        sl = start_price * (1 - pt_sl[1])
        max_j = min(N-1, i + t_days)
        triggered = False
        for j in range(i+1, max_j+1):
            price_j = close[j]
            if np.isnan(price_j):
                continue
            # check profit taker
            if price_j >= pt:
                labels[i] = 1
                exit_dates[i] = df.index[j]
                exit_rets[i] = (price_j - start_price) / start_price
                triggered = True
                break
            if price_j <= sl:
                labels[i] = -1
                exit_dates[i] = df.index[j]
                exit_rets[i] = (price_j - start_price) / start_price
                triggered = True
                break
        if not triggered:
            # time barrier
            j = max_j
            price_j = close[j]
            labels[i] = 1 if price_j > start_price else -1 if price_j < start_price else 0
            exit_dates[i] = df.index[j]
            exit_rets[i] = (price_j - start_price) / start_price

    out = df.copy()
    out['tb_label'] = labels
    out['tb_exit_date'] = exit_dates
    out['tb_exit_ret'] = exit_rets
    # Filter tiny moves
    out.loc[out['tb_exit_ret'].abs() < min_ret_vol, 'tb_label'] = 0
    return out

def meta_labeling(df: pd.DataFrame, prob_model_preds: np.ndarray, thr=0.5):
    """
    Ejemplo: transformar las predicciones crudas en etiquetas meta (si queremos)
    """
    return (prob_model_preds >= thr).astype(int)

# ---------------------------
# 4. PURGED K-FOLD (con embargo)
# ---------------------------
def purged_kfold_indices(times: pd.DatetimeIndex, n_splits=5, embargo_pct=0.01):
    """
    Returns list of (train_idx, test_idx). Embargo aplicado al final de cada train block.
    times: index datetime
    """
    n = len(times)
    indices = np.arange(n)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    out = []
    for train_idx, test_idx in tscv.split(indices):
        # apply embargo: remove last embargo_pct of train from being used (by index)
        emb = int(n * embargo_pct)
        if emb > 0:
            max_train = train_idx.max()
            embargo_start = max_train - emb + 1
            if embargo_start < 0:
                embargo_start = 0
            train_idx = train_idx[train_idx < embargo_start]
        out.append((train_idx, test_idx))
    return out

# ---------------------------
# 5. BACKTEST ENGINE (realista)
# ---------------------------
def compute_equity_curve(returns_series):
    return (1 + returns_series.fillna(0)).cumprod()

def max_drawdown(cum_series):
    roll_max = cum_series.cummax()
    dd = (roll_max - cum_series) / roll_max
    return dd.max()

def sharpe_ratio(returns, freq=FREQ):
    mean = returns.mean()
    vol = returns.std()
    if vol == 0:
        return 0.0
    return (mean / vol) * np.sqrt(freq)

def backtest_with_preds(df: pd.DataFrame, preds: np.ndarray, prob: np.ndarray=None,
                        le: LabelEncoder=None, capital=1.0):
    """
    Preds: 0/1/2 encoded by labelencoder or direct numeric mapping.
    Interpretación: Buy->long ret, Sell->short ret, Hold->0
    Aplica comisiones + slippage + turnover.
    """
    df = df.copy().reset_index()
    if le is not None:
        classes = list(le.inverse_transform([0,1,2]))
    else:
        classes = ['Buy','Hold','Sell']

    # Map preds to positions
    pos = np.zeros(len(df))
    # asumimos buy -> +1, sell -> -1
    mapping = {}
    try:
        buy_idx = int(np.where(le.classes_=='Buy')[0][0])
        sell_idx = int(np.where(le.classes_=='Sell')[0][0])
        mapping[buy_idx] = 1
        mapping[sell_idx] = -1
    except:
        # fallback: try labels existing
        for i, c in enumerate(le.classes_):
            if c == 'Buy':
                mapping[i] = 1
            elif c == 'Sell':
                mapping[i] = -1

    for i, p in enumerate(preds):
        pos[i] = mapping.get(p, 0)

    df['position'] = pos
    # returns: use close-to-close simple returns
    df['market_ret'] = df['Close'].pct_change().fillna(0.0)
    # strategy raw ret
    df['strat_ret_raw'] = df['position'].shift(0) * df['market_ret']  # assume intraday entry not considered here
    # trading costs: apply cost when position changes (turnover)
    df['pos_change'] = df['position'].diff().fillna(0).abs()
    df['trading_costs'] = df['pos_change'] * COMMISSION_PER_TRADE + df['pos_change'] * SLIPPAGE_PCT
    df['strat_ret_after_costs'] = df['strat_ret_raw'] - df['trading_costs']
    df['cum_strategy'] = compute_equity_curve(df['strat_ret_after_costs'])
    df['cum_buyhold'] = compute_equity_curve(df['market_ret'])
    metrics = {
        'Strategy Return': df['cum_strategy'].iloc[-1] - 1,
        'BuyHold Return': df['cum_buyhold'].iloc[-1] - 1,
        'Sharpe': sharpe_ratio(df['strat_ret_after_costs']),
        'Max Drawdown': max_drawdown(df['cum_strategy'])
    }
    return df.set_index('Date'), metrics

# ---------------------------
# 6. TRAIN + CV + SAVE
# ---------------------------
def train_pipeline(data_dict: Dict[str, pd.DataFrame], pt_sl=(0.02, 0.02), t_days=5):
    """
    data_dict: {ticker: df}
    Retorna modelo final, scaler, features list, label encoder, backtest folds info.
    """
    # 1) Build per-ticker features + labeling
    rows = []
    for ticker, raw_df in data_dict.items():
        df = build_features(raw_df)
        df = triple_barrier_labels(df, pt_sl=pt_sl, t_days=t_days)
        df['ticker'] = ticker
        rows.append(df)
    full = pd.concat(rows).dropna()
    # Keep only rows where tb_label != 0 (reduce micro noise) or keep all? we'll keep all and experiment
    # features
    feature_cols = [c for c in full.columns if c not in ['tb_label','tb_exit_date','tb_exit_ret','ticker','Close','Open','High','Low','Adj Close','Volume','typ_price','Date','index'] and not c.startswith('Unnamed')]
    # safe intersection
    feature_cols = [c for c in feature_cols if full[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
    print(f"Features used: {feature_cols}")

    X = full[feature_cols].copy()
    y = full['tb_label'].copy()
    # map labels to Buy/Hold/Sell strings for compatibility
    y_str = y.map({1:'Buy', -1:'Sell', 0:'Hold'}).values

    le = LabelEncoder()
    y_enc = le.fit_transform(y_str)

    # Purged K-Fold
    times = full.index
    pkf = purged_kfold_indices(times, n_splits=5, embargo_pct=0.01)

    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(pkf, start=1):
        print(f"Fold {fold}: train {len(train_idx)} test {len(test_idx)}")
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y_enc[train_idx]
        y_test = y_enc[test_idx]
        # scaler
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.03,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=SEED, use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        probs = model.predict_proba(X_test_s)
        print(classification_report(y_test, preds, target_names=le.classes_))
        # backtest per-fold (use full rows indexing)
        df_test = full.iloc[test_idx]
        bt_df, metrics = backtest_with_preds(df_test, preds, prob=probs, le=le)
        fold_results.append({'fold': fold, 'metrics': metrics, 'curve': bt_df})
        print("Fold metrics:", metrics)

    # Entrenar final sobre todo el dataset (usando scaler del último fold para consistencia)
    final_scaler = StandardScaler().fit(X)
    X_s_full = final_scaler.transform(X)
    final_model = xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03,
                                    subsample=0.8, colsample_bytree=0.8,
                                    random_state=SEED, use_label_encoder=False, eval_metric='mlogloss')
    final_model.fit(X_s_full, y_enc)

    # Guardar artefactos
    os.makedirs('models', exist_ok=True)
    dump(final_model, 'models/final_xgb.joblib')
    dump(final_scaler, 'models/final_scaler.joblib')
    with open('models/feature_cols.pkl','wb') as f:
        pickle.dump(feature_cols, f)
    with open('models/label_encoder.pkl','wb') as f:
        pickle.dump(le, f)
    with open('models/fold_results.pkl','wb') as f:
        pickle.dump(fold_results, f)
    print("Model saved in models/")

    return final_model, final_scaler, feature_cols, le, fold_results, full

# ---------------------------
# 7. SHAP EXPLAIN (post-train)
# ---------------------------
def shap_explain(model, scaler, X_raw: pd.DataFrame, feature_cols: List[str], n_samples=200):
    """
    Calcula y guarda shap summary plot y valores para interpretabilidad.
    """
    X = X_raw[feature_cols].copy()
    Xs = scaler.transform(X)
    explainer = shap.Explainer(model)
    # sample for speed
    idx = np.random.choice(np.arange(len(Xs)), size=min(n_samples, len(Xs)), replace=False)
    shap_values = explainer(Xs[idx]).values
    # summary plot guardado
    os.makedirs('reports', exist_ok=True)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    # shap.summary_plot works mejor interactivo; guardamos una figura básica
    shap.summary_plot(shap_values, X.iloc[idx], show=False)
    plt.savefig('reports/shap_summary.png', dpi=150)
    plt.close()
    # guardar valores
    with open('reports/shap_values.pkl','wb') as f:
        pickle.dump({'shap_values': shap_values, 'X_sample': X.iloc[idx]}, f)
    print("SHAP saved in reports/")

# ---------------------------
# 8. MAIN
# ---------------------------
if __name__ == "__main__":
    # EJEMPLO de uso
    # 1) sustituye por tu función que coge tickers de la bolsa española y trae 5 años
   
    data, no = get_tickers_data()
    model, scaler, features, le, folds, full_df = train_pipeline(data, pt_sl=(0.015, 0.01), t_days=5)
    shap_explain(model, scaler, full_df, features, n_samples=300)
    print("Pipeline completado.")
