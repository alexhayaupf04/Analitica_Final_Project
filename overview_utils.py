import yfinance as yf
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import pickle
import os
from datetime import datetime

def get_ibex_tickers() -> List[str]:
    
    tickers = [
    "ACS.MC", "ACX.MC", "AENA.MC", "AMS.MC", "ANA.MC", "ANE.MC",
    "BBVA.MC", "BKT.MC", "CABK.MC", "CLNX.MC", "COL.MC", "ELE.MC",
    "ENG.MC", "FDR.MC", "FER.MC", "GRF.MC", "IAG.MC", "IBE.MC",
    "ITX.MC", "LOG.MC", "MAP.MC", "MRL.MC", "MTS.MC", "NTGY.MC",
    "PUIG.MC", "RED.MC", "SAB.MC", "SAN.MC", "TEF.MC", "UNI.MC"
    ]
    return tickers

def download_tickers_data(tickers: List[str], period="1y", interval="1d") -> Tuple[Dict[str, pd.DataFrame], Dict[str, dict]]:
    """
    Descarga series históricas (Adj Close) para cada ticker con yfinance.
    Devuelve (price_data_dict, info_dict)
    """
    price_dict = {}
    info_dict = {}
    # yfinance supports batch download
    try:
        yf_tickers = yf.Tickers(" ".join(tickers))
        for t in tickers:
            try:
                tk = yf.Ticker(t)
                hist = tk.history(period=period, interval=interval, auto_adjust=False)
                if not hist.empty:
                    # asegurar columna 'Adj Close'
                    price_dict[t] = hist.rename_axis("Date")
                # info (para marketCap / sector)
                info = tk.info or {}
                info_dict[t] = {
                    "marketCap": info.get("marketCap"),
                    "sector": info.get("sector") or "Unknown",
                    "longName": info.get("longName") or t
                }
            except Exception:
                continue
    except Exception:
        # fallback: descargar uno a uno
        for t in tickers:
            try:
                tk = yf.Ticker(t)
                hist = tk.history(period=period, interval=interval, auto_adjust=False)
                if not hist.empty:
                    price_dict[t] = hist.rename_axis("Date")
                info = tk.info or {}
                info_dict[t] = {
                    "marketCap": info.get("marketCap"),
                    "sector": info.get("sector") or "Unknown",
                    "longName": info.get("longName") or t
                }
            except Exception:
                continue
    return price_dict, info_dict

def compute_top_n_by_marketcap(info_meta: Dict[str, dict], n=5) -> List[str]:
    """
    Ordena tickers por marketCap si está disponible y devuelve top n.
    """
    df = pd.DataFrame.from_dict(info_meta, orient="index")
    if df.empty or "marketCap" not in df.columns:
        return []
    df_sorted = df.sort_values(by="marketCap", ascending=False).dropna(subset=["marketCap"])
    return df_sorted.head(n).index.tolist()

def get_index_kpi(period="1y", interval="1d"):
    """
    Devuelve un dict con {last, prev, pct_change} para el ticker ^IBEX si está disponible.
    """
    try:
        t = yf.Ticker("^IBEX")
        hist = t.history(period=period, interval=interval)
        if hist.empty:
            return None
        last = hist["Close"].iloc[-1]
        prev = hist["Close"].iloc[0]
        pct = (last - prev) / prev if prev != 0 else 0.0
        return {"last": float(last), "first": float(prev), "pct_change": float(pct)}
    except Exception:
        return None

def compute_sector_changes(info_meta: Dict[str, dict], price_data: Dict[str, pd.DataFrame]):
    """
    Calcula para cada ticker su cambio porcentual en el periodo y agrupa por sector.
    Devuelve un DataFrame con columnas: ticker, sector, pct_change, marketCap_fallback
    """
    rows = []
    for t, meta in info_meta.items():
        df = price_data.get(t)
        if df is None or df.empty:
            continue
        # usar 'Adj Close' si existe, si no 'Close'
        if "Adj Close" in df.columns:
            series = df["Adj Close"]
        elif "Adjclose" in df.columns:
            series = df["Adjclose"]
        else:
            series = df["Close"]
        if series.empty:
            continue
        start = series.iloc[0]
        end = series.iloc[-1]
        pct = (end - start) / start if start != 0 else np.nan
        marketcap = meta.get("marketCap") or 0
        sector = meta.get("sector") or "Unknown"
        rows.append({
            "ticker": t,
            "sector": sector,
            "pct_change": float(pct) if not np.isnan(pct) else 0.0,
            "marketCap_fallback": marketcap if marketcap else 1.0
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # ensure numeric types
    df["marketCap_fallback"] = pd.to_numeric(df["marketCap_fallback"], errors="coerce").fillna(1.0)
    return df

def save_cache(price_data, info_meta, filename="ibex_cache.pkl"):
    """Guarda los diccionarios de datos en un archivo pickle."""
    with open(filename, "wb") as f:
        pickle.dump({
            "timestamp": datetime.now(),
            "price_data": price_data,
            "info_meta": info_meta
        }, f)

def load_cache(filename="ibex_cache.pkl"):
    """Carga los datos desde un pickle si existe."""
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
            return data
    except Exception:
        return None