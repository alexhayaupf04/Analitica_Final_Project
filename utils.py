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

def get_ibex_tickers_name() -> List[str]:
    tickers = [
    'ACS, Actividades de Construcción y Servicios, S.A.',
    'Acerinox, S.A.',
    'Aena S.M.E., S.A.',
    'Amadeus IT Group, S.A.',
    'Acciona, S.A.',
    'Corporación Acciona Energías Renovables, S.A.',
    'Banco Bilbao Vizcaya Argentaria, S.A.',
    'Bankinter, S.A.',
    'CaixaBank, S.A.',
    'Cellnex Telecom, S.A.',
    'Inmobiliaria Colonial, SOCIMI, S.A.',
    'Endesa, S.A.',
    'Enagás, S.A.',
    'Fluidra, S.A.',
    'Ferrovial SE',
    'Grifols, S.A.',
    'International Consolidated Airlines Group S.A.',
    'Iberdrola, S.A.',
    'Industria de Diseño Textil, S.A.',
    'Logista Integral, S.A.',
    'Mapfre, S.A.',
    'MERLIN Properties SOCIMI, S.A.',
    'ArcelorMittal S.A.',
    'Naturgy Energy Group, S.A.',
    'Puig Brands SA',
    'Redeia Corporación, S.A.',
    'Banco de Sabadell, S.A.',
    'Banco Santander, S.A.',
    'Telefónica, S.A.',
    'Unicaja Banco, S.A.'
    ]
    return tickers

def get_indexes()-> List[str]:
    indexes = [
        "^IBEX"
        ]
    return indexes

def download_tickers_data(tickers: List[str], period="5y", interval="1d") -> Tuple[Dict[str, pd.DataFrame], Dict[str, dict]]:
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

def download_index_data(indexes: List[str], period="5y", interval="1d") -> Dict[str, pd.DataFrame]:
    """
    Descarga series históricas (Adj Close) para cada ticker con yfinance.
    Devuelve (price_data_dict, info_dict)
    """
    price_dict = {}
    # yfinance supports batch download
    try:
        yf_tickers = yf.Tickers(" ".join(indexes))
        for i in indexes:
            try:
                tk = yf.Ticker(i)
                hist = tk.history(period=period, interval=interval, auto_adjust=False)
                if not hist.empty:
                    # asegurar columna 'Adj Close'
                    price_dict[i] = hist.rename_axis("Date")            
            except Exception:
                continue
    except Exception:
        # fallback: descargar uno a uno
        for i in indexes:
            try:
                tk = yf.Ticker(i)
                hist = tk.history(period=period, interval=interval, auto_adjust=False)
                if not hist.empty:
                    price_dict[i] = hist.rename_axis("Date")
            except Exception:
                continue
    return price_dict

def compute_top_n_by_marketcap(info_meta: Dict[str, dict], n=5) -> List[str]:
    """
    Ordena tickers por marketCap si está disponible y devuelve top n.
    """
    df = pd.DataFrame.from_dict(info_meta, orient="index")
    if df.empty or "marketCap" not in df.columns:
        return []
    df_sorted = df.sort_values(by="marketCap", ascending=False).dropna(subset=["marketCap"])
    return df_sorted.head(n).index.tolist()

def get_index_kpi(index, price_data):

    df= price_data.get(index)
    """
    Devuelve un dict con {last, prev, pct_change} para el ticker ^IBEX si está disponible.
    """
    last = df["Close"].iloc[-1]
    prev_max = df["Close"].iloc[0]
    prev_day = df["Close"].iloc[-2]
    prev_month = df["Close"].iloc[-30]
    prev_year = df["Close"].iloc[-365]
    
    pct_max = (last - prev_max) / prev_max if prev_max != 0 else 0.0
    pct_day = (last - prev_day) / prev_day if prev_day != 0 else 0.0
    pct_month = (last - prev_month) / prev_month if prev_month != 0 else 0.0
    pct_year = (last - prev_year) / prev_year if prev_year != 0 else 0.0
    
    dict = {"last": float(last), "first": float(prev_max), "pct_change_year": float(pct_year), 
            "pct_change_month": float(pct_month), "pct_change_day": float(pct_day), "pct_change_max":
            float(pct_max)}
    
    return dict
    

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
      
        series = df["Close"]
        
        start = series.iloc[-2]
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

def save_cache_tickers(price_data, info_meta, filename="data/ibex_tickers.pkl"):
    """Guarda los diccionarios de datos en un archivo pickle."""
    os.makedirs("data", exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump({
            "timestamp": datetime.now(),
            "price_data": price_data,
            "info_meta": info_meta
        }, f)

def load_cache_tickers(filename="data/ibex_tickers.pkl"):
    """Carga los datos desde un pickle si existe."""
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
            return data
    except Exception:
        return None
    
def save_cache_index(price_data, filename="data/ibex.pkl"):
    """Guarda los diccionarios de datos en un archivo pickle."""
    os.makedirs("data", exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump({
            "timestamp": datetime.now(),
            "price_data": price_data,
        }, f)
    return None

def load_cache_index(filename="data/ibex.pkl"):
    """Carga los datos desde un pickle si existe."""
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
            return data
    except Exception:
        return None

def refresh_tickers_data():
    cache = load_cache_tickers()
    cache_time = cache["timestamp"].date()
    now_time = datetime.now().date()

    # If we already have up to date data no need to download
    if cache_time == now_time: 
        return False
    return True

def refresh_indexes_data():
    cache = load_cache_index()
    cache_time = cache["timestamp"].date()
    now_time = datetime.now().date()

    # If we already have up to date data no need to download
    if cache_time == now_time: 
        return False
    return True

def get_tickers_data():
    if refresh_tickers_data():
        price, info = download_tickers_data(get_ibex_tickers())
        save_cache_tickers(price, info)
    else:
        cache = load_cache_tickers()
        price = cache["price_data"]
        info = cache["info_meta"]

    return price, info

def get_indexes_data():
    if refresh_indexes_data():
        price = download_index_data(get_indexes())
        save_cache_index(price)
    else:
        cache = load_cache_index()
        price = cache["price_data"]

    return price

if __name__ == "__main__": 
    price_data, info_data = download_index_data(get_ibex_tickers())
    i_price_data = download_tickers_data(get_indexes())
    save_cache_tickers(price_data, info_data)
    save_cache_index(i_price_data)