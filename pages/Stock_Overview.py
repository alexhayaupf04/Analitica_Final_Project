import streamlit as st
from overview_utils import (
    get_ibex_tickers,
    download_tickers_data,
    compute_top_n_by_marketcap,
    get_index_kpi,
    compute_sector_changes,
    load_cache,
    save_cache
)
import pandas as pd
import plotly.express as px


st.set_page_config(page_title="IBEX35 - Overview", layout="wide")

# --- Header ---
st.title("IBEX 35 - Overview")
st.markdown(
    """
    **Descripción:** App interactiva que descarga datos desde *yfinance*, calcula KPIs del IBEX-35,
    grafica la evolución de los 5 valores más importantes por capitalización y muestra un treemap por sector
    con crecimiento/decrecimiento.
    """
)

# --- Sidebar controls ---
st.sidebar.header("Controles")
period = st.sidebar.selectbox("Periodo para series históricas", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
interval = st.sidebar.selectbox("Intervalo (yfinance)", ["1d", "1wk", "1mo"], index=0)
top_n = st.sidebar.number_input("Número de top tickers a mostrar", min_value=3, max_value=10, value=5, step=1)
refresh = st.sidebar.button("Actualizar datos")

st.sidebar.subheader("Cache")
use_cache = st.sidebar.checkbox("Usar cache si existe", value=True)
force_refresh = st.sidebar.checkbox("Forzar actualización", value=False)

# --- Load tickers ---
with st.spinner("Obteniendo lista de componentes del IBEX..."):
    tickers = get_ibex_tickers()

if not tickers:
    st.error("No se pudieron obtener los tickers del IBEX. Revisa la conexión.")
    st.stop()

st.sidebar.write(f"Componentes detectados: {len(tickers)}")

# --- Download data or cache---
cache = load_cache("ibex_cache.pkl") if use_cache else None

if cache and not force_refresh:
    st.success("Datos cargados desde cache.")
    price_data = cache["price_data"]
    info_meta = cache["info_meta"]
else:
    with st.spinner("Descargando datos desde yfinance..."):
        price_data, info_meta = download_tickers_data(tickers, period=period, interval=interval)

    if use_cache:
        save_cache(price_data, info_meta, "ibex_cache.pkl")
        st.info("Cache actualizado.")


# --- KPI index ---
st.header("KPI — IBEX35")
index_kpi = get_index_kpi(period=period, interval=interval)
if index_kpi is not None:
    col1, col2 = st.columns([1,3])
    with col1:
        up = index_kpi["pct_change"] >= 0
        color = "green" if up else "red"
        arrow = "▲" if up else "▼"
        st.metric(label="IBEX35 (último cierre)", value=f"{index_kpi['last']:.2f}", delta=f"{index_kpi['pct_change']*100:.2f}%")
    with col2:
        st.write(f"Periodo usado: **{period}**, intervalo: **{interval}**")
else:
    st.warning("No se pudo descargar el índice IBEX (^IBEX).")

# --- Top N by market cap ---
st.header(f"Evolución de precio — Top {top_n} por capitalización")
top_tickers = compute_top_n_by_marketcap(info_meta, n=top_n)
if not top_tickers:
    st.warning("No hay datos de capitalización para ordenar; se mostrarán los primeros tickers.")
    top_tickers = list(price_data.keys())[:top_n]

# prepare dataframe for plotting
plot_df = pd.DataFrame()
for t in top_tickers:
    df = price_data.get(t)
    if df is None or df.empty:
        continue
    tmp = df[["Adj Close"]].rename(columns={"Adj Close": t})
    plot_df = pd.concat([plot_df, tmp], axis=1)

if plot_df.empty:
    st.error("No hay series de precios disponibles para los top tickers.")
else:
    fig = px.line(plot_df, labels={"value":"Adj Close", "index":"Date"}, title="Precio Ajustado — Top tickers")
    st.plotly_chart(fig, use_container_width=True)

# --- Treemap por sector ---
st.header("Treemap por sector — crecimiento / decrecimiento")
sector_df = compute_sector_changes(info_meta, price_data)
if sector_df is None or sector_df.empty:
    st.warning("No se pudo calcular treemap por sector (faltan sectores o datos).")
else:
    # Treemap: path sector -> ticker, values = marketcap (or abs change) and color by pct_change
    fig_treemap = px.treemap(
        sector_df,
        path=["sector", "ticker"],
        values="marketCap_fallback",
        color="pct_change",
        hover_data=["pct_change", "marketCap_fallback"],
        color_continuous_scale="RdYlGn",
        title="Treemap: sectores (color = % cambio en periodo seleccionado)"
    )
    st.plotly_chart(fig_treemap, use_container_width=True)


