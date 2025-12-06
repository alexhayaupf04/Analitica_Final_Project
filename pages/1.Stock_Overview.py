import streamlit as st
import pandas as pd
import plotly.express as px
from utils import (
    get_ibex_tickers,
    get_indexes,
    get_tickers_data,
    get_indexes_data,
    compute_top_n_by_marketcap,
    get_index_kpi,
    compute_sector_changes
)


st.set_page_config(page_title="IBEX35 - Overview", layout="wide")

# --- Header ---
st.title("IBEX 35 - Overview")

st.markdown(
    """
    Dinamically dowloads data from yfinance API (Yahoo). Shows general state of spanish stock market. 
    """
)

tickers = get_ibex_tickers()
indexes = get_indexes()
index = "".join(indexes)

with st.spinner("Loading data..."):

    price_data,info_meta = get_tickers_data()
    i_price_data = get_indexes_data()

# --- KPI index ---
index_kpi = get_index_kpi(index, i_price_data)

col1, col2, col3 = st.columns(3)

with col1:
    st.header("IBEX 35")

with col3:
    options = ["Historical", "1Y", "1M", "1D"]
    evolution = st.selectbox(label = "Performance:", index=1, options= options, placeholder= "1Y")
    ev_to_pct = {"Historical": "pct_change_max","1Y": "pct_change_year", 
                    "1M": "pct_change_month", "1D": "pct_change_day"}
    pct = ev_to_pct[evolution]

with col2:
    up = index_kpi[pct] >= 0
    color = "green" if up else "red"
    arrow = "▲" if up else "▼"
    st.metric(label="Last closing price", value=f"{index_kpi['last']:.2f} €", delta=f"{index_kpi[pct]*100:.2f}%")

st.markdown("---")
# --- Top N by market cap ---
st.header("Stock price evolution")

st.markdown(
    """
    View the price performance of the selected tickers.
    By default, the chart displays the top 5 companies by market capitalization.
    """
)

selected_tickers = st.sidebar.multiselect("Tickers:",options=tickers, placeholder= "Top5 Marketcap")
select_all = st.sidebar.toggle("Select all", value=False)
if select_all:
    selected_tickers = get_ibex_tickers()
if not selected_tickers:

    top_n= 5
    top_tickers = compute_top_n_by_marketcap(info_meta, n=top_n)
    if not top_tickers:
        top_tickers = list(price_data.keys())[:top_n]
else: 
    top_tickers = [t for t in selected_tickers if t in price_data]     

# prepare dataframe for plotting
plot_df = pd.DataFrame()
for t in top_tickers:
    df = price_data.get(t)
    if df is None or df.empty:
        continue
    tmp = df[["Close"]].rename(columns={"Close": t})
    plot_df = pd.concat([plot_df, tmp], axis=1)

if plot_df.empty:
    st.error("No hay series de precios disponibles para los top tickers.")
else:
    fig = px.line(plot_df, labels={"value":"Closing Price €", "index":"Date"}, title= None)
    st.plotly_chart(fig, width="stretch")

col5,col6 = st.columns(2)

with col5:
    st.markdown(
    """
    Use the sidebar to:
    - Select any combination of tickers manually, or
    - Enable **Select all** to display every IBEX 35 ticker.
    """
)
with col6:
    st.markdown(
    """
    You can interact with the chart:
    - Click and drag to zoom into any area.
    - Hover over the lines to see detailed price information
    """
)

st.markdown("---")
# --- Treemap por sector ---
st.header("Sector Performance")
sector_df = compute_sector_changes(info_meta, price_data)
sector_df["pct_change"] = round(sector_df["pct_change"]*100,2)
st.markdown("""
    This treemap shows today's assets performance, grouped by sector.
    Green indicates the asset is up compared to yesterday, while red indicates it is down.
            """)
if sector_df is None or sector_df.empty:
    st.warning("No se pudo calcular treemap por sector.")
else:
    # Treemap: path sector -> ticker, values = marketcap (or abs change) and color by pct_change
    fig_treemap = px.treemap(
        sector_df,
        path=["sector", "ticker"],
        values="marketCap_fallback",
        color="pct_change",
        hover_data=["pct_change", "marketCap_fallback"],
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        title = None,
    )
    st.plotly_chart(fig_treemap, use_container_width=True)

st.markdown(
    """ The treemap is interactive: click on a sector or an individual asset to zoom in,
    and hover to see detailed information.""")


