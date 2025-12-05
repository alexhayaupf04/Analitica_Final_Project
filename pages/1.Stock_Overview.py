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
    **Descripción:** App interactiva que descarga datos desde *yfinance*, calcula KPIs del IBEX-35,
    grafica la evolución de los 5 valores más importantes por capitalización y muestra un treemap por sector
    con crecimiento/decrecimiento.
    """
)

tickers = get_ibex_tickers()
indexes = get_indexes()
index = "".join(indexes)

with st.spinner("Cargando los datos..."):

    price_data,info_meta = get_tickers_data()
    i_price_data = get_indexes_data()

# --- KPI index ---
index_kpi = get_index_kpi(index, i_price_data)
if index_kpi is not None:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.header("IBEX 35")
    
    with col3:
        options = ["Histórico", "Anual", "Mensual", "Diario"]
        evolution = st.selectbox(label = "Evolución:", index=1, options= options, placeholder= "Anual")
        ev_to_pct = {"Histórico": "pct_change_max","Anual": "pct_change_year", 
                     "Mensual": "pct_change_month", "Diario": "pct_change_day"}
        pct = ev_to_pct[evolution]

    with col2:
        up = index_kpi[pct] >= 0
        color = "green" if up else "red"
        arrow = "▲" if up else "▼"
        st.metric(label="IBEX35 (último cierre)", value=f"{index_kpi['last']:.2f}", delta=f"{index_kpi[pct]*100:.2f}%")

    
else:
    st.warning("No se pudo descargar el índice IBEX (^IBEX).")

# --- Top N by market cap ---
st.header("Evolución de precio")
top_n= 30
top_tickers = compute_top_n_by_marketcap(info_meta, n=top_n)
if not top_tickers:
    st.warning("No hay datos de capitalización para ordenar; se mostrarán los primeros tickers.")
    top_tickers = list(price_data.keys())[:top_n]
    
selected_tickers = st.sidebar.multiselect("Activos:",options=tickers, placeholder= "Top5 Marketcap")

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
    fig = px.line(plot_df, labels={"value":"Close", "index":"Date"}, title= None)
    st.plotly_chart(fig, use_container_width=True)
st.text("En los controles puedes ajustar los parámetros del gráfico (período, intervalo y activos)." \
"Para ampliar una zona, haz clic y arrastra sobre el área que quieras ver con más detalle. " \
"Al pasar el ratón por encima aparecerá información adicional.")

# --- Treemap por sector ---
st.header("Treemap por sector")
sector_df = compute_sector_changes(info_meta, price_data)
st.text("Mostramos el comportamiento de los activos en la sesión actual ya sea por sector o de " \
"forma individual. El color verde indica que el activo ha subido respecto a ayer, y el rojo que ha bajado.")
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
        title = None,
    )
    st.plotly_chart(fig_treemap, use_container_width=True)
st.text("El gráfico es interactivo: puedes hacer clic en un sector o un activo para " \
"verlos en detalle. Si pasas el ratón por encima, aparecerá información adicional.")


