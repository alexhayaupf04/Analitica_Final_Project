# app.py
import streamlit as st

st.set_page_config(page_title="Visual Analytics- IBEX35", layout="wide")

# --- Header ---
st.title("Analítica Visual - IBEX 35")
st.markdown(
    """
    **Descripción:** App interactiva que descarga datos desde *yfinance*, calcula KPIs del IBEX-35,
    grafica la evolución de los 5 valores más importantes por capitalización y muestra un treemap por sector
    con crecimiento/decrecimiento.
    """
)

# show image
st.image("trading.jpg", use_container_width=True)
