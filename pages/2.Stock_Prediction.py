import streamlit as st
import yfinance as yf
import pandas as pd
import pickle
import numpy as np
from utils import get_ibex_tickers, get_tickers_data
from ml_models import build_features, load_r_forest, load_xgboost, predict_today_for_all
st.set_page_config(page_title="ML IBEX35 Signals", layout="wide")

st.title("IBEX 35 - ML Predictions")
st.write("Selecciona el modelo para predecir Buy/Hold/Sell")

tickers = get_ibex_tickers()
price_data, _ = get_tickers_data()

xgboost = load_xgboost()
r_forest = load_r_forest()

df_xpreds = predict_today_for_all(xgboost, price_data)
df_rpreds = predict_today_for_all(r_forest, price_data)

buy_xpreds = df_xpreds[df_xpreds["prediction"] == "Buy"]
top_5_xbuy = buy_xpreds.sort_values(ascending=False, ignore_index=True, by="prob_buy").head(3)

buy_rpreds = df_rpreds[df_rpreds["prediction"] == "Buy"]
top_5_rbuy = buy_rpreds.sort_values(ascending=False, ignore_index=True, by="prob_buy").head(3)


sell_xpreds = df_xpreds[df_xpreds["prediction"] == "Sell"]
top_5_xsell = sell_xpreds.sort_values(ascending=False, ignore_index=True, by="prob_sell").head(3)

sell_rpreds = df_rpreds[df_rpreds["prediction"] == "Sell"]
top_5_rsell = sell_rpreds.sort_values(ascending=False, ignore_index=True, by="prob_sell").head(3)


hold_xpreds = df_xpreds[df_xpreds["prediction"] == "Hold"]
top_5_xsell = hold_xpreds.sort_values(ascending=False, ignore_index=True, by="prob_hold").head(3)

sell_rpreds = df_rpreds[df_rpreds["prediction"] == "Hold"]
top_5_rsell = sell_rpreds.sort_values(ascending=False, ignore_index=True, by="prob_hold").head(3)


tab1, tab2 = st.tabs(["Xgboost", "Random Forest"])

with tab1:
    st.header("Xgboost")
    st.subheader("Buy")
    st.dataframe(top_5_xbuy)
    st.subheader("Hold")
    st.dataframe(top_5_xsell)
    st.subheader("Sell")
    st.dataframe(top_5_xsell)

with tab2:
    st.header("Random Forest")
    st.subheader("Buy")
    st.dataframe(top_5_rbuy)
    st.subheader("Hold")
    st.dataframe(top_5_rsell)
    st.subheader("Sell")
    st.dataframe(top_5_xsell)

st.subheader("Predicciones de Hoy")

data, _ = get_tickers_data()

st.head
st.multiselect()
