import streamlit as st
import yfinance as yf
import pandas as pd
import pickle
import numpy as np
from utils import get_ibex_tickers, get_tickers_data
from ml_models import load_r_forest, load_xgboost, predict_today_for_all
st.set_page_config(page_title="ML IBEX35 Signals", layout="wide")

st.title("IBEX 35 - ML Predictions")
st.markdown("""Select a model to generate 5-day ahead Buy / Hold / Sell signals.""")

tickers = get_ibex_tickers()
price_data, _ = get_tickers_data()

with st.spinner(""" Loading today's predictions..."""):
    xgboost = load_xgboost()
    r_forest = load_r_forest()

    df_xpreds = predict_today_for_all(xgboost, price_data)
    df_rpreds = predict_today_for_all(r_forest, price_data)

buy_xpreds = df_xpreds[df_xpreds["prediction"] == "Buy"]
top_3_xbuy = buy_xpreds.sort_values(ascending=False, ignore_index=True, by="prob_buy")[["ticker","prob_buy"]].head(3)

buy_rpreds = df_rpreds[df_rpreds["prediction"] == "Buy"]
top_3_rbuy = buy_rpreds.sort_values(ascending=False, ignore_index=True, by="prob_buy")[["ticker", "prob_buy"]].head(3)


sell_xpreds = df_xpreds[df_xpreds["prediction"] == "Sell"]
top_3_xsell = sell_xpreds.sort_values(ascending=False, ignore_index=True, by="prob_sell")[["ticker", "prob_sell"]].head(3)

sell_rpreds = df_rpreds[df_rpreds["prediction"] == "Sell"]
top_3_rsell = sell_rpreds.sort_values(ascending=False, ignore_index=True, by="prob_sell")[["ticker", "prob_sell"]].head(3)


hold_xpreds = df_xpreds[df_xpreds["prediction"] == "Hold"]
top_3_xhold = hold_xpreds.sort_values(ascending=False, ignore_index=True, by="prob_hold")[["ticker", "prob_hold"]].head(3)

hold_rpreds = df_rpreds[df_rpreds["prediction"] == "Hold"]
top_3_rhold = hold_rpreds.sort_values(ascending=False, ignore_index=True, by="prob_hold")[["ticker", "prob_hold"]].head(3)


tab1, tab2 = st.tabs(["Xgboost", "Random Forest"])

with tab1:
    st.header("Xgboost")
    st.markdown("""
        **XGBoost** is a gradient-boosted trees model optimized for structured financial data.
                
        **Pros:** High predictive accuracy, captures nonlinear patterns.
        **Cons:** More complex and slower to train than Random Forest.
    """)
    col1,col2,col3 = st.columns(3)
    with col1:
        st.subheader("🟢 Buy ")
        st.dataframe(top_3_xbuy, hide_index=True)
    with col2:
        st.subheader("🟡 Hold")
        st.dataframe(top_3_xhold, hide_index = True)
    with col3:
        st.subheader("🔴 Sell ")
        st.dataframe(top_3_xsell, hide_index = True)
    st.markdown("""
    These are the most recommended actions from the model. The probability of Buy/Hold/Sell ranges from 0 to 1.
    The sum of the three probabilities does not exceed 1. A probability above 0.45 indicates a strong signal for that action.
    """)
    st.markdown("---")
    st.subheader("Predict an asset:")
    st.markdown("View the 5-day ahead prediction for a specific ticker.")
    xselected_ticker = st.selectbox("Select an asset",tickers, key=6767, index = None)
    xpred_ticker = df_xpreds[df_xpreds["ticker"] == xselected_ticker][["ticker", "prediction", "prediction_proba"]]
    if xselected_ticker:
        st.dataframe(xpred_ticker, hide_index=True)

    
with tab2:
    st.header("Random Forest")
    st.markdown("""
        **Random Forest** is an ensemble of decision trees designed for strong stability.
                
        **Pros:** Very robust, low risk of overfitting, easy to interpret.
        **Cons:** May miss subtle patterns that XGBoost can capture.
        """)
    col4,col5,col6 = st.columns(3)
    with col4:
        st.subheader("🟢 Buy ")
        st.dataframe(top_3_rbuy, hide_index= True)
    with col5:    
        st.subheader("🟡 Hold")
        st.dataframe(top_3_rhold, hide_index = True)
    with col6:
        st.subheader("🔴 Sell ")
        st.dataframe(top_3_rsell, hide_index=True)
    st.markdown("""
    These are the most recommended actions from the model. The probability of Buy/Hold/Sell ranges from 0 to 1.
    The sum of the three probabilities does not exceed 1. A probability above 0.45 indicates a strong signal for that action.
    """)
    st.markdown("---")
    st.subheader("Predict an asset:")
    st.markdown("View the 5-day ahead prediction for a specific ticker.")
    rselected_ticker = st.selectbox("Select an asset:",tickers, key=6969, index = None)
    rpred_ticker = df_rpreds[df_rpreds["ticker"] == rselected_ticker][["ticker","prediction", "prediction_proba"]]
    if rselected_ticker:
        st.dataframe(rpred_ticker, hide_index=True)
    


