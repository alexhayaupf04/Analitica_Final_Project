import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from ml_models import load_r_forest, load_xgboost
st.title("IBEX35 - Trading Simulation")
st.markdown("""
Explain features model etc, what are we doing
            """)
tab1, tab2 = st.tabs(["Xgboost", "Random Forest"])

xgboost = load_xgboost()
r_forest = load_r_forest()

x_backtest = xgboost["backtest"]
r_backtest = r_forest["backtest"]

x_fold = x_backtest[-1]
r_fold = r_backtest[-1]

x_metrics = x_fold["metrics"]
x_sr = x_metrics["Strategy Return"]
x_br = x_metrics["BuyHold Return"]
x_md = x_metrics["Max Drawdown"]
x_sh = x_metrics["Sharpe Ratio"]
x_curve = x_fold["curve"]

r_metrics = r_fold["metrics"]
r_curve = r_fold["curve"]
r_sr = r_metrics["Strategy Return"]
r_br = r_metrics["BuyHold Return"]
r_md = r_metrics["Max Drawdown"]
r_sh = r_metrics["Sharpe Ratio"]
r_curve = r_fold["curve"]



with tab1:
    st.header("Xgboost")
    st.subheader("Equity Curve")
    
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(x_curve["cum_strategy"], label="Strategy")
    ax.plot(x_curve["cum_buyhold"], label="Buy & Hold")
    ax.legend()
    st.pyplot(fig)

    xcol1, xcol2 = st.columns(2)
    with xcol1:
        st.subheader("Strategy Return")
        st.text(f"{round(x_sr,0)} EUR")
        st.subheader("Max Drawdown")
        st.text(f"{round(x_md,2)} EUR")

    with xcol2:
        st.subheader("BuyHold Return")
        st.text(f"{round(x_br,2)} EUR")
        st.subheader("Sharpe Ratio")
        st.text(f"{round(x_sh,2)}")

with tab2:
    st.header("Random Forest")
    st.subheader("Equity Curve")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(r_curve["cum_strategy"], label="Strategy")
    ax.plot(r_curve["cum_buyhold"], label="Buy & Hold")
    ax.legend()
    st.pyplot(fig)

    xcol1, xcol2 = st.columns(2)
    with xcol1:
        st.subheader("Strategy Return")
        st.text(f"{round(r_sr,0)} EUR")
        st.subheader("Max Drawdown")
        st.text(f"{round(r_md,2)} EUR")

    with xcol2:
        st.subheader("BuyHold Return")
        st.text(f"{round(r_br,2)} EUR")
        st.subheader("Sharpe Ratio")
        st.text(f"{round(r_sh,2)}")
   




