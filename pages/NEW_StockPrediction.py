import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

st.title("📈 Quant Classification Backtest — IBEX35")

with open("model_classif.pkl", "rb") as f:
    data = pickle.load(f)

backtest = data["backtest"]

for fold_data in backtest:
    fold = fold_data["fold"]
    metrics = fold_data["metrics"]
    curve = fold_data["curve"]

    st.header(f"Fold {fold}")

    st.subheader("Metrics")
    st.table(pd.DataFrame([metrics]))

    st.subheader("Equity Curve")

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(curve["cum_strategy"], label="Strategy")
    ax.plot(curve["cum_buyhold"], label="Buy & Hold")
    ax.legend()
    st.pyplot(fig)

    st.markdown("---")
