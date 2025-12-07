import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
from ml_models import load_r_forest, load_xgboost

st.title("IBEX 35 – Trading Simulation")
st.markdown("""
This section presents a backtest simulation of trading strategies using XGBoost and Random Forest models.  

""")

# Load models
xgboost = load_xgboost()
r_forest = load_r_forest()

x_backtest = xgboost["backtest"][-1]
r_backtest = r_forest["backtest"][-1]

x_curve = x_backtest["curve"]
x_metrics = x_backtest["metrics"]

r_curve = r_backtest["curve"]
r_metrics = r_backtest["metrics"]

tab1, tab2 = st.tabs(["XGBoost", "Random Forest"])

def plot_equity_curve(curve):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve.index, y=curve["cum_strategy"], mode='lines', name='Strategy',
        line=dict(color='green')
    ))
    fig.add_trace(go.Scatter(
        x=curve.index, y=curve["cum_buyhold"], mode='lines', name='Buy & Hold',
        line=dict(color='blue')
    ))
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Cumulative Returns %",
        hovermode="x unified",
        template="plotly_white"
    )
    return fig

with tab1:
    st.header("XGBoost")
    st.subheader("Equity Curve")
    st.text("View the cumulative returns of a trading strategy over time.")
    st.plotly_chart(plot_equity_curve(x_curve), use_container_width=True)

    st.markdown(
    """
    You can interact with the chart:
    - Click and drag to zoom into any area.
    - Hover over the lines to see detailed price information
    """
    )
    st.markdown("---")

    col1, col2, col3 = st.columns([1.1,1.2,0.9])
    with col1:
        st.subheader("Strategy Return")
        st.text(f"{round(x_metrics['Strategy Return'], 2)} %")
        st.text("Following model signals")
    with col2:
        st.subheader("Buy&Hold Return")
        st.text(f"{round(x_metrics['BuyHold Return'], 2)} %")  
    with col3:
        st.subheader("Sharpe Ratio")
        st.text(f"{round(x_metrics['Sharpe Ratio'], 2)}")
        st.markdown("""**Good**""")

with tab2:
    st.header("Random Forest")
    st.subheader("Equity Curve")
    st.text("View the cumulative returns of a trading strategy over time.")
    st.plotly_chart(plot_equity_curve(r_curve), use_container_width=True)
    
    st.markdown(
    """
    You can interact with the chart:
    - Click and drag to zoom into any area.
    - Hover over the lines to see detailed price information
    """
    )

    st.markdown("---")

    col1, col2, col3 = st.columns([1.1,1.2,0.9])
    with col1:
        st.subheader("Strategy Return")
        st.text(f"{round(r_metrics['Strategy Return'], 2)} %")
        st.text("Following model signals.")
    with col2:
        st.subheader("Buy&Hold Return")
        st.text(f"{round(r_metrics['BuyHold Return'], 2)} %")
    with col3:
        st.subheader("Sharpe Ratio")
        st.text(f"{round(r_metrics['Sharpe Ratio'], 2)}")
        st.markdown("""**Good**""")
