# app.py
import streamlit as st
from utils import get_ibex_tickers, get_ibex_tickers_name
st.set_page_config(page_title="Visual Analytics- IBEX35", layout="wide")

# --- Header ---
col1, col2 = st.columns(2)
with col1:
    st.title("VA - IBEX 35")
    st.markdown(
    """
    You can navigate through the different project sections using the tab.
    1. Stock Overview
    2. Stock Prediction
    3. Trading Simulation
    4. AI Explainability

    """
)
with col2: 
    st.image("imgs/trading.jpg", width="content")

col3, col4 = st.columns(2)
with col3:

    st.subheader("Key concepts")
    st.markdown(
    """
    **Ticker:** Short code representing a stock (e.g., SAN for Santander).  

    **Stock / Share:** Ownership unit of a company that can be traded.  

    **Stock Market:** Platform where shares are bought and sold.  

    **RSI (Relative Strength Index):** Measures momentum and overbought/oversold conditions.

    **Momentum / MAC:** Rate of price change; identifies trend strength.  

    **Moving Average (MA):** Average price over a period; helps smooth out trends.  

    **Volatility:** How much price fluctuates; higher volatility = higher risk.

    **Backtest:** Simulation of a trading strategy using historical data to evaluate performance.  

    **Buy & Hold:** Simple strategy of buying an asset and holding it until the end.  

    **Strategy Return:** Profit or loss from following a model's signals (Buy/Sell/Hold).  

    **Max Drawdown:** Largest drop from peak equity in a strategy; indicates risk.  
    
    **Sharpe Ratio:** Risk-adjusted return; higher means better reward per unit of risk.   
    """
    )
with col4:

    st.subheader("IBEX35 companies", help= "")
    tickers_abr = get_ibex_tickers()
    tickers_name = get_ibex_tickers_name()
    for abr, name in zip(tickers_abr, tickers_name):
        st.write(f"**{abr}** → {name}")

    