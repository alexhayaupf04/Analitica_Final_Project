import streamlit as st
import yfinance as yf
import pandas as pd
import pickle
import numpy as np

st.set_page_config(page_title="ML IBEX35 Signals", layout="wide")

st.title("IBEX 35 - ML Predictions")
st.write("Selecciona el modelo para predecir Buy/Hold/Sell")


