import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from utils.shocks import apply_rate_shocks

st.title("📈 Interest Rate Forecasting & Yield Curve Shock Simulator")

uploaded = st.file_uploader("Upload interest rate time-series CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    df.columns = ["date", "rate"]
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    
    st.line_chart(df["rate"], height=250)
    
    st.subheader("ARIMA Forecast")
    model = ARIMA(df["rate"], order=(2,1,2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    
    st.line_chart(forecast, height=200)
    
    st.subheader("Shock Scenarios")
    shock_type = st.selectbox("Select shock", ["+50bps", "+100bps", "-50bps"])
    shocked = apply_rate_shocks(df["rate"], shock_type)
    st.line_chart(shocked, height=200)
    
    st.success("MVP generated successfully!")
