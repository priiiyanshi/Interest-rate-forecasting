import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from utils.shocks import apply_rate_shocks
from utils.preprocessing import clean_rate_data
st.title("Interest Rate Forecasting & Yield Curve Shock Simulator")
st.write("Upload a CSV with columns: date, rate")

uploaded = st.file_uploader("Upload interest rate CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    df.columns = ["date", "rate"]
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    st.subheader("Raw Rate Data")
    st.line_chart(df["rate"])

    st.subheader("ARIMA Forecast (30 days)")
    model = ARIMA(df["rate"], order=(2,1,2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)

    st.line_chart(forecast)

    st.subheader("Shock Scenarios")
    shock = st.selectbox("Choose shock", ["+50bps", "+100bps", "-50bps"])
    shocked_series = apply_rate_shocks(df["rate"], shock)

    st.line_chart(shocked_series)

    st.success("Analysis complete!")
