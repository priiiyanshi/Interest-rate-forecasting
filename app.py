import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from utils.shocks import apply_rate_shocks
from utils.preprocessing import clean_rate_data
st.title("Interest Rate Forecasting & Yield Curve Shock Simulator")
st.write("Upload a CSV with columns: date, rate")
import streamlit as st

# Inject Custom CSS
st.markdown("""
    <style>

    /* Center everything and control width */
    .main {
        padding: 2rem;
    }

    /* Beautiful card container */
    .card {
        background: #ffffff10;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #cccccc33;
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }

    /* Page title styling */
    .title {
        font-size: 2.3rem;
        text-align: center;
        font-weight: 700;
        margin-bottom: 1.2rem;
    }

    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }

    /* Better buttons */
    .stButton button {
        width: 100%;
        border-radius: 10px;
        padding: 10px 0;
        font-size: 1rem !important;
        background-color: #4A7BFF !important;
        color: white;
        border: none;
    }

    .stButton button:hover {
        background-color: #2f62e8 !important;
        transform: scale(1.01);
        transition: 0.2s ease-in-out;
    }

    /* File uploader style */
    .uploadedFile {
        border: 2px dashed #bbbbbb55 !important;
        border-radius: 12px !important;
        padding: 10px !important;
        background: #ffffff05 !important;
    }

    </style>
""", unsafe_allow_html=True)


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
