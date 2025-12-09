import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from utils.shocks import apply_rate_shocks

# --------------------------------------------------------
# Polished UI Setup (SAFE + DOES NOT BREAK IMPORTS)
# --------------------------------------------------------
st.set_page_config(page_title="Interest Rate Forecasting", layout="wide")

st.markdown(
    """
    <style>
        .main {
            padding: 2rem;
        }

        .title {
            font-size: 2.6rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: -10px;
        }

        .subtitle {
            text-align: center;
            color: #bbbbbb;
            font-size: 1.1rem;
            margin-bottom: 25px;
        }

        .card {
            background: rgba(255,255,255,0.08);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.15);
            backdrop-filter: blur(8px);
            margin-bottom: 20px;
        }

        .stButton button {
            width: 100%;
            border-radius: 8px;
            font-size: 1.05rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------------
# Title
# --------------------------------------------------------
st.markdown("<div class='title'>📈 Interest Rate Forecasting</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload → Forecast → Apply Shocks</div>", unsafe_allow_html=True)

# --------------------------------------------------------
# File Upload Card
# --------------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
uploaded = st.file_uploader("📤 Upload your interest rate time-series CSV", type=["csv"])
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------
# Forecasting Logic (UNCHANGED)
# --------------------------------------------------------

if uploaded:
    # ---- Data Prep ----
    df = pd.read_csv(uploaded)
    df.columns = ["date", "rate"]
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # ---- Plot Raw Data ----
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Historical Rate Curve")
    st.line_chart(df["rate"], height=250)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- ARIMA Forecast ----
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔮 ARIMA Forecast (30 days)")
    model = ARIMA(df["rate"], order=(2,1,2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    st.line_chart(forecast, height=250)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Shock Scenarios ----
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("⚡ Shock Scenarios")

    shock_type = st.selectbox("Choose a shock", ["+50bps", "+100bps", "-50bps"])
    shocked = apply_rate_shocks(df["rate"], shock_type)

    st.line_chart(shocked, height=250)
    st.markdown("</div>", unsafe_allow_html=True)

    st.success("✨ Forecast and shocks generated successfully!")
