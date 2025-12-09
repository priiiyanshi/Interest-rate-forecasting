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
# Page Config
# --------------------------------------------------------
st.set_page_config(page_title="Interest Rate Forecasting", layout="wide")

# --------------------------------------------------------
# CSS Styling
# --------------------------------------------------------
st.markdown(
    """
    <style>
        /* Page background */
        .stApp {
            background: linear-gradient(135deg, #1f2c34, #2c3e50);
            color: #ffffff;
        }

        /* Main padding */
        .main {
            padding: 2rem;
        }

        /* Titles */
        .title {
            font-size: 2.8rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 5px;
            color: #ffdd59;
        }

        .subtitle {
            text-align: center;
            color: #bbbbbb;
            font-size: 1.2rem;
            margin-bottom: 30px;
        }

        /* Card styling */
        .card {
            background: rgba(255,255,255,0.07);
            padding: 25px;
            border-radius: 15px;
            border: 1px solid rgba(255,255,255,0.12);
            backdrop-filter: blur(10px);
            margin-bottom: 25px;
        }

        /* Full-width buttons */
        .stButton button {
            width: 100%;
            border-radius: 10px;
            font-size: 1.05rem;
            background-color: #ffdd59;
            color: #1f2c34;
            font-weight: 600;
        }

        /* File uploader label */
        .css-1avcm0n {
            font-size: 1rem;
            font-weight: 500;
        }

        /* Streamlit line charts text color */
        .stLineChart {
            color: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------------
# Header
# --------------------------------------------------------
st.markdown("<div class='title'>📈 Interest Rate Forecasting</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload → Forecast → Apply Shocks</div>", unsafe_allow_html=True)

# --------------------------------------------------------
# File Upload
# --------------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
uploaded = st.file_uploader("📤 Upload your interest rate CSV (date, rate)", type=["csv"])
st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------
# Forecasting Logic
# --------------------------------------------------------
if uploaded:
    df = pd.read_csv(uploaded)
    df.columns = ["date", "rate"]
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # ---------------- Raw Data ----------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Historical Rate Curve")
    st.line_chart(df["rate"], height=300)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Forecast ----------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔮 ARIMA Forecast (30 days)")
    model = ARIMA(df["rate"], order=(2,1,2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    st.line_chart(forecast, height=300)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Shock Scenarios ----------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("⚡ Shock Scenarios")

    col1, col2 = st.columns([1, 2])
    with col1:
        shock_type = st.selectbox("Choose a shock:", ["+50bps", "+100bps", "-50bps"])
    with col2:
        shocked = apply_rate_shocks(df["rate"], shock_type)
        st.line_chart(shocked, height=300)

    st.markdown("</div>", unsafe_allow_html=True)

    st.success("✨ Forecast and shocks generated successfully!")
