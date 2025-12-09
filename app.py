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
# CSS Styling (Final, Targeted Fix)
# --------------------------------------------------------
st.markdown(
    """
    <style>
        /* --- General Colors --- */
        .stApp {
            background: linear-gradient(145deg, #121212, #1e2630); 
            color: #f0f0f0; 
        }

        /* Main content area padding */
        .main {
            padding: 2rem 5rem;
        }

        /* --- Header Styling --- */
        .title {
            font-size: 3.2rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 5px;
            color: #00bcd4; 
            letter-spacing: 1px;
        }

        .subtitle {
            text-align: center;
            color: #bdbdbd; 
            font-size: 1.2rem;
            margin-bottom: 40px;
        }

        /* --- Card Styling (Containers) --- */
        .card {
            background: rgba(255, 255, 255, 0.08); 
            padding: 30px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.15); 
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }

        /* Streamlit Subheaders in cards */
        h3 {
            color: #ff9800; 
            font-weight: 600;
            margin-bottom: 20px;
            border-bottom: 2px solid rgba(255, 152, 0, 0.3);
            padding-bottom: 5px;
        }

        /* --- Component Styling --- */
        
        /* Full-width buttons */
        .stButton button {
            width: 100%;
            border-radius: 8px;
            font-size: 1.1rem;
            background-color: #00bcd4; 
            color: #121212;
            font-weight: 700;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #4dd0e1;
            color: #000000;
        }
        
        /* Dropzone area background/border -> Bright Yellow */
        div[data-testid="stFileUploaderDropzone"] {
            background-color: #ffffb3 !important; /* Bright Yellow/Cream background for black text */
            border: 2px dashed #00bcd4 !important;
            border-radius: 10px;
            padding: 20px;
        }

        /* File uploader label and text */
        /* BROWSE FILES TEXT/LABEL -> BLACK */
        label[data-testid="stFileUploadDropzone"] {
            font-size: 1.1rem;
            font-weight: 600;
            color: #000000 !important; /* Main Label Text */
        }
        
        /* TARGETING THE SPECIFIC 'BROWSE FILES' LINK INSIDE THE DROPZONE: BLACK */
        div[data-testid="stFileUploaderDropzone"] a {
            color: #000000 !important; /* Forces the clickable link to black */
            font-weight: 700;
            text-decoration: underline;
        }

        /* Uploaded File Name Text Color -> WHITE */
        div[data-testid="stFileUploader"] p {
            color: #FFFFFF !important; /* UPLOADED FILE NAME IS WHITE */
            font-weight: 500;
            background-color: transparent !important; 
        }
        
        /* Selectbox/Input styling for dark mode */
        div[data-baseweb="select"] > div, 
        div[data-baseweb="input"] > div {
            background-color: #212121 !important; 
            border: 1px solid #424242 !important;
            color: #f0f0f0 !important;
            border-radius: 8px;
        }
        
        /* Streamlit success message */
        .stAlert div[role="alert"] {
            background-color: #388e3c;
            color: white;
            border-radius: 10px;
            font-weight: 500;
        }

    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------------
# Header
# --------------------------------------------------------
st.markdown("<div class='title'>💰 Rate Curve Modeler</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A financial tool for Interest Rate Forecasting and Stress Testing</div>", unsafe_allow_html=True)

# --------------------------------------------------------
# Main App Structure 
# --------------------------------------------------------

# Section 1: File Upload and Configuration
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("⚙️ Data Input & Configuration")
    
    col_upload, col_space = st.columns([2, 1])
    with col_upload:
        uploaded = st.file_uploader("Upload Historical Interest Rate Data (CSV with 'date' and 'rate' columns)", type=["csv"])
    
    st.markdown("</div>", unsafe_allow_html=True) 

# --------------------------------------------------------
# Forecasting Logic and Results
# --------------------------------------------------------
if uploaded:
    df = pd.read_csv(uploaded)
    
    # Ensure columns are correctly named and dated
    if len(df.columns) < 2:
        st.error("CSV must contain at least two columns: one for date and one for rate.")
    else:
        # Standardize column names
        df.columns = ["date", "rate"][:len(df.columns)]
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        
        # --- Visualization Section (2 columns for side-by-side charts) ---
        
        col_hist, col_forecast = st.columns(2)
        
        # ---------------- Raw Data ----------------
        with col_hist:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("📈 Historical Rate Curve")
            st.line_chart(df["rate"], height=300, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------- Forecast ----------------
        with col_forecast:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("🔮 ARIMA Forecast (30 days)")
            try:
                # Assuming (2,1,2) is the intended ARIMA order
                model = ARIMA(df["rate"], order=(2, 1, 2))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=30)
                st.line_chart(forecast, height=300, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error during ARIMA modeling: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        
        # ---------------- Shock Scenarios (Full Width Section) ----------------
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("⚡ Stress Testing Scenarios")

            shock_col1, shock_col2 = st.columns([1, 3])
            
            with shock_col1:
                shock_type = st.selectbox(
                    "Select Rate Shock Scenario (bps = basis points):", 
                    ["No Shock", "+50bps", "+100bps", "-50bps"],
                    index=0 
                )
            
            with shock_col2:
                if shock_type == "No Shock":
                     st.info("Select a shock scenario on the left to apply it to the historical data.")
                     shocked_data = df["rate"] 
                else:
                    try:
                        shocked_data = apply_rate_shocks(df["rate"], shock_type)
                    except Exception as e:
                        st.error(f"Error applying rate shocks: {e}")
                        shocked_data = df["rate"] 
                        
                if shocked_data is not None:
                    st.line_chart(shocked_data, height=300, use_container_width=True)
                
            st.markdown("</div>", unsafe_allow_html=True) 

        st.success("✅ Analysis Complete! Forecast generated and stress test results displayed.")
