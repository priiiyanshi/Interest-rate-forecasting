import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from utils.shocks import apply_rate_shocks # Assuming this function exists and works

# --------------------------------------------------------
# Page Config
# --------------------------------------------------------
st.set_page_config(page_title="Interest Rate Forecasting", layout="wide")

# --------------------------------------------------------
# CSS Styling (Enhanced for Dark Theme and Readability)
# --------------------------------------------------------
st.markdown(
    """
    <style>
        /* Base Streamlit App container */
        .stApp {
            /* Deep, dark blue gradient for background */
            background: linear-gradient(145deg, #0f1c2b, #1a2a3a);
            color: #e0e0e0; /* Light text for readability */
        }

        /* Main content area padding */
        .main {
            padding: 2rem 5rem;
        }

        /* Titles and Headers */
        .title {
            font-size: 3rem;
            font-weight: 800;
            text-align: center;
            margin-bottom: 5px;
            color: #58a6ff; /* A striking but professional blue for main title */
            letter-spacing: 1px;
        }

        .subtitle {
            text-align: center;
            color: #90a4ae;
            font-size: 1.3rem;
            margin-bottom: 40px;
        }

        /* Card styling (Sections) */
        .card {
            background: rgba(255, 255, 255, 0.05); /* Slightly transparent dark background */
            padding: 30px;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1); /* Subtle border */
            backdrop-filter: blur(5px);
            margin-bottom: 30px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }

        /* Streamlit Subheaders in cards */
        h3 {
            color: #fdd835; /* Gold/Yellow for section headers */
            font-weight: 600;
            margin-bottom: 20px;
            border-bottom: 2px solid rgba(253, 216, 53, 0.3); /* Subtle separator */
            padding-bottom: 5px;
        }

        /* Full-width buttons */
        .stButton button {
            width: 100%;
            border-radius: 10px;
            font-size: 1.1rem;
            background-color: #58a6ff; /* Primary button color */
            color: #0f1c2b;
            font-weight: 700;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #79c0ff;
            color: #000000;
        }

        /* File uploader label */
        label[data-testid="stFileUploadDropzone"] {
            font-size: 1.1rem;
            font-weight: 500;
            color: #e0e0e0;
        }
        
        /* Selectbox/Input styling for dark mode */
        div[data-baseweb="select"] > div, 
        div[data-baseweb="input"] > div {
            background-color: #1a2a3a !important; 
            border: 1px solid #37474f !important;
            color: #e0e0e0 !important;
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
    
    # Use a column to place the uploader
    col_upload, col_space = st.columns([2, 1])
    with col_upload:
        uploaded = st.file_uploader("Upload Historical Interest Rate Data (CSV with 'date' and 'rate' columns)", type=["csv"])
    
    # Placeholder for configuration options if you add them later
    # with col_space:
    #     forecast_steps = st.number_input("Forecast Period (days):", min_value=1, value=30, step=1)

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
            # Set chart color for better dark mode visibility
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
                # Moved shock type selection into a column
                shock_type = st.selectbox(
                    "Select Rate Shock Scenario (bps = basis points):", 
                    ["No Shock", "+50bps", "+100bps", "-50bps"],
                    index=0 # Default to No Shock
                )
            
            with shock_col2:
                if shock_type == "No Shock":
                     st.info("Select a shock scenario on the left to apply it to the historical data.")
                     shocked_data = df["rate"] # Display original data
                else:
                    try:
                        shocked_data = apply_rate_shocks(df["rate"], shock_type)
                    except Exception as e:
                        st.error(f"Error applying rate shocks: {e}")
                        shocked_data = df["rate"] # Fallback to original data
                        
                # Ensure the chart is only displayed once the data is ready
                if shocked_data is not None:
                    st.line_chart(shocked_data, height=300, use_container_width=True)
                
            st.markdown("</div>", unsafe_allow_html=True)

        st.success("✅ Analysis Complete! Forecast generated and stress test results displayed.")
