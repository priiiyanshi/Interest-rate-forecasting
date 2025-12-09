import streamlit as st
import pandas as pd

from utils.preprocessing import clean_rate_data
from utils.shocks import apply_rate_shocks
from models.arima_model import forecast_rates

# --------------------------------------------------------
# CUSTOM UI THEME (Glassmorphism + Responsive Design)
# --------------------------------------------------------
st.markdown("""
    <style>

    /* Main Page Styling */
    .main {
        padding: 2rem;
        max-width: 900px;
        margin: auto;
    }

    /* Card container */
    .card {
        background: rgba(255, 255, 255, 0.08);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(200, 200, 200, 0.2);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }

    /* Title styling */
    .title {
        font-size: 2.4rem;
        text-align: center;
        font-weight: 700;
        margin-top: -10px;
        margin-bottom: 0.3rem;
    }

    .subtitle {
        font-size: 1.1rem;
        text-align: center;
        color: #cccccc;
        margin-bottom: 2rem;
    }

    /* Buttons */
    .stButton button {
        width: 100%;
        border-radius: 10px;
        padding: 10px 0;
        font-size: 1.05rem;
        background-color: #4A7BFF !important;
        color: white;
        border: none;
        transition: 0.2s ease-in-out;
    }

    .stButton button:hover {
        background-color: #2f62e8 !important;
        transform: scale(1.01);
    }

    /* File uploader */
    .uploadedFile {
        border: 2px dashed rgba(255,255,255,0.3) !important;
        border-radius: 12px !important;
        padding: 10px !important;
        background: rgba(255,255,255,0.05) !important;
    }

    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------------
# PAGE TITLE
# --------------------------------------------------------
st.markdown("<div class='title'>📈 Interest Rate Forecasting Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload → Clean → Forecast → Shock Analysis</div>", unsafe_allow_html=True)

# --------------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📤 Upload Interest Rate CSV File")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
st.markdown("</div>", unsafe_allow_html=True)

df = None
forecast_df = None
shocked_df = None

# --------------------------------------------------------
# PREPROCESSING
# --------------------------------------------------------
if uploaded_file:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧹 Preprocess Data")

    if st.button("Run Preprocessing"):
        try:
            df = pd.read_csv(uploaded_file)
            df = preprocess_data(df)
            st.success("Data preprocessing completed successfully ✔")
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Error during preprocessing: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------
# FORECASTING SECTION
# --------------------------------------------------------
if uploaded_file:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔮 Forecast Future Interest Rates")

    model_choice = st.selectbox("Select a Forecasting Model", ["ARIMA"])

    if st.button("Run Forecast"):
        try:
            if df is None:
                df = pd.read_csv(uploaded_file)
                df = preprocess_data(df)

            st.info("Running ARIMA model… Please wait.")
            forecast_df = forecast_rates(df)

            st.success("Forecast generated successfully!")
            st.line_chart(forecast_df.set_index("Date")["Forecast"])
        except Exception as e:
            st.error(f"Forecasting error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------
# SHOCK APPLICATION SECTION
# --------------------------------------------------------
if uploaded_file:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("⚡ Apply Market Stress Shocks")

    shock_value = st.slider("Select Shock Value (in basis points)", -300, 300, 100)

    if st.button("Apply Shock"):
        try:
            if forecast_df is None:
                df = pd.read_csv(uploaded_file)
                df = preprocess_data(df)
                forecast_df = forecast_rates(df)

            shocked_df = apply_rate_shocks(forecast_df, shock_value)
            st.success("Shock applied successfully!")
            st.line_chart(shocked_df.set_index("Date")["Shocked_Forecast"])
        except Exception as e:
            st.error(f"Shock application error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------
# FOOTER / CREDITS
# --------------------------------------------------------
st.write("---")
st.markdown(
    "<p style='text-align:center; opacity:0.6;'>Made with ❤️ using Streamlit</p>",
    unsafe_allow_html=True
)
