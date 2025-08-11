import streamlit as st
import joblib
import pandas as pd
from streamlit_lottie import st_lottie
import requests
import time
import os

# ========================  CUSTOM CSS ========================
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to right, #f0f2f6, #d9e4ec);
            font-family: 'Segoe UI', sans-serif;
        }
        .main-title {
            text-align: center;
            color: #2e7d32;
            font-size: 36px;
            font-weight: bold;
        }
        .sub-title {
            text-align: center;
            color: #555;
            font-size: 18px;
            margin-bottom: 20px;
        }
        .css-1d391kg { padding: 1rem 2rem; }
        .stMetric {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
        }
    </style>
""", unsafe_allow_html=True)

# ========================  LOTTIE LOADER ========================
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ========================  PAGE CONFIG ========================
st.set_page_config(
    page_title="🏠 California Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

house_anim = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_touohxv0.json")
money_anim = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")

# ========================  TITLE ========================
st.markdown("<h1 class='main-title'>🏠 California Housing Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Estimate the median house price in California with real-time predictions</p>", unsafe_allow_html=True)
st_lottie(house_anim, height=180, key="house")

# ========================  LOAD MODEL ========================
MODEL_PATH = "housing_model.pkl"

@st.cache_data
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        st.error("Model file missing. Please upload 'housing_model.pkl'")
        return None
    return joblib.load(path)

model = load_model()

# ========================  INPUT SECTION ========================
with st.expander("🔧 Adjust House Details"):
    col1, col2, col3 = st.columns(3)
    with col1:
        MedInc = st.slider("Median Income (10k USD)", 0.5, 15.0, 3.0, 0.1)
        HouseAge = st.slider("House Age", 1, 50, 20)
        AveRooms = st.slider("Average Rooms", 1.0, 10.0, 5.0)
    with col2:
        AveBedrms = st.slider("Average Bedrooms", 0.5, 3.0, 1.0)
        Population = st.slider("Population", 1, 5000, 1000)
        AveOccup = st.slider("Average Occupancy", 0.5, 10.0, 3.0)
    with col3:
        Latitude = st.slider("Latitude", 32.0, 42.0, 37.0)
        Longitude = st.slider("Longitude", -125.0, -112.0, -122.0)

# ========================  FEATURE ENGINEERING ========================
rooms_per_household = AveRooms / (AveOccup + 1e-6)
input_df = pd.DataFrame([{
    'MedInc': MedInc,
    'HouseAge': HouseAge,
    'AveRooms': AveRooms,
    'AveBedrms': AveBedrms,
    'Population': Population,
    'AveOccup': AveOccup,
    'Latitude': Latitude,
    'Longitude': Longitude,
    'rooms_per_household': rooms_per_household
}])

# ========================  PREDICT BUTTON ========================
col_pred1, col_pred2, col_pred3 = st.columns([1,2,1])
with col_pred2:
    if st.button("🚀 Predict Price", use_container_width=True):
        if model is None:
            st.error("❌ Model not loaded")
        else:
            with st.spinner("🔮 Analyzing market trends..."):
                time.sleep(1.5)
                pred = model.predict(input_df)[0]

            st.success("✅ Prediction Complete!")
            st.metric("🏡 Predicted Median House Value", f"${pred*100000:,.2f}")

            st_lottie(money_anim, height=140, key="money")

            # Extra insight
            if pred < 2.0:
                st.info("💡 This seems to be a relatively affordable area.")
            elif pred < 4.0:
                st.warning("⚠️ Mid-range pricing detected.")
            else:
                st.error("💎 This is a high-value property area!")
