import streamlit as st
import joblib
import pandas as pd
from streamlit_lottie import st_lottie
import requests
import time
import os

# --- Lottie Animation Loader ---
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- Page Config ---
st.set_page_config(
    page_title="🏠 California Housing Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# --- Load Animation ---
house_anim = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_touohxv0.json")
money_anim = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")

# --- Title ---
st.markdown(
    "<h1 style='text-align:center; color:#4CAF50;'>🏠 California Housing Price Predictor</h1>",
    unsafe_allow_html=True
)

st_lottie(house_anim, height=200, key="house")

# --- Load Model ---
MODEL_PATH = "housing_model.pkl"
@st.cache_data
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        st.error("Model file missing. Please upload 'housing_model.pkl'")
        return None
    return joblib.load(path)

model = load_model()

# --- Sidebar Inputs ---
st.sidebar.header("🔧 Input House Details")
MedInc = st.sidebar.slider("Median Income (10k USD)", 0.5, 15.0, 3.0, 0.1)
HouseAge = st.sidebar.slider("House Age", 1, 50, 20)
AveRooms = st.sidebar.slider("Average Rooms", 1.0, 10.0, 5.0)
AveBedrms = st.sidebar.slider("Average Bedrooms", 0.5, 3.0, 1.0)
Population = st.sidebar.slider("Population", 1, 5000, 1000)
AveOccup = st.sidebar.slider("Average Occupancy", 0.5, 10.0, 3.0)
Latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 37.0)
Longitude = st.sidebar.slider("Longitude", -125.0, -112.0, -122.0)

# --- Feature Engineering ---
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

# --- Predict Button ---
if st.button("🚀 Predict Price"):
    if model is None:
        st.error("❌ Model not loaded")
    else:
        with st.spinner("🔮 Analyzing market trends..."):
            time.sleep(1.5)
            pred = model.predict(input_df)[0]

        st.success("✅ Prediction Complete!")
        st.metric("🏡 Predicted Median House Value", f"${pred*100000:,.2f}")

        st_lottie(money_anim, height=150, key="money")
