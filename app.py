import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="California Housing Price Predictor", page_icon="🏠", layout="centered")

st.markdown("""
# 🏠 California Housing Price Predictor
Adjust the inputs and click **Predict**.  
Model is compact so it works well on Streamlit Cloud + GitHub.
""")

MODEL_PATH = "housing_model.pkl"

@st.cache_data(show_spinner=False)
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        st.warning("Model file not found locally. Make sure 'housing_model.pkl' is in the app folder.")
        return None
    return joblib.load(path)

model = load_model()

# Input controls in two columns
col1, col2 = st.columns(2)
with col1:
    MedInc = st.slider("Median Income (10k USD)", min_value=0.5, max_value=15.0, value=3.0, step=0.1)
    HouseAge = st.slider("House Age", 1, 50, 20)
    AveRooms = st.slider("Average Rooms", 1.0, 10.0, 5.0)
    AveBedrms = st.slider("Average Bedrooms", 0.5, 3.0, 1.0)
with col2:
    Population = st.slider("Population", 1, 5000, 1000)
    AveOccup = st.slider("Average Occupancy", 0.5, 10.0, 3.0)
    Latitude = st.slider("Latitude", 32.0, 42.0, 37.0)
    Longitude = st.slider("Longitude", -125.0, -112.0, -122.0)

# engineered feature consistent with training notebook
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

if st.button("Predict"):
    if model is None:
        st.error("Model not loaded. Upload 'housing_model.pkl' to the app folder in your repo.")
    else:
        with st.spinner("Predicting..."):
            pred = model.predict(input_df)[0]
            st.success(f"🏡 Predicted Median House Value: **${pred*100000:,.2f}**")
            st.balloons()
