import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_lottie import st_lottie
import requests

# Load Lottie animation from URL
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animations
lottie_welcome = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_3rwasyjy.json")
lottie_house = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_rzvt4rvy.json")
lottie_loading = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_j1adxtyb.json")

# Load model
model = joblib.load("housing_model.pkl")

# Page config
st.set_page_config(page_title="🏡 House Price Predictor", page_icon="🏠", layout="centered")

# Custom CSS for background and styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(120deg, #f6d365, #fda085);
    }
    .stApp {
        background: transparent;
    }
    .css-18e3th9 {
        padding-top: 2rem;
    }
    .card {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st_lottie(lottie_welcome, height=200, key="welcome")
st.title("🏠 House Price Prediction")
st.write("Welcome! Let's estimate your dream home's price with a friendly touch. 🏡✨")

# Input form in a nice card
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("📋 Enter House Details")
    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("📐 Area (sq ft)", min_value=500, max_value=10000, value=2000)
        bedrooms = st.number_input("🛏 Bedrooms", min_value=1, max_value=10, value=3)
    with col2:
        bathrooms = st.number_input("🚿 Bathrooms", min_value=1, max_value=10, value=2)
        stories = st.number_input("🏢 Stories", min_value=1, max_value=5, value=1)
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction button
if st.button("🔮 Predict Price"):
    with st.spinner("Calculating the magic price... ✨"):
        st_lottie(lottie_loading, height=150, key="loading")
        features = np.array([[area, bedrooms, bathrooms, stories]])
        prediction = model.predict(features)
    st.success(f"💰 Estimated Price: **${prediction[0]:,.2f}**")
    st_lottie(lottie_house, height=200, key="house")

