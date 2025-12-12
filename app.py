import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------------------------
# Load Models
# ----------------------------------------------------
@st.cache_resource
def load_models():
    classifier = joblib.load("best_classifier_XGB_py313.pkl")
    regressor = joblib.load("best_regressor_LR_py313.pkl")
    return classifier, regressor

classifier, regressor = load_models()

# ----------------------------------------------------
# Streamlit UI
# ----------------------------------------------------
st.set_page_config(page_title="Real Estate Investment Predictor", layout="wide")

st.title("🏡 Real Estate Investment Predictor")
st.write("Upload features to classify *Good Investment* and predict *Resale Value*.")

# ----------------------------------------------------
# User Inputs
# ----------------------------------------------------
st.header("📌 Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    state = st.selectbox("State",
                         ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "Gujarat",
                          "West Bengal", "Rajasthan", "Uttar Pradesh", "Telangana"])
    city = st.text_input("City", "Mumbai")
    area_sqft = st.number_input("Area (sqft)", min_value=200, max_value=10000, value=1200, step=50)
    bhk = st.selectbox("BHK", [1, 2, 3, 4, 5], index=1)

with col2:
    furnishing = st.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Fully-Furnished"])
    crime_rate = st.slider("Crime Rate (per 1000)", 0.0, 10.0, 2.5, 0.1)
    infrastructure_score = st.slider("Infrastructure Score", 1.0, 10.0, 7.0, 0.1)
    age = st.number_input("Age of Property (Years)", min_value=0, max_value=50, value=10)

# ----------------------------------------------------
# Build Input Data
# ----------------------------------------------------
input_data = pd.DataFrame({
    "State": [state],
    "City": [city],
    "Area_sqft": [area_sqft],
    "BHK": [bhk],
    "Furnishing": [furnishing],
    "Crime_Rate": [crime_rate],
    "Infrastructure_Score": [infrastructure_score],
    "Age": [age]
})

# Fix dtypes
input_data = input_data.astype({
    "State": "object",
    "City": "object",
    "Furnishing": "object",
    "Area_sqft": "float64",
    "BHK": "int64",
    "Crime_Rate": "float64",
    "Infrastructure_Score": "float64",
    "Age": "int64"
