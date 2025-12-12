import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------
# Load Models
# ----------------------------------
@st.cache_resource
def load_models():
    classifier = joblib.load("best_classifier_XGB_py313.pkl")
    regressor = joblib.load("best_regressor_LR_py313.pkl")
    return classifier, regressor

classifier, regressor = load_models()

# ----------------------------------
# Streamlit UI
# ----------------------------------
st.set_page_config(page_title="Real Estate Investment Predictor", layout="wide")

st.title("🏡 Real Estate Investment Predictor")
st.write("Upload property features to classify *Good Investment* and estimate *Resale Value*.")

st.header("📌 Enter Property Details")

# ----------------------------------
# Input Fields
# ----------------------------------
col1, col2 = st.columns(2)

with col1:
    state = st.selectbox(
        "State",
        [
            "Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "Gujarat",
            "West Bengal", "Rajasthan", "Uttar Pradesh", "Telangana"
        ]
    )

    city = st.text_input("City", "Mumbai")
    area_sqft = st.number_input("Area (sqft)", min_value=200, max_value=10000, value=1200)
    bhk = st.selectbox("BHK", [1, 2, 3, 4, 5], index=1)

with col2:
    furnishing = st.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Fully-Furnished"])
    crime_rate = st.slider("Crime Rate (per 1000)", 0.0, 10.0, 2.5, 0.1)
    infrastructure_score = st.slider("Infrastructure Score", 1.0, 10.0, 7.0, 0.1)
    age = st.number_input("Age of Property (Years)", min_value=0, max_value=50, value=10)

# ----------------------------------
# Construct Input DataFrame
# ----------------------------------
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

# 🔥 FIX APPLIED: Ensure correct datatypes for pipeline
input_data = input_data.astype({
    "State": "object",
    "City": "object",
    "Furnishing": "object",
    "Area_sqft": "float64",
    "BHK": "int64",
    "Crime_Rate": "float64",
    "Infrastructure_Score": "float64",
    "Age": "int64"
})

st.write("### 🔍 Input Data Preview")
st.dataframe(input_data)

# ----------------------------------
# Predict Button
# ----------------------------------
if st.button("Predict"):

    # Classification (Good Investment or Not)
    class_pred = classifier.predict(input_data)[0]
    class_label = "✅ Good Investment" if class_pred == 1 else "❌ Not a Good Investment"

    # Regression (Resale Value)
    reg_pred = regressor.predict(input_data)[0]

    st.subheader("📈 Results")
    st.markdown(f"### Investment Classification: **{class_label}**")
    st.markdown(f"### Estimated Resale Value: **₹ {reg_pred:,.0f}**")

    # Optional: Probability
    try:
        prob = classifier.predict_proba(input_data)[0][1]
        st.write(f"Probability of GOOD investment: **{prob:.2f}**")
    except:
        pass
