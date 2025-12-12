import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    classifier = joblib.load("best_classifier_XGBClassifier_py313.pkl")
    regressor = joblib.load("best_regressor_LR_py313.pkl")
    return classifier, regressor

classifier, regressor = load_models()

# -----------------------------
# SAFE DEBUG: Show model expected input columns
# -----------------------------
try:
    # This works for all sklearn Pipelines
    expected_cols = classifier.named_steps["preprocess"].feature_names_in_
    st.write("📌 DEBUG — Model expects columns:", list(expected_cols))
except Exception as e:
    st.write("⚠ DEBUG: Could not extract feature names:", str(e))

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Real Estate Investment Predictor", layout="wide")

st.title("🏡 Real Estate Investment Predictor")
st.write("Enter property details to classify *Good Investment* and predict *Resale Value*.")

# -----------------------------
# User Inputs
# -----------------------------
st.header("📌 Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    state = st.selectbox("State", 
                         ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "Gujarat",
                          "West Bengal", "Rajasthan", "Uttar Pradesh", "Telangana"], index=0)
    city = st.text_input("City", "Mumbai")
    area_sqft = st.number_input("Area (sqft)", min_value=200, max_value=10000, value=1200, step=50)
    bhk = st.selectbox("BHK", [1, 2, 3, 4, 5], index=1)

with col2:
    furnishing = st.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Fully-Furnished"])
    crime_rate = st.slider("Crime Rate (per 1000)", 0.0, 10.0, 2.5, 0.1)
    infrastructure_score = st.slider("Infrastructure Score", 1.0, 10.0, 7.0, 0.1)
    age = st.number_input("Age of Property (Years)", min_value=0, max_value=50, value=10)

# -----------------------------
# Prepare DataFrame
# -----------------------------
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
})

st.write("### 🔍 Input Data Preview")
st.dataframe(input_data)

# -----------------------------
# Predict
# -----------------------------
if st.button("Predict"):

    try:
        class_pred = classifier.predict(input_data)[0]
        class_label = "✅ Good Investment" if class_pred == 1 else "❌ Not a Good Investment"

        reg_pred = regressor.predict(input_data)[0]

        st.subheader("📈 Results")
        st.markdown(f"### Investment Classification: **{class_label}**")
        st.markdown(f"### Estimated Resale Value: **₹ {reg_pred:,.0f}**")

        try:
            prob = classifier.predict_proba(input_data)[0][1]
            st.write(f"Probability of GOOD investment: **{prob:.2f}**")
        except:
            pass

    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")

