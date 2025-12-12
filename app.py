import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# Load Models
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    classifier = joblib.load("best_classifier_XGBClassifier_py313.pkl")
    regressor = joblib.load("best_regressor_LR_py313.pkl")
    return classifier, regressor

classifier, regressor = load_models()

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Real Estate Investment Predictor", layout="wide")

st.title("🏡 Real Estate Investment Predictor")
st.write("Enter details to classify **Good Investment** and predict **Resale Value**.")

# ---------------------------------------------------------
# User Inputs
# ---------------------------------------------------------
st.header("📌 Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    state = st.selectbox("State", [
        "Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "Gujarat",
        "West Bengal", "Rajasthan", "Uttar Pradesh", "Telangana"
    ])
    city = st.text_input("City", "Mumbai")
    area_sqft = st.number_input("Area (sqft)", min_value=200, max_value=10000, value=1200)
    bhk = st.selectbox("BHK", [1, 2, 3, 4, 5], index=1)

with col2:
    furnishing = st.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Fully-Furnished"])
    crime_rate = st.slider("Crime Rate (per 1000)", 0.0, 10.0, 2.5, 0.1)
    infrastructure_score = st.slider("Infrastructure Score", 1.0, 10.0, 7.0, 0.1)
    age = st.number_input("Age of Property (Years)", min_value=0, max_value=50, value=10)

# ---------------------------------------------------------
# Create DataFrame
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# Prediction Section
# ---------------------------------------------------------
if st.button("Predict"):

    try:
        # -----------------------------------------------
        # 1) Get model's expected training columns
        # -----------------------------------------------
        expected_cols = list(classifier.feature_names_in_)

        # -----------------------------------------------
        # 2) Create full row with expected columns
        # -----------------------------------------------
        full_row = pd.DataFrame([{col: None for col in expected_cols}])

        # -----------------------------------------------
        # 3) Insert our user input data into matching columns
        # -----------------------------------------------
        for col in input_data.columns:
            if col in full_row.columns:
                full_row[col] = input_data[col].values[0]

        # -----------------------------------------------
        # 4) Fill missing features safely
        # -----------------------------------------------
        for col in full_row.columns:
            if pd.isna(full_row.at[0, col]):

                if pd.api.types.is_numeric_dtype(full_row[col]):
                    full_row.at[0, col] = 0
                else:
                    full_row.at[0, col] = "Unknown"

        st.write("### 🔧 Prepared Model Input Row (after alignment)")
        st.dataframe(full_row)

        # -----------------------------------------------
        # 5) Classification
        # -----------------------------------------------
        class_pred = classifier.predict(full_row)[0]
        label = "✅ Good Investment" if class_pred == 1 else "❌ Not a Good Investment"

        # -----------------------------------------------
        # 6) Regression
        # -----------------------------------------------
        reg_pred = regressor.predict(full_row)[0]

        # -----------------------------------------------
        # 7) Display Results
        # -----------------------------------------------
        st.subheader("📈 Results")
        st.markdown(f"### Investment Classification: **{label}**")
        st.markdown(f"### Estimated Resale Value: **₹ {reg_pred:,.0f}**")

        # Probability (optional)
        try:
            prob = classifier.predict_proba(full_row)[0][1]
            st.write(f"Probability of Good Investment: **{prob:.2f}**")
        except:
            pass

    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
