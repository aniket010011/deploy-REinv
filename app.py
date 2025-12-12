import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Real Estate Investment Predictor", layout="wide")

# ============================================================
# 1. Load Models (Classifier + Regressor)
# ============================================================

@st.cache_resource
def load_models():
    clf = joblib.load("best_classifier_XGBClassifier_py313.pkl")
    reg = joblib.load("best_regressor_LR_py313.pkl")
    return clf, reg

classifier, regressor = load_models()

# ============================================================
# 2. Fixed Feature Order (MUST MATCH TRAINING DATA)
# ============================================================

MODEL_FEATURES = [
    "State",
    "City",
    "Furnished_Status",
    "Size_in_SqFt",
    "BHK",
    "Crime_Rate",
    "Infrastructure_Score",
    "Age_of_Property"
]

# ============================================================
# 3. Streamlit Title
# ============================================================

st.title("🏡 Real Estate Investment Predictor")

st.write("Enter property features below to predict **Good Investment** and **Future 5Y Price**.")

# ============================================================
# 4. User Input Form
# ============================================================

state = st.selectbox("State", ["Maharashtra", "Karnataka", "Delhi", "Tamil Nadu"])
city = st.text_input("City", "Mumbai")
size_sqft = st.number_input("Area (SqFt)", min_value=100, max_value=10000, value=1200)
bhk = st.selectbox("BHK", [1, 2, 3, 4, 5])
furnish = st.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Furnished"])
crime = st.slider("Crime Rate (per 1000)", 0.0, 10.0, 2.5, 0.1)
infra = st.slider("Infrastructure Score", 0.0, 10.0, 7.0, 0.1)
age = st.number_input("Age of Property (Years)", min_value=0, max_value=100, value=10)

# ------------------------------------------------------------
# 5. Prepare Input Preview
# ------------------------------------------------------------

raw_input_df = pd.DataFrame([{
    "State": state,
    "City": city,
    "Furnished_Status": furnish,
    "Size_in_SqFt": size_sqft,
    "BHK": bhk,
    "Crime_Rate": crime,
    "Infrastructure_Score": infra,
    "Age_of_Property": age
}])

st.subheader("🔍 Input Data Preview")
st.dataframe(raw_input_df)

# ------------------------------------------------------------
# 6. Prediction Button
# ------------------------------------------------------------

if st.button("Predict"):

    try:
        # Ensure column order matches model training
        input_prepared = raw_input_df[MODEL_FEATURES]

        st.subheader("🔧 Prepared Input Row (Aligned to Model Columns)")
        st.dataframe(input_prepared)

        # CLASSIFICATION
        class_pred = classifier.predict(input_prepared)[0]
        class_label = "Good Investment" if class_pred == 1 else "Bad Investment"

        # REGRESSION
        reg_pred = float(regressor.predict(input_prepared)[0])

        st.success(f"🏷 Investment Classification: **{class_label}**")
        st.success(f"💰 Predicted Future Price (5Y): **₹ {reg_pred:,.0f}**")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
