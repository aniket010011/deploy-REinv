import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Real Estate Investment Predictor", layout="wide")

# --------------------------------------
# Load models
# --------------------------------------
@st.cache_resource
def load_models():
    clf = joblib.load("best_classifier_XGBClassifier_py313.pkl")        # retrained model
    reg = joblib.load("best_regressor_LR_final.pkl")          # retrained model
    return clf, reg

classifier, regressor = load_models()

# --------------------------------------
# Input UI
# --------------------------------------
st.title("🏡 Real Estate Investment Predictor")

st.header("📌 Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    state = st.selectbox("State",
                         ["Maharashtra","Delhi","Karnataka","Tamil Nadu","Gujarat",
                          "West Bengal","Rajasthan","Uttar Pradesh","Telangana"])

    city = st.text_input("City", "Mumbai")

    area_sqft = st.number_input("Area (sqft)", min_value=100, max_value=10000, value=1200)

    bhk = st.selectbox("BHK", [1,2,3,4,5], index=0)

with col2:
    furnishing = st.selectbox("Furnishing", 
                              ["Unfurnished","Semi-Furnished","Furnished"])

    crime_rate = st.slider("Crime Rate (per 1000)", 0.0, 10.0, 2.5)

    infrastructure = st.slider("Infrastructure Score", 1.0, 10.0, 7.0)

    age = st.number_input("Age of Property (Years)", 0, 50, 10)

# --------------------------------------
# Build dataframe matching retrained model
# --------------------------------------
input_row = pd.DataFrame([{
    "State": state,
    "City": city,
    "Size_in_SqFt": area_sqft,
    "BHK": bhk,
    "Furnished_Status": furnishing,
    "Crime_Rate": crime_rate,
    "Infrastructure_Score": infrastructure,
    "Age_of_Property": age
}])

st.write("### 🔍 Input Data Preview")
st.dataframe(input_row)

# --------------------------------------
# Prediction
# --------------------------------------
if st.button("Predict"):
    try:
        class_pred = classifier.predict(input_row)[0]
        investment_label = "✅ Good Investment" if class_pred == 1 else "❌ Not a Good Investment"

        resale_pred = regressor.predict(input_row)[0]

        st.subheader("📈 Prediction Results")
        st.write(f"### Investment Classification: **{investment_label}**")
        st.write(f"### Estimated Resale Value: **₹ {resale_pred:,.0f}**")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
