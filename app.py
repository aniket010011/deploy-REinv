import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# Load saved models
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    classifier = joblib.load("best_classifier_XGBClassifier_py313.pkl")
    regressor = joblib.load("best_regressor_LR_py313.pkl")
    return classifier, regressor


classifier, regressor = load_models()

st.set_page_config(page_title="Real Estate Investment Predictor", layout="wide")

st.title("🏡 Real Estate Investment Predictor")
st.write("Enter property details to classify *Good Investment* and predict *Resale Value*.")


# ---------------------------------------------------------
# Streamlit Input UI
# ---------------------------------------------------------
st.header("📌 Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    state = st.selectbox("State",
                         ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "Gujarat",
                          "West Bengal", "Rajasthan", "Uttar Pradesh", "Telangana"])
    city = st.text_input("City", "Mumbai")
    area_sqft = st.number_input("Area (sqft)", 200, 10000, 1200, 50)
    bhk = st.selectbox("BHK", [1, 2, 3, 4, 5], index=1)

with col2:
    furnishing = st.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Fully-Furnished"])
    crime_rate = st.slider("Crime Rate (per 1000)", 0.0, 10.0, 2.5, 0.1)
    infrastructure_score = st.slider("Infrastructure Score", 1.0, 10.0, 7.0, 0.1)
    age = st.number_input("Age of Property (Years)", 0, 50, 10)


# ---------------------------------------------------------
# Raw input preview
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

st.subheader("🔍 Input Data Preview")
st.dataframe(input_data)


# ---------------------------------------------------------
# MODEL EXPECTED 29 FEATURES → DEFINE THEM
# ---------------------------------------------------------
EXPECTED_FEATURES = [
    "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
    "Price_per_SqFt", "Year_Built", "Furnished_Status", "Floor_No", "Total_Floors",
    "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals", "Public_Transport_Accessibility",
    "Parking_Space", "Security", "Amenities", "Facing", "Owner_Type", "Availability_Status",
    "Return_5Y_pct", "growth_rate", "Infrastructure_Score", "Resale_Base",
    "Infra_Adjustment", "Resale_Value"
]


# ---------------------------------------------------------
# Build full 29-column row with defaults for missing fields
# ---------------------------------------------------------
def prepare_model_input():
    row = {}

    for col in EXPECTED_FEATURES:
        if col == "State":
            row[col] = state
        elif col == "City":
            row[col] = city
        elif col == "Size_in_SqFt":
            row[col] = area_sqft
        elif col == "BHK":
            row[col] = bhk
        elif col == "Furnished_Status":
            row[col] = furnishing
        elif col == "Age_of_Property":
            row[col] = age
        elif col == "Infrastructure_Score":
            row[col] = infrastructure_score
        else:
            # Missing feature → assign default placeholder
            # Numeric defaults = 0, Categorical defaults = "Unknown"
            row[col] = 0

    return pd.DataFrame([row])


# ---------------------------------------------------------
# Predict Button
# ---------------------------------------------------------
if st.button("Predict"):

    full_row = prepare_model_input()

    st.subheader("🛠 Prepared Model Input Row (after alignment)")
    st.dataframe(full_row)

    # Try Prediction
    try:
        class_pred = classifier.predict(full_row)[0]
        reg_pred = regressor.predict(full_row)[0]

        class_label = "✅ Good Investment" if class_pred == 1 else "❌ Not a Good Investment"

        st.subheader("📈 Results")
        st.write(f"### Investment Classification: **{class_label}**")
        st.write(f"### Estimated Resale Value: **₹ {reg_pred:,.0f}**")

        # Probability
        try:
            prob = classifier.predict_proba(full_row)[0][1]
            st.write(f"Probability of GOOD investment: **{prob:.2f}**")
        except:
            pass

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
