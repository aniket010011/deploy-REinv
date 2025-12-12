import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Real Estate Investment Predictor", layout="wide")

# ======================================================
# Load Models
# ======================================================
@st.cache_resource
def load_models():
    classifier = joblib.load("best_classifier_XGBClassifier_py313.pkl")
    regressor = joblib.load("best_regressor_LR_py313.pkl")
    return classifier, regressor

classifier, regressor = load_models()

# ======================================================
# Columns used during training (EXCLUDING BOTH TARGETS)
# ======================================================
MODEL_FEATURES = [
    'ID', 'State', 'City', 'Locality', 'Property_Type', 'BHK',
    'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt', 'Year_Built',
    'Furnished_Status', 'Floor_No', 'Total_Floors', 'Age_of_Property',
    'Nearby_Schools', 'Nearby_Hospitals', 'Public_Transport_Accessibility',
    'Parking_Space', 'Security', 'Amenities', 'Facing', 'Owner_Type',
    'Availability_Status', 'Return_5Y_pct', 'growth_rate',
    'Infrastructure_Score', 'Resale_Base', 'Infra_Adjustment', 'Resale_Value'
]

# ======================================================
# Default values for columns NOT collected in UI
# ======================================================
DEFAULT_VALUES = {
    'ID': 0,
    'Locality': "Unknown",
    'Property_Type': "Unknown",
    'Price_in_Lakhs': 0,
    'Price_per_SqFt': 0,
    'Year_Built': 2000,
    'Floor_No': 0,
    'Total_Floors': 0,
    'Nearby_Schools': 0,
    'Nearby_Hospitals': 0,
    'Public_Transport_Accessibility': 0,
    'Parking_Space': 0,
    'Security': 0,
    'Amenities': 0,
    'Facing': "Unknown",
    'Owner_Type': "Unknown",
    'Availability_Status': "Unknown",
    'Return_5Y_pct': 0,
    'growth_rate': 0,
    'Resale_Base': 0,
    'Infra_Adjustment': 0,
    'Resale_Value': 0
}

# ======================================================
# Streamlit UI
# ======================================================
st.title("🏡 Real Estate Investment Predictor")

state = st.selectbox("State", ["Maharashtra"])
city = st.selectbox("City", ["Mumbai"])
area_sqft = st.number_input("Area (sqft)", min_value=100, max_value=10000, value=1200)
bhk = st.selectbox("BHK", [1, 2, 3, 4])
furnishing = st.selectbox("Furnishing", ["Furnished", "Semi-Furnished", "Unfurnished"])
crime_rate = st.slider("Crime Rate (per 1000)", 0.0, 10.0, 2.5)
infrastructure_score = st.slider("Infrastructure Score", 1, 10, 7)
age = st.number_input("Age of Property (Years)", min_value=0, max_value=100, value=10)

# Preview
input_df = pd.DataFrame([{
    "State": state,
    "City": city,
    "Area_sqft": area_sqft,
    "BHK": bhk,
    "Furnishing": furnishing,
    "Crime_Rate": crime_rate,
    "Infrastructure_Score": infrastructure_score,
    "Age": age
}])

st.subheader("📌 Input Data Preview")
st.dataframe(input_df)

# ======================================================
# Prepare full model row
# ======================================================
def prepare_full_feature_row(row):
    full = {}

    full["ID"] = 0
    full["State"] = row["State"]
    full["City"] = row["City"]
    full["Locality"] = DEFAULT_VALUES["Locality"]
    full["Property_Type"] = DEFAULT_VALUES["Property_Type"]
    full["BHK"] = row["BHK"]
    full["Size_in_SqFt"] = row["Area_sqft"]
    full["Price_in_Lakhs"] = DEFAULT_VALUES["Price_in_Lakhs"]
    full["Price_per_SqFt"] = DEFAULT_VALUES["Price_per_SqFt"]
    full["Year_Built"] = DEFAULT_VALUES["Year_Built"]
    full["Furnished_Status"] = row["Furnishing"]
    full["Floor_No"] = DEFAULT_VALUES["Floor_No"]
    full["Total_Floors"] = DEFAULT_VALUES["Total_Floors"]
    full["Age_of_Property"] = row["Age"]
    full["Nearby_Schools"] = DEFAULT_VALUES["Nearby_Schools"]
    full["Nearby_Hospitals"] = DEFAULT_VALUES["Nearby_Hospitals"]
    full["Public_Transport_Accessibility"] = DEFAULT_VALUES["Public_Transport_Accessibility"]
    full["Parking_Space"] = DEFAULT_VALUES["Parking_Space"]
    full["Security"] = DEFAULT_VALUES["Security"]
    full["Amenities"] = DEFAULT_VALUES["Amenities"]
    full["Facing"] = DEFAULT_VALUES["Facing"]
    full["Owner_Type"] = DEFAULT_VALUES["Owner_Type"]
    full["Availability_Status"] = DEFAULT_VALUES["Availability_Status"]
    full["Return_5Y_pct"] = DEFAULT_VALUES["Return_5Y_pct"]
    full["growth_rate"] = DEFAULT_VALUES["growth_rate"]
    full["Infrastructure_Score"] = row["Infrastructure_Score"]
    full["Resale_Base"] = DEFAULT_VALUES["Resale_Base"]
    full["Infra_Adjustment"] = DEFAULT_VALUES["Infra_Adjustment"]
    full["Resale_Value"] = DEFAULT_VALUES["Resale_Value"]

    df = pd.DataFrame([full])
    df = df[MODEL_FEATURES]  # enforce correct order
    return df

# ======================================================
# Prediction
# ======================================================
if st.button("Predict"):
    try:
        full_row = prepare_full_feature_row(input_df.iloc[0])

        st.subheader("🛠 Prepared Model Input Row")
        st.dataframe(full_row)

        # Classification → Good_Investment prediction
        class_pred = classifier.predict(full_row)[0]

        # Regression → Future_Price_5Y prediction
        reg_pred = regressor.predict(full_row)[0]

        st.success(f"🏷 Investment Recommendation: **{'Good Investment' if class_pred == 1 else 'Not Good'}**")
        st.success(f"💰 Predicted Price After 5 Years: **₹ {reg_pred:,.2f} Lakhs**")

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
