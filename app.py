import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    clf = joblib.load("best_classifier_XGBClassifier_py313.pkl")
    reg = joblib.load("best_regressor_LR_py313.pkl")
    return clf, reg


classifier, regressor = load_models()

# Extract expected column names from pipeline
EXPECTED_FEATURES = classifier.named_steps["preprocess"].feature_names_in_.tolist()

# Columns that should stay as strings
CATEGORICAL_COLS = {
    "State", "City", "Locality", "Property_Type", "Furnished_Status",
    "Facing", "Owner_Type", "Availability_Status", "Amenities", "Security"
}

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Real Estate Investment Predictor", layout="wide")
st.title("🏡 Real Estate Investment Predictor")

col1, col2 = st.columns(2)

with col1:
    state = st.selectbox("State", ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu",
                                   "Gujarat", "West Bengal", "Rajasthan",
                                   "Uttar Pradesh", "Telangana"])
    city = st.text_input("City", "Mumbai")
    area_sqft = float(st.number_input("Area (sqft)", min_value=200, max_value=10000, value=1200))
    bhk = float(st.selectbox("BHK", [1, 2, 3, 4, 5]))

with col2:
    furnishing = st.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Fully-Furnished"])
    crime_rate = float(st.slider("Crime Rate (per 1000)", 0.0, 10.0, 2.5, 0.1))
    infrastructure_score = float(st.slider("Infrastructure Score", 1.0, 10.0, 7.0, 0.1))
    age = float(st.number_input("Age of Property (Years)", 0, 50, value=10))

# Show input for reference
input_preview = pd.DataFrame([{
    "State": state,
    "City": city,
    "Area_sqft": area_sqft,
    "BHK": bhk,
    "Furnishing": furnishing,
    "Crime_Rate": crime_rate,
    "Infrastructure_Score": infrastructure_score,
    "Age": age
}])

st.subheader("🔍 Input Data Preview")
st.dataframe(input_preview, use_container_width=True)

# -----------------------------
# Prepare row for pipeline
# -----------------------------
def prepare_model_row():

    # Start row dictionary
    row = {}

    # Fill user-provided fields
    row["State"] = state
    row["City"] = city
    row["BHK"] = float(bhk)
    row["Size_in_SqFt"] = float(area_sqft)
    row["Furnished_Status"] = furnishing
    row["Crime_Rate"] = float(crime_rate)
    row["Infrastructure_Score"] = float(infrastructure_score)
    row["Age_of_Property"] = float(age)

    # Fill all other model columns with NaN (numeric) or "Unknown" (categorical)
    for col in EXPECTED_FEATURES:

        if col not in row:  # not user-entered
            if col in CATEGORICAL_COLS:
                row[col] = "Unknown"
            else:
                row[col] = np.nan  # numeric missing

    # Convert to DataFrame
    df = pd.DataFrame([row])

    # FINAL CRITICAL FIX: Force numeric dtypes
    for col in df.columns:
        if col not in CATEGORICAL_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")  # ensures float64 + NaN

    return df
    
# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    full_row = prepare_model_row()

    st.subheader("🛠 Prepared Model Input Row (after alignment)")
    st.dataframe(full_row, use_container_width=True)

    try:
        class_pred = classifier.predict(full_row)[0]
        reg_pred = regressor.predict(full_row)[0]

        class_label = "✅ Good Investment" if class_pred == 1 else "❌ Not a Good Investment"

        st.success(f"### Investment Classification: **{class_label}**")
        st.info(f"### Estimated Resale Value: **₹ {reg_pred:,.0f}**")

    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")



