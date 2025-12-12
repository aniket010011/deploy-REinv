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
    row = {}

    for col in EXPECTED_FEATURES:

        if col == "State":
            row[col] = state

        elif col == "City":
            row[col] = city

        elif col == "Size_in_SqFt":
            row[col] = float(area_sqft)

        elif col == "BHK":
            row[col] = float(bhk)

        elif col == "Crime_Rate":
            row[col] = float(crime_rate)

        elif col == "Infrastructure_Score":
            row[col] = float(infrastructure_score)

        elif col == "Age_of_Property":
            row[col] = float(age)

        elif col == "Furnished_Status":
            row[col] = furnishing

        else:
            # Missing values
            if col in CATEGORICAL_COLS:
                row[col] = "Unknown"
            else:
                row[col] = float(0)

    # Convert everything to DataFrame
    df = pd.DataFrame([row])

    # Make ABSOLUTELY sure all numeric columns are floats
    for c in df.columns:
        if c not in CATEGORICAL_COLS:
            df[c] = df[c].astype(float)

    return df


# -----------------------------
# Prediction
# ------------------------
