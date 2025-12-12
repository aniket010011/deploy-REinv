import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    clf = joblib.load("best_classifier_XGBClassifier_py313.pkl")
    reg = joblib.load("best_regressor_LR_py313.pkl")
    return clf, reg


classifier, regressor = load_models()

# Extract expected feature names from classifier pipeline
EXPECTED_FEATURES = classifier.named_steps["preprocess"].feature_names_in_.tolist()

# Columns that should remain categorical (strings)
CATEGORICAL_COLS = {
    "State", "City", "Locality", "Property_Type", "Furnished_Status",
    "Facing", "Owner_Type", "Availability_Status", "Amenities", "Security"
}

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Real Estate Investment Predictor", layout="wide")
st.title("🏡 Real Estate Investment Predictor")

st.write("Enter property details to classify **Good Investment** and predict **Resale Value**.")

# -----------------------------
# User Inputs
# -----------------------------
col1, col2 = st.columns(2)

# Column 1
with col1:
    state = st.selectbox("State", ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "Gujarat",
                                   "West Bengal", "Rajasthan", "Uttar Pradesh", "Telangana"])
    city = st.text_input("City", "Mumbai")
    area_sqft = st.number_input("Area (sqft)", 200, 10000, 1200)
    bhk = st.selectbox("BHK", [1, 2, 3, 4, 5])

# Column 2
with col2:
    furnishing = st.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Fully-Furnished"])
    crime_rate = st.slider("Crime Rate (per 1000)", 0.0, 10.0, 2.5, 0.1)
    infrastructure_score = st.slider("Infrastructure Score", 1.0, 10.0, 7.0, 0.1)
    age = st.number_input("Age of Property (Years)", 0, 50, 10)

# -----------------------------
# Show Input Data Preview
# -----------------------------
input_data_display = pd.DataFrame([{
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
st.dataframe(input_data_display, use_container_width=True)

# -----------------------------
# Function to prepare model input
# -----------------------------
def prepare_model_row():
    row = {}

    for col in EXPECTED_FEATURES:

        # Map real values to expected columns
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

        elif col == "Crime_Rate":
            row[col] = crime_rate

        elif col == "Infrastructure_Score":
            row[col] = infrastructure_score

        elif col == "Age_of_Property":
            row[col] = age

        else:
            # Missing values
            if col in CATEGORICAL_COLS:
                row[col] = "Unknown"
            else:
                row[col] = 0  # numeric fallback

    return pd.DataFrame([row])


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    full_row = prepare_model_row()

    st.subheader("🛠 Prepared Model Input Row (after alignment)")
    st.dataframe(full_row, use_container_width=True)

    try:
        class_pred = classifier.predict(full_row)[0]
        class_label = "✅ Good Investment" if class_pred == 1 else "❌ Not a Good Investment"

        reg_pred = regressor.predict(full_row)[0]

        st.success(f"### Investment Classification: **{class_label}**")
        st.info(f"### Estimated Resale Value: **₹ {reg_pred:,.0f}**")

    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
