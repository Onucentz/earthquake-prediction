import streamlit as st
import numpy as np
import joblib

# Load trained earthquake model and scaler
model = joblib.load('earthquake_model.pkl')  # Ensure this file exists in the same folder
scaler = joblib.load('scaler.pkl')

# Define mapping for model prediction classes
likelihood_mapping = {
    0: "None",
    1: "Low",
    2: "Moderate",
    3: "Highly Likely"
}

# Streamlit App
st.set_page_config(page_title="Earthquake Likelihood Predictor", page_icon="🌍", layout="centered")
st.title("🌍 Earthquake Likelihood Prediction App")

st.write("""
This app predicts the **likelihood of an earthquake** based on key seismic parameters.
Enter the required values below and click **Predict**.
""")

# Input fields for user data
with st.form("prediction_form"):
    magnitude = st.number_input("Magnitude (e.g., 5.5)", min_value=0.0, step=0.1)
    depth = st.number_input("Depth (km)", min_value=0.0, step=1.0)
    cdi = st.number_input("CDI (Community Determined Intensity)", min_value=0.0, step=0.1)
    mmi = st.number_input("MMI (Modified Mercalli Intensity)", min_value=0.0, step=0.1)
    sig = st.number_input("Significance", min_value=0.0, step=1.0)

    submitted = st.form_submit_button("Predict")

# Handle prediction
if submitted:
    try:
        # Prepare data for model
        input_data = np.array([[magnitude, depth, cdi, mmi, sig]])
        scaled_data = scaler.transform(input_data)

        # Predict with model
        prediction = model.predict(scaled_data)[0]
        likelihood = likelihood_mapping.get(prediction, "Unknown")

        st.success(f"### 🌋 Earthquake Likelihood: **{likelihood}**")

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

# Optional JSON API simulation
st.divider()
st.subheader("🔧 JSON Input (API Simulation)")
st.write("You can simulate API input here by entering raw JSON data:")

sample_json = {
    "magnitude": 5.5,
    "depth": 10.0,
    "cdi": 3.2,
    "mmi": 4.1,
    "sig": 120
}

user_json = st.text_area("Enter JSON data:", value=str(sample_json))

if st.button("Predict from JSON"):
    try:
        import ast
        data = ast.literal_eval(user_json)
        features = [
            data['magnitude'],
            data['depth'],
            data['cdi'],
            data['mmi'],
            data['sig']
        ]
        input_data = np.array([features])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        likelihood = likelihood_mapping.get(prediction, "Unknown")
        st.info(f"### 🧭 Prediction from JSON: **{likelihood}**")

    except Exception as e:
        st.error(f"Invalid JSON or Prediction Error: {str(e)}")


