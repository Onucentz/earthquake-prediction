from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load your trained earthquake model
model = joblib.load('earthquake_model.pkl')  # Ensure this file exists in the same folder
scaler = joblib.load('scaler.pkl')

# Define mapping for model prediction classes
likelihood_mapping = {
    0: "None",
    1: "Low",
    2: "Moderate",
    3: "Highly Likely"
}

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Route for form-based prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve form data (convert each to float)
        magnitude = float(request.form['magnitude'])
        depth = float(request.form['depth'])
        cdi = float(request.form['cdi'])
        mmi = float(request.form['mmi'])
        sig = float(request.form['sig'])

        # Combine features into a numpy array for model input
        input_data = np.array([[magnitude, depth, cdi, mmi, sig]])

        # Predict with trained model
        prediction = model.predict(input_data)[0]
        likelihood = likelihood_mapping.get(prediction, "Unknown")

        # Return result to frontend
        return render_template('index.html', prediction_text=f"Earthquake Likelihood: {likelihood}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

# JSON API endpoint
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(force=True)
        features = [
            data['magnitude'],
            data['depth'],
            data['cdi'],
            data['mmi'],
            data['sig']
        ]

        input_data = np.array([features])
        prediction = model.predict(input_data)[0]
        likelihood = likelihood_mapping.get(prediction, "Unknown")

        return jsonify({'prediction': likelihood})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
